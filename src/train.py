import argparse
import json
import os
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from metrics import MetricLogger, StepTimer, aggregate_expert_counts, compute_expert_stats
from rebalance import (
    ExpertLoadHistory,
    RebalanceManager,
    build_deepspeed_ep_map,
    build_deepspeed_startup_layout,
    build_global_to_local_expert_index,
    build_initial_expert_map,
    build_rank_local_expert_modules,
    compute_load_metrics,
    estimate_remote_assignments,
    estimate_gpu_load,
    get_local_experts_for_rank,
    init_cost_model,
    is_map_ep_compatible,
    load_expert_map_json,
    propose_rebalanced_mapping,
    project_map_to_ep_compatible,
    save_expert_map_json,
    should_rebalance_now,
    summarize_rebalance_decision,
)
from synthetic_data import build_dataloader
from utils import (
    barrier_if_distributed,
    ensure_dir,
    get_env_info,
    get_git_commit_hash,
    get_rank_world_size_local_rank,
    init_distributed_if_needed,
    is_main_process,
    load_json,
    save_json,
    set_seed,
)


class ExpertMLP(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SimpleMoE(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = max(1, min(top_k, num_experts))
        self.gate = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList([ExpertMLP(hidden_size) for _ in range(num_experts)])

    def forward(
        self,
        x: torch.Tensor,
        expert_to_gpu_map: Optional[List[int]] = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        bsz, seq_len, hidden = x.shape
        flat = x.reshape(-1, hidden)
        gate_logits = self.gate(flat)
        topk_vals, topk_idx = torch.topk(gate_logits, k=self.top_k, dim=-1)
        topk_weights = torch.softmax(topk_vals, dim=-1)

        out_flat = torch.zeros_like(flat)
        expert_counts = torch.zeros(self.num_experts, device=flat.device, dtype=torch.float32)
        local_token_assignments = 0.0
        remote_token_assignments = 0.0

        map_now = expert_to_gpu_map if expert_to_gpu_map is not None else build_initial_expert_map(self.num_experts, world_size)
        if len(map_now) != self.num_experts:
            map_fixed = build_initial_expert_map(self.num_experts, world_size)
            for i in range(min(len(map_now), self.num_experts)):
                map_fixed[i] = int(map_now[i])
            map_now = map_fixed

        for expert_id, expert in enumerate(self.experts):
            token_mask = (topk_idx == expert_id)
            token_positions, expert_slot = token_mask.nonzero(as_tuple=True)
            if token_positions.numel() == 0:
                continue
            expert_in = flat[token_positions]
            owner_gpu = int(map_now[expert_id]) if expert_id < len(map_now) else rank
            is_local = world_size <= 1 or owner_gpu == rank
            if is_local:
                expert_out = expert(expert_in)
                local_token_assignments += float(token_positions.numel())
            else:
                # Simulate remote execution in fallback mode: local rank dispatches but does not backprop expert weights.
                with torch.no_grad():
                    expert_out = expert(expert_in)
                expert_out = expert_out.detach()
                remote_token_assignments += float(token_positions.numel())
            weights = topk_weights[token_positions, expert_slot].unsqueeze(-1)
            out_flat[token_positions] += expert_out * weights
            expert_counts[expert_id] += float(token_positions.numel())

        out = out_flat.view(bsz, seq_len, hidden)

        # Small balancing loss proxy for fallback path.
        probs = torch.softmax(gate_logits, dim=-1).mean(dim=0)
        l_aux = (probs * probs).sum() * self.num_experts
        runtime_stats = {
            "local_token_assignments": float(local_token_assignments),
            "remote_token_assignments": float(remote_token_assignments),
            "communication_proxy": float(remote_token_assignments),
        }
        return out, l_aux, expert_counts, runtime_stats


class FallbackMoEBlock(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.moe = SimpleMoE(hidden_size, num_experts, top_k)

    def forward(
        self,
        x: torch.Tensor,
        expert_to_gpu_map: Optional[List[int]],
        rank: int,
        world_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        y, l_aux, counts, runtime_stats = self.moe(
            self.norm(x), expert_to_gpu_map=expert_to_gpu_map, rank=rank, world_size=world_size
        )
        return x + y, l_aux, counts, runtime_stats


def build_seeded_expert_for_global(hidden_size: int, global_expert_id: int, base_seed: int) -> ExpertMLP:
    cpu_state = torch.get_rng_state()
    seed = int(base_seed) + 104729 * (int(global_expert_id) + 1)
    torch.manual_seed(seed)
    expert = ExpertMLP(hidden_size)
    torch.set_rng_state(cpu_state)
    return expert


class DeepSpeedMoEBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        ep_size: int,
        ds_moe_cls,
        internal_to_global_expert_ids: Optional[List[int]] = None,
        rank_local_global_expert_ids: Optional[List[int]] = None,
        expert_init_seed: int = 0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.num_experts = int(num_experts)
        self.hidden_size = int(hidden_size)
        self.expert_init_seed = int(expert_init_seed)

        if internal_to_global_expert_ids is None:
            self.internal_to_global_expert_ids = list(range(self.num_experts))
        else:
            self.internal_to_global_expert_ids = [int(x) for x in internal_to_global_expert_ids[: self.num_experts]]
            if len(self.internal_to_global_expert_ids) < self.num_experts:
                missing = [idx for idx in range(self.num_experts) if idx not in set(self.internal_to_global_expert_ids)]
                self.internal_to_global_expert_ids.extend(missing)
            self.internal_to_global_expert_ids = self.internal_to_global_expert_ids[: self.num_experts]

        self.global_to_internal_expert_ids = [-1 for _ in range(self.num_experts)]
        for internal_expert_id, global_expert_id in enumerate(self.internal_to_global_expert_ids):
            if 0 <= int(global_expert_id) < self.num_experts:
                self.global_to_internal_expert_ids[int(global_expert_id)] = int(internal_expert_id)
        for global_expert_id in range(self.num_experts):
            if self.global_to_internal_expert_ids[global_expert_id] < 0:
                self.global_to_internal_expert_ids[global_expert_id] = global_expert_id

        self.rank_local_global_expert_ids = [int(x) for x in (rank_local_global_expert_ids or [])]

        self.expert = ExpertMLP(hidden_size)
        self.moe = ds_moe_cls(
            hidden_size=hidden_size,
            expert=self.expert,
            num_experts=num_experts,
            k=top_k,
            ep_size=ep_size,
        )
        self._apply_startup_mapping()

    def _apply_startup_mapping(self) -> None:
        gate = self.moe.deepspeed_moe.gate.wg
        perm = torch.tensor(self.internal_to_global_expert_ids, dtype=torch.long)
        if perm.numel() == gate.weight.shape[0]:
            with torch.no_grad():
                gate.weight.data = gate.weight.data.index_select(0, perm)
                if gate.bias is not None:
                    gate.bias.data = gate.bias.data.index_select(0, perm)

        local_target = int(getattr(self.moe, "num_local_experts", 0))
        if local_target <= 0 or not self.rank_local_global_expert_ids:
            return
        if len(self.rank_local_global_expert_ids) != local_target:
            raise ValueError(
                "Rank-local expert count from map does not match DeepSpeed local expert count: "
                f"{len(self.rank_local_global_expert_ids)} vs {local_target}"
            )

        experts_container = self.moe.deepspeed_moe.experts
        local_modules = build_rank_local_expert_modules(
            hidden_size=self.hidden_size,
            local_expert_ids=self.rank_local_global_expert_ids,
            expert_builder=lambda h, gid: build_seeded_expert_for_global(h, gid, self.expert_init_seed),
        )
        experts_container.deepspeed_experts = nn.ModuleList(local_modules)
        experts_container.num_local_experts = int(len(local_modules))
        for expert in experts_container.deepspeed_experts:
            for param in expert.parameters():
                param.allreduce = False
                param.group_name = self.moe.expert_group_name

    def _convert_counts_internal_to_global(self, internal_counts: torch.Tensor) -> torch.Tensor:
        values = internal_counts.flatten()
        out = torch.zeros(self.num_experts, device=values.device, dtype=torch.float32)
        n = min(values.numel(), len(self.internal_to_global_expert_ids))
        if n <= 0:
            return out
        idx = torch.tensor(self.internal_to_global_expert_ids[:n], device=values.device, dtype=torch.long)
        out.index_add_(0, idx, values[:n].to(dtype=torch.float32))
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        bsz, seq_len, hidden = x.shape
        flat = self.norm(x).reshape(-1, hidden)
        out_flat, l_aux, exp_counts = self.moe(flat)
        out = out_flat.view(bsz, seq_len, hidden)
        counts = self._convert_counts_internal_to_global(exp_counts.to(out.device, dtype=torch.float32))
        runtime_stats = {
            "local_token_assignments": 0.0,
            "remote_token_assignments": 0.0,
            "communication_proxy": 0.0,
        }
        return x + out, l_aux, counts, runtime_stats


class TinyMoELM(nn.Module):
    def __init__(
        self,
        config: Dict,
        backend: str,
        ds_moe_cls=None,
        deepspeed_ep_size: int = 1,
        deepspeed_internal_to_global_expert_ids: Optional[List[int]] = None,
        deepspeed_rank_local_expert_ids: Optional[List[int]] = None,
        expert_init_seed: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = int(config["vocab_size"])
        self.hidden_size = int(config["hidden_size"])
        self.num_layers = int(config["num_layers"])
        self.num_experts = int(config["num_experts"])
        top_k = int(config["top_k"])

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        blocks = []
        for layer_idx in range(self.num_layers):
            if backend == "deepspeed":
                blocks.append(
                    DeepSpeedMoEBlock(
                        self.hidden_size,
                        self.num_experts,
                        top_k,
                        ep_size=max(1, int(deepspeed_ep_size)),
                        ds_moe_cls=ds_moe_cls,
                        internal_to_global_expert_ids=deepspeed_internal_to_global_expert_ids,
                        rank_local_global_expert_ids=deepspeed_rank_local_expert_ids,
                        expert_init_seed=expert_init_seed + 10007 * int(layer_idx),
                    )
                )
            else:
                blocks.append(FallbackMoEBlock(self.hidden_size, self.num_experts, top_k))
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = nn.LayerNorm(self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        expert_to_gpu_map: Optional[List[int]] = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        x = self.embedding(input_ids)
        total_aux = torch.zeros(1, device=x.device, dtype=x.dtype)
        total_counts = torch.zeros(self.num_experts, device=x.device, dtype=torch.float32)
        total_local_assign = 0.0
        total_remote_assign = 0.0
        total_comm_proxy = 0.0
        for block in self.blocks:
            if isinstance(block, FallbackMoEBlock):
                x, l_aux, counts, runtime_stats = block(
                    x,
                    expert_to_gpu_map=expert_to_gpu_map,
                    rank=rank,
                    world_size=world_size,
                )
            else:
                x, l_aux, counts, runtime_stats = block(x)
            total_aux = total_aux + l_aux
            total_counts = total_counts + counts
            total_local_assign += float(runtime_stats["local_token_assignments"])
            total_remote_assign += float(runtime_stats["remote_token_assignments"])
            total_comm_proxy += float(runtime_stats["communication_proxy"])
        logits = self.lm_head(self.final_norm(x))
        runtime_totals = {
            "local_token_assignments": float(total_local_assign),
            "remote_token_assignments": float(total_remote_assign),
            "communication_proxy": float(total_comm_proxy),
        }
        return logits, total_aux.squeeze(), total_counts, runtime_totals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 MoE baseline training")
    parser.add_argument("--config", type=str, default="configs/base.json")
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--ds_config", type=str, default="configs/ds_config_moe.json")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--run_name", type=str, default="")
    return parser.parse_args()


def autocast_context(precision: str, device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def pick_precision(requested: str, device: torch.device, is_main: bool) -> str:
    p = requested.lower()
    if p not in {"fp32", "fp16", "bf16"}:
        p = "fp32"
    if device.type != "cuda":
        return "fp32"
    if p == "bf16":
        bf16_ok = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        if not bf16_ok:
            if is_main:
                print("bf16 requested but not supported, falling back to fp32")
            return "fp32"
    return p


def setup_device(local_rank: int, config: Dict) -> torch.device:
    if config.get("device"):
        dev = torch.device(config["device"])
    elif torch.cuda.is_available():
        dev = torch.device(f"cuda:{local_rank}")
    else:
        dev = torch.device("cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    return dev


def main() -> None:
    args = parse_args()
    config = load_json(args.config)

    if not args.run_name:
        args.run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    init_distributed_if_needed()
    rank_info = get_rank_world_size_local_rank()
    rank = rank_info["rank"]
    world_size = rank_info["world_size"]
    local_rank = rank_info["local_rank"]
    is_main = is_main_process()

    device = setup_device(local_rank, config)
    set_seed(int(config["seed"]) + rank)
    precision = pick_precision(str(config.get("precision", "fp32")), device, is_main)
    log_every = int(config["log_every"])

    enable_rebalance_planner = bool(config.get("enable_rebalance_planner", False))
    rebalance_eval_interval = max(1, int(config.get("rebalance_eval_interval", log_every)))
    rebalance_history_size = max(1, int(config.get("rebalance_history_size", 5)))
    rebalance_threshold = float(config.get("rebalance_threshold", 0.05))
    rebalance_metric = str(config.get("rebalance_metric", "wasted_fraction"))
    rebalance_min_steps = max(0, int(config.get("rebalance_min_steps", 20)))
    rebalance_cooldown = max(0, int(config.get("rebalance_cooldown", 100)))
    rebalance_use_ema = bool(config.get("rebalance_use_ema", True))
    rebalance_ema_beta = float(config.get("rebalance_ema_beta", 0.8))
    rebalance_dry_run = bool(config.get("rebalance_dry_run", True))
    rebalance_apply_live_fallback = bool(config.get("rebalance_apply_live_fallback", True))
    rebalance_log_remote_stats = bool(config.get("rebalance_log_remote_stats", True))
    rebalance_min_expected_improvement = float(config.get("rebalance_min_expected_improvement", 0.0))

    deepspeed_ep_size_cfg = int(config.get("deepspeed_ep_size", 0))
    deepspeed_enable_mapped_experts = bool(config.get("deepspeed_enable_mapped_experts", False))
    deepspeed_allow_rebuild_on_rebalance = bool(config.get("deepspeed_allow_rebuild_on_rebalance", False))
    deepspeed_rebalance_mode = str(config.get("deepspeed_rebalance_mode", "save_next_requires_restart"))
    deepspeed_initial_map_path_raw = config.get("deepspeed_initial_map_path")
    deepspeed_initial_map_path = (
        str(deepspeed_initial_map_path_raw).strip()
        if deepspeed_initial_map_path_raw not in (None, "")
        else None
    )

    deepspeed_ep_size_effective = int(deepspeed_ep_size_cfg) if int(deepspeed_ep_size_cfg) > 0 else int(world_size)
    if deepspeed_enable_mapped_experts and deepspeed_ep_size_effective != int(world_size):
        if is_main:
            print(
                "deepspeed_enable_mapped_experts=true requires ep_size==world_size in Step 4; "
                f"overriding ep_size {deepspeed_ep_size_effective} -> {world_size}"
            )
        deepspeed_ep_size_effective = int(world_size)
    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))
    config_copy_path = out_dir / f"{args.run_name}_config_used.json"
    summary_path = out_dir / f"{args.run_name}_summary.json"
    jsonl_path = out_dir / f"{args.run_name}.jsonl"

    config_used = dict(config)
    config_used["precision_effective"] = precision
    config_used["enable_rebalance_planner"] = enable_rebalance_planner
    config_used["rebalance_eval_interval"] = rebalance_eval_interval
    config_used["rebalance_history_size"] = rebalance_history_size
    config_used["rebalance_threshold"] = rebalance_threshold
    config_used["rebalance_metric"] = rebalance_metric
    config_used["rebalance_min_steps"] = rebalance_min_steps
    config_used["rebalance_cooldown"] = rebalance_cooldown
    config_used["rebalance_use_ema"] = rebalance_use_ema
    config_used["rebalance_ema_beta"] = rebalance_ema_beta
    config_used["rebalance_dry_run"] = rebalance_dry_run
    config_used["rebalance_apply_live_fallback"] = rebalance_apply_live_fallback
    config_used["rebalance_log_remote_stats"] = rebalance_log_remote_stats
    config_used["rebalance_min_expected_improvement"] = rebalance_min_expected_improvement
    config_used["deepspeed_ep_size"] = deepspeed_ep_size_effective
    config_used["deepspeed_enable_mapped_experts"] = deepspeed_enable_mapped_experts
    config_used["deepspeed_allow_rebuild_on_rebalance"] = deepspeed_allow_rebuild_on_rebalance
    config_used["deepspeed_rebalance_mode"] = deepspeed_rebalance_mode
    config_used["deepspeed_initial_map_path"] = deepspeed_initial_map_path
    save_json(str(config_copy_path), config_used)

    env_info = get_env_info()
    metadata = {
        "seed": int(config["seed"]),
        "command": " ".join(sys.argv),
        "git_commit": get_git_commit_hash(),
        "config_path": str(args.config),
        "config_copy_path": str(config_copy_path),
        "env": env_info,
        "deepspeed_requested": bool(args.deepspeed),
        "nccl_debug": os.environ.get("NCCL_DEBUG"),
    }

    if is_main:
        print(json.dumps({"run_name": args.run_name, "metadata": metadata}, indent=2))

    ds_available = False
    ds_module = None
    ds_moe_cls = None
    if args.deepspeed and bool(config.get("use_deepspeed_moe", True)):
        try:
            import deepspeed  # type: ignore
            from deepspeed.moe.layer import MoE as DSMoE  # type: ignore

            ds_available = True
            ds_module = deepspeed
            ds_moe_cls = DSMoE
        except Exception as e:
            if is_main:
                print(f"DeepSpeed import failed, fallback MoE will be used: {e}")

    backend = "deepspeed" if ds_available and device.type == "cuda" else "fallback"

    num_experts = int(config["num_experts"])
    initial_active_map = build_initial_expert_map(num_experts, world_size)
    deepspeed_mapping_apply_mode = "not_deepspeed_backend"
    deepspeed_startup_map_apply_reason = "not_using_deepspeed"
    deepspeed_startup_requested_map_path = deepspeed_initial_map_path
    deepspeed_startup_map_was_projected = False
    deepspeed_pending_map_path: Optional[str] = None
    deepspeed_map_rebuild_count = 0
    deepspeed_internal_to_global_expert_ids: List[int] = list(range(num_experts))
    deepspeed_global_to_internal_expert_ids: List[int] = list(range(num_experts))
    deepspeed_rank_local_expert_ids_startup: List[int] = get_local_experts_for_rank(initial_active_map, rank)
    deepspeed_startup_layout: Dict[str, Any] = {
        "internal_to_global_expert_ids": deepspeed_internal_to_global_expert_ids,
        "global_to_internal_expert_ids": deepspeed_global_to_internal_expert_ids,
        "rank_to_global_expert_ids": [deepspeed_rank_local_expert_ids_startup[:] for _ in range(max(1, world_size))],
    }

    if backend == "deepspeed":
        if num_experts % max(1, deepspeed_ep_size_effective) != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by "
                f"deepspeed_ep_size ({deepspeed_ep_size_effective}) for DeepSpeed MoE"
            )
        initial_active_map = build_deepspeed_ep_map(num_experts, deepspeed_ep_size_effective)
        deepspeed_mapping_apply_mode = "ep_layout_only_no_custom_startup_map"
        deepspeed_startup_map_apply_reason = "applied_default_deepspeed_ep_layout"

        if deepspeed_enable_mapped_experts:
            deepspeed_mapping_apply_mode = "startup_map_controls_local_experts_restart_updates_only"
            deepspeed_startup_map_apply_reason = "applied_default_deepspeed_mapping"
            if deepspeed_rebalance_mode != "save_next_requires_restart" and is_main:
                print(
                    f"deepspeed_rebalance_mode={deepspeed_rebalance_mode} is not supported in Step 4; "
                    "using save_next_requires_restart"
                )
            if deepspeed_initial_map_path is not None:
                loaded_map = load_expert_map_json(
                    deepspeed_initial_map_path,
                    num_experts=num_experts,
                    default_map=initial_active_map,
                )
                projected = project_map_to_ep_compatible(loaded_map, deepspeed_ep_size_effective)
                if not is_map_ep_compatible(projected, deepspeed_ep_size_effective):
                    projected = build_deepspeed_ep_map(num_experts, deepspeed_ep_size_effective)
                if loaded_map != projected:
                    deepspeed_startup_map_was_projected = True
                    if is_main:
                        print(
                            "Loaded DeepSpeed initial map was not EP-compatible, "
                            "auto-projected to equal experts per rank"
                        )
                initial_active_map = projected
                if deepspeed_startup_map_was_projected:
                    deepspeed_startup_map_apply_reason = "applied_initial_deepspeed_mapping_after_projection"
                else:
                    deepspeed_startup_map_apply_reason = "applied_initial_deepspeed_mapping"

        deepspeed_startup_layout = build_deepspeed_startup_layout(initial_active_map, deepspeed_ep_size_effective)
        deepspeed_internal_to_global_expert_ids = [
            int(x) for x in deepspeed_startup_layout["internal_to_global_expert_ids"]
        ]
        deepspeed_global_to_internal_expert_ids = [
            int(x) for x in deepspeed_startup_layout["global_to_internal_expert_ids"]
        ]
        rank_to_global = deepspeed_startup_layout["rank_to_global_expert_ids"]
        if rank < len(rank_to_global):
            deepspeed_rank_local_expert_ids_startup = [int(x) for x in rank_to_global[rank]]
        else:
            deepspeed_rank_local_expert_ids_startup = get_local_experts_for_rank(initial_active_map, rank)

    model = TinyMoELM(
        config,
        backend=backend,
        ds_moe_cls=ds_moe_cls,
        deepspeed_ep_size=deepspeed_ep_size_effective,
        deepspeed_internal_to_global_expert_ids=(
            deepspeed_internal_to_global_expert_ids
            if (backend == "deepspeed" and deepspeed_enable_mapped_experts)
            else None
        ),
        deepspeed_rank_local_expert_ids=(
            deepspeed_rank_local_expert_ids_startup
            if (backend == "deepspeed" and deepspeed_enable_mapped_experts)
            else None
        ),
        expert_init_seed=int(config["seed"]),
    ).to(device)

    train_steps = int(config["train_steps"])
    tokens_per_step_global = int(config["micro_batch_size"]) * int(config["seq_len"]) * int(world_size)

    dataloader = build_dataloader(config, rank=rank, world_size=world_size)
    data_iter = iter(dataloader)

    deepspeed_engine = None
    optimizer = None
    scaler: Optional[torch.cuda.amp.GradScaler] = None
    using_deepspeed = False

    if backend == "deepspeed" and ds_module is not None:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"])
        )
        ds_runtime_config = load_json(args.ds_config)
        ds_runtime_config["train_micro_batch_size_per_gpu"] = int(config["micro_batch_size"])
        ds_runtime_config.setdefault("bf16", {})["enabled"] = precision == "bf16"
        ds_runtime_config.setdefault("fp16", {})["enabled"] = precision == "fp16"
        try:
            deepspeed_engine, optimizer, _, _ = ds_module.initialize(
                model=model,
                model_parameters=model.parameters(),
                optimizer=optimizer,
                config=ds_runtime_config,
            )
            using_deepspeed = True
            if is_main:
                print("Running with DeepSpeed MoE backend")
        except Exception as e:
            if is_main:
                print(f"DeepSpeed initialize failed, using fallback path: {e}")
            backend = "fallback"
            model = TinyMoELM(config, backend=backend).to(device)
            deepspeed_mapping_apply_mode = "not_deepspeed_backend"
            deepspeed_startup_map_apply_reason = "deepspeed_init_failed_fallback"
            initial_active_map = build_initial_expert_map(num_experts, world_size)

    if not using_deepspeed:
        if world_size > 1 and dist.is_initialized():
            ddp_device_ids = [local_rank] if device.type == "cuda" else None
            model = DDP(model, device_ids=ddp_device_ids, find_unused_parameters=(backend == "fallback"))
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"])
        )
        if precision == "fp16" and device.type == "cuda":
            scaler = torch.cuda.amp.GradScaler()
        if is_main:
            print("Running with fallback MoE backend")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    logger = MetricLogger(str(jsonl_path), is_main=is_main, ma_window=20)
    timer = StepTimer()
    moe_aux_loss_coef = float(config.get("moe_aux_loss_coef", 0.01))
    planner_history: Optional[ExpertLoadHistory] = None
    rebalance_manager = RebalanceManager(initial_active_map)
    last_rebalance_step = -10**9

    startup_active_map = rebalance_manager.get_active_map()
    startup_local = get_local_experts_for_rank(startup_active_map, rank)
    startup_local_index = build_global_to_local_expert_index(startup_local)
    startup_payload = {
        "rank": int(rank),
        "world_size": int(world_size),
        "deepspeed_ep_size": deepspeed_ep_size_effective if backend == "deepspeed" else None,
        "deepspeed_mapping_enabled": bool(backend == "deepspeed" and deepspeed_enable_mapped_experts),
        "deepspeed_mapping_apply_mode": deepspeed_mapping_apply_mode,
        "deepspeed_rebalance_mode": deepspeed_rebalance_mode,
        "deepspeed_allow_rebuild_on_rebalance": deepspeed_allow_rebuild_on_rebalance,
        "deepspeed_startup_map_apply_reason": deepspeed_startup_map_apply_reason,
        "deepspeed_initial_map_path": deepspeed_startup_requested_map_path,
        "deepspeed_startup_map_projected": bool(deepspeed_startup_map_was_projected),
        "expert_to_gpu_map_active_startup": startup_active_map,
        "local_expert_ids_active_startup": startup_local,
        "num_local_experts_active_startup": int(len(startup_local)),
        "rank_local_expert_count_startup": int(len(startup_local)),
        "global_to_local_expert_index_startup": {str(k): int(v) for k, v in startup_local_index.items()},
        "deepspeed_internal_to_global_expert_ids_startup": deepspeed_internal_to_global_expert_ids
        if backend == "deepspeed"
        else None,
        "deepspeed_global_to_internal_expert_ids_startup": deepspeed_global_to_internal_expert_ids
        if backend == "deepspeed"
        else None,
    }
    print(json.dumps(startup_payload)) 

    if is_main:
        _1gpu_path = Path(args.out_dir) / "phase1_1gpu.jsonl"
        if _1gpu_path.exists():
            _1gpu_steps = [json.loads(l) for l in _1gpu_path.open() if l.strip()]
            init_cost_model(steps_1gpu=_1gpu_steps,
                H=int(config["hidden_size"]),
                num_layers=int(config["num_layers"]),)
        else:
            if is_main:
                print("[CostModel] phase1_1gpu.jsonl not found, using fallback metrics")

    if enable_rebalance_planner:
        planner_history = ExpertLoadHistory(
            history_size=rebalance_history_size,
            use_ema=rebalance_use_ema,
            ema_beta=rebalance_ema_beta,
            num_experts=int(config["num_experts"]),
        )

    for step in range(1, train_steps + 1):
        batch = next(data_iter)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        active_map_before_step = rebalance_manager.get_active_map()
        local_experts_before_step = get_local_experts_for_rank(active_map_before_step, rank)
        global_to_local_before_step = build_global_to_local_expert_index(local_experts_before_step)
        runtime_stats = {
            "local_token_assignments": 0.0,
            "remote_token_assignments": 0.0,
            "communication_proxy": 0.0,
        }

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timer.start()

        if not using_deepspeed:
            optimizer.zero_grad(set_to_none=True)

        if using_deepspeed:
            ds_out = deepspeed_engine(input_ids)
            if isinstance(ds_out, tuple) and len(ds_out) == 4:
                logits, l_aux, exp_counts, runtime_stats = ds_out
            elif isinstance(ds_out, tuple) and len(ds_out) == 3:
                logits, l_aux, exp_counts = ds_out
            else:
                raise RuntimeError("Unexpected DeepSpeed output format")
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss = loss + moe_aux_loss_coef * l_aux
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timer.mark_forward_done()

            deepspeed_engine.backward(loss)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timer.mark_backward_done()

            deepspeed_engine.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timer.mark_optim_done()
        else:
            with autocast_context(precision, device):
                model_out = model(input_ids, expert_to_gpu_map=active_map_before_step, rank=rank, world_size=world_size)
                if isinstance(model_out, tuple) and len(model_out) == 4:
                    logits, l_aux, exp_counts, runtime_stats = model_out
                else:
                    raise RuntimeError("Unexpected fallback model output format")
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                loss = loss + moe_aux_loss_coef * l_aux
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timer.mark_forward_done()

            if scaler is not None:
                scaler.scale(loss).backward()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timer.mark_backward_done()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timer.mark_backward_done()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
                optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timer.mark_optim_done()

        timing = timer.end()
        step_time = float(timing["step_time_sec"])
        tokens_per_sec = float(tokens_per_step_global / step_time)
        steps_per_sec = float(1.0 / step_time)

        if exp_counts is None:
            exp_counts = torch.zeros(int(config["num_experts"]), device=device, dtype=torch.float32)
        exp_counts = exp_counts.to(device=device, dtype=torch.float32).flatten()
        if exp_counts.numel() != int(config["num_experts"]):
            fixed = torch.zeros(int(config["num_experts"]), device=device, dtype=torch.float32)
            n = min(fixed.numel(), exp_counts.numel())
            fixed[:n] = exp_counts[:n]
            exp_counts = fixed
        local_exp_counts = exp_counts.detach().clone()
        exp_counts = aggregate_expert_counts(exp_counts)
        expert_counts_list = exp_counts.detach().cpu().tolist()
        expert_stats = compute_expert_stats(expert_counts_list)
        local_expert_counts_list = local_exp_counts.detach().cpu().tolist()

        local_token_assignments = float(runtime_stats.get("local_token_assignments", 0.0))
        remote_token_assignments = float(runtime_stats.get("remote_token_assignments", 0.0))
        communication_proxy_runtime = float(runtime_stats.get("communication_proxy", 0.0))
        if dist.is_available() and dist.is_initialized():
            runtime_vec = torch.tensor(
                [local_token_assignments, remote_token_assignments, communication_proxy_runtime],
                device=device,
                dtype=torch.float32,
            )
            dist.all_reduce(runtime_vec, op=dist.ReduceOp.SUM)
            local_token_assignments = float(runtime_vec[0].item())
            remote_token_assignments = float(runtime_vec[1].item())
            communication_proxy_runtime = float(runtime_vec[2].item())
        total_assignments = local_token_assignments + remote_token_assignments
        remote_fraction = float(remote_token_assignments / total_assignments) if total_assignments > 0 else 0.0

        if device.type == "cuda":
            max_alloc = int(torch.cuda.max_memory_allocated(device))
            max_res = int(torch.cuda.max_memory_reserved(device))
        else:
            max_alloc = 0
            max_res = 0

        planner_info = {
            "rebalance_enabled": enable_rebalance_planner,
            "rebalance_dry_run": rebalance_dry_run,
            "rebalance_evaluated_this_step": False,
            "rebalance_triggered": False,
            "rebalance_applied": False,
            "rebalance_apply_reason": "not_triggered",
            "rebalance_apply_backend": backend,
            "rebalance_num_experts_moved": 0,
            "rebalance_event_id": None,
            "rebalance_reason": "planner_disabled",
            "rebalance_summary": "planner disabled",
            "rebalance_metric": rebalance_metric if enable_rebalance_planner else None,
            "rebalance_metric_current": None,
            "rebalance_metric_proposed": None,
            "rebalance_expected_improvement": None,
            "rebalance_min_expected_improvement": rebalance_min_expected_improvement,
            "predicted_gpu_loads_current": None,
            "predicted_gpu_loads_proposed": None,
            "expert_to_gpu_map_current": active_map_before_step[:],
            "expert_to_gpu_map_active": active_map_before_step[:],
            "expert_to_gpu_map_previous": None,
            "proposed_expert_to_gpu_map": None,
            "smoothed_expert_loads": None,
            "local_token_assignments": local_token_assignments if rebalance_log_remote_stats else None,
            "remote_token_assignments": remote_token_assignments if rebalance_log_remote_stats else None,
            "remote_fraction": remote_fraction if rebalance_log_remote_stats else None,
            "communication_proxy_runtime": communication_proxy_runtime if rebalance_log_remote_stats else None,
            "communication_proxy_current": None,
            "communication_proxy_proposed": None,
            "communication_proxy_improvement": None,
            "remote_dispatch_estimate_current": None,
            "remote_dispatch_estimate_proposed": None,
            "num_local_experts_active": len(local_experts_before_step),
            "rank_local_expert_count": len(local_experts_before_step),
            "local_expert_ids_active": local_experts_before_step,
            "global_to_local_expert_index": {str(k): int(v) for k, v in global_to_local_before_step.items()},
            "deepspeed_ep_size": deepspeed_ep_size_effective if backend == "deepspeed" else None,
            "deepspeed_mapping_enabled": bool(backend == "deepspeed" and deepspeed_enable_mapped_experts),
            "deepspeed_mapping_apply_mode": deepspeed_mapping_apply_mode,
            "deepspeed_rebalance_mode": deepspeed_rebalance_mode,
            "deepspeed_allow_rebuild_on_rebalance": deepspeed_allow_rebuild_on_rebalance,
            "deepspeed_map_rebuild_count": int(deepspeed_map_rebuild_count),
            "deepspeed_pending_map_path": deepspeed_pending_map_path,
            "deepspeed_startup_map_apply_reason": deepspeed_startup_map_apply_reason,
            "deepspeed_initial_map_path": deepspeed_startup_requested_map_path,
            "deepspeed_startup_map_projected": bool(deepspeed_startup_map_was_projected),
            "proposed_expert_to_gpu_map_ep_compatible": None,
        }

        if enable_rebalance_planner and planner_history is not None:
            planner_history.update(expert_counts_list)
            planner_info["rebalance_reason"] = "not_evaluated_this_step"
            planner_info["rebalance_summary"] = "planner enabled, waiting for eval step"

            should_eval = step % rebalance_eval_interval == 0
            if should_eval:
                planner_info["rebalance_evaluated_this_step"] = True
                smoothed_expert_loads = planner_history.get_smoothed_load()
                planner_info["smoothed_expert_loads"] = smoothed_expert_loads

                if not planner_history.is_ready():
                    reason = "history_not_ready"
                    planner_info["rebalance_reason"] = reason
                    planner_info["rebalance_summary"] = summarize_rebalance_decision(
                        step=step,
                        triggered=False,
                        metric_name=rebalance_metric,
                        current_value=None,
                        proposed_value=None,
                        expected_improvement=None,
                        reason=reason,
                    )
                else:
                    current_map = rebalance_manager.get_active_map()
                    planner_info["expert_to_gpu_map_current"] = current_map[:]
                    current_gpu_loads = estimate_gpu_load(
                        smoothed_expert_loads, current_map, world_size
                    )
                    current_metrics = compute_load_metrics(current_gpu_loads)

                    proposal = propose_rebalanced_mapping(smoothed_expert_loads, world_size)
                    proposed_gpu_loads = proposal["gpu_loads"]  # type: ignore[index]
                    proposed_map = proposal["expert_to_gpu_map"]  # type: ignore[index]
                    proposed_metrics = proposal["metrics"]  # type: ignore[index]
                    proposed_map_ep_compatible = proposed_map[:]

                    metric_current = current_metrics.get(rebalance_metric)
                    metric_proposed = proposed_metrics.get(rebalance_metric)

                    expected_improvement = None
                    if metric_current is not None and metric_proposed is not None:
                        expected_improvement = float(metric_current - metric_proposed)

                    comm_proxy_current = float(current_metrics["max_load"] - current_metrics["mean_load"])
                    comm_proxy_proposed = float(proposed_metrics["max_load"] - proposed_metrics["mean_load"])
                    comm_proxy_improvement = float(comm_proxy_current - comm_proxy_proposed)

                    base_trigger, reason = should_rebalance_now(
                        metrics=current_metrics,
                        metric_name=rebalance_metric,
                        threshold=rebalance_threshold,
                        step=step,
                        min_steps=rebalance_min_steps,
                        cooldown=rebalance_cooldown,
                        last_trigger_step=last_rebalance_step,
                    )

                    triggered = bool(
                        base_trigger
                        and expected_improvement is not None
                        and expected_improvement > rebalance_min_expected_improvement
                    )
                    if base_trigger and not triggered:
                        reason = (
                            f"{reason},below_min_expected_improvement={rebalance_min_expected_improvement:.4f}"
                        )

                    apply_reason = "not_triggered"
                    apply_event = {
                        "applied": False,
                        "event_id": None,
                        "num_experts_moved": 0,
                        "previous_map": None,
                        "active_map": rebalance_manager.get_active_map(),
                    }
                    if triggered:
                        last_rebalance_step = step
                        if rebalance_dry_run:
                            apply_reason = "triggered_but_dry_run"
                        elif backend == "fallback":
                            if not rebalance_apply_live_fallback:
                                apply_reason = "triggered_but_live_apply_disabled_for_fallback"
                            else:
                                apply_event = rebalance_manager.apply_rebalanced_mapping(
                                    new_map=proposed_map,
                                    step=step,
                                    backend=backend,
                                    reason="live_apply_fallback",
                                )
                                if bool(apply_event["applied"]):
                                    apply_reason = "applied_live_fallback"
                                else:
                                    apply_reason = "mapping_unchanged"
                        elif backend == "deepspeed":
                            if not deepspeed_enable_mapped_experts:
                                apply_reason = "triggered_but_deepspeed_mapping_disabled"
                            else:
                                proposed_map_ep_compatible = project_map_to_ep_compatible(
                                    proposed_map, deepspeed_ep_size_effective
                                )
                                if proposed_map_ep_compatible == rebalance_manager.get_active_map():
                                    apply_reason = "mapping_unchanged"
                                elif deepspeed_rebalance_mode != "save_next_requires_restart":
                                    apply_reason = "triggered_but_unsupported_deepspeed_mode"
                                else:
                                    next_map_path = out_dir / f"{args.run_name}_deepspeed_next_map_step{step}.json"
                                    save_expert_map_json(
                                        str(next_map_path),
                                        proposed_map_ep_compatible,
                                        metadata={
                                            "step": int(step),
                                            "backend": backend,
                                            "reason": "saved_next_deepspeed_map_requires_restart",
                                            "requested_map": proposed_map,
                                            "requested_map_ep_compatible": bool(
                                                is_map_ep_compatible(proposed_map, deepspeed_ep_size_effective)
                                            ),
                                            "deepspeed_ep_size": int(deepspeed_ep_size_effective),
                                        },
                                    )
                                    deepspeed_pending_map_path = str(next_map_path)
                                    apply_reason = "saved_next_deepspeed_map_requires_restart"
                        else:
                            apply_event = rebalance_manager.apply_rebalanced_mapping(
                                new_map=proposed_map,
                                step=step,
                                backend=backend,
                                reason="live_apply_other_backend",
                            )
                            if bool(apply_event["applied"]):
                                apply_reason = "applied_live_other_backend"
                            else:
                                apply_reason = "mapping_unchanged"

                    planner_info["rebalance_triggered"] = triggered
                    planner_info["rebalance_reason"] = reason
                    planner_info["rebalance_applied"] = bool(apply_event["applied"])
                    planner_info["rebalance_apply_reason"] = apply_reason
                    planner_info["rebalance_event_id"] = apply_event["event_id"]
                    planner_info["rebalance_num_experts_moved"] = int(apply_event["num_experts_moved"])
                    planner_info["rebalance_metric_current"] = metric_current
                    planner_info["rebalance_metric_proposed"] = metric_proposed
                    planner_info["rebalance_expected_improvement"] = expected_improvement
                    planner_info["predicted_gpu_loads_current"] = current_gpu_loads
                    planner_info["predicted_gpu_loads_proposed"] = proposed_gpu_loads
                    planner_info["proposed_expert_to_gpu_map"] = proposed_map
                    planner_info["proposed_expert_to_gpu_map_ep_compatible"] = proposed_map_ep_compatible
                    planner_info["communication_proxy_current"] = comm_proxy_current
                    planner_info["communication_proxy_proposed"] = comm_proxy_proposed
                    planner_info["communication_proxy_improvement"] = comm_proxy_improvement
                    planner_info["expert_to_gpu_map_previous"] = apply_event["previous_map"]
                    planner_info["expert_to_gpu_map_active"] = apply_event["active_map"]
                    planner_info["remote_dispatch_estimate_current"] = estimate_remote_assignments(
                        local_expert_counts_list, current_map, rank
                    )
                    planner_info["remote_dispatch_estimate_proposed"] = estimate_remote_assignments(
                        local_expert_counts_list, proposed_map_ep_compatible, rank
                    )
                    if planner_info["rebalance_applied"]:
                        active_after = rebalance_manager.get_active_map()
                        locals_after = get_local_experts_for_rank(active_after, rank)
                        planner_info["num_local_experts_active"] = len(locals_after)
                        planner_info["rank_local_expert_count"] = len(locals_after)
                        planner_info["local_expert_ids_active"] = locals_after
                        planner_info["global_to_local_expert_index"] = {
                            str(k): int(v) for k, v in build_global_to_local_expert_index(locals_after).items()
                        }
                    planner_info["rebalance_summary"] = summarize_rebalance_decision(
                        step=step,
                        triggered=triggered,
                        metric_name=rebalance_metric,
                        current_value=metric_current,
                        proposed_value=metric_proposed,
                        expected_improvement=expected_improvement,
                        reason=reason,
                    )
            planner_info["expert_to_gpu_map_active"] = rebalance_manager.get_active_map()
            planner_info["deepspeed_pending_map_path"] = deepspeed_pending_map_path

        if step % log_every == 0 or bool(planner_info["rebalance_evaluated_this_step"]):
            row = {
                "step": step,
                "rank": rank,
                "world_size": world_size,
                "step_time_sec": step_time,
                "tokens_per_sec": tokens_per_sec,
                "steps_per_sec": steps_per_sec,
                "forward_time_sec": float(timing["forward_time_sec"]),
                "backward_time_sec": float(timing["backward_time_sec"]),
                "optim_time_sec": float(timing["optim_time_sec"]),
                "cuda_max_memory_allocated": max_alloc,
                "cuda_max_memory_reserved": max_res,
                "expert_counts": expert_counts_list,
                "deepspeed_enabled": using_deepspeed,
                "moe_backend": backend,
                "nccl_debug": os.environ.get("NCCL_DEBUG"),
            }
            row.update(expert_stats)
            row.update(planner_info)
            if is_main:
                logger.log(row)

    barrier_if_distributed()
    if is_main:
        logger.finalize_summary(
            summary_path=str(summary_path),
            run_name=args.run_name,
            metadata=metadata,
            world_size=world_size,
            tokens_per_step_global=tokens_per_step_global,
        )
        print(f"Saved run summary to {summary_path}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
