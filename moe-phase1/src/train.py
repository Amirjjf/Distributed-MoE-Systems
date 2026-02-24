import argparse
import json
import os
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from metrics import MetricLogger, StepTimer, aggregate_expert_counts, compute_expert_stats
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, hidden = x.shape
        flat = x.reshape(-1, hidden)
        gate_logits = self.gate(flat)
        topk_vals, topk_idx = torch.topk(gate_logits, k=self.top_k, dim=-1)
        topk_weights = torch.softmax(topk_vals, dim=-1)

        out_flat = torch.zeros_like(flat)
        expert_counts = torch.zeros(self.num_experts, device=flat.device, dtype=torch.float32)

        for expert_id, expert in enumerate(self.experts):
            token_mask = (topk_idx == expert_id)
            token_positions, expert_slot = token_mask.nonzero(as_tuple=True)
            if token_positions.numel() == 0:
                continue
            expert_in = flat[token_positions]
            expert_out = expert(expert_in)
            weights = topk_weights[token_positions, expert_slot].unsqueeze(-1)
            out_flat[token_positions] += expert_out * weights
            expert_counts[expert_id] += float(token_positions.numel())

        out = out_flat.view(bsz, seq_len, hidden)

        # Small balancing loss proxy for fallback path.
        probs = torch.softmax(gate_logits, dim=-1).mean(dim=0)
        l_aux = (probs * probs).sum() * self.num_experts
        return out, l_aux, expert_counts


class FallbackMoEBlock(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.moe = SimpleMoE(hidden_size, num_experts, top_k)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y, l_aux, counts = self.moe(self.norm(x))
        return x + y, l_aux, counts


class DeepSpeedMoEBlock(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int, ds_moe_cls) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.expert = ExpertMLP(hidden_size)
        self.moe = ds_moe_cls(hidden_size=hidden_size, expert=self.expert, num_experts=num_experts, k=top_k, ep_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, hidden = x.shape
        flat = self.norm(x).reshape(-1, hidden)
        out_flat, l_aux, exp_counts = self.moe(flat)
        out = out_flat.view(bsz, seq_len, hidden)
        counts = exp_counts.to(out.device, dtype=torch.float32)
        return x + out, l_aux, counts


class TinyMoELM(nn.Module):
    def __init__(self, config: Dict, backend: str, ds_moe_cls=None) -> None:
        super().__init__()
        self.vocab_size = int(config["vocab_size"])
        self.hidden_size = int(config["hidden_size"])
        self.num_layers = int(config["num_layers"])
        self.num_experts = int(config["num_experts"])
        top_k = int(config["top_k"])

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        blocks = []
        for _ in range(self.num_layers):
            if backend == "deepspeed":
                blocks.append(DeepSpeedMoEBlock(self.hidden_size, self.num_experts, top_k, ds_moe_cls))
            else:
                blocks.append(FallbackMoEBlock(self.hidden_size, self.num_experts, top_k))
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = nn.LayerNorm(self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.embedding(input_ids)
        total_aux = torch.zeros(1, device=x.device, dtype=x.dtype)
        total_counts = torch.zeros(self.num_experts, device=x.device, dtype=torch.float32)
        for block in self.blocks:
            x, l_aux, counts = block(x)
            total_aux = total_aux + l_aux
            total_counts = total_counts + counts
        logits = self.lm_head(self.final_norm(x))
        return logits, total_aux.squeeze(), total_counts


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

    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))
    config_copy_path = out_dir / f"{args.run_name}_config_used.json"
    summary_path = out_dir / f"{args.run_name}_summary.json"
    jsonl_path = out_dir / f"{args.run_name}.jsonl"

    config_used = dict(config)
    config_used["precision_effective"] = precision
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
    model = TinyMoELM(config, backend=backend, ds_moe_cls=ds_moe_cls).to(device)

    train_steps = int(config["train_steps"])
    log_every = int(config["log_every"])
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

    if not using_deepspeed:
        if world_size > 1 and dist.is_initialized():
            ddp_device_ids = [local_rank] if device.type == "cuda" else None
            model = DDP(model, device_ids=ddp_device_ids)
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

    for step in range(1, train_steps + 1):
        batch = next(data_iter)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timer.start()

        if not using_deepspeed:
            optimizer.zero_grad(set_to_none=True)

        if using_deepspeed:
            logits, l_aux, exp_counts = deepspeed_engine(input_ids)
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
                logits, l_aux, exp_counts = model(input_ids)
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
        exp_counts = aggregate_expert_counts(exp_counts)
        expert_counts_list = exp_counts.detach().cpu().tolist()
        expert_stats = compute_expert_stats(expert_counts_list)

        if device.type == "cuda":
            max_alloc = int(torch.cuda.max_memory_allocated(device))
            max_res = int(torch.cuda.max_memory_reserved(device))
        else:
            max_alloc = 0
            max_res = 0

        if step % log_every == 0:
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
