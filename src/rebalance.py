import json
import statistics
from collections import deque
from cost_model import CostModel, tokens_per_gpu_from_map, compute_cost_metrics, propose_cost_aware_mapping
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, TypeVar

T = TypeVar("T")

_cost_model: "CostModel | None" = None

def init_cost_model( steps_1gpu: list, H: int, num_layers: int) -> None:
    global _cost_model
    _cost_model = CostModel.from_1gpu_steps(steps_1gpu, H, num_layers)


def build_initial_expert_map(num_experts: int, world_size: int) -> List[int]:
    if world_size <= 0:
        return [0 for _ in range(max(num_experts, 0))]
    return [expert_id % world_size for expert_id in range(max(num_experts, 0))]


def build_deepspeed_ep_map(num_experts: int, ep_size: int) -> List[int]:
    n = max(int(num_experts), 0)
    if ep_size <= 0:
        return [0 for _ in range(n)]
    if n == 0:
        return []
    if n % ep_size != 0:
        return build_initial_expert_map(n, ep_size)

    per_rank = n // ep_size
    mapping = [0 for _ in range(n)]
    for expert_id in range(n):
        mapping[expert_id] = min(expert_id // per_rank, ep_size - 1)
    return mapping


def count_mapping_changes(old_map: List[int], new_map: List[int]) -> int:
    n = min(len(old_map), len(new_map))
    changed = sum(1 for i in range(n) if int(old_map[i]) != int(new_map[i]))
    changed += abs(len(old_map) - len(new_map))
    return int(changed)


def get_local_experts_for_rank(expert_to_gpu: List[int], rank: int) -> List[int]:
    return [expert_id for expert_id, gpu_id in enumerate(expert_to_gpu) if int(gpu_id) == int(rank)]


def build_global_to_local_expert_index(local_expert_ids: List[int]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for local_idx, global_expert_id in enumerate(local_expert_ids):
        out[int(global_expert_id)] = int(local_idx)
    return out


def build_rank_local_expert_modules(
    hidden_size: int,
    local_expert_ids: List[int],
    expert_builder: Callable[[int, int], T],
) -> List[T]:
    modules: List[T] = []
    for global_expert_id in local_expert_ids:
        modules.append(expert_builder(int(hidden_size), int(global_expert_id)))
    return modules


def build_deepspeed_startup_layout(expert_to_gpu_map: List[int], ep_size: int) -> Dict[str, List[Any]]:
    n = len(expert_to_gpu_map)
    e = int(ep_size)
    if e <= 0:
        e = 1

    rank_to_global_expert_ids: List[List[int]] = [[] for _ in range(e)]
    for global_expert_id, gpu_id in enumerate(expert_to_gpu_map):
        r = int(gpu_id)
        if r < 0 or r >= e:
            r = 0
        rank_to_global_expert_ids[r].append(int(global_expert_id))

    internal_to_global_expert_ids: List[int] = []
    for r in range(e):
        internal_to_global_expert_ids.extend(rank_to_global_expert_ids[r])

    if len(internal_to_global_expert_ids) != n:
        seen = {int(x) for x in internal_to_global_expert_ids}
        for global_expert_id in range(n):
            if global_expert_id not in seen:
                internal_to_global_expert_ids.append(int(global_expert_id))
        internal_to_global_expert_ids = internal_to_global_expert_ids[:n]

    global_to_internal_expert_ids = [-1 for _ in range(n)]
    for internal_expert_id, global_expert_id in enumerate(internal_to_global_expert_ids):
        g = int(global_expert_id)
        if 0 <= g < n:
            global_to_internal_expert_ids[g] = int(internal_expert_id)
    for global_expert_id in range(n):
        if global_to_internal_expert_ids[global_expert_id] < 0:
            global_to_internal_expert_ids[global_expert_id] = int(global_expert_id)

    return {
        "internal_to_global_expert_ids": [int(x) for x in internal_to_global_expert_ids],
        "global_to_internal_expert_ids": [int(x) for x in global_to_internal_expert_ids],
        "rank_to_global_expert_ids": [[int(x) for x in row] for row in rank_to_global_expert_ids],
    }


def load_expert_map_json(path: str, num_experts: int, default_map: Optional[List[int]] = None) -> List[int]:
    fallback = default_map[:] if default_map is not None else [0 for _ in range(max(int(num_experts), 0))]
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return fallback

    raw_map: Optional[List[Any]] = None
    if isinstance(payload, list):
        raw_map = payload
    elif isinstance(payload, dict):
        maybe_map = payload.get("expert_to_gpu_map")
        if isinstance(maybe_map, list):
            raw_map = maybe_map

    if raw_map is None:
        return fallback

    values = [int(x) for x in raw_map]
    target = max(int(num_experts), 0)
    if len(values) < target:
        values += fallback[len(values):target]
    if len(values) > target:
        values = values[:target]
    return values


def save_expert_map_json(path: str, expert_to_gpu_map: List[int], metadata: Optional[Dict[str, Any]] = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"expert_to_gpu_map": [int(x) for x in expert_to_gpu_map]}
    if metadata is not None:
        payload["metadata"] = metadata
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def is_map_ep_compatible(expert_to_gpu_map: List[int], ep_size: int) -> bool:
    if ep_size <= 0:
        return False
    if len(expert_to_gpu_map) % ep_size != 0:
        return False

    target_per_rank = len(expert_to_gpu_map) // ep_size
    counts = [0 for _ in range(ep_size)]
    for gpu_id in expert_to_gpu_map:
        g = int(gpu_id)
        if g < 0 or g >= ep_size:
            return False
        counts[g] += 1
    return all(c == target_per_rank for c in counts)


def project_map_to_ep_compatible(expert_to_gpu_map: List[int], ep_size: int) -> List[int]:
    n = len(expert_to_gpu_map)
    if n == 0:
        return []
    if ep_size <= 0:
        return [0 for _ in range(n)]
    if n % ep_size != 0:
        return build_deepspeed_ep_map(n, ep_size)

    target_per_rank = n // ep_size
    projected = [-1 for _ in range(n)]
    counts = [0 for _ in range(ep_size)]

    for expert_id, gpu_id in enumerate(expert_to_gpu_map):
        g = int(gpu_id)
        if 0 <= g < ep_size and counts[g] < target_per_rank:
            projected[expert_id] = g
            counts[g] += 1

    fill_order: List[int] = []
    for rank in range(ep_size):
        need = target_per_rank - counts[rank]
        if need > 0:
            fill_order.extend([rank] * need)

    fill_ptr = 0
    for expert_id in range(n):
        if projected[expert_id] >= 0:
            continue
        if fill_ptr >= len(fill_order):
            projected[expert_id] = 0
        else:
            projected[expert_id] = fill_order[fill_ptr]
            fill_ptr += 1

    return [int(x) for x in projected]


class ExpertLoadHistory:
    def __init__(self, history_size: int, use_ema: bool, ema_beta: float, num_experts: int) -> None:
        self.history_size = max(1, int(history_size))
        self.use_ema = bool(use_ema)
        self.ema_beta = min(max(float(ema_beta), 0.0), 0.9999)
        self.num_experts = max(1, int(num_experts))
        self._history: Deque[List[float]] = deque(maxlen=self.history_size)
        self._ema_state: Optional[List[float]] = None

    def _fit_num_experts(self, expert_counts: List[float]) -> List[float]:
        values = [float(x) for x in expert_counts]
        if len(values) == self.num_experts:
            return values
        if len(values) > self.num_experts:
            return values[: self.num_experts]
        return values + [0.0 for _ in range(self.num_experts - len(values))]

    def update(self, expert_counts: List[float]) -> None:
        values = self._fit_num_experts(expert_counts)
        self._history.append(values)

        if self.use_ema:
            if self._ema_state is None:
                self._ema_state = values[:]
            else:
                beta = self.ema_beta
                self._ema_state = [
                    beta * old + (1.0 - beta) * new for old, new in zip(self._ema_state, values)
                ]

    def is_ready(self) -> bool:
        if len(self._history) == 0:
            return False
        if self.use_ema:
            return self._ema_state is not None
        return len(self._history) >= self.history_size

    def get_smoothed_load(self) -> List[float]:
        if len(self._history) == 0:
            return [0.0 for _ in range(self.num_experts)]
        if self.use_ema and self._ema_state is not None:
            return self._ema_state[:]

        n = float(len(self._history))
        smoothed = [0.0 for _ in range(self.num_experts)]
        for row in self._history:
            for i, value in enumerate(row):
                smoothed[i] += value
        return [v / n for v in smoothed]


def estimate_gpu_load(expert_loads: List[float], expert_to_gpu: List[int], world_size: int) -> List[float]:
    if world_size <= 0:
        return []
    return tokens_per_gpu_from_map(expert_loads, expert_to_gpu, world_size)


def estimate_remote_assignments(expert_loads: List[float], expert_to_gpu: List[int], rank: int) -> float:
    n = min(len(expert_loads), len(expert_to_gpu))
    remote = 0.0
    for expert_id in range(n):
        if int(expert_to_gpu[expert_id]) != int(rank):
            remote += float(expert_loads[expert_id])
    return float(remote)


def compute_load_metrics(gpu_loads: List[float]) -> Dict[str, float]:
    if _cost_model is not None:
        return compute_cost_metrics(gpu_loads, _cost_model.P_w, _cost_model.H, _cost_model.nu)
    
    if not gpu_loads:
        return {
            "max_over_mean": 0.0,
            "cv": 0.0,
            "wasted_fraction": 0.0,
            "max_load": 0.0,
            "mean_load": 0.0,
        }

    max_load = float(max(gpu_loads))
    mean_load = float(statistics.mean(gpu_loads))
    std_load = float(statistics.pstdev(gpu_loads)) if len(gpu_loads) > 1 else 0.0

    max_over_mean = float(max_load / mean_load) if mean_load > 0 else 0.0
    cv = float(std_load / mean_load) if mean_load > 0 else 0.0
    wasted_fraction = float((max_load - mean_load) / max_load) if max_load > 0 else 0.0

    return {
        "max_over_mean": max_over_mean,
        "cv": cv,
        "wasted_fraction": wasted_fraction,
        "max_load": max_load,
        "mean_load": mean_load,
    }


def should_rebalance_now(
    metrics: Dict[str, float],
    metric_name: str,
    threshold: float,
    step: int,
    min_steps: int,
    cooldown: int,
    last_trigger_step: int,
) -> Tuple[bool, str]:
    metric_key = str(metric_name)
    if metric_key not in metrics:
        return False, f"metric_not_found:{metric_key}"

    if step < int(min_steps):
        return False, f"step<{int(min_steps)}"

    cooldown_steps = int(cooldown)
    if step - int(last_trigger_step) < cooldown_steps:
        remain = cooldown_steps - (step - int(last_trigger_step))
        return False, f"cooldown_active:{remain}"

    value = float(metrics[metric_key])
    th = float(threshold)
    if value > th:
        return True, f"{metric_key}={value:.4f}>{th:.4f}"
    return False, f"{metric_key}={value:.4f}<={th:.4f}"


def propose_rebalanced_mapping(expert_loads: List[float], world_size: int) -> Dict[str, object]:
    if _cost_model is not None:
        return propose_cost_aware_mapping(expert_loads, world_size,
                                           _cost_model.P_w, _cost_model.H,
                                           _cost_model.num_layers)    
    num_experts = len(expert_loads)
    if world_size <= 0:
        return {
            "expert_to_gpu_map": [0]*num_experts,
            "gpu_loads": [],
            "metrics": compute_load_metrics([]),
        }

    gpu_loads = [0.0] * world_size
    mapping = [0] * num_experts
    ordered   = sorted(enumerate(expert_loads), key=lambda x: (-x[1], x[0]))
    for expert_id, load in ordered:
        min_gpu = min(range(world_size), key=lambda g: (gpu_loads[g], g))
        mapping[expert_id]  = min_gpu
        gpu_loads[min_gpu] += load

    return {
        "expert_to_gpu_map": mapping,
        "gpu_loads": gpu_loads,
        "metrics": compute_load_metrics(gpu_loads),
    }


def summarize_rebalance_decision(
    step: int,
    triggered: bool,
    metric_name: str,
    current_value: Optional[float],
    proposed_value: Optional[float],
    expected_improvement: Optional[float],
    reason: str,
) -> str:
    cur = "na" if current_value is None else f"{float(current_value):.4f}"
    prop = "na" if proposed_value is None else f"{float(proposed_value):.4f}"
    impr = "na" if expected_improvement is None else f"{float(expected_improvement):.4f}"
    action = "TRIGGER" if triggered else "SKIP"
    return (
        f"{action} step={step} metric={metric_name} current={cur} "
        f"proposed={prop} improvement={impr} reason={reason}"
    )


class RebalanceManager:
    def __init__(self, initial_map: List[int]) -> None:
        self._active_map = [int(x) for x in initial_map]
        self.event_count = 0
        self.applied_history: List[Dict[str, Any]] = []

    def get_active_map(self) -> List[int]:
        return self._active_map[:]

    def apply_rebalanced_mapping(self, new_map: List[int], step: int, backend: str, reason: str) -> Dict[str, Any]:
        prev = self.get_active_map()
        next_map = [int(x) for x in new_map]
        moved = count_mapping_changes(prev, next_map)
        applied = moved > 0

        if applied:
            self._active_map = next_map
            self.event_count += 1

        event = {
            "applied": applied,
            "event_id": self.event_count if applied else None,
            "step": int(step),
            "backend": str(backend),
            "reason": str(reason),
            "num_experts_moved": int(moved),
            "previous_map": prev,
            "active_map": self.get_active_map(),
        }
        if applied:
            self.applied_history.append(event)
        return event
