import statistics
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple


def build_initial_expert_map(num_experts: int, world_size: int) -> List[int]:
    if world_size <= 0:
        return [0 for _ in range(max(num_experts, 0))]
    return [expert_id % world_size for expert_id in range(max(num_experts, 0))]


def count_mapping_changes(old_map: List[int], new_map: List[int]) -> int:
    n = min(len(old_map), len(new_map))
    changed = sum(1 for i in range(n) if int(old_map[i]) != int(new_map[i]))
    changed += abs(len(old_map) - len(new_map))
    return int(changed)


def get_local_experts_for_rank(expert_to_gpu: List[int], rank: int) -> List[int]:
    return [expert_id for expert_id, gpu_id in enumerate(expert_to_gpu) if int(gpu_id) == int(rank)]


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
    gpu_loads = [0.0 for _ in range(world_size)]
    n = min(len(expert_loads), len(expert_to_gpu))
    for expert_id in range(n):
        gpu_id = int(expert_to_gpu[expert_id])
        if 0 <= gpu_id < world_size:
            gpu_loads[gpu_id] += float(expert_loads[expert_id])
    return gpu_loads


def estimate_remote_assignments(expert_loads: List[float], expert_to_gpu: List[int], rank: int) -> float:
    n = min(len(expert_loads), len(expert_to_gpu))
    remote = 0.0
    for expert_id in range(n):
        if int(expert_to_gpu[expert_id]) != int(rank):
            remote += float(expert_loads[expert_id])
    return float(remote)


def compute_load_metrics(gpu_loads: List[float]) -> Dict[str, float]:
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
    num_experts = len(expert_loads)
    if world_size <= 0:
        return {
            "expert_to_gpu_map": [0 for _ in range(num_experts)],
            "gpu_loads": [],
            "metrics": compute_load_metrics([]),
        }

    gpu_loads = [0.0 for _ in range(world_size)]
    mapping = [0 for _ in range(num_experts)]

    ordered = sorted(
        [(idx, float(load)) for idx, load in enumerate(expert_loads)],
        key=lambda item: (-item[1], item[0]),
    )

    for expert_id, load in ordered:
        min_gpu = min(range(world_size), key=lambda gpu_id: (gpu_loads[gpu_id], gpu_id))
        mapping[expert_id] = min_gpu
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
