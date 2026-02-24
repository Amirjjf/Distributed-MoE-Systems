import json
import math
import statistics
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import torch
import torch.distributed as dist


class StepTimer:
    def __init__(self) -> None:
        self._start = 0.0
        self._forward_done = 0.0
        self._backward_done = 0.0
        self._optim_done = 0.0

    def start(self) -> None:
        self._start = time.perf_counter()

    def mark_forward_done(self) -> None:
        self._forward_done = time.perf_counter()

    def mark_backward_done(self) -> None:
        self._backward_done = time.perf_counter()

    def mark_optim_done(self) -> None:
        self._optim_done = time.perf_counter()

    def end(self) -> Dict[str, float]:
        end_t = time.perf_counter()
        return {
            "forward_time_sec": max(self._forward_done - self._start, 0.0),
            "backward_time_sec": max(self._backward_done - self._forward_done, 0.0),
            "optim_time_sec": max(self._optim_done - self._backward_done, 0.0),
            "step_time_sec": max(end_t - self._start, 1e-12),
        }


def aggregate_expert_counts(expert_counts: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        out = expert_counts.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out
    return expert_counts


def compute_expert_stats(counts: List[float]) -> Dict[str, float]:
    if not counts:
        return {"expert_mean": 0.0, "expert_std": 0.0, "expert_max_min_ratio": 0.0, "expert_cv": 0.0}
    mean_v = float(statistics.mean(counts))
    std_v = float(statistics.pstdev(counts)) if len(counts) > 1 else 0.0
    min_v = min(counts)
    max_v = max(counts)
    if min_v <= 0:
        max_min_ratio = float("inf") if max_v > 0 else 0.0
    else:
        max_min_ratio = float(max_v / min_v)
    cv = float(std_v / mean_v) if mean_v > 0 else 0.0
    return {
        "expert_mean": mean_v,
        "expert_std": std_v,
        "expert_max_min_ratio": max_min_ratio,
        "expert_cv": cv,
    }


class MetricLogger:
    def __init__(self, jsonl_path: str, is_main: bool, ma_window: int = 20) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.is_main = is_main
        self.step_time_window: Deque[float] = deque(maxlen=ma_window)
        self.logged_rows: List[Dict[str, Any]] = []

    def _safe_for_json(self, row: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
                out[k] = None
            else:
                out[k] = v
        return out

    def log(self, row: Dict[str, Any]) -> None:
        self.step_time_window.append(float(row["step_time_sec"]))
        row["step_time_avg_sec"] = float(sum(self.step_time_window) / len(self.step_time_window))
        serializable = self._safe_for_json(row)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(serializable) + "\n")
        self.logged_rows.append(serializable)

        if self.is_main:
            print(
                f"step={row['step']} "
                f"step_time={row['step_time_sec']:.4f}s "
                f"tok/s={row['tokens_per_sec']:.2f} "
                f"mem_alloc={row['cuda_max_memory_allocated']}"
            )

    def finalize_summary(
        self,
        summary_path: str,
        run_name: str,
        metadata: Dict[str, Any],
        world_size: int,
        tokens_per_step_global: int,
    ) -> None:
        rows = self.logged_rows
        if not rows:
            summary = {
                "run_name": run_name,
                "metadata": metadata,
                "world_size": world_size,
                "global_batch_tokens_per_step": tokens_per_step_global,
                "message": "no rows logged",
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, sort_keys=True)
            return

        tps = [float(r["tokens_per_sec"]) for r in rows]
        step_t = [float(r["step_time_sec"]) for r in rows]
        fwd_t = [float(r["forward_time_sec"]) for r in rows]
        bwd_t = [float(r["backward_time_sec"]) for r in rows]
        opt_t = [float(r["optim_time_sec"]) for r in rows]
        mem_alloc = [float(r["cuda_max_memory_allocated"]) for r in rows]
        mem_res = [float(r["cuda_max_memory_reserved"]) for r in rows]
        cv_vals = [float(r["expert_cv"]) for r in rows]
        ratio_vals = [
            float(r["expert_max_min_ratio"])
            for r in rows
            if r["expert_max_min_ratio"] is not None and not math.isinf(float(r["expert_max_min_ratio"]))
        ]

        summary = {
            "run_name": run_name,
            "metadata": metadata,
            "world_size": world_size,
            "global_batch_tokens_per_step": tokens_per_step_global,
            "throughput": {
                "tokens_per_sec_avg": float(statistics.mean(tps)),
                "tokens_per_sec_median": float(statistics.median(tps)),
                "tokens_per_sec_max": float(max(tps)),
                "steps_per_sec_avg": float(statistics.mean([1.0 / x for x in step_t])),
            },
            "timing": {
                "avg_step_time_sec": float(statistics.mean(step_t)),
                "avg_forward_time_sec": float(statistics.mean(fwd_t)),
                "avg_backward_time_sec": float(statistics.mean(bwd_t)),
                "avg_optim_time_sec": float(statistics.mean(opt_t)),
            },
            "memory": {
                "max_allocated_bytes": float(max(mem_alloc)),
                "max_reserved_bytes": float(max(mem_res)),
            },
            "expert_imbalance": {
                "cv_avg": float(statistics.mean(cv_vals)),
                "cv_max": float(max(cv_vals)),
                "max_min_ratio_avg": float(statistics.mean(ratio_vals)) if ratio_vals else None,
            },
            "num_logged_points": len(rows),
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

