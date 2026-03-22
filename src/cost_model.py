import statistics
from typing import Dict, List

import numpy as np 

ALPHA=4
MI250X_PEAK_TFLOPS= 383e12

def calibrate_P_w(steps_1gpu: List[Dict], H:int, 
                    num_layers: int, alpha: int = ALPHA) -> float:

    estimates = []
    for s in steps_1gpu:
        t_fwd   = float(s.get("forward_time_sec", 0))
        if t_fwd <= 0:
            continue
        B_total = sum(float(x) for x in s.get("expert_counts", []))
        if B_total <= 0:
            continue
        flops = 4 * B_total * alpha * (H ** 2) * num_layers
        estimates.append(flops / t_fwd)
 
    if not estimates:
        return 1e12  
 
    P_w = float(np.median(estimates))
    print(f"[CostModel] P_w (median) = {P_w/1e9:.1f} GFLOPs/sec "
          f"| range = {min(estimates)/1e9:.1f}–{max(estimates)/1e9:.1f} "
          f"| {100*P_w/MI250X_PEAK_TFLOPS:.2f}% of MI250X peak")
    return P_w

def tokens_per_gpu_from_map(expert_counts: List[float], expert_to_gpu: List[int],
                            world_size: int) -> List[float]:

    if world_size <= 0:
        return []
    gpu_loads = [0.0]* world_size
    n =  min(len(expert_counts), len(expert_to_gpu))

    for expert_id in range(n):
        gpu = int(expert_to_gpu[expert_id]) % world_size
        gpu_loads[gpu] += float(expert_counts[expert_id])
    
    return gpu_loads


def compute_cost_metrics(gpu_loads: List[float], P_w: float,
                            H: int, num_layers: int, alpha: int = ALPHA) -> Dict[str, float]:

    if not gpu_loads or max(gpu_loads) == 0:
        return {
            "wasted_fraction": 0.0,
            "cv": 0.0,
            "max_over_mean": 0.0,
            "wasted_ms": 0.0,
            "lat_comp_ms": 0.0,
            "lat_balanced_ms":  0.0,
            "imbalance_factor": 0.0,
            "max_load": 0.0,
            "mean_load": 0.0,
        }
    
    B_max = float(max(gpu_loads))
    B_avg = float(statistics.mean(gpu_loads))
    std = float(statistics.pstdev(gpu_loads)) if len(gpu_loads) > 1 else 0.0

    wasted_fraction = (B_max - B_avg)/B_max if B_max > 0 else 0.0
    cv = std/B_avg if B_avg > 0 else 0.0
    max_over_mean = B_max/B_avg if B_avg > 0 else 0.0
    imbalance_factor = B_max/B_avg if B_avg > 0 else 1.0

    lat_comp_ms     = (4* B_max * alpha * H**2 * num_layers)/P_w * 1000
    lat_balanced_ms = (4 * B_avg * alpha * H**2 * num_layers)/P_w * 1000
    wasted_ms       = lat_comp_ms - lat_balanced_ms

    return {
            "wasted_fraction": wasted_fraction,
            "cv": cv,
            "max_over_mean": max_over_mean,
            "wasted_ms": wasted_ms,
            "lat_comp_ms": lat_comp_ms,
            "lat_balanced_ms": lat_balanced_ms,
            "imbalance_factor": imbalance_factor,
            "max_load": B_max,
            "mean_load": B_avg,
        }

def propose_cost_aware_mapping(expert_counts: List[float], world_size: int,
                                P_w: float, H: int, num_layers: int, 
                                alpha: int = ALPHA) -> Dict:

    num_experts = len(expert_counts)
    if world_size <= 0:
        return{
            "expert_to_gpu_map": [0] * num_experts,
            "gpu_loads": [],
            "metrics": compute_cost_metrics([], P_w, H, num_layers),
        }

    gpu_loads = [0.0] * world_size
    mapping = [0] * num_experts

    ordered = sorted(enumerate(expert_counts), key=lambda x: -float(x[1]))
    for expert_id, load in ordered:
        best_gpu = min(range(world_size), key=lambda g: gpu_loads[g])
        mapping[expert_id]  = best_gpu
        gpu_loads[best_gpu] += float(load)
 
    metrics = compute_cost_metrics(gpu_loads, P_w, H, num_layers, alpha)
 
    return {
        "expert_to_gpu_map": mapping,
        "gpu_loads": gpu_loads,
        "metrics": metrics,
    }

class CostModel:
    def __init__(self, P_w: float, H: int, num_layers: int, alpha: int = ALPHA):
        self.P_w = float(P_w)
        self.H = int(H)
        self.num_layers = int(num_layers)
        self.alpha = int(alpha)
    
    @classmethod

    def from_1gpu_steps(cls, steps_1gpu: List[Dict], H: int,
                         num_layers: int, alpha: int = ALPHA) -> "CostModel":
        P_w = calibrate_P_w(steps_1gpu, H, num_layers, alpha)
        return cls(P_w, H, num_layers, alpha)
 
    @classmethod
    def from_P_w(cls, P_w: float, H: int, num_layers: int,
                  alpha: int = ALPHA) -> "CostModel":
        return cls(P_w, H, num_layers, alpha)
 
    def gpu_loads(self, expert_counts: List[float],
                  expert_to_gpu: List[int],
                  world_size: int) -> List[float]:
        return tokens_per_gpu_from_map(expert_counts, expert_to_gpu, world_size)
 
    def metrics(self, gpu_loads: List[float]) -> Dict[str, float]:
        return compute_cost_metrics(gpu_loads, self.P_w,
                                     self.H, self.num_layers, self.alpha)
 
    def propose(self, expert_counts: List[float], world_size: int) -> Dict:
        return propose_cost_aware_mapping(expert_counts, world_size,
                                           self.P_w, self.H,
                                           self.num_layers, self.alpha)
 
