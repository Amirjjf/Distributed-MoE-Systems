import json
import os
import numpy as np 
import argparse

MI250X_PEAK_TFLOPS = 383e12

def load_config(results_dir, run_name):
    path = os.path.join(results_dir, f"{run_name}_config_used.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"hidden_size": 512, "num_layers": 2, "num_experts": 8, "top_k": 1}

def load_results(results_dir: str) -> dict:
    runs = {}

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".jsonl"):
            continue

        run_name = fname.replace(".jsonl", "")
        steps = []
        seen = set()
        with open(os.path.join(results_dir, fname)) as f:
            for line in f:
                line = line.strip()
                if line:
                    s = json.loads(line)
                    if s["step"] not in seen:
                        steps.append(s)
                        seen.add(s["step"])

        runs[run_name] = steps
        cfg = load_config(results_dir, run_name)
        print(f"  Loaded {len(steps):>3} steps from {fname} "
              f"| H={cfg.get('hidden_size')} experts={cfg.get('num_experts')} "
              f"top_k={cfg.get('top_k')} world_size={steps[0]['world_size']}")
    return runs

def tokens_per_gpu(expert_counts: list, world_size: int) -> list:
    n = len(expert_counts)
    experts_per_gpu = n // world_size
    return [
        sum(expert_counts[i*experts_per_gpu:(i+1)*experts_per_gpu])
        for i in range(world_size)
    ]

def calibrate(steps_1gpu: list, H: int, alpha: int, num_layers: int) -> float:
    estimates = []
    for s in steps_1gpu:
        B_total  = sum(s["expert_counts"])
        fwd_time = s["forward_time_sec"]
        flops    = 4 * B_total * alpha * H**2 * num_layers
        estimates.append(flops / fwd_time)

    P_w = float(np.median(estimates))

    print(f"  P_w (median)   = {P_w/1e9:.1f} GFLOPs/sec")
    print(f"  P_w (range)    = {min(estimates)/1e9:.1f} – {max(estimates)/1e9:.1f} GFLOPs/sec")
    print(f"  % of MI250X peak = {100*P_w/MI250X_PEAK_TFLOPS:.2f}%")

    return P_w  

def predict_lat_comp(B_w_list: list, H: int, alpha: int,
                     P_w: float, num_layers: int = 1) -> dict:
    B     = np.array(B_w_list, dtype=float)
    B_max = float(np.max(B))
    B_avg = float(np.mean(B))

    lat_comp     = (4 * B_max * alpha * H**2 / P_w) * num_layers
    lat_balanced = (4 * B_avg * alpha * H**2 / P_w) * num_layers

    imbalance_factor = B_max / B_avg
    wasted_fraction  = (B_max - B_avg) / B_max
    savings_sec      = lat_comp - lat_balanced

    return {
        "lat_comp_sec":      lat_comp,
        "lat_balanced_sec":  lat_balanced,
        "imbalance_factor":  imbalance_factor,
        "slowest_GPU_wasted": wasted_fraction,
        "savings_sec":       savings_sec,
        "B_max":             B_max,
        "B_avg":             B_avg,
        "tokens_per_gpu":    B.tolist(),
    }

def validate(runs: dict, alpha: int, P_w: float):
    base_times = {}
    for run_name, steps in runs.items():
        if "1gpu" in run_name and "heavy" not in run_name:
            for s in steps:
                base_times[s["step"]] = s["forward_time_sec"]
    base_fallback = float(np.mean(list(base_times.values())))

    print(f"\n-- Validation results --")

    for run_name, steps in sorted(runs.items()):
        world_size = steps[0]["world_size"]
        if world_size == 1 and "heavy" not in run_name:
            continue

        cfg        = load_config(os.path.dirname(
                         next(iter(runs))), run_name) if False else {}

        predicted, measured, errors = [], [], []

        for s in steps:
            B_w     = tokens_per_gpu(s["expert_counts"], world_size)
            result  = predict_lat_comp(B_w, 512, alpha, P_w, 2)  
            imb_fac = result["imbalance_factor"]
            base    = base_times.get(s["step"], base_fallback)
            pred    = base * imb_fac
            real    = s["forward_time_sec"]
            predicted.append(pred)
            measured.append(real)
            errors.append(abs(pred - real) / real * 100)

        meas_arr = np.array(measured)
        pred_arr = np.array(predicted)
        ss_res   = np.sum((meas_arr - pred_arr) ** 2)
        ss_tot   = np.sum((meas_arr - np.mean(meas_arr)) ** 2)
        r2       = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        print(f"  {run_name}: steps={len(steps)}  "
              f"mean_err={np.mean(errors):.1f}%  "
              f"max_err={max(errors):.1f}%  "
              f"R²={r2:.3f}")

def should_rebalance(expert_counts: list, world_size: int, H: int,
                     alpha: int, P_w: float, num_layers: int,
                     threshold: float = 0.05) -> dict:
    B_w    = tokens_per_gpu(expert_counts, world_size)
    result = predict_lat_comp(B_w, H, alpha, P_w, num_layers)

    rebalance = result["slowest_GPU_wasted"] > threshold

    return {
        "rebalance":       rebalance,
        "wasted_pct":      result["slowest_GPU_wasted"] * 100,
        "imbalance_ratio": result["imbalance_factor"],
        "savings_ms":      result["savings_sec"] * 1000,
        "lat_pred_ms":     result["lat_comp_sec"] * 1000,
        "tokens_per_gpu":  result["tokens_per_gpu"],
        "reason": (
            f"{'REBALANCE' if rebalance else 'BALANCED'}: "
            f"imbalance={result['imbalance_factor']:.3f}x, "
            f"wasted={result['slowest_GPU_wasted']*100:.1f}% "
            f"({'>' if rebalance else '<='} {threshold*100:.0f}% threshold)"
        ),
    }

def print_summary(runs: dict, results_dir: str, alpha: int,
                  P_w: float, threshold: float = 0.05):
    print(f"\n-- Per-step Rebalance Decisions (every 5th step) --")
    print(f"{'Run':<30} {'Step':<6} {'Imbalance':<12} {'Wasted':<10} "
          f"{'Savings(ms)':<13} {'Decision'}")
    print("-" * 75)

    for run_name, steps in sorted(runs.items()):
        world_size = steps[0]["world_size"]
        if world_size == 1 and "heavy" not in run_name:
            continue

        cfg        = load_config(results_dir, run_name)
        H          = cfg.get("hidden_size", 512)
        num_layers = cfg.get("num_layers", 2)

        for s in steps[::5]:  
            dec = should_rebalance(
                s["expert_counts"], world_size,
                H, alpha, P_w, num_layers, threshold
            )
            print(f"  {run_name:<28} {s['step']:<6} "
                  f"{dec['imbalance_ratio']:.4f}x     "
                  f"{dec['wasted_pct']:.2f}%      "
                  f"{dec['savings_ms']:.3f}ms        "
                  f"{'REBALANCE' if dec['rebalance'] else 'no'}")

def main():
    parser = argparse.ArgumentParser(description="MoE computation cost model")
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()

    alpha = 4

    print("=" * 50)
    print("  MoE Computation Cost Model (FasterMoE 3.1)")
    print("=" * 50)
    print(f"\nLoading results from: {args.results_dir}")
    runs = load_results(args.results_dir)

    key_1gpu   = next(k for k in runs if "1gpu" in k and "heavy" not in k)
    cfg_1gpu   = load_config(args.results_dir, key_1gpu)
    H_1gpu     = cfg_1gpu.get("hidden_size", 512)
    nl_1gpu    = cfg_1gpu.get("num_layers", 2)

    print(f"\n-- Calibration (using {key_1gpu}) --")
    P_w = calibrate(runs[key_1gpu], H_1gpu, alpha, nl_1gpu)

    validate(runs, alpha, P_w)

    print_summary(runs, args.results_dir, alpha, P_w, args.threshold)

if __name__ == "__main__":
    main()