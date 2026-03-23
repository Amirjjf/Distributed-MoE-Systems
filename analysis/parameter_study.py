import sys

import json
import statistics
import itertools
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import pandas as pd 
import matplotlib.pyplot as plt 

sys.path.insert(0, 'src')
from src.cost_model import tokens_per_gpu_from_map, compute_cost_metrics, propose_cost_aware_mapping, calibrate_P_w

plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11

DATA_FILES = {
    "phase1_1gpu": "results/phase1_1gpu.jsonl",
    "phase1_2gpu": "results/phase1_2gpu.jsonl",
    "phase1_4gpu": "results/phase1_4gpu.jsonl",
    "phase2_heavy_8gpu": "results/phase2_heavy_8gpu.jsonl",
    "phase2_step4_ds_map_8gpu": "results/phase2_step4_ds_map_8gpu_8gpu.jsonl",
}

HIDDEN_SIZE = 512
NUM_LAYERS = 2
ALPHA = 4.0
TOKEN_SIZE_BYTES = HIDDEN_SIZE * 2

METRICS = ["wasted_fraction", "cv", "max_over_mean"]
THRESHOLDS = [0.03, 0.05, 0.08, 0.10]
INTERVALS = [10, 20, 50]
COOLDOWNS = [50, 100, 200]
STRATEGIES = ["round_robin", "load_sorted"]
LAMBDAS = [0.0, 0.5, 1.0]

def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

runs = {}
for name, path in DATA_FILES.items():
    p = Path(path)
    if not p.exists():
        print(f"{path} not found, skipping")
        continue
    runs[name] = load_jsonl(str(p))
    print(f"Loaded {name}: {len(runs[name])} steps")

def calibrate_P_w(rows_1gpu: List[Dict], hidden_size: int, num_layers: int, alpha: float) -> float:
    pw_samples = []
    for row in rows_1gpu:
        t_fwd = float(row.get('forward_time_sec', 0))
        if t_fwd <= 0:
            continue
        B_total = sum(float(x) for x in row.get('expert_counts', []))
        if B_total <= 0:
            continue
        F = 4.0 * B_total * alpha * (hidden_size ** 2) * num_layers
        pw_samples.append(F / t_fwd)
    return float(statistics.median(pw_samples)) if pw_samples else 1e12  

if 'phase1_1gpu' in runs:
    P_w = calibrate_P_w(runs['phase1_1gpu'], HIDDEN_SIZE, NUM_LAYERS, ALPHA)
else:
    P_w = calibrate_P_w(list(runs.values())[0], HIDDEN_SIZE, NUM_LAYERS, ALPHA)
print(f"Calibrated P_w = {P_w:.4e} FLOPs/sec")

def partition_experts_to_gpus(expert_counts, expert_to_gpu, world_size):
    gpu_loads = [0.0] * world_size
    for eid, load in enumerate(expert_counts):
        gpu_loads[int(expert_to_gpu[eid]) % world_size] += float(load)
    return gpu_loads

def compute_cost_metrics(gpu_loads, P_w, H, L, alpha=4.0):
    if not gpu_loads or max(gpu_loads) == 0:
        return {"wasted_fraction": 0.0, "cv": 0.0, "max_over_mean": 0.0,
                "wasted_ms": 0.0, "max_load": 0.0, "mean_load": 0.0}
    B_max = max(gpu_loads)
    B_avg = statistics.mean(gpu_loads)
    std   = statistics.pstdev(gpu_loads) if len(gpu_loads) > 1 else 0.0
    wasted_fraction = (B_max - B_avg) / B_max if B_max > 0 else 0.0
    cv = std / B_avg if B_avg > 0 else 0.0
    max_over_mean = B_max / B_avg if B_avg > 0 else 0.0
    t_max_ms = (4 * B_max * alpha * H**2 * L) / P_w * 1000
    t_avg_ms = (4 * B_avg * alpha * H**2 * L) / P_w * 1000
    return {
        "wasted_fraction": wasted_fraction,
        "cv": cv,
        "max_over_mean": max_over_mean,
        "wasted_ms": t_max_ms - t_avg_ms,
        "max_load": B_max,
        "mean_load": B_avg,
    }
 
def estimate_communication_cost(expert_counts, expert_to_gpu, world_size, token_size_bytes,
                                 bandwidth_bytes_per_sec=200e9, latency_sec=1e-6):
    if world_size <= 1:
        return 0.0
    remote_tokens = 0.0
    for eid, load in enumerate(expert_counts):
        owner = int(expert_to_gpu[eid]) % world_size
        if owner != 0:
            remote_tokens += float(load)
    return (latency_sec + remote_tokens * token_size_bytes / bandwidth_bytes_per_sec) * 1000

def build_map(strategy, expert_counts, world_size):
    n = len(expert_counts)
    if strategy == "round_robin":
        return [i % world_size for i in range(n)]
    gpu_loads = [0.0] * world_size
    mapping   = [0] * n
    for eid, load in sorted(enumerate(expert_counts), key=lambda x: -x[1]):
        best = min(range(world_size), key=lambda g: gpu_loads[g])
        mapping[eid] = best
        gpu_loads[best] += load
    return mapping

def total_cost(expert_counts, mapping, world_size, P_w, H, L, lambda_comm, token_size_bytes):
    gpu_loads = partition_experts_to_gpus(expert_counts, mapping, world_size)
    metrics   = compute_cost_metrics(gpu_loads, P_w, H, L)
    comm_ms   = estimate_communication_cost(expert_counts, mapping, world_size, token_size_bytes)
    return metrics["wasted_ms"] + lambda_comm * comm_ms

@dataclass
class StudyConfig:
    metric: str
    threshold: float
    eval_interval: int
    cooldown: int
    map_strategy: str
    lambda_comm: float

def simulate_planner(rows, cfg, P_w, H, L, token_size_bytes, min_steps=20):
    world_size_run = rows[0].get('world_size', 1)
    last_trigger   = -(10**9)
    num_triggers   = 0
    savings_ms_list, metric_vals, comm_cost_list, wasted_ms_list = [], [], [], []
 
    for row in rows:
        step          = int(row['step'])
        expert_counts = [float(x) for x in row.get('expert_counts', [])]
        if not expert_counts:
            continue
 
        current_map = build_map(cfg.map_strategy, expert_counts, world_size_run)
        gpu_loads = partition_experts_to_gpus(expert_counts, current_map, world_size_run)
        metrics = compute_cost_metrics(gpu_loads, P_w, H, L)
        comm_ms = estimate_communication_cost(expert_counts, current_map,
                                                   world_size_run, token_size_bytes)
        metric_val = metrics.get(cfg.metric, 0.0)
        metric_vals.append(metric_val)
        wasted_ms_list.append(metrics['wasted_ms'])
        comm_cost_list.append(comm_ms)
 
        if step % cfg.eval_interval != 0: continue
        if step < min_steps: continue
        if step - last_trigger < cfg.cooldown: continue
        if metric_val <= cfg.threshold: continue

        proposed_map   = build_map('load_sorted', expert_counts, world_size_run)
        current_cost   = total_cost(expert_counts, current_map,  world_size_run,
                                     P_w, H, L, cfg.lambda_comm, token_size_bytes)
        proposed_cost  = total_cost(expert_counts, proposed_map, world_size_run,
                                     P_w, H, L, cfg.lambda_comm, token_size_bytes)
        savings_ms_list.append(max(0.0, current_cost - proposed_cost))
        num_triggers += 1
        last_trigger = step
 
    total_steps = len(rows)
    return {
        "metric": cfg.metric,
        "threshold": cfg.threshold,
        "eval_interval": cfg.eval_interval,
        "cooldown": cfg.cooldown,
        "map_strategy": cfg.map_strategy,
        "lambda_comm": cfg.lambda_comm,
        "num_triggers": num_triggers,
        "trigger_rate": num_triggers / total_steps if total_steps > 0 else 0.0,
        "avg_savings_ms": statistics.mean(savings_ms_list) if savings_ms_list else 0.0,
        "total_savings_ms": sum(savings_ms_list),
        "avg_metric_val": statistics.mean(metric_vals)    if metric_vals    else 0.0,
        "avg_wasted_ms": statistics.mean(wasted_ms_list) if wasted_ms_list else 0.0,
        "avg_comm_ms": statistics.mean(comm_cost_list) if comm_cost_list else 0.0,
    }

all_results = []
for run_name, rows in runs.items():
    ws = rows[0].get('world_size', 1)
    if ws < 2:
        print(f"Skipping {run_name} (world_size=1)")
        continue
    print(f"Sweeping {run_name} (world_size={ws})...")
    for metric, threshold, interval, cooldown, strategy, lam in itertools.product(
        METRICS, THRESHOLDS, INTERVALS, COOLDOWNS, STRATEGIES, LAMBDAS
    ):
        cfg = StudyConfig(metric, threshold, interval, cooldown, strategy, lam)
        result = simulate_planner(rows, cfg, P_w, HIDDEN_SIZE, NUM_LAYERS, TOKEN_SIZE_BYTES)
        result["run"] = run_name
        result["world_size"] = ws
        all_results.append(result)
 
df = pd.DataFrame(all_results)
print(f"\nTotal configurations evaluated: {len(df)}")

display_cols = ["run", "metric", "threshold", "eval_interval", "cooldown",
                "map_strategy", "lambda_comm", "num_triggers", "avg_savings_ms", "total_savings_ms"]
top = (df[df['num_triggers'] > 0]
       .sort_values('total_savings_ms', ascending=False)
       .head(20)[display_cols]
       .reset_index(drop=True))
top['avg_savings_ms']   = top['avg_savings_ms'].round(2)
top['total_savings_ms'] = top['total_savings_ms'].round(2)
print("\nTop 20 configurations by total predicted savings:")
print(top.to_string())

plot_df = df[
    (df['eval_interval'] == 20) & (df['cooldown'] == 100) &
    (df['map_strategy'] == 'load_sorted') & (df['lambda_comm'] == 0.0)
].copy()
 
multi_gpu_runs = [n for n, r in runs.items() if r[0].get('world_size', 1) >= 2]
fig, axes = plt.subplots(1, max(1, len(multi_gpu_runs)), figsize=(7 * len(multi_gpu_runs), 4))
if not hasattr(axes, '__iter__'):
    axes = [axes]
 
colors = {'wasted_fraction': '#e63946', 'cv': '#457b9d', 'max_over_mean': '#2a9d8f'}
for ax, run_name in zip(axes, multi_gpu_runs):
    ws  = runs[run_name][0].get('world_size', 1)
    sub = plot_df[plot_df['run'] == run_name]
    for metric in METRICS:
        mdf = sub[sub['metric'] == metric].groupby('threshold')['trigger_rate'].mean().reset_index()
        ax.plot(mdf['threshold'], mdf['trigger_rate'],
                marker='o', label=metric, color=colors[metric], linewidth=2)
    ax.axvline(x=0.05, color='orange', linestyle='--', alpha=0.7, label='θ=0.05 baseline')
    ax.set(title=f"{run_name} (ws={ws})", xlabel="Threshold", ylabel="Trigger rate")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
 
fig.suptitle("Trigger rate vs threshold by metric", fontweight='bold')
plt.tight_layout()
plt.savefig("trigger_rate_vs_threshold.png", bbox_inches='tight')
print("Saved: trigger_rate_vs_threshold.png")


plot_df2 = df[
    (df['eval_interval'] == 20) & (df['cooldown'] == 100) &
    (df['threshold'] == 0.05)  & (df['metric'] == 'wasted_fraction')
].copy()
 
fig, axes = plt.subplots(1, max(1, len(multi_gpu_runs)), figsize=(7 * len(multi_gpu_runs), 4))
if not hasattr(axes, '__iter__'):
    axes = [axes]
 
strat_colors = {'round_robin': '#e76f51', 'load_sorted': '#264653'}
for ax, run_name in zip(axes, multi_gpu_runs):
    ws  = runs[run_name][0].get('world_size', 1)
    sub = plot_df2[plot_df2['run'] == run_name]
    for strat in STRATEGIES:
        sdf = sub[sub['map_strategy'] == strat].groupby('lambda_comm')['avg_savings_ms'].mean().reset_index()
        ax.plot(sdf['lambda_comm'], sdf['avg_savings_ms'],
                marker='s', label=strat, color=strat_colors[strat], linewidth=2)
    ax.set(title=f"{run_name} (ws={ws})", xlabel="Lambda_comm", ylabel="Avg Savings (ms)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
 
fig.suptitle("Predicted savings vs lambda_comm by map strategy", fontweight='bold')
plt.tight_layout()
plt.savefig("savings_vs_lambda_comm.png", bbox_inches='tight')
print("Saved: savings_vs_lambda_comm.png")

plot_df3 = df[
    (df['metric'] == 'max_over_mean') & (df['threshold'] == 0.05) &
    (df['map_strategy'] == 'load_sorted') & (df['lambda_comm'] == 0.0)
].copy()
 
fig, axes = plt.subplots(1, max(1, len(multi_gpu_runs)), figsize=(7 * len(multi_gpu_runs), 5))
if not hasattr(axes, '__iter__'):
    axes = [axes]
 
for ax, run_name in zip(axes, multi_gpu_runs):
    ws    = runs[run_name][0].get('world_size', 1)
    sub   = plot_df3[plot_df3['run'] == run_name]
    pivot = sub.groupby(['eval_interval', 'cooldown'])['num_triggers'].mean().unstack()
    im    = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"cd={c}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"ev={i}" for i in pivot.index])
    ax.set_title(f"{run_name} (ws={ws})")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.0f}",
                    ha='center', va='center', fontsize=10)
    plt.colorbar(im, ax=ax, label='Num triggers')
 
fig.suptitle("Trigger count: Eval interval vs cooldown heatmap", fontweight='bold')
plt.tight_layout()
plt.savefig("trigger_heatmap_interval_cooldown.png", bbox_inches='tight')
print("Saved: trigger_heatmap_interval_cooldown.png")

best = (df[df['num_triggers'] >= 2]
        .sort_values('total_savings_ms', ascending=False)
        .iloc[0])
 
print("\n" + "=" * 55)
print("BEST CONFIGURATION FROM PARAMETER STUDY")
print("=" * 55)
for k, v in [("Run", best['run']), ("Metric", best['metric']),
             ("Threshold", best['threshold']), ("Eval interval", f"{best['eval_interval']} steps"),
             ("Cooldown", f"{best['cooldown']} steps"), ("Map strategy", best['map_strategy']),
             ("Lambda_comm", best['lambda_comm'])]:
    print(f"  {k:<18} {v}")
print("-" * 55)
print(f"  Num triggers:        {best['num_triggers']}")
print(f"  Avg savings/trigger: {best['avg_savings_ms']:.2f} ms")
print(f"  Total savings:       {best['total_savings_ms']:.2f} ms")
print("=" * 55)
 
best_rows  = df[(df['metric'] == best['metric']) & (df['threshold'] == best['threshold']) &
                (df['eval_interval'] == best['eval_interval']) & (df['cooldown'] == best['cooldown']) &
                (df['map_strategy'] == best['map_strategy']) & (df['run'] == best['run'])]
avg_wasted = best_rows['avg_wasted_ms'].mean()
avg_comm   = best_rows['avg_comm_ms'].mean()
total_avg  = avg_wasted + avg_comm or 1
print(f"\nCost breakdown (avg per step):")
print(f"  Compute waste:  {avg_wasted:.2f} ms  ({100*avg_wasted/total_avg:.1f}%)")
print(f"  Communication:  {avg_comm:.2f} ms  ({100*avg_comm/total_avg:.1f}%)")

df.to_csv("parameter_study_results.csv", index=False)
print(f"\nExported {len(df)} rows to parameter_study_results.csv")