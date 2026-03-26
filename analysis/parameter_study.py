import sys

import json
import statistics
import itertools
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D


matplotlib.rcParams.update({
    'figure.dpi': 200,
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'legend.framealpha': 0.92,
    'legend.edgecolor': '#cccccc',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.18,
    'grid.linestyle': ':',
    'grid.color': '#888888',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
    'lines.linewidth': 1.6,
    'lines.markersize': 5,
})

DATA_FILES = {
    "phase1_1gpu": "results/phase1_1gpu.jsonl",
    "phase1_2gpu": "results/phase1_2gpu.jsonl",
    "phase1_4gpu": "results/phase1_4gpu.jsonl",
    "phase2_heavy_8gpu": "results/phase2_heavy_8gpu.jsonl",
    "phase2_step4_ds_map_8gpu": "results/phase2_step4_ds_map_8gpu_8gpu.jsonl",
    "phase2_heavy_16gpu": "results/phase2_heavy_16gpu.jsonl",
}

HIDDEN_SIZE = 512
NUM_LAYERS = 2
ALPHA = 4.0
TOKEN_SIZE_BYTES = HIDDEN_SIZE * 2

METRICS = ["wasted_fraction", "cv", "max_over_mean", "ema_wasted_fraction", "jain_fairness"]
EMA_BETA = 0.9         
 
THRESHOLDS = [0.03, 0.05, 0.08, 0.10]
INTERVALS = [10, 20, 50]
COOLDOWNS = [50, 100, 200]
STRATEGIES = ["round_robin", "load_sorted"]
LAMBDAS = [0.0, 0.5, 1.0]
 

RUN_COLORS = {
    'phase1_2gpu':              '#4393c3',
    'phase1_4gpu':              '#2166ac',
    'phase2_heavy_8gpu':        '#d6604d',
    'phase2_step4_ds_map_8gpu': '#b2182b',
}
RUN_LABELS = {
    'phase1_2gpu':              'Phase 1 — 2 GPU',
    'phase1_4gpu':              'Phase 1 — 4 GPU',
    'phase2_heavy_8gpu':        'Phase 2 — 8 GPU (heavy)',
    'phase2_step4_ds_map_8gpu': 'Phase 2 — 8 GPU (DS map)',
}
METRIC_COLORS = {
    'wasted_fraction':     '#d6604d',
    'cv':                  '#4393c3',
    'max_over_mean':       '#1a9850',
    'ema_wasted_fraction': '#9970ab',   
    'jain_fairness':       '#e08214',   
}
METRIC_LABELS = {
    'wasted_fraction':     'Wasted fraction',
    'cv':                  'CV',
    'max_over_mean':       'Max / mean',
    'ema_wasted_fraction': 'EMA wasted fraction (β=0.9)',
    'jain_fairness':       "Jain's fairness index",
}
STRAT_COLORS  = {'round_robin': '#e08214', 'load_sorted': '#2166ac'}
STRAT_LABELS  = {'round_robin': 'Round-robin', 'load_sorted': 'Load-sorted'}
METRIC_MARKER = {'wasted_fraction': 'o', 'cv': '^', 'max_over_mean': 's',
                 'ema_wasted_fraction': 'D', 'jain_fairness': 'P'}

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

def _compute_metrics(gpu_loads, P_w, H, L, alpha=4.0,
                     ema_wasted_state=None, ema_beta=EMA_BETA):
    if not gpu_loads or max(gpu_loads) == 0:
        return {"wasted_fraction": 0.0, "cv": 0.0, "max_over_mean": 0.0,
                "ema_wasted_fraction": 0.0, "jain_fairness": 0.0,
                "wasted_ms": 0.0, "max_load": 0.0, "mean_load": 0.0}
    B_max = max(gpu_loads)
    B_avg = statistics.mean(gpu_loads)
    std = statistics.pstdev(gpu_loads) if len(gpu_loads) > 1 else 0.0
    wasted_fraction = (B_max - B_avg) / B_max if B_max > 0 else 0.0
    cv = std / B_avg if B_avg > 0 else 0.0
    max_over_mean = B_max / B_avg if B_avg > 0 else 0.0
    _ema = ema_wasted_state if ema_wasted_state is not None else [wasted_fraction]
    _ema[0] = ema_beta * _ema[0] + (1.0 - ema_beta) * wasted_fraction
    ema_wasted = _ema[0]
    if ema_wasted_state is not None:
        ema_wasted_state[0] = _ema[0]
    sum_b = sum(gpu_loads)
    sum_b2 = sum(b * b for b in gpu_loads)
    n = len(gpu_loads)
    jain_J = (sum_b ** 2) / (n * sum_b2) if sum_b2 > 0 else 1.0
    jain_waste = 1.0 - jain_J
    t_max_ms = (4 * B_max * alpha * H**2 * L) / P_w * 1000
    t_avg_ms = (4 * B_avg * alpha * H**2 * L) / P_w * 1000
    return {
        "wasted_fraction": wasted_fraction,
        "cv": cv,
        "max_over_mean": max_over_mean,
        "ema_wasted_fraction": ema_wasted,
        "jain_fairness": jain_waste,
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
    mapping = [0] * n
    for eid, load in sorted(enumerate(expert_counts), key=lambda x: -x[1]):
        best = min(range(world_size), key=lambda g: gpu_loads[g])
        mapping[eid] = best
        gpu_loads[best] += load
    return mapping

def total_cost(expert_counts, mapping, world_size, P_w, H, L, lambda_comm, token_size_bytes):
    gpu_loads = partition_experts_to_gpus(expert_counts, mapping, world_size)
    metrics = _compute_metrics(gpu_loads, P_w, H, L)
    comm_ms = estimate_communication_cost(expert_counts, mapping, world_size, token_size_bytes)
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
    last_trigger = -(10**9)
    num_triggers = 0
    savings_ms_list, metric_vals, comm_cost_list, wasted_ms_list = [], [], [], []

    ema_state = [None]
 
    for row in rows:
        step = int(row['step'])
        expert_counts = [float(x) for x in row.get('expert_counts', [])]
        if not expert_counts:
            continue
 
        current_map = build_map(cfg.map_strategy, expert_counts, world_size_run)
        gpu_loads = partition_experts_to_gpus(expert_counts, current_map, world_size_run)
        if ema_state[0] is None:
            raw_wf = (max(gpu_loads) - statistics.mean(gpu_loads)) / max(gpu_loads) \
                     if max(gpu_loads) > 0 else 0.0
            ema_state[0] = raw_wf
        
        metrics = _compute_metrics(gpu_loads, P_w, H, L,
                               ema_wasted_state=ema_state)
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
 
multi_gpu_runs = [n for n, r in runs.items() if r[0].get('world_size', 1) >= 2]
# No map_strategy filter — average across both so EMA/Jain show signal
plot_df = df[
    (df['eval_interval'] == 20) & (df['cooldown'] == 100) &
    (df['lambda_comm'] == 0.0)
].copy()
 
fig, axes = plt.subplots(1, len(METRICS), figsize=(13, 3.0), sharey=False)
 
for ax, metric in zip(axes, METRICS):
    sub = plot_df[plot_df['metric'] == metric]
    for run in multi_gpu_runs:
        rdf = sub[sub['run'] == run].groupby('threshold')['trigger_rate'].mean().reset_index()
        if rdf.empty:
            continue
        ws = runs[run][0].get('world_size', 1)
        ax.plot(rdf['threshold'], rdf['trigger_rate'],
                marker='o', color=RUN_COLORS.get(run, '#888'),
                label=f'ws={ws}', linewidth=1.6, markersize=4)
 
    ax.set_title(METRIC_LABELS[metric], fontweight='bold', pad=5)
    ax.set_xlabel('Threshold θ')
    if metric == METRICS[0]:
        ax.set_ylabel('Trigger rate')
    ax.set_xlim(0.02, 0.112)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    ax.set_ylim(bottom=0)
    if metric == 'jain_fairness':
        ax.set_ylim(-0.001, 0.01)  # near-zero range to show flat line clearly

    if metric in ('ema_wasted_fraction', 'jain_fairness'):
        ax.set_facecolor('#faf5ff')
        extra = '\n(thresholds too high for this workload)' if metric == 'jain_fairness' else ''
        ax.set_title(METRIC_LABELS[metric] + '\n[P1 — new]' + extra,
                     fontweight='bold', pad=5, color='#7b3f9e', fontsize=8)
 
handles = [Line2D([0], [0], color=RUN_COLORS.get(r, '#888'), marker='o',
                  markersize=4, label=RUN_LABELS.get(r, r))
           for r in multi_gpu_runs]
fig.legend(handles=handles, loc='lower center', ncol=len(multi_gpu_runs),
           bbox_to_anchor=(0.5, -0.14), frameon=True)
fig.suptitle('Trigger rate vs threshold — by metric  '
             '(eval_interval=20, cooldown=100, avg across strategies, λ=0)',
             fontsize=10, fontweight='bold', y=1.02)
plt.tight_layout(w_pad=1.4)
plt.savefig("trigger_rate_vs_threshold.png", bbox_inches='tight')
plt.close()
print("Saved: trigger_rate_vs_threshold.png")
 

plot_df2 = df[
    (df['eval_interval'] == 20) & (df['cooldown'] == 100) &
    (df['threshold'] == 0.03) & (df['metric'] == 'max_over_mean') &
    (df['lambda_comm'] == 0.0)
].copy()
 
agg2     = plot_df2.groupby(['run', 'map_strategy'])['avg_savings_ms'].mean().unstack(fill_value=0)
run_order = [r for r in multi_gpu_runs if r in agg2.index]
x         = np.arange(len(run_order))
width     = 0.32
 
fig, ax = plt.subplots(figsize=(7, 3.6))
for i, strat in enumerate(['round_robin', 'load_sorted']):
    if strat not in agg2.columns:
        continue
    vals = [agg2.loc[r, strat] if r in agg2.index else 0.0 for r in run_order]
    bars = ax.bar(x + (i - 0.5) * width, vals, width,
                  color=STRAT_COLORS[strat], label=STRAT_LABELS[strat],
                  edgecolor='white', linewidth=0.5, zorder=3)
    for bar, val in zip(bars, vals):
        if val > 0.005:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom',
                    fontsize=7, color='#444')
 
ax.set_xticks(x)
ax.set_xticklabels([RUN_LABELS.get(r, r) for r in run_order], fontsize=7.5,
                   rotation=15, ha='right')
ax.set_ylabel('Avg predicted savings per trigger (ms)')
ax.set_title('Avg savings per trigger by map strategy\n'
             '(θ=0.03, metric=max_over_mean, eval_interval=20, cooldown=100, λ=0)',
             fontweight='bold')
ax.legend(loc='upper left', frameon=True)
ax.set_ylim(bottom=0)
ax.yaxis.grid(True, alpha=0.18, linestyle=':')
ax.set_axisbelow(True)
ax.annotate('load_sorted savings = 0:\nproposed map identical to current',
            xy=(x[0] + 0.5 * width, 0.005),
            xytext=(x[0] + 1.0, max(agg2.values.max() * 0.4, 0.05)),
            fontsize=7, color='#666',
            arrowprops=dict(arrowstyle='->', color='#aaa', lw=0.8))
plt.tight_layout()
plt.savefig("savings_vs_lambda_comm.png", bbox_inches='tight')
plt.close()
print("Saved: savings_vs_lambda_comm.png")
 
 
plot_df3 = df[
    (df['metric'] == 'max_over_mean') & (df['threshold'] == 0.05) &
    (df['map_strategy'] == 'round_robin') & (df['lambda_comm'] == 0.0)
].copy()
 
pivots = {}
for run in multi_gpu_runs:
    sub = plot_df3[plot_df3['run'] == run]
    pivots[run] = sub.groupby(['eval_interval', 'cooldown'])['avg_savings_ms'].mean().unstack()
 
vmin = 0
vmax = max(p.values.max() for p in pivots.values())
if vmax == 0:
    vmax = 1.0
 
fig, axes = plt.subplots(1, len(multi_gpu_runs),
                          figsize=(3.2 * len(multi_gpu_runs), 3.0))
if len(multi_gpu_runs) == 1:
    axes = [axes]
 
im_ref = None
for ax, run in zip(axes, multi_gpu_runs):
    pivot = pivots[run]
    ws    = runs[run][0].get('world_size', 1)
    im    = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd',
                      vmin=vmin, vmax=vmax, interpolation='nearest')
    im_ref = im
 
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index], fontsize=8)
    ax.set_xlabel('Cooldown (steps)', fontsize=8)
    if ax == axes[0]:
        ax.set_ylabel('Eval interval (steps)', fontsize=8)
    ax.set_title(RUN_LABELS.get(run, run), fontsize=8.5, fontweight='bold', pad=4)
 
    best_idx = np.unravel_index(np.argmax(pivot.values), pivot.values.shape)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val      = pivot.values[i, j]
            txt_col  = 'white' if val > vmax * 0.55 else '#333'
            weight   = 'bold' if (i, j) == best_idx else 'normal'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=txt_col, fontweight=weight)
    ax.add_patch(plt.Rectangle(
        (best_idx[1] - 0.48, best_idx[0] - 0.48), 0.96, 0.96,
        fill=False, edgecolor='#2166ac', linewidth=2.0, zorder=5
    ))
 
cbar = fig.colorbar(im_ref, ax=axes, shrink=0.82, pad=0.02)
cbar.set_label('Avg savings per trigger (ms)', fontsize=8)
cbar.ax.tick_params(labelsize=7)
fig.suptitle('Avg savings per trigger: eval interval × cooldown\n'
             '(metric=max_over_mean, θ=0.05, round_robin, λ=0)',
             fontsize=9.5, fontweight='bold', y=1.04)
plt.savefig("trigger_heatmap_interval_cooldown.png", bbox_inches='tight')
plt.close()
print("Saved: trigger_heatmap_interval_cooldown.png")
 
def pareto_frontier(trigger_rates, savings):
    n = len(trigger_rates)
    efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not efficient[i]:
            continue
        dominated = (
            (trigger_rates >= trigger_rates[i]) & (savings >= savings[i]) &
            ((trigger_rates > trigger_rates[i]) | (savings > savings[i]))
        )
        dominated[i] = False
        if dominated.any():
            efficient[i] = False
    return efficient
 
 
pareto_df = df[(df['world_size'] >= 2) & (df['num_triggers'] > 0)].copy()
 
fig, ax = plt.subplots(figsize=(7.5, 4.8))
 
for metric in METRICS:
    sub = pareto_df[pareto_df['metric'] == metric]
    ax.scatter(sub['trigger_rate'], sub['avg_savings_ms'],
               color=METRIC_COLORS[metric],
               marker=METRIC_MARKER[metric],
               alpha=0.22, s=14, linewidths=0, zorder=2,
               label=METRIC_LABELS[metric])
 
tr_all = pareto_df['trigger_rate'].values
sv_all = pareto_df['avg_savings_ms'].values
pf_mask = pareto_frontier(tr_all, sv_all)
pf_pts = pareto_df[pf_mask].sort_values('trigger_rate')
 
ax.scatter(pf_pts['trigger_rate'], pf_pts['avg_savings_ms'],
           color='#b2182b', marker='D', s=55, zorder=5,
           linewidths=0.8, edgecolors='white',
           label='Pareto-efficient config')
ax.step(pf_pts['trigger_rate'], pf_pts['avg_savings_ms'],
        where='post', color='#b2182b', linewidth=1.2,
        linestyle='--', alpha=0.7, zorder=4)
 
for _, row in pf_pts.iterrows():
    short = {'wasted_fraction': 'WF', 'cv': 'CV', 'max_over_mean': 'MoM',
             'ema_wasted_fraction': 'EMA', 'jain_fairness': 'Jain'}[row['metric']]
    ax.annotate(f"θ={row['threshold']:.2f}, {short}",
                xy=(row['trigger_rate'], row['avg_savings_ms']),
                xytext=(5, 3), textcoords='offset points',
                fontsize=6.5, color='#b2182b', zorder=6)
 
ax.set_xlabel('Trigger rate (triggers per training step)')
ax.set_ylabel('Avg predicted savings per trigger (ms)')
ax.set_title('Pareto frontier: trigger frequency vs savings per trigger\n'
             '(all multi-GPU runs, all parameter configurations)',
             fontweight='bold')
ax.text(0.02, 0.97, 'Low frequency,\nhigh savings\n(ideal)',
        transform=ax.transAxes, fontsize=7.5, va='top',
        color='#2166ac', style='italic')
ax.text(0.75, 0.06, 'High frequency,\nlow savings\n(noisy)',
        transform=ax.transAxes, fontsize=7.5, va='bottom',
        color='#888', style='italic')
ax.legend(loc='center right', fontsize=7.5, frameon=True)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig("pareto_front.png", bbox_inches='tight')
plt.close()
print("Saved: pareto_front.png")
 
 
best = (df[df['num_triggers'] >= 2]
        .sort_values('total_savings_ms', ascending=False)
        .iloc[0])
 
print("\n" + "=" * 55)
print("BEST CONFIGURATION FROM PARAMETER STUDY")
print("=" * 55)
for k, v in [("Run",          best['run']),
             ("Metric",       best['metric']),
             ("Threshold",    best['threshold']),
             ("Eval interval", f"{best['eval_interval']} steps"),
             ("Cooldown",     f"{best['cooldown']} steps"),
             ("Map strategy", best['map_strategy']),
             ("Lambda_comm",  best['lambda_comm'])]:
    print(f"  {k:<18} {v}")
print("-" * 55)
print(f"  Num triggers:        {best['num_triggers']}")
print(f"  Avg savings/trigger: {best['avg_savings_ms']:.2f} ms")
print(f"  Total savings:       {best['total_savings_ms']:.2f} ms")
print("=" * 55)
 
best_rows  = df[
    (df['metric'] == best['metric'])       &
    (df['threshold'] == best['threshold'])    &
    (df['eval_interval'] == best['eval_interval'])&
    (df['cooldown'] == best['cooldown'])     &
    (df['map_strategy'] == best['map_strategy']) &
    (df['run'] == best['run'])
]
avg_wasted = best_rows['avg_wasted_ms'].mean()
avg_comm   = best_rows['avg_comm_ms'].mean()
total_avg  = avg_wasted + avg_comm or 1
print(f"\nCost breakdown (avg per step):")
print(f"  Compute waste:  {avg_wasted:.2f} ms  ({100*avg_wasted/total_avg:.1f}%)")
print(f"  Communication:  {avg_comm:.2f} ms  ({100*avg_comm/total_avg:.1f}%)")
 
df.to_csv("parameter_study_results.csv", index=False)
print(f"\nExported {len(df)} rows to parameter_study_results.csv")
print("Figures saved: trigger_rate_vs_threshold, savings_vs_lambda_comm,")
print("               trigger_heatmap_interval_cooldown, pareto_front")
