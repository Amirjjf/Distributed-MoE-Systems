# MoE Phase 1 (Benchmarking + Setup)

This folder is a reproducible Phase 1 baseline for distributed MoE training on LUMI.
It uses synthetic token data and logs throughput, memory, expert load balance, and timing.

## Folder layout

```text
moe-phase1/
  configs/
  logs/
  results/
  scripts/
  src/
```

## Phase 1 – Running MoE Project on LUMI (Step-by-Step Log)

### 0️ Assumptions (read once)
- You have a LUMI account with allocation:
  project_462001047
- Project directory name:
  moe-phase1
- Main training script:
  src/train.py
- DeepSpeed config:
  configs/ds_config_moe.json
- You will run everything from SCRATCH after copying.

### 1️ Upload project in VSCode (HOME)

VSCode opened at:
  /users/jafaria/

Project path in home:
  ~/moe-benchmark/moe-phase1

Verify structure (on login node):
  cd ~/moe-benchmark/moe-phase1
  ls


### 2️ Move project to SCRATCH

From login node:
  mkdir -p /scratch/project_462001047/$USER

Copy project to scratch:
  rsync -a ~/moe-benchmark/moe-phase1/ /scratch/project_462001047/$USER/moe-phase1/

Switch to scratch (this is your main working directory):
  cd /scratch/project_462001047/$USER/moe-phase1
  pwd
  ls

From now on, all runs happen here.


### 3️ Check Slurm account and partitions

Check usable Slurm accounts:
  sacctmgr -Pn show assoc user=$USER format=Account,Partition,QOS | head -n 50

Confirmed account:
  project_462001047

Check available GPU partitions:
  sinfo | egrep 'g|G' | head -n 30

Typical usage:
- small-g  (quick 1 GPU tests)
- standard-g (full-node + longer runs)


### 4️ Open an interactive GPU shell (sanity check)

From login node:
  srun -A project_462001047 -p small-g -N 1 -n 1 --gpus=1 -c 8 --time=00:10:00 --pty bash

You land on a compute node:
  jafaria@nidXXXXX:~


### 5️ Load LUMI AI container bindings

On the compute node:
  module use /appl/local/laifs/modules
  module load lumi-aif-singularity-bindings


### 6️ Set container image (LUMI AI Factory)

On the compute node:
  export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260216_093549/lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif

Verify container:
  singularity run "$SIF" python -V
  singularity run "$SIF" python -c "import torch; print(torch.__version__)"


### 7️ GPU sanity check (inside allocation)

On the compute node:
  cd /scratch/project_462001047/$USER/moe-phase1

  singularity run "$SIF" python -c "import torch; print('cuda_available:', torch.cuda.is_available()); print('gpus:', torch.cuda.device_count()); print('name0:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NA')"

Expected:
  cuda_available: True
  gpus: 1
  name0: AMD Instinct MI250X


### 8️ Create Python venv inside the project (in SCRATCH)

On the compute node:
  cd /scratch/project_462001047/$USER/moe-phase1

Create venv using container Python, reuse container site-packages:
  singularity run "$SIF" bash -lc "cd $PWD && python -m venv .venv --system-site-packages"

Install small extras (if needed):
  singularity run "$SIF" bash -lc "cd $PWD && source .venv/bin/activate && pip install -U pip && pip install pandas numpy"

Verify:
  singularity run "$SIF" bash -lc "cd $PWD && source .venv/bin/activate && python -c 'import pandas, numpy; print(\"deps ok\")'"



### 9️ Exit interactive session (back to login node)

  exit


### 10 Submit the 1-GPU job (login node)

From:
  /scratch/project_462001047/$USER/moe-phase1

Submit:
  sbatch scripts/run_1gpu.sbatch

Check queue:
  squeue -u $USER


### 11 Monitor logs (login node)

When you get a job id, e.g. 163XXXXX:

  tail -n 200 logs/phase1_1gpu_163XXXXX.out
  tail -n 200 logs/phase1_1gpu_163XXXXX.err

Success indicators in .out:
- cuda_available: True
- training steps progressing
- "Saved run summary to results/phase1_1gpu_summary.json"


### 12 Verify results produced (login node)

  cd /scratch/project_462001047/$USER/moe-phase1
  ls -lah results
  cat results/phase1_1gpu_summary.json | head -n 80

## Phase 2 Step 1 (Dry-Run Rebalancing Planner)

This step adds **planning-only** routing-aware rebalancing during training.

What it does:
- tracks recent expert loads with EMA or moving average
- estimates GPU load from a logical expert-to-GPU map
- checks imbalance with `wasted_fraction`
- proposes a better map with a simple greedy heuristic
- logs current map/load and proposed map/load

What dry-run means:
- no live expert migration
- no expert weight movement between ranks
- no DeepSpeed MoE internals changed at runtime
- training continues normally after each planner decision

### New config keys (used in training)

In `configs/base.json` and `configs/heavy.json`:
- `enable_rebalance_planner`
- `rebalance_eval_interval`
- `rebalance_history_size`
- `rebalance_threshold`
- `rebalance_metric`
- `rebalance_min_steps`
- `rebalance_cooldown`
- `rebalance_use_ema`
- `rebalance_ema_beta`
- `rebalance_dry_run`

Dedicated Step 1 config:
- `configs/heavy_phase2_step1.json` (planner enabled, dry-run enabled)

### Run locally (quick fallback smoke run)

1. From project root:
```bash
cd moe-phase1
```

2. Run with planner disabled (baseline behavior):
```bash
python src/train.py --config configs/base.json --run_name smoke_planner_off --out_dir results
```

3. Run with planner enabled (dry-run):
```bash
python src/train.py --config configs/heavy_phase2_step1.json --run_name smoke_planner_on --out_dir results
```

Note: on CPU this can be slow with heavy config. For a quick test, copy the config and reduce `train_steps`, `hidden_size`, and `num_layers`.

### Run on LUMI (8 GPU dry-run planner)

1. Submit dedicated Step 1 job:
```bash
sbatch scripts/run_phase2_step1_8gpu.sbatch
```

2. Monitor logs:
```bash
tail -n 200 logs/phase2_step1_8gpu_<JOBID>.out
tail -n 200 logs/phase2_step1_8gpu_<JOBID>.err
```

### Output files and where to look

For run name `X`, outputs are:
- `results/X.jsonl` (step logs)
- `results/X_summary.json` (aggregated summary)
- `results/X_config_used.json` (effective config saved at run start)

Planner fields in `X.jsonl`:
- `rebalance_evaluated_this_step`
- `rebalance_triggered`
- `rebalance_reason`
- `rebalance_summary`
- `expert_to_gpu_map_current`
- `proposed_expert_to_gpu_map`
- `predicted_gpu_loads_current`
- `predicted_gpu_loads_proposed`
- `rebalance_metric_current`
- `rebalance_metric_proposed`
- `rebalance_expected_improvement`
- `smoothed_expert_loads`

Quick inspection examples:
```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("results/phase2_step1_8gpu.jsonl")
for line in path.open():
    row = json.loads(line)
    if row.get("rebalance_evaluated_this_step"):
        print(
            row["step"],
            row["rebalance_triggered"],
            row["rebalance_reason"],
            row["rebalance_metric_current"],
            row["rebalance_metric_proposed"],
        )
PY
```

### Difference vs real live migration

Step 1 only computes and logs "what should be remapped".
It does **not** apply remapping in the model runtime.
Live migration (moving expert weights and switching placement during training) is a later phase.

## Phase 2 Step 2 (Live Rebalancing in Fallback Backend)

Step 2 adds **live application** of rebalance decisions for the fallback backend.

What Step 2 adds vs Step 1:
- planner can still trigger and propose mappings
- when `rebalance_dry_run=false` and backend is fallback, proposed mapping becomes active
- next training steps run with the updated active map
- logs now show trigger vs apply, moved experts, event IDs, and remote/local assignment stats

### What is real vs simulated in Step 2

Real in this project:
- active `expert_to_gpu_map` is runtime state
- map updates happen live during training when conditions are met
- fallback forward path uses active ownership map for local vs remote expert handling
- local/remote assignment counters change after map updates

Simulated (honestly):
- experts are still physically present on each process in fallback path
- remote expert execution is simulated (dispatch accounted, no local expert backprop for remote-owned experts)
- this is not full DeepSpeed expert migration across ranks

DeepSpeed status:
- planner still runs
- live apply is intentionally blocked
- logs use reason: `triggered_but_live_apply_not_supported_for_deepspeed_yet`

### Step 2 configs

- `configs/heavy_phase2_step2_live_fallback.json`:
  - planner on
  - `rebalance_dry_run=false`
  - fallback backend (`use_deepspeed_moe=false`)

New practical flags used:
- `rebalance_apply_live_fallback`
- `rebalance_log_remote_stats`
- `rebalance_min_expected_improvement`

### Run Step 2 locally (small fallback test)

1. Make a small local config (copy `configs/base.json`, set `enable_rebalance_planner=true`, `rebalance_dry_run=false`, small `train_steps`).
2. Run fallback training:
```bash
python src/train.py --config <your_small_step2_config>.json --run_name step2_live_local --out_dir results
```
3. Confirm at least one apply event in JSONL:
```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("results/step2_live_local.jsonl")
for line in path.open():
    row = json.loads(line)
    if row.get("rebalance_applied"):
        print("applied_step:", row["step"], "event:", row["rebalance_event_id"], "moved:", row["rebalance_num_experts_moved"])
        break
PY
```

### Run Step 2 on LUMI (8 GPU fallback live apply)

Submit:
```bash
sbatch scripts/run_phase2_step2_live_fallback_8gpu.sbatch
```

Monitor:
```bash
tail -n 200 logs/phase2_step2_live_fb_8gpu_<JOBID>.out
tail -n 200 logs/phase2_step2_live_fb_8gpu_<JOBID>.err
```

### Step 2 fields to inspect in JSONL

Planner/apply:
- `rebalance_triggered`
- `rebalance_applied`
- `rebalance_apply_reason`
- `rebalance_apply_backend`
- `rebalance_event_id`
- `rebalance_num_experts_moved`
- `expert_to_gpu_map_current`
- `expert_to_gpu_map_previous`
- `expert_to_gpu_map_active`

Runtime remote/local behavior:
- `local_token_assignments`
- `remote_token_assignments`
- `remote_fraction`
- `communication_proxy_runtime`
- `communication_proxy_current`
- `communication_proxy_proposed`
- `communication_proxy_improvement`

### Current limitations (important)

- Live apply works only for fallback backend in this step.
- DeepSpeed backend stays planner-only for now.
- Fallback distributed dispatch is a practical simulation, not full expert migration.
