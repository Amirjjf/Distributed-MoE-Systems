# Reproducible MoE Benchmarking and Rebalancing on LUMI

## 1. Project Overview

This repository provides a reproducible MoE benchmarking and rebalancing workflow for distributed training on LUMI.

- Synthetic token data is used for controlled experiments.
- The training entry point is `src/train.py`.
- DeepSpeed runs use `configs/ds_config_moe.json`.
- Outputs are written to `results/` and logs are written to `logs/`.
- Reported metrics include throughput, timing, memory, expert imbalance, and rebalancing behavior.
- Both DeepSpeed MoE and fallback MoE paths are supported.

## 2. Project Structure

- `configs/`: experiment configs
- `scripts/`: LUMI `sbatch` launchers
- `src/`: training and utility code
- `results/`: JSONL logs, summaries, saved maps
- `analysis/`: analysis scripts and visuals
- `Documentation/`: reports and presentations

## 3. LUMI Assumptions / Working Directory

Assume the repository is already available on LUMI. Use the scratch project area as your main working location:

`/scratch/project_462001047/$USER`

Go to your project directory before running any step:

```powershell
cd /scratch/project_462001047/$USER/<project-directory>
pwd
ls
```

## 4. One-Time LUMI Setup

1. Load the required LUMI modules.

```powershell
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
```

2. Set the container image path.

```powershell
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260216_093549/lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif
```

3. Run a GPU sanity check using `srun`.

```powershell
srun -A project_462001047 -p small-g -N 1 -n 1 --gpus=1 -c 8 --time=00:10:00 --pty bash
```

4. Inside the allocation, verify GPU visibility from the container.

```powershell
cd /scratch/project_462001047/$USER/<project-directory>
singularity run "$SIF" python -c "import torch; print('cuda_available:', torch.cuda.is_available()); print('gpus:', torch.cuda.device_count()); print('name0:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NA')"
```

5. Create the virtual environment inside the project directory and install extras.

```powershell
singularity run "$SIF" bash -lc "cd $PWD && python -m venv .venv --system-site-packages"
singularity run "$SIF" bash -lc "cd $PWD && source .venv/bin/activate && pip install -U pip && pip install pandas numpy"
```

## 5. Phase 1: Baseline Benchmarking

Phase 1 is the baseline benchmarking stage.

- Configs: `configs/base.json`, `configs/ds_config_moe.json`

### 5.1 1 GPU

```powershell
sbatch scripts/run_1gpu.sbatch
```

Running job:

```powershell
squeue -u $USER
```

Expected summary:

- `results/phase1_1gpu_summary.json`

### 5.2 2 GPU

```powershell
sbatch scripts/run_2gpu.sbatch
```

Expected summary:

- `results/phase1_2gpu_summary.json`

### 5.3 4 GPU

```powershell
sbatch scripts/run_4gpu.sbatch
```

Expected summary:

- `results/phase1_4gpu_summary.json`

### 5.4 Monitoring

Monitor queue and logs (`<JOBID>` placeholder). The example below uses the 1 GPU job log pattern:

```powershell
squeue -u $USER
tail -n 200 logs/phase1_1gpu_<JOBID>.out
tail -n 200 logs/phase1_1gpu_<JOBID>.err
```

## 6. Phase 2: Advanced Experiments

### 6.1 Heavy Baseline 8 GPU

Purpose: baseline heavy DeepSpeed run on 8 GPUs.

- Config: `configs/heavy.json`
- Script: `scripts/run_heavy_8gpu.sbatch`
- Expected summary: `results/phase2_heavy_8gpu_summary.json`

Submit:

```powershell
sbatch scripts/run_heavy_8gpu.sbatch
```

### 6.2 Heavy Baseline 16 GPU

Purpose: baseline heavy DeepSpeed run on 16 GPUs.

- Config: `configs/heavy.json`
- Script: `scripts/run_heavy_16gpu.sbatch`
- Expected summary: `results/phase2_heavy_16gpu_summary.json`

Submit:

```powershell
sbatch scripts/run_heavy_16gpu.sbatch
```

### 6.3 Step 1 Planner Dry-Run

Purpose: planner dry-run on 8 GPUs.

- Config: `configs/heavy_phase2_step1.json`
- Script: `scripts/run_phase2_step1_8gpu.sbatch`
- Expected summary: `results/phase2_step1_8gpu_summary.json`

Submit:

```powershell
sbatch scripts/run_phase2_step1_8gpu.sbatch
```

### 6.4 Step 2 Fallback Live Apply

Purpose: fallback backend with live apply enabled.

This step uses the fallback path (not DeepSpeed mapping-aware EP).

- Config: `configs/heavy_phase2_step2_live_fallback.json`
- Script: `scripts/run_phase2_step2_live_fallback_8gpu.sbatch`
- Expected run name: `phase2_step2_live_fb_8gpu`
- Expected summary: `results/phase2_step2_live_fb_8gpu_summary.json`

Submit:

```powershell
sbatch scripts/run_phase2_step2_live_fallback_8gpu.sbatch
```

### 6.5 Step 3 DeepSpeed Mapping-Aware Run

Purpose: generate and save proposed DeepSpeed expert-placement maps for later use.

Step 3 runs mapping-aware logic and writes proposed map artifacts. It does not apply a new DeepSpeed map during the same job.

- Config: `configs/heavy_phase2_step3_deepspeed_mapping.json`
- Script: `scripts/run_phase2_step3_deepspeed_mapping_8gpu.sbatch`
- Supported overrides: `CONFIG`, `RUN_NAME`, `DS_CONFIG`

Default submit:

```powershell
sbatch scripts/run_phase2_step3_deepspeed_mapping_8gpu.sbatch
```

Default run name produced by script behavior:

- `phase2_step3_ds_map_8gpu_8gpu`

Expected Step 3 outputs:

- `results/phase2_step3_ds_map_8gpu_8gpu.jsonl`
- `results/phase2_step3_ds_map_8gpu_8gpu_summary.json`
- `results/phase2_step3_ds_map_8gpu_8gpu_deepspeed_next_map_step*.json`

Step 3 limitations:

- no in-job DeepSpeed remap/rebuild
- no live application of the proposed map in the same run
- proposed maps are saved for later startup-based use

Monitor:

```powershell
tail -n 200 logs/phase2_step3_ds_map_8gpu_<JOBID>.out
tail -n 200 logs/phase2_step3_ds_map_8gpu_<JOBID>.err
grep -n "ep_size_" logs/phase2_step3_ds_map_8gpu_<JOBID>.out | head
```

### 6.6 Inspecting the Proposed Map from Step 3

1. Run Step 3 first.

2. Inspect `deepspeed_pending_map_path` in the Step 3 JSONL.

```powershell
grep -n "deepspeed_pending_map_path" results/phase2_step3_ds_map_8gpu_8gpu.jsonl | tail -n 20
```

3. Use the discovered file path (for example `results/phase2_step3_ds_map_8gpu_8gpu_deepspeed_next_map_step1460.json`) as the Step 4 input map artifact.

### 6.7 Step 4 Startup Mapping Run

Purpose: start a new DeepSpeed run using a proposed map created by Step 3.

- Config: `configs/heavy_phase2_step4_deepspeed_mapping_startup_real.json`
- Script: `scripts/run_phase2_step4_deepspeed_mapping_8gpu.sbatch`
- Supported overrides: `CONFIG`, `RUN_NAME`, `DS_CONFIG`, `INITIAL_MAP_PATH`
- Runtime config written to: `results/${RUN_NAME}_runtime_config.json`

Step 4 is the execution step. It consumes a saved map from Step 3 and applies that map at startup through `INITIAL_MAP_PATH`.

Default submit:

```powershell
sbatch scripts/run_phase2_step4_deepspeed_mapping_8gpu.sbatch
```

Monitor:

```powershell
tail -n 200 logs/phase2_step4_ds_map_8gpu_<JOBID>.out
tail -n 200 logs/phase2_step4_ds_map_8gpu_<JOBID>.err
grep -n "local_expert_ids_active_startup" logs/phase2_step4_ds_map_8gpu_<JOBID>.out
```

### 6.8 Step 3 -> Step 4 Handoff (Apply Saved Map at Startup)

Use a map file produced by Step 3 when launching Step 4:

```powershell
sbatch --export=ALL,INITIAL_MAP_PATH=results/phase2_step3_ds_map_8gpu_8gpu_deepspeed_next_map_step1460.json,RUN_NAME=phase2_step4_ds_map_restart scripts/run_phase2_step4_deepspeed_mapping_8gpu.sbatch
```

Handoff flow:

1. Step 3 proposes and saves a DeepSpeed map.
2. Step 4 receives that saved map through `INITIAL_MAP_PATH`.
3. Step 4 starts a new run with that proposed map applied at startup.

Expected outputs:

- `phase2_step4_ds_map_restart_8gpu`
- `results/phase2_step4_ds_map_restart_8gpu_summary.json`

## 7. Where Logs and Results Are Written

- Job logs:
  - `logs/*.out`
  - `logs/*.err`
- Result artifacts:
  - `results/*.jsonl`
  - `results/*_summary.json`
  - `results/*_config_used.json`
  - `results/*_deepspeed_next_map_step*.json`

## 8. Optional Verification Checks / Fields to Inspect

Use these checks to validate runtime behavior after jobs complete:

```powershell
grep -n "deepspeed_pending_map_path" results/phase2_step3_ds_map_8gpu_8gpu.jsonl | tail -n 5
grep -n "deepspeed_startup_map_apply_reason" results/phase2_step4_ds_map_restart_8gpu.jsonl | tail -n 5
grep -n "local_expert_ids_active_startup" logs/phase2_step4_ds_map_8gpu_<JOBID>.out
```

Recommended JSONL fields to inspect:

- `deepspeed_pending_map_path`
- `deepspeed_startup_map_apply_reason`
- `local_expert_ids_active_startup`
- `tokens_per_sec`
- `step_time_sec`
- `cuda_max_memory_allocated`
- `cuda_max_memory_reserved`
- `expert_std`
- `expert_cv`
- `expert_max_min_ratio`

### **AI Usage Disclosure**:
>We have used AI tools in particular ChatGPT and Claude for several aspects:
> 1. finding articles and summarizing them and learning more abot the MoE concepts, Rebalancing Methods, etc.  
> 2. for debugging the programming related issues in the code for example fixing python errors etc.
> 3. to help us write the README.md file in a much more structured way so that the instructions are more clear. 
> 4. Assisting in generating and improving some technical code and configuration scripts. AI tools were used to help refine components, configuration files, and testing scripts. All generated code was reviewed, tested, and fully integrated by us.
