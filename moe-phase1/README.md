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
