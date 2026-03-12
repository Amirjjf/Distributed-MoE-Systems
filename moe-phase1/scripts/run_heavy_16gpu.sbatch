#!/bin/bash

#SBATCH --job-name=phase2_heavy_16gpu

#SBATCH --account=project_462001047

#SBATCH --partition=standard-g

#SBATCH --time=00:30:00

#SBATCH --nodes=2

#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:mi250:8

#SBATCH --cpus-per-task=56

#SBATCH --mem=400G

#SBATCH --hint=nomultithread

#SBATCH --output=logs/%x_%j.out

#SBATCH --error=logs/%x_%j.err

#SBATCH --chdir=/scratch/project_462001047/ngop/moe-phase1/moe-phase1



set -euo pipefail



PROJECT_ROOT="$SLURM_SUBMIT_DIR"

VENV_PATH="$PROJECT_ROOT/.venv"

mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/results" "$PROJECT_ROOT/.miopen"



module purge

module use /appl/local/laifs/modules

module load lumi-aif-singularity-bindings



export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260216_093549/lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif



export NCCL_DEBUG=INFO

export MIOPEN_USER_DB_PATH="$PROJECT_ROOT/.miopen"

export MIOPEN_CACHE_DIR="$PROJECT_ROOT/.miopen"



export OMP_NUM_THREADS=7

export MKL_NUM_THREADS=$OMP_NUM_THREADS

export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS

export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS

export TORCH_COMPILE_DISABLE=1

export TORCHDYNAMO_DISABLE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MALLOC_TRIM_THRESHOLD_=0



# Multi-node torchrun setup

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export MASTER_PORT=29500

export WORLD_SIZE=$((SLURM_NNODES * 8))

export NODE_RANK=$SLURM_NODEID



srun --cpu-bind=cores singularity run "$SIF" bash -lc "

  set -euo pipefail

  cd '$PROJECT_ROOT'

  source '$VENV_PATH/bin/activate'

  echo allowed_cpus_in_job=\$(nproc)

  python -c 'import torch; print("visible_gpus:", torch.cuda.device_count())'

  torchrun 
    --nproc_per_node=8 
    --nnodes=2 
    --node_rank=\$SLURM_NODEID 
    --master_addr=\$MASTER_ADDR 
    --master_port=\$MASTER_PORT 
    src/train.py 
    --config configs/heavy.json 
    --deepspeed 
    --ds_config configs/ds_config_moe.json 
    --run_name phase2_heavy_16gpu 
    --out_dir results

"

