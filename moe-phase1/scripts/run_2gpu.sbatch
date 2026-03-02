#!/bin/bash
#SBATCH --job-name=phase1_2gpu
#SBATCH --account=project_462001047
#SBATCH --partition=small-g
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=14
#SBATCH --chdir=/pfs/lustrep4/scratch/project_462001047/jafaria/moe-phase1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Project paths
PROJECT_ROOT="$SLURM_SUBMIT_DIR"
VENV_PATH="$PROJECT_ROOT/.venv"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "VENV_PATH=$VENV_PATH"
echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA}"

mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/results" "$PROJECT_ROOT/.miopen"

# LUMI container bindings
module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# Container image
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260216_093549/lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif

# ROCm/MIOpen cache
export MIOPEN_USER_DB_PATH="$PROJECT_ROOT/.miopen"
export MIOPEN_CACHE_DIR="$PROJECT_ROOT/.miopen"

# debug
export NCCL_DEBUG=INFO

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

srun singularity run "$SIF" bash -lc "
  set -euo pipefail
  cd '$PROJECT_ROOT'
  source '$VENV_PATH/bin/activate'

  python -c 'import torch; print(\"cuda_available:\", torch.cuda.is_available()); print(\"device_count:\", torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NA\")'

  torchrun --nproc_per_node=2 src/train.py \
    --config configs/base.json \
    --deepspeed \
    --ds_config configs/ds_config_moe.json \
    --run_name phase1_2gpu \
    --out_dir results
"