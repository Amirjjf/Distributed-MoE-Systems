#!/bin/bash
#SBATCH --job-name=phase2_heavy_16gpu
#SBATCH --account=project_462001047
#SBATCH --partition=standard-g
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
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

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260216_093549/lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif

export NCCL_DEBUG=INFO
export MIOPEN_USER_DB_PATH="$PROJECT_ROOT/.miopen"
export MIOPEN_CACHE_DIR="$PROJECT_ROOT/.miopen"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

srun --cpu-bind=cores singularity run "$SIF" bash -lc "
  set -euo pipefail
  cd '$PROJECT_ROOT'
  source '$VENV_PATH/bin/activate'

  python src/train.py \
    --config configs/heavy.json \
    --deepspeed \
    --ds_config configs/ds_config_moe.json \
    --run_name phase2_heavy_16gpu \
    --out_dir results
"