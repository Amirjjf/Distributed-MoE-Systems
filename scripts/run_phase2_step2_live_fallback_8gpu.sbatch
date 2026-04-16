#!/bin/bash
#SBATCH --job-name=phase2_step2_live_fb_8gpu
#SBATCH --account=project_462001047
#SBATCH --partition=standard-g
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:mi250:8
#SBATCH --cpus-per-task=56
#SBATCH --mem=400G
#SBATCH --hint=nomultithread
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --chdir=/pfs/lustrep4/scratch/project_462001047/jafaria

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

srun --cpu-bind=cores singularity run "$SIF" bash -lc "
  set -euo pipefail
  cd '$PROJECT_ROOT'
  source '$VENV_PATH/bin/activate'

  echo allowed_cpus_in_job=\$(nproc)
  python -c 'import torch; print(\"visible_gpus:\", torch.cuda.device_count())'

  torchrun --standalone --nproc_per_node=8 src/train.py \
    --config configs/heavy_phase2_step2_live_fallback.json \
    --run_name phase2_step2_live_fb_8gpu \
    --out_dir results
"
