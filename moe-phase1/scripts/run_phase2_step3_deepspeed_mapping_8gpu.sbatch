#!/bin/bash
#SBATCH --job-name=phase2_step3_ds_map_8gpu
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
#SBATCH --chdir=/pfs/lustrep4/scratch/project_462001047/jafaria/moe-phase1

set -euo pipefail

PROJECT_ROOT="$SLURM_SUBMIT_DIR"
VENV_PATH="$PROJECT_ROOT/.venv"
CONFIG="${CONFIG:-configs/heavy_phase2_step3_deepspeed_mapping.json}"
RUN_NAME="${RUN_NAME:-phase2_step3_ds_map_8gpu}"
DS_CONFIG="${DS_CONFIG:-configs/ds_config_moe.json}"

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
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MALLOC_TRIM_THRESHOLD_=0

srun --cpu-bind=cores singularity run "$SIF" bash -lc "
  set -euo pipefail
  cd '$PROJECT_ROOT'
  source '$VENV_PATH/bin/activate'

  echo allowed_cpus_in_job=\$(nproc)

  python - <<'PY'
import os, torch
print('visible_gpus =', torch.cuda.device_count())
print('CUDA_VISIBLE_DEVICES =', os.environ.get('CUDA_VISIBLE_DEVICES'))
print('HIP_VISIBLE_DEVICES =', os.environ.get('HIP_VISIBLE_DEVICES'))
print('ROCR_VISIBLE_DEVICES =', os.environ.get('ROCR_VISIBLE_DEVICES'))
PY

  echo config='$CONFIG'
  echo run_name='$RUN_NAME'

  NPROC=\$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)

  if [ \"\$NPROC\" -lt 1 ]; then
    echo 'ERROR: No visible GPUs inside the container.'
    exit 1
  fi

  echo using_nproc_per_node=\$NPROC

  torchrun --standalone --nproc_per_node=\$NPROC src/train.py \
    --config '$CONFIG' \
    --deepspeed \
    --ds_config '$DS_CONFIG' \
    --run_name '${RUN_NAME}_'\${NPROC}'gpu' \
    --out_dir results
"