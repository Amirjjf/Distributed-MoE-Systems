#!/bin/bash
#SBATCH --job-name=phase2_step4_ds_map_8gpu
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
CONFIG="${CONFIG:-configs/heavy_phase2_step4_deepspeed_mapping_startup_real.json}"
RUN_NAME="${RUN_NAME:-phase2_step4_ds_map_8gpu}"
DS_CONFIG="${DS_CONFIG:-configs/ds_config_moe.json}"
INITIAL_MAP_PATH="${INITIAL_MAP_PATH:-}"
RUNTIME_CONFIG="results/${RUN_NAME}_runtime_config.json"
export CONFIG RUN_NAME DS_CONFIG INITIAL_MAP_PATH RUNTIME_CONFIG

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

srun --cpu-bind=cores /usr/bin/singularity run --env CONFIG="$CONFIG",RUN_NAME="$RUN_NAME",DS_CONFIG="$DS_CONFIG",INITIAL_MAP_PATH="$INITIAL_MAP_PATH",RUNTIME_CONFIG="$RUNTIME_CONFIG" "$SIF" bash -lc "  set -euo pipefail
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
  echo initial_map_path='${INITIAL_MAP_PATH:-<none>}'

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

  python - <<'PY'
import json
import os
from pathlib import Path

config_path = Path(os.environ['CONFIG'])
runtime_path = Path(os.environ['RUNTIME_CONFIG'])
initial_map = os.environ.get('INITIAL_MAP_PATH', '').strip()

with open(config_path, 'r', encoding='utf-8') as f:
    cfg = json.load(f)

if initial_map:
    cfg['deepspeed_initial_map_path'] = initial_map

runtime_path.parent.mkdir(parents=True, exist_ok=True)
with open(runtime_path, 'w', encoding='utf-8') as f:
    json.dump(cfg, f, indent=2)

print('runtime_config:', str(runtime_path))
print('runtime_deepspeed_initial_map_path:', cfg.get('deepspeed_initial_map_path'))
PY

  torchrun --standalone --nproc_per_node=\$NPROC src/train.py \
    --config '$RUNTIME_CONFIG' \
    --deepspeed \
    --ds_config '$DS_CONFIG' \
    --run_name '${RUN_NAME}_'\${NPROC}'gpu' \
    --out_dir results
"