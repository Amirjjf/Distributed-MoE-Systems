#!/bin/bash
#  Fill the placeholders below for LUMI.
#SBATCH --job-name=phase1_4gpu
#SBATCH --account=ACCOUNT_PLACEHOLDER
#SBATCH --partition=PARTITION_PLACEHOLDER
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

ACCOUNT="${ACCOUNT:-ACCOUNT_PLACEHOLDER}"
PARTITION="${PARTITION:-PARTITION_PLACEHOLDER}"
TIME_LIMIT="${TIME_LIMIT:-00:45:00}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"

echo "ACCOUNT=$ACCOUNT PARTITION=$PARTITION TIME=$TIME_LIMIT"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "VENV_PATH=$VENV_PATH"

module purge
# TODO: adjust these for LUMI.
module load python
module load cuda

source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"
mkdir -p logs results

export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

torchrun --nproc_per_node=4 src/train.py \
  --config configs/base.json \
  --deepspeed \
  --ds_config configs/ds_config_moe.json \
  --run_name phase1_4gpu \
  --out_dir results
