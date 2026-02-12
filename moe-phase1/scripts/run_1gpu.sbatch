#!/bin/bash
#SBATCH --job-name=moe_1gpu_run
#SBATCH --account=YOUR_PROJECT_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out

set -e

# Load environment
module purge
module load python
module load cuda

# Activate virtual environment
source .venv/bin/activate

# Move to project directory
cd $SLURM_SUBMIT_DIR
mkdir -p logs results

# debugging info for distributed communication
export NCCL_DEBUG=INFO

# Launch training
torchrun --nproc_per_node=1 src/train.py \
  --config configs/base.json \
  --deepspeed \
  --ds_config configs/ds_config_moe.json \
  --run_name moe_1gpu_test \
  --out_dir results
