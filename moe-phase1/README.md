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

## Setup

```bash
cd moe-phase1
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch deepspeed numpy pandas
```

`pandas` is optional. It is only used if you want to inspect `phase1_table.csv` more easily.

## Quick local test (1 GPU)

```bash
cd moe-phase1
source .venv/bin/activate
torchrun --nproc_per_node=1 src/train.py \
  --config configs/base.json \
  --deepspeed \
  --ds_config configs/ds_config_moe.json \
  --run_name smoke1 \
  --out_dir results
```

If DeepSpeed import/init fails, training automatically falls back to a simple PyTorch MoE path and still logs the same metrics.

## Submit on LUMI

Remember to edit placeholders in each script first:
- `ACCOUNT_PLACEHOLDER`
- `PARTITION_PLACEHOLDER`
- `PROJECT_ROOT`
- `VENV_PATH`

Then submit:

```bash
cd moe-phase1
sbatch scripts/run_1gpu.sbatch
sbatch scripts/run_2gpu.sbatch
sbatch scripts/run_4gpu.sbatch
```

## Outputs

For each run name:
- `results/<run_name>.jsonl` (metrics every log interval)
- `results/<run_name>_summary.json` (aggregates)
- `results/<run_name>_config_used.json` (exact config used)

SLURM logs:
- `logs/<job_name>_<job_id>.out`

## Metrics (short meaning)

- `tokens_per_sec`: global tokens processed per second.
- `step_time_sec`: total step time.
- `forward/backward/optim_time_sec`: timing breakdown (communication overhead proxy).
- `cuda_max_memory_allocated` and `cuda_max_memory_reserved`: GPU memory peaks.
- `expert_counts`: tokens routed to each expert.
- `expert_cv` and `expert_max_min_ratio`: load imbalance indicators.

## Compare runs

After 1/2/4 GPU runs finish:

```bash
cd moe-phase1
python src/collect_results.py --results_dir results --out_csv results/phase1_table.csv
```

This prints a small table and writes `results/phase1_table.csv` with:
- GPUs (`world_size`)
- avg tokens/sec
- max memory
- imbalance CV
- avg step time
- scaling efficiency (`throughput_N / (N * throughput_1GPU)`)

## Troubleshooting

- DeepSpeed not found: install it or use fallback path (already automatic).
- bf16 unsupported: script falls back to fp32.
- NCCL issues: set `NCCL_DEBUG=INFO`, check GPU visibility, and check that your partition supports requested GPU count.

