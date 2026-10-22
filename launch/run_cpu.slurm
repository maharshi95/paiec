#!/usr/bin/env bash
#SBATCH --job-name=paiec-cpu
#SBATCH --account=clip
#SBATCH --partition=clip
#SBATCH --qos=medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=/fs/clip-quiz/mgor/aims-neurips-2026/paiec/logs/%x_%j.out
#SBATCH --error=/fs/clip-quiz/mgor/aims-neurips-2026/paiec/logs/%x_%j.err
# CPU-only harness job. Usage: hpc-submit launch/run_cpu.slurm --job-name=sel_pool --export=ALL,CONFIG=configs/sel/sel_pool.yaml
set -euo pipefail
source /fs/clip-quiz/mgor/miniconda3/etc/profile.d/conda.sh
conda activate aims26
cd /fs/clip-quiz/mgor/aims-neurips-2026/paiec
export PYTHONPATH=src OMP_NUM_THREADS=8
echo "job $SLURM_JOB_ID on $(hostname) config ${CONFIG}"
python -u scripts/run.py --config "${CONFIG}"
