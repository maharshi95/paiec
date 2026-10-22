#!/usr/bin/env bash
#SBATCH --job-name=hier-a
#SBATCH --account=clip
#SBATCH --partition=clip
#SBATCH --qos=medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=/fs/clip-quiz/mgor/aims-neurips-2026/paiec/logs/research/%x_%j.out
#SBATCH --error=/fs/clip-quiz/mgor/aims-neurips-2026/paiec/logs/research/%x_%j.out
# Generic analysis job: CMD is eval'd in the worktree.
set -euo pipefail
source /fs/clip-quiz/mgor/miniconda3/etc/profile.d/conda.sh
conda activate aims26
cd /fs/clip-quiz/mgor/aims-neurips-2026/paiec-wt/hier
export PYTHONPATH=src OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "job ${SLURM_JOB_ID:-local} on $(hostname) $(git rev-parse --short HEAD) cmd: ${CMD}"
eval "${CMD}"
