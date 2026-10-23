#!/usr/bin/env bash
#SBATCH --job-name=auto_dump
#SBATCH --account=clip
#SBATCH --partition=clip
#SBATCH --qos=medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=/fs/clip-quiz/mgor/aims-neurips-2026/paiec/logs/research/%x_%j.out
#SBATCH --error=/fs/clip-quiz/mgor/aims-neurips-2026/paiec/logs/research/%x_%j.out
# Component dump on one replay trajectory: FOLD=<k> [ARGS="--comps ..."]
set -euo pipefail
source /fs/clip-quiz/mgor/miniconda3/etc/profile.d/conda.sh
conda activate aims26
cd /fs/clip-quiz/mgor/aims-neurips-2026/paiec-wt/auto
export PYTHONPATH=src OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
eval python -u analysis/auto/dump.py --fold "${FOLD}" ${ARGS:-}
