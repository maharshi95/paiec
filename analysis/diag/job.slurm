#!/usr/bin/env bash
#SBATCH --job-name=diag
#SBATCH --account=clip
#SBATCH --partition=clip
#SBATCH --qos=medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=6:00:00
#SBATCH --output=/fs/clip-quiz/mgor/aims-neurips-2026/paiec/logs/research/%x_%j.out
#SBATCH --error=/fs/clip-quiz/mgor/aims-neurips-2026/paiec/logs/research/%x_%j.out
# Diag-direction analysis job: SCRIPT and ARGS env vars, run from the diag worktree.
set -euo pipefail
source /fs/clip-quiz/mgor/miniconda3/etc/profile.d/conda.sh
conda activate aims26
cd /fs/clip-quiz/mgor/aims-neurips-2026/paiec-wt/diag
export PYTHONPATH=/fs/clip-quiz/mgor/aims-neurips-2026/paiec-wt/diag/src OMP_NUM_THREADS=8
echo "job ${SLURM_JOB_ID:-local} on $(hostname) script ${SCRIPT} args ${ARGS:-}"
eval python -u analysis/diag/${SCRIPT} ${ARGS:-}
