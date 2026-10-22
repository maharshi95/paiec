#!/usr/bin/env bash
#SBATCH --job-name=item__ctrl1
#SBATCH --account=clip
#SBATCH --partition=clip
#SBATCH --qos=medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=/fs/clip-quiz/mgor/aims-neurips-2026/paiec/logs/research/%x_%j.out
#SBATCH --error=/fs/clip-quiz/mgor/aims-neurips-2026/paiec/logs/research/%x_%j.out
# Research-loop job, runnable from any worktree.
#   hpc-submit launch/research.slurm --job-name=X --export=ALL,REPO=$PWD,CONFIG=configs/...,ARGS="--set a.b=1 --name X"
# Run several configs serially in one job: CONFIG="a.yaml b.yaml" (ARGS apply to each).
set -euo pipefail
export REPO=/fs/clip-quiz/mgor/aims-neurips-2026/paiec-wt/item CONFIG=configs/research/champion.yaml ARGS=--set\ predictor.name=irt_item\ --set\ \'cv.seeds=\[0\]\'\ \ --name\ item__ctrl1 OMP=3
source /fs/clip-quiz/mgor/miniconda3/etc/profile.d/conda.sh
conda activate aims26
cd "${REPO:-/fs/clip-quiz/mgor/aims-neurips-2026/paiec}"
export PYTHONPATH=src OMP_NUM_THREADS=${OMP:-3}
echo "job ${SLURM_JOB_ID:-local} on $(hostname) repo $(pwd) ($(git rev-parse --abbrev-ref HEAD)@$(git rev-parse --short HEAD)) config ${CONFIG} args ${ARGS:-}"
for c in ${CONFIG}; do
  eval python -u scripts/run.py --config "$c" ${ARGS:-}
done
