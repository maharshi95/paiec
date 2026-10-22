#!/bin/bash
#SBATCH --job-name=paiec
#SBATCH --output=/fs/clip-quiz/mgor/aims-neurips-2026/paiec/logs/%x-%j.out
#SBATCH --error=/fs/clip-quiz/mgor/aims-neurips-2026/paiec/logs/%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
# CPU-only harness (numpy). Usage:
#   CMD="scripts/sweep.py --config configs/irt.yaml --grid predictor.var_scale=0.5,1,2 --jobs 8" hpc-submit launch/run_experiment.slurm
#   ws-submit lapine launch/run_experiment.slurm      # (CPU workstation; CMD via env)
# Default CMD runs the IRT config. Account/partition: see `hpc-info`.
set -euo pipefail
cd /fs/clip-quiz/mgor/aims-neurips-2026/paiec
export PYTHONPATH=src
python ${CMD:-scripts/run.py --config configs/irt.yaml}
