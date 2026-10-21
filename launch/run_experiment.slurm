#!/bin/bash
#SBATCH --job-name=paiec
#SBATCH --output=/fs/clip-quiz/mgor/aims-neurips-2026/paiec/logs/%x-%j.out
#SBATCH --error=/fs/clip-quiz/mgor/aims-neurips-2026/paiec/logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
# Usage: hpc-submit launch/run_experiment.slurm   |   ws-submit <host> launch/run_experiment.slurm
# Set CONFIG=configs/<name>.yaml in the environment or edit below. Account/partition: see `hpc-info`.
set -euo pipefail
cd /fs/clip-quiz/mgor/aims-neurips-2026/paiec
mkdir -p logs
export PYTHONPATH=src
python scripts/train.py --config "${CONFIG:-configs/empirical_mean.yaml}"
