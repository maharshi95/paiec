#!/usr/bin/env bash
#SBATCH --job-name=__NAME__
#SBATCH --gres=gpu:1
#SBATCH --output=/fs/clip-quiz/mgor/aims-neurips-2026/paiec/logs/research/%x_%j.out
#SBATCH --error=/fs/clip-quiz/mgor/aims-neurips-2026/paiec/logs/research/%x_%j.out
# Research run on a lab workstation GPU (1080 Ti): needs the `advscore` env (torch 2.7+cu126 ships sm_60; aims26's has no Pascal kernels).
# Generate per-job copies with sed (ws-submit does not forward env): __NAME__ and __CONFIG__.
set -euo pipefail
export PATH=/fs/clip-quiz/mgor/miniconda3/envs/advscore/bin:$PATH
export HF_HOME=/fs/clip-scratch/mgor/hf HF_HUB_OFFLINE=1 PYTHONPATH=src OMP_NUM_THREADS=8
cd /fs/clip-quiz/mgor/aims-neurips-2026/paiec
echo "job $SLURM_JOB_ID on $(hostname) config __CONFIG__"
python -u scripts/run.py --config __CONFIG__ --name __NAME__
