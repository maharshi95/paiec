#!/usr/bin/env bash
#SBATCH --job-name=embed-test
#SBATCH --gres=gpu:1
#SBATCH --output=/fs/clip-quiz/mgor/aims-neurips-2026/paiec/logs/%x_%j.out
#SBATCH --error=/fs/clip-quiz/mgor/aims-neurips-2026/paiec/logs/%x_%j.err
set -euo pipefail
export PATH=/fs/clip-quiz/mgor/miniconda3/envs/advscore/bin:$PATH
export HF_HOME=/fs/clip-scratch/mgor/hf HF_HUB_OFFLINE=1 PYTHONPATH=src PAIEC_MPNET=/fs/clip-scratch/mgor/paiec-cache/test.npz
cd /fs/clip-quiz/mgor/aims-neurips-2026/paiec
python -u scripts/embed_items.py --config configs/research/dev.yaml --limit 300000
