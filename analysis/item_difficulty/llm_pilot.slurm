#!/usr/bin/env bash
#SBATCH --job-name=paiec-llm-pilot
#SBATCH --account=clip
#SBATCH --partition=clip
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/fs/clip-quiz/mgor/aims-neurips-2026/paiec/data/analysis/item_difficulty/logs/%x_%j.out
#SBATCH --error=/fs/clip-quiz/mgor/aims-neurips-2026/paiec/data/analysis/item_difficulty/logs/%x_%j.err
# usage: hpc-submit analysis/item_difficulty/llm_pilot.slurm   (model: env MODEL, default Qwen/Qwen3-8B)
set -euo pipefail
source /fs/clip-quiz/mgor/miniconda3/etc/profile.d/conda.sh
conda activate llm-infer
export HF_HOME=/fs/clip-scratch/mgor/.cache/huggingface HF_HUB_OFFLINE=1
cd /fs/clip-quiz/mgor/aims-neurips-2026/paiec/analysis/item_difficulty
echo "job $SLURM_JOB_ID on $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
time python -u llm_pilot.py run "${MODEL:-Qwen/Qwen3-8B}"
