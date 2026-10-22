#!/usr/bin/env bash
#SBATCH --job-name=item__time
#SBATCH --output=/fs/clip-quiz/mgor/aims-neurips-2026/paiec/logs/research/%x_%j.out
set -euo pipefail
source /fs/clip-quiz/mgor/miniconda3/etc/profile.d/conda.sh
conda activate aims26
cd /fs/clip-quiz/mgor/aims-neurips-2026/paiec-wt/item
export PYTHONPATH=src OMP_NUM_THREADS=2
for c in configs/research/champion.yaml configs/research/item.yaml; do echo "== $c"; python -u scripts/time_predict.py --config $c --benchmark mmlupro --subjects 20; done
