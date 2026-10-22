#!/usr/bin/env bash
# usage: wsub.sh <host> <variant> [extra --set args...]  -> runs on a lab workstation (1 seed unless SEEDS set)
h=$1; v=$2; shift 2
SEEDS=${SEEDS:-"[0]"}
R=/fs/clip-quiz/mgor/aims-neurips-2026/paiec-wt/item
CFG=${CONFIG:-configs/research/champion.yaml}
A="--set predictor.name=irt_item --set 'cv.seeds=$SEEDS' $* --name item__$v"
f=$R/analysis/item_online/.ws_$v.slurm
{ sed -n '1,/^set -euo/p' $R/analysis/item_online/ws.slurm | sed "s/^#SBATCH --job-name=.*/#SBATCH --job-name=item__$v/"
  printf 'export REPO=%q CONFIG=%q ARGS=%q OMP=%s\n' "$R" "$CFG" "$A" "${OMP:-3}"
  sed -n '/^set -euo/,$p' $R/analysis/item_online/ws.slurm | tail -n +2; } > $f
ws-submit $h $f | tail -1
