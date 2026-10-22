#!/usr/bin/env bash
# usage: sub.sh <variant> [extra --set args...]   (1 seed unless SEEDS set; uses champion.yaml + irt_item)
v=$1; shift
SEEDS=${SEEDS:-"[0]"}
ARGS="--set predictor.name=irt_item --set 'cv.seeds=$SEEDS' $* --name item__$v"
hpc-submit launch/research.slurm --job-name=item__$v --export=ALL,REPO=$PWD,CONFIG=configs/research/champion.yaml,ARGS="$ARGS" | tail -1
