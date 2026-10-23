#!/usr/bin/env bash
# fit_mix variants on the dump2 (A3+lr+VI driver) trajectory
P='analysis/auto/out/dump2/f*_s*.npz'
for args in "--kind gate --b0fixed A3lrvi05" "--kind gate --b0 A3lrvi05" "--kind gate_n --b0fixed A3lrvi05" "--kind gate_icpt --b0fixed A3lrvi05" "--kind gate --b0fixed A3lrvi05 --comps A3lrvi05 A3vi05" "--kind gate --b0fixed A3lrvi05 --comps A3lrvi05 A3lr" "--kind gate --b0fixed A3lrvi05 --comps A3lrvi05"; do
  echo ">> $args"
  python analysis/auto/fit_mix.py --pattern "$P" --comps A3lrvi05 A3vi05 A3lr A3 $args 2>&1 | grep -v Warn
done
