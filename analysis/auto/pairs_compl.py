"""Complementarity of completed dev runs from their pairs.csv (per pair-seed Brier by budget).

    python analysis/auto/pairs_compl.py
Prints, per run: ALC (fold-macro), corr of per-pair dALC with item A3's dALC (both vs champion), and the
pair-level oracle 'pick the better of A3 and X per (pair, budget)' (an upper bound for any gate that switches
between whole models by pair and budget; a blend can beat it only through within-pair averaging).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/fs/clip-quiz/mgor/aims-neurips-2026")
W = np.array([.1, .2, .2, .2, .2, .1])
BS = ["B0", "B1", "B3", "B7", "B15", "B31"]
RUNS = {
    "champion": "paiec/runs/R_champion_dev", "A3": "paiec-wt/item/runs/item__A3_mix_eb10",
    "B3": "paiec-wt/item/runs/item__B3_d32kv30", "stack_vi1": "paiec/runs/stack__item_vi1.0",
    "stack_vi05": "paiec/runs/stack__item_vi0.5", "stack_lr": "paiec/runs/stack__item_lr",
    "hier_vi_nuu10": "paiec-wt/hier/runs/hier__vi_nuu10", "hier_vi": "paiec-wt/hier/runs/hier__vi",
    "subj_lr": "paiec-wt/subj/runs/subj__lr", "lex": "paiec/runs/ref__lex", "lex_unc": "paiec/runs/ref__lex_unc",
    "acq_random": "paiec-wt/acq/runs/acq__random", "acq_voi": "paiec-wt/acq/runs/acq__voi_l1",
    "feat_p2": "paiec-wt/feat/runs/feat__p2_all4k",
}
KEY = ["subject_id", "benchmark", "seed"]


def load(p):
    f = ROOT / p / "pairs.csv"
    return pd.read_csv(f).set_index(KEY).sort_index() if f.exists() else None


def fold_macro(df, bench_fold):
    """mean over folds of the per-fold pair-mean (per seed, then averaged), as experiment.py."""
    d = df.reset_index()
    d["fold"] = d.benchmark.map(bench_fold)
    return d.groupby(["fold", "seed"])[BS].mean().groupby("fold").mean().mean()


def main():
    import yaml
    folds = yaml.safe_load(open(ROOT / "paiec/configs/split_seen/dev_base.yaml"))["cv"]["test_benchmarks"]
    bf = {b: k for k, g in enumerate(folds) for b in g}
    runs = {k: load(v) for k, v in RUNS.items()}
    runs = {k: v for k, v in runs.items() if v is not None and len(v) == len(runs["champion"])}
    C, A = runs["champion"], runs["A3"]
    dA = (A[BS].values - C[BS].values) @ W
    print(f"{'run':>14} {'ALC':>7} {'dALC':>8} {'corr_dA3':>8} {'oracle(A3,X)':>12} {'gain_vs_A3':>10} | per-budget oracle gain")
    for k, R in runs.items():
        assert (R.index == C.index).all()
        alc = float(fold_macro(R, bf) @ W)
        dX = (R[BS].values - C[BS].values) @ W
        cor = np.corrcoef(dA, dX)[0, 1] if k not in ("champion", "A3") else np.nan
        orc = np.minimum(A[BS].values, R[BS].values)
        o = pd.DataFrame(orc, index=C.index, columns=BS)
        oalc = float(fold_macro(o, bf) @ W)
        ga = fold_macro(o, bf) - fold_macro(A, bf)
        print(f"{k:>14} {alc:7.4f} {alc - float(fold_macro(C, bf) @ W):+8.4f} {cor:8.2f} {oalc:12.4f} {oalc - float(fold_macro(A, bf) @ W):+10.4f} | "
              + " ".join(f"{x:+.4f}" for x in ga.values))


if __name__ == "__main__":
    main()
