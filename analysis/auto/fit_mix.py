"""Fit irt_item_mix gate weights on the component dumps, leave-one-fold-out, and write a run config.

    PYTHONPATH=src python analysis/auto/fit_mix.py --comps A3 A3vi A3lr --kind gate --name auto__mixfit_gate

kinds (n_own bins 0, 1, 2-3, 4-7, 8-15, 16+; bin 0 = budget 0, where only the lr member differs):
  gate     bin 0: weights on the members listed in --b0 (default A3, A3lr); bins 1+: one shared weight vector
  gate_n   bin 0 as gate; one weight vector per bin 1..5
  +icpt    adds per-bin intercepts for bins 1+ (never bin 0: the B0 mean shift is a dev subject-mix artefact)
Weights for fold f are fitted on the other folds (mix_fold_w per dev benchmark); mix_w is the all-fold fit.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402
from blend import fit_brier, lg, nbins, report, sig, weights  # noqa: E402

MEMBERS = {"A3": {"lr_on": False, "vi": False}, "A3vi": {"lr_on": False, "vi": True, "vi_frac": 1.0},
           "A3lr": {"lr_on": True, "vi": False}, "A3vi05": {"lr_on": False, "vi": True, "vi_frac": 0.5},
           "A3lrvi05": {"lr_on": True, "vi": True, "vi_frac": 0.5}}
NB = 6


def layout(comps, kind, b0):
    """List of (member index, bin set or None for intercept, bin set) columns."""
    cols = [(comps.index(c), (0,)) for c in b0]
    rest = [(1, 2, 3, 4, 5)] if kind.startswith("gate_n") is False else [(b,) for b in range(1, NB)]
    for bs in rest:
        cols += [(m, bs) for m in range(len(comps))]
    if kind.endswith("icpt"):
        cols += [(None, (b,)) for b in range(1, NB)]
    return cols


def design(df, comps, cols):
    L = [lg(df["p_" + c].values) for c in comps]
    bins = [b.astype(float) for b in nbins(df.n_own.values)]
    return np.column_stack([(L[m] if m is not None else 1.0) * sum(bins[b] for b in bs) for m, bs in cols])


def to_mix(coef, cols, nm, fixed=None):
    W, c = np.zeros((nm, NB)), np.zeros(NB)
    if fixed is not None:
        W[fixed, 0] = 1.0
    for x, (m, bs) in zip(coef, cols):
        for b in bs:
            if m is None:
                c[b] += x
            else:
                W[m, b] += x
    return W, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="analysis/auto/out/dump/f*_s*.npz")
    ap.add_argument("--comps", nargs="+", default=["A3", "A3vi", "A3lr"])
    ap.add_argument("--b0", nargs="+", default=["A3", "A3lr"])
    ap.add_argument("--kind", default="gate")
    ap.add_argument("--b0fixed", help="bin 0 uses this member with weight 1 (not fitted); overrides --b0")
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--name")
    a = ap.parse_args()
    df = C.load(a.pattern)
    folds = sorted(df.fold.unique())
    w = weights(df)
    cols = layout(a.comps, a.kind, [] if a.b0fixed else a.b0)
    A = design(df, a.comps, cols)
    off = np.zeros(len(df))
    if a.b0fixed:   # fixed bin-0 logit enters as an offset column with a pinned weight (ridge-free, 1.0)
        off = np.where(df.n_own.values == 0, lg(df["p_" + a.b0fixed].values), 0.0)
    b_init = np.array([1.0 if (m == 0) else 0.0 for m, _ in cols])
    fx = a.comps.index(a.b0fixed) if a.b0fixed else None
    p = np.empty(len(df))
    fold_w = {}
    for f in folds:
        tr, te = (df.fold != f).values, (df.fold == f).values
        b = fit_brier(A[tr], df.s.values[tr], df.k.values[tr], w[tr], a.lam, b_init, off[tr])
        p[te] = sig(A[te] @ b + off[te])
        fold_w[int(f)] = to_mix(b, cols, len(a.comps), fx)
    ball = fit_brier(A, df.s.values, df.k.values, w, a.lam, b_init, off)
    Wall, call = to_mix(ball, cols, len(a.comps), fx)
    ref = C.pair_brier(df, df["p_" + a.comps[0]].values)
    report(f"{'+'.join(a.comps)} {a.kind} LOFO", df, p, ref)
    print("all-fold W (member x bin):\n", np.round(Wall, 3), "\n c:", np.round(call, 3))
    if a.name:
        dev = yaml.safe_load(open("configs/split_seen/dev_base.yaml"))["cv"]["test_benchmarks"]
        fw = {b: np.round(fold_w[k][0], 4).tolist() for k, g in enumerate(dev) for b in g}
        fc = {b: np.round(fold_w[k][1], 4).tolist() for k, g in enumerate(dev) for b in g}
        pred = {"name": "irt_item_mix", "item_target": "absolute", "beta_var_mult": 4.0, "rp_frac": 0.3, "rp_dim": 64,
                "kv_frac": 0.2, "dup_frac": 0.2, "eb_nu": 10, "lr_wvar_mult": 2.0,
                "mix_members": [MEMBERS[c] for c in a.comps], "mix_w": np.round(Wall, 4).tolist(),
                "mix_c": np.round(call, 4).tolist(), "mix_fold_w": fw, "mix_fold_c": fc}
        cfg = {"name": a.name, "base": "configs/research/champion.yaml", "predictor": pred}
        out = Path("configs/research/auto") / (a.name.split("__", 1)[-1] + ".yaml")
        out.write_text(f"# fitted by analysis/auto/fit_mix.py --comps {' '.join(a.comps)} --kind {a.kind} "
                       f"--b0 {' '.join(a.b0)} --b0fixed {a.b0fixed} (mix_fold_w: leave-one-fold-out weights per dev benchmark)\n"
                       + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None, width=200))
        print("wrote", out)


if __name__ == "__main__":
    main()
