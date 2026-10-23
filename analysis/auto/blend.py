"""Offline complementarity on the shared A3 trajectory (analysis/auto/dump.py output).

    PYTHONPATH=src python analysis/auto/blend.py [--pattern ...]

1. each component's ALC on the A3 trajectory, paired vs A3;
2. logit blends A3 + w (X - A3) for each X: in-sample best w overall and per budget, and leave-one-fold-out (LOFO);
3. LOFO-fitted stackers over component logits with label-count interactions (Brier objective, ALC weights).
LOFO: the weights scored on fold f are fitted on the other four folds only.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
import common as C


def lg(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sig(x):
    return 1 / (1 + np.exp(-x))


def weights(df):
    """Per-row weight so that sum w * se == fold-macro ALC (up to a constant)."""
    key = [df.fold, df.seed, df.pair, df.budget]
    nresp = df.groupby(["fold", "seed", "pair", "budget"]).k.transform("sum")
    npairs = df.groupby(["fold", "seed"]).pair.transform("nunique")
    nseeds = df.groupby("fold").seed.transform("nunique")
    del key
    return df.budget.map(C.W).values / nresp.values / npairs.values / nseeds.values


def fit_brier(A, s, k, w, lam=1e-4, b0=None):
    def f(b):
        p = sig(A @ b)
        r = k * p - s   # d/dp (k p^2 - 2 p s) / 2
        g = A.T @ (w * 2 * r * p * (1 - p))
        return float(w @ (k * p * p - 2 * p * s)) + lam * b @ b, g + 2 * lam * b
    b0 = np.zeros(A.shape[1]) if b0 is None else b0
    return minimize(f, b0, jac=True, method="L-BFGS-B", options={"maxiter": 500}).x


def nbins(n):
    return [(n == 0), (n == 1), (n >= 2) & (n <= 3), (n >= 4) & (n <= 7), (n >= 8) & (n <= 15), (n >= 16)]


def design(df, comps, kind):
    L = {c: lg(df["p_" + c].values) for c in comps}
    base = L[comps[0]]
    one = np.ones(len(df))
    if kind == "blend":       # logit-linear blend, one weight per component (+ intercept-free)
        return np.column_stack([L[c] for c in comps])
    bins = [b.astype(float) for b in nbins(df.n_own.values)]
    if kind == "blend_n":     # per own-label-count bin weights
        return np.column_stack([L[c] * b for c in comps for b in bins])
    if kind == "blend_n_icpt":
        return np.column_stack([L[c] * b for c in comps for b in bins] + bins)
    if kind == "full":
        lb, li = np.log1p(df.n_bench.values), np.log1p(df.n_item.values)
        X = [L[c] * b for c in comps for b in bins] + bins
        X += [base * lb, base * li, lb, li, one]
        return np.column_stack(X)
    raise ValueError(kind)


def lofo(df, comps, kind, w, lam=1e-4):
    A = design(df, comps, kind)
    p = np.empty(len(df))
    coefs = {}
    b0 = None
    if kind.startswith("blend") and kind != "blend_n_icpt":
        b0 = np.zeros(A.shape[1])
        nb = A.shape[1] // len(comps)
        b0[:nb] = 1.0    # start at the base component
    for f in sorted(df.fold.unique()):
        tr, te = (df.fold != f).values, (df.fold == f).values
        b = fit_brier(A[tr], df.s.values[tr], df.k.values[tr], w[tr], lam, b0)
        coefs[f] = b
        p[te] = sig(A[te] @ b)
    ball = fit_brier(A, df.s.values, df.k.values, w, lam, b0)
    return p, coefs, ball


def report(name, df, p, ref):
    pb = C.pair_brier(df, p)
    a, m = C.fold_macro(pb)
    d, ci, dfm, win = C.paired(ref, pb)
    _, mr = C.fold_macro(ref)
    print(f"{name:>34} ALC {a:.4f}  Δ {d:+.4f}±{ci:.4f} fm {dfm:+.4f} win {win:4.1f} | "
          + " ".join(f"{m[b] - mr[b]:+.4f}" for b in C.BUD), flush=True)
    return pb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="analysis/auto/out/dump/f*_s*.npz")
    a = ap.parse_args()
    df = C.load(a.pattern)
    comps = [c[2:] for c in df.columns if c.startswith("p_")]
    print("folds", sorted(df.fold.unique()), "seeds", sorted(df.seed.unique()), "rows", len(df), "comps", comps)
    ref = C.pair_brier(df, df.p_A3.values)
    print("== 1. components on the A3 trajectory (Δ vs A3; per-budget Δ B0..B31)")
    for c in comps:
        report(c, df, df["p_" + c].values, ref)
    w = weights(df)
    print("== 2. two-way logit blends A3 + w (X - A3): in-sample best global w; LOFO global; LOFO per-n_own-bin")
    LA = lg(df.p_A3.values)
    for c in comps[1:]:
        LX = lg(df["p_" + c].values)
        best = min(np.linspace(-0.5, 1, 31), key=lambda t: float(w @ (df.k * sig(LA + t * (LX - LA)) ** 2 - 2 * sig(LA + t * (LX - LA)) * df.s)))
        report(f"A3+{c} w={best:.2f} (in-sample)", df, sig(LA + best * (LX - LA)), ref)
        for kind in ("blend", "blend_n"):
            p, co, ball = lofo(df, ["A3", c], kind, w)
            report(f"A3+{c} {kind} LOFO", df, p, ref)
            print("      all-fold coefs", np.round(ball, 2).tolist())
    print("== 3. multi-component stackers (LOFO)")
    for cs in (["A3", "A3vi", "A3lr"], ["A3", "A3vi", "A3lr", "lex"], comps):
        for kind in ("blend", "blend_n", "blend_n_icpt", "full"):
            cs_ = [c for c in cs if c in comps]
            p, co, ball = lofo(df, cs_, kind, w)
            report(f"{'+'.join(cs_)} {kind}", df, p, ref)
            if kind in ("blend", "blend_n"):
                print("      all-fold coefs", np.round(ball, 2).tolist())
    print("== 4. calibration of A3 alone (LOFO)")
    for kind in ("blend", "blend_n", "blend_n_icpt", "full"):
        p, co, ball = lofo(df, ["A3"], kind, w)
        report(f"A3 recal {kind}", df, p, ref)
        print("      all-fold coefs", np.round(ball, 2).tolist())


if __name__ == "__main__":
    main()
