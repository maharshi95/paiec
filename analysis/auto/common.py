"""Load component dumps (analysis/auto/dump.py) and compute pair Brier / ALC exactly as the harness."""
import glob

import numpy as np
import pandas as pd

BUD = (0, 1, 3, 7, 15, 31)
W = dict(zip(BUD, (.1, .2, .2, .2, .2, .1)))


def load(pattern="analysis/auto/out/dump/f*_s*.npz"):
    parts = []
    for f in sorted(glob.glob(pattern)):
        z = np.load(f, allow_pickle=True)
        fold, seed = int(z["fold"]), int(z["seed"])
        d = {"fold": fold, "seed": seed, "pair": z["pair"], "budget": z["budget"], "n_own": z["n_own"],
             "n_bench": z["n_bench"], "n_item": z["n_item"], "s": z["s"], "k": z["k"]}
        df = pd.DataFrame(d)
        df["bench"] = z["pair_bench"][z["pair"]]
        df["subject"] = z["pair_subject"][z["pair"]]
        for c, p in zip(z["comps"], z["P"]):
            df["p_" + str(c)] = p
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def pair_brier(df, p):
    """Per (fold, seed, pair, budget) Brier of prob vector p (aligned with df)."""
    se = df.k * p ** 2 - 2 * p * df.s + df.s
    g = pd.DataFrame({"fold": df.fold, "seed": df.seed, "pair": df.pair, "budget": df.budget, "se": se, "k": df.k})
    g = g.groupby(["fold", "seed", "pair", "budget"], sort=True)[["se", "k"]].sum()
    return (g.se / g.k).unstack("budget")


def alc_pairs(pb):
    """Per pair ALC from a pair_brier table."""
    return sum(W[b] * pb[b] for b in BUD)


def fold_macro(pb):
    """Fold-macro ALC and per-budget (mean over folds of mean over seeds of pair mean)."""
    m = pb.groupby(["fold", "seed"]).mean().groupby("fold").mean().mean()
    return float(sum(W[b] * m[b] for b in BUD)), m


def paired(pa, pb):
    """Paired Δ (b - a) over pairs: mean, 95% CI half-width, fold-macro Δ, win%."""
    d = alc_pairs(pb) - alc_pairs(pa)
    fa, _ = fold_macro(pa)
    fb, _ = fold_macro(pb)
    return float(d.mean()), float(1.96 * d.std() / np.sqrt(len(d))), fb - fa, float((d < 0).mean() * 100)
