"""Where each architectural decision helps or hurts, on one shared trajectory (component dump).

    PYTHONPATH=src python analysis/auto/effects.py [--pattern analysis/auto/out/dump/f*_s*.npz]

Contrasts (Brier difference, response-weighted within pair, pair-macro, negative = the change helps):
  item  = A3 - champ        (online item amortisation)
  vi_i  = A3vi - A3         (mean-field VI on top of item amortisation)
  vi_c  = hvi - champ       (VI + subject-variance EM on the champion)
  lr_i  = A3lr - A3         (subject-descriptor regression on top of item amortisation)
  lr_c  = lr - champ
  lex   = lex - A3          (pair-local lexical kernel instead of the model)
Breakdowns: budget; benchmark base rate (extreme < .2 / > .8 vs mid); evaluated pairs per benchmark;
item-level: eval items that other pairs have labelled (n_item > 0) vs not, at budgets >= 7.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

CONTRASTS = {"item": ("A3", "champ"), "vi_i": ("A3vi", "A3"), "vi_c": ("hvi", "champ"), "lr_i": ("A3lr", "A3"),
             "lr_c": ("lr", "champ"), "lex": ("lex", "A3"), "lrvi_i": ("A3lrvi05", "A3"),
             "vi05_i": ("A3vi05", "A3"), "lr_on_vi": ("A3lrvi05", "A3vi05")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="analysis/auto/out/dump/f*_s*.npz")
    a = ap.parse_args()
    df = C.load(a.pattern)
    comps = {c[2:] for c in df.columns if c.startswith("p_")}
    cons = {k: v for k, v in CONTRASTS.items() if set(v) <= comps}
    se = {c: df.k * df["p_" + c] ** 2 - 2 * df["p_" + c] * df.s + df.s for c in comps}
    key = ["fold", "seed", "pair", "budget"]
    nresp = df.groupby(key).k.transform("sum")
    # pair attributes
    rate = df.groupby("bench").s.transform("sum") / df.groupby("bench").k.transform("sum")
    npairs = df.groupby(["bench", "seed"]).pair.transform("nunique")
    attrs = pd.DataFrame({
        "budget": df.budget,
        "base_rate": np.where((rate < .2) | (rate > .8), "extreme", "mid"),
        "pairs/bench": pd.cut(npairs, [0, 6, 15, 20], labels=["1-6", "7-15", "16-20"]).astype(str),
    })
    for name, (x, y) in cons.items():
        d = (se[x] - se[y]) / nresp
        pb = d.groupby([df.fold, df.seed, df.pair, df.budget]).sum().groupby(level="budget").mean()
        print(f"\n== {name}: {x} - {y}   (pair-macro Brier Δ; ALC-weighted {sum(C.W[b] * pb[b] for b in C.BUD):+.4f})")
        for col in attrs:
            t = pd.DataFrame({"d": d, "g": attrs[col], "fold": df.fold, "seed": df.seed, "pair": df.pair, "budget": df.budget})
            per_pair = t.groupby(["g", "fold", "seed", "pair", "budget"]).d.sum().reset_index()
            if col == "budget":
                s = per_pair.groupby("g").d.mean()
            else:
                s = per_pair.assign(w=per_pair.budget.map(C.W)).groupby(["g", "fold", "seed", "pair"]).apply(
                    lambda z: (z.d * z.w).sum(), include_groups=False).groupby("g").agg(["mean", "size"])
            print(f"  by {col}: " + ", ".join(f"{k}: {v:+.4f}" if np.isscalar(v) else f"{k}: {v['mean']:+.4f} (n={int(v['size'])})"
                                             for k, v in (s.items() if col == "budget" else s.iterrows())))
        late = df.budget >= 7
        for lab, m in (("labelled by others", late & (df.n_item > 0)), ("unlabelled", late & (df.n_item == 0))):
            dd = (se[x] - se[y])[m].sum() / df.k[m].sum()
            print(f"  item-level, budgets>=7, eval items {lab}: {dd:+.4f} per response (n items={int(m.sum())})")


if __name__ == "__main__":
    main()
