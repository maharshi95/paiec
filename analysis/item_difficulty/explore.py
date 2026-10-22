"""Step 2: what makes items hard, per benchmark.

Writes data/analysis/item_difficulty/{feature_corr.csv, kv_groups.csv, extremes.txt} and prints summaries.
Items with fewer than MIN_SUBJ distinct subjects are dropped (except swe_rebench, which has one subject).
"""
import re

import numpy as np
import pandas as pd
from common import OUT, targets
from features import _kv, build
from scipy.stats import spearmanr

MIN_SUBJ = 5


def main():
    df = targets()
    feats = build()
    X = feats.drop(columns=["idx", "bench"])
    df = pd.concat([df.reset_index(drop=True), X], axis=1)
    trusted = (df.n_subj >= MIN_SUBJ) | (df.bench == "swe_rebench")
    print("trusted items per bench:", df[trusted].groupby("bench").size().to_dict())
    d = df[trusted]

    # ---- per-feature Spearman with d, per benchmark ----
    rows = []
    for col in X.columns:
        r = {"feature": col}
        for b, g in d.groupby("bench"):
            v = g[col]
            if v.notna().sum() > 20 and v.nunique() > 1:
                r[b] = spearmanr(v, g.d, nan_policy="omit")[0]
        rows.append(r)
    corr = pd.DataFrame(rows).set_index("feature")
    benches = [c for c in corr.columns]
    vis = corr.loc[[c for c in corr.index if not c.startswith("h_")]]
    # transferability: same sign in >= 4 of the 5 non-trivial benchmarks with |rho| >= .1
    sign = np.sign(vis.fillna(0)).where(vis.abs() >= 0.1, 0)
    corr["n_pos"] = (np.sign(corr[benches]).where(corr[benches].abs() >= 0.1, 0) > 0).sum(1)
    corr["n_neg"] = (np.sign(corr[benches]).where(corr[benches].abs() >= 0.1, 0) < 0).sum(1)
    corr["mean_abs"] = corr[benches].abs().mean(1)
    corr.round(3).to_csv(OUT / "feature_corr.csv")
    pd.set_option("display.width", 220)
    print("\nSpearman(feature, d) per benchmark (|rho|>=.1 counted in n_pos/n_neg):")
    print(corr.sort_values("mean_abs", ascending=False).round(2).to_string())
    del sign

    # ---- selection confounds ----
    print("\nSpearman(d, n_subj) and (d, n_resp): does who-was-tested-on-what correlate with difficulty?")
    for b, g in d.groupby("bench"):
        print(f"  {b:18s} n_subj {spearmanr(g.n_subj, g.d)[0]:+.2f}  n_resp {spearmanr(g.n_resp, g.d)[0]:+.2f}"
              f"  rate~d {spearmanr(g.rate, g.d)[0]:+.2f}  a2pl~d {spearmanr(g.a2pl, g.d)[0]:+.2f}"
              f"  d_strong~d_weak {spearmanr(g.d_strong, g.d_weak)[0]:+.2f}")

    # ---- categorical kv groups: share of variance explained (eta^2) and group means ----
    out = []
    for b, g in d.groupby("bench"):
        kvs = g.item_features.map(_kv)
        keys = sorted({k for m in kvs for k in m})
        for k in keys:
            v = kvs.map(lambda m: m.get(k, ""))
            if v.nunique() < 2:
                continue
            if k == "problem_idx":
                print(f"\n{b}: Spearman(problem_idx, d) = {spearmanr(v.astype(float), g.d)[0]:+.2f}")
                # within-competition (so the comparison is not across competitions)
                comp = kvs.map(lambda m: m.get("competition", ""))
                within = []
                for c, gg in g.groupby(comp.values):
                    if len(gg) >= 10:
                        within.append(spearmanr(v[gg.index].astype(float), gg.d)[0])
                print(f"  mean within-competition Spearman(problem_idx, d) = {np.nanmean(within):+.2f} over {len(within)} competitions")
                continue
            gm = g.d.groupby(v.values).agg(["mean", "size"])
            eta2 = float(((gm["mean"] - g.d.mean()) ** 2 * gm["size"]).sum() / ((g.d - g.d.mean()) ** 2).sum())
            print(f"\n{b}: key {k!r}: {v.nunique()} values, eta^2 = {eta2:.2f}")
            print("  " + ", ".join(f"{i}={r['mean']:+.2f}(n={int(r['size'])})" for i, r in gm.sort_values("mean").iterrows()))
            for i, r in gm.iterrows():
                out.append({"bench": b, "key": k, "value": i, "mean_d": r["mean"], "n": r["size"], "eta2": eta2})
    pd.DataFrame(out).to_csv(OUT / "kv_groups.csv", index=False)

    # ---- prompt templates (first 60 chars) as a hidden "sub-benchmark" ----
    for b in ["matharena", "researchcodebench", "multi_swebench"]:
        g = d[d.bench == b]
        tpl = g.content.str[:60]
        gm = g.d.groupby(tpl.values).agg(["mean", "size"])
        gm = gm[gm["size"] >= 10]
        if len(gm) > 1:
            print(f"\n{b}: prompt-template prefixes with >=10 items")
            for i, r in gm.sort_values("mean").iterrows():
                print(f"  {r['mean']:+.2f} n={int(r['size']):4d}  {i!r}")

    # ---- extremes dump ----
    with open(OUT / "extremes.txt", "w") as f:
        for b, g in d.groupby("bench"):
            g = g.sort_values("d")
            for lab, sub in (("EASY", g.head(10)), ("HARD", g.tail(10))):
                for r in sub.itertuples():
                    c = re.sub(r"\s+", " ", r.content)
                    f.write(f"{b}\t{lab}\td={r.d:+.2f}\trate={r.rate:.2f}\tn_subj={r.n_subj}\t{r.item_features}\t{c[:400]}\n")


if __name__ == "__main__":
    main()
