"""Test-time (transductive) item-difficulty sharing within one benchmark.

At test time, global evidence reveals labels on a subset of the unseen benchmark's items. How much does that
tell us about the *other* items' difficulty, via (a) shared item_features values (competition / website / paper /
lang), (b) nearest neighbours in embedding space, (c) a within-benchmark ridge on embeddings?
Simulation: a random fraction f of the benchmark's items have an observed difficulty d_obs = d + noise
(noise sd sigma mimics a few labels per item); predict the rest; report skill = 1 - MSE/Var(d) on held-out items.
Optimistic in that it ignores the IRT coupling with abilities; meant to rank the sharing mechanisms.

usage: python transductive.py [EMB_NAME]   (default minilm__focus)
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from common import OUT, TEST_BENCHES, targets
from features import _kv

FRACS = (0.02, 0.05, 0.2)
SIGMAS = (0.0, 1.5, 3.0)
REPS = 5


def kv_group(features):
    kv = _kv(features)
    kv.pop("problem_idx", None)
    kv.pop("prompt_source", None)
    kv.pop("image_detail", None)
    return ";".join(f"{k}={v}" for k, v in sorted(kv.items()))


def main():
    emb = sys.argv[1] if len(sys.argv) > 1 else "minilm__focus"
    df = targets().reset_index(drop=True)
    E = np.load(OUT / "emb" / f"{emb}.npy")
    rng = np.random.default_rng(0)
    rows = []
    for b in TEST_BENCHES:
        m = ((df.n_subj >= 5) & (df.bench == b)).values
        idx = np.nonzero(m)[0]
        d = df.d.values[idx]
        groups = np.array([kv_group(f) for f in df.item_features.values[idx]])
        content = df.content.values[idx]
        X = E[idx]
        var = d.var()
        for f in FRACS:
            for sig in SIGMAS:
                acc = {k: [] for k in ("kv_group_mean", "same_content", "knn10", "ridge_emb", "kv+knn")}
                for _ in range(REPS):
                    obs = rng.random(len(idx)) < f
                    if obs.sum() < 3:
                        continue
                    y = d[obs] + rng.normal(0, sig, obs.sum())
                    te = ~obs
                    mu = y.mean()
                    # (a) kv group means, shrunk toward the benchmark mean (k = 3 pseudo-items)
                    gs = pd.Series(y).groupby(groups[obs]).agg(["sum", "count"])
                    gm = {g: (r["sum"] + 3 * mu) / (r["count"] + 3) for g, r in gs.iterrows()}
                    p_kv = np.array([gm.get(g, mu) for g in groups[te]])
                    # same exact content (duplicates), unless it is a template shared by many items
                    cs = pd.Series(y).groupby(content[obs]).agg(["mean", "count"])
                    ccount = pd.Series(content).value_counts()
                    p_c = np.array([cs["mean"][c] if (c in cs.index and ccount[c] <= 5) else pk
                                    for c, pk in zip(content[te], p_kv)])
                    # (b) kNN in embedding space (k=10, shrunk)
                    S = X[te] @ X[obs].T
                    k = min(10, obs.sum())
                    nn = np.argpartition(-S, k - 1, axis=1)[:, :k]
                    p_knn = (y[nn].sum(1) + 3 * mu) / (k + 3)
                    # (c) ridge on embeddings
                    p_r = Ridge(alpha=10.0).fit(X[obs], y).predict(X[te])
                    for name, p in (("kv_group_mean", p_kv), ("same_content", p_c), ("knn10", p_knn),
                                    ("ridge_emb", p_r), ("kv+knn", 0.5 * (p_kv + p_knn))):
                        acc[name].append(1 - np.mean((d[te] - p) ** 2) / var)
                for name, v in acc.items():
                    rows.append({"bench": b, "frac": f, "noise_sd": sig, "method": name, "skill": np.mean(v) if v else np.nan})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "transductive.csv", index=False)
    pd.set_option("display.width", 220)
    print(R.pivot_table(index=["frac", "noise_sd", "method"], columns="bench", values="skill").round(2).to_string())


if __name__ == "__main__":
    main()
