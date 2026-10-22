"""Shared loading + target fitting for the item-difficulty analysis.

Targets (per item, all benchmarks, matharena/mmdocrag binarised at grade >= 0.5 as in configs/base.yaml):
  d          Rasch difficulty from paiec.predictors.irt.rasch, centred per benchmark (same as IRT.fit)
  rate       empirical solve rate over all trials
  n_resp     number of responses (trials), n_subj distinct subjects
  d_strong / d_weak   logit-smoothed failure rate among the top / bottom half of subjects by theta
  a2pl       2PL discrimination (per-benchmark joint fit, analysis only)
Hidden metadata (NOT visible at test time) is joined from the raw items table for analysis:
  raw_item_id, grading_criterion, verifier.

Outputs are cached under data/analysis/item_difficulty/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from paiec.data.dataset import load_dataset  # noqa: E402
from paiec.predictors.irt import rasch  # noqa: E402
from paiec.predictors.simple import sigmoid  # noqa: E402

BENCHES = ["matharena", "mmdocrag", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]
TEST_BENCHES = ["matharena", "mmdocrag", "multi_swebench", "real_webagents", "researchcodebench"]
OUT = ROOT / "data" / "analysis" / "item_difficulty"
OUT.mkdir(parents=True, exist_ok=True)


def load():
    return load_dataset(str(ROOT / "data/raw"), BENCHES, {"mode": "threshold", "threshold": 0.5},
                        str(ROOT / "data/processed"))


def fit_targets(ds, var=9.0, iters=40):
    """Rasch exactly as IRT.fit builds it: pairs = (subject, benchmark), cells aggregated over trials."""
    nb, ni = len(ds.bench_names), len(ds.items)
    rb = ds.resp_b
    pk, P_resp = np.unique(ds.resp_s * nb + rb, return_inverse=True)
    cells, inv = np.unique(P_resp * ni + ds.resp_i, return_inverse=True)
    ysum, n = np.bincount(inv, ds.resp_y), np.bincount(inv).astype(float)
    P, I = cells // ni, cells % ni
    items_used, J = np.unique(I, return_inverse=True)
    theta, d = rasch(P, J, ysum, n, len(pk), len(items_used), var, iters)
    pair_b = pk % nb
    item_b = ds.item_bench[items_used]
    for b in np.unique(item_b):
        c = d[item_b == b].mean()
        d[item_b == b] -= c
        theta[pair_b == b] -= c

    # strong / weak split per benchmark (by pair ability)
    strong = np.zeros(len(pk), bool)
    for b in np.unique(pair_b):
        m = pair_b == b
        strong[m] = theta[m] >= np.median(theta[m])
    s_cell = strong[P]
    nJ = len(items_used)

    def smoothed_logit_fail(mask):
        ys, nn = np.bincount(J[mask], ysum[mask], nJ), np.bincount(J[mask], n[mask], nJ)
        return np.log((nn - ys + 0.5) / (ys + 0.5)), nn

    d_strong, n_strong = smoothed_logit_fail(s_cell)
    d_weak, n_weak = smoothed_logit_fail(~s_cell)

    # 2PL (analysis only): alternate Newton on theta, d, log a with N(0,var) priors, per benchmark jointly
    a = np.ones(nJ)
    th, dd = theta.copy(), d.copy()
    for _ in range(30):
        z = a[J] * (th[P] - dd[J]); mu = sigmoid(z); w = n * mu * (1 - mu); r = ysum - n * mu
        th += (np.bincount(P, a[J] * r, len(pk)) - th / var) / (np.bincount(P, a[J] ** 2 * w, len(pk)) + 1 / var)
        z = a[J] * (th[P] - dd[J]); mu = sigmoid(z); w = n * mu * (1 - mu); r = ysum - n * mu
        dd += (-np.bincount(J, a[J] * r, nJ) - dd / var) / (np.bincount(J, a[J] ** 2 * w, nJ) + 1 / var)
        z = a[J] * (th[P] - dd[J]); mu = sigmoid(z); w = n * mu * (1 - mu); r = ysum - n * mu
        la = np.log(a); x = th[P] - dd[J]
        g = np.bincount(J, r * x * a[J], nJ) - la / 0.25
        h = np.bincount(J, w * (x * a[J]) ** 2, nJ) + 1 / 0.25
        a = np.exp(np.clip(la + g / h, -2.5, 1.5))

    df = pd.DataFrame({
        "idx": items_used,
        "bench": [ds.bench_names[b] for b in item_b],
        "d": d,
        "rate": np.bincount(J, ysum, nJ) / np.bincount(J, n, nJ),
        "n_resp": np.bincount(J, n, nJ).astype(int),
        "n_subj": np.bincount(J, minlength=nJ),
        "d_strong": d_strong, "n_strong": n_strong.astype(int),
        "d_weak": d_weak, "n_weak": n_weak.astype(int),
        "a2pl": a,
        # posterior sd of d under the Rasch fit (to flag untrustworthy items)
        "d_se": 1 / np.sqrt(np.bincount(J, n * sigmoid(theta[P] - d[J]) * (1 - sigmoid(theta[P] - d[J])), nJ) + 1 / var),
    })
    df["item_id"] = [ds.item_ids[i] for i in items_used]
    df["content"] = [ds.items[i]["item_content"] for i in items_used]
    df["item_features"] = [ds.items[i]["item_features"] for i in items_used]
    pair_df = pd.DataFrame({"pair": pk, "bench": [ds.bench_names[b] for b in pair_b], "theta": theta,
                            "subject": [ds.subject_ids[s] for s in pk // nb]})
    return df, pair_df


def hidden_meta():
    rows = []
    for b in BENCHES:
        it = pd.read_parquet(ROOT / f"data/raw/{b}/formatted_tables/items.parquet",
                             columns=["item_id", "raw_item_id", "grading_criterion", "verifier", "asset_manifest"])
        rows.append(it)
    return pd.concat(rows, ignore_index=True).drop_duplicates("item_id")


def targets(refresh=False):
    """Per-item target table (cached)."""
    f = OUT / "targets.pkl"
    if f.exists() and not refresh:
        return pd.read_pickle(f)
    ds = load()
    df, pair_df = fit_targets(ds)
    df = df.merge(hidden_meta(), on="item_id", how="left")
    df.to_pickle(f)
    pair_df.to_pickle(OUT / "pairs.pkl")
    return df


if __name__ == "__main__":
    df = targets(refresh=True)
    pd.set_option("display.width", 200)
    g = df.groupby("bench")
    print(pd.DataFrame({
        "items": g.size(), "mean_rate": g.rate.mean(), "sd_d": g.d.std(),
        "med_n_resp": g.n_resp.median(), "med_n_subj": g.n_subj.median(),
        "frac_all0": g.rate.apply(lambda r: (r == 0).mean()), "frac_all1": g.rate.apply(lambda r: (r == 1).mean()),
        "frac_nsubj<5": g.n_subj.apply(lambda r: (r < 5).mean()), "med_d_se": g.d_se.median(),
        "corr_dstrong_dweak": g.apply(lambda x: x.d_strong.corr(x.d_weak, method="spearman")),
        "med_a2pl": g.a2pl.median(),
    }).round(3).to_string())
