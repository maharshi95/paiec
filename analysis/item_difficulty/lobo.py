"""Step 3: leave-one-benchmark-out (LOBO) difficulty prediction experiments.

For each held-out benchmark b (5 harness test folds + swe_rebench as a 6th, reported separately):
  fit on the trusted items of the other benchmarks, predict d for b's trusted items, then report
  spearman, pearson, rmse_c (RMSE after centring the predictions within b; the quantity that sets the
  IRT prior variance), rmse0 (= sd of d in b: the "predict 0" baseline) and skill = 1 - rmse_c^2/rmse0^2.
Hyper-parameters (ridge alpha, k for kNN) are chosen by an inner LOBO over the training benchmarks only.
Secondary: within-benchmark 5-fold CV (GroupKFold on content, so duplicated problems stay together).

usage: python lobo.py [--quick] [--only NAME[,NAME]]   -> data/analysis/item_difficulty/lobo_results.csv
"""
import argparse
import hashlib
import sys
import time

import numpy as np
import pandas as pd
from common import OUT, TEST_BENCHES, targets
from features import build as build_feats
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from views import focus_views, headtail

sys.path.insert(0, str(OUT.parents[2] / "src"))
from paiec.features.item import ItemFeaturizer

ALPHAS = (0.1, 1, 10, 100, 1000, 10000)
FOLDS = TEST_BENCHES + ["swe_rebench"]


# ---------------------------------------------------------------- data
def beta_by_bench():
    """beta_b = mean pair ability on b (as in IRT.fit), from the cached Rasch fit."""
    return pd.read_pickle(OUT / "pairs.pkl").groupby("bench").theta.mean().to_dict()


def load_all():
    df = targets().reset_index(drop=True)
    feats = build_feats().reset_index(drop=True)
    trusted = ((df.n_subj >= 5) | (df.bench == "swe_rebench")).values
    return df, feats, trusted


def zstd(X, groups=None, train=None):
    """Standardise columns; with groups, per group (transductive within-benchmark standardisation)."""
    X = np.asarray(X, float).copy()
    if groups is None:
        ref = X[train] if train is not None else X
        mu, sd = ref.mean(0), ref.std(0) + 1e-6
        return (X - mu) / sd
    for g in np.unique(groups):
        m = groups == g
        X[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-6)
    return X


# ---------------------------------------------------------------- models
class RidgeModel:
    def __init__(self, alpha=10.0, std=True):
        self.alpha, self.std = alpha, std

    def fit(self, X, y, w=None):
        self.sp = sparse.issparse(X)
        if self.std and not self.sp:
            self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
            X = (X - self.mu) / self.sd
        self.m = Ridge(alpha=self.alpha).fit(X, y, sample_weight=w)
        return self

    def predict(self, X):
        if self.std and not self.sp:
            X = (X - self.mu) / self.sd
        return self.m.predict(X)


class KNNModel:
    """Mean target of the k most cosine-similar training items (rows assumed L2-normalised)."""

    def __init__(self, k=20):
        self.k = int(k)

    def fit(self, X, y, w=None):
        self.X, self.y = np.asarray(X, np.float32), np.asarray(y, float)
        return self

    def predict(self, X):
        S = np.asarray(X, np.float32) @ self.X.T
        k = min(self.k, S.shape[1])
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        return self.y[idx].mean(1)


class GBDTModel:
    def __init__(self, max_iter=200):
        self.max_iter = int(max_iter)

    def fit(self, X, y, w=None):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.m = HistGradientBoostingRegressor(max_iter=self.max_iter, learning_rate=0.05, max_leaf_nodes=15,
                                               min_samples_leaf=40, l2_regularization=1.0).fit(X, y, sample_weight=w)
        return self

    def predict(self, X):
        return self.m.predict(X)


def make(kind, hp):
    return {"ridge": lambda: RidgeModel(hp), "ridge_raw": lambda: RidgeModel(hp, std=False), "knn": lambda: KNNModel(hp), "gbdt": lambda: GBDTModel(hp)}[kind]()


GRID = {"ridge_raw": (10.0,), "ridge": ALPHAS, "knn": (5, 20, 50, 200), "gbdt": (100, 300)}


# ---------------------------------------------------------------- evaluation
def metrics(y, p):
    p = p - p.mean()
    rmse0 = float(np.sqrt(np.mean(y ** 2)))
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    return {"spearman": spearmanr(p, y)[0] if np.std(p) > 0 else 0.0,
            "pearson": pearsonr(p, y)[0] if np.std(p) > 0 else 0.0,
            "rmse_c": rmse, "rmse0": rmse0, "skill": 1 - rmse ** 2 / rmse0 ** 2 if rmse0 > 0 else np.nan}


def _rows(X, m):
    return X[m] if not isinstance(X, list) else [X[i] for i in np.nonzero(m)[0]]


def weights_for(bench, mode):
    if mode == "equal":  # every training benchmark gets the same total weight
        _, inv, cnt = np.unique(bench, return_inverse=True, return_counts=True)
        return 1.0 / cnt[inv] * len(bench) / len(cnt)
    return None


def fit_predict(X, y, bench, tr, te, kind, hp, wmode, y_scale=None):
    w = weights_for(bench[tr], wmode)
    yt = y[tr] if y_scale is None else y[tr] / y_scale[tr]
    m = make(kind, hp).fit(_rows(X, tr), yt, w)
    p = m.predict(_rows(X, te))
    if y_scale is not None:
        p = p * np.mean(np.unique(y_scale[tr]))
    return p


def inner_select(X, y, bench, tr, kind, wmode, y_scale):
    """Pick the hyper-parameter and a shrink factor c in [0, 1] for the centred prediction by inner LOBO.

    c is the pooled least-squares slope of y on the centred inner-fold predictions: the prior mean is pulled
    toward the benchmark mean as much as the cross-benchmark evidence says it should be."""
    grid = GRID[kind]
    inner = [b for b in np.unique(bench[tr])]
    best = (-np.inf, grid[0], 1.0)
    for hp in grid:
        ys, ps = [], []
        for b in inner:
            itr, ite = tr & (bench != b), tr & (bench == b)
            p = fit_predict(X, y, bench, itr, ite, kind, hp, wmode, y_scale)
            ys.append(y[ite] - y[ite].mean())
            ps.append(p - p.mean())
        c = float(np.clip(sum(a @ b for a, b in zip(ys, ps)) / max(sum(b @ b for b in ps), 1e-12), 0, 1))
        sk = np.mean([1 - np.mean((a - c * b) ** 2) / max(np.mean(a ** 2), 1e-12) for a, b in zip(ys, ps)])
        if sk > best[0]:
            best = (sk, hp, c)
    return best[1], best[2]


def run_lobo(name, X, df, trusted, kind="ridge", target="d", wmode="equal", zscore=False, folds=FOLDS, eval_target="d",
             absolute=False):
    """absolute=True: train on d - beta_b (= -logit of the average pair's success, i.e. the offset is in the
    target, like IRT item_target=absolute) and also report offset_err = mean prediction - true mean on b."""
    bench = df.bench.values
    y = df[target].values.astype(float)
    if target != "d":  # centre auxiliary targets per benchmark too
        y = y - df.groupby("bench")[target].transform("mean").values
    if absolute:
        y = y - df.bench.map(beta_by_bench()).values
    y_eval = df[eval_target].values
    y_scale = df.groupby("bench")[target].transform("std").values if zscore else None
    out = []
    for b in folds:
        tr, te = trusted & (bench != b), trusted & (bench == b)
        tr_inner = tr & (bench != "swe_rebench") if b == "swe_rebench" else tr
        hp, c = inner_select(X, y, bench, tr_inner, kind, wmode, y_scale)
        p = fit_predict(X, y, bench, tr, te, kind, hp, wmode, y_scale)
        r = metrics(y_eval[te], p)
        r["skill_shrunk"] = metrics(y_eval[te], c * (p - p.mean()))["skill"]
        r["shrink"] = c
        bb = beta_by_bench()
        if absolute:
            r["offset_err"] = float(p.mean() - y[te].mean())
        else:  # implied offset error when the centred prior's mean shift on b acts as a benchmark offset
            r["offset_err"] = float(p.mean() - np.mean([bb[c] for c in bb if c != b]) + bb[b])
        r.update({"variant": name, "fold": b, "hp": hp, "n_test": int(te.sum())})
        out.append(r)
    return out


def run_within(name, X, df, trusted, kind="ridge", hp=None, folds=FOLDS):
    """Secondary diagnostic: 5-fold CV within each benchmark (groups = content hash)."""
    out = []
    for b in folds:
        m = trusted & (df.bench.values == b)
        idx = np.nonzero(m)[0]
        y = df.d.values[idx]
        groups = [hashlib.md5(c.encode()).hexdigest() for c in df.content.values[idx]]
        p = np.zeros(len(idx))
        Xb = _rows(X, m)
        for tr, te in GroupKFold(5).split(idx, y, groups):
            h = hp if hp is not None else 100.0
            mod = make(kind, h).fit(Xb[tr] if not isinstance(Xb, list) else [Xb[i] for i in tr], y[tr])
            p[te] = mod.predict(Xb[te] if not isinstance(Xb, list) else [Xb[i] for i in te])
        r = {"variant": name, "fold": b, "within_spearman": spearmanr(p, y)[0], "within_skill": metrics(y, p)["skill"]}
        out.append(r)
    return out


# ---------------------------------------------------------------- feature sets
def feature_sets(df, feats, quick=False):
    items = [{"item_content": c, "item_features": k} for c, k in zip(df.content, df.item_features)]
    groups = df.bench.values
    vis_cols = [c for c in feats.columns if c not in ("idx", "bench") and not c.startswith("h_")]
    H = feats[vis_cols].fillna(0).values.astype(float)
    hid_cols = [c for c in feats.columns if c.startswith("h_")]
    Hh = feats[hid_cols].fillna(0).values.astype(float)
    S = {}

    def lazy(name, fn):
        S[name] = fn

    # the current IRT item features (hashed BOW unigrams of first 4000 chars + 4 length stats + kv)
    lazy("bow_irt", lambda: ItemFeaturizer(("text", "length", "kv"), 1024, 4000).transform(items).astype(float))
    lazy("bow_text_only", lambda: ItemFeaturizer(("text",), 1024, 4000).transform(items)[:, :1024].astype(float))
    lazy("length4", lambda: ItemFeaturizer(("length",), 1024, 4000).transform(items)[:, 1024:].astype(float))
    lazy("hand", lambda: H)
    lazy("hand_bz", lambda: zstd(H, groups))  # transductive per-benchmark standardisation
    lazy("hand_bz+hidden", lambda: np.hstack([zstd(H, groups), zstd(Hh, groups)]))
    lazy("hidden_only_bz", lambda: zstd(Hh, groups))

    def tfidf(analyzer, ngram, view, max_features=20000, min_df=3):
        def f():
            texts = [view(c) for c in df.content]
            return TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, max_features=max_features, min_df=min_df,
                                   sublinear_tf=True, lowercase=True).fit_transform(texts).tocsr()
        return f

    lazy("tfidf_w12_head4k", tfidf("word", (1, 2), lambda c: c[:4000]))
    lazy("tfidf_c35_head4k", tfidf("char_wb", (3, 5), lambda c: c[:4000]))
    fv = None

    def focus_tfidf():
        nonlocal fv
        if fv is None:
            fv = focus_views(df.content.tolist(), list(groups), 4000)
        return TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=3, sublinear_tf=True).fit_transform(fv).tocsr()
    lazy("tfidf_w12_focus4k", focus_tfidf)
    lazy("tfidf_w12_headtail4k", tfidf("word", (1, 2), lambda c: headtail(c, 4000)))

    for f in sorted((OUT / "llm").glob("feat_*.npy")):
        lazy("llm_" + f.stem[5:], (lambda f=f: np.load(f).astype(float)))
        lazy("llm_" + f.stem[5:] + "+hand_bz", (lambda f=f: np.hstack([np.load(f).astype(float), zstd(H, groups)])))
        lazy("llmdiff_" + f.stem[5:], (lambda f=f: np.load(f).astype(float)[:, :1]))
    for f in sorted((OUT / "emb").glob("*.npy")):
        key = f.stem
        lazy("emb_" + key, (lambda f=f: np.load(f).astype(float)))
        lazy("emb_" + key + "+hand_bz", (lambda f=f: np.hstack([np.load(f).astype(float), zstd(H, groups)])))
    return S


EXPERIMENTS = [
    # (name, feature set, kind, extra kwargs)
    ("zero", None, None, {}),
    ("bow_irt|ridge(lam=10, unweighted, as in IRT)", "bow_irt", "ridge_fixed10", {}),
    ("bow_irt|ridge", "bow_irt", "ridge", {}),
    ("bow_text_only|ridge", "bow_text_only", "ridge", {}),
    ("length4|ridge", "length4", "ridge", {}),
    ("hand|ridge", "hand", "ridge", {}),
    ("hand|ridge|unweighted", "hand", "ridge", {"wmode": None}),
    ("hand_bz|ridge", "hand_bz", "ridge", {}),
    ("hand_bz|ridge|zscore-target", "hand_bz", "ridge", {"zscore": True}),
    ("hand_bz|gbdt", "hand_bz", "gbdt", {}),
    ("hand|gbdt", "hand", "gbdt", {}),
    ("hand_bz|ridge|target=d_strong", "hand_bz", "ridge", {"target": "d_strong"}),
    ("hand_bz|ridge|target=d_weak", "hand_bz", "ridge", {"target": "d_weak"}),
    ("hand_bz+hidden|ridge (oracle: hidden fields)", "hand_bz+hidden", "ridge", {}),
    ("hidden_only_bz|ridge (oracle)", "hidden_only_bz", "ridge", {}),
    ("bow_irt|ridge|absolute", "bow_irt", "ridge", {"absolute": True}),
    ("hand|ridge|absolute", "hand", "ridge", {"absolute": True}),
    ("hand|gbdt|absolute", "hand", "gbdt", {"absolute": True}),
    ("tfidf_w12_head4k|ridge", "tfidf_w12_head4k", "ridge", {}),
    ("tfidf_c35_head4k|ridge", "tfidf_c35_head4k", "ridge", {}),
    ("tfidf_w12_headtail4k|ridge", "tfidf_w12_headtail4k", "ridge", {}),
    ("tfidf_w12_focus4k|ridge", "tfidf_w12_focus4k", "ridge", {}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--within", action="store_true")
    ap.add_argument("--tag", default="lobo_results")
    a = ap.parse_args()
    df, feats, trusted = load_all()
    S = feature_sets(df, feats)
    exps = list(EXPERIMENTS)
    for k in S:
        if k.startswith("llm"):
            exps.append((k + "|ridge", k, "ridge", {}))
        if k.startswith("emb_"):
            exps.append((k + "|ridge", k, "ridge", {}))
            if "+hand" not in k:
                exps.append((k + "|knn", k, "knn", {}))
    if a.only:
        keys = a.only.split(",")
        exps = [e for e in exps if any(k in e[0] for k in keys)]
    res, within = [], []
    cache = {}
    for name, fs, kind, kw in exps:
        t = time.time()
        if fs is None:
            for b in FOLDS:
                m = trusted & (df.bench.values == b)
                r = metrics(df.d.values[m], np.zeros(m.sum()))
                bb = beta_by_bench()
                r["offset_err"] = float(np.mean([-bb[c] for c in bb if c != b]) + bb[b])
                r.update({"variant": name, "fold": b, "hp": None, "n_test": int(m.sum())})
                res.append(r)
            continue
        if fs not in cache:
            cache = {fs: S[fs]()}  # keep only one matrix in memory
        X = cache[fs]
        if kind == "ridge_fixed10":  # exactly IRT._fit_prior: lam=10, no weighting, no standardisation
            out = run_lobo(name, X, df, trusted, kind="ridge_raw", wmode=None, **kw)
        else:
            out = run_lobo(name, X, df, trusted, kind=kind, **kw)
        res += out
        if a.within and kind in ("ridge", "knn") and "target" not in kw:
            within += run_within(name, X, df, trusted, kind=kind, hp=out[0]["hp"])
        o = pd.DataFrame(out).set_index("fold")
        print(f"{name:55s} {time.time() - t:6.1f}s  sp " + " ".join(f"{o.spearman[b]:+.2f}" for b in FOLDS)
              + f" | mean5 sp {o.spearman[TEST_BENCHES].mean():+.3f} skill {o.skill[TEST_BENCHES].mean():+.3f}", flush=True)
    R = pd.DataFrame(res)
    R.to_csv(OUT / f"{a.tag}.csv", index=False)
    if within:
        pd.DataFrame(within).to_csv(OUT / f"{a.tag}_within.csv", index=False)
    summarize(R, pd.DataFrame(within) if within else None)


def summarize(R, W=None):
    t5 = R[R.fold.isin(TEST_BENCHES)]
    g = t5.groupby("variant", sort=False)
    S = pd.DataFrame({"sp_mean5": g.spearman.mean(), "pe_mean5": g.pearson.mean(), "rmse_c_mean5": g.rmse_c.mean(),
                      "skill_mean5": g.skill.mean(), "skill_shrunk5": g.skill_shrunk.mean() if "skill_shrunk" in t5 else np.nan,
                      "shrink_mean5": g.shrink.mean() if "shrink" in t5 else np.nan})
    for b in FOLDS:
        S["sp_" + b[:8]] = R[R.fold == b].set_index("variant").spearman
    S["skill_swe_reb"] = R[R.fold == "swe_rebench"].set_index("variant").skill
    if "offset_err" in t5:
        S["offset_rmse5"] = np.sqrt(t5.groupby("variant", sort=False).offset_err.apply(lambda v: np.mean(v ** 2)))
    if W is not None and len(W):
        S["within_sp_mean5"] = W[W.fold.isin(TEST_BENCHES)].groupby("variant").within_spearman.mean()
    pd.set_option("display.width", 250)
    print(S.round(3).to_string())
    return S


if __name__ == "__main__":
    main()
