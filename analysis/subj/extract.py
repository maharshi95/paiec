"""Extract per-pair Rasch abilities for the whole dev universe + champion subject priors per fold.

Rasch is separable per benchmark (pair and item params are benchmark-local), so one joint fit on all
data gives every pair's theta. Output: analysis/subj/pairs.pkl
    PYTHONPATH=src python analysis/subj/extract.py
"""
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, "src")
from paiec.config import load_config
from paiec.data.dataset import load_dataset
from paiec.features.subject import subject_identity
from paiec.fitcache import fit_cached
from paiec.predictors import build_predictor
from paiec.predictors.irt import rasch
from paiec.sim.replay import build_pairs

cfg = load_config("configs/research/champion.yaml")
d = cfg["data"]
t0 = time.time()
ds = load_dataset(d["root"], d["benchmarks"], d["binarize"], d["cache_dir"], max_responses=d.get("max_responses"),
                  max_item_chars=d.get("max_item_chars"), prefer_seen_from=d.get("prefer_seen_from"))
print("loaded", time.time() - t0, flush=True)
nb, ni = len(ds.bench_names), len(ds.items)
rb = ds.resp_b
pk, P_resp = np.unique(ds.resp_s * nb + rb, return_inverse=True)
cells, inv = np.unique(P_resp * ni + ds.resp_i, return_inverse=True)
ysum, n = np.bincount(inv, ds.resp_y), np.bincount(inv).astype(float)
P, I = cells // ni, cells % ni
items_used, J = np.unique(I, return_inverse=True)
theta, dd = rasch(P, J, ysum, n, len(pk), len(items_used), 9.0, 40)
pair_b, pair_s = pk % nb, pk // nb
item_b = ds.item_bench[items_used]
for b in np.unique(item_b):
    c = dd[item_b == b].mean()
    dd[item_b == b] -= c
    theta[pair_b == b] -= c
n_items = np.bincount(P, minlength=len(pk))
pair_mean = np.bincount(P, ysum, len(pk)) / np.bincount(P, n, len(pk))
print("rasch", time.time() - t0, flush=True)

ident = [subject_identity(s) for s in ds.subjects]
cv = cfg["cv"]
ref = np.isin(ds.resp_b, [ds.bench_names.index(b) for b in cv["eval_seen_from"] if b in ds.bench_names])
seen = {ident[s] for s in np.unique(ds.resp_s[ref])}
allowed = {k for k in range(len(ds.subjects)) if ident[k] in seen}
benches = ds.benchmarks()
folds = []
for fold in cv["test_benchmarks"]:
    group = [b for b in fold if b in benches]
    train_b = [b for b in benches if b not in group]
    pred = fit_cached(build_predictor(cfg["predictor"]), ds.restrict(train_b), train_b, cfg["data"], ds.revision)
    evp = {}
    for seed in (0, 1, 2):
        for pr in build_pairs(ds, group, cv["split_seed"], seed, cv["min_items"], cv["max_subjects_per_benchmark"],
                              cv["max_eval_items_per_pair"], allowed):
            evp[(pr.subject, pr.bench)] = 1
    subs = sorted({s for s, _ in evp} | set(np.unique(pair_s[np.isin(pair_b, [ds.bench_names.index(b) for b in group])]).tolist()))
    u = {s: pred.subject_prior(ds.subjects[s]) for s in subs}
    folds.append({"group": group, "train": train_b, "eval_pairs": sorted(evp), "u": u, "var_u": pred.var_u,
                  "var_beta": pred.var_beta, "var_d": pred.var_d, "hist": pred.hist, "w_s": pred.w_s, "b_s": pred.b_s})
    print("fold", group, len(evp), "var_u", pred.var_u, time.time() - t0, flush=True)

out = {"bench_names": ds.bench_names, "subjects": ds.subjects, "subject_ids": ds.subject_ids, "ident": ident,
       "pair_s": pair_s, "pair_b": pair_b, "theta": theta, "n_items": n_items, "pair_mean": pair_mean, "folds": folds}
with open("analysis/subj/pairs.pkl", "wb") as f:
    pickle.dump(out, f)
print("done", time.time() - t0)
