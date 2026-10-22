"""Diagnose budget-0 (cold start) Brier of IRT prior variants per LOBO fold.

Variants are applied post-fit by editing prior parameters, so each fold is fitted once.
    PYTHONPATH=src python analysis/b0/b0_prior.py
"""
import copy
import sys

import numpy as np

sys.path.insert(0, "src")
from paiec.config import load_config
from paiec.data.dataset import load_dataset
from paiec.predictors import build_predictor
from paiec.sim.replay import build_pairs


def pair_brier(pred, ds, pairs):
    out = []
    for pr in pairs:
        p = pred.predict_batch([(ds.subjects[pr.subject], ds.items[i]) for i in pr.evl], [])
        se = sum(float(((pk - ys) ** 2).sum()) for pk, ys in zip(p, pr.evl_ys))
        out.append(se / sum(len(y) for y in pr.evl_ys))
    return float(np.mean(out))


cfg = load_config("configs/irt.yaml")
d = cfg["data"]
ds = load_dataset(d["root"], d["benchmarks"], d["binarize"], d["cache_dir"])
benches = d["benchmarks"]
rows = {}
for test_b in cfg["cv"]["test_benchmarks"]:
    train = ds.restrict([b for b in benches if b != test_b])
    base = build_predictor(cfg["predictor"]).fit(train)
    pairs = build_pairs(ds, [test_b], max_eval_items=200)
    rate = np.mean([y.mean() for pr in pairs for y in pr.evl_ys])
    r = {"const .5": 0.25}
    # oracle per-pair constant = pair base rate (upper bound for 'right offset, no item info')
    r["oracle pair rate"] = float(np.mean([np.mean(np.concatenate(pr.evl_ys)) * (1 - np.mean(np.concatenate(pr.evl_ys))) for pr in pairs]))
    tr_mean = float(train.resp_y.mean())
    r["const train mean"] = float(np.mean([np.mean((tr_mean - np.concatenate(pr.evl_ys)) ** 2) for pr in pairs]))
    r["irt"] = pair_brier(base, ds, pairs)
    for k in (2, 4, 8, 16, 64):
        p = copy.deepcopy(base); p.var_beta *= k; p.reset_runtime()
        r[f"irt beta_var x{k}"] = pair_brier(p, ds, pairs)
    p = copy.deepcopy(base); p.w_s = p.w_s * 0; p.hist = {}; p.reset_runtime()
    r["irt no subject prior"] = pair_brier(p, ds, pairs)
    p = copy.deepcopy(base); p.w_i = p.w_i * 0; p.reset_runtime()
    r["irt no item prior"] = pair_brier(p, ds, pairs)
    p = copy.deepcopy(base); p.marginalize = False; p.reset_runtime()
    r["irt plug-in (no marginalize)"] = pair_brier(p, ds, pairs)
    # oracle offset: set mu_beta so the mean prediction matches the test base rate (shows offset cost)
    lo, hi = -6.0, 6.0
    for _ in range(30):
        p = copy.deepcopy(base); p.mu_beta = (lo + hi) / 2; p.var_beta = 1e-6; p.reset_runtime()
        m = np.mean([p.predict_batch([(ds.subjects[pr.subject], ds.items[i]) for i in pr.evl[:40]], []).mean() for pr in pairs])
        lo, hi = (p.mu_beta, hi) if m < rate else (lo, p.mu_beta)
    r["irt ORACLE offset"] = pair_brier(p, ds, pairs)
    rows[test_b] = r
    print(test_b, f"rate={rate:.3f} mu_beta={base.mu_beta:.2f} var_beta={base.var_beta:.2f} var_u={base.var_u:.2f} var_d={base.var_d:.2f}", flush=True)
names = list(next(iter(rows.values())))
print("\n| variant | " + " | ".join(rows) + " | mean |\n|---|" + "---|" * (len(rows) + 1))
for n in names:
    vals = [rows[b][n] for b in rows]
    print(f"| {n} | " + " | ".join(f"{v:.4f}" for v in vals) + f" | {np.mean(vals):.4f} |")
