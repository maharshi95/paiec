"""Why do the static CaimiraLite retrains score ~0.24 on dev? Rank quality (AUC), base-rate error and an oracle base-rate
shift on the fold's unseen benchmarks, from the cached fits (src/paiec/fitcache.py).

    PYTHONPATH=src python scripts/diag_ref_static.py [n_pickles]
"""
import glob
import pickle
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from paiec.config import load_config
from paiec.data.dataset import load_dataset

d = load_config("configs/research/dev.yaml")["data"]
ds = load_dataset(d["root"], d["benchmarks"], d["binarize"], d["cache_dir"], max_responses=d.get("max_responses"),
                  max_item_chars=d.get("max_item_chars"), prefer_seen_from=d.get("prefer_seen_from"))
rng = np.random.default_rng(0)
done, limit = 0, int(sys.argv[1]) if len(sys.argv) > 1 else 2
for f in sorted(glob.glob("/fs/clip-scratch/mgor/paiec-cache/fits/RefCaimira_*.pkl")):
    p = pickle.load(open(f, "rb"))
    if getattr(p, "kind", "") != "irtm":
        continue
    unseen = [b for b in range(len(ds.bench_names)) if ds.bench_names[b] not in p.b_map and (ds.resp_b == b).any()]
    print(f.split("/")[-1], "unseen benchmarks:", len(unseen))
    for b in unseen:
        m = rng.choice(np.nonzero(ds.resp_b == b)[0], 3000, replace=True)
        inputs = [(ds.subjects[ds.resp_s[k]], ds.items[ds.resp_i[k]]) for k in m]
        y = ds.resp_y[m]
        lg = p._logits(inputs)
        pr = 1 / (1 + np.exp(-lg))
        lo, hi = -8, 8
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if (1 / (1 + np.exp(-(lg + mid)))).mean() < y.mean() else (lo, mid)
        pr2 = 1 / (1 + np.exp(-(lg + mid)))
        auc = roc_auc_score(y, lg) if 0 < y.mean() < 1 else float("nan")
        print(f"  {ds.bench_names[b][:22]:22s} base {y.mean():.2f} meanp {pr.mean():.2f} brier {((pr - y) ** 2).mean():.4f} "
              f"const.5 {((0.5 - y) ** 2).mean():.4f} oracle-shift {((pr2 - y) ** 2).mean():.4f} AUC {auc:.3f}")
    done += 1
    if done >= limit:
        break
