"""Which identity score best predicts held-out-benchmark pair ability?  For each candidate score x(s):
  raw:  centred MSE of (delta - x) [only for scores on delta scale]
  aff:  LOO affine fit delta ~ a + c x on the other eval subjects of the same benchmark (what test-time
        learning of a per-benchmark slope could reach with many labels), MSE on held-out subject.
"""
import pickle
from collections import defaultdict
import numpy as np
D = pickle.load(open("analysis/subj/pairs.pkl", "rb"))
bn, ident = D["bench_names"], D["ident"]
ps, pb, th = D["pair_s"], D["pair_b"], D["theta"]
pmean = D["pair_mean"]
beta = np.zeros(len(bn)); sd = np.ones(len(bn))
for b in np.unique(pb):
    beta[b] = th[pb == b].mean(); sd[b] = max(th[pb == b].std(), 0.3)
delta = th - beta[pb]
lookup = {(ps[k], pb[k]): delta[k] for k in range(len(ps))}
# per-benchmark percentile rank
rank = np.zeros(len(ps))
for b in np.unique(pb):
    m = np.nonzero(pb == b)[0]
    rank[m] = (np.argsort(np.argsort(th[m])) + 0.5) / len(m) - 0.5
lg = np.log(np.clip(pmean, .02, .98) / (1 - np.clip(pmean, .02, .98)))
lgc = lg.copy()
for b in np.unique(pb):
    m = pb == b; lgc[m] -= lg[m].mean()
agg = defaultdict(lambda: defaultdict(list))
R = defaultdict(lambda: defaultdict(list))
for f in D["folds"]:
    tr = np.isin(pb, [bn.index(b) for b in f["train"]])
    acc = defaultdict(lambda: defaultdict(list))
    for k in np.nonzero(tr)[0]:
        i = ident[ps[k]]
        acc[i]["d"].append(delta[k]); acc[i]["z"].append(delta[k] / sd[pb[k]]); acc[i]["r"].append(rank[k])
        acc[i]["l"].append(lgc[k])
    def score(s, name):
        a = acc.get(ident[s])
        if not a: return 0.0
        if name == "u": return f["u"][s]
        if name == "hist": return np.mean(a["d"])
        if name == "hz": return np.mean(a["z"])
        if name == "rank": return np.mean(a["r"])
        if name == "logit": return np.mean(a["l"])
        if name == "hz_shr": return np.sum(a["z"]) / (len(a["z"]) + 3)
        if name == "rank_shr": return np.sum(a["r"]) / (len(a["r"]) + 3)
    names = ["u", "hist", "hz", "rank", "logit", "hz_shr", "rank_shr"]
    evb = defaultdict(list)
    for s, b in f["eval_pairs"]:
        evb[b].append(s)
    for b, subs in evb.items():
        if len(subs) < 4: continue
        y = np.array([lookup[(s, b)] for s in subs])
        for nm in names:
            x = np.array([score(s, nm) for s in subs])
            for j in range(len(subs)):
                o = np.arange(len(subs)) != j
                A = np.c_[np.ones(o.sum()), x[o]]
                coef = np.linalg.solve(A.T @ A + np.diag([0, 1e-3]), A.T @ y[o])
                R[nm]["aff"].append((y[j] - coef[0] - coef[1] * x[j]) ** 2)
                R[nm]["raw"].append((y[j] - y[o].mean() - (x[j] - x[o].mean())) ** 2)
                R[nm]["corr"].append(0)
            R[nm]["c"].append(np.corrcoef(x, y)[0, 1] if x.std() > 0 else 0)
        for j in range(len(subs)):
            o = np.arange(len(subs)) != j
            R["mean"]["aff"].append((y[j] - y[o].mean()) ** 2)
for nm, r in R.items():
    print(f"{nm:10s} LOO-raw {np.mean(r['raw']) if r['raw'] else float('nan'):.3f}  LOO-affine {np.mean(r['aff']):.3f}  "
          f"median corr {np.median(r['c']) if r['c'] else float('nan'):.2f} mean corr {np.mean(r['c']) if r['c'] else float('nan'):.2f}")
