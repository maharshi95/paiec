import pickle
from collections import defaultdict
import numpy as np
D = pickle.load(open("analysis/subj/pairs.pkl", "rb"))
bn, ident = D["bench_names"], D["ident"]
ps, pb, th = D["pair_s"], D["pair_b"], D["theta"]
beta = np.zeros(len(bn))
for b in np.unique(pb):
    beta[b] = th[pb == b].mean()
delta = th - beta[pb]
lookup = {(ps[k], pb[k]): delta[k] for k in range(len(ps))}
pm = {(ps[k], pb[k]): D["pair_mean"][k] for k in range(len(ps))}
tot = defaultdict(float)
print(f"{'bench':28s} n  sd_true sd_u  corr slope  mse0  mse_slope  base")
for f in D["folds"]:
    evb = defaultdict(list)
    for s, b in f["eval_pairs"]:
        evb[b].append(s)
    for b, subs in evb.items():
        y = np.array([lookup[(s, b)] for s in subs]); u = np.array([f["u"][s] for s in subs])
        if len(subs) < 3:
            continue
        yc, uc = y - y.mean(), u - u.mean()
        c = np.corrcoef(y, u)[0, 1]; sl = (yc @ uc) / (uc @ uc)
        print(f"{bn[b]:28s} {len(subs):2d} {yc.std():6.2f} {uc.std():5.2f} {c:5.2f} {sl:5.2f} {np.mean((yc-uc)**2):6.2f} {np.mean((yc-sl*uc)**2):6.2f}  {np.mean([pm[(s,b)] for s in subs]):.2f}")
        tot["n"] += len(subs); tot["m0"] += np.sum((yc-uc)**2); tot["ms"] += np.sum((yc-sl*uc)**2); tot["v"] += np.sum(yc**2)
print({k: v / tot["n"] for k, v in tot.items()})
