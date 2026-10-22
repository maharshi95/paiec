"""Hierarchical Bayesian linear regression of pair ability on subject descriptors, per benchmark.
delta_{s,b} = w_b . x_s + r,  w_b ~ N(wbar, Sigma) estimated from per-training-benchmark ridge fits.
Oracle LOO on held-out benchmarks: w_b posterior from the other eval subjects' true delta.
Also partial-information version: other subjects' delta observed with noise (var tau2) to mimic few labels.
"""
import pickle
import sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, "src")
from paiec.features.subject import SubjectFeaturizer
D = pickle.load(open("analysis/subj/pairs.pkl", "rb"))
bn, ident, subjects = D["bench_names"], D["ident"], D["subjects"]
ps, pb, th = D["pair_s"], D["pair_b"], D["theta"]
CLIP = float(sys.argv[1]) if len(sys.argv) > 1 else 99
import os
PROV = os.environ.get('PROV', '').split(',') if os.environ.get('PROV') else []
KEEP = [0] + [int(c) for c in sys.argv[2].split(",")] if len(sys.argv) > 2 else list(range(10))
beta = np.zeros(len(bn)); sd = np.ones(len(bn))
for b in np.unique(pb):
    beta[b] = th[pb == b].mean(); sd[b] = max(th[pb == b].std(), 0.3)
delta = np.clip(th - beta[pb], -CLIP, CLIP)
lookup = {(ps[k], pb[k]): delta[k] for k in range(len(ps))}
sf = SubjectFeaturizer()
dense = {}
def dn(s):
    if s not in dense: dense[s] = sf.transform_one(subjects[s])[-7:]
    return dense[s]
out = defaultdict(list)
for f in D["folds"]:
    trb = [bn.index(b) for b in f["train"]]
    tr = np.isin(pb, trb)
    acc = defaultdict(list)
    for k in np.nonzero(tr)[0]:
        acc[ident[ps[k]]].append(delta[k] / sd[pb[k]])
    hz = {i: np.sum(v) / (len(v) + 3) for i, v in acc.items()}
    hd = {}
    for k in np.nonzero(tr)[0]:
        hd.setdefault(ident[ps[k]], []).append(delta[k])
    def X(s, u):
        d = dn(s)
        full = np.r_[1.0, u, hz.get(ident[s], 0.0), d[0], d[1], d[2], d[3], d[4], d[5], d[6]]
        pv = subjects[s].get("provider", "").lower()
        return np.r_[full[KEEP], [float(pv == q) for q in PROV]]
    # per-training-benchmark regressions on pairs whose identity has history (use champion-like u = hist mean shrunk)
    W = []
    for c in trb:
        m = np.nonzero(pb == c)[0]
        if len(m) < 15: continue
        # u for training pairs: leave-this-benchmark-out history mean (approx)
        rows, ys = [], []
        for k in m:
            i = ident[ps[k]]
            v = hd.get(i, [])
            n = len(v) - 1
            uu = (np.sum(v) - delta[k]) / (n + 1) if n > 0 else 0.0
            zz = (hz.get(i, 0) * (len(v) + 3) - delta[k] / sd[c]) / (len(v) + 2) if v else 0.0
            x = X(ps[k], uu)
            rows.append(x); ys.append(delta[k])
        A, y = np.array(rows), np.array(ys)
        A[:, 0] = 1
        w = np.linalg.solve(A.T @ A + 1.0 * np.diag([0.01] + [1] * (A.shape[1] - 1)), A.T @ y)
        W.append(w)
    W = np.array(W)
    wbar, S = W.mean(0), np.cov(W.T) + 1e-3 * np.eye(W.shape[1])
    S[0, 0] = 100.0   # intercept free (absorbed by beta)
    if f is D["folds"][0]:
        print("wbar", np.round(wbar, 2), "\nsd", np.round(np.sqrt(np.diag(S)), 2), len(W))
    evb = defaultdict(list)
    for s, b in f["eval_pairs"]:
        evb[b].append(s)
    for b, subs in evb.items():
        if len(subs) < 4: continue
        y = np.array([lookup[(s, b)] for s in subs])
        Xb = np.array([X(s, f["u"][s]) for s in subs])
        u = Xb[:, 1]
        for tau2 in (0.5,):
            for sig2 in (1.0,):
                for j in range(len(subs)):
                    o = np.arange(len(subs)) != j
                    noise = sig2 + tau2
                    Si = np.linalg.inv(S)
                    A = Xb[o].T @ Xb[o] / noise + Si
                    rng = np.random.default_rng(j)
                    yo = y[o] + rng.normal(0, np.sqrt(tau2), o.sum()) if tau2 else y[o]
                    w = np.linalg.solve(A, Xb[o].T @ yo / noise + Si @ wbar)
                    out[f"hblr tau2={tau2}"].append((y[j] - Xb[j] @ w) ** 2)
                    # baseline: u with free offset
                    off = np.mean(yo - u[o])
                    out[f"u+offset tau2={tau2}"].append((y[j] - u[j] - off) ** 2)
                    # prior-only wbar with free intercept
                    off2 = np.mean(yo - Xb[o, 1:] @ wbar[1:])
                    out[f"wbar+offset tau2={tau2}"].append((y[j] - Xb[j, 1:] @ wbar[1:] - off2) ** 2)
for k, v in sorted(out.items()):
    print(f"{k:25s} {np.mean(v):.3f}  (n={len(v)})")
