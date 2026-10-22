"""Offline check of benchmark-level low-rank transfer of subject abilities.

For each dev fold: delta_{s,b} = theta_{s,b} - mean_b(theta). Training matrix identity x benchmark (mean delta).
Factorise R[id, b] ~ a_id + e_id . w_b (ALS, ridge). For each held-out benchmark b and eval subject s, predict
delta_{s,b} (centred over b's eval subjects) by
  P0  champion prior u(s)
  MF  u(s) + e_id . w_b with w_b fit (ridge toward the training-loading prior) on the OTHER eval subjects'
      true residuals delta - u (oracle: noise-free other subjects; an upper bound).
Reports residual MSE (centred per benchmark) vs var_u.
"""
import pickle
import sys
from collections import defaultdict

import numpy as np

D = pickle.load(open("analysis/subj/pairs.pkl", "rb"))
bn = D["bench_names"]
ident = D["ident"]
ps, pb, th, nit = D["pair_s"], D["pair_b"], D["theta"], D["n_items"]
beta = np.zeros(len(bn))
for b in np.unique(pb):
    beta[b] = th[pb == b].mean()
delta = th - beta[pb]
Ks = [int(k) for k in (sys.argv[1:] or ["1", "2", "4", "8"])]


def als(rows, cols, vals, nr, nc, K, lam=1.0, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    E = rng.normal(0, 0.3, (nr, K))
    W = rng.normal(0, 0.3, (nc, K))
    a = np.zeros(nr)
    byr, byc = defaultdict(list), defaultdict(list)
    for k, (r, c) in enumerate(zip(rows, cols)):
        byr[r].append(k)
        byc[c].append(k)
    for _ in range(iters):
        for r, ks in byr.items():   # [a_r, e_r]
            X = np.hstack([np.ones((len(ks), 1)), W[cols[ks]]])
            reg = lam * np.eye(K + 1)
            reg[0, 0] = 0.1
            sol = np.linalg.solve(X.T @ X + reg, X.T @ vals[ks])
            a[r], E[r] = sol[0], sol[1:]
        for c, ks in byc.items():
            X = E[rows[ks]]
            W[c] = np.linalg.solve(X.T @ X + lam * np.eye(K), X.T @ (vals[ks] - a[rows[ks]]))
    return a, E, W


for f in D["folds"]:
    tr = {bn.index(b) for b in f["train"]}
    m = np.isin(pb, list(tr))
    # training matrix identity x benchmark
    acc = defaultdict(list)
    for s, b, v in zip(ps[m], pb[m], delta[m]):
        acc[(ident[s], b)].append(v)
    ids = sorted({k[0] for k in acc})
    iid = {x: k for k, x in enumerate(ids)}
    bl = sorted(tr)
    bid = {b: k for k, b in enumerate(bl)}
    rows = np.array([iid[k[0]] for k in acc])
    cols = np.array([bid[k[1]] for k in acc])
    vals = np.array([np.mean(v) for v in acc.values()])
    cnt = np.bincount(rows, minlength=len(ids))
    ev = f["eval_pairs"]
    evb = defaultdict(list)
    for s, b in ev:
        evb[b].append(s)
    ecnt = [cnt[iid[ident[s]]] if ident[s] in iid else 0 for s, _ in ev]
    print(f"fold {f['group'][0]}..: {len(ev)} eval pairs, train ids {len(ids)}, var_u {f['var_u']:.3f}; eval-id train-bench "
          f"count median {np.median(ecnt):.0f}, q10 {np.quantile(ecnt, .1):.0f}, q90 {np.quantile(ecnt, .9):.0f}")
    lookup = {(ps[k], pb[k]): delta[k] for k in range(len(ps))}
    res0 = []
    resk = {K: [] for K in Ks}
    for K in Ks:
        a, E, W = als(rows, cols, vals, len(ids), len(bl), K)
        wm, wc = W.mean(0), np.cov(W.T).reshape(K, K) + 1e-6 * np.eye(K)
        for b, subs in evb.items():
            y = np.array([lookup[(s, b)] for s in subs])
            u = np.array([f["u"][s] for s in subs])
            e = np.array([E[iid[ident[s]]] if ident[s] in iid else np.zeros(K) for s in subs])
            r = y - u
            r0 = r - r.mean()
            if K == Ks[0]:
                res0.append(r0)
            out = []
            sig2 = f["var_u"]
            for j in range(len(subs)):
                o = np.arange(len(subs)) != j
                # residual model r = c + e.w + noise; c is absorbed by centring
                X = e[o] - e[o].mean(0)
                rr = r[o] - r[o].mean()
                Pinv = np.linalg.inv(wc)
                A = X.T @ X / sig2 + Pinv
                w = np.linalg.solve(A, X.T @ rr / sig2 + Pinv @ wm)
                out.append(r[j] - r[o].mean() - (e[j] - e[o].mean(0)) @ w)
            resk[K].append(np.array(out))
    r0 = np.concatenate(res0)
    print(f"   P0 centred resid MSE {np.mean(r0**2):.3f}  " + "  ".join(
        f"K={K}: {np.mean(np.concatenate(resk[K])**2):.3f}" for K in Ks))
