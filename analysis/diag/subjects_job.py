"""Q4: subject-transfer potential on the dev universe.

Per-benchmark Rasch on every benchmark -> pair ability delta = theta - beta per (subject, benchmark), keyed by model
identity (features.subject.subject_identity). Reports identity coverage for dev eval subjects, cross-benchmark
correlation of pair abilities, and a low-rank completion simulation. Writes out/subjects.pkl and prints a summary.
"""
import pickle
import time
from collections import defaultdict

import numpy as np
from common import OUT, allowed_subjects, fold_pairs, load

from paiec.features.subject import subject_identity
from paiec.predictors.irt import rasch


def main():
    t0 = time.time()
    cfg, ds = load()
    print(f"loaded {time.time() - t0:.0f}s", flush=True)
    rb = ds.resp_b
    ident_of = [subject_identity(s) for s in ds.subjects]
    # per-benchmark Rasch
    M = defaultdict(dict)   # identity -> {bench: delta}
    for b in np.unique(rb):
        m = rb == b
        rs, ri, ry = ds.resp_s[m], ds.resp_i[m], ds.resp_y[m]
        su, P0 = np.unique(rs, return_inverse=True)
        iu, J0 = np.unique(ri, return_inverse=True)
        nI = len(iu)
        c, inv = np.unique(P0 * nI + J0, return_inverse=True)
        th, d = rasch(c // nI, c % nI, np.bincount(inv, ry), np.bincount(inv).astype(float), len(su), nI, 9.0, 60)
        th = th - d.mean()
        delta = th - th.mean()
        acc = defaultdict(list)
        for s, v in zip(su, delta):
            acc[ident_of[s]].append(v)
        for idn, v in acc.items():
            M[idn][ds.bench_names[b]] = float(np.mean(v))
    print(f"rasch done {time.time() - t0:.0f}s: {len(M)} identities", flush=True)
    out = {"M": dict(M)}
    # coverage of dev eval subjects
    allowed = allowed_subjects(cfg, ds)
    cov = []
    for k, group in enumerate(cfg["cv"]["test_benchmarks"]):
        prs = fold_pairs(cfg, ds, group, 0, allowed)
        for pr in prs:
            idn = ident_of[pr.subject]
            bs = set(M[idn]) - set(group)
            same_fold = (set(M[idn]) & set(group)) - {ds.bench_names[pr.bench]}
            cov.append({"fold": k, "bench": ds.bench_names[pr.bench], "ident": idn, "n_train": len(bs),
                        "n_fold_other": len(same_fold)})
    out["coverage"] = cov
    n = np.array([c["n_train"] for c in cov])
    nf = np.array([c["n_fold_other"] for c in cov])
    print(f"eval pairs {len(cov)}, distinct identities {len({c['ident'] for c in cov})}")
    print("train benchmarks per eval identity: mean %.1f median %.0f; >=1: %.0f%%  >=5: %.0f%%  >=20: %.0f%%  >=50: %.0f%%"
          % (n.mean(), np.median(n), 100 * (n >= 1).mean(), 100 * (n >= 5).mean(), 100 * (n >= 20).mean(), 100 * (n >= 50).mean()))
    print("other test benchmarks in same fold with this identity: mean %.2f, >=1: %.0f%%" % (nf.mean(), 100 * (nf >= 1).mean()))
    # cross-benchmark correlation of pair abilities
    benches = sorted({b for v in M.values() for b in v})
    bi = {b: k for k, b in enumerate(benches)}
    ids = [i for i in M if len(M[i]) >= 2]
    X = np.full((len(ids), len(benches)), np.nan)
    for r, i in enumerate(ids):
        for b, v in M[i].items():
            X[r, bi[b]] = v
    obs = ~np.isnan(X)
    rs = []
    for a in range(len(benches)):
        for c in range(a + 1, len(benches)):
            m = obs[:, a] & obs[:, c]
            if m.sum() >= 10 and X[m, a].std() > 0 and X[m, c].std() > 0:
                rs.append((np.corrcoef(X[m, a], X[m, c])[0, 1], int(m.sum()), benches[a], benches[c]))
    r = np.array([x[0] for x in rs])
    w = np.array([x[1] for x in rs])
    out["pair_corr"] = rs
    print(f"benchmark pairs with >=10 shared identities: {len(rs)}; r median {np.median(r):.2f} IQR "
          f"[{np.percentile(r, 25):.2f}, {np.percentile(r, 75):.2f}] weighted mean {np.average(r, weights=w):.2f}")
    # variance decomposition: identity mean vs within-identity
    within, tot = [], []
    for i in ids:
        v = np.array(list(M[i].values()))
        within.append(((v - v.mean()) ** 2).sum())
        tot.append(len(v) - 1)
    allv = X[obs]
    var_within = sum(within) / sum(tot)
    print(f"var(delta) over identities with >=2 benchmarks {allv.var():.3f}; within-identity var {var_within:.3f} "
          f"-> identity share {1 - var_within / allv.var():.2f}")
    out["var_decomp"] = (float(allv.var()), float(var_within))
    # low-rank completion simulation
    rng = np.random.default_rng(0)
    rows_ok = obs.sum(1) >= 3
    Xr, Or = X[rows_ok], obs[rows_ok]
    targets = [c for c in range(len(benches)) if Or[:, c].sum() >= 15]
    targets = sorted(targets, key=lambda c: -Or[:, c].sum())[:40]
    print(f"simulation: {Xr.shape[0]} identities with >=3 benchmarks, {len(targets)} target benchmarks with >=15 of them")
    sim = defaultdict(list)
    for c in targets:
        O2 = Or.copy()
        O2[:, c] = False
        keep = O2.sum(1) >= 2
        rows = np.nonzero(keep & Or[:, c])[0]
        if len(rows) < 12:
            continue
        y = Xr[rows, c]
        vy = y.var()
        for K in (0, 1, 2, 3):
            E = als(Xr, O2, K) if K else None
            mu_id = np.array([Xr[r_, O2[r_]].mean() for r_ in range(len(Xr))])
            F = np.column_stack([mu_id[rows]] + ([E[rows]] if K else []))
            for nrev in (0, 3, 5, 10):
                errs = []
                for rep in range(20):
                    perm = rng.permutation(len(rows))
                    if nrev == 0:   # identity mean on other benchmarks, no fitting
                        errs.append(np.mean((y - mu_id[rows]) ** 2))
                        break
                    if nrev >= len(rows) - 2:
                        continue
                    A, B = perm[:nrev], perm[nrev:]
                    pred = ridge_pred(F[A], y[A], F[B], lam=1.0)
                    errs.append(np.mean((y[B] - pred) ** 2))
                if errs:
                    sim[(K, nrev, "clean")].append(1 - np.mean(errs) / vy)
            # noisy: every other subject has k labels (noise var 4/k); predict held-out subject, then combine with own
            for k in (1, 3, 7, 15, 31):
                nv = 4.0 / k
                e_lr, e_own, e_comb, e_pri = [], [], [], []
                for rep in range(5):
                    yn = y + rng.normal(0, np.sqrt(nv), len(y))
                    for t in range(len(rows)):
                        oth = np.delete(np.arange(len(rows)), t)
                        p_lr = ridge_pred(F[oth], yn[oth], F[t:t + 1], lam=1.0 + nv)[0]
                        res_v = max(np.mean((yn[oth] - ridge_pred(F[oth], yn[oth], F[oth], lam=1.0 + nv)) ** 2) - nv, 0.05)
                        own = yn[t] * vy / (vy + nv)
                        comb = (p_lr / res_v + yn[t] / nv) / (1 / res_v + 1 / nv)
                        pri = mu_id[rows[t]]
                        e_lr.append((y[t] - p_lr) ** 2); e_own.append((y[t] - own) ** 2)
                        e_comb.append((y[t] - comb) ** 2); e_pri.append((y[t] - pri) ** 2)
                for tag, e in (("lr_only", e_lr), ("own_only", e_own), ("combined", e_comb), ("idmean", e_pri)):
                    sim[(K, k, "noisy_" + tag)].append(1 - np.mean(e) / vy)
    out["sim"] = dict(sim)
    print("R^2 of held-out-benchmark subject abilities (median over target benchmarks):")
    for key in sorted(sim, key=str):
        print(f"  K={key[0]} n/k={key[1]:>2} {key[2]:<16} R2 median {np.median(sim[key]):+.3f} mean {np.mean(sim[key]):+.3f} "
              f"(n={len(sim[key])})")
    with open(OUT / "subjects.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"done {time.time() - t0:.0f}s")


def ridge_pred(F, y, Fq, lam=1.0):
    F1 = np.column_stack([np.ones(len(F)), F])
    Q1 = np.column_stack([np.ones(len(Fq)), Fq])
    R = lam * np.eye(F1.shape[1])
    R[0, 0] = 1e-6
    w = np.linalg.solve(F1.T @ F1 + R, F1.T @ y)
    return Q1 @ w


def als(X, O, K, lam=1.0, iters=15):
    """Low-rank fit of the observed entries of X (rows: identities, cols: benchmarks) with row/col means."""
    n, m = X.shape
    rng = np.random.default_rng(1)
    U, V = 0.1 * rng.standard_normal((n, K)), 0.1 * rng.standard_normal((m, K))
    rmean = np.array([X[r, O[r]].mean() if O[r].any() else 0.0 for r in range(n)])
    R = np.where(O, X - rmean[:, None], 0.0)
    for _ in range(iters):
        for r in range(n):
            c = O[r]
            if c.any():
                U[r] = np.linalg.solve(V[c].T @ V[c] + lam * np.eye(K), V[c].T @ R[r, c])
        for c_ in range(m):
            r_ = O[:, c_]
            if r_.any():
                V[c_] = np.linalg.solve(U[r_].T @ U[r_] + lam * np.eye(K), U[r_].T @ R[r_, c_])
    return U


if __name__ == "__main__":
    main()
