"""Offline calibration probes on a dump (analysis/hier/dump.py).

Brier per pair = sum_trials (p - y)^2 / n_trials; ALC = weights . (pair-macro Brier per budget).
"""
import sys

import numpy as np
from scipy.optimize import minimize_scalar

W = {0: .1, 1: .2, 3: .2, 7: .2, 15: .2, 31: .1}
z = np.load(sys.argv[1])
bn = z["bench_names"]
c = z["comps"]
bm, bv, dm, dv, im, iv = c.T
m = bm + dm - im
v = bv + dv + iv
ys, nt, bud = z["ysum"], z["ntr"].astype(float), z["budget"]
pair = z["bench"].astype(np.int64) * 100000 + z["subj"].astype(np.int64) * 10 + z["seed"]
sig = lambda x: 1 / (1 + np.exp(-x))


def sq(p, mask=slice(None)):
    return nt[mask] * p ** 2 - 2 * p * ys[mask] + ys[mask]


def alc_of(p):
    """pair-macro Brier per budget and ALC (pairs pooled)."""
    out = {}
    for n in W:
        k = bud == n
        pk = pair[k]
        u, inv = np.unique(pk, return_inverse=True)
        out[n] = float(np.mean(np.bincount(inv, sq(p[k], k)) / np.bincount(inv, nt[k])))
    return sum(W[n] * out[n] for n in W), out


p0 = sig(m / np.sqrt(1 + np.pi * v / 8))
A0, b0 = alc_of(p0)
print(f"base ALC {A0:.4f} | " + " ".join(f"B{n}={b0[n]:.4f}" for n in W))
for n in W:
    k = bud == n
    print(f"  B{n}: mean p {np.average(p0[k], weights=nt[k]):.3f} mean y {ys[k].sum()/nt[k].sum():.3f} "
          f"mean v {v[k].mean():.2f} (bv {bv[k].mean():.2f} dv {dv[k].mean():.2f} iv {iv[k].mean():.2f}) "
          f"frac items labeled {np.mean(z['nlab_item'][k] > 0):.2f}")


def brier_flat(p, k):
    return float(sq(p[k], k).sum() / nt[k].sum())


print("\nper-budget optimal variance multiplier k (p = sig(m / sqrt(1 + pi k v / 8))) and temperature t:")
best_p = p0.copy()
for n in W:
    kk = bud == n
    f = lambda lk: brier_flat(sig(m / np.sqrt(1 + np.pi * np.exp(lk) * v / 8)), kk)
    r = minimize_scalar(f, bounds=(-3, 3), method="bounded")
    g = lambda t: brier_flat(sig(t * m / np.sqrt(1 + np.pi * v / 8)), kk)
    r2 = minimize_scalar(g, bounds=(0.2, 3), method="bounded")
    print(f"  B{n}: k*={np.exp(r.x):.2f} Brier {brier_flat(p0, kk):.4f} -> {r.fun:.4f} | t*={r2.x:.2f} -> {r2.fun:.4f}")
    best_p[kk] = sig(m[kk] / np.sqrt(1 + np.pi * np.exp(r.x) * v[kk] / 8))
A, b = alc_of(best_p)
print(f"  ALC with per-budget k*: {A:.4f} ({A - A0:+.4f})")

print("\nper-benchmark oracle (fit on the eval data itself, all budgets): temperature t_b, floor c_b, both")
pt, pc = p0.copy(), p0.copy()
rows = []
for b in np.unique(z["bench"]):
    kk = z["bench"] == b
    g = lambda t: brier_flat(sig(t * m / np.sqrt(1 + np.pi * v / 8)), kk)
    rt = minimize_scalar(g, bounds=(0.1, 5), method="bounded")
    pt[kk] = sig(rt.x * m[kk] / np.sqrt(1 + np.pi * v[kk] / 8))
    h = lambda cc: brier_flat(cc + (1 - cc) * p0, kk)
    rc = minimize_scalar(h, bounds=(0, 0.6), method="bounded")
    pc[kk] = rc.x + (1 - rc.x) * p0[kk]
    rows.append((str(bn[b]), rt.x, rc.x, brier_flat(p0, kk), rt.fun, rc.fun, ys[kk].sum() / nt[kk].sum()))
for r in sorted(rows, key=lambda r: r[4] - r[3]):
    print(f"  {r[0]:>30}: t*={r[1]:.2f} c*={r[2]:.2f} base rate {r[6]:.2f} Brier {r[3]:.4f} -> t {r[4]:.4f} c {r[5]:.4f}")
print(f"  ALC oracle t_b: {alc_of(pt)[0]:.4f} ({alc_of(pt)[0] - A0:+.4f}); oracle c_b: {alc_of(pc)[0]:.4f} ({alc_of(pc)[0] - A0:+.4f})")

print("\nper-budget x benchmark oracle temperature (upper bound for budget-dependent per-benchmark scale):")
pb = p0.copy()
for b in np.unique(z["bench"]):
    for n in W:
        kk = (z["bench"] == b) & (bud == n)
        g = lambda t: brier_flat(sig(t * m / np.sqrt(1 + np.pi * v / 8)), kk)
        rt = minimize_scalar(g, bounds=(0.1, 5), method="bounded")
        pb[kk] = sig(rt.x * m[kk] / np.sqrt(1 + np.pi * v[kk] / 8))
Ab, bb = alc_of(pb)
print(f"  ALC {Ab:.4f} ({Ab - A0:+.4f}) | " + " ".join(f"B{n}={bb[n]:.4f}" for n in W))

print("\nlabeled vs unlabeled eval items: Brier and calibration by budget")
for n in W:
    for lab in (False, True):
        kk = (bud == n) & ((z["nlab_item"] > 0) == lab)
        if kk.sum():
            print(f"  B{n} labeled={lab:d}: n={kk.sum():6d} Brier {brier_flat(p0, kk):.4f} mean p {np.average(p0[kk], weights=nt[kk]):.3f} "
                  f"mean y {ys[kk].sum()/nt[kk].sum():.3f}")
