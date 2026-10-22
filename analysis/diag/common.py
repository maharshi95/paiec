"""Shared helpers for the diag direction: dev-protocol data, folds, pairs, in-sample benchmark fits."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from paiec.config import load_config  # noqa: E402
from paiec.features.subject import subject_identity, subject_key  # noqa: E402
from paiec.predictors.base import item_key  # noqa: E402
from paiec.sim.replay import build_pairs  # noqa: E402

CHAMPION = str(ROOT / "configs/research/champion.yaml")
OUT = ROOT / "analysis/diag/out"


def load(cfg_path=CHAMPION, overrides=("cv.seeds=[0]",)):
    from paiec.data.dataset import load_dataset
    cfg = load_config(cfg_path, list(overrides))
    d = cfg["data"]
    ds = load_dataset(d["root"], d["benchmarks"], d["binarize"], d["cache_dir"], max_responses=d.get("max_responses"),
                      max_item_chars=d.get("max_item_chars"), prefer_seen_from=d.get("prefer_seen_from"))
    return cfg, ds


def allowed_subjects(cfg, ds):
    cv = cfg["cv"]
    if not cv.get("eval_seen_from"):
        return None
    ref = np.isin(ds.resp_b, [ds.bench_names.index(b) for b in cv["eval_seen_from"] if b in ds.bench_names])
    seen = {subject_identity(ds.subjects[s]) for s in np.unique(ds.resp_s[ref])}
    return {k for k, s in enumerate(ds.subjects) if subject_identity(s) in seen}


def fold_pairs(cfg, ds, group, seed=0, allowed=None):
    cv = cfg["cv"]
    return build_pairs(ds, group, cv["split_seed"], seed, cv["min_items"], cv["max_subjects_per_benchmark"],
                       cv["max_eval_items_per_pair"], allowed)


class BenchCells:
    """All responses of one benchmark aggregated per (subject_key, item_key) cell."""

    def __init__(self, ds, b, resp_mask=None):
        m = ds.resp_b == b
        if resp_mask is not None:
            m &= resp_mask
        rs, ri, ry = ds.resp_s[m], ds.resp_i[m], ds.resp_y[m]
        self.skeys, self.ikeys = {}, {}
        self.s_of_ds, self.i_of_ds = {}, {}
        for s in np.unique(rs):
            self.s_of_ds[int(s)] = self.skeys.setdefault(subject_key(ds.subjects[s]), len(self.skeys))
        for i in np.unique(ri):
            self.i_of_ds[int(i)] = self.ikeys.setdefault(item_key(ds.items[i]), len(self.ikeys))
        S = np.array([self.s_of_ds[int(s)] for s in rs])
        Ii = np.array([self.i_of_ds[int(i)] for i in ri])
        nI = len(self.ikeys)
        c, inv = np.unique(S * nI + Ii, return_inverse=True)
        self.P, self.J = c // nI, c % nI
        self.ysum, self.n = np.bincount(inv, ry), np.bincount(inv).astype(float)
        self.nS, self.nI = len(self.skeys), nI
        self.items_ds = {}
        for i, j in self.i_of_ds.items():
            self.items_ds.setdefault(j, i)


def fit_model(c, kind="rasch", K=0, lam=1.0, iters=300, prior_var=9.0):
    """MAP fit on aggregated cells with torch LBFGS.

    eta = a_j (theta_s - d_j) + U_s.V_j;   p = lo + (1 - lo - hi) sigmoid(eta)
    kind: rasch | 2pl | 4pl (per-benchmark lower/upper asymptotes) | lr (rasch + rank-K interaction)
    Returns a dict of numpy params and a predict(s_idx, j_idx) closure.
    """
    import torch
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    P, J = torch.as_tensor(c.P), torch.as_tensor(c.J)
    ys, n = torch.as_tensor(c.ysum), torch.as_tensor(c.n)
    th = torch.zeros(c.nS, requires_grad=True)
    d = torch.zeros(c.nI, requires_grad=True)
    params = [th, d]
    la = U = V = q = None
    if kind == "2pl":
        la = torch.zeros(c.nI, requires_grad=True)
        params.append(la)
    if kind == "4pl":
        q = torch.tensor([-4.0, -4.0], requires_grad=True)
        params.append(q)
    if kind == "lr":
        g = torch.Generator().manual_seed(0)
        U = (0.1 * torch.randn(c.nS, K, generator=g)).requires_grad_()
        V = (0.1 * torch.randn(c.nI, K, generator=g)).requires_grad_()
        params += [U, V]

    def prob(Pi, Ji):
        eta = th[Pi] - d[Ji]
        if la is not None:
            eta = torch.exp(la[Ji]) * eta
        if U is not None:
            eta = eta + (U[Pi] * V[Ji]).sum(1)
        p = torch.sigmoid(eta)
        if q is not None:
            lo, hi = 0.5 * torch.sigmoid(q[0]), 0.5 * torch.sigmoid(q[1])
            p = lo + (1 - lo - hi) * p
        return p.clamp(1e-9, 1 - 1e-9)

    def loss():
        p = prob(P, J)
        nll = -(ys * torch.log(p) + (n - ys) * torch.log1p(-p)).sum()
        reg = (th ** 2).sum() / (2 * prior_var) + (d ** 2).sum() / (2 * prior_var)
        if la is not None:
            reg = reg + (la ** 2).sum() / (2 * 0.25)
        if U is not None:
            reg = reg + lam * ((U ** 2).sum() + (V ** 2).sum()) / 2
        return nll + reg

    opt = torch.optim.LBFGS(params, lr=1, max_iter=iters, line_search_fn="strong_wolfe", tolerance_grad=1e-6,
                            tolerance_change=1e-10, history_size=20)

    def closure():
        opt.zero_grad()
        lo_ = loss()
        lo_.backward()
        return lo_
    opt.step(closure)
    with torch.no_grad():
        out = {"theta": th.numpy().copy(), "d": d.numpy().copy()}
        if q is not None:
            out["asym"] = (0.5 * torch.sigmoid(q)).numpy().copy()
        if la is not None:
            out["a"] = torch.exp(la).numpy().copy()

    def predict(Si, Jj):
        with torch.no_grad():
            return prob(torch.as_tensor(np.asarray(Si)), torch.as_tensor(np.asarray(Jj))).numpy()
    out["predict"] = predict
    return out
