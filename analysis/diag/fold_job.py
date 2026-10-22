"""Per-fold diagnostics for the champion (dev protocol, seed 0).

    python analysis/diag/fold_job.py --fold K

Writes analysis/diag/out/fold{K}.pkl with
  replays[variant] = list of per-pair dicts {benchmark, subject, B0..B31, probs{n: array}, ys[list of arrays]}
    variants: champion and hybrid oracles (beta, delta, d, and combinations) plugged into the champion's adaptation
  static[model] = list of per-pair Brier (constant over budgets): in-sample and cross-fitted (acquisition pools only)
    Rasch / 2PL / 4PL / low-rank K=2,3, per-pair base rate
  bench[b] = benchmark characteristics (items, subjects, trials, base rate, MCQ-likeness, trial-noise floor)
  items[b] = within-benchmark item-difficulty transfer (ridge R2, Spearman) for several feature sets
"""
import argparse
import copy
import pickle
import re
import time

import numpy as np
from common import OUT, BenchCells, allowed_subjects, fit_model, fold_pairs, load

from paiec.acquisition import build_policy
from paiec.features.item import ItemFeaturizer
from paiec.features.subject import subject_key
from paiec.fitcache import fit_cached
from paiec.predictors import build_predictor
from paiec.predictors.base import item_key
from paiec.predictors.irt import IRT
from paiec.predictors.simple import sigmoid
from paiec.sim.replay import replay, score_pairs

BUDGETS = (0, 1, 3, 7, 15, 31)


class OracleIRT(IRT):
    """Champion IRT with selected parameters replaced by in-sample oracle values (fixed, zero variance).
    Parametrisation matches item_target=absolute: beta' = beta + mean(yh), d'_i = d_c,i + mean(yh), delta = theta - beta."""
    oracle_beta = False
    oracle_delta = False
    oracle_d = False

    def _fit_bench(self, obs, warm=None):
        bid = obs[0][4].get("benchmark_id", "")
        orc = self.oracle_tab.get(bid)
        vs = self.var_scale
        var_b, var_u, var_d = self.var_beta * vs, self.var_u * vs, self.var_d * vs
        pidx, iidx, subj, items = {}, {}, [], []
        for sk, ik, _, s, i in obs:
            if sk not in pidx:
                pidx[sk] = len(subj)
                subj.append(s)
            if ik not in iidx:
                iidx[ik] = len(items)
                items.append(i)
        P = np.array([pidx[o[0]] for o in obs])
        J = np.array([iidx[o[1]] for o in obs])
        y = np.array([o[2] for o in obs])
        u = np.array([self.subject_prior(s) for s in subj])
        yh = self.item_priors(items)
        nP, nJ = len(u), len(yh)
        beta, delta, d = self.mu_beta, u.copy(), yh.copy()
        fb, fdel, fd = False, np.zeros(nP, bool), np.zeros(nJ, bool)
        if orc is not None:
            if self.oracle_beta:
                beta, fb = orc["beta"], True
            if self.oracle_delta:
                for sk, k in pidx.items():
                    if sk in orc["delta"]:
                        delta[k], fdel[k] = orc["delta"][sk], True
            if self.oracle_d:
                for ik, k in iidx.items():
                    if ik in orc["d"]:
                        d[k], fd[k] = orc["d"][ik], True
        if warm:
            if not fb:
                beta = warm["beta"][0]
            for sk, k in pidx.items():
                if sk in warm["delta"] and not fdel[k]:
                    delta[k] = warm["delta"][sk][0]
            for ik, k in iidx.items():
                if ik in warm["d"] and not fd[k]:
                    d[k] = warm["d"][ik][0]
        for _ in range(self.adapt_iters):
            if not fb:
                mu = sigmoid(beta + delta[P] - d[J])
                beta += ((y - mu).sum() - (beta - self.mu_beta) / var_b) / ((mu * (1 - mu)).sum() + 1 / var_b)
            mu = sigmoid(beta + delta[P] - d[J])
            step = (np.bincount(P, y - mu, nP) - (delta - u) / var_u) / (np.bincount(P, mu * (1 - mu), nP) + 1 / var_u)
            delta += np.where(fdel, 0.0, step)
            mu = sigmoid(beta + delta[P] - d[J])
            step = (-np.bincount(J, y - mu, nJ) - (d - yh) / var_d) / (np.bincount(J, mu * (1 - mu), nJ) + 1 / var_d)
            d += np.where(fd, 0.0, step)
        w = sigmoid(beta + delta[P] - d[J])
        w = w * (1 - w)
        vb = 0.0 if fb else 1 / (w.sum() + 1 / var_b)
        vdel = np.where(fdel, 0.0, 1 / (np.bincount(P, w, nP) + 1 / var_u))
        vd = np.where(fd, 0.0, 1 / (np.bincount(J, w, nJ) + 1 / var_d))
        return {"beta": (float(beta), float(vb)),
                "delta": {sk: (float(delta[k]), float(vdel[k])) for sk, k in pidx.items()},
                "d": {ik: (float(d[k]), float(vd[k])) for ik, k in iidx.items()}}

    def score_batch(self, state, inputs):
        vs = self.var_scale
        yh = self.item_priors([i for _, i in inputs])
        m, v = np.empty(len(inputs)), np.empty(len(inputs))
        fits = state["fits"]
        for k, (s, i) in enumerate(inputs):
            bid = i.get("benchmark_id", "")
            orc = self.oracle_tab.get(bid, {"delta": {}, "d": {}})
            st = fits.get(bid)
            sk, ik = subject_key(s), item_key(i)
            if self.oracle_beta and "beta" in orc:
                bm, bv = orc["beta"], 0.0
            else:
                bm, bv = st["beta"] if st else (self.mu_beta, self.var_beta * vs)
            if self.oracle_delta and sk in orc["delta"]:
                dm, dv = orc["delta"][sk], 0.0
            else:
                dm, dv = st["delta"].get(sk, (None, None)) if st else (None, None)
                if dm is None:
                    dm, dv = self.subject_prior(s), self.var_u * vs
            if self.oracle_d and ik in orc["d"]:
                im, iv = orc["d"][ik], 0.0
            else:
                im, iv = st["d"].get(ik, (None, None)) if st else (None, None)
                if im is None:
                    im, iv = yh[k], self.var_d * vs
            m[k], v[k] = bm + dm - im, bv + dv + iv
        return sigmoid(m / np.sqrt(1 + np.pi * v / 8)) if self.marginalize else sigmoid(m)


MCQ = re.compile(r"(^|\n)\s*[\(\[]?[A-D][\)\].:]\s|\boptions?\s*:|\bchoices?\s*:|answer with (the )?(letter|option)",
                 re.I)


def pair_brier(p, ys):
    return sum(float(((pk - y) ** 2).sum()) for pk, y in zip(p, ys)) / sum(len(y) for y in ys)


def static_eval(pairs, cells, fit, fallback):
    """Per-pair Brier of a fitted static model on eval items (keys through cells' key maps)."""
    out = []
    for pr in pairs:
        si = cells.skeys.get(subject_key(DS.subjects[pr.subject]))
        js = [cells.ikeys.get(item_key(DS.items[i])) for i in pr.evl]
        ok = np.array([j is not None for j in js]) & (si is not None)
        p = np.full(len(js), fallback(pr))
        if ok.any():
            p[ok] = fit["predict"](np.full(ok.sum(), si), np.array([j for j in js if j is not None]))
        out.append(pair_brier(p, pr.evl_ys))
    return out


def main():
    global DS
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--skip-replay", action="store_true")
    ap.add_argument("--source", default="full", help="oracle source: full (in-sample) | cf (all data except eval pools of evaluated pairs)")
    a = ap.parse_args()
    t0 = time.time()
    cfg, ds = load()
    DS = ds
    print(f"loaded {time.time() - t0:.0f}s", flush=True)
    group = cfg["cv"]["test_benchmarks"][a.fold]
    benches = [b for b in cfg["data"]["benchmarks"] if b in ds.bench_names and b in ds.benchmarks()]
    group = [b for b in group if b in benches]
    train_b = [b for b in benches if b not in group]
    pred = fit_cached(build_predictor(cfg["predictor"]), ds.restrict(train_b), train_b, cfg["data"], ds.revision)
    print(f"fit {time.time() - t0:.0f}s var_beta={pred.var_beta:.3f} var_u={pred.var_u:.3f} var_d={pred.var_d:.3f} "
          f"var_pair={pred.var_pair:.3f}", flush=True)
    allowed = allowed_subjects(cfg, ds)
    pairs0 = fold_pairs(cfg, ds, group, 0, allowed)
    res = {"fold": a.fold, "group": group, "vars": {k: getattr(pred, k) for k in ("var_beta", "var_u", "var_d", "var_pair", "var_g")},
           "pairs": [(ds.subject_ids[p.subject], ds.bench_names[p.bench]) for p in pairs0],
           "static": {}, "bench": {}, "items": {}, "oracle_vals": {}}
    oracle_tab = {}
    by_b = {}
    for pr in pairs0:
        by_b.setdefault(pr.bench, []).append(pr)
    for b, prs in by_b.items():
        name = ds.bench_names[b]
        anon = ds.items[prs[0].evl[0]]["benchmark_id"]
        full = BenchCells(ds, b)
        # cross-fit: drop eval-pool responses of the evaluated pairs (keep only their acquisition source items)
        excl = np.zeros(len(ds.resp_s), bool)
        mb = np.nonzero(ds.resp_b == b)[0]
        acq_src = {pr.subject: {ds.item_ids[i].split("|", 1)[0] for i in pr.acq} for pr in prs}
        for k in mb:
            s = int(ds.resp_s[k])
            if s in acq_src and ds.item_ids[ds.resp_i[k]].split("|", 1)[0] not in acq_src[s]:
                excl[k] = True
        cf = BenchCells(ds, b, ~excl)
        # pair base rates
        def base_full(pr, c=full):
            m = c.P == c.skeys[subject_key(ds.subjects[pr.subject])]
            return c.ysum[m].sum() / c.n[m].sum()

        def base_cf(pr, c=cf):
            si = c.skeys.get(subject_key(ds.subjects[pr.subject]))
            m = c.P == si
            return (c.ysum[m].sum() + 1) / (c.n[m].sum() + 2)
        st = res["static"]
        st.setdefault("base_full", []).extend(pair_brier(np.full(len(pr.evl), base_full(pr)), pr.evl_ys) for pr in prs)
        st.setdefault("base_cf", []).extend(pair_brier(np.full(len(pr.evl), base_cf(pr)), pr.evl_ys) for pr in prs)
        specs = [("rasch", {}), ("2pl", {}), ("4pl", {}), ("lr2", {"K": 2, "lam": 1.0}), ("lr3", {"K": 3, "lam": 1.0}),
                 ("lr2_l4", {"K": 2, "lam": 4.0}), ("lr3_l4", {"K": 3, "lam": 4.0})]
        fits = {}
        if a.source == "cf":
            specs = [("rasch", {})]
        for tag, cells, fb in (("full", full, base_full), ("cf", cf, base_cf)):
            if a.source == "cf" and tag == "full":
                continue
            for kind, kw in specs:
                k0 = "lr" if kind.startswith("lr") else kind
                tt = time.time()
                f = fit_model(cells, k0, **kw)
                fits[(tag, kind)] = f
                st.setdefault(f"{kind}_{tag}", []).extend(static_eval(prs, cells, f, fb))
                if kind == "4pl":
                    res.setdefault("asym", {}).setdefault(name, {})[tag] = f["asym"].tolist()
                print(f"  {name} {tag} {kind} {time.time() - tt:.1f}s", flush=True)
        # oracle values from the in-sample Rasch (all subjects, all items)
        oc = full if a.source == "full" else cf
        fr = fits[(a.source, "rasch")]
        theta, dd = fr["theta"], fr["d"]
        cmean = dd.mean()
        dd_c, theta_c = dd - cmean, theta - cmean
        beta = theta_c.mean()
        item_dicts = [ds.items[oc.items_ds[j]] for j in range(oc.nI)]
        yh = pred.item_priors(item_dicts)
        my = float(yh.mean())
        oracle_tab[anon] = {"beta": float(beta + my),
                            "delta": {sk: float(theta_c[k] - beta) for sk, k in oc.skeys.items()},
                            "d": {ik: float(dd_c[k] + my) for ik, k in oc.ikeys.items()}}
        res["oracle_vals"][name] = {"beta_true": float(beta), "mean_yh": my, "beta_prime": float(beta + my),
                                    "sd_delta": float(np.std(theta_c - beta)), "sd_d": float(dd_c.std()),
                                    "spearman_yh_d": float(__import__("scipy.stats").stats.spearmanr(yh, dd_c)[0])}
        if a.source == "cf":
            continue
        # benchmark characteristics
        mbr = ds.resp_b == b
        items_b = item_dicts
        mcq = np.mean([bool(MCQ.search(it["item_content"][:6000] + "\n" + it["item_features"])) for it in items_b])
        noise_num, noise_den = 0.0, 0
        for pr in prs:
            for yv in pr.evl_ys:
                noise_num += float(((yv - yv.mean()) ** 2).sum())
                noise_den += len(yv)
        pair_floor = []
        for pr in prs:
            pair_floor.append(sum(float(((yv - yv.mean()) ** 2).sum()) for yv in pr.evl_ys) / sum(len(yv) for yv in pr.evl_ys))
        res["bench"][name] = {"n_items": full.nI, "n_subjects": full.nS, "n_resp": int(mbr.sum()),
                              "trials_per_cell": float(full.n.mean()), "frac_multi_trial": float((full.n > 1).mean()),
                              "base_rate": float(ds.resp_y[mbr].mean()), "mcq_frac": float(mcq),
                              "eval_trials_per_item": float(np.mean([len(y) for pr in prs for y in pr.evl_ys])),
                              "trial_floor": noise_num / noise_den, "pair_floor": pair_floor,
                              "n_acq_mean": float(np.mean([len(pr.acq) for pr in prs])), "n_pairs": len(prs),
                              "kv_keys": sorted({k.split("=")[0].strip() for it in items_b[:200] for k in it["item_features"].split(";") if "=" in k})[:30],
                              "sample": items_b[0]["item_content"][:300]}
        # within-benchmark item transfer: features -> in-sample centred difficulty, random half split
        from scipy.stats import spearmanr
        from sklearn.linear_model import RidgeCV
        tr = {}
        rng = np.random.default_rng(0)
        for fset in (("text", "length", "kv"), ("text",), ("kv",), ("length",), ("text", "length", "kv", "inter")):
            X = ItemFeaturizer(fset, 1024, 4000).transform(item_dicts)
            r2s, sps = [], []
            for rep in range(5):
                perm = rng.permutation(full.nI)
                h = full.nI // 2
                A, B = perm[:h], perm[h:]
                if X[A].std(0).sum() == 0:
                    r2s.append(0.0); sps.append(0.0); continue
                m = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(X[A], dd_c[A])
                pB = m.predict(X[B])
                r2s.append(1 - np.mean((dd_c[B] - pB) ** 2) / np.var(dd_c[B]))
                sps.append(spearmanr(pB, dd_c[B])[0] if np.std(pB) > 0 else 0.0)
            tr["+".join(fset)] = (float(np.mean(r2s)), float(np.nanmean(sps)))
        tr["prior_yh_spearman"] = res["oracle_vals"][name]["spearman_yh_d"]
        # duplicates: fraction of items sharing identical content with another item
        cont = [it["item_content"] for it in item_dicts]
        from collections import Counter
        cc = Counter(cont)
        tr["dup_content_frac"] = float(np.mean([cc[c] > 1 for c in cont]))
        res["items"][name] = tr
        print(f"  {name}: static done {time.time() - t0:.0f}s items {tr}", flush=True)
    with open(OUT / f"fold{a.fold}{'' if a.source == 'full' else '_cf'}.pkl", "wb") as f:
        pickle.dump(res, f)
    if a.skip_replay:
        return
    variants = {"champion": {}, "o_beta": {"oracle_beta": True}, "o_delta": {"oracle_delta": True},
                "o_d": {"oracle_d": True}, "o_beta_delta": {"oracle_beta": True, "oracle_delta": True},
                "o_beta_d": {"oracle_beta": True, "oracle_d": True}, "o_delta_d": {"oracle_delta": True, "oracle_d": True},
                "o_all": {"oracle_beta": True, "oracle_delta": True, "oracle_d": True}}
    res["replays"] = {}
    for vname, flags in variants.items():
        o = OracleIRT.__new__(OracleIRT)
        o.__dict__.update(pred.__dict__)
        o.oracle_tab = oracle_tab
        for k, v in flags.items():
            setattr(o, k, v)
        o.reset_runtime()
        prs = copy.deepcopy(pairs0)
        tt = time.time()
        rows, _ = replay(o, build_policy(cfg["acquisition"]), ds, prs, "global", True)
        for r, pr in zip(rows, prs):
            r["probs"] = {n: pr.probs[n].astype(np.float32) for n in BUDGETS}
            if vname == "champion":
                r["ys"] = pr.evl_ys
                r["n_acq_calls"] = pr.cursor
        res["replays"][vname] = rows
        print(f"  replay {vname}: ALC {np.mean([r['alc'] for r in rows]):.4f} {time.time() - tt:.0f}s", flush=True)
        with open(OUT / f"fold{a.fold}{'' if a.source == 'full' else '_cf'}.pkl", "wb") as f:
            pickle.dump(res, f)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
