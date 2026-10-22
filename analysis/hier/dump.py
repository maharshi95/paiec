"""Run a config's dev replay and dump per-(budget, pair, eval item) score components for offline
calibration analysis.

    python analysis/hier/dump.py --config configs/research/champion.yaml --set 'cv.seeds=[0]' --out X.npz
Rows: fold, bench, subj, seed, budget, m (logit mean), v (summed var), comps (bm,bv,dm,dv,im,iv),
      ysum, ntr (trials), nlab_item (global labels on that item), nlab_pair.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from paiec.acquisition import build_policy
from paiec.config import load_config
from paiec.data.dataset import load_dataset
from paiec.features.subject import subject_key
from paiec.fitcache import fit_cached
from paiec.metrics import BUDGETS
from paiec.predictors import build_predictor
from paiec.predictors.base import item_key
from paiec.sim.replay import build_pairs, replay


def comps(pred, state, inputs):
    vs = pred.var_scale
    yh = pred.item_priors([i for _, i in inputs])
    out = np.empty((len(inputs), 6))
    fits = state["fits"]
    for k, (s, i) in enumerate(inputs):
        st = fits.get(i.get("benchmark_id", ""))
        bm, bv = st["beta"] if st else (pred.mu_beta, pred.var_beta * vs)
        dm, dv = st["delta"].get(subject_key(s), (None, None)) if st else (None, None)
        if dm is None:
            dm, dv = pred.subject_prior(s), pred.var_u * vs
        im, iv = st["d"].get(item_key(i), (None, None)) if st else (None, None)
        if im is None:
            im, iv = yh[k], pred.var_d * vs
        out[k] = bm, bv, dm, dv, im, iv
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    d = cfg["data"]
    ds = load_dataset(d["root"], d["benchmarks"], d["binarize"], d["cache_dir"], max_responses=d.get("max_responses"),
                      max_item_chars=d.get("max_item_chars"), prefer_seen_from=d.get("prefer_seen_from"))
    benches = [b for b in (d["benchmarks"] or ds.benchmarks()) if b in ds.benchmarks()]
    cv = cfg["cv"]
    rec = {k: [] for k in ("fold", "bench", "subj", "seed", "budget", "comps", "ysum", "ntr", "nlab_item", "nlab_pair")}
    for f, group in enumerate(cv["test_benchmarks"]):
        group = [b for b in group if b in benches]
        train_b = [b for b in benches if b not in group]
        pred = fit_cached(build_predictor(cfg["predictor"]), ds.restrict(train_b), train_b, cfg["data"], ds.revision)
        for seed in cv["seeds"]:
            pairs = build_pairs(ds, group, cv["split_seed"], seed, cv["min_items"], cv["max_subjects_per_benchmark"],
                                cv["max_eval_items_per_pair"], None if not cv.get("eval_seen_from") else _allowed(ds, cv))
            pred.reset_runtime()
            calls = [0]
            orig = pred.predict_batch

            def pb(inputs, labeled=None, pairs=pairs, seed=seed, f=f):
                k = calls[0]
                calls[0] += 1
                n, pr = BUDGETS[k // len(pairs)], pairs[k % len(pairs)]
                st = pred._state(labeled)
                c = comps(pred, st, inputs)
                cnt = {}
                for (s_, i_), _ in labeled:
                    kk = item_key(i_)
                    cnt[kk] = cnt.get(kk, 0) + 1
                rec["fold"].append(np.full(len(inputs), f))
                rec["bench"].append(np.full(len(inputs), pr.bench))
                rec["subj"].append(np.full(len(inputs), pr.subject))
                rec["seed"].append(np.full(len(inputs), seed))
                rec["budget"].append(np.full(len(inputs), n))
                rec["comps"].append(c)
                rec["ysum"].append(np.array([y.sum() for y in pr.evl_ys]))
                rec["ntr"].append(np.array([len(y) for y in pr.evl_ys]))
                rec["nlab_item"].append(np.array([cnt.get(item_key(i_), 0) for _, i_ in inputs]))
                rec["nlab_pair"].append(np.full(len(inputs), len(pr.labeled)))
                return orig(inputs, labeled)
            pred.predict_batch = pb
            res, _ = replay(pred, build_policy(cfg["acquisition"]), ds, pairs, cfg["sim"]["evidence_scope"],
                            cfg["sim"]["fresh_each_checkpoint"])
            del pred.predict_batch
            print(f"fold {f} seed {seed}: ALC {np.mean([r['alc'] for r in res]):.4f}", flush=True)
    out = {k: np.concatenate(v) for k, v in rec.items()}
    out["bench_names"] = np.array(ds.bench_names)
    np.savez_compressed(a.out, **out)


def _allowed(ds, cv):
    from paiec.features.subject import subject_identity
    ref = np.isin(ds.resp_b, [ds.bench_names.index(b) for b in cv["eval_seen_from"] if b in ds.bench_names])
    seen = {subject_identity(ds.subjects[s]) for s in np.unique(ds.resp_s[ref])}
    return {k for k, s in enumerate(ds.subjects) if subject_identity(s) in seen}


if __name__ == "__main__":
    main()
