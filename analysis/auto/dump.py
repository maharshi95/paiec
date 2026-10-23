"""Score several component predictors on the *same* replay trajectory (one driver predictor + policy).

    PYTHONPATH=src python analysis/auto/dump.py --fold 0 [--seeds 0 1 2] [--out analysis/auto/out/dump]

The driver (first component) drives acquisition exactly as in a normal run, so its Brier equals its run's.
At each budget checkpoint every component scores every eval item against the same global `labeled`.
Output npz per (fold, seed): P [n_comp, N] probabilities, s/k (sum/count of recorded trials per eval
item), pair, budget, n_own, n_bench (labels on the benchmark), n_item (labels on this item from others),
bench; plus comps (names) and the pair table. Pair Brier = sum_i k p^2 - 2 p s + s over items / sum k.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from paiec.acquisition import build_policy  # noqa: E402
from paiec.config import load_config  # noqa: E402
from paiec.fitcache import fit_cached  # noqa: E402
from paiec.metrics import BUDGETS  # noqa: E402
from paiec.predictors import build_predictor  # noqa: E402
from paiec.predictors.base import item_key  # noqa: E402
from paiec.sim.replay import build_pairs  # noqa: E402

BASE = dict(item_target="absolute", beta_var_mult=4.0)
A3 = dict(BASE, rp_frac=0.3, rp_dim=64, kv_frac=0.2, dup_frac=0.2, eb_nu=10)
COMPONENTS = {
    "A3": dict(A3, name="irt_item"),
    "champ": dict(BASE, name="irt"),
    "A3vi": dict(A3, name="irt_item", vi=True),
    "A3lr": dict(A3, name="irt_item_lr", lr_wvar_mult=2.0),
    "hvi": dict(BASE, name="irt_h", vi=True, nu_u=10),
    "lr": dict(BASE, name="irt_lr"),
    "lex": dict(name="ref_lex"),
    "A3lrvi05": dict(A3, name="irt_item_lr", lr_wvar_mult=2.0, vi=True, vi_frac=0.5),
    "A3vi05": dict(A3, name="irt_item", vi=True, vi_frac=0.5),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--comps", nargs="+", default=list(COMPONENTS))
    ap.add_argument("--out", default="analysis/auto/out/dump")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = load_config("configs/research/champion.yaml", [])
    from paiec.data.dataset import load_dataset
    d = cfg["data"]
    ds = load_dataset(d["root"], d["benchmarks"], d["binarize"], d["cache_dir"], max_responses=d.get("max_responses"),
                      max_item_chars=d.get("max_item_chars"), prefer_seen_from=d.get("prefer_seen_from"))
    benches = [b for b in d["benchmarks"] if b in ds.benchmarks()]
    cv = cfg["cv"]
    group = [b for b in cv["test_benchmarks"][a.fold] if b in benches]
    train_b = [b for b in benches if b not in group]
    train = ds.restrict(train_b)
    from paiec.features.subject import subject_identity
    ref = np.isin(ds.resp_b, [ds.bench_names.index(b) for b in cv["eval_seen_from"] if b in ds.bench_names])
    seen = {subject_identity(ds.subjects[s]) for s in np.unique(ds.resp_s[ref])}
    allowed = {k for k, s in enumerate(ds.subjects) if subject_identity(s) in seen}
    preds = [fit_cached(build_predictor(dict(COMPONENTS[c])), train, train_b, d, ds.revision) for c in a.comps]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        pairs = build_pairs(ds, group, cv["split_seed"], seed, cv["min_items"], cv["max_subjects_per_benchmark"],
                            cv["max_eval_items_per_pair"], allowed)
        for p in preds:
            p.reset_runtime()
        driver, policy = preds[0], build_policy(cfg["acquisition"])
        if hasattr(policy, "bind"):
            policy.bind(driver)
        evidence = []
        cols = {k: [] for k in ("pair", "budget", "n_own", "n_bench", "n_item", "s", "k")}
        P = [[] for _ in preds]
        for n in BUDGETS:
            for pr in pairs:
                subj = ds.subjects[pr.subject]
                while len(pr.labeled) < n and pr.cursor < len(pr.acq):
                    i = pr.acq[pr.cursor]
                    inp = [subj, ds.items[i]]
                    ctx = {"subject_id": ds.subject_ids[pr.subject], "benchmark_id": ds.items[i]["benchmark_id"],
                           "labels_acquired": len(pr.labeled), "labels_remaining": 31 - len(pr.labeled),
                           "max_labels": 31, "items_remaining": len(pr.acq) - pr.cursor}
                    if policy(inp, driver.predict(inp, evidence), evidence, ctx):
                        e = [inp, int(pr.outcomes[i])]
                        pr.labeled.append(e)
                        evidence.append(e)
                    pr.cursor += 1
            for p in preds:
                p.reset_runtime()
            nb, ni = {}, {}
            for (s_, it), _ in evidence:
                nb[it["benchmark_id"]] = nb.get(it["benchmark_id"], 0) + 1
                ni[item_key(it)] = ni.get(item_key(it), 0) + 1
            for k, pr in enumerate(pairs):
                subj = ds.subjects[pr.subject]
                inputs = [(subj, ds.items[i]) for i in pr.evl]
                for c, p in enumerate(preds):
                    P[c].append(p.predict_batch(inputs, list(evidence)))
                bid = ds.bench_names[pr.bench]
                m = len(inputs)
                cols["pair"].append(np.full(m, k))
                cols["budget"].append(np.full(m, n))
                cols["n_own"].append(np.full(m, len(pr.labeled)))
                cols["n_bench"].append(np.full(m, nb.get(bid, 0)))
                cols["n_item"].append(np.array([ni.get(item_key(ds.items[i]), 0) for i in pr.evl]))
                cols["s"].append(np.array([y.sum() for y in pr.evl_ys]))
                cols["k"].append(np.array([len(y) for y in pr.evl_ys]))
        np.savez_compressed(out / f"f{a.fold}_s{seed}{a.tag}.npz", comps=np.array(a.comps),
                            P=np.stack([np.concatenate(x) for x in P]),
                            **{k: np.concatenate(v) for k, v in cols.items()},
                            pair_subject=np.array([ds.subject_ids[pr.subject] for pr in pairs]),
                            pair_bench=np.array([ds.bench_names[pr.bench] for pr in pairs]), fold=a.fold, seed=seed)
        print(f"fold {a.fold} seed {seed}: {len(pairs)} pairs saved", flush=True)


if __name__ == "__main__":
    main()
