"""Q6: sensitivity of the champion to replay mechanics the hosted coordinator may do differently.

Variants of the acquisition phase (evaluation unchanged):
  default      pairs advanced one after another in sorted (subject, benchmark) order, evidence grows live
  reversed     same, reversed pair order
  shuffled     pair order reshuffled every budget round
  roundrobin   one acquisition *label* at a time across pairs (interleaved), evidence grows live
  frozen       acquisition predictions see only the evidence frozen at the previous checkpoint
Writes out/order.pkl.
"""
import copy
import pickle
import random
import time

import numpy as np
from common import OUT, allowed_subjects, fold_pairs, load

from paiec.acquisition import build_policy
from paiec.fitcache import fit_cached
from paiec.metrics import BUDGETS
from paiec.predictors import build_predictor
from paiec.sim.replay import score_pairs


def step(pred, policy, ds, pr, ev_view, evidence, n):
    """Advance pair pr by candidates until it has one more label (or n reached). Returns True if a label was added."""
    subj = ds.subjects[pr.subject]
    while len(pr.labeled) < n and pr.cursor < len(pr.acq):
        i = pr.acq[pr.cursor]
        inp = [subj, ds.items[i]]
        ctx = {"subject_id": ds.subject_ids[pr.subject], "benchmark_id": ds.items[i]["benchmark_id"],
               "labels_acquired": len(pr.labeled), "labels_remaining": 31 - len(pr.labeled), "max_labels": 31,
               "items_remaining": len(pr.acq) - pr.cursor}
        ev = ev_view()
        p = pred.predict(inp, ev)
        take = policy(inp, p, ev, ctx)
        pr.cursor += 1
        if take:
            e = [inp, int(pr.outcomes[i])]
            pr.labeled.append(e)
            evidence.append(e)
            return True
    return False


def run(pred, policy, ds, pairs, mode, seed=0):
    evidence = []
    rng = random.Random(seed)
    for n in BUDGETS:
        frozen = list(evidence)
        order = list(pairs)
        if mode == "reversed":
            order = order[::-1]
        elif mode == "shuffled":
            rng.shuffle(order)
        view = (lambda: frozen) if mode == "frozen" else (lambda: evidence)
        if mode == "roundrobin":
            active = True
            while active:
                active = False
                for pr in order:
                    if step(pred, policy, ds, pr, view, evidence, n):
                        active = True
        else:
            for pr in order:
                while step(pred, policy, ds, pr, view, evidence, n):
                    pass
        pred.reset_runtime()
        for pr in pairs:
            pr.probs[n] = pred.predict_batch([(ds.subjects[pr.subject], ds.items[i]) for i in pr.evl], list(evidence))
    return score_pairs(ds, pairs)


def main():
    t0 = time.time()
    cfg, ds = load()
    benches = [b for b in cfg["data"]["benchmarks"] if b in ds.bench_names and b in ds.benchmarks()]
    allowed = allowed_subjects(cfg, ds)
    out = {}
    for k, group in enumerate(cfg["cv"]["test_benchmarks"]):
        group = [b for b in group if b in benches]
        train_b = [b for b in benches if b not in group]
        pred = fit_cached(build_predictor(cfg["predictor"]), ds.restrict(train_b), train_b, cfg["data"], ds.revision)
        pairs0 = fold_pairs(cfg, ds, group, 0, allowed)
        for mode in ("default", "reversed", "shuffled", "roundrobin", "frozen"):
            pred.reset_runtime()
            rows = run(pred, build_policy(cfg["acquisition"]), ds, copy.deepcopy(pairs0), mode)
            out.setdefault(mode, []).extend(rows)
            print(f"fold {k} {mode:>10}: ALC {np.mean([r['alc'] for r in rows]):.4f} {time.time() - t0:.0f}s", flush=True)
        with open(OUT / "order.pkl", "wb") as f:
            pickle.dump(out, f)
    base = np.array([r["alc"] for r in out["default"]])
    for mode, rows in out.items():
        a = np.array([r["alc"] for r in rows])
        d = a - base
        print(f"{mode:>10}: ALC {a.mean():.4f}  Δ {d.mean():+.4f} ± {1.96 * d.std() / np.sqrt(len(d)):.4f}  "
              + " ".join(f"B{n} {np.mean([r[f'B{n}'] for r in rows]):.4f}" for n in BUDGETS))


if __name__ == "__main__":
    main()
