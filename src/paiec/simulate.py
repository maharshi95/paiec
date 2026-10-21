"""Local simulation of the hosted evaluator (approximate; the real splitter is private).

For each (subject, benchmark) pair with >=80 distinct items: split items 50/50 into
acquisition / evaluation pools; for each budget n in BUDGETS reveal n labels from the
acquisition pool (nested across budgets, chosen by acquisition_function or randomly)
and score Brier on the evaluation pool. Aggregate by averaging over pairs, then ALC.
"""
import random
from .metrics import BUDGETS, brier, brier_alc


def split_pools(item_ids, seed=0):
    ids = sorted(item_ids)
    random.Random(seed).shuffle(ids)
    h = len(ids) // 2
    return ids[:h], ids[h:]


def evaluate_pair(predict, rows, acquisition=None, seed=0, max_labels=31):
    """rows: list of dicts with keys subject, item, item_id, label (one per trial)."""
    by_item = {}
    for r in rows:
        by_item.setdefault(r["item_id"], []).append(r)
    acq_ids, eval_ids = split_pools(by_item, seed)
    # choose acquisition order (random default; acquisition fn is called sequentially in the real
    # platform with the running state -- TODO: mirror its exact protocol once starting kit is read)
    order = list(acq_ids)
    random.Random(seed + 1).shuffle(order)
    labeled_full = [([r["subject"], r["item"]], r["label"]) for i in order for r in by_item[i][:1]]
    out = {}
    for n in BUDGETS:
        labeled = labeled_full[:n]
        ps, ys = [], []
        for i in eval_ids:
            for r in by_item[i]:
                ps.append(predict([r["subject"], r["item"]], labeled))
                ys.append(r["label"])
        out[n] = brier(ps, ys)
    return out, brier_alc(out)
