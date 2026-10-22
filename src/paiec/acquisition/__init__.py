"""Streaming acquisition policies: __call__(input, prediction, labeled, context) -> bool.

Hosted protocol (from the starting kit, via the collaborator's replay):
candidates of one pair's acquisition pool arrive in random order; each is accepted (label
revealed) or skipped forever. context = {subject_id, benchmark_id, labels_acquired,
labels_remaining (of 31), max_labels, items_remaining (incl. this candidate)}.
Every policy here is stateless and enforces the fill rule: accept when
items_remaining <= labels_remaining, reject when labels_remaining == 0.

Skip room per pair is |acquisition pool| - 31, so selective policies only matter on
larger pairs. Selective policies bias predictors that ignore item difficulty.
"""
import hashlib
import json

import numpy as np


class Policy:
    name = "base"

    def __init__(self, **params):
        for k, v in params.items():
            if not hasattr(type(self), k):
                raise ValueError(f"{type(self).__name__} has no parameter {k!r}")
            setattr(self, k, v)

    def __call__(self, input, prediction=None, labeled=None, context=None) -> bool:
        rem, items = context["labels_remaining"], context["items_remaining"]
        if rem <= 0:
            return False
        if items <= rem:
            return True
        return bool(self.decide(input, prediction, context, rem / items, labeled or []))

    def bind(self, predictor) -> "Policy":
        """Give the policy access to the fitted predictor (used by model-aware policies)."""
        self._predictor = predictor
        return self

    def decide(self, input, p, context, rate, labeled) -> bool:
        raise NotImplementedError


class AcceptAll(Policy):
    """Take the first 31 candidates. Uniform subset (order is random); fills budgets earliest."""
    name = "accept_all"

    def decide(self, input, p, context, rate, labeled):
        return True


class OfficialRandom(Policy):
    """Platform default: deterministic hash-uniform u < labels_remaining / items_remaining."""
    name = "official_random"

    def decide(self, input, p, context, rate, labeled):
        key = json.dumps([context, input], sort_keys=True, separators=(",", ":"))
        u = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2 ** 64
        return u < rate


class Uncertainty(Policy):
    """Accept when |p - 0.5| <= radius (the paiec_baseline BLE rule, plus fill guarantee)."""
    name = "uncertainty"
    radius = 0.15

    def decide(self, input, p, context, rate, labeled):
        return abs(p - 0.5) <= self.radius


class Information(Policy):
    """Accept when Fisher information 4p(1-p) >= (1 - rate)^gamma.

    rate = labels_remaining / items_remaining: when skip room is plentiful (rate small) only
    near-0.5 items pass; as room runs out the bar drops to 0. gamma=0 -> accept_all.
    """
    name = "information"
    gamma = 1.0

    def decide(self, input, p, context, rate, labeled):
        return 4 * p * (1 - p) >= (1 - rate) ** self.gamma


class Quantile(Policy):
    """Accept iff the candidate's information 4p(1-p) is in the top `rate` fraction of a reference set.

    The reference set is the same benchmark's distinct items already revealed for other subjects (global
    evidence), scored for *this* subject with the bound predictor. They come from the same benchmark's
    acquisition pools, so they stand in for the candidates still to come. Accepting the top
    labels_remaining / items_remaining fraction fills the budget at the right rate while keeping the
    most informative items. Falls back to Information(gamma) without a predictor or with < min_ref items.
    """
    name = "quantile"
    min_ref = 30
    max_ref = 400
    slack = 1.0          # >1 accepts more freely (less risk of running out of candidates)
    gamma = 1.0

    def decide(self, input, p, context, rate, labeled):
        pred = getattr(self, "_predictor", None)
        subject, item = input
        bid = item.get("benchmark_id", "")
        ref, seen = [], set()
        if pred is not None:
            for (s, i), _ in reversed(labeled):
                if i.get("benchmark_id", "") != bid:
                    continue
                k = (i.get("item_content", "")[:256], i.get("item_features", ""))
                if k not in seen:
                    seen.add(k)
                    ref.append((subject, i))
                    if len(ref) >= self.max_ref:
                        break
        info = 4 * p * (1 - p)
        if len(ref) < self.min_ref:
            return info >= (1 - rate) ** self.gamma
        pr = pred.predict_batch(ref, labeled)
        thr = np.quantile(4 * pr * (1 - pr), max(0.0, 1 - rate * self.slack))
        return info >= thr


POLICIES = {c.name: c for c in (AcceptAll, OfficialRandom, Uncertainty, Information, Quantile)}


def build_policy(cfg: dict) -> Policy:
    cfg = dict(cfg)
    return POLICIES[cfg.pop("name")](**cfg)
