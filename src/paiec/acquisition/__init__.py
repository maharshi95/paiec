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
        return bool(self.decide(input, prediction, context, rem / items))

    def decide(self, input, p, context, rate) -> bool:
        raise NotImplementedError


class AcceptAll(Policy):
    """Take the first 31 candidates. Uniform subset (order is random); fills budgets earliest."""
    name = "accept_all"

    def decide(self, input, p, context, rate):
        return True


class OfficialRandom(Policy):
    """Platform default: deterministic hash-uniform u < labels_remaining / items_remaining."""
    name = "official_random"

    def decide(self, input, p, context, rate):
        key = json.dumps([context, input], sort_keys=True, separators=(",", ":"))
        u = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2 ** 64
        return u < rate


class Uncertainty(Policy):
    """Accept when |p - 0.5| <= radius (the paiec_baseline BLE rule, plus fill guarantee)."""
    name = "uncertainty"
    radius = 0.15

    def decide(self, input, p, context, rate):
        return abs(p - 0.5) <= self.radius


class Information(Policy):
    """Accept when Fisher information 4p(1-p) >= (1 - rate)^gamma.

    rate = labels_remaining / items_remaining: when skip room is plentiful (rate small) only
    near-0.5 items pass; as room runs out the bar drops to 0. gamma=0 -> accept_all.
    """
    name = "information"
    gamma = 1.0

    def decide(self, input, p, context, rate):
        return 4 * p * (1 - p) >= (1 - rate) ** self.gamma


POLICIES = {c.name: c for c in (AcceptAll, OfficialRandom, Uncertainty, Information)}


def build_policy(cfg: dict) -> Policy:
    cfg = dict(cfg)
    return POLICIES[cfg.pop("name")](**cfg)
