from .base import Predictor
from .irt import IRT
from .mirt import MIRT
from .simple import BetaShrink, Constant, EmpiricalMean

PREDICTORS = {cls.name: cls for cls in (Constant, EmpiricalMean, BetaShrink, IRT, MIRT)}


def build_predictor(cfg: dict) -> Predictor:
    """Drop parameters that belong to *other* predictors (so a config can switch `name` over a base
    that sets e.g. `item_target`); unknown keys still raise."""
    cfg = dict(cfg)
    cls = PREDICTORS[cfg.pop("name")]
    others = {k for c in PREDICTORS.values() if c is not cls for c_ in c.__mro__ for k in vars(c_)
              if not k.startswith("_")}
    return cls(**{k: v for k, v in cfg.items() if hasattr(cls, k) or k not in others})
