from .base import Predictor
from .irt import IRT
from .simple import BetaShrink, Constant, EmpiricalMean

PREDICTORS = {cls.name: cls for cls in (Constant, EmpiricalMean, BetaShrink, IRT)}


def build_predictor(cfg: dict) -> Predictor:
    cfg = dict(cfg)
    return PREDICTORS[cfg.pop("name")](**cfg)
