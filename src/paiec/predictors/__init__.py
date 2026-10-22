from .base import Predictor
from .irt import IRT
from .mirt import MIRT
from .simple import BetaShrink, Constant, EmpiricalMean

PREDICTORS = {cls.name: cls for cls in (Constant, EmpiricalMean, BetaShrink, IRT, MIRT)}


def build_predictor(cfg: dict) -> Predictor:
    cfg = dict(cfg)
    return PREDICTORS[cfg.pop("name")](**cfg)
