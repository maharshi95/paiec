"""End-to-end check: does a different item prior change Brier ALC in the harness replay?

Registers (at runtime, without touching src/) an `irt_lookup` predictor: the IRT predictor whose item
featurizer is replaced by a lookup into a precomputed feature table (keyed by item content + features), and an
optional `item_shrink` that scales the ridge weights (prior mean pulled toward the benchmark mean).

  python e2e.py export FEATSET            # write data/analysis/item_difficulty/tables/FEATSET.npz from lobo.feature_sets
  python e2e.py run NAME [--set k=v ...]  # configs/irt.yaml + overrides; runs -> data/analysis/item_difficulty/runs/NAME
  e.g. python e2e.py run no_item_prior --set predictor.use_item_features=false
       python e2e.py run lookup_hand_bz --set predictor.name=irt_lookup --set predictor.table=hand_bz
"""
import hashlib
import sys

import numpy as np
from common import OUT, ROOT

from paiec.config import load_config
from paiec.experiment import run_experiment
from paiec.predictors import PREDICTORS
from paiec.predictors import irt as irtmod

TABLES = OUT / "tables"


def ikey(item):
    return hashlib.md5((item.get("item_features", "") + "\x00" + item.get("item_content", "")).encode()).hexdigest()


class LookupFeaturizer:
    def __init__(self, path):
        z = np.load(path)
        self.X = z["X"].astype(np.float32)
        self.index = {k: j for j, k in enumerate(z["keys"].tolist())}

    @property
    def n_features(self):
        return self.X.shape[1]

    def transform(self, items):
        out = np.zeros((len(items), self.n_features), np.float32)
        for r, it in enumerate(items):
            j = self.index.get(ikey(it))
            if j is not None:
                out[r] = self.X[j]
        return out

    def transform_one(self, item):
        return self.transform([item])[0]


class IRTLookup(irtmod.IRT):
    name = "irt_lookup"
    table = "hand_bz"        # "" keeps the stock hashed-BOW ItemFeaturizer
    item_shrink = 1.0
    var_d_mult = 1.0         # multiplies the item prior variance only

    def fit(self, train):
        if not self.table:
            super().fit(train)
            return self._post()
        orig = irtmod.ItemFeaturizer
        irtmod.ItemFeaturizer = lambda *a, **k: LookupFeaturizer(TABLES / f"{self.table}.npz")
        try:
            super().fit(train)
        finally:
            irtmod.ItemFeaturizer = orig
        return self._post()

    def _post(self):
        self.var_d *= self.var_d_mult
        if self.item_shrink != 1.0:
            # shrink the centred prediction: w -> c w, keep the training mean
            Xm = self.ifeat.X.mean(0) if hasattr(self.ifeat, "X") else np.zeros_like(self.w_i)
            self.b_i = float(self.b_i + (1 - self.item_shrink) * (Xm @ self.w_i))
            self.w_i = self.w_i * self.item_shrink
        return self



PREDICTORS[IRTLookup.name] = IRTLookup


def export(featset, raw=False):
    """FEATSET may be 'a+b' to concatenate two lobo feature sets; raw=True skips the global z-score."""
    from lobo import feature_sets, load_all
    df, feats, _ = load_all()
    S = feature_sets(df, feats)
    parts = []
    for name in featset.split("&"):
        X = S[name]()
        parts.append(X.toarray() if hasattr(X, "toarray") else np.asarray(X, float))
    X = np.hstack(parts)
    if not raw:
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)          # global z-score so ridge_item acts uniformly
    keys = np.array([ikey({"item_content": c, "item_features": f}) for c, f in zip(df.content, df.item_features)])
    TABLES.mkdir(parents=True, exist_ok=True)
    np.savez(TABLES / f"{featset}.npz", X=X.astype(np.float32), keys=keys)
    print("wrote", TABLES / f"{featset}.npz", X.shape)


def run(name, sets):
    import os
    os.chdir(ROOT)
    cfg = load_config("configs/irt.yaml", sets)
    cfg["name"] = name
    cfg["out_root"] = str(OUT / "runs")
    run_experiment(cfg)


if __name__ == "__main__":
    if sys.argv[1] == "export":
        export(sys.argv[2], raw="--raw" in sys.argv)
    else:
        args = sys.argv[3:]
        sets = [args[i + 1] for i in range(len(args)) if args[i] == "--set"]
        run(sys.argv[2], sets)
