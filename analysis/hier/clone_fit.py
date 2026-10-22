"""Clone cached IRT fits into IRTH cache entries (IRTH.fit is IRT.fit; only the class differs)."""
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from paiec.config import load_config
from paiec.fitcache import DEFAULT_DIR, cache_key
from paiec.predictors import build_predictor
from paiec.predictors.irt_h import IRTH

cfg = load_config(sys.argv[1], [])
d = cfg["data"]
rev = (Path(d["root"]) / "REVISION").read_text().strip()
benches = list(d["benchmarks"])
root = Path(os.environ.get("PAIEC_FIT_CACHE", DEFAULT_DIR))
for group in cfg["cv"]["test_benchmarks"]:
    train_b = [b for b in benches if b not in group]
    src = build_predictor(cfg["predictor"])
    dst = build_predictor({**cfg["predictor"], "name": "irt_h"})
    ps = root / f"IRT_{cache_key(src, train_b, d, rev)}.pkl"
    pd = root / f"IRTH_{cache_key(dst, train_b, d, rev)}.pkl"
    print(ps.name, ps.exists(), "->", pd.name, pd.exists())
    if ps.exists() and not pd.exists():
        with open(ps, "rb") as f:
            obj = pickle.load(f)
        obj.__class__ = IRTH
        tmp = pd.with_suffix(".tmpclone")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, pd)
