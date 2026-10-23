"""Check the ref_lex port against the original submission's `predict` on real pairs.

    python scripts/check_ref_lex.py /fs/clip-scratch/mgor/collab_zip/c14_lex_cosine_p4_a1
"""
import importlib.util
import sys

import numpy as np

sys.path.insert(0, "src")
from paiec.config import load_config  # noqa: E402
from paiec.data.dataset import load_dataset  # noqa: E402
from paiec.predictors import build_predictor  # noqa: E402

zdir = sys.argv[1]
sys.path.insert(0, zdir)   # the zip imports its sibling `contract`
spec = importlib.util.spec_from_file_location("collab_model", f"{zdir}/model.py")
orig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orig)
cfg = orig.CONFIG
mine = build_predictor({"name": "ref_lex", "alpha": cfg["alpha"], "prior": cfg["prior"], "power": cfg["power"],
                        "floor": cfg["weight_floor"], "metric": cfg["metric"], "blend": cfg["blend"]})

d = load_config("configs/research/dev.yaml")["data"]
ds = load_dataset(d["root"], d["benchmarks"], d["binarize"], d["cache_dir"], max_responses=d.get("max_responses"),
                  max_item_chars=d.get("max_item_chars"), prefer_seen_from=d.get("prefer_seen_from"))
rng = np.random.default_rng(0)
worst, n_cmp = 0.0, 0
for b in rng.choice(np.unique(ds.resp_b), 8, replace=False):
    m = np.nonzero(ds.resp_b == b)[0]
    for s in rng.choice(np.unique(ds.resp_s[m]), 2, replace=False):
        rows = m[ds.resp_s[m] == s]
        if len(rows) < 45:
            continue
        rows = rng.permutation(rows)
        subj = ds.subjects[s]
        for n_lab in (0, 3, 15, 31):
            lab = [([subj, ds.items[ds.resp_i[k]]], float(ds.resp_y[k])) for k in rows[:n_lab]]
            lab_t = [((subj, ds.items[ds.resp_i[k]]), float(ds.resp_y[k])) for k in rows[:n_lab]]
            for k in rows[31:36]:
                it = ds.items[ds.resp_i[k]]
                a = orig.predict([subj, it], lab)
                mine.reset_runtime()
                p = float(mine.predict_batch([(subj, it)], lab_t)[0])
                worst, n_cmp = max(worst, abs(a - p)), n_cmp + 1
print(f"{n_cmp} comparisons, max |orig - port| = {worst:.2e}")
assert worst < 1e-9
