"""Per-benchmark overlap with models seen in the six reference benchmarks.

For every benchmark in benchmark_report.csv: subjects whose canonical model identity (features/identity.py,
display_name included since this is offline selection) matches a ref6 model, and how many of those form
eligible pairs (>= 80 distinct items after binarisation). Writes <root>/ref6_overlap.csv.

    PYTHONPATH=src python analysis/subjects/ref6_overlap.py
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from paiec.data.dataset import _binarize, parse_scale
from paiec.features.identity import identity_set, model_identity

ROOT = Path("/fs/clip-scratch/mgor/measurement-db")
REF6 = ["matharena", "mmdocrag", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]


def subjects_of(b):
    s = pd.read_parquet(ROOT / b / "formatted_tables/subjects.parquet")
    return {r["subject_id"]: {k: ("" if pd.isna(v) else str(v)) for k, v in r.items()} for r in s.to_dict("records")}


def ref6_identities():
    R = set()
    for b in REF6:
        for r in subjects_of(b).values():
            R |= identity_set(r, include_display=True)
    return R


def one(args):
    b, R = args
    subs = subjects_of(b)
    seen = {sid for sid, r in subs.items() if identity_set(r, include_display=True) & R}
    vis = {sid for sid, r in subs.items() if model_identity(r) in R}
    cat = pd.read_parquet(ROOT / b / "formatted_tables/benchmarks.parquet").iloc[0]
    resp = pd.read_parquet(ROOT / b / "formatted_tables/responses.parquet", columns=["subject_id", "item_id", "response"])
    _, keep, _ = _binarize(b, resp.response.values, {"mode": "threshold", "threshold": 0.5},
                           parse_scale(cat.get("response_scale")))
    per = resp[keep].groupby("subject_id").item_id.nunique()
    elig = set(per[per >= 80].index)
    ids = {model_identity(subs[s], True) for s in seen & elig}
    return {"benchmark": b, "n_subj": len(subs), "seen_subj": len(seen), "seen_subj_visible": len(vis),
            "seen_eval_pairs": len(seen & elig), "seen_eval_identities": len(ids),
            "seen_frac": len(seen) / max(len(subs), 1)}


def main():
    rep = pd.read_csv(ROOT / "benchmark_report.csv")
    R = ref6_identities()
    with ProcessPoolExecutor(6) as ex:
        rows = list(ex.map(one, [(b, R) for b in rep.benchmark], chunksize=4))
    out = rep.merge(pd.DataFrame(rows), on="benchmark")
    out.to_csv(ROOT / "ref6_overlap.csv", index=False)
    print(f"{len(R)} ref6 identities; benchmarks with >=3 seen eval pairs: {(out.seen_eval_pairs >= 3).sum()}, "
          f">=8: {(out.seen_eval_pairs >= 8).sum()}")


if __name__ == "__main__":
    main()
