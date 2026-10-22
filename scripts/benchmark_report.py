"""Per-benchmark usefulness report for choosing training / internal-eval sets.

Reads each downloaded benchmark once (responses + subjects; items only for size stats) and records:
  binarisation rule and binary base rate, #subjects, #items, #responses, trials,
  eval_pairs     subject-benchmark pairs with >= 80 distinct items (the platform's eligibility rule)
  id_overlap     share of subjects whose model identity also appears in another benchmark
                 (needed for cross-benchmark subject priors)
  label_sd       spread of per-subject accuracy (is there subject signal to learn?)
  item_sd        spread of per-item accuracy (is there item signal to learn?)
  median_chars   item text length; modality / domain from the catalog
and assigns a tier:
  A  test-like: binarisable, >= 3 eligible pairs  -> internal eval + train
  B  train-only: binarisable, >= 2 subjects, but < 3 eligible pairs
  C  weak: binarisable but 1 subject (item easiness for one model only)
  X  excluded: not binarisable (unordered/unbounded) or withheld

    python scripts/benchmark_report.py --root /fs/clip-scratch/mgor/measurement-db --workers 8
"""
import argparse
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paiec.data.dataset import _binarize, parse_scale
from paiec.features.subject import subject_identity

BINARIZE = {"mode": "threshold", "threshold": 0.5}


def one(args):
    root, slug = args
    t = Path(root) / slug / "formatted_tables"
    row = {"benchmark": slug}
    try:
        cat = pd.read_parquet(t / "benchmarks.parquet").iloc[0]
        row.update(release=cat.get("release"), response_type=cat.get("response_type"),
                   modality=",".join(map(str, list(cat.get("modality") if cat.get("modality") is not None else []))),
                   domain=",".join(map(str, list(cat.get("domain") if cat.get("domain") is not None else []))))
        resp = pd.read_parquet(t / "responses.parquet", columns=["subject_id", "item_id", "trial", "response"])
        y, keep, rule = _binarize(slug, resp.response.values, BINARIZE, parse_scale(cat.get("response_scale")))
        row["rule"] = rule
        row["n_responses"] = len(resp)
        row["n_subjects"] = resp.subject_id.nunique()
        row["n_items"] = resp.item_id.nunique()
        row["max_trial"] = int(resp.trial.max()) if len(resp) else 0
        if keep.any():
            r = resp[keep].assign(y=y[keep])
            row["base_rate"] = float(r.y.mean())
            per = r.groupby("subject_id").item_id.nunique()
            row["eval_pairs"] = int((per >= 80).sum())
            sacc = r.groupby("subject_id").y.mean()
            iacc = r.groupby("item_id").y.mean()
            row["label_sd"] = float(sacc.std()) if len(sacc) > 1 else 0.0
            row["item_sd"] = float(iacc.std()) if len(iacc) > 1 else 0.0
        subs = pd.read_parquet(t / "subjects.parquet")
        row["identities"] = sorted({subject_identity({k: ("" if pd.isna(v) else str(v)) for k, v in s.items()})
                                    for s in subs.to_dict("records")})
        lens = []   # stream: some items tables are several GB
        for batch in pq.ParquetFile(t / "items.parquet").iter_batches(batch_size=20000, columns=["content"]):
            lens.append(pc.utf8_length(pc.fill_null(batch.column(0), "")).to_numpy(zero_copy_only=False))
        row["median_chars"] = int(np.median(np.concatenate(lens))) if lens else 0
    except Exception as e:  # noqa: BLE001 - one bad benchmark must not kill the report
        row["error"] = repr(e)[:200]
    return row


def tier(r):
    if r.get("release") != "public" or r.get("rule") in (None, "excluded") or pd.isna(r.get("base_rate")):
        return "X"
    if r.get("eval_pairs", 0) >= 3:
        return "A"
    return "B" if r.get("n_subjects", 0) >= 2 else "C"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/fs/clip-scratch/mgor/measurement-db")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    slugs = sorted(p.name for p in Path(a.root).iterdir() if (p / "formatted_tables" / "responses.parquet").exists())
    size = {s: (Path(a.root) / s / "formatted_tables" / "responses.parquet").stat().st_size for s in slugs}
    small = [s for s in slugs if size[s] < 2e8]
    big = [s for s in slugs if size[s] >= 2e8]   # run alone to bound memory
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, [(a.root, s) for s in small], chunksize=1))
    rows += [one((a.root, s)) for s in big]
    owners = defaultdict(set)
    for r in rows:
        for i in r.get("identities", []):
            owners[i].add(r["benchmark"])
    for r in rows:
        ids = r.pop("identities", [])
        r["id_overlap"] = float(np.mean([len(owners[i]) > 1 for i in ids])) if ids else 0.0
        r["tier"] = tier(r)
    df = pd.DataFrame(rows)
    out = Path(a.root) / "benchmark_report.csv"
    df.to_csv(out, index=False)
    print(df.tier.value_counts().to_dict(), "->", out)
    print(df.groupby("tier")[["n_subjects", "eval_pairs", "n_responses"]].sum())


if __name__ == "__main__":
    main()
