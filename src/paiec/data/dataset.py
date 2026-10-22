"""measurement-db tables -> Dataset in the competition's visible-input format.

Subjects and items are stored as the exact dicts `predict` receives (all strings,
missing -> ""), with an anonymised `benchmark_id`. True benchmark names and private
ids live only in parallel metadata lists, which predictors never see at score time.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

SUBJECT_FIELDS = ("normalized_name", "provider", "release_date", "access_date", "harness",
                  "harness_version", "reasoning_effort", "subject_features_extra")
ITEM_FIELDS = ("item_content", "item_features", "interactors", "benchmark_id")
TABLES = ("benchmarks", "subjects", "items", "responses")


def anon_benchmark_id(name: str) -> str:
    return "bm_" + hashlib.sha1(name.encode()).hexdigest()[:10]


@dataclass
class Dataset:
    subjects: list            # visible subject dicts; index = subject idx
    items: list               # visible item dicts; index = item idx
    item_bench: np.ndarray    # item idx -> benchmark idx
    bench_names: list         # private benchmark names
    subject_ids: list         # private subject ids (pair identity on the platform)
    item_ids: list            # private item ids
    resp_s: np.ndarray        # one entry per response (trial)
    resp_i: np.ndarray
    resp_y: np.ndarray        # binary, float64
    revision: str = ""

    @property
    def resp_b(self) -> np.ndarray:
        return self.item_bench[self.resp_i]

    def benchmarks(self) -> list:
        return [self.bench_names[b] for b in np.unique(self.resp_b)]

    def restrict(self, benchmarks) -> "Dataset":
        """Keep responses from these benchmarks; subject/item index spaces are unchanged."""
        keep = np.isin(self.resp_b, [self.bench_names.index(b) for b in benchmarks])
        return replace(self, resp_s=self.resp_s[keep], resp_i=self.resp_i[keep],
                       resp_y=self.resp_y[keep])

    def summary(self) -> str:
        rows = []
        for b in np.unique(self.resp_b):
            m = self.resp_b == b
            rows.append(f"{self.bench_names[b]}: {m.sum()} responses, "
                        f"{len(np.unique(self.resp_s[m]))} subjects, "
                        f"{len(np.unique(self.resp_i[m]))} items, mean {self.resp_y[m].mean():.3f}")
        return "\n".join(rows)


def _s(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(v)


def _binarize(name, values, cfg):
    """Return (binary values, keep mask). cfg: {mode: exclude|threshold, threshold, per_benchmark}."""
    vals = np.asarray(values, float)
    finite = np.isfinite(vals)
    if set(np.unique(vals[finite])) <= {0.0, 1.0}:
        return vals, finite
    rule = (cfg.get("per_benchmark") or {}).get(name, cfg.get("mode", "exclude"))
    if rule == "exclude":
        return vals, np.zeros(len(vals), bool)
    t = float(rule) if not isinstance(rule, str) else float(cfg.get("threshold", 0.5))
    return (vals >= t).astype(float), finite


def load_dataset(root="data/raw", benchmarks=None, binarize=None, cache_dir="data/processed",
                 use_cache=True) -> Dataset:
    import pandas as pd

    root = Path(root)
    binarize = binarize or {"mode": "exclude"}
    revision = (root / "REVISION").read_text().strip() if (root / "REVISION").exists() else ""
    slugs = sorted(p.name for p in root.iterdir() if (p / "formatted_tables").is_dir())
    if benchmarks:
        slugs = [s for s in slugs if s in benchmarks]
    key = hashlib.sha1(json.dumps([revision, slugs, binarize], sort_keys=True).encode()).hexdigest()[:12]
    cache = Path(cache_dir) / f"dataset_{key}.pkl"
    if use_cache and cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    subjects, subject_ids, sidx = [], [], {}
    items, item_ids, item_bench, iidx = [], [], [], {}
    rs, ri, ry, bench_names = [], [], [], []
    for slug in slugs:
        t = root / slug / "formatted_tables"
        resp = pd.read_parquet(t / "responses.parquet")
        y, keep = _binarize(slug, resp.response.values, binarize)
        if not keep.any():
            continue
        b = len(bench_names)
        bench_names.append(slug)
        anon = anon_benchmark_id(slug)
        for row in pd.read_parquet(t / "subjects.parquet").itertuples(index=False):
            if row.subject_id not in sidx:
                sidx[row.subject_id] = len(subjects)
                subjects.append({f: _s(getattr(row, f, None)) for f in SUBJECT_FIELDS})
                subject_ids.append(row.subject_id)
        for row in pd.read_parquet(t / "items.parquet", columns=["item_id", "content", "item_features"]).itertuples(index=False):
            if row.item_id not in iidx:
                iidx[row.item_id] = len(items)
                items.append({"item_content": _s(row.content), "item_features": _s(row.item_features),
                              "interactors": "", "benchmark_id": anon})
                item_ids.append(row.item_id)
                item_bench.append(b)
        resp = resp[keep]
        rs.append(resp.subject_id.map(sidx).values)
        ri.append(resp.item_id.map(iidx).values)
        ry.append(y[keep])
    ds = Dataset(subjects, items, np.array(item_bench), bench_names, subject_ids, item_ids,
                 np.concatenate(rs).astype(np.int64), np.concatenate(ri).astype(np.int64),
                 np.concatenate(ry).astype(float), revision)
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(ds, f)
    return ds
