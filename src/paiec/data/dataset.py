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


def parse_scale(raw) -> dict:
    try:
        return json.loads(raw) if isinstance(raw, str) else dict(raw or {})
    except (TypeError, ValueError):
        return {}


def _binarize(name, values, cfg, scale=None):
    """Map recorded grades to {0,1}. Returns (y, keep mask, rule used).

    cfg:
      mode          exclude (binary benchmarks only) | threshold
      threshold     cut on the scale normalised to [0, 1] by its declared bounds/values
                    (observed min/max for `mixed` or missing bounds); default 0.5
      per_benchmark {name: threshold | "exclude"} overrides
      orient        flip lower_is_better scales (incl. binary) so 1 = good outcome; default False
                    (keep recorded values, which is presumably what the platform scores)
    Unordered (nominal) and unbounded scales are excluded under threshold mode.
    """
    scale = scale or {}
    vals = np.asarray(values, float)
    finite = np.isfinite(vals)
    flip = bool(cfg.get("orient")) and scale.get("direction") == "lower_is_better"
    if set(np.unique(vals[finite])) <= {0.0, 1.0}:
        return (1 - vals if flip else vals), finite, "binary"
    rule = (cfg.get("per_benchmark") or {}).get(name, cfg.get("mode", "exclude"))
    none = (vals, np.zeros(len(vals), bool), "excluded")
    if rule == "exclude" or scale.get("direction") == "unordered" or not finite.any():
        return none
    t = float(rule) if not isinstance(rule, str) else float(cfg.get("threshold", 0.5))
    lo, hi = scale.get("min"), scale.get("max")
    if scale.get("values"):
        lo, hi = min(scale["values"]), max(scale["values"])
    if scale.get("kind") == "interval" and (lo is None or hi is None):
        return none   # unbounded: no meaningful normalisation
    if lo is None or hi is None:
        lo, hi = float(vals[finite].min()), float(vals[finite].max())
    if hi <= lo:
        return none
    z = (vals - lo) / (hi - lo)
    if scale.get("direction") == "lower_is_better":
        z = 1 - z   # graded "lower is better" scales are oriented before thresholding
    y = (z >= t).astype(float)
    return y, finite, f"threshold@{t}"


def load_dataset(root="data/raw", benchmarks=None, binarize=None, cache_dir="data/processed",
                 use_cache=True, max_responses=None, max_item_chars=None) -> Dataset:
    """max_responses: cap per benchmark by keeping a seeded random subset of *subjects* (each kept
    pair keeps all of its items). max_item_chars: truncate stored item text (memory; featurizers
    read at most a few thousand characters anyway)."""
    import pandas as pd

    root = Path(root)
    binarize = binarize or {"mode": "exclude"}
    revision = (root / "REVISION").read_text().strip() if (root / "REVISION").exists() else ""
    slugs = sorted(p.name for p in root.iterdir() if (p / "formatted_tables").is_dir())
    if benchmarks:
        slugs = [s for s in slugs if s in benchmarks]
    key = hashlib.sha1(json.dumps(["v2", revision, slugs, binarize, max_responses, max_item_chars], sort_keys=True).encode()).hexdigest()[:12]
    cache = Path(cache_dir) / f"dataset_{key}.pkl"
    if use_cache and cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    subjects, subject_ids, sidx = [], [], {}
    items, item_ids, item_bench, iidx = [], [], [], {}
    rs, ri, ry, bench_names = [], [], [], []
    rules = {}
    for slug in slugs:
        t = root / slug / "formatted_tables"
        if not all((t / f"{x}.parquet").exists() for x in TABLES):
            continue   # partially downloaded
        cat = pd.read_parquet(t / "benchmarks.parquet").iloc[0]
        if cat.get("release", "public") != "public" and not binarize.get("include_withheld"):
            continue
        resp = pd.read_parquet(t / "responses.parquet", columns=["subject_id", "item_id", "interactors", "response"])
        y, keep, rules[slug] = _binarize(slug, resp.response.values, binarize, parse_scale(cat.get("response_scale")))
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
        resp = resp[keep]
        y = y[keep]
        if max_responses and len(resp) > max_responses:
            counts = resp.subject_id.value_counts()
            order = sorted(counts.index, key=lambda x: hashlib.md5(f"{slug}|{x}".encode()).hexdigest())
            keep_s, total = set(), 0
            for sid in order:
                if total and total + counts[sid] > max_responses:
                    continue
                keep_s.add(sid)
                total += counts[sid]
            m = resp.subject_id.isin(keep_s).values
            resp, y = resp[m], y[m]
        # a visible item is (item, interactors): interactors live on responses and are part of the input
        inter = resp.interactors.fillna("").astype(str).values
        needed = set(resp.item_id.unique())
        meta = {}
        for row in pd.read_parquet(t / "items.parquet", columns=["item_id", "content", "item_features"]).itertuples(index=False):
            if row.item_id in needed and row.item_id not in meta:
                c = _s(row.content)
                meta[row.item_id] = (c[:max_item_chars] if max_item_chars else c, _s(row.item_features))
        idx = np.empty(len(resp), np.int64)
        for k, (iid, it) in enumerate(zip(resp.item_id.values, inter)):
            key = (iid, it)
            if key not in iidx:
                iidx[key] = len(items)
                content, feats = meta.get(iid, ("", ""))
                items.append({"item_content": content, "item_features": feats, "interactors": it, "benchmark_id": anon})
                item_ids.append(iid if not it else f"{iid}|{hashlib.sha1(it.encode()).hexdigest()[:8]}")
                item_bench.append(b)
            idx[k] = iidx[key]
        s_idx = resp.subject_id.map(sidx).values
        ok = ~np.isnan(s_idx.astype(float))   # drop responses whose subject is missing from subjects.parquet
        rs.append(s_idx[ok].astype(np.int64))
        ri.append(idx[ok])
        ry.append(y[ok])
    ds = Dataset(subjects, items, np.array(item_bench), bench_names, subject_ids, item_ids,
                 np.concatenate(rs).astype(np.int64), np.concatenate(ri).astype(np.int64),
                 np.concatenate(ry).astype(float), revision)
    ds.binarize_rules = rules
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(ds, f)
    return ds
