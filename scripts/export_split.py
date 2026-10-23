"""Export the local train/val split as two measurement-db-layout datasets (and optionally push them to HF).

    python scripts/export_split.py --out /fs/clip-scratch/mgor/aims-split
    python scripts/export_split.py --out ... --push   # private repos; also deletes remote files removed locally
    python scripts/export_split.py --root data/mdb --out data/aims   # rebuild data/aims/{train,val} from a local db

The default config, configs/aims/export.yaml, is the seen-dev split minus `data.exclude` (docs/todo.md).
From the config's universe (`data.benchmarks`, which already excludes the locked hold-out, minus `data.exclude`):
  val   = the benchmarks of `cv.test_benchmarks` (5 grouped folds; fold ids go to split.json),
  train = every other benchmark.
Each benchmark keeps the measurement-db layout ({benchmark}/formatted_tables/{benchmarks,subjects,items,
responses}.parquet) with the loader's preprocessing pre-applied, so `load_dataset` on the export returns
the same Dataset as on the full db with this config:
  - responses: only those the config's binarisation keeps, capped at `data.max_responses` per benchmark by
    the loader's own subject subsampling (`cap_subjects`); raw `response` values are kept, not binarised;
  - items: only items that have a kept response (full content, not truncated); subjects: the full table;
  - a graded scale without declared bounds is normalised by its observed range, which subsampling changes,
    so its full-data min/max are written into `response_scale`.
Traces, assets and raw/ archives are not exported.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paiec.config import load_config
from paiec.data.dataset import (
    _binarize,
    cap_subjects,
    find_benchmarks,
    parse_scale,
    seen_identities,
)

UPSTREAM = "aims-foundations/measurement-db"
PINNED = "54074453da7b9911be0f48e4c560be547bc84d21"   # upstream revision of the published split
REPOS = {"train": "mgor/aims-measurement-db-train", "val": "mgor/aims-measurement-db-val"}


def export_benchmark(slug, src, dst, d, seen_ids):
    cat = pd.read_parquet(src / "benchmarks.parquet")
    if cat.iloc[0].get("release", "public") != "public":
        return None
    resp = pd.read_parquet(src / "responses.parquet")
    scale = parse_scale(cat.iloc[0].get("response_scale"))
    _, keep, rule = _binarize(slug, resp.response.values, d["binarize"], scale)
    if not keep.any():
        return None
    resp = resp[keep]
    pinned = None
    if rule.startswith("threshold") and not scale.get("values") and (scale.get("min") is None or scale.get("max") is None):
        # the loader normalises by the *observed* range here, which subsampling would change: pin the full range
        pinned = [float(resp.response.min()), float(resp.response.max())]
        scale.update(min=pinned[0], max=pinned[1])
        cat = cat.assign(response_scale=json.dumps(scale))
    subjects = pd.read_parquet(src / "subjects.parquet")
    cap = (d.get("cap_overrides") or {}).get(slug, d.get("max_responses"))
    resp = resp[cap_subjects(slug, resp.subject_id, subjects, seen_ids, cap)]
    items = pd.read_parquet(src / "items.parquet")
    items = items[items.item_id.isin(set(resp.item_id))]
    dst.mkdir(parents=True, exist_ok=True)
    for name, df in (("benchmarks", cat), ("subjects", subjects), ("items", items), ("responses", resp)):
        df.to_parquet(dst / f"{name}.parquet", index=False, compression="zstd")
    return {"benchmark": slug, "rule": rule, "cap": cap, "subjects": int(resp.subject_id.nunique()),
            "items": int(resp.item_id.nunique()), "responses": len(resp),
            "pinned_bounds": pinned, "license": str(cat.iloc[0].get("license", "")), "domain": [str(x) for x in (cat.iloc[0].get("domain") if cat.iloc[0].get("domain") is not None else [])]}


CARD = """---
license: cc-by-sa-4.0
pretty_name: AI Measurement Data Bank ({split} split for PAIEC)
tags: [evaluation, item-response-theory, benchmarks, llm-evaluation]
---

# AI Measurement Data Bank: {split} split

The **{split}** half of the benchmark-level train/val split used by
[maharshi95/paiec](https://github.com/maharshi95/paiec) for the
[Predictive AI Evaluation Competition](https://aims.stanford.edu/competition) (NeurIPS 2026).
Its partner is [`{other_repo}`](https://huggingface.co/datasets/{other_repo}). The two share no benchmark,
which mimics the hosted test (benchmarks never seen in training).

Derived from [`{upstream}`](https://huggingface.co/datasets/{upstream}) at revision `{revision}`
(CC-BY-SA-4.0; individual benchmarks keep their own licences, listed in `split.json` and each
`benchmarks.parquet`). Please cite the original work:

> Truong, Truong, Koyejo. *The AI Measurement Data Bank*, 2026. <https://aimslab.stanford.edu/measurement-db>

## Contents

{n} benchmarks, {total:,} responses. Same layout and schema as the upstream dataset:

```
{{benchmark}}/formatted_tables/{{benchmarks,subjects,items,responses}}.parquet
REVISION      # upstream measurement-db revision
split.json    # benchmark lists, folds, preprocessing, per-benchmark stats
```

Preprocessing relative to upstream (the paiec loader's settings for this split):
- only responses that binarise under `{binarize}` (the raw `response` value is kept);
- about {cap:,} responses per benchmark at most, by seeded subject subsampling that keeps models seen in the
  six reference benchmarks first (each kept subject keeps all its responses, so a benchmark whose first
  subject alone exceeds the cap keeps just that subject);
- only items with a kept response; traces, assets and raw archives dropped;
- graded scales without declared bounds get the full upstream observed `min`/`max` written into
  `response_scale` (`pinned_bounds` in `split.json`), so binarisation matches the full data;
- withheld benchmarks and benchmarks with no binarisable responses dropped; the 12 locked hold-out
  benchmarks of the paiec harness are in neither split;
- excluded on purpose: {excluded}. A mixed-scale benchmark (0/1 grades plus 1-5 or 0-100 ratings) is
  binarised against its whole observed range, which mislabels the small-range part; see `docs/todo.md` in the repo.
{folds}
## Use

```bash
git clone https://github.com/maharshi95/paiec && cd paiec
python scripts/download_data.py --split   # both splits -> data/aims/{{train,val}}
python scripts/run.py --config configs/aims/irt.yaml
```

Or read a table directly:

```python
import pandas as pd
pd.read_parquet("hf://datasets/{repo}/{example}/formatted_tables/responses.parquet")
```

## Benchmarks

| benchmark | subjects | items | responses | binarisation |
|---|---|---|---|---|
"""
FOLDS_NOTE = """
`split.json` assigns the val benchmarks to 5 grouped folds (`folds`); the harness evaluates each group
together.
"""


def card(split, meta, rows):
    other = "val" if split == "train" else "train"
    head = CARD.format(split=split, other_repo=REPOS[other], upstream=UPSTREAM, revision=meta["upstream_revision"],
                       n=len(rows), total=sum(r["responses"] for r in rows), binarize=json.dumps(meta["binarize"]),
                       cap=meta["max_responses"], folds=FOLDS_NOTE if split == "val" else "", repo=REPOS[split],
                       example=rows[0]["benchmark"], excluded=", ".join(f"`{b}`" for b in meta.get("excluded", [])) or "none")
    body = [f"| {r['benchmark']} | {r['subjects']} | {r['items']:,} | {r['responses']:,} | {r['rule']} |" for r in rows]
    return head + "\n".join(body) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/aims/export.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", help="measurement-db directory (default: the config's data.root)")
    ap.add_argument("--push", action="store_true", help="upload to the HF repos (created private if missing)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.root:
        cfg["data"]["root"] = a.root
    d = cfg["data"]
    folds = [[f] if isinstance(f, str) else list(f) for f in cfg["cv"]["test_benchmarks"]]
    val = [b for f in folds for b in f]
    tables = find_benchmarks(d["root"])
    exclude = set(d.get("exclude") or [])
    universe = [b for b in sorted(tables) if b in d["benchmarks"] and b not in exclude]
    split = {"train": [b for b in universe if b not in val], "val": [b for b in universe if b in val]}
    seen_ids = seen_identities(tables, d.get("prefer_seen_from"))
    revision = (Path(d["root"]) / "REVISION").read_text().strip()
    if revision != PINNED:
        print(f"warning: {d['root']} is at revision {revision[:8]}, not {PINNED[:8]}; the split will differ from the "
              "published one (download with --revision " + PINNED + ")")
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    out = Path(a.out)
    for name, benches in split.items():
        root = out / name
        if root.exists():
            shutil.rmtree(root)
        rows = []
        for b in benches:
            r = export_benchmark(b, tables[b], root / b / "formatted_tables", d, seen_ids)
            if r:
                rows.append(r)
                print(f"  {name} {b}: {r['responses']:,} responses", flush=True)
            else:
                print(f"  {name} {b}: dropped (withheld or nothing binarisable)", flush=True)
        kept = [r["benchmark"] for r in rows]
        meta = {"split": name, "upstream": UPSTREAM, "upstream_revision": revision, "source_config": a.config,
                "paiec_commit": sha, "binarize": d["binarize"], "max_responses": d.get("max_responses"), "cap_overrides": d.get("cap_overrides"),
                "prefer_seen_from": d.get("prefer_seen_from"), "excluded": sorted(exclude), "benchmarks": kept,
                "folds": [[b for b in f if b in kept] for f in folds] if name == "val" else None, "stats": rows}
        (root / "REVISION").write_text(revision + "\n")
        (root / "split.json").write_text(json.dumps(meta, indent=1))
        (root / "README.md").write_text(card(name, meta, rows))
        print(f"{name}: {len(kept)} benchmarks, {sum(r['responses'] for r in rows):,} responses -> {root}")
    if a.push:
        from huggingface_hub import HfApi
        api = HfApi()
        for name, repo in REPOS.items():
            api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
            api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(out / name),
                              delete_patterns=["*/formatted_tables/*"],   # drop remote benchmarks removed locally
                              commit_message=f"export {name} split (paiec {sha})")
            print(f"pushed {out / name} -> https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    main()
