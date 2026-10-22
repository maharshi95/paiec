"""Inspect item_features kv structure and duplicate content on the dev test benchmarks."""
import sys
from collections import Counter
sys.path.insert(0, "src")
from paiec.config import load_config
from paiec.data.dataset import load_dataset
from paiec.features.hashing import parse_kv

cfg = load_config("configs/research/dev.yaml", [])
d = cfg["data"]
ds = load_dataset(d["root"], d["benchmarks"], d["binarize"], d["cache_dir"], max_responses=d.get("max_responses"),
                  max_item_chars=d.get("max_item_chars"), prefer_seen_from=d.get("prefer_seen_from"))
import numpy as np
tests = [b for f in cfg["cv"]["test_benchmarks"] for b in f]
used = set(np.unique(ds.resp_i).tolist())
for b in tests:
    bi = ds.bench_names.index(b)
    items = [ds.items[i] for i in np.nonzero(ds.item_bench == bi)[0] if i in used]
    kvc = Counter()
    vals = {}
    for it in items:
        for k, v in parse_kv(it["item_features"]).items():
            vals.setdefault(k, Counter())[v] += 1
    cont = Counter(it["item_content"] for it in items)
    ndup = sum(c for c in cont.values() if c > 1)
    print(f"== {b}: {len(items)} items, distinct content {len(cont)}, items in dup groups {ndup}, max dup {max(cont.values())}, "
          f"mean len {np.mean([len(it['item_content']) for it in items]):.0f}")
    for k, c in vals.items():
        print(f"   {k}: {len(c)} values, top {c.most_common(3)}"[:200])
    print("   ex:", repr(items[0]["item_content"][:150]))
