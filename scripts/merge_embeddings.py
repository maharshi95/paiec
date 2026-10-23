"""Merge mpnet_items.shard{k}of{n}.npz -> mpnet_items.npz (keys, fp16 emb)."""
import sys

import numpy as np

sys.path.insert(0, "src")
from paiec.features.mpnet import CACHE  # noqa: E402

parts = sorted(CACHE.parent.glob(f"{CACHE.stem}.shard*of*.npz"))
z = [np.load(p) for p in parts]
keys = np.concatenate([a["keys"] for a in z])
emb = np.concatenate([a["emb"] for a in z])
assert len(np.unique(keys)) <= len(keys)
print(len(parts), "shards,", len(keys), "rows,", len(np.unique(keys)), "unique keys, zero rows:", int((np.abs(emb).sum(1) == 0).sum()))
np.savez(CACHE, keys=keys, emb=emb)
print("saved", CACHE)
