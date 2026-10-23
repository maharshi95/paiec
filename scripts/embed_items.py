"""Embed every item of the dev dataset with all-mpnet-base-v2 -> mpnet_items.npz (keys, fp16 emb). GPU job.

    python scripts/embed_items.py --config configs/research/dev.yaml
"""
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, "src")
from paiec.config import load_config  # noqa: E402
from paiec.data.dataset import load_dataset  # noqa: E402
from paiec.features.mpnet import CACHE, ENCODER, item_hash, item_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/research/dev.yaml")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    import torch
    from sentence_transformers import SentenceTransformer
    d = load_config(a.config)["data"]
    ds = load_dataset(d["root"], d["benchmarks"], d["binarize"], d["cache_dir"], max_responses=d.get("max_responses"),
                      max_item_chars=d.get("max_item_chars"), prefer_seen_from=d.get("prefer_seen_from"))
    items = ds.items[:a.limit] if a.limit else ds.items
    items = items[a.shard::a.nshards]
    texts = [item_text(i) for i in items]
    keys = np.array([item_hash(i) for i in items], dtype=np.uint64)
    model = SentenceTransformer(ENCODER, device="cuda")
    model.half()
    model.max_seq_length = 128
    order = np.argsort([len(t) for t in texts])   # length-sorted batches
    emb = np.zeros((len(texts), 768), np.float16)
    t0 = time.time()
    for s in range(0, len(order), a.batch * 50):
        idx = order[s:s + a.batch * 50]
        emb[idx] = model.encode([texts[k] for k in idx], batch_size=a.batch, normalize_embeddings=False,
                                convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
        print(f"{s + len(idx)}/{len(order)} {time.time() - t0:.0f}s", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out = CACHE.with_name(f"{CACHE.stem}.shard{a.shard}of{a.nshards}.npz") if a.nshards > 1 else CACHE
    np.savez(out, keys=keys, emb=emb)
    print("saved", out)


if __name__ == "__main__":
    main()
