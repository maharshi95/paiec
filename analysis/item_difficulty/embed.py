"""Sentence embeddings of item text views (CPU; the local 1080 Ti is sm_61, unsupported by the installed torch cu130).

usage: python embed.py MODEL VIEW [VIEW ...]     e.g. python embed.py minilm head focus chunks
Writes data/analysis/item_difficulty/emb/<model>__<view>.npy (rows aligned with targets()) and prints throughput.
"""
import sys
import time

import numpy as np
import torch

from common import OUT, targets
from views import chunks, focus_views, head, headtail, tail

MODELS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "bge-small": ("BAAI/bge-small-en-v1.5", ""),
    "e5-small": ("intfloat/e5-small-v2", "query: "),
    "bge-base": ("BAAI/bge-base-en-v1.5", ""),
    "gte-small": ("thenlper/gte-small", ""),
    "jina-code": ("jinaai/jina-embeddings-v2-base-code", ""),
}
N_CHARS = 2000


def main():
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", 4)))
    from sentence_transformers import SentenceTransformer

    key, views = sys.argv[1], sys.argv[2:]
    repo, prefix = MODELS[key]
    model = SentenceTransformer(repo, device="cpu", trust_remote_code=key.startswith("jina"))
    model.max_seq_length = 512
    df = targets()
    contents = df.content.tolist()
    (OUT / "emb").mkdir(exist_ok=True)
    for view in views:
        t = time.time()
        if view == "chunks":
            parts, owner = [], []
            for j, c in enumerate(contents):
                for p in chunks(c, N_CHARS, 8):
                    parts.append(prefix + p)
                    owner.append(j)
            E = model.encode(parts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
            owner = np.array(owner)
            X = np.zeros((len(contents), E.shape[1]), np.float32)
            np.add.at(X, owner, E)
            X /= np.bincount(owner)[:, None]
        else:
            if view == "focus":
                texts = focus_views(contents, df.bench.tolist(), N_CHARS)
            else:
                texts = [{"head": head, "tail": tail, "headtail": headtail}[view](c, N_CHARS) for c in contents]
            X = model.encode([prefix + s for s in texts], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        np.save(OUT / "emb" / f"{key}__{view}.npy", X.astype(np.float32))
        dt = time.time() - t
        print(f"{key} {view}: {X.shape} in {dt:.0f}s ({len(contents) / dt:.1f} items/s)", flush=True)


if __name__ == "__main__":
    main()
