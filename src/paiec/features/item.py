"""Item featurizer. Each entry in `fields` is one named extractor; ablate by editing the list.

text    hashed word unigrams (+bigrams) of item_content[:max_chars], log-tf, L2-normalised
length  log chars, log words, digit fraction, code-likeness (dense)
kv      hashed `key` and `key=value` tokens from item_features
inter   hashed tokens from interactors
"""
from collections import Counter
from itertools import pairwise

import numpy as np

from .hashing import hash_tokens, l2_normalize, parse_kv, tokenize

ALL_FIELDS = ("text", "length", "kv", "inter")
N_DENSE = 4


class ItemFeaturizer:
    def __init__(self, fields=("text", "length", "kv"), dim=1024, max_chars=4000, bigrams=False):
        unknown = set(fields) - set(ALL_FIELDS)
        assert not unknown, f"unknown item fields {unknown}"
        self.fields, self.dim, self.max_chars, self.bigrams = tuple(fields), dim, max_chars, bigrams

    @property
    def n_features(self) -> int:
        return self.dim + N_DENSE

    def transform_one(self, item: dict) -> np.ndarray:
        x = np.zeros(self.n_features, np.float32)
        h = x[: self.dim]
        content = item.get("item_content", "")
        if "text" in self.fields:
            toks = tokenize(content, self.max_chars)
            grams = toks + ([a + "_" + b for a, b in pairwise(toks)] if self.bigrams else [])
            t = np.zeros(self.dim, np.float32)
            for g, c in Counter(grams).items():
                hash_tokens(t, [g], "w:", float(np.log1p(c)))
            h += l2_normalize(t)
        if "kv" in self.fields:
            for k, v in parse_kv(item.get("item_features", "")).items():
                hash_tokens(h, [k, k + "=" + v], "kv:", 0.5)
        if "inter" in self.fields and item.get("interactors"):
            hash_tokens(h, tokenize(item["interactors"], 500), "in:", 0.5)
        if "length" in self.fields:
            n = max(len(content), 1)
            x[self.dim:] = [np.log1p(len(content)) / 10, np.log1p(len(content.split())) / 8,
                            sum(ch.isdigit() for ch in content[:20000]) / min(n, 20000),
                            float("def " in content or "{" in content or "```" in content)]
        return x

    def transform(self, items) -> np.ndarray:
        return np.stack([self.transform_one(it) for it in items]) if items else np.zeros((0, self.n_features), np.float32)
