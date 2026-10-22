"""Truncation / text views for long item content. numpy + stdlib only (runtime-safe).

head       first N chars
tail       last N chars
headtail   first N/2 + last N/2
chunks     K evenly spaced N-char chunks (embedded separately and mean-pooled)
focus      transductive boilerplate removal: drop lines that occur in many items of the same benchmark
           (line document frequency within the benchmark pool), keep the rare lines plus +-W neighbouring
           rare-or-not lines, then take the first N chars. Needs the whole pool of the benchmark's items, which
           is visible at test time (every eval/acq input carries the same anonymised benchmark_id).
"""
from collections import Counter, defaultdict


def head(c, n=2000):
    return c[:n]


def tail(c, n=2000):
    return c[-n:]


def headtail(c, n=2000):
    return c if len(c) <= n else c[: n // 2] + "\n...\n" + c[-n // 2:]


def chunks(c, n=2000, k=8):
    if len(c) <= n:
        return [c]
    step = (len(c) - n) / max(k - 1, 1)
    return [c[int(j * step): int(j * step) + n] for j in range(k)]


def _norm(line):
    return " ".join(line.split()).lower()


def line_df(contents):
    df = Counter()
    for c in contents:
        df.update({_norm(l) for l in c.split("\n") if l.strip()})
    return df


def focus(c, df, n_items, n=2000, window=10, frac=0.0, min_df=2):
    """Keep lines rare within the benchmark (df <= max(min_df, frac*n_items)) and +-window neighbours."""
    lines = c.split("\n")
    thr = max(min_df, frac * n_items)
    rare = [bool(l.strip()) and df.get(_norm(l), 0) <= thr for l in lines]
    if not any(rare):
        return c[:n]
    keep = [False] * len(lines)
    for j, r in enumerate(rare):
        if r:
            for t in range(max(0, j - window), min(len(lines), j + window + 1)):
                keep[t] = True
    # drop very common lines even inside windows (templates), keep code context otherwise
    out = [l for j, l in enumerate(lines) if keep[j] and (rare[j] or df.get(_norm(l), 0) <= 0.5 * n_items)]
    s = "\n".join(out)
    return s[:n] if s.strip() else c[:n]


def focus_views(contents, groups, n=2000, **kw):
    """Apply `focus` per group (benchmark)."""
    by = defaultdict(list)
    for j, g in enumerate(groups):
        by[g].append(j)
    out = [None] * len(contents)
    for g, idx in by.items():
        df = line_df([contents[j] for j in idx])
        for j in idx:
            out[j] = focus(contents[j], df, len(idx), n=n, **kw)
    return out
