"""Hand-crafted, interpretable item features.

visible_features(item)   uses only test-time-visible fields (item_content, item_features). numpy/stdlib/re only,
                         so it can ship in the submission runtime unchanged.
hidden_features(row)     uses grading_criterion / verifier (NOT visible at test time). Analysis only: tells us
                         what drives difficulty and gives auxiliary targets for privileged-information training.

Run as a script to write data/analysis/item_difficulty/features.pkl.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np
import pandas as pd

from common import OUT, targets

_WORD = re.compile(r"[A-Za-z]+")
_NUM = re.compile(r"(?<![A-Za-z_])\d+(?:\.\d+)?")
_URL = re.compile(r"https?://\S+")
_PATHLIKE = re.compile(r"[\w\-./]+\.(?:py|js|ts|tsx|go|rs|java|c|cc|cpp|h|hpp|rb|php|kt|scala|cs|json|yml|yaml|toml|md|rst|txt|cfg|sh)\b")
_IDENT = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b|\b[a-z]+[A-Z][A-Za-z0-9]+\b")  # snake_case / camelCase
_STACK = re.compile(r"Traceback \(most recent call last\)|^\s+at [\w.$]+\(|panic:|Exception in thread|stack trace|goroutine \d+", re.M | re.I)
_ERRWORD = re.compile(r"\b(?:error|exception|fail(?:s|ed|ure)?|crash(?:es|ed)?|segfault|panic)\b", re.I)
_FEATWORD = re.compile(r"\b(?:feature request|enhancement|support for|add support|would be nice|proposal|ENH|RFE)\b", re.I)
_REPRO = re.compile(r"\b(?:steps to reproduce|to reproduce|reproduc(?:e|ible|tion)|minimal example|mcve)\b", re.I)
_EXPECT = re.compile(r"\b(?:expected behavior|actual behavior|expected result|expected:|actual:)\b", re.I)
_MATH_ENV = re.compile(r"\\(?:frac|sum|prod|int|binom|sqrt|lfloor|lceil|cdot|le|ge|leq|geq|neq|pmod|mod|equiv)\b")
_PROOF = re.compile(r"\b(?:prove|proof|show that|determine all|find all)\b", re.I)
_MULTI = re.compile(r"\b(?:and then|then|after that|afterwards|finally|also|as well as|both|each|every|all)\b", re.I)
_COMPARE = re.compile(r"\b(?:compare|comparison|differ|difference|rank|most|least|highest|lowest|largest|smallest|best|worst)\b", re.I)
_COUNT = re.compile(r"\bhow many\b|\bnumber of\b|\bcount\b|\bsum of\b|\btotal\b", re.I)
_DATE = re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}|\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", re.I)
_QUOTED = re.compile(r"\"[^\"]{2,80}\"|'[^']{2,80}'")
_TODO = re.compile(r'TODO: Implement block "([^"]+)"\s*\n\s*# Approximately (\d+) line')
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)]|\\item)\s", re.M)


def _safe_log(x):
    return math.log1p(max(x, 0))


def _kv(s):
    out = {}
    for part in (s or "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def visible_features(item: dict, max_scan: int = 200_000) -> dict:
    c = item.get("item_content", "") or ""
    s = c[:max_scan]
    n = max(len(c), 1)
    words = _WORD.findall(s)
    nw = max(len(words), 1)
    lines = s.split("\n")
    kv = _kv(item.get("item_features", ""))
    code_lines = sum(1 for l in lines if re.match(r"\s{4,}\S|\t\S", l) or l.rstrip().endswith((";", "{", "}", ":")))
    f = {
        # ---- generic size / surface ----
        "log_chars": math.log1p(len(c)),
        "log_words": _safe_log(len(words)),
        "log_lines": _safe_log(len(lines)),
        "frac_digit": sum(ch.isdigit() for ch in s) / min(n, max_scan),
        "frac_upper": sum(ch.isupper() for ch in s) / min(n, max_scan),
        "frac_nonascii": sum(ord(ch) > 127 for ch in s) / min(n, max_scan),
        "frac_punct": sum(ch in "{}[]()<>;=$\\_^" for ch in s) / min(n, max_scan),
        "mean_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "ttr": len(set(w.lower() for w in words[:2000])) / min(nw, 2000),
        "n_numbers": _safe_log(len(_NUM.findall(s))),
        "n_urls": _safe_log(len(_URL.findall(s))),
        "n_code_fences": _safe_log(s.count("```")),
        "frac_code_lines": code_lines / max(len(lines), 1),
        "n_paths": _safe_log(len(set(_PATHLIKE.findall(s)))),
        "n_identifiers": _safe_log(len(set(_IDENT.findall(s)))),
        "has_stack": float(bool(_STACK.search(s))),
        "n_errwords": _safe_log(len(_ERRWORD.findall(s))),
        "is_feature_req": float(bool(_FEATWORD.search(s[:3000]))),
        "has_repro": float(bool(_REPRO.search(s))),
        "has_expected": float(bool(_EXPECT.search(s))),
        "n_bullets": _safe_log(len(_BULLET.findall(s))),
        "n_quoted": _safe_log(len(_QUOTED.findall(s))),
        "n_questions": _safe_log(s.count("?")),
        "n_sentences": _safe_log(len(re.findall(r"[.!?](?:\s|$)", s))),
        # ---- reasoning / task-structure cues ----
        "n_math_env": _safe_log(len(_MATH_ENV.findall(s)) + s.count("$") // 2),
        "has_proof": float(bool(_PROOF.search(s))),
        "n_multi": _safe_log(len(_MULTI.findall(s))),
        "n_compare": _safe_log(len(_COMPARE.findall(s))),
        "n_count": _safe_log(len(_COUNT.findall(s))),
        "n_and": _safe_log(len(re.findall(r"\band\b", s, re.I))),
        "n_dates": _safe_log(len(_DATE.findall(s))),
        "mentions_image": float(bool(re.search(r"\b(?:image|figure|picture|diagram|screenshot|photo)\b", s[:4000], re.I))),
        "mentions_table_page": float(bool(re.search(r"\b(?:table|page|chart|graph|slide)\b", s[:4000], re.I))),
        # ---- kv metadata (generic) ----
        "kv_n_keys": len(kv),
        "kv_has_index": float(any(v.isdigit() for v in kv.values())),
        "kv_index": math.log1p(max([int(v) for v in kv.values() if v.isdigit()] or [0])),
        # ---- code-task specific (fire only when present) ----
    }
    todos = [(m.group(1), int(m.group(2))) for m in _TODO.finditer(c) if m.group(1) != "calculate area"]
    f["todo_lines"] = math.log1p(sum(t[1] for t in todos)) if todos else 0.0
    f["todo_n_blocks"] = float(len(todos))
    i_code = c.find("Here is the code that you need to complete")
    f["code_part_log_chars"] = math.log1p(len(c) - i_code) if i_code >= 0 else 0.0
    f["paper_part_log_chars"] = math.log1p(i_code) if i_code >= 0 else 0.0
    # issue-shaped texts: title = first line
    first = c.split("\n", 1)[0]
    f["title_log_chars"] = math.log1p(len(first))
    f["title_is_bug"] = float(bool(re.search(r"\b(?:bug|error|fail|crash|wrong|incorrect|broken|not work|doesn't|does not|issue)\b", first, re.I)))
    return f


def hidden_features(row) -> dict:
    """Training-only features from grading_criterion / verifier."""
    out = {}
    try:
        g = json.loads(row.grading_criterion) if isinstance(row.grading_criterion, str) else {}
    except json.JSONDecodeError:
        g = {}
    ref = g.get("reference_answer") if isinstance(g, dict) else None
    ref = ref if isinstance(ref, str) else ""
    out["h_ref_log_chars"] = math.log1p(len(ref))
    if ref.startswith("diff --git"):
        files = re.findall(r"^diff --git a/(\S+)", ref, re.M)
        add = len(re.findall(r"^\+(?!\+\+)", ref, re.M))
        rem = len(re.findall(r"^-(?!--)", ref, re.M))
        out.update({"h_patch_files": math.log1p(len(files)), "h_patch_hunks": math.log1p(ref.count("\n@@")),
                    "h_patch_lines": math.log1p(add + rem), "h_patch_add": math.log1p(add), "h_patch_rem": math.log1p(rem),
                    "h_patch_test_files": float(sum("test" in f.lower() for f in files))})
    else:
        out["h_ref_lines"] = math.log1p(ref.count("\n") + 1 if ref else 0)
    try:
        v = json.loads(row.verifier) if isinstance(row.verifier, str) else {}
        spec = json.loads(v.get("spec", "{}")) if isinstance(v.get("spec"), str) and v.get("spec", "").startswith("{") else {}
    except (json.JSONDecodeError, AttributeError):
        spec = {}
    if isinstance(spec, dict):
        f2p, p2p = spec.get("FAIL_TO_PASS"), spec.get("PASS_TO_PASS")
        if isinstance(f2p, list):
            out["h_f2p"] = math.log1p(len(f2p))
        if isinstance(p2p, list):
            out["h_p2p"] = math.log1p(len(p2p))
        tp = spec.get("test_patch")
        if isinstance(tp, str):
            out["h_test_patch_lines"] = math.log1p(tp.count("\n"))
    out["h_is_proof_criterion"] = float(isinstance(g, dict) and "grading_scheme_desc" in json.dumps(g)[:200])
    out["h_has_asset"] = float(isinstance(row.asset_manifest, str) and len(row.asset_manifest) > 4)
    return out


def build(refresh=False):
    f = OUT / "features.pkl"
    if f.exists() and not refresh:
        return pd.read_pickle(f)
    df = targets()
    vis = pd.DataFrame([visible_features({"item_content": c, "item_features": k}) for c, k in zip(df.content, df.item_features)])
    hid = pd.DataFrame([hidden_features(r) for r in df.itertuples()])
    feats = pd.concat([df[["idx", "bench"]].reset_index(drop=True), vis, hid], axis=1)
    feats.to_pickle(f)
    return feats


if __name__ == "__main__":
    feats = build(refresh=True)
    print(feats.shape)
    print(feats.groupby("bench").mean(numeric_only=True).T.round(2).to_string())
