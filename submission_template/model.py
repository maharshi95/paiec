"""Submission entry point (copied to ZIP root by scripts/build_zip.py).

predict(input, labeled) -> float in [0,1].
  input   = [subject: dict[str,str], item: dict[str,str]]   (see docs/submission-format.md)
  labeled = [[[subject, item], label], ...]  label in {0,1}; empty at budget 0
No network except approved inference APIs; no runtime model downloads; 8h total budget.
"""


def predict(input, labeled):
    subject, item = input
    bid = item.get("benchmark_id", "")
    ys = [y for (s, it), y in labeled if s == subject and it.get("benchmark_id") == bid]
    return float(sum(ys) / len(ys)) if ys else 0.5
