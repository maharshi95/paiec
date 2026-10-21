"""Reference baseline: mean of acquired labels in the same (subject, benchmark); 0.5 if none."""


def predict(input, labeled):
    subject, item = input
    bid = item["benchmark_id"]
    ys = [y for (s, it), y in labeled if s == subject and it["benchmark_id"] == bid]
    return sum(ys) / len(ys) if ys else 0.5
