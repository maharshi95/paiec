"""Minimal sanity check of a submission ZIP (not a substitute for the official starting-kit tools)."""
import importlib.util
import math
import sys
import tempfile
import zipfile
from pathlib import Path


def main(path):
    with tempfile.TemporaryDirectory() as d, zipfile.ZipFile(path) as z:
        names = z.namelist()
        assert "model.py" in names, "model.py must be at ZIP root"
        z.extractall(d)
        spec = importlib.util.spec_from_file_location("model", Path(d) / "model.py")
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        subj = {k: "" for k in ["normalized_name", "provider", "release_date", "access_date", "harness",
                                "harness_version", "reasoning_effort", "subject_features_extra"]}
        item = {"item_content": "2+2?", "item_features": "", "interactors": "", "benchmark_id": "b0"}
        for labeled in ([], [[[subj, item], 1], [[subj, item], 0]]):
            p = m.predict([subj, item], labeled)
            assert isinstance(p, float) and math.isfinite(p) and 0 <= p <= 1, p
    print("ok:", path, names)


if __name__ == "__main__":
    main(sys.argv[1])
