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
        if "labeling.py" in names:   # a broken acquisition config fails at import hosted (failure code 12)
            spec = importlib.util.spec_from_file_location("labeling", Path(d) / "labeling.py")
            lab = importlib.util.module_from_spec(spec); spec.loader.exec_module(lab)
            ctx = {"subject_id": "s0", "benchmark_id": "b0", "labels_acquired": 0, "labels_remaining": 31,
                   "max_labels": 31, "items_remaining": 120}
            take = lab.acquisition_function([subj, item], 0.5, [], ctx)
            assert type(take) is bool, f"acquisition_function must return a bool, got {take!r}"
    print("ok:", path, names)


if __name__ == "__main__":
    main(sys.argv[1])
