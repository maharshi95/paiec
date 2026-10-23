"""Fit a config on all included benchmarks and package a Codabench ZIP.

    python scripts/build_submission.py --config configs/irt.yaml [--out dist/irt.zip]

ZIP root: model.py, labeling.py (unless official_random), models.txt, predictor.json,
acquisition.json, paiec/ (runtime package: numpy + stdlib at score time).
Then run the official kit checks on the result:
    python <kit>/tools/check_submission_zip.py dist/irt.zip
    python <kit>/tools/run_smoke_test.py dist/irt/
"""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from paiec.acquisition import build_policy
from paiec.config import load_config
from paiec.data.dataset import load_from_config
from paiec.predictors import build_predictor
from paiec.runtime import POLICY_FILE, STATE_FILE, save_predictor

DEV_ONLY = ("ref_caimira",)   # need a 4.4 GB embedding table at predict time, which the ZIP does not contain


def build(cfg, out_zip: Path, ds=None) -> Path:
    if cfg["predictor"]["name"] in DEV_ONLY:
        raise SystemExit(f"{cfg['predictor']['name']} is a dev-only reference model and cannot be packaged: "
                         "its predictions need the mpnet embedding table, which is not in the ZIP")
    build_policy(cfg["acquisition"])   # a typo'd parameter would otherwise only fail when labeling.py is imported hosted
    if ds is None:
        d = cfg["data"]
        ds = load_from_config(d)
    benches = [b for b in (cfg["data"]["benchmarks"] or ds.benchmarks()) if b in ds.benchmarks()]
    pred = build_predictor(cfg["predictor"]).fit(ds.restrict(benches))
    stage = out_zip.with_suffix("")
    shutil.rmtree(stage, ignore_errors=True)
    shutil.copytree(REPO / "src" / "paiec", stage / "paiec", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy(REPO / "submission_template" / "model.py", stage / "model.py")
    if cfg["acquisition"]["name"] != "official_random":
        shutil.copy(REPO / "submission_template" / "labeling.py", stage / "labeling.py")
    save_predictor(pred, stage / STATE_FILE)
    (stage / POLICY_FILE).write_text(json.dumps(cfg["acquisition"]))
    (stage / "models.txt").write_text("")
    (stage / "build_info.json").write_text(json.dumps(
        {"name": cfg["name"], "benchmarks": benches, "data_revision": ds.revision, "config": cfg}, indent=2))
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(stage.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(stage).as_posix())
    return out_zip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    out = build(cfg, Path(a.out or f"dist/{cfg['name']}.zip"))
    print("wrote", out, f"({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
