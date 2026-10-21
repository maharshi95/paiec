"""Package a submission: model.py (+labeling.py, models.txt, requirements.txt, artifacts) at ZIP root."""
import argparse, zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="submission_template", help="directory containing model.py")
    ap.add_argument("--out", default="dist/submission.zip")
    a = ap.parse_args()
    src = Path(a.dir)
    assert (src / "model.py").exists(), "model.py required"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(src.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts and f.name != "submission_config.json":
                z.write(f, f.relative_to(src).as_posix())
    print("wrote", a.out)


if __name__ == "__main__":
    main()
