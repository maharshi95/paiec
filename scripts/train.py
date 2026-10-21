"""STUB: fit a model and evaluate with leave-one-benchmark-out CV; write runs/<name>/metrics.json.

TODO: implement after choosing the record format (see paiec.data.load.build_records) and
adding a Model interface (fit(records) / predict / acquisition) in paiec.models.
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    print("not implemented yet:", ap.parse_args().config)


if __name__ == "__main__":
    main()
