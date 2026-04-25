"""Merge classification and segmentation predictions into one submission."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from shared import EXPERIMENTS, OUTPUT_DIR, get_experiment_config  # noqa: E402

DEFAULT_SEG_CSV = OUTPUT_DIR / "submission_exp_v7_segformer_b3.csv"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENTS),
        default=None,
        help="Classification experiment to merge.",
    )
    parser.add_argument(
        "--seg-csv",
        type=Path,
        default=DEFAULT_SEG_CSV,
        help="CSV containing *_segmentation rows.",
    )
    return parser.parse_args()


def main(config=None, seg_csv: Path | None = None):
    if config is None:
        args = parse_args()
        config = get_experiment_config(args.experiment)
        seg_csv = args.seg_csv

    seg_csv = Path(seg_csv or DEFAULT_SEG_CSV)
    clf_csv = config.submission_path
    out_csv = config.submissions_dir / f"submission_final_{config.name}.csv"
    config.submissions_dir.mkdir(parents=True, exist_ok=True)

    if not clf_csv.exists():
        raise FileNotFoundError(f"Classification CSV not found: {clf_csv}")
    if not seg_csv.exists():
        raise FileNotFoundError(f"Segmentation CSV not found: {seg_csv}")

    clf = pd.read_csv(clf_csv, index_col="Id")
    seg = pd.read_csv(seg_csv, index_col="Id")

    clf_rows = clf[clf.index.str.endswith("_classification")]
    seg_rows = seg[seg.index.str.endswith("_segmentation")]

    print(f"Classification rows : {len(clf_rows)}")
    print(f"Segmentation rows   : {len(seg_rows)}")

    merged = pd.concat([clf_rows, seg_rows]).sort_index()
    merged["_id"] = merged.index.str.extract(r"^(\d+)_")[0].astype(int)
    merged["_type"] = merged.index.str.extract(r"_(\w+)$")[0]
    merged = merged.sort_values(["_id", "_type"]).drop(columns=["_id", "_type"])

    merged.to_csv(out_csv)
    print(f"Saved {len(merged)} rows to {out_csv}")
    print(merged.head(4))
    return out_csv


if __name__ == "__main__":
    main()
