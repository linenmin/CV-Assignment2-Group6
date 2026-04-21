"""
Merge classification and segmentation predictions into one Kaggle submission.

    classification rows  ← submission_classification.csv
    segmentation rows    ← submission_exp_v1_ade20k_main.csv

Run from project root:
    python "Image classification/merge_submission.py"

Output:
    output/submission_final.csv
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from shared import OUTPUT_DIR  # noqa: E402

CLF_CSV = OUTPUT_DIR / "submission_classification.csv"
SEG_CSV = OUTPUT_DIR / "submission_exp_v1_ade20k_main.csv"
OUT_CSV = OUTPUT_DIR / "submission_final.csv"


def main():
    clf = pd.read_csv(CLF_CSV, index_col="Id")
    seg = pd.read_csv(SEG_CSV, index_col="Id")

    clf_rows = clf[clf.index.str.endswith("_classification")]
    seg_rows = seg[seg.index.str.endswith("_segmentation")]

    print(f"Classification rows : {len(clf_rows)}")
    print(f"Segmentation rows   : {len(seg_rows)}")

    merged = pd.concat([clf_rows, seg_rows]).sort_index()

    # Interleave: 0_classification, 0_segmentation, 1_classification, ...
    merged["_id"]  = merged.index.str.extract(r"^(\d+)_")[0].astype(int)
    merged["_type"] = merged.index.str.extract(r"_(\w+)$")[0]
    merged = merged.sort_values(["_id", "_type"]).drop(columns=["_id", "_type"])

    merged.to_csv(OUT_CSV)
    print(f"Saved {len(merged)} rows → {OUT_CSV}")
    print(merged.head(4))


if __name__ == "__main__":
    main()
