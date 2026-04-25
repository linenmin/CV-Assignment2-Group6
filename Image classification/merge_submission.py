"""Merge classification and segmentation predictions into one submission."""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from shared import (  # noqa: E402
    CLASSIFICATION_OUTPUT_DIR,
    EXPERIMENTS,
    LABELS,
    OUTPUT_DIR,
    get_experiment_config,
)

DEFAULT_SEG_CSV = OUTPUT_DIR / "submission_exp_v10_segman_b_iter25000.csv"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENTS),
        default=None,
        help="Classification experiment to merge. Inferred from --clf-csv when possible.",
    )
    parser.add_argument(
        "--clf-csv",
        type=Path,
        default=None,
        help=(
            "Classification CSV to merge. Accepts submission_classification_*.csv "
            "or test_binary_predictions_*.csv. Defaults to the experiment's "
            "submission_classification CSV."
        ),
    )
    parser.add_argument(
        "--seg-csv",
        type=Path,
        default=DEFAULT_SEG_CSV,
        help="CSV containing *_segmentation rows.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional explicit output CSV path. Defaults to the model's submissions folder.",
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    """Return a filename-safe token."""

    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return slug or "csv"


def infer_experiment_from_path(path: Path | None) -> str | None:
    if path is None:
        return None

    try:
        relative = path.resolve().relative_to(CLASSIFICATION_OUTPUT_DIR.resolve())
    except ValueError:
        relative = None

    if relative is not None and relative.parts and relative.parts[0] in EXPERIMENTS:
        return relative.parts[0]

    path_text = str(path).lower()
    for experiment_name in sorted(EXPERIMENTS, key=len, reverse=True):
        if experiment_name.lower() in path_text:
            return experiment_name
    return None


def resolve_config(experiment: str | None, clf_csv: Path | None):
    inferred = infer_experiment_from_path(clf_csv)
    return get_experiment_config(experiment or inferred)


def rle_encode(arr: np.ndarray) -> str:
    pixels = np.concatenate([[0], arr.flatten(), [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def load_classification_rows(clf_csv: Path) -> pd.DataFrame:
    clf = pd.read_csv(clf_csv, index_col="Id")

    if clf.index.astype(str).str.endswith("_classification").any():
        return clf[clf.index.astype(str).str.endswith("_classification")]

    if all(label in clf.columns for label in LABELS):
        values = clf[LABELS].to_numpy()
        unique_values = set(np.unique(values[~pd.isna(values)]).tolist())
        if not unique_values.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(
                f"{clf_csv} looks like probabilities, not binary predictions. "
                "Use submission_classification_*.csv or test_binary_predictions_*.csv."
            )

        rows = {"Id": [], "Predicted": []}
        for idx, label_vec in clf[LABELS].astype(int).iterrows():
            rows["Id"].append(f"{idx}_classification")
            rows["Predicted"].append(rle_encode(label_vec.to_numpy()))
        return pd.DataFrame(rows).set_index("Id")

    raise ValueError(
        f"Could not read classification rows from {clf_csv}. Expected either "
        "*_classification rows or the 20 binary label columns."
    )


def load_segmentation_rows(seg_csv: Path) -> pd.DataFrame:
    seg = pd.read_csv(seg_csv, index_col="Id")
    if not seg.index.astype(str).str.endswith("_segmentation").any():
        raise ValueError(f"No *_segmentation rows found in {seg_csv}")
    return seg[seg.index.astype(str).str.endswith("_segmentation")]


def default_output_path(config, clf_csv: Path, seg_csv: Path) -> Path:
    clf_tag = slugify(clf_csv.stem)
    seg_tag = slugify(seg_csv.stem)
    prefix = f"submission_classification_{config.name}"
    if clf_tag == prefix:
        name = f"submission_final_{config.name}__seg_{seg_tag}.csv"
    else:
        name = f"submission_final_{config.name}__clf_{clf_tag}__seg_{seg_tag}.csv"
    return config.submissions_dir / name


def main(
    config=None,
    clf_csv: Path | None = None,
    seg_csv: Path | None = None,
    out_csv: Path | None = None,
):
    if config is None:
        args = parse_args()
        clf_csv = args.clf_csv
        config = resolve_config(args.experiment, clf_csv)
        seg_csv = args.seg_csv
        out_csv = args.out_csv

    seg_csv = Path(seg_csv or DEFAULT_SEG_CSV)
    clf_csv = Path(clf_csv or config.submission_path)
    out_csv = Path(out_csv) if out_csv else default_output_path(config, clf_csv, seg_csv)
    config.submissions_dir.mkdir(parents=True, exist_ok=True)

    if not clf_csv.exists():
        raise FileNotFoundError(f"Classification CSV not found: {clf_csv}")
    if not seg_csv.exists():
        raise FileNotFoundError(f"Segmentation CSV not found: {seg_csv}")

    clf_rows = load_classification_rows(clf_csv)
    seg_rows = load_segmentation_rows(seg_csv)

    print(f"Classification rows : {len(clf_rows)}")
    print(f"Segmentation rows   : {len(seg_rows)}")
    print(f"Classification CSV  : {clf_csv}")
    print(f"Segmentation CSV    : {seg_csv}")

    merged = pd.concat([clf_rows, seg_rows]).sort_index()
    ids = merged.index.astype(str).str.extract(r"^(\d+)_", expand=False)
    row_types = merged.index.astype(str).str.extract(r"_(\w+)$", expand=False)
    if ids.isna().any() or row_types.isna().any():
        raise ValueError("Submission row IDs must look like '<number>_classification/segmentation'.")

    merged["_id"] = ids.astype(int).to_numpy()
    merged["_row_order"] = row_types.map({"classification": 0, "segmentation": 1}).to_numpy()
    if merged["_row_order"].isna().any():
        raise ValueError("Submission rows must end with _classification or _segmentation.")
    merged = merged.sort_values(["_id", "_row_order"]).drop(columns=["_id", "_row_order"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv)
    print(f"Saved {len(merged)} rows to {out_csv}")
    print(merged.head(4))
    return out_csv


if __name__ == "__main__":
    main()
