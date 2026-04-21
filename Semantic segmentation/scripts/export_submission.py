import argparse
from pathlib import Path

from ga2_seg.submission import (
    attach_segmentation_predictions,
    build_submission_dataframe,
    get_submission_output_dir,
    initialize_placeholder_classification,
    load_test_dataframe,
    write_submission_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Kaggle submission.csv from predicted segmentation masks.")
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        required=True,
        help="Directory containing test_<id>.npy segmentation predictions.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=get_submission_output_dir() / "submission.csv",
        help="Path where submission.csv will be written.",
    )
    parser.add_argument(
        "--classification-fill",
        type=int,
        default=0,
        help="Placeholder value for all classification labels.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick smoke runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_df = load_test_dataframe(limit=args.limit)
    test_df = initialize_placeholder_classification(
        test_df,
        fill_value=args.classification_fill,
    )
    test_df = attach_segmentation_predictions(test_df, args.prediction_dir)
    submission_df = build_submission_dataframe(test_df)
    output_path = write_submission_csv(submission_df, args.output_path)

    print(f"submission_csv={output_path}")
    print(f"submission_rows={len(submission_df)}")


if __name__ == "__main__":
    main()
