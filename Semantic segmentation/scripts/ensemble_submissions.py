import argparse
from pathlib import Path

from ga2_seg.paths import get_project_root
from ga2_seg.submission_ensemble import build_hard_vote_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a hard-vote ensemble from Kaggle submission CSVs.")
    parser.add_argument(
        "--submission",
        action="append",
        required=True,
        help="Input submission CSV. Pass multiple times in weak-to-strong order.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=get_project_root() / "outputs" / "submissions" / "submission_ensemble.csv",
        help="Output submission CSV path.",
    )
    parser.add_argument(
        "--primary-index",
        type=int,
        default=-1,
        help="Submission index used to break voting ties. Defaults to the last input.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path, stats = build_hard_vote_submission(
        submission_paths=args.submission,
        output_path=args.output_path,
        primary_index=args.primary_index,
    )
    print(f"submission_csv={output_path}")
    print(f"num_images={stats.num_images}")
    print(f"mean_changed_vs_primary={stats.mean_changed_vs_primary:.6f}")
    print(f"mean_pairwise_disagreement={stats.mean_pairwise_disagreement:.6f}")


if __name__ == "__main__":
    main()
