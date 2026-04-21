import argparse
from pathlib import Path

from ga2_seg.paths import get_project_root
from ga2_seg.training_analysis import (
    build_training_summary,
    find_run_artifacts,
    infer_best_checkpoint_name,
    load_scalar_records,
    lookup_kaggle_score,
    plot_runtime_diagnostics,
    plot_training_curves,
    write_summary_markdown,
    write_validation_metrics_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots and summaries for a training run.")
    parser.add_argument("--work-dir", type=str, required=True, help="Training work directory.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for analysis outputs.")
    parser.add_argument(
        "--score-tracker",
        type=str,
        default=None,
        help="Optional Kaggle score tracker markdown file.",
    )
    parser.add_argument(
        "--submission-file",
        type=str,
        default=None,
        help="Optional submission filename for score lookup.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=None,
        help="Optional checkpoint name override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = find_run_artifacts(args.work_dir)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else artifacts.analysis_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_scalar_records(artifacts.scalars_path)
    checkpoint_name = args.checkpoint_name or infer_best_checkpoint_name(artifacts.work_dir)
    score_tracker = (
        Path(args.score_tracker).resolve()
        if args.score_tracker
        else get_project_root() / "kaggle_scores.md"
    )
    kaggle_entry = lookup_kaggle_score(
        score_tracker,
        submission_filename=args.submission_file,
        checkpoint_name=checkpoint_name,
    )
    summary = build_training_summary(
        val_records=val_df.to_dict(orient="records"),
        submission_filename=args.submission_file,
        checkpoint_name=checkpoint_name,
        kaggle_entry=kaggle_entry,
    )

    curves_path = output_dir / "training_curves.png"
    diagnostics_path = output_dir / "runtime_diagnostics.png"
    metrics_path = output_dir / "validation_metrics.csv"
    summary_path = output_dir / "summary.md"

    plot_training_curves(train_df, val_df, curves_path)
    plot_runtime_diagnostics(train_df, diagnostics_path)
    write_validation_metrics_csv(val_df, metrics_path)
    write_summary_markdown(summary, summary_path)

    print(f"analysis_dir={output_dir}")
    print(f"training_curves={curves_path}")
    print(f"runtime_diagnostics={diagnostics_path}")
    print(f"validation_metrics={metrics_path}")
    print(f"summary_markdown={summary_path}")


if __name__ == "__main__":
    main()
