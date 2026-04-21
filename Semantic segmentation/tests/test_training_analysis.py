import json
from pathlib import Path

from ga2_seg.training_analysis import (
    build_training_summary,
    find_run_artifacts,
    load_scalar_records,
    lookup_kaggle_score,
)


def test_load_scalar_records_splits_train_and_val_metrics(tmp_path):
    scalars_path = tmp_path / "scalars.json"
    records = [
        {"iter": 50, "step": 50, "loss": 1.2, "decode.acc_seg": 70.0, "lr": 1e-4},
        {"iter": 100, "step": 100, "loss": 0.9, "decode.acc_seg": 75.0, "lr": 1e-4},
        {"step": 100, "mIoU": 40.5, "mAcc": 50.0, "aAcc": 88.0},
    ]
    scalars_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    train_df, val_df = load_scalar_records(scalars_path)

    assert train_df["step"].tolist() == [50, 100]
    assert val_df["step"].tolist() == [100]
    assert val_df["mIoU"].tolist() == [40.5]


def test_find_run_artifacts_resolves_latest_run_directory(tmp_path):
    work_dir = tmp_path / "exp"
    run_dir = work_dir / "20260421_190406"
    vis_dir = run_dir / "vis_data"
    vis_dir.mkdir(parents=True)
    (vis_dir / "scalars.json").write_text("", encoding="utf-8")
    (run_dir / "20260421_190406.log").write_text("", encoding="utf-8")
    (work_dir / "analysis").mkdir(parents=True)

    artifacts = find_run_artifacts(work_dir)

    assert artifacts.work_dir == work_dir
    assert artifacts.run_dir == run_dir
    assert artifacts.scalars_path == vis_dir / "scalars.json"
    assert artifacts.log_path == run_dir / "20260421_190406.log"
    assert artifacts.analysis_dir == work_dir / "analysis"


def test_lookup_kaggle_score_matches_submission_filename(tmp_path):
    tracker_path = tmp_path / "kaggle_scores.md"
    tracker_path.write_text(
        "\n".join(
            [
                "# Kaggle Score Tracker",
                "",
                "| Date | Submission File | Checkpoint | Local Val mIoU | Kaggle Score | Classification Placeholder | Notes |",
                "| --- | --- | --- | --- | --- | --- | --- |",
                "| 2026-04-21 | `outputs/submissions/submission_exp_v1_ade20k_main.csv` | `outputs/logs/exp_v1_ade20k_main/best_mIoU_iter_8000.pth` | `62.46` | `0.36281` | `Yes (all zero)` | baseline |",
            ]
        ),
        encoding="utf-8",
    )

    matched = lookup_kaggle_score(
        tracker_path,
        submission_filename="submission_exp_v1_ade20k_main.csv",
    )

    assert matched is not None
    assert matched["kaggle_score"] == "0.36281"
    assert matched["local_val_miou"] == "62.46"


def test_build_training_summary_reports_best_and_stop_steps():
    summary = build_training_summary(
        val_records=[
            {"step": 1000, "mIoU": 36.53, "mAcc": 41.75, "aAcc": 88.2},
            {"step": 8000, "mIoU": 62.46, "mAcc": 77.8, "aAcc": 92.66},
            {"step": 14000, "mIoU": 60.39, "mAcc": 70.4, "aAcc": 92.51},
        ],
        submission_filename="submission_exp_v1_ade20k_main.csv",
        checkpoint_name="best_mIoU_iter_8000.pth",
        kaggle_entry={
            "kaggle_score": "0.36281",
            "classification_placeholder": "Yes (all zero)",
        },
    )

    assert summary["best_step"] == 8000
    assert summary["best_miou"] == 62.46
    assert summary["stop_step"] == 14000
    assert summary["kaggle_score"] == "0.36281"
