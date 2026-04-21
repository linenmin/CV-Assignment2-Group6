# Kaggle Score Tracker

This file records leaderboard-facing submissions for the semantic-segmentation workflow so the team can compare local validation against Kaggle outcomes.


| Date       | Submission File                                         | Checkpoint                                                | Local Val mIoU | Kaggle Score | Classification Placeholder | Notes                                                                                                                                                 |
| ---------- | ------------------------------------------------------- | --------------------------------------------------------- | -------------- | ------------ | -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-04-21 | `outputs/submissions/submission_exp_v1_ade20k_main.csv` | `outputs/logs/exp_v1_ade20k_main/best_mIoU_iter_8000.pth` | `62.46`        | `0.36281`    | `Yes (all zero)`           | First full ADE20K-initialized run. Good local segmentation baseline, but leaderboard score is strongly limited by placeholder classification outputs. |


## Usage Notes

- Add one row per Kaggle upload.
- Keep the `Submission File` and `Checkpoint` relative to `Semantic segmentation/` so the table remains portable.
- Record whether `classification` was real or placeholder, because it materially affects the Kaggle score.
- Use the notes column to capture the main experimental difference for each iteration.

