# Kaggle Score Tracker

This file records leaderboard-facing submissions for the semantic-segmentation workflow so the team can compare local validation against Kaggle outcomes.


| Date       | Submission File                                         | Checkpoint                                                | Local Val mIoU | Kaggle Score | Classification Placeholder | Notes                                                                                                                                                 |
| ---------- | ------------------------------------------------------- | --------------------------------------------------------- | -------------- | ------------ | -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-04-21 | `outputs/submissions/submission_exp_v1_ade20k_main.csv` | `outputs/logs/exp_v1_ade20k_main/best_mIoU_iter_8000.pth` | `62.46`        | `0.36281`    | `Yes (all zero)`           | First full ADE20K-initialized run. Good local segmentation baseline, but leaderboard score is strongly limited by placeholder classification outputs. |
| 2026-04-21 | `outputs/submissions/submission_exp_v2_rare_focus.csv`  | `outputs/logs/exp_v2_rare_focus/best_mIoU_iter_4000.pth`  | `60.29`        | `0.3583`     | `Yes (all zero)`           | Second-round rare-focus split plus focused crop. Both local validation and Kaggle score regressed versus V1, which confirms the strategy was too aggressive. |
| 2026-04-22 | `outputs/submissions/submission_exp_v3_region_rebalance.csv` | `outputs/logs/exp_v3_region_rebalance/best_mIoU_iter_5000.pth` | `62.54` | `Pending` | `Yes (all zero)` | Third-round region-rebalance auxiliary branch. Local validation slightly exceeds V1 and clearly recovers from V2; waiting for manual Kaggle upload. |


## Usage Notes

- Add one row per Kaggle upload.
- Keep the `Submission File` and `Checkpoint` relative to `Semantic segmentation/` so the table remains portable.
- Record whether `classification` was real or placeholder, because it materially affects the Kaggle score.
- Use the notes column to capture the main experimental difference for each iteration.
