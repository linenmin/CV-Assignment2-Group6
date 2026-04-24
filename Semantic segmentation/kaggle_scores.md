# Kaggle Score Tracker

This file records leaderboard-facing submissions for the semantic-segmentation workflow so the team can compare local validation against Kaggle outcomes.


| Date       | Submission File                                         | Checkpoint                                                | Local Val mIoU | Kaggle Score | Classification Placeholder | Notes                                                                                                                                                 |
| ---------- | ------------------------------------------------------- | --------------------------------------------------------- | -------------- | ------------ | -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-04-21 | `outputs/submissions/submission_exp_v1_ade20k_main.csv` | `outputs/logs/exp_v1_ade20k_main/best_mIoU_iter_8000.pth` | `62.46`        | `0.36281`    | `Yes (all zero)`           | First full ADE20K-initialized run. Good local segmentation baseline, but leaderboard score is strongly limited by placeholder classification outputs. |
| 2026-04-21 | `outputs/submissions/submission_exp_v2_rare_focus.csv`  | `outputs/logs/exp_v2_rare_focus/best_mIoU_iter_4000.pth`  | `60.29`        | `0.3583`     | `Yes (all zero)`           | Second-round rare-focus split plus focused crop. Both local validation and Kaggle score regressed versus V1, which confirms the strategy was too aggressive. |
| 2026-04-22 | `outputs/submissions/submission_exp_v3_region_rebalance.csv` | `outputs/logs/exp_v3_region_rebalance/best_mIoU_iter_5000.pth` | `62.54` | `0.36011` | `Yes (all zero)` | Third-round region-rebalance auxiliary branch. Local validation slightly exceeds V1, but Kaggle still stays below V1, so the local gain did not transfer cleanly to the hidden test distribution. |
| 2026-04-23 | `outputs/submissions/submission_exp_v4_ohem.csv` | `outputs/logs/exp_v4_ohem/best_mIoU_iter_13000.pth` | `63.90` | `0.35237` | `Yes (all zero)` | Fourth-round OHEM pixel sampling. Local validation improved clearly, but the public score regressed, so the gain did not transfer to the leaderboard submission. |
| 2026-04-23 | `outputs/submissions/submission_exp_v5_ce_dice.csv` | `outputs/logs/exp_v5_ce_dice/best_mIoU_iter_7000.pth` | `63.62` | `0.35553` | `Yes (all zero)` | Fifth-round CE + Dice loss. Local validation stayed strong and beat V1/V3, but public score still stayed below the original V1 baseline. |
| 2026-04-24 | `outputs/submissions/submission_exp_v6_segformer_b2.csv` | `outputs/logs/exp_v6_segformer_b2/best_mIoU_iter_14000.pth` | `63.46` | `0.37348` | `Yes (all zero)` | Sixth-round SegFormer-B2 baseline. Switching model family improved leaderboard transfer noticeably even though local validation stayed close to the stronger SegNeXt variants. |
| 2026-04-24 | `outputs/submissions/submission_exp_v7_segformer_b3.csv` | `outputs/logs/exp_v7_segformer_b3/best_mIoU_iter_8000.pth` | `64.72` | `0.38084` | `Yes (all zero)` | Seventh-round SegFormer-B3 scale-up. Both local validation and Kaggle score improved over V6, confirming the new model family generalized better on the hidden test set. |
| 2026-04-24 | `outputs/submissions/submission_exp_v8_segformer_b5.csv` | `outputs/logs/exp_v8_segformer_b5/best_mIoU_iter_11000.pth` | `68.97` | `0.38567` | `Yes (all zero)` | Eighth-round SegFormer-B5 scale-up. This is the current best single-model segmentation submission and the strongest public leaderboard result so far. |
| 2026-04-24 | `outputs/submissions/submission_exp_v9_segformer_vote_v6_v7_v8.csv` | `N/A (submission-level ensemble)` | `N/A` | `0.39064` | `Yes (all zero)` | Ninth-round hard-vote ensemble from V6/V7/V8 SegFormer submissions, using V8 as the tie-break baseline. This is the current best public leaderboard result. |


## Usage Notes

- Add one row per Kaggle upload.
- Keep the `Submission File` and `Checkpoint` relative to `Semantic segmentation/` so the table remains portable.
- Record whether `classification` was real or placeholder, because it materially affects the Kaggle score.
- Use the notes column to capture the main experimental difference for each iteration.
