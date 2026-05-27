# DM-6: Chapter 6 — Final Submission Generation

**Chapter goal**: produce `submission.csv` that the Kaggle scorer
reads. Chapter 1 already demonstrated the RLE encoder on a toy mask,
so Chapter 6 is the assembly step: load the pre-merged 1500-row CSV
from the bundled dataset, write it to the correct path (Kaggle's
`/kaggle/working/submission.csv` when running on Kaggle, a local
output path otherwise), and sanity-check the result.

## Proposed cell list (5 cells: 3 markdown + 2 light code)

| # | Type | Content |
|---|---|---|
| 6.0 | md | Chapter header + provenance of the 1500 rows (750 cls rows from ConvNeXt-Small, 750 seg rows from V17 EoMT-DINOv3 with multi-scale TTA) |
| 6.1 | code | Load the pre-merged submission CSV and write `/kaggle/working/submission.csv` (or local fallback) |
| 6.2 | md | Sanity-check description |
| 6.3 | code | Assertions: total rows, classification row count, segmentation row count, no NaN, unique Id; print one classification and one segmentation sample row |
| 6.4 | md | Closing: leaderboard score `0.89139`, how to reproduce from `dataset/weights/eomt_dinov3_v17/` plus `dataset/code/`, and why we do not run end-to-end inference inside the notebook (~20 min GPU on Kaggle) |

## Asset to add

`dataset/submissions/submission_v17_final_merged.csv` copied from `Semantic segmentation/outputs/submissions/submission_exp_v17_merge_val_tta_ms496_512_528_noflip_with_convnext_small_320.csv`. This is the 1500-row CSV that scores `0.89139` on the joint Kaggle leaderboard.

## Sub-options for user

### A. Include a reproducibility demo
- **A1 (recommended)**: no extra demo. Chapter 1 already proved the RLE round-trip works.
- A2: add a cell that loads one V17 `.npy` test prediction, runs `rle_encode`, and asserts the resulting string matches the corresponding row in the merged CSV. Cost: ~145 MB of `.npy` files added to the dataset.

### B. Sanity-check depth
- **B1 (recommended)**: five assertions (total rows, classification rows, segmentation rows, no NaN, unique Id) plus two printed sample rows plus a decoded classification vector showing the predicted class names for image 0.
- B2: minimal (total row count only).

### C. Include reproduction instructions
- **C1 (recommended)**: a short paragraph at the end of 6.4 explaining how to rebuild the submission from scratch: load `weights/eomt_dinov3_v17/best/`, run the inference helper in `code/`, then call `rle_encode` from `code/submission.py`. State explicitly that we do not run this inside the notebook because end-to-end inference on 750 images takes about 20 GPU-minutes and would push the notebook beyond the practical Kaggle runtime limit.
- C2: skip.

## Estimated effort

About 15 minutes for the CSV copy, five cells of writing, and the
Chrome slice review.

## Pending user input

- A / B / C selections.
