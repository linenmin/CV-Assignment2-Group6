# DM-1: Chapter 1 — Overview / Data / Submission framework

**Chapter goal**: establish the shared baseline every later chapter
relies on. State what the dataset is, what intrinsic challenges it has,
and what counts as a valid Kaggle submission. Do **not** anticipate
task-specific design here; that is the job of Chapters 2-4.

## Proposed cell list

| # | Type | Content |
|---|---|---|
| 1.1 | markdown | Section header + dataset background (PASCAL VOC 2009 subset, 749 train / 750 test, 20 classes + background, two derived tasks). |
| 1.2 | code (light) | Imports, Kaggle dataset paths, `sys.path` injection of `dataset/code/`. ~10 lines, English comments. |
| 1.3 | code (light) | Call `data_overview.compute_class_distribution()` and plot a horizontal bar chart inline. Long tail visible at a glance. |
| 1.4 | code (light) | Call `data_overview.show_sample_images_with_masks()` to load and display a 4-image grid with masks (pre-rendered PNG asset). |
| 1.5 | markdown | Three data observations (long tail / image-size variance / label noise), each pointing forward to the chapter where it informed a design choice. |
| 1.6 | markdown | Kaggle submission format: 1500 rows = 750 images x (classification + segmentation) rows; classification row encodes present classes, segmentation row uses RLE; both scored by Dice. |
| 1.7 | code (light) | Import `submission.rle_encode` / `rle_decode`, run a round-trip on a 4x4 toy mask, print the RLE string. |

7 cells total: 4 markdown + 3 light code.

## Figures

| Figure | Source | Cell |
|---|---|---|
| Class-count bar chart | rendered inline from `train_set.csv` | 1.3 |
| 4-image sample grid (image + mask) | pre-rendered PNG at `dataset/figures/sample_images_with_masks.png` | 1.4 |

Rationale for the split: the class-count data is tiny (20 rows), so
in-notebook rendering reinforces that the data is real and the notebook
runs. The sample grid requires reading 4 `.npy` arrays and colourising
masks; pre-rendering keeps the Kaggle execution fast.

## Tables

None in Chapter 1. The bar chart is the data-overview surface; tables
appear in Chapter 2 (classification per-class AP) and Chapter 3
(segmentation per-class IoU).

## Modules to add under `dataset/code/`

| Module | Functions | Used by |
|---|---|---|
| `data_overview.py` | `compute_class_distribution(data_dir)`, `plot_class_distribution(counts, ax)`, `show_sample_images_with_masks(figure_path, ax)` | cells 1.3, 1.4 |
| `submission.py` | `rle_encode(mask)`, `rle_decode(rle_str, shape)` | cell 1.7, and Chapter 6 final submission. Port of the verified implementation in `Semantic segmentation/src/ga2_seg/submission.py`. |

## Text tone

- Background paragraph ~80 words, factual.
- Three observations: one short sentence per item, plus one sentence
  pointing forward to the chapter where the choice paid off.
  Example: "Class `person` appears 207 times while `sheep` appears 27;
  this imbalance motivates the AsymmetricLoss choice in Section 2.1."
- Submission format paragraph ~100 words, references the
  `submit_df_to_competition` helper from the starter notebook.
- Total markdown ~300-400 words.

## Sub-options for user

### A. Number of data observations
- **A1 (recommended)**: 3 (long tail / size variance / label noise).
  Each maps to a concrete design choice later.
- A2: 5 (add train-vs-test distribution check, mean 1.43 classes per
  image). Broader but dilutes the focus.
- A3: 1 + figure (only long tail, with the figure carrying the rest).

### B. Sample-image grid size
- **B1 (recommended)**: 4 images (2x2). Fits on one screen.
- B2: 6 images (2x3). More class coverage but cramped layout.
- B3: 2 images (1x2). Minimal, may under-represent the variety.

### C. RLE encoding demonstration
- **C1 (recommended)**: 4x4 toy mask round-trip. Proves the
  encode/decode pair is invertible without depending on any model
  output.
- C2: use a real predicted mask (truncated to first 60 chars). More
  realistic but couples Chapter 1 to Chapter 3 ordering.
- C3: skip the demo, explain in text only.

## Estimated effort

~25 minutes including module authoring, figure pre-rendering, local
test run, Chrome screenshot self-review.

## Deliverables after approval

1. 7 cells appended to `notebook/group6_final.ipynb`.
2. `dataset/figures/sample_images_with_masks.png` rendered once.
3. `dataset/code/data_overview.py` and `dataset/code/submission.py`
   added and unit-tested against a local data path.
4. Local-vs-Kaggle path indirection inside cell 1.2 so the same
   notebook runs in both environments.
5. Updated `STORY_MAP.md` chapter status to draft v1.
6. Discord ping when ready for verification.
