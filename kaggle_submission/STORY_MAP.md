# Story Map — KUL CV GA2 Group 6 Submission

This file is the **navigation entry point** for the final Kaggle
submission. The TA reads a notebook + a Kaggle dataset; the team reads
this map to know where every important artifact lives.

> All paths are relative to the project root
> `D:\BaiduNetdiskWorkspace\Leuven\8th\Computer Vision\assignment\Group2\`

## What the TA Sees

| Surface | Path | Purpose |
|---|---|---|
| Final notebook | `kaggle_submission/notebook/group6_final.ipynb` | The report. Self-contained, well-documented, generates `submission.csv`. |
| Kaggle dataset | `kaggle_submission/dataset/` | Heavy artifacts the notebook imports at runtime. |

## What the Team Sees (planning + source experiments)

| Folder | Purpose |
|---|---|
| `kaggle_submission/plan/` | pi-planning files: `task_plan.md`, `findings.md`, `progress.md`, plus per-chapter decision memos under `decisions/`. |
| `kaggle_submission/STORY_MAP.md` | This file. |
| `Semantic segmentation/` (local git branch `segmentation`) | Source of V17 EoMT segmentation model, Cityscapes evaluator, V26 adversarial attack outputs, trainable-vs-PGD visualizations. |
| `Image classification/` (local git branch `Image-Classification`, read via `git show`) | Source of ConvNeXt-Small classification model, threshold calibration, FGSM/PGD classifier attack outputs. |
| `CV_GA2.pdf` | Original assignment requirements; the spec. |
| `ga2_group_6.ipynb` | Reference starter notebook from the lecturers. |

## Kaggle Dataset Layout

The contents of `kaggle_submission/dataset/` are uploaded as one Kaggle
dataset. The notebook reads from
`/kaggle/input/<dataset-slug>/...`.

```
kaggle_submission/dataset/
├── weights/
│   └── eomt_dinov3_v17/        # 1.2 GB best segmentation checkpoint
├── predictions/
│   └── eomt_dinov3_v17/        # 750 precomputed test masks (.npy)
├── submissions/                # CSV files that the notebook re-builds
│   ├── submission_classification_convnext_small_320.csv
│   └── submission_seg_v17_with_classification.csv
├── figures/                    # all figures embedded in the notebook
├── tables/                     # CSV tables loaded and rendered as DataFrames
└── code/                       # importable modules called by the notebook
```

## Chapter -> Asset Map

This is populated as each chapter's decision memo is approved. See
`plan/decisions/DM-*.md` for the per-chapter plans.

| Chapter | Decision Memo | Status |
|---|---|---|
| 0. Title + Abstract + Summary of Results | `plan/decisions/DM-0.md` | draft v2, awaiting final verification |
| 1. Overview / Data / Submission framework | `plan/decisions/DM-1.md` | draft v3, awaiting verification |
| 2. Image Classification | `plan/decisions/DM-2.md` | draft v3 (incl. BCE ablation), awaiting verification |
| 3. Semantic Segmentation | `plan/decisions/DM-3.md` | draft v1, awaiting verification |
| 4. Adversarial Attacks | `plan/decisions/DM-4.md` | draft v1, awaiting verification |
| 5. Discussion | `plan/decisions/DM-5.md` | draft v1, awaiting verification |
| 6. Final submission cell | `plan/decisions/DM-6.md` | draft v1, awaiting verification |
| Review pass (all 6 chapters) | `plan/scripts/apply_review_edits.py` | v2 complete — PDF rubric A1-A12 + internal contracts B1-B8 all pass; from-scratch baseline measured (`val mAP 0.20` vs fine-tuned `0.90`); EoMT + trainable-vs-PGD mermaid diagrams added; References section added |

## Source-of-Truth References

| Asset | Source (where the team actually produced it) |
|---|---|
| V17 EoMT checkpoint | `Semantic segmentation/outputs/checkpoints/eomt_dinov3_v17_merge_val_lr1e5/best/` |
| V17 test predictions | `Semantic segmentation/outputs/predictions/eomt_dinov3_v17_tta_ms496_512_528_noflip/` |
| V17 inference script | `Semantic segmentation/scripts/predict_eomt_tta.py` |
| Classification submission CSV | `Semantic segmentation/outputs/submissions/submission_classification_convnext_small_320.csv` (committed on segmentation branch) |
| V17 merged submission CSV | `Semantic segmentation/outputs/submissions/submission_exp_v17_merge_val_tta_ms496_512_528_noflip_with_convnext_small_320.csv` |
| Cityscapes V17 metrics | `Semantic segmentation/outputs/cityscapes_generalization/eomt_dinov3_v17_tta_ms496_512_528_noflip/` |
| Train-vs-tram visual panels | `Semantic segmentation/outputs/figures/train_class_voc_vs_cityscapes/` |
| Segmentation attack V26 | `Semantic segmentation/outputs/attacks/attack_v26_final_report_attacks_eomt_v17/` |
| Trainable-vs-PGD visualization | `Semantic segmentation/outputs/attacks/trainable_vs_pgd_visual/` |
| Classification attack script | `Image classification/07_adversarial_attack.py` (branch `Image-Classification`) |
| Classification README narrative | `README.md` (branch `Image-Classification`) |
| Classification Discussion draft | `Discussion.md` (branch `Image-Classification`) |
