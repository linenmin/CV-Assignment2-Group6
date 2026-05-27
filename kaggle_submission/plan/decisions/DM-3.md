# DM-3: Chapter 3 — Semantic Segmentation

**Chapter goal**: deliver Section 2.2 of the assignment. Cover the
difference from classification, the journey across 17 segmentation
versions, the loss ablation evidence, the V17 training curve, the
per-class IoU breakdown, and a brief reflection. No retraining; every
number is loaded from precomputed CSVs and the production V17 weights
are loaded as a sanity check.

## Proposed cell list (14 total: 9 markdown + 5 light code)

| # | Type | Content |
|---|---|---|
| 3.0 | md | Chapter header + five concrete differences between classification and segmentation tasks |
| 3.1 | md | 3.2.1 Model journey: V1 SegNeXt-S baseline -> V4/V5 loss tweaks (failed) -> V6-V8 SegFormer scaling -> V10 SegMAN-B family switch -> V11-V17 EoMT-DINOv3 refinements |
| 3.2 | code | Load `segmentation_experiments.csv`, styled DataFrame highlighting V17 |
| 3.3 | md | Experiments-table interpretation: three pivots (loss tweaks did not transfer, family change was the breakthrough, merge-val capped the gains) |
| 3.4 | md | 3.2.2 Loss ablation: introduce V1 CE, V4 OHEM, V5 CE+Dice comparison |
| 3.5 | code | Load `segmentation_loss_ablation.csv`, grouped bar chart (val mIoU vs Kaggle Dice) |
| 3.6 | md | Loss-ablation interpretation: val mIoU rose but Kaggle Dice fell; 112-image val does not represent 750-image test |
| 3.7 | code | Load `segmentation_v17_training_log.csv`, plot val mIoU vs step |
| 3.8 | md | Training-curve interpretation: best at step 1000, early-stopped at 1800, plateau visible |
| 3.9 | code | V17 per-class IoU horizontal bar chart, weak classes highlighted below 0.5 |
| 3.10 | md | Per-class interpretation + cross-task pattern (same weak classes as classification) |
| 3.11 | code | Load V17 best checkpoint, report parameter count + state dict size |
| 3.12 | md | Weight summary + pretrained provenance: DINOv3 ViT-L/16 on LVD-1689M, fine-tuned on ADE20K; ADE20K is disjoint from PASCAL VOC |
| 3.13 | md | 3.2.3 Mini-reflection linking forward to Discussion chapter |

## Figures

| Figure | Source | Cell |
|---|---|---|
| Experiments comparison table (17 rows) | `segmentation_experiments.csv` | 3.2 |
| Loss ablation grouped bar chart | `segmentation_loss_ablation.csv` | 3.5 |
| V17 training curve | `segmentation_v17_training_log.csv` | 3.7 |
| Per-class IoU horizontal bar chart | `segmentation_per_class_iou.csv` | 3.9 |

## Dataset assets to add

- `dataset/tables/segmentation_experiments.csv` (V1 through V17 with val mIoU and Kaggle Dice)
- `dataset/tables/segmentation_loss_ablation.csv` (V1 CE, V4 OHEM, V5 CE+Dice)
- `dataset/tables/segmentation_per_class_iou.csv` (V17 best on validation set)
- `dataset/tables/segmentation_v17_training_log.csv` (copied from `outputs/checkpoints/.../training_log.csv`)
- `dataset/weights/eomt_dinov3_v17_merge_val_lr1e5/best/` (full HF checkpoint, ~1.2 GB)
- `dataset/code/segmentation_metrics.py` (4 helpers: load_experiments_table, load_loss_ablation, plot_loss_ablation, load_training_log, plot_training_curve, load_per_class_iou, plot_per_class_iou)
- `dataset/code/segmentation_model.py` (lightweight load_eomt_v17_summary helper)

## Existing source material (segmentation branch outputs)

- Experiments narrative: `Semantic segmentation/kaggle_scores.md` and `findings.md`
- V17 training log: `Semantic segmentation/outputs/checkpoints/eomt_dinov3_v17_merge_val_lr1e5/training_log.csv`
- V17 per-class IoU: `Semantic segmentation/outputs/predictions/tta_v17_ms496_512_528_noflip_val37/tta_validation_summary.json`
- V17 checkpoint: `Semantic segmentation/outputs/checkpoints/eomt_dinov3_v17_merge_val_lr1e5/best/`

## How the five hard rules are satisfied

- **Pretrained-Weight Provenance**: 3.12 names DINOv3 ViT-L/16 pretrained on LVD-1689M and fine-tuned on ADE20K, and states explicitly that ADE20K does not contain PASCAL VOC images.
- **Numerical-Claim Sourcing**: every mIoU / Kaggle Dice in the chapter is loaded from a CSV under `dataset/tables/`. No isolated numbers in the narrative.
- **Train / Val / Test Split Disclosure**: 3.1 documents both the original 637 / 112 split (V1 through V14) and the 712 / 37 re-split (V17). Each metric in the table indicates which split produced it.
- **Metric-Definition Disambiguation**: both the experiments table and the loss-ablation chart explicitly separate `val_mIoU` from `kaggle_dice`, with the merged-submission scope noted.
- **Loss-Ablation Pairing**: 3.4-3.6 is the dedicated ablation, contrasting V1 CE / V4 OHEM / V5 CE+Dice across both val mIoU and Kaggle Dice. The cross-class V17 per-class breakdown later in 3.9 plays the same role for the model-side choice.

## Sub-options for user

### A. V1-V17 narrative depth
- **A1 (recommended)**: one short paragraph for the arc, full 17-row table speaks for itself.
- A2: one paragraph per major version.
- A3: table only, no narrative.

### B. Loss-ablation visualisation
- **B1 (recommended)**: grouped bar chart, blue = val mIoU, orange = Kaggle Dice, side by side.
- B2: two-column table.
- B3: chart and table both.

### C. Training-curve content
- **C1 (recommended)**: single line of val mIoU vs step, the cleanest answer to "is the model learning".
- C2: 3-panel loss / mIoU / lr.
- C3: overlay V12 + V14 + V15 + V17 to make the diminishing-returns plateau visible.

### D. Per-class IoU presentation
- **D1 (recommended)**: diverging-bar chart identical in style to the classification per-class AP chart, with `IoU < 0.5` weak-class highlighting.
- D2: side-by-side comparison with classification per-class AP, exposing the cross-task weak-class pattern.
- D3: table only.

D2 is more informative (it surfaces a genuine cross-task finding) but doubles the work.

### E. Weight loading demo
- **E1 (recommended)**: same pattern as Chapter 2, load safetensors and report tensor count + model class.
- E2: skip, cite architecture in text only.

## Estimated effort

About 45 minutes including CSV compilation, two helper modules, the four
visualisations, the V17 checkpoint copy, and Chrome screenshot review.

## Pending user input

- A / B / C / D / E selections.
- OK to copy the 1.2 GB V17 best checkpoint into `dataset/weights/`?
  (Or upload it directly to the Kaggle dataset later and skip the
  local copy.)
