# DM-2: Chapter 2 — Image Classification

**Chapter goal**: deliver Section 2.1 of the assignment. Cover backbone
choice, loss design, the three-stage training recipe, the per-class
threshold and TTA tricks, the final numbers, and a clean honest section
on what the model still misses. No retraining inside the notebook; we
load the team's pre-trained ConvNeXt-Small weights from the bundled
Kaggle dataset.

## Proposed cell list (9 total: 6 markdown + 3 light code)

| # | Type | Content |
|---|---|---|
| 2.1 | md | Chapter header + task statement (20 classes + background, multi-label binary classification) |
| 2.2 | md | 2.1.1 Approach: backbone rationale (ConvNeXt-Small 50M, Base/Large overfit, ViT lacks inductive bias on small data) |
| 2.3 | md | 2.1.2 Loss design: AsymmetricLoss with clip=0.05, gamma_neg=4; directly attacks false-negative annotation noise |
| 2.4 | md | 2.1.3 Training recipe: three stages (head warmup -> full fine-tune -> all-data retrain), per-class threshold, hflip TTA |
| 2.5 | code (light) | Load `tables/classification_experiments.csv` and render styled DataFrame |
| 2.6 | code (light) | Load `tables/classification_per_class_ap.csv` and plot horizontal bar chart (ascending by AP) |
| 2.7 | code (light) | Load `weights/convnext_small_320/final_model.pth`, print param count, confirm state_dict keys |
| 2.8 | md | 2.1.4 What the model misses: diningtable noise / pottedplant scale / sheep long-tail |
| 2.9 | md | 2.1.5 Mini-reflection linking forward to Chapter 5 discussion |

## Figures

| Figure | Source | Cell |
|---|---|---|
| Experiments comparison table | live load CSV -> pandas `.style` | 2.5 |
| Per-class AP horizontal bar chart | live render from CSV (<15 lines matplotlib) | 2.6 |

## Tables to add under `dataset/tables/`

- `classification_experiments.csv` columns: `experiment_name, backbone,
  loss, input_size, val_map, kaggle_dice, notes`. Rows: ResNet-50,
  EfficientNet-B3, ConvNeXt-Tiny, ConvNeXt-Small, ConvNeXt-Base,
  ConvNeXt V2-Tiny, EfficientNet-V2-S, ViT-B/16 (selected from
  classification README).
- `classification_per_class_ap.csv` columns: `class_name, ap,
  optimal_threshold`. 20 rows for the final ConvNeXt-Small model.

## Modules to add under `dataset/code/`

| Module | Functions |
|---|---|
| `classification_metrics.py` | `load_experiments_table(path)`, `load_per_class_ap(path)`, `plot_per_class_ap(df, ax)` |
| `classification_model.py` | `load_convnext_small_weights(path)` -- light loader that `torch.load`s the state_dict and reports the parameter count without rebuilding the model. |

## Pre-trained weight location (team-side setup)

Notebook expects:

```
kaggle_submission/dataset/weights/convnext_small_320/
└── final_model.pth
```

This path holds both on Kaggle (under `/kaggle/input/<slug>/weights/...`)
and locally. The team must download the file from Zhenyang's Google
Drive into the local path before re-rendering the notebook outputs.
Once `STORY_MAP.md` records the exact Drive URL, every contributor
mirrors the file to the same relative location.

**Pending user input**: confirm the Drive path for classification
weights (or whether to ask Zhenyang to upload to the existing team
folder `https://drive.google.com/drive/u/1/folders/1ImiIyTvxtMVIlrptKXfl3w3qF5CBdPCG/`).

## Text tone

- Approach / loss / training paragraphs ~80 words each, factual.
- Each design choice gets one sentence motivating it and one sentence
  citing the metric it moved.
- Mistakes paragraph ~100 words, three weak classes, one explanation
  each.
- Mini-reflection ~50 words, points to Chapter 5.

## Sub-options for user

### A. Weight loading demo depth
- **A1 (recommended)**: `torch.load` only, print param count and
  state_dict shape check. No backbone import needed.
- A2: rebuild `MultiLabelClassifier`, `load_state_dict`, run one
  sample prediction. Most convincing but pulls in torchvision ConvNeXt.
- A3: skip the demo, text only.

### B. Experiments comparison presentation
- **B1 (recommended)**: styled pandas DataFrame, bold the chosen row.
- B2: static markdown table.
- B3: bar chart of val mAP / Kaggle Dice across backbones.

### C. Per-class AP visualisation
- **C1 (recommended)**: horizontal bar chart sorted ascending, weak
  classes immediately visible.
- C2: table + colour-coded weak rows.
- C3: chart + table both.

### D. Weight download instructions location
- **D1 (recommended)**: in `STORY_MAP.md` team-side setup section,
  notebook stays code-only.
- D2: small markdown "setup checklist" box at start of Chapter 2.
- D3: inline comment in the loader cell.

## Estimated effort

~30 minutes including CSV authoring, helper module writing, weight
loading sanity test, Chrome screenshot self-review.

## Deliverables after approval

1. Two CSV tables in `dataset/tables/`.
2. Two helper modules in `dataset/code/`.
3. 9 cells appended to `notebook/group6_final.ipynb` after the existing
   Chapter 1 cells.
4. `STORY_MAP.md` updated with weight download path (placeholder until
   Drive URL is provided).
5. Self-review screenshot, plan files updated, Discord ping.
