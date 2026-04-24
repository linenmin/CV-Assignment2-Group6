# CV Assignment 2 — Group 6

KUL H02A5a Computer Vision, Group Assignment 2.
Multi-label image classification on PASCAL VOC 2009 (20 classes) using PyTorch transfer learning.

## Task

Given 750 training images and 750 test images stored as `.npy` numpy arrays, predict which of 20 PASCAL VOC object categories are present in each image. Predictions are evaluated on Kaggle using the Dice coefficient (equivalent to F1 for binary predictions).

**Classes:** aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor

## Results

| Version | Backbone | Loss | Input | val mAP | Kaggle (classification) | Kaggle (full submission) |
|---------|----------|------|-------|---------|------------------------|--------------------------|
| v1 | ResNet-50 | NegativeSmoothBCE | 224×224 | 0.801 | 0.381 | — |
| v2 | EfficientNet-B3 | AsymmetricLoss | 320×320 + TTA | **0.956** | **0.430** | **0.816** |

The full submission score (0.816) is computed by Kaggle across all 1500 rows (750 classification + 750 segmentation) in a single submission — it is not the sum of two separate scores.

## Setup

All scripts require the `biometrics` conda environment:

```bash
conda activate biometrics
# PyTorch 2.11.0+cu130, torchvision 0.26, scikit-learn, pandas, PIL, tqdm
# timm is NOT installed — torchvision.models is used for all backbones
```

## Running

All commands are run from the **project root**:

```bash
# Full pipeline from scratch (~40–50 min on a local GPU)
python "Image classification/run_pipeline.py"

# PNG export already done, retrain from scratch
python "Image classification/run_pipeline.py" --skip-explore

# Use existing checkpoint, skip training
python "Image classification/run_pipeline.py" --skip-explore --skip-train

# Individual steps
python "Image classification/04_train.py"       # train → output/checkpoints/
python "Image classification/05_evaluate.py"    # val mAP + best_thresholds.npy
python "Image classification/06_predict.py"     # submission_classification.csv
python "Image classification/merge_submission.py"  # merge with segmentation → submission_final.csv
```

## Architecture

```
Image classification/
├── shared.py            # LABELS, DATA_DIR, OUTPUT_DIR, load_module(), AsymmetricLoss
├── 01_explore_data.py   # Data stats + export all .npy → PNG
├── 02_dataset.py        # VOCDataset, 320×320 transforms, RandomErasing
├── 03_model.py          # MultiLabelClassifier (EfficientNet-B3 / ResNet-50)
├── 04_train.py          # Three-stage training
├── 05_evaluate.py       # mAP evaluation + per-class threshold search
├── 06_predict.py        # TTA inference → submission CSV
├── merge_submission.py  # Combine classification + segmentation submissions
└── run_pipeline.py      # End-to-end orchestration script
```

### Model

`MultiLabelClassifier` wraps EfficientNet-B3 (default) or ResNet-50 from `torchvision.models`. The head is `Dropout(0.4) → Linear(1536→512) → ReLU → Dropout(0.2) → Linear(512→20)`. The model outputs raw logits; sigmoid is applied at loss/inference time.

### Three-Stage Training

| Stage | Backbone | Epochs | LR | Purpose |
|-------|----------|--------|----|---------|
| 1 | Frozen | 5 | 1e-3 | Warm up classification head |
| 2 | Unfrozen | 20 | 1e-4 | Full fine-tuning, saves `best_model.pth` |
| 3 | Unfrozen | 5 | 5e-5 | Retrain on all 750 samples → `final_model.pth` |

### Key Design Decisions

**AsymmetricLoss (ICCV 2021)** replaces BCE. PASCAL VOC has systematic false-negative noise — objects appear in images but are unannotated (e.g., people in the background of a "bicycle" image). `clip=0.05` zeros out the loss for negatives where `P < 0.05`; `gamma_neg=4` down-weights easy negatives. This improved the hardest class (diningtable) from AP=0.28 to AP≈0.68.

**Per-class thresholds** are searched on the validation split to maximise per-class F1, then saved to `output/best_thresholds.npy`. Kaggle Dice = F1, so this directly optimises the submission metric.

**TTA** (Test-Time Augmentation): averages sigmoid probabilities over the original image and its horizontal flip at inference time, with no training cost.

## Output Files

| File | Description |
|------|-------------|
| `output/checkpoints/best_model.pth` | Best checkpoint by val loss (after Stage 2) |
| `output/checkpoints/final_model.pth` | Checkpoint after Stage 3 full-data retrain (used for submission) |
| `output/best_thresholds.npy` | Per-class optimal F1 thresholds |
| `output/submission_classification.csv` | Classification predictions (RLE-encoded) |
| `output/submission_final.csv` | Final merged submission (classification + segmentation) |
| `output/eval_ap_per_class.png` | Per-class AP bar chart |

## Potential Improvements

- **K-fold cross-validation** — val set is only 150 images; k-fold gives more robust thresholds
- **MixUp / CutMix** — strong regularisation for small datasets
- **Larger backbone** — EfficientNet-B4/B5 if GPU memory allows
- **Ensemble** — average probabilities from multiple checkpoints or backbones
