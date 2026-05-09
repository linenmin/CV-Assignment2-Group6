# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project

KUL H02A5a Computer Vision — Group Assignment 2. PASCAL VOC 2009 multi-label image classification (20 classes, 750 train / 750 test images stored as `.npy` numpy arrays). Final deliverable is a Kaggle submission CSV scored by Dice coefficient.

## Python Environment

All scripts require the **`biometrics` conda environment**:

```bash
/c/Users/31667/.conda/envs/biometrics/python.exe <script>
```

PyTorch (`torch==2.11.0+cu130`), torchvision `0.26`, scikit-learn, pandas, PIL, tqdm are available. `timm` is **not** installed — use `torchvision.models` for all backbones.

## Running the Pipeline

All commands are run from the **project root**:

```bash
# Full pipeline (explore → train → evaluate → predict)
python "Image classification/run_pipeline.py"

# Skip PNG export (already done), retrain from scratch
python "Image classification/run_pipeline.py" --skip-explore

# Skip training, regenerate submission from existing checkpoint
python "Image classification/run_pipeline.py" --skip-explore --skip-train

# Individual steps
python "Image classification/01_explore_data.py"   # export all .npy → PNG
python "Image classification/04_train.py"           # train model
python "Image classification/05_evaluate.py"        # compute mAP + save thresholds
python "Image classification/06_predict.py"         # generate submission_classification.csv
python "Image classification/merge_submission.py"   # merge classification + segmentation
```

Each numbered script can also be run standalone — they each contain a `main()` function and a `if __name__ == "__main__"` guard.

## Architecture

### Import pattern

Script filenames start with digits (`02_dataset.py`, etc.), so they cannot be imported with standard `import`. All cross-script imports go through `shared.load_module()`:

```python
sys.path.insert(0, str(Path(__file__).parent))
from shared import load_module
_ds = load_module("dataset", Path(__file__).parent / "02_dataset.py")
VOCDataset = _ds.VOCDataset
```

### `shared.py` — single source of truth

Contains everything shared across scripts:
- `LABELS` — the 20-class list in the exact order of `train_set.csv` columns
- `DATA_DIR`, `OUTPUT_DIR`, `PROJECT_ROOT` — path constants relative to `shared.py`
- `load_module(name, filepath)` — the importlib loader described above
- `AsymmetricLoss` — ICCV 2021 loss for multi-label with false-negative noise
- `NUM_WORKERS` — always `0` on Windows (spawn multiprocessing can't re-import dynamic modules)

### Data flow

```
kul-computer-vision-ga-2-2026/
  train/img/train_{idx}.npy  →  VOCDataset  →  DataLoader  →  MultiLabelClassifier
  train/train_set.csv        (20 binary label columns, index=Id)
  test/img/test_{idx}.npy    →  VOCDataset (no labels)
```

Images are `(H, W, 3)` uint8 arrays, variable size (~333×500). Resized to 320×320 before feeding to the model.

### Model (`03_model.py`)

`MultiLabelClassifier(backbone, num_classes, pretrained)` wraps `efficientnet_b3` (1536-dim), `convnext_tiny` (768-dim), `convnext_small` (768-dim), `convnext_base` (1024-dim), `convnext_large` (1536-dim), or `resnet50` (2048-dim). The classifier head is `Linear(feat_dim→512)→ReLU→Linear(512→20)`. Outputs raw logits — sigmoid is applied at loss/inference time, not inside `forward()`.

### Training (`04_train.py`)

Three-stage strategy:
1. **Stage 1** (5 ep, lr=1e-3): backbone frozen, train head only
2. **Stage 2** (20 ep, lr=1e-4): full fine-tune
3. **Stage 3** (5 ep, lr=5e-5): reload `best_model.pth`, retrain on all 750 samples

Saves `output/checkpoints/best_model.pth` (best val loss) and `output/checkpoints/final_model.pth` (after Stage 3).

### Inference (`06_predict.py`)

Uses `final_model.pth` if it exists, else falls back to `best_model.pth`. Applies TTA: averages sigmoid probabilities over the original image and its horizontal flip. Loads per-class thresholds from `output/best_thresholds.npy` (written by `05_evaluate.py`).

### Submission format

Kaggle requires one row per `{idx}_classification` and one row per `{idx}_segmentation`, with predictions RLE-encoded. `merge_submission.py` combines `submission_classification.csv` (from `06_predict.py`) with `submission_exp_v1_ade20k_main.csv` (external segmentation predictions) into `submission_final.csv`.

## Key Design Decisions

**`AsymmetricLoss` (not `BCEWithLogitsLoss`)**: PASCAL VOC has systematic false-negative noise — objects appear in images but are not annotated. `clip=0.05` zeroes out loss for negatives where `P < 0.05`; `gamma_neg=4` down-weights easy negatives. This specifically helps low-AP classes like `diningtable` (v1 AP=0.28 → v2 AP≈0.68 after this fix).

**No `pos_weight` in loss**: `AsymmetricLoss` handles class imbalance via focal decay; adding `pos_weight` on top would double-count the correction.

**Per-class thresholds**: `05_evaluate.py` sweeps thresholds per class to maximise F1 on the val split and saves them as `best_thresholds.npy`. Kaggle Dice = F1 for binary predictions, so this directly optimises the submission metric.

## Experiment Results

| Version | Experiment folder | Backbone | Loss | Input | val mAP | Kaggle display | Classification Dice (x2) |
|---------|-------------------|----------|------|-------|---------|----------------|--------------------------|
| v1.0 | *(pre-refactor)* | ResNet-50 | NegativeSmoothBCE | 224 | 0.801 | 0.38084 | 0.76168 |
| v1.1 | `resnet50_224` | ResNet-50 | AsymmetricLoss | 224 + TTA + Stage3 | 0.8175 | 0.39165 | 0.78330 |
| v2 | `efficientnet_b3_320` | EfficientNet-B3 | AsymmetricLoss | 320 + TTA + Stage3 | **0.8599** | 0.42813 | 0.85626 |
| v3 | `convnext_tiny_320` | ConvNeXt-Tiny | AsymmetricLoss | 320 + TTA + Stage3 | **0.8933** | **0.43673** | **0.87346** |
| v4 | `convnext_small_320` | ConvNeXt-Small | AsymmetricLoss | 320 + TTA + Stage3 | **0.8995** | **0.44905** | **0.89810** |

**About the 0.81610 score**: this is the Kaggle public score from a single complete submission containing **all 1500 rows** (750 classification + 750 segmentation). It is **not** the sum of two separate scores. Kaggle computes one Dice score across all rows together; classification-only submissions (missing segmentation rows) receive 0 for those rows, yielding a lower total.

**Classification score convention**: the Kaggle display score for
classification-only comparisons should be multiplied by 2 because the submitted
file includes both classification and segmentation rows. The table records both
the raw Kaggle display score and the adjusted classification Dice.

**ConvNeXt-Tiny result**: `convnext_tiny_320` completed local training,
evaluation, prediction, submission generation, and Kaggle scoring. It produced
`evaluation_summary.csv` with val mAP **0.8932642162** and scored **0.43673**
on Kaggle display, equivalent to **0.87346** adjusted classification Dice.

**ConvNeXt-Small result**: `convnext_small_320` is now the best classification
model. It produced local val mAP **0.8995381256** and scored **0.44905** on
Kaggle display, equivalent to **0.89810** adjusted classification Dice. The
latest complete submission using this classifier and
`submission_exp_v10_segman_b_iter25000.csv` scored **0.87588** overall.
`DEFAULT_EXPERIMENT` is now `convnext_small_320`.

**Next ConvNeXt experiments**: `torchvision 0.26` provides ImageNet-1K weights
for ConvNeXt Tiny/Small/Base/Large. The repo now registers
`convnext_small_320`, `convnext_base_320`, and `convnext_large_320`; Small is
confirmed best so far, so only try Base next if GPU time is worthwhile. Large
uses batch size 2 on the local 8 GB RTX 4060 Laptop GPU
and may be slower or more overfit-prone on this 750-image dataset.

## Planning Files

`task_plan.md`, `findings.md`, `progress.md` in the project root are persistent working notes maintained across sessions. Read them at the start of any non-trivial task to recover context.

For complex multi-step tasks (research, pipeline changes, anything requiring >5 tool calls), invoke the `/pi-planning-with-files` skill first. It creates and maintains these three files as working memory across sessions.
