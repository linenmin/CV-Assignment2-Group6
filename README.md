# CV Assignment 2 - Group 6

KUL H02A5a Computer Vision, Group Assignment 2.

This repository contains the Group 6 solution code. Section 2.1 is the
multi-label image classification task on PASCAL VOC 2009.

## Task

The dataset contains 750 training images and 750 test images stored as `.npy`
arrays. For each image, the classifier predicts which of the 20 PASCAL VOC
classes are present.

Classes:

`aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train,
tvmonitor`

## Classification Layout

Shared implementation lives in `Image classification/`:

```
Image classification/
  shared.py
  01_explore_data.py
  02_dataset.py
  03_model.py
  04_train.py
  05_evaluate.py
  06_predict.py
  run_pipeline.py
  merge_submission.py
  experiments/
    efficientnet_b3_320/
      train.py
      evaluate.py
      predict.py
    convnext_tiny_320/
      train.py
      evaluate.py
      predict.py
    resnet50_224/
      train.py
      evaluate.py
      predict.py
```

Each experiment writes to its own output folder:

```
output/image_classification/<experiment>/
  checkpoints/
    best_model.pth
    last_model.pth
    final_model.pth
  metrics/
    training_history.csv
    ap_per_class.csv
    best_thresholds.npy
    evaluation_summary.csv
  figures/
    eval_ap_per_class.png
  predictions/
    test_probabilities_<experiment>.csv
    test_binary_predictions_<experiment>.csv
  submissions/
    submission_classification_<experiment>.csv
    submission_final_<experiment>__seg_<segmentation_csv_name>.csv
```

Data exploration figures are stored in:

```
output/image_classification/_data/
```

## Experiments

| Experiment | Backbone | Input | Loss | Purpose |
|------------|----------|-------|------|---------|
| `efficientnet_b3_320` | EfficientNet-B3 | 320 x 320 | AsymmetricLoss | Strong model |
| `convnext_tiny_320` | ConvNeXt-Tiny | 320 x 320 | AsymmetricLoss | Modern CNN comparison |
| `resnet50_224` | ResNet-50 | 224 x 224 | AsymmetricLoss | Baseline comparison |

## Running

Run from the project root.

```bash
# Full pipeline for the strong model
python "Image classification/run_pipeline.py" --experiment efficientnet_b3_320

# Full pipeline for ConvNeXt-Tiny
python "Image classification/run_pipeline.py" --experiment convnext_tiny_320

# Full pipeline for the ResNet baseline
python "Image classification/run_pipeline.py" --experiment resnet50_224

# Skip data exploration and train only one model
python "Image classification/run_pipeline.py" --experiment efficientnet_b3_320 --skip-explore

# Use an existing checkpoint
python "Image classification/run_pipeline.py" --experiment efficientnet_b3_320 --skip-explore --skip-train --ckpt best
```

You can also run model-specific entry points:

```bash
python "Image classification/experiments/efficientnet_b3_320/train.py"
python "Image classification/experiments/efficientnet_b3_320/evaluate.py"
python "Image classification/experiments/efficientnet_b3_320/predict.py"

python "Image classification/experiments/convnext_tiny_320/train.py"
python "Image classification/experiments/convnext_tiny_320/evaluate.py"
python "Image classification/experiments/convnext_tiny_320/predict.py"

python "Image classification/experiments/resnet50_224/train.py"
python "Image classification/experiments/resnet50_224/evaluate.py"
python "Image classification/experiments/resnet50_224/predict.py"
```

## Kaggle GPU Notebook

For Kaggle training, upload or open:

```text
Image classification/kaggle_train.ipynb
```

Enable a Kaggle GPU and run all cells. The notebook is self-contained and writes
to:

```text
/kaggle/working/image_classification/efficientnet_b3_320/
```

The classification submission file is:

```text
/kaggle/working/image_classification/efficientnet_b3_320/submissions/submission_classification_efficientnet_b3_320.csv
```

For the ConvNeXt-Tiny experiment, use:

```text
Image classification/kaggle_train_convnext_tiny_320.ipynb
```

It writes to:

```text
/kaggle/working/image_classification/convnext_tiny_320/
```

## Merging With Segmentation

After classification prediction and segmentation prediction are both available:

```bash
python "Image classification/merge_submission.py" --experiment efficientnet_b3_320 --seg-csv output/submission_exp_v7_segformer_b3.csv
```

The merged file is written to the selected experiment's `submissions/` folder.
Its default name includes the classification experiment and segmentation CSV,
for example:

```text
output/image_classification/efficientnet_b3_320/submissions/submission_final_efficientnet_b3_320__seg_submission_exp_v7_segformer_b3.csv
```

To merge a specific classification CSV, pass `--clf-csv`. The script accepts
either `submission_classification_*.csv` files or binary prediction files:

```bash
python "Image classification/merge_submission.py" --clf-csv output/image_classification/resnet50_224/predictions/test_binary_predictions_resnet50_224.csv --seg-csv output/submission_exp_v10_segman_b_iter25000.csv
```
