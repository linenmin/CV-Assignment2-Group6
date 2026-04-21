# Semantic Segmentation V1 Design

## Goal

Build a maintainable semantic segmentation workspace for Group Assignment 2 that supports fast leaderboard iteration, starts from a recent official model, and validates the data pipeline before any long training run.

## Chosen Direction

- Framework: `MMSegmentation`
- Baseline model: `SegNeXt-S`
- Pretraining: `ADE20K` segmentation checkpoint initialization
- Execution environment: `conda activate gpu_env`

This direction is chosen because the dataset is small, imbalanced, and spatially irregular. `SegNeXt-S` is recent enough, stronger than old CNN baselines, and easier to maintain than heavier segmentation stacks. `MMSegmentation` gives a clean config-driven workflow, standardized training scripts, and good long-term extensibility.

## Why Not the Other Main Options

### `SegFormer-B2`

Still a valid second candidate, but less attractive as the first pass here. On this dataset we want a model that is robust on small data and straightforward to stabilize. `SegNeXt-S` is the lower-risk first baseline.

### `Mask2Former`

Too heavy for the first iteration. It is stronger as a later experiment but has higher engineering cost and more moving parts.

## Assignment-Safe Pretraining Policy

We should not use a checkpoint that has already been trained for segmentation on `PASCAL VOC`. The safest compliant route is:

- use an `ImageNet-1K` pretrained `MSCAN-S` backbone
- initialize the segmentation decode head from scratch
- fine-tune on the assignment dataset only

Official evidence:

- SegNeXt official repository states `MSCAN-S` uses `IN-1K` pretraining.
- MMSegmentation publishes a converted pretrained backbone checkpoint for direct training use.

## Dataset Risks That Must Drive the Pipeline

### 1. Variable image sizes

The dataset does not use a single fixed resolution. Shapes cluster around `375x500`, `333x500`, and `500x375`, but many outliers exist. The pipeline must therefore standardize geometry using `RandomResize` and `RandomCrop` for training, and deterministic resize for validation/inference.

### 2. Heavy background dominance

Background occupies about `77.74%` of all pixels. This creates a real risk that an apparently stable model learns to over-predict background.

### 3. Long-tail classes

Classes such as `sheep` and `cow` are much rarer than `person`. Validation must report per-class IoU and not only aggregate metrics.

## Proposed Project Layout

The root folder should stay narrow and scriptable. Put the actual project under the existing `Semantic segmentation/` directory.

```text
Semantic segmentation/
  task_plan.md
  notes.md
  README.md
  pyproject.toml
  requirements.txt
  data/
    splits/
      train.txt
      val.txt
    metadata/
      class_names.json
      palette.json
      stats.json
  configs/
    _base_/
    dataset/
      ga2_voc_like.py
    model/
      segnext_s_ga2.py
    runtime/
      default_runtime.py
    experiments/
      segnext_s_512x512_adamw_poly_v1.py
      segnext_s_512x512_adamw_poly_v1_weighted_ce.py
  src/
    ga2_seg/
      __init__.py
      paths.py
      labels.py
      split.py
      mmseg_dataset.py
      metrics.py
      visualize.py
      callbacks.py
  scripts/
    analyze_dataset.py
    visualize_samples.py
    build_splits.py
    train.py
    validate.py
    infer.py
    export_submission.py
  outputs/
    figures/
    checkpoints/
    predictions/
    logs/
  notebooks/
    exploration.ipynb
```

## Data Analysis and Visualization Deliverables

Before training, generate and save the following outputs:

1. image size and aspect ratio histograms
2. class-presence bar chart from `train_set.csv`
3. pixel-frequency bar chart from segmentation masks
4. random sample grid with raw image, mask, and overlay
5. augmentation sanity check grid:
   raw, resized, cropped, flipped, photometric-distorted
6. train/val split diagnostics:
   class presence counts and mean mask area by class

These artifacts should go to `outputs/figures/` and be reproducible via scripts.

## First Training Strategy

### Model

- backbone: `MSCAN-S`
- decode head: `LightHamHead`
- classes: `21` total if background is explicit, or `20 + background` mapped to the dataset convention
- input crop: `512x512`

### Initialization

- initialize backbone from the official `ImageNet-1K` pretrained `MSCAN-S` checkpoint
- initialize decode head randomly

### Dataset Split

- create a fixed train/val split from the `749` training samples
- start with `85/15` split
- save split files to `data/splits/`
- keep the split deterministic with a recorded random seed

### Training Pipeline

Adapted from official MMSeg Pascal VOC and SegNeXt configs:

- `LoadImageFromFile`
- `LoadAnnotations`
- `RandomResize`
  - base scale around `(2048, 512)` in MMSeg VOC is too wide for our custom pipeline literal copy
  - for this project, use a resize policy that preserves aspect ratio and targets a `512` crop later
  - practical first-pass choice: target long side in the `512..1024` band
- `RandomCrop(crop_size=(512, 512), cat_max_ratio=0.75)`
- `RandomFlip(prob=0.5)`
- `PhotoMetricDistortion`
- optional normalization through MMSeg defaults
- `PackSegInputs`

### Validation / Inference Pipeline

- deterministic resize preserving aspect ratio
- no random crop
- whole-image inference first
- add sliding-window inference only if whole-image memory or quality becomes a problem

### Optimizer and Schedule

Start close to the official `SegNeXt-S` MMSeg config, but adapt for a single `RTX 4060 16GB`.

- optimizer: `AdamW`
- base learning rate: start from `6e-5` if effective batch size stays close to official scaling
- betas: `(0.9, 0.999)`
- weight decay: `1e-2`
- parameter-wise rules:
  - no decay on norm layers
  - no decay on positional blocks if retained from official config
  - `head` learning-rate multiplier `10x`
- scheduler: `poly`
- warmup: linear
- warmup iters: `1500`
- warmup ratio: `1e-6`
- power: `1.0`

### Batch Size and Runtime

First-pass assumption for `16 GB` VRAM:

- `batch_size = 4` or `6`
- mixed precision enabled
- gradient accumulation only if needed
- `num_workers = 4`

We should profile a single forward pass before locking the batch size.

### Loss

Start with:

- `CrossEntropyLoss`

Planned second-pass upgrades:

- class-weighted cross entropy if tail classes collapse
- CE + Dice hybrid if boundary quality and minority-class recall lag

The first version should stay closer to the official baseline unless the initial diagnostics already show catastrophic imbalance.

### Validation Metrics

- primary: `mIoU`
- secondary:
  - per-class IoU
  - mean accuracy
  - qualitative overlays on validation images

### Checkpointing and Early Stopping

Do not use aggressive early stopping.

Recommended control logic:

- validate every fixed interval
- save `best_mIoU` checkpoint
- save latest checkpoint
- enable early-stopping patience only after warmup and enough validation points
- suggested first-pass patience:
  - wait at least `5` validation evaluations before stopping logic starts
  - stop only after `6-8` consecutive non-improving validations

This is intentionally conservative because segmentation often improves late.

## Baseline Hyperparameter Draft

This is the concrete first run to build around:

- model: `SegNeXt-S`
- pretrained backbone checkpoint:
  `https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_s_20230227-f33ccdf2.pth`
- crop size: `512x512`
- batch size: target `4`
- optimizer: `AdamW`
- lr: `6e-5`
- weight decay: `1e-2`
- warmup iters: `1500`
- scheduler: `poly`
- max iters: start in the `20k-40k` range for this smaller dataset rather than blindly copying `160k`
- eval interval: `500` or `1000` iterations
- checkpoint interval: same as eval interval
- seed: fixed and logged
- amp: enabled

## Expected V1 Milestones

1. reproducible dataset analysis outputs
2. fixed train/val split saved to disk
3. working MMSeg custom dataset config
4. successful pretrained initialization
5. successful single-batch sanity run
6. first full training run with validation curves
7. best checkpoint inference and submission export

## Review Criteria Before Implementation

The design is good enough to start coding if you agree with the following:

- `SegNeXt-S` is the V1 baseline
- the default initialization uses the official `ADE20K` SegNeXt-S checkpoint
- the `ImageNet-1K` backbone-only initialization remains as a control experiment
- the project is built around a `scripts + src + configs + outputs` layout
- preprocessing validation is mandatory before full training
- the first training run optimizes stability and observability, not maximum complexity
