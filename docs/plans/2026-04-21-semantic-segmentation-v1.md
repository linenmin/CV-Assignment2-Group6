# Semantic Segmentation V1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a maintainable `MMSegmentation + SegNeXt-S` semantic segmentation project with dataset analysis scripts, validated preprocessing, pretrained initialization, and a first baseline training run.

**Architecture:** The project will use a config-driven MMSeg workflow. Lightweight project-specific Python modules under `src/ga2_seg/` will own dataset metadata, split logic, metrics, and visualization. Thin scripts under `scripts/` will call those modules for analysis, training, validation, inference, and submission export.

**Tech Stack:** Python, PyTorch, CUDA, MMSegmentation, MMEngine, OpenMMLab config system, matplotlib, pandas, numpy

---

### Task 1: Create the semantic segmentation project skeleton

**Files:**
- Create: `Semantic segmentation/README.md`
- Create: `Semantic segmentation/pyproject.toml`
- Create: `Semantic segmentation/requirements.txt`
- Create: `Semantic segmentation/src/ga2_seg/__init__.py`
- Create: `Semantic segmentation/src/ga2_seg/paths.py`
- Create: `Semantic segmentation/src/ga2_seg/labels.py`
- Create: `Semantic segmentation/src/ga2_seg/split.py`
- Create: `Semantic segmentation/scripts/analyze_dataset.py`
- Create: `Semantic segmentation/scripts/visualize_samples.py`
- Create: `Semantic segmentation/scripts/build_splits.py`

**Step 1: Write a failing smoke test plan**

Define one minimal import test to verify the package can be imported and paths resolve to the assignment dataset.

**Step 2: Create the package skeleton**

Add the package files and expose:
- dataset root resolution
- class names and palette
- split file path helpers

**Step 3: Add command-line entry stubs**

Each script should parse arguments and call package functions, even before full implementation.

**Step 4: Run a smoke import**

Run:

```bash
conda run -n gpu_env python -c "from ga2_seg.paths import get_dataset_root; print(get_dataset_root())"
```

Expected: the dataset root prints successfully.

**Step 5: Commit**

```bash
git add "Semantic segmentation"
git commit -m "feat(backend): scaffold semantic segmentation project structure"
```

### Task 2: Implement dataset analysis and visualization

**Files:**
- Modify: `Semantic segmentation/src/ga2_seg/labels.py`
- Create: `Semantic segmentation/src/ga2_seg/stats.py`
- Create: `Semantic segmentation/src/ga2_seg/visualize.py`
- Modify: `Semantic segmentation/scripts/analyze_dataset.py`
- Modify: `Semantic segmentation/scripts/visualize_samples.py`
- Create: `Semantic segmentation/outputs/figures/.gitkeep`

**Step 1: Implement dataset statistics utilities**

Add functions for:
- image size distribution
- aspect ratio distribution
- class-presence counts from `train_set.csv`
- pixel-frequency counts from masks

**Step 2: Implement visualization helpers**

Add functions to render:
- raw image
- segmentation mask
- overlay
- preprocessing comparison panels

**Step 3: Write analysis outputs to disk**

Save figures and JSON summaries into `outputs/figures/` and `data/metadata/`.

**Step 4: Run the scripts**

Run:

```bash
conda run -n gpu_env python "Semantic segmentation/scripts/analyze_dataset.py"
conda run -n gpu_env python "Semantic segmentation/scripts/visualize_samples.py"
```

Expected: figures and metadata files are generated without errors.

**Step 5: Commit**

```bash
git add "Semantic segmentation"
git commit -m "feat(data): add dataset analysis and visualization scripts"
```

### Task 3: Build deterministic train/val splits

**Files:**
- Modify: `Semantic segmentation/src/ga2_seg/split.py`
- Modify: `Semantic segmentation/scripts/build_splits.py`
- Create: `Semantic segmentation/data/splits/train.txt`
- Create: `Semantic segmentation/data/splits/val.txt`
- Create: `Semantic segmentation/data/metadata/stats.json`

**Step 1: Implement split logic**

Create a deterministic split function that:
- reads all training ids
- uses a fixed seed
- saves `train.txt` and `val.txt`
- reports class-presence diagnostics for both splits

**Step 2: Validate split balance**

Ensure the validation split is not pathologically missing rare classes.

**Step 3: Run split generation**

Run:

```bash
conda run -n gpu_env python "Semantic segmentation/scripts/build_splits.py"
```

Expected: split files exist and summary stats are printed.

**Step 4: Commit**

```bash
git add "Semantic segmentation"
git commit -m "feat(data): add deterministic train-val split generation"
```

### Task 4: Create MMSeg dataset and experiment configs

**Files:**
- Create: `Semantic segmentation/configs/dataset/ga2_voc_like.py`
- Create: `Semantic segmentation/configs/model/segnext_s_ga2.py`
- Create: `Semantic segmentation/configs/runtime/default_runtime.py`
- Create: `Semantic segmentation/configs/experiments/segnext_s_512x512_adamw_poly_v1.py`
- Create: `Semantic segmentation/src/ga2_seg/mmseg_dataset.py`

**Step 1: Define dataset metadata**

Register dataset class names and palette in a custom MMSeg dataset definition.

**Step 2: Port the training and test pipelines**

Use the approved:
- random resize
- random crop `512x512`
- flip
- photometric distortion
- deterministic validation resize

**Step 3: Define the SegNeXt-S experiment**

Include:
- official pretrained backbone checkpoint
- `21` classes including background
- AdamW optimizer
- poly scheduler
- mixed precision if supported

**Step 4: Print the resolved config**

Run:

```bash
conda run -n gpu_env python -c "from mmengine.config import Config; cfg=Config.fromfile(r'Semantic segmentation/configs/experiments/segnext_s_512x512_adamw_poly_v1.py'); print(cfg.pretty_text[:4000])"
```

Expected: the config resolves without import or syntax errors.

**Step 5: Commit**

```bash
git add "Semantic segmentation"
git commit -m "feat(config): add mmseg dataset and SegNeXt-S experiment config"
```

### Task 5: Add training control, validation, and inference scripts

**Files:**
- Create: `Semantic segmentation/scripts/train.py`
- Create: `Semantic segmentation/scripts/validate.py`
- Create: `Semantic segmentation/scripts/infer.py`
- Create: `Semantic segmentation/src/ga2_seg/metrics.py`
- Create: `Semantic segmentation/src/ga2_seg/callbacks.py`

**Step 1: Implement thin script wrappers**

Each script should:
- load the experiment config
- inject project paths
- call MMSeg train/test runners

**Step 2: Add best-checkpoint and conservative early-stop logic**

Track:
- best validation `mIoU`
- latest checkpoint
- patience after warmup

**Step 3: Add validation visualization hooks**

Save a few overlays each evaluation interval for quick inspection.

**Step 4: Smoke-test one iteration**

Run a minimal sanity command such as:

```bash
conda run -n gpu_env python "Semantic segmentation/scripts/train.py" --config "Semantic segmentation/configs/experiments/segnext_s_512x512_adamw_poly_v1.py" --max-iters 2
```

Expected: one short training run completes and writes logs.

**Step 5: Commit**

```bash
git add "Semantic segmentation"
git commit -m "feat(backend): add training validation and inference entrypoints"
```

### Task 6: Run the first real baseline training

**Files:**
- Modify: `Semantic segmentation/configs/experiments/segnext_s_512x512_adamw_poly_v1.py`
- Write outputs: `Semantic segmentation/outputs/logs/`
- Write outputs: `Semantic segmentation/outputs/checkpoints/`

**Step 1: Set initial hyperparameters**

Use:
- crop size `512x512`
- batch size target `4`
- lr `6e-5`
- weight decay `1e-2`
- warmup `1500`
- eval interval `500` or `1000`
- max iters in the `20k-40k` range for first pass

**Step 2: Launch training**

Run:

```bash
conda run -n gpu_env python "Semantic segmentation/scripts/train.py" --config "Semantic segmentation/configs/experiments/segnext_s_512x512_adamw_poly_v1.py"
```

**Step 3: Review validation metrics**

Check:
- overall `mIoU`
- per-class IoU
- whether tail classes collapse
- whether overlays show background overprediction

**Step 4: Save the best checkpoint path**

Record the winning checkpoint for inference and submission.

**Step 5: Commit**

```bash
git add "Semantic segmentation"
git commit -m "feat(backend): run first SegNeXt-S baseline experiment"
```

### Task 7: Export test predictions and competition submission

**Files:**
- Create: `Semantic segmentation/scripts/export_submission.py`
- Modify: `Semantic segmentation/scripts/infer.py`
- Write outputs: `Semantic segmentation/outputs/predictions/`

**Step 1: Run inference on the test set**

Generate segmentation masks in the assignment format.

**Step 2: Convert predictions into the notebook-compatible submission format**

Map predictions back to the required dataframe / RLE export path.

**Step 3: Verify submission schema**

Check that every test example gets one segmentation prediction row.

**Step 4: Commit**

```bash
git add "Semantic segmentation"
git commit -m "feat(data): add inference export and submission generation"
```

### Task 8: Add the first ablation branch

**Files:**
- Create: `Semantic segmentation/configs/experiments/segnext_s_512x512_adamw_poly_v1_weighted_ce.py`
- Optionally modify: `Semantic segmentation/src/ga2_seg/metrics.py`

**Step 1: Duplicate the baseline config**

Create a weighted-loss experiment from the baseline.

**Step 2: Add class weighting only if baseline evidence justifies it**

Keep one variable changed at a time.

**Step 3: Train and compare**

Compare against baseline on:
- `mIoU`
- tail-class IoU
- visual quality

**Step 4: Commit**

```bash
git add "Semantic segmentation"
git commit -m "feat(config): add weighted-loss ablation for SegNeXt-S"
```
