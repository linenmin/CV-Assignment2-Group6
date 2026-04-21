# Progress: Semantic Segmentation V1

## Session Log

### 2026-04-21

- Created branch `segmentation`.
- Confirmed the user-approved baseline direction: `MMSegmentation + SegNeXt-S`.
- Built and validated a dedicated `seg_gpu_env` for this project.
- Confirmed official `SegNeXt` sources provide:
  - `IN-1K` pretrained `MSCAN-S` backbone weights
  - `ADE20K` full segmentation checkpoints in MMSegmentation
- Recorded the current working decision:
  - main initialization: `ADE20K segmentation checkpoint`
  - control initialization: `ImageNet-1K backbone checkpoint`
- Created persistent planning files in `Semantic segmentation/`:
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
- Added the initial maintainable project skeleton:
  - `src/ga2_seg/`
  - `scripts/`
  - `tests/`
  - `configs/`
- Implemented and verified:
  - dataset path helpers
  - class metadata
  - deterministic split builder
  - dataset statistics generation
  - sample and transform visualization generation
- Installed and verified core Python dependencies used so far:
  - `pytest`
  - `mmengine`
  - `mmsegmentation`
  - `mmcv-lite`
  - `ftfy`
- Added `SegNeXt-S` experiment config using the official `ADE20K` checkpoint as the default initialization.
- Added shared runtime compatibility helpers for:
  - `mmseg` module registration
  - Windows OpenMP duplicate-runtime handling
  - Windows `mmengine collect_env` locale decoding fallback
- Added maintainable entrypoints:
  - `scripts/train.py`
  - `scripts/validate.py`
- Fixed the validation pipeline shape mismatch by removing `Resize` from the val/test pipeline.
- Verified in `seg_gpu_env`:
  - full pytest suite passes
  - `analyze_dataset.py`, `visualize_samples.py`, and `build_splits.py` run successfully
  - a one-iteration training smoke run completes and writes checkpoints
  - validation from the same training run reports metrics successfully

## Current Focus

- Package the working progress into a clean branch commit, push the branch, and notify via Discord.

## Verification Notes

- Latest verification completed in `seg_gpu_env`:
  - `python -m pytest .\\tests -v`
  - `python .\\scripts\\analyze_dataset.py`
  - `python .\\scripts\\visualize_samples.py --num-samples 2`
  - `python .\\scripts\\build_splits.py`
  - `python .\\scripts\\train.py --max-iters 1 --val-interval 1 --batch-size 1 --num-workers 0 --no-pretrained --work-dir .\\outputs\\logs\\smoke_train`
