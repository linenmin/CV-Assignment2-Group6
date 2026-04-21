# Progress: Semantic Segmentation V1

## Session Log

### 2026-04-21

- Created branch `segmentation`.
- Confirmed the user-approved baseline direction: `MMSegmentation + SegNeXt-S`.
- Verified `gpu_env` is the correct execution environment for this project.
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
- Confirmed current blocker:
  - `Config.fromfile(...)` works
  - `mmseg` dataset build still fails on Windows because `mmseg.datasets` imports `mmcv.ops`, which expects `mmcv._ext`

## Current Focus

- Package the working progress into a clean branch commit, while clearly recording the current `mmcv._ext` blocker for the next iteration.

## Verification Notes

- Fresh verification required before commit and push:
  - full pytest suite in `Semantic segmentation/tests`
  - smoke runs of `analyze_dataset.py`, `visualize_samples.py`, and `build_splits.py`
