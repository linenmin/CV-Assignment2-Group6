# Task Plan: Semantic Segmentation V1

## Goal

Build a maintainable `Semantic segmentation` project around `MMSegmentation + SegNeXt-S` with validated data analysis, preprocessing visualization, pretrained initialization, and a first training baseline suitable for leaderboard iteration.

## Phases

- [x] Phase 1: Research candidate models and verify official pretrained support
- [x] Phase 2: Inspect dataset structure and identify preprocessing risks
- [x] Phase 3: Review design and directory structure
- [x] Phase 4: Implement data analysis and visualization scripts
- [x] Phase 5: Implement custom dataset conversion / config / training entrypoints
- [x] Phase 6: Run baseline training and validate outputs
- [x] Phase 7: Commit, push branch, and notify via Discord
- [ ] Phase 8: Commit submission-export pipeline, push branch, and launch first full training run

## Key Questions

1. Which first-pass model offers the best tradeoff between recency, stability, and maintainability?
2. Which preprocessing choices are required by the dataset shape and class imbalance?
3. Which pretrained checkpoints are allowed under the assignment constraints?
4. Which initialization should be the default first-run leaderboard baseline?

## Decisions Made

- Model family: `SegNeXt-S`
  Rationale: recent enough, strong segmentation backbone, lower engineering risk than heavier alternatives.
- Framework: `MMSegmentation`
  Rationale: mature config system, standard training scripts, easier long-term iteration.
- Main pretraining strategy: `ADE20K` segmentation checkpoint initialization
  Rationale: official checkpoint is not trained on `PASCAL VOC`, aligns with the current reading of the assignment restriction, and should transfer better than backbone-only initialization on a 749-image training set.
- Control experiment: `ImageNet-1K` pretrained MSCAN-S backbone only
  Rationale: keeps a compliant lower-assumption baseline for comparison if ADE20K initialization underperforms or raises policy questions later.
- Training control: best-checkpoint selection plus conservative early stopping
  Rationale: segmentation curves often improve late; aggressive stopping is counterproductive.
- Git workflow: work on branch `segmentation`
  Rationale: isolates semantic segmentation work from the rest of the repository.

## Errors Encountered

- Base Python environment has a broken `numpy/pandas` binary combination.
  Resolution: use the dedicated `seg_gpu_env` instead of the base environment.
- Windows training hit three framework/runtime issues:
  - missing `mmcv._ext` in the old environment
  - `OpenMP` duplicate runtime crashes in plotting
  - `mmengine` locale decode failure while collecting compiler info
  Resolution: use `seg_gpu_env` with full `mmcv`, add a Windows compatibility layer, and patch `mmengine collect_env` narrowly in project runtime code.
- Validation originally resized masks before evaluation, which broke `IoUMetric` shape alignment.
  Resolution: keep val/test samples at original resolution.
- Resource probing before the first full run showed only `2.29 GB` free RAM on the host.
  Resolution: use a conservative launch profile for the first long run with `num_workers=0`, dedicated `work_dir`, and a short pretrained sanity check before the main run.

## Status

**Currently in Phase 8** - Finalizing the submission-export pipeline commit and launching the first ADE20K-initialized training run.
