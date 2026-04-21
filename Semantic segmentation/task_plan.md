# Task Plan: Semantic Segmentation V1

## Goal

Build a maintainable `Semantic segmentation` project around `MMSegmentation + SegNeXt-S` with validated data analysis, preprocessing visualization, pretrained initialization, and a first training baseline suitable for leaderboard iteration.

## Phases

- [x] Phase 1: Research candidate models and verify official pretrained support
- [x] Phase 2: Inspect dataset structure and identify preprocessing risks
- [x] Phase 3: Review design and directory structure
- [ ] Phase 4: Implement data analysis and visualization scripts
- [ ] Phase 5: Implement custom dataset conversion / config / training entrypoints
- [ ] Phase 6: Run baseline training and validate outputs
- [ ] Phase 7: Commit, push branch, and notify via Discord

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
  Resolution: all analysis and training work must run through `conda run -n gpu_env ...` or `conda activate gpu_env`.
- `mmseg` dataset construction currently fails on Windows with `ModuleNotFoundError: No module named 'mmcv._ext'`.
  Resolution: not yet resolved in this session; current checkpoint stops before training loop startup and records the blocker for the next iteration.

## Status

**Currently in Phase 5** - Packaging the working data-analysis and config progress, with the `mmcv._ext` runtime blocker documented for the next implementation pass.
