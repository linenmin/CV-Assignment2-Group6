# Findings: Semantic Segmentation V1

## Dataset Findings

### Source: local dataset inspection

- Path: `kul-computer-vision-ga-2-2026/`
- Training examples: `749`
- Image and mask shapes are variable, with major modes around `375x500`, `333x500`, and `500x375`
- Example image arrays are `uint8` RGB with values in `0..255`
- Example mask arrays are `uint8` with class ids in `0..20`
- Pixel-level imbalance is severe:
  - background pixels: about `77.74%`
  - foreground pixels: about `22.26%`
- Image-level labels are long-tailed:
  - common examples: `person`, `chair`, `car`
  - rare examples: `sheep`, `cow`

## Model Research Findings

### Source: SegNeXt official repository

- URL: https://github.com/Visual-Attention-Network/SegNeXt
- The official results table marks `MSCAN-S` as `IN-1K` pretrained.
- The repository README explicitly states that `ImageNet Pre-trained models` are provided.
- The same repository reports leaderboard results on `Pascal VOC`, but those are separate downstream results, not proof that the released `IN-1K` backbone weights touched VOC.

### Source: MMSegmentation SegNeXt configs

- URL: https://github.com/open-mmlab/mmsegmentation/tree/main/configs/segnext
- Official `SegNeXt MSCAN-S` config exists for `ADE20K`.
- The config points to the converted backbone pretrained checkpoint:
  `https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_s_20230227-f33ccdf2.pth`
- MMSeg also publishes the full `SegNeXt-S ADE20K` segmentation checkpoint in the SegNeXt model zoo.

## Policy Interpretation Findings

- The assignment restriction is about avoiding models pretrained on `PASCAL VOC`.
- Current official evidence does not show the `ADE20K` SegNeXt checkpoint being trained on `PASCAL VOC`.
- Therefore, the current working assumption is:
  - `ADE20K segmentation checkpoint` is allowed
  - `ImageNet-1K backbone checkpoint` is also allowed
- This should still be treated as an interpretation of the assignment, not a TA confirmation.

## Engineering Findings

- The base Python environment is broken for `numpy/pandas`.
- A dedicated `seg_gpu_env` is the stable execution environment for this project.
- `seg_gpu_env` requires:
  - `python 3.11`
  - full `mmcv 2.1.0` wheel matching `torch 2.1 + cu121`
  - `numpy<2`
  - `setuptools<81`
- Because the dataset is small and imbalanced, the first implementation milestone should validate preprocessing visually before any long run.
- `MMSegmentation` config loading works with the local project configs and custom imports.
- The repository now contains a shared runtime helper that:
  - registers `mmseg` plus project-specific modules
  - patches the Windows `mmengine collect_env` locale issue
  - gives consistent behavior for tests and CLI scripts
- Windows-specific runtime issues resolved in this session:
  - `mmcv._ext` blocker avoided by moving to `seg_gpu_env` with full `mmcv`
  - `OpenMP` duplicate runtime crash handled by a small compatibility layer
  - `mmengine` compiler-probe decode failure handled by a narrow runtime patch
- Validation/test pipeline should not resize masks before evaluation on this dataset.
  - Keeping original spatial resolution avoids `pred/label` shape mismatches during `IoUMetric`.
- The repository now has a local path to generate a Kaggle-ready `submission.csv`.
  - `classification` rows can be exported with placeholder defaults.
  - `segmentation` rows are encoded from per-image predicted masks using the same RLE scheme as the teacher notebook.
- Host resource probe before the first long run:
  - available RAM was low (`2.29 GB`)
  - GPU memory headroom was moderate (about `5.6 GB` free on the `RTX 4060 Laptop GPU`)
  - this host should prefer `num_workers=0` and conservative launch settings for long training jobs
- The official `ADE20K` SegNeXt-S checkpoint has now been exercised end-to-end in a short sanity run.
  - checkpoint download and initialization succeeded
  - a 2-iteration training run produced a valid best checkpoint
  - this removes the main remaining uncertainty before the first long run
- The project now contains a local delayed early-stopping hook for iterative training.
  - default policy: wait for `5` validations before activation
  - stop only after `6` non-improving `mIoU` validations with `min_delta=0.1`
  - this matches the agreed training strategy better than relying on `save_best` alone
- The first full `ADE20K`-initialized run completed successfully on this host.
  - best validation checkpoint: `best_mIoU_iter_8000.pth`
  - best validation score: `mIoU=62.46`
  - training stopped at `14000` iterations due to the configured delayed early stopping
- The first Kaggle submission from this run scored `0.36281`.
  - submission source: `submission_exp_v1_ade20k_main.csv`
  - this score should not be interpreted as a pure segmentation ceiling because `classification` was still exported as all-zero placeholder values
  - the leaderboard gap is therefore partly a modeling gap and partly an intentionally incomplete submission pipeline
- Current repository state is good enough for:
  - project structure
  - data analysis and visualization
  - split generation
  - config definition
  - one-iteration training and validation smoke runs
  - test-time inference over the whole test set
  - `submission.csv` generation for manual Kaggle upload
