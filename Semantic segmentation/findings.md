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
- `gpu_env` is healthy and should be the only environment used for analysis and training.
- Because the dataset is small and imbalanced, the first implementation milestone should validate preprocessing visually before any long run.
- `MMSegmentation` config loading works with the current local project configs and custom imports.
- On this Windows + Python 3.12 environment, actual dataset building through `mmseg` still triggers a `mmcv._ext` import path via `mmseg.datasets`, which blocks full training startup for now.
- This means the current repository state is:
  - good enough for project structure, data analysis, visualization, split generation, and config definition
  - not yet good enough for an end-to-end `mmseg` training launch
