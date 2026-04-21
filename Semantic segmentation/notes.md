# Notes: Semantic Segmentation V1 Research

## Dataset Findings

### Source: local dataset inspection

- Path: `kul-computer-vision-ga-2-2026/`
- Training examples: `749`
- Test examples: read from `test/test_set.csv`
- Image and mask shapes are variable, with many samples around `375x500`, `333x500`, and `500x375`.
- Example arrays:
  - image: `uint8`, shape like `(333, 500, 3)`, range `0..255`
  - mask: `uint8`, shape like `(333, 500)`, values in `0..20`
- Image-level class distribution is long-tailed:
  - high-frequency examples include `person`, `chair`, `car`
  - low-frequency examples include `sheep`, `cow`
- Pixel-level imbalance is severe:
  - background pixels: about `77.74%`
  - all foreground pixels together: about `22.26%`

## Official Model Sources

### Source: SegNeXt official repository

- URL: https://github.com/Visual-Attention-Network/SegNeXt
- Key points:
  - official PyTorch implementation
  - repository states that released backbones are `ImageNet` pretrained
  - README explicitly says `ImageNet Pre-trained models can be found in TsingHua Cloud`
  - official results table marks `MSCAN-S` as `IN-1K` pretrained

### Source: MMSegmentation SegNeXt configs

- URL: https://github.com/open-mmlab/mmsegmentation/tree/main/configs/segnext
- Key points:
  - official config for `SegNeXt MSCAN-S` is available
  - config uses `LightHamHead` with `MSCAN-S`
  - config references a converted pretrained backbone checkpoint:
    `https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_s_20230227-f33ccdf2.pth`
  - README notes that OpenMMLab converted official weights and they can be used directly for training

### Source: MMSegmentation Pascal VOC base dataset config

- URL: https://github.com/open-mmlab/mmsegmentation/blob/main/configs/_base_/datasets/pascal_voc12.py
- Key points:
  - standard VOC pipeline uses `RandomResize`, `RandomCrop`, `RandomFlip`, `PhotoMetricDistortion`
  - crop size is `512x512`
  - `cat_max_ratio=0.75` is used in random crop to avoid overly empty crops

## Constraints and Implications

- Avoid any segmentation checkpoint pretrained on `PASCAL VOC`.
- Prefer an `ImageNet-1K` pretrained backbone and train the decode head for this assignment dataset.
- Because the dataset is small and imbalanced, the first iteration should prioritize:
  - stable train/val split
  - per-class diagnostics
  - visual inspection of masks and transforms
  - conservative regularization and augmentation

## Recommended First Baseline

- Framework: `MMSegmentation`
- Model: `SegNeXt-S`
- Pretrained init: `MSCAN-S ImageNet-1K` backbone checkpoint
- Training focus:
  - validate preprocessing visually before full training
  - start with official AdamW + poly schedule style
  - adapt batch size and learning rate to single `RTX 4060 16GB`
  - save best checkpoint on validation `mIoU`
