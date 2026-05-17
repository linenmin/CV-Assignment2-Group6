# Cityscapes External Generalization Summary (EoMT-DINOv3 V17)

- Samples evaluated: `500`
- Dataset roots: `leftImg8bit=G:\Datasets\leftImg8bit_trainvaltest\leftImg8bit; gtFine=G:\Datasets\gtFine_trainvaltest\gtFine; split=val`
- Checkpoint: `D:\BaiduNetdiskWorkspace\Leuven\8th\Computer Vision\assignment\Group2\Semantic segmentation\outputs\checkpoints\eomt_dinov3_v17_merge_val_lr1e5\best`
- Inference: `scales=[496, 512, 528] ref=512 hflip=False`
- Metric: `overlap-class mIoU`
- Overlap mIoU: `0.4253`

This is not a full VOC-20 evaluation. Non-overlap Cityscapes classes are ignored.

## Class Mapping

- Cityscapes `24` -> VOC `person` (`15`)
- Cityscapes `26` -> VOC `car` (`7`)
- Cityscapes `28` -> VOC `bus` (`6`)
- Cityscapes `31` -> VOC `train` (`19`)
- Cityscapes `32` -> VOC `motorbike` (`14`)
- Cityscapes `33` -> VOC `bicycle` (`2`)

## Per-Class IoU

- `bicycle`: IoU `0.1375`, target pixels `6504475`, union `6561009`
- `bus`: IoU `0.5264`, target pixels `3563120`, union `5747621`
- `car`: IoU `0.8728`, target pixels `59731217`, union `59838281`
- `motorbike`: IoU `0.4389`, target pixels `729415`, union `744984`
- `person`: IoU `0.5685`, target pixels `11913424`, union `12221724`
- `train`: IoU `0.0079`, target pixels `1031648`, union `1031648`
