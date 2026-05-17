# Cityscapes External Generalization Summary (EoMT-DINOv3 V17)

- Source: `checkpoint=eomt_dinov3_v17_merge_val_lr1e5/best; scales=[496,512,528] hflip=False; Cityscapes val 500 samples`
- Primary metric: 5-class transferable mIoU (excludes `train`): **`0.5088`**
- Secondary: 6-class overlap mIoU (includes `train`, kept for backward comparison): `0.4253`

`train` is excluded from the primary metric because VOC trains and
Cityscapes 'trains' are essentially different visual concepts: VOC trains
are full-frame intercity / steam locomotives photographed as subject,
while Cityscapes 'trains' are urban trams that appear small and
incidentally in street scenes. See
`outputs/figures/train_class_voc_vs_cityscapes/` for visual evidence.

## Class Mapping

- Cityscapes `24` -> VOC `person` (`15`)
- Cityscapes `26` -> VOC `car` (`7`)
- Cityscapes `28` -> VOC `bus` (`6`)
- Cityscapes `31` -> VOC `train` (`19`)  (excluded from primary metric)
- Cityscapes `32` -> VOC `motorbike` (`14`)
- Cityscapes `33` -> VOC `bicycle` (`2`)

## Per-Class IoU

- `bicycle`: IoU `0.1375`, target pixels `6504475`, union `6561009`
- `bus`: IoU `0.5264`, target pixels `3563120`, union `5747621`
- `car`: IoU `0.8728`, target pixels `59731217`, union `59838281`
- `motorbike`: IoU `0.4389`, target pixels `729415`, union `744984`
- `person`: IoU `0.5685`, target pixels `11913424`, union `12221724`
- `train`: IoU `0.0079`, target pixels `1031648`, union `1031648`  (excluded)
