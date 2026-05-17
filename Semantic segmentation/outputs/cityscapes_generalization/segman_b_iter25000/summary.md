# Cityscapes External Generalization Summary (SegMAN-B V10)

- Source: `checkpoint=external/SegMAN/.../best_mIoU_iter_25000.pth; Cityscapes val 500 samples`
- Primary metric: 5-class transferable mIoU (excludes `train`): **`0.4168`**
- Secondary: 6-class overlap mIoU (includes `train`, kept for backward comparison): `0.3484`

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

- `bicycle`: IoU `0.0382`, target pixels `6504475`, union `6507274`
- `bus`: IoU `0.5092`, target pixels `3563120`, union `4970397`
- `car`: IoU `0.8249`, target pixels `59731217`, union `60395987`
- `motorbike`: IoU `0.1874`, target pixels `729415`, union `761721`
- `person`: IoU `0.5241`, target pixels `11913424`, union `12125509`
- `train`: IoU `0.0066`, target pixels `1031648`, union `1058982`  (excluded)
