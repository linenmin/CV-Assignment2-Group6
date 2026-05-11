# Cityscapes External Generalization Summary

- Samples evaluated: `500`
- Dataset roots: `leftImg8bit=/mnt/g/Datasets/leftImg8bit_trainvaltest/leftImg8bit; gtFine=/mnt/g/Datasets/gtFine_trainvaltest/gtFine; split=val`
- Metric: `overlap-class mIoU`
- Overlap mIoU: `0.3484`

This is not a full VOC-20 evaluation. Non-overlap Cityscapes classes are ignored.

## Class Mapping

- Cityscapes `24` -> VOC `person` (`15`)
- Cityscapes `26` -> VOC `car` (`7`)
- Cityscapes `28` -> VOC `bus` (`6`)
- Cityscapes `31` -> VOC `train` (`19`)
- Cityscapes `32` -> VOC `motorbike` (`14`)
- Cityscapes `33` -> VOC `bicycle` (`2`)

## Per-Class IoU

- `bicycle`: IoU `0.0382`, target pixels `6504475`, union `6507274`
- `bus`: IoU `0.5092`, target pixels `3563120`, union `4970397`
- `car`: IoU `0.8249`, target pixels `59731217`, union `60395987`
- `motorbike`: IoU `0.1874`, target pixels `729415`, union `761721`
- `person`: IoU `0.5241`, target pixels `11913424`, union `12125509`
- `train`: IoU `0.0066`, target pixels `1031648`, union `1058982`
