# Adversarial Attack Findings

This file records adversarial-attack results separately from the main semantic-segmentation training logs.

## Summary Table

| Attack | Target model | Attack type | Trainable adversary | Clean mIoU | Attack mIoU | mIoU drop | Notes |
|---|---|---|---:|---:|---:|---:|---|
| V1 | V7 SegFormer-B3 | tiny learned perturbation generator | Yes | 64.720578 | 64.610856 | 0.109722 | First PDF-style trainable adversary; too weak. |
| V2 FGSM 2/255 | V7 SegFormer-B3 | FGSM white-box | No | 64.720578 | 36.305314 | 28.415264 | Strong non-trainable gradient baseline. |
| V2 FGSM 4/255 | V7 SegFormer-B3 | FGSM white-box | No | 64.720578 | 33.878626 | 30.841952 | Stronger FGSM setting. |
| V2 PGD 2/255 | V7 SegFormer-B3 | PGD white-box, 5 steps | No | 64.720578 | 19.002467 | 45.718110 | Iterative white-box attack. |
| V2 PGD 4/255 | V7 SegFormer-B3 | PGD white-box, 10 steps | No | 64.720578 | 7.917722 | 56.802856 | Strongest practical baseline at the same 4/255 budget. |
| V2 PGD 6/255 | V7 SegFormer-B3 | PGD white-box, 10 steps | No | 64.720578 | 7.856018 | 56.864560 | Slightly stronger budget, similar to PGD 4/255. |
| V3 smoke | V7 SegFormer-B3 | U-Net generator, 300 iters | Yes | 64.720578 | 63.947911 | 0.772667 | Learned generator improves over V1 but remains weak. |
| V3 full | V7 SegFormer-B3 | U-Net generator, 2000 iters | Yes | 64.720578 | 63.880034 | 0.840543 | Longer training gives only a small extra gain. |
| V4 | V10 SegMAN-B | U-Net generator design | Yes | N/A | N/A | N/A | Designed but blocked by Colab Python 3.12 dependency incompatibility. |
| V5 | V8 SegFormer-B5 | ResNet generator | Yes | 68.975346 | 66.829134 | 2.146212 | First stronger trainable generator on the B5 target. |
| V6 | V8 SegFormer-B5 | ASPP generator | Yes | 68.975346 | 64.941451 | 4.033895 | Multi-scale generator clearly improves over V5. |
| V7 | V8 SegFormer-B5 | Feature-ASPP generator | Yes | 68.975346 | 60.354332 | 8.621014 | Best trainable attack so far; adds feature-level loss. |
| V8 | V17 EoMT-DINOv3 | Feature-ASPP generator | Yes | 79.232563 | 79.599480 | -0.366917 | V7-style targeted-background + feature loss does not transfer to V17. |
| V9 | V17 EoMT-DINOv3 | CE-margin Feature-ASPP generator | Yes | 79.232563 | 79.576007 | -0.343444 | Direct CE + margin objective still fails to reduce V17 global mIoU. |

## Key Findings

- The strongest non-trainable baseline is V2 PGD:
  - PGD 4/255 reduces mIoU from `64.720578` to `7.917722`.
  - This shows the segmentation model is highly vulnerable under per-image iterative white-box optimization.
- The strongest trainable adversarial model is V7:
  - V7 reduces mIoU from `68.975346` to `60.354332`.
  - The drop of `8.621014` is more than double V6's `4.033895`.
- The trainable-adversary progression is meaningful:
  - V1 tiny generator: `0.109722` drop
  - V3 U-Net generator: `0.772667` to `0.840543` drop
  - V5 ResNet generator: `2.146212` drop
  - V6 ASPP generator: `4.033895` drop
  - V7 Feature-ASPP generator: `8.621014` drop
- V6 confirms that generator architecture matters.
- V7 confirms that attack objective design matters, not only generator capacity.
- V8 and V9 show that the V7 attack family does not directly transfer to V17 EoMT-DINOv3:
  - V8 targeted-background + feature loss gives `-0.366917` mIoU drop.
  - V9 untargeted CE + margin + feature loss gives `-0.343444` mIoU drop.
  - Negative drops mean attacked mIoU is slightly higher than clean mIoU on the 37-image V17 validation split.
- The V17 result should be interpreted carefully:
  - V17 appears substantially more robust than V8 SegFormer-B5 under these trainable generator attacks.
  - The 37-image validation split is small, so per-class damage can be offset by small improvements elsewhere.
  - EoMT is query-based, so dense-output attack objectives may be too indirect.

## Per-Class Observations

### Attack V1

Top affected classes:

| Class | Clean IoU | Attack IoU | IoU drop |
|---|---:|---:|---:|
| sheep | 50.012595 | 46.876313 | 3.136283 |
| pottedplant | 31.650976 | 30.235263 | 1.415713 |
| bus | 78.484505 | 77.689804 | 0.794701 |

V1 shows the learned generator can perturb the target, but the global effect is very small.

### Attack V3 Smoke

Top affected classes:

| Class | Clean IoU | Attack IoU | IoU drop |
|---|---:|---:|---:|
| sheep | 50.012595 | 36.766754 | 13.245841 |
| bus | 78.484505 | 75.032597 | 3.451908 |
| cat | 80.992944 | 77.684418 | 3.308525 |
| bottle | 53.942844 | 50.958756 | 2.984088 |

V3 improves over V1 mainly by using a more capable U-Net-style generator and targeted-background loss.

### Attack V3 Full

Top affected classes:

| Class | Clean IoU | Attack IoU | IoU drop |
|---|---:|---:|---:|
| sheep | 50.012595 | 38.428083 | 11.584513 |
| bus | 78.484505 | 71.710441 | 6.774064 |
| cat | 80.992944 | 76.153408 | 4.839536 |
| bird | 75.181786 | 70.342973 | 4.838813 |

The full 2000-step run is only slightly stronger globally than the 300-step smoke run.

### Attack V5

Top affected classes:

| Class | Clean IoU | Attack IoU | IoU drop |
|---|---:|---:|---:|
| sofa | 71.594797 | 58.881768 | 12.713029 |
| pottedplant | 35.018361 | 26.641197 | 8.377164 |
| cat | 86.345633 | 80.901593 | 5.444040 |
| tvmonitor | 75.297619 | 69.861263 | 5.436356 |

The ResNet generator is the first trainable adversary that produces a clearly visible global mIoU drop on the stronger SegFormer-B5 target.

### Attack V6

Top affected classes:

| Class | Clean IoU | Attack IoU | IoU drop |
|---|---:|---:|---:|
| tvmonitor | 75.297619 | 54.377188 | 20.920431 |
| sheep | 62.973650 | 44.347029 | 18.626621 |
| sofa | 71.594797 | 57.418544 | 14.176253 |
| pottedplant | 35.018361 | 26.876817 | 8.141544 |

The ASPP generator improves over V5, likely because multi-scale dilated context better attacks both small and large object structures.

### Attack V7

Top affected classes:

| Class | Clean IoU | Attack IoU | IoU drop |
|---|---:|---:|---:|
| tvmonitor | 75.297619 | 24.911344 | 50.386275 |
| cow | 78.830888 | 43.227444 | 35.603445 |
| sheep | 62.973650 | 40.652874 | 22.320776 |
| cat | 86.345633 | 70.415131 | 15.930502 |
| bird | 78.961182 | 63.326735 | 15.634447 |
| sofa | 71.594797 | 58.294027 | 13.300770 |

V7 is the best trainable attack result. The feature-level objective substantially improves the attack beyond output-level targeted-background loss alone.

### Attack V8

Top affected classes:

| Class | Clean IoU | Attack IoU | IoU drop |
|---|---:|---:|---:|
| bicycle | 27.638337 | 21.845910 | 5.792427 |
| bottle | 80.975120 | 77.408081 | 3.567040 |
| dog | 37.691163 | 35.162859 | 2.528304 |
| person | 92.460652 | 91.001641 | 1.459010 |
| train | 92.983455 | 91.678999 | 1.304457 |

V8 attacks V17 EoMT-DINOv3 using the same broad idea as V7: Feature-ASPP generator, targeted-background output loss, and feature deviation. It causes visible per-class degradation on bicycle and bottle, but overall mIoU changes from `79.232563` to `79.599480`, giving a negative global drop.

### Attack V9

Top affected classes:

| Class | Clean IoU | Attack IoU | IoU drop |
|---|---:|---:|---:|
| bicycle | 27.638337 | 22.743824 | 4.894513 |
| bottle | 80.975120 | 78.284972 | 2.690148 |
| train | 92.983455 | 91.156860 | 1.826596 |
| cow | 85.336040 | 84.064208 | 1.271832 |
| person | 92.460652 | 91.877069 | 0.583583 |

V9 keeps the Feature-ASPP generator but changes the objective to untargeted CE plus a correct-vs-wrong margin loss, with feature deviation only as an auxiliary term. This more direct objective still does not reduce global V17 mIoU: `79.232563` clean vs. `79.576007` attacked.

## Current Conclusion

For report writing, the most defensible story is:

1. V2 PGD is the strongest white-box non-trainable baseline.
2. V1 and V3 show basic trainable adversarial generators are possible but weak.
3. V5 to V7 show a systematic trainable-generator design progression.
4. V7 should be used as the main trainable adversarial model result.
5. V8 and V9 should be reported as negative but informative attempts against V17 EoMT-DINOv3.
6. Before designing another trainable V17 generator, run a V17 FGSM/PGD sanity check to verify whether V17 is attackable under direct per-image white-box optimization at the same perturbation budget.
