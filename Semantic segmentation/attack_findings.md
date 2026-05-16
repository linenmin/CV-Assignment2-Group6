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
| V10 FGSM 2/255 | V17 EoMT-DINOv3 | FGSM white-box | No | 79.232563 | 49.506473 | 29.726090 | Direct gradient baseline confirms V17 is attackable. |
| V10 FGSM 4/255 | V17 EoMT-DINOv3 | FGSM white-box | No | 79.232563 | 45.792824 | 33.439740 | Strong one-step attack on V17. |
| V10 PGD 4/255 | V17 EoMT-DINOv3 | PGD white-box, 10 steps | No | 79.232563 | 4.777270 | 74.455294 | Very strong V17 white-box diagnostic; not trainable. |
| V11 | V17 EoMT-DINOv3 | PGD-distilled Feature-ASPP generator | Yes | 79.232563 | 79.243165 | -0.010602 | PGD final-delta distillation produces too little effective perturbation. |
| V12 | V17 EoMT-DINOv3 | query-level Feature-ASPP generator | Yes | 79.232563 | 79.997189 | -0.764625 | Query-level EoMT objective still fails to reduce global mIoU. |>>>>>>> 6dffe60 (Document adversarial attack experiments)| V12 | V17 EoMT-DINOv3 | query-level Feature-ASPP generator | Yes | 79.232563 | 79.997189 | -0.764625 | Query-level EoMT objective still fails to reduce global mIoU. |
| V13 | V17 EoMT-DINOv3 | PGD-teacher Feature-ASPP generator | Yes | 79.232563 | 79.247728 | -0.015164 | PGD-teacher objective still collapses to tiny ineffective perturbations. |
| V14 | V17 EoMT-DINOv3 | Transformer generator, targeted background | Yes | 79.232563 | 78.974004 | 0.258559 | First V17 transformer generator with small positive drop, but visible stripe artifacts. |
| V15 | V17 EoMT-DINOv3 | DINO feature-conditioned Transformer generator | Yes | 79.232563 | 79.883066 | -0.650503 | Feature conditioning did not help; global mIoU increases. |
| V16 | V17 EoMT-DINOv3 | SEA-inspired Transformer targeted-background loss | Yes | 79.232563 | 78.949332 | 0.283231 | Best V17 targeted-background trainable result, but still very weak. |
| V17 | V17 EoMT-DINOv3 | Transformer generator, target person | Yes | 79.232563 | 79.503825 | -0.271262 | Targeting person is ineffective. |
| V18 | V17 EoMT-DINOv3 | Data-driven target-class Transformer generator | Yes | 79.232563 | 79.523864 | -0.291301 | Data-driven selection chooses person and remains ineffective. |
| V19 | V17 EoMT-DINOv3 | Per-image latent Transformer generator, untargeted | Yes | 79.232563 | 79.316299 | -0.083735 | Mechanism-level change; lowers GT probability and affects local classes but not global mIoU. |
| V20 | V17 EoMT-DINOv3 | CosPGD-inspired Transformer generator objective | Yes | 79.232563 | 79.260748 | -0.028184 | Objective-level attempt; valid experiment, but ineffective on V17. |
| V21 | V17 EoMT-DINOv3 | Target-mask distillation generator | Yes | 79.232563 | 79.092610 | 0.139953 | Trainable mask-level target attempt; measurable but weak. |
| V22 PGD 4/255 | V17 EoMT-DINOv3 | per-image PGD, 10 steps | No | 79.232563 | 5.404918 | 73.827646 | Reproducible per-image PGD baseline. |
| V22 CosPGD 4/255 | V17 EoMT-DINOv3 | per-image CosPGD-inspired PGD, 10 steps | No | 79.232563 | 4.639892 | 74.592671 | Stronger than PGD on pixel accuracy and GT probability drop. |
| V23 Momentum PGD 4/255 | V17 EoMT-DINOv3 | Momentum PGD, 10 steps | No | 79.232563 | 4.569797 | 74.662766 | Slightly stronger than V23 PGD. |
| V23 Universal PGD 4/255 | V17 EoMT-DINOv3 | universal perturbation | No | 79.232563 | 79.573316 | -0.340752 | Ineffective as a reusable universal perturbation under this setup. |
| V24 PGD 6/255 | V17 EoMT-DINOv3 | stronger per-image PGD, 10 steps | No | 79.232563 | 3.531095 | 75.701469 | Higher epsilon gives the strongest untargeted PGD mIoU drop among V24 runs. |
| V24 non-person FG -> person | V17 EoMT-DINOv3 | targeted PGD, 4/255, 10 steps | No | 79.232563 | 9.892714 | 69.339849 | First clean semantic target-class attack story; target match 0.919196. |
| V25 non-person FG -> person | V17 EoMT-DINOv3 | targeted PGD, 6/255, 20 steps | No | 79.232563 | 6.059096 | 73.173467 | Best targeted-person run so far; target match 0.993359. |
| V26 | V17 EoMT-DINOv3 | final report rerun notebook | No | 79.232563 | TBD | TBD | Consolidates historical reruns and final visual evidence for best untargeted/targeted attacks. |
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
<<<<<<< HEAD>>>>>>> e078db8 (Add semantic PGD attack report artifacts)
- V8 and V9 show that the V7 attack family does not directly transfer to V17 EoMT-DINOv3:
  - V8 targeted-background + feature loss gives `-0.366917` mIoU drop.
  - V9 untargeted CE + margin + feature loss gives `-0.343444` mIoU drop.
  - Negative drops mean attacked mIoU is slightly higher than clean mIoU on the 37-image V17 validation split.
- V10 confirms that V17 itself is not immune to white-box attacks:
  - FGSM 4/255 reduces V17 mIoU from `79.232563` to `45.792824`.
  - PGD 4/255 with 10 steps reduces V17 mIoU to `4.777270`.
  - Therefore, the failure of V8/V9/V11/V12 is a trainable-generator/objective bottleneck, not proof that V17 is generally robust.
- V11 and V12 are negative trainable-generator attempts on V17:
  - V11 PGD-distilled generator gives `-0.010602` mIoU drop.
  - V12 query-level generator gives `-0.764625` mIoU drop.
  - Both keep the image-space `eps=4/255` perturbation budget.
<<<<<<< HEAD
=======
- V13 confirms that a second PGD-teacher attempt still does not transfer PGD's strength into a trainable generator:
  - V13 gives `-0.015164` mIoU drop and very small mean perturbation.
- V14 to V18 test Transformer-based trainable attacks on V17:
  - V14 targeted-background Transformer gives a small positive drop of `0.258559`, but adversarial images show visible stripe artifacts.
  - V15 DINO feature-conditioned Transformer gives `-0.650503` drop, so feature conditioning did not help.
  - V16 SEA-inspired targeted-background Transformer gives the best V17 targeted-background trainable result so far: `0.283231` drop.
  - V17 target-person Transformer gives `-0.271262` drop.
  - V18 data-driven target selection chooses `person` and gives `-0.291301` drop.
- V19 changes the attack mechanism rather than only target/loss:
  - Previous trainable V17 attacks mostly use a universal generator, `delta = G(image)`.
  - V19 uses per-image latent adaptation, `delta_i = G(image_i, z_i)`.
  - V19 still does not reduce global mIoU: `79.232563` clean vs. `79.316299` attacked.
  - It does reduce mean GT probability by `0.009674` and changes `0.002282` of valid pixels, so it creates measurable attack pressure even when argmax mIoU does not fall.
- V20 and V21 close the trainable-generator branch:
  - V20 tries a CosPGD-inspired objective in a Transformer generator but gives `-0.028184` mIoU drop.
  - V21 tries target-mask distillation and gives only `0.139953` mIoU drop.
  - These runs support the conclusion that direct per-image optimization is currently much stronger than trainable generator attacks on V17.
- V22 to V25 define the final white-box attack evidence:
  - V22 shows CosPGD-inspired per-image optimization is stronger than plain PGD on several confidence/change metrics.
  - V23 shows Momentum PGD is slightly stronger than plain PGD, while Universal Perturbation is ineffective.
  - V24 shows `eps=6/255` and foreground-focused losses can further reduce mIoU.
  - V25 gives the cleanest targeted semantic task: perturb images so non-person foreground regions are segmented as `person`.
- V26 is the final report notebook:
  - it reruns the historical comparisons instead of relying only on manually copied tables;
  - it keeps final visual evidence focused on `cospgd_untargeted_eps6_255_steps20` and `pgd_target_non_person_fg_to_person_eps6_255_steps20`;
  - targeted visual examples are filtered so the ground truth contains no `person`, making the task definition closer to the classification team's no-target-class setup.- The V17 result should be interpreted carefully:
  - V17 appears substantially more robust than V8 SegFormer-B5 under these trainable generator attacks.
  - The 37-image validation split is small, so per-class damage can be offset by small improvements elsewhere.
  - EoMT is query-based, so dense-output attack objectives may be too indirect.
=======## Per-Class Observations

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

### Attack V10

V10 is a direct white-box diagnostic on V17 EoMT-DINOv3, not a trainable adversarial model.

| Condition | Clean mIoU | Attack mIoU | mIoU drop | max delta | mean delta |
|---|---:|---:|---:|---:|---:|
| FGSM 2/255 | 79.232563 | 49.506473 | 29.726090 | 2.0 | 1.984618 |
| FGSM 4/255 | 79.232563 | 45.792824 | 33.439740 | 4.0 | 3.957107 |
| PGD 4/255, 10 steps | 79.232563 | 4.777270 | 74.455294 | 4.0 | 2.305559 |

V10 is important because it proves V17 can be attacked when the perturbation is optimized per image. This separates target-model robustness from trainable-generator weakness.

### Attack V11

Top affected classes:

| Class | Clean IoU | Attack IoU | IoU drop |
|---|---:|---:|---:|
| pottedplant | 36.828596 | 36.789139 | 0.039457 |
| bottle | 80.975120 | 80.944800 | 0.030320 |
| horse | 88.991211 | 88.967874 | 0.023337 |
| person | 92.460652 | 92.450478 | 0.010174 |
| boat | 70.750845 | 70.743825 | 0.007020 |

V11 tries to distill PGD into a reusable Feature-ASPP generator. It fails globally: `79.232563` clean vs. `79.243165` attacked. The learned perturbation has very small mean magnitude (`0.094721` pixel), so the generator does not reproduce the strong PGD attack.

### Attack V12

Top affected classes:

| Class | Clean IoU | Attack IoU | IoU drop |
|---|---:|---:|---:|
| bicycle | 27.638337 | 25.193608 | 2.444729 |
| bottle | 80.975120 | 79.929822 | 1.045298 |
| tvmonitor | 89.671692 | 88.688258 | 0.983434 |
| dog | 37.691163 | 37.175043 | 0.516119 |
| cow | 85.336040 | 84.904602 | 0.431439 |

V12 attacks EoMT's query outputs more directly, including query no-object, foreground-class, mask-suppression, entropy, dense CE, and dense margin losses. It still does not reduce global V17 mIoU: `79.232563` clean vs. `79.997189` attacked.
>>>>>>> 6dffe60 (Document adversarial attack experiments)
### Attack V13

Top affected classes:

| Class | Clean IoU | Attack IoU | IoU drop |
|---|---:|---:|---:|
| horse | 88.991211 | 88.965215 | 0.025996 |
| boat | 70.750845 | 70.728224 | 0.022621 |
| bottle | 80.975120 | 80.956335 | 0.018785 |
| bird | 95.105845 | 95.098092 | 0.007753 |
| train | 92.983455 | 92.976675 | 0.006781 |

V13 uses a PGD-teacher Feature-ASPP setup. It fails globally: `79.232563` clean vs. `79.247728` attacked, with mIoU drop `-0.015164`.

### Attack V14

Top affected classes:

| Class | Clean IoU | Attack IoU | IoU drop |
|---|---:|---:|---:|
| boat | 70.750845 | 67.167778 | 3.583067 |
| bicycle | 27.638337 | 24.401537 | 3.236800 |
| aeroplane | 95.205407 | 92.441946 | 2.763461 |
| bus | 96.180344 | 93.500352 | 2.679992 |
| bottle | 80.975120 | 78.618056 | 2.357065 |

V14 introduces a Transformer generator for V17 targeted-background attack. It gives a small positive global drop: `79.232563` clean vs. `78.974004` attacked. However, adversarial images showed visible horizontal stripe artifacts.

### Attack V15

Top affected classes:

| Class | Clean IoU | Attack IoU | IoU drop |
|---|---:|---:|---:|
| bottle | 80.975120 | 77.473260 | 3.501860 |
| dog | 37.691163 | 36.266721 | 1.424442 |
| chair | 77.175400 | 75.897151 | 1.278249 |
| cat | 92.419893 | 91.261992 | 1.157901 |
| tvmonitor | 89.671692 | 89.034578 | 0.637114 |

V15 conditions the Transformer generator on frozen V17/DINO features and uses a smoother decoder. This does not improve attack effectiveness: mIoU changes from `79.232563` to `79.883066`.

### Attack V16

Top affected classes:

| Class | Clean IoU | Attack IoU | IoU drop |
|---|---:|---:|---:|
| boat | 70.750845 | 68.774799 | 1.976046 |
| pottedplant | 36.828596 | 34.887232 | 1.941364 |
| tvmonitor | 89.671692 | 88.912917 | 0.758775 |
| dog | 37.691163 | 36.988364 | 0.702798 |
| sofa | 74.873516 | 74.481626 | 0.391890 |

V16 keeps the Transformer generator but changes the objective to a SEA-inspired targeted-background loss. It gives the strongest V17 targeted-background trainable result so far, but the effect remains small: mIoU drop `0.283231`.

### Attack V17

Top affected classes:

| Class | Clean IoU | Attack IoU | IoU drop |
|---|---:|---:|---:|
| bottle | 80.975120 | 80.580976 | 0.394144 |
| tvmonitor | 89.671692 | 89.283818 | 0.387874 |
| pottedplant | 36.828596 | 36.503709 | 0.324887 |
| cat | 92.419893 | 92.191465 | 0.228427 |
| horse | 88.991211 | 88.837923 | 0.153288 |

V17 manually targets `person`. The non-person foreground to person rate increases only from `0.001545` to `0.001673`, and global mIoU increases from `79.232563` to `79.503825`.

### Attack V18

Top affected classes:

| Class | Clean IoU | Attack IoU | IoU drop |
|---|---:|---:|---:|
| chair | 77.175400 | 76.219620 | 0.955780 |
| bottle | 80.975120 | 80.365075 | 0.610045 |
| cow | 85.336040 | 85.038963 | 0.297077 |
| person | 92.460652 | 92.169681 | 0.290970 |
| car | 91.115991 | 90.939262 | 0.176730 |

V18 selects a target class from clean validation confusion statistics. The selected class is again `person`. The non-target foreground to selected-target rate increases only from `0.001545` to `0.001694`, and global mIoU increases to `79.523864`.

### Attack V19

V19 is the first per-image latent trainable generator attempt:

```text
delta_i = G(image_i, z_i)
```

Summary metrics:

| Metric | Clean | Attack | Drop / Change |
|---|---:|---:|---:|
| mIoU | 79.232563 | 79.316299 | -0.083735 |
| foreground mIoU | 78.286862 | 78.374661 | -0.087799 |
| pixel accuracy | 97.574698 | 97.571821 | 0.002876 |
| mean GT probability | 0.753120 | 0.743446 | 0.009674 |
| prediction change rate | 0.000000 | 0.002282 | 0.002282 |
| max delta pixel | 0.0 | 4.0 | 4.0 |
| mean delta pixel | 0.000000 | 2.454069 | 2.454069 |

Top affected classes:

| Class | Clean IoU | Attack IoU | Per-class IoU drop |
|---|---:|---:|---:|
| pottedplant | 36.828596 | 35.214925 | 1.613671 |
| bottle | 80.975120 | 79.972988 | 1.002132 |
| chair | 77.175400 | 76.350998 | 0.824402 |
| horse | 88.991211 | 88.416465 | 0.574746 |
| cow | 85.336040 | 84.873441 | 0.462599 |
| person | 92.460652 | 92.046820 | 0.413832 |

V19 does not reduce global mIoU, but it does show partial attack pressure:

- mean GT probability drops by `0.009674`;
- prediction change rate reaches `0.002282`;
- pottedplant, bottle, and chair show the largest local per-class damage.

This means per-image latent adaptation is a meaningful mechanism change, but the current setting is still not strong enough to create broad argmax-level segmentation failure on V17.
## Current Conclusion

For report writing, the most defensible story is:

1. V2 PGD is the strongest white-box non-trainable baseline.
2. V1 and V3 show basic trainable adversarial generators are possible but weak.
3. V5 to V7 show a systematic trainable-generator design progression.
4. V7 should be used as the main trainable adversarial model result.
<<<<<<< HEAD
5. V8, V9, V11, and V12 should be reported as negative but informative trainable-generator attempts against V17 EoMT-DINOv3.
6. V10 should be reported as the V17 white-box diagnostic showing that V17 is attackable under direct per-image FGSM/PGD.
7. The next trainable V17 attempt should use PGD as teacher supervision rather than relying only on dense or query-level generator losses.=======
5. V8, V9, and V11-V18 should be reported as negative but informative trainable-generator attempts against V17 EoMT-DINOv3.
6. V10 should be reported as the V17 white-box diagnostic showing that V17 is attackable under direct per-image FGSM/PGD.
7. V19 should be reported as a mechanism-level negative result: it introduces per-image latent adaptation and richer metrics, lowers GT-class confidence, but still does not lower global V17 mIoU.
8. V20-V21 should be reported as final trainable-generator attempts that remain weak.
9. V22-V25 should be used for the final per-image white-box attack comparison:
   - untargeted degradation: PGD, CosPGD-inspired PGD, Momentum PGD, Universal Perturbation;
   - targeted semantic attack: foreground/person target-mask PGD variants.
10. V26 should be treated as the final evidence notebook because it consolidates the reproducible historical rerun code and the report-ready visualization pipeline.