# Adversarial Attack Notes

This file keeps conceptual and implementation notes for the adversarial-attack part of the assignment.

## Core Definitions

- `clean mIoU`: validation mIoU of the frozen segmentation model on the original validation images.
- `attack mIoU`: validation mIoU after applying adversarial perturbations.
- `mIoU_drop`: `clean mIoU - attack mIoU`.
- `eps=4/255`: each image channel can be changed by at most 4 pixel values on the 0-255 scale.
- `max_delta_pixel`: maximum absolute perturbation value in pixel space.
- `mean_delta_pixel`: average absolute perturbation value in pixel space.

## Human-Visible Effect

All trainable generator attacks in V1, V3, V5, V6, and V7 produce:

```text
adv_image = clip(clean_image + delta, 0, 255)
```

The perturbation `delta` is bounded by `eps`, so the adversarial image should look very close to the original image to humans. The perturbation visualization is usually amplified so the noise pattern can be inspected.

Attack V8 and V9 use the same image-space perturbation principle on V17 EoMT-DINOv3:

```text
adv_image = clip(clean_image + delta, 0, 255)
```

They also keep `eps=4/255`, so the human-visible image difference should remain very small.
>>>>>>> 6dffe60 (Document adversarial attack experiments)## Trainable vs Non-Trainable Attacks

### Trainable Attacks

Trainable attacks learn a generator network. The segmentation model is frozen, and only the adversarial generator is optimized.

Used in:

- Attack V1: tiny trainable perturbation generator
- Attack V3: U-Net-style generator
- Attack V4: SegMAN-B U-Net design, blocked by environment
- Attack V5: ResNet generator
- Attack V6: ASPP generator
- Attack V7: Feature-ASPP generator
<<<<<<< HEAD>>>>>>> e078db8 (Add semantic PGD attack report artifacts)
- Attack V8: Feature-ASPP generator adapted to V17 EoMT-DINOv3
- Attack V9: CE-margin Feature-ASPP generator adapted to V17 EoMT-DINOv3
- Attack V11: PGD-distilled Feature-ASPP generator adapted to V17 EoMT-DINOv3
- Attack V12: query-level Feature-ASPP generator adapted to V17 EoMT-DINOv3
<<<<<<< HEAD
=======
=======
- Attack V13: PGD-teacher Feature-ASPP generator adapted to V17 EoMT-DINOv3
- Attack V14: Transformer generator, targeted-background attack on V17
- Attack V15: DINO feature-conditioned Transformer generator on V17
- Attack V16: SEA-inspired Transformer targeted-background attack on V17
- Attack V17: Transformer generator, target-person attack on V17
- Attack V18: data-driven target-class Transformer generator on V17
- Attack V19: per-image latent Transformer generator on V17
- Attack V20: CosPGD-inspired Transformer generator objective on V17
- Attack V21: target-mask distillation generator on V17
### Non-Trainable Attacks

Non-trainable attacks directly optimize each input image using gradients, without learning a reusable generator.

Used in:

- Attack V2: FGSM and PGD
- Attack V10: FGSM and PGD on V17 EoMT-DINOv3>>>>>>> 6dffe60 (Document adversarial attack experiments)- Attack V10: FGSM and PGD on V17 EoMT-DINOv3
- Attack V22: per-image PGD and CosPGD-inspired attacks on V17
- Attack V23: Momentum PGD and Universal Perturbation on V17
- Attack V24: stronger PGD and semantic targeted PGD on V17
- Attack V25: stronger targeted non-person foreground to person PGD on V17
- Attack V26: final report notebook that reruns historical per-image attacks and produces final figures
FGSM, PGD, CosPGD-inspired PGD, Momentum PGD, and Universal Perturbation are white-box baselines, but they are not trainable adversarial generator models. In the final report they are useful because they show what is possible when the adversary can optimize each image directly.

## Attack Objectives

### Untargeted CE

Maximizes the normal segmentation loss so the prediction becomes wrong:

```text
loss = -cross_entropy(prediction, ground_truth)
```

This is simple but can be unstable for a reusable generator.

### Targeted Background

Forces foreground pixels toward background:

```text
target = background
foreground pixels get higher weight
```

This objective was used because the assignment target is semantic segmentation, and suppressing foreground objects gives an interpretable failure mode.

### Feature-Level Deviation

Used in Attack V7.

The frozen SegFormer-B5 model computes features for both clean and adversarial images:

```text
clean image -> SegFormer intermediate features
adv image   -> SegFormer intermediate features
```

The generator is trained to push the adversarial features away from clean features while still using targeted-background output loss.

This makes V7 a design upgrade over V6:

```text
V6: attack final segmentation output
V7: attack final segmentation output + internal feature representation
```

<<<<<<< HEAD>>>>>>> e078db8 (Add semantic PGD attack report artifacts)
### EoMT Query-Based Semantic Scores

Attack V8 and V9 target V17 EoMT-DINOv3. Unlike SegFormer, EoMT is query-based:

```text
image -> mask queries + class queries -> semantic segmentation map
```

The attack notebooks combine the query outputs into dense semantic scores before applying loss functions. This is necessary for comparison with normal semantic-segmentation mIoU, but it may make the gradient less direct than attacking a dense-logit model such as SegFormer.

### CE + Margin Objective

Used in Attack V9.

The CE part tries to make the model wrong with respect to the ground-truth mask:

```text
loss_ce = -cross_entropy(prediction, ground_truth)
```

The margin part tries to reduce the gap between the correct class and the strongest wrong class:

```text
margin = correct_class_score - strongest_wrong_class_score
```

The generator minimizes this margin, so the strongest wrong class can overtake the correct class. V9 also keeps feature deviation as a low-weight auxiliary term.

### Direct V17 FGSM / PGD

Used in Attack V10.

Attack V10 directly optimizes each input image against V17 EoMT-DINOv3:

```text
FGSM: one gradient-sign step
PGD: iterative projected gradient steps
```

This is not a trainable adversarial model, but it is an essential diagnostic. It shows whether the frozen V17 target can be attacked at all under the same `eps=4/255` budget.

V10 result:

```text
clean mIoU: 79.232563
FGSM 4/255 mIoU: 45.792824
PGD 4/255 steps10 mIoU: 4.777270
```

Therefore V17 is attackable under direct per-image white-box optimization. The weaker V8/V9/V11/V12 results should be interpreted as limitations of the reusable trainable generator designs.

### PGD Final-Delta Distillation

Used in Attack V11.

The idea is:

```text
1. run PGD to produce a strong teacher perturbation
2. train a Feature-ASPP generator to imitate that final perturbation
```

This remains a trainable-generator approach because the final attack is produced by a learned generator. However, V11 failed because the learned perturbation stayed too weak:

```text
mean_delta_pixel: 0.094721
mIoU_drop: -0.010602
```

This suggests that simply imitating the final PGD delta is not enough for V17.

### EoMT Query-Level Objective

Used in Attack V12.

EoMT predicts segmentation through query class logits and query mask logits. V12 therefore attacks the query structure more explicitly:

```text
query no-object loss
query foreground-class suppression
query mask suppression over GT foreground
query entropy loss
dense CE loss
dense margin loss
```

The goal is to disrupt query-to-object assignment, not only the final dense semantic map. V12 still fails globally:

```text
clean mIoU: 79.232563
attack mIoU: 79.997189
mIoU_drop: -0.764625
```

This suggests that query-level losses alone are not enough for the current Feature-ASPP generator on V17.

<<<<<<< HEAD
=======## Model Progression
=======
### PGD-Teacher Feature-ASPP

Used in Attack V13.

This is a second attempt to transfer PGD behavior to a trainable Feature-ASPP generator. It still does not reproduce PGD's strength:

```text
clean mIoU: 79.232563
attack mIoU: 79.247728
mIoU_drop: -0.015164
```

The result reinforces the conclusion from V11: naive PGD teacher supervision is not enough.

### Transformer Generator

Used in Attack V14 to V18.

The generator changes from CNN/ASPP-style perturbation synthesis to patch-token attention:

```text
image -> patch tokens -> Transformer encoder -> perturbation map
```

This is a model-level change from V5-V13. However, on V17 it remains weak:

```text
V14 targeted background: +0.258559 mIoU drop
V16 SEA-inspired targeted background: +0.283231 mIoU drop
V17 target person: -0.271262 mIoU drop
V18 data-driven target class: -0.291301 mIoU drop
```

The main lesson is that changing to a Transformer generator alone is not sufficient for V17.

### DINO Feature Conditioning

Used in Attack V15.

V15 conditions the Transformer generator on frozen V17/DINO feature tokens:

```text
delta = G(image, frozen_v17_features)
```

The goal was to make the adversary aware of the target model's representation. The result was negative:

```text
clean mIoU: 79.232563
attack mIoU: 79.883066
mIoU_drop: -0.650503
```

This suggests that feature conditioning alone may regularize the generator rather than making it more destructive.

### SEA-Inspired Targeted Loss

Used in Attack V16.

V16 keeps the Transformer generator but changes the objective to be more segmentation-specific:

```text
targeted foreground-to-background loss
class-balanced foreground loss
soft IoU suppression
prediction divergence
query-level object-erasing losses
TV regularization
```

This improves slightly over V14:

```text
V14 mIoU_drop: 0.258559
V16 mIoU_drop: 0.283231
```

The improvement is real but too small to be considered a strong attack.

### Targeted Class Attack

Used in Attack V17 and V18.

Targeted class attacks force pixels into a selected wrong class:

```text
foreground -> person
foreground -> selected target class
```

This is different from simply lowering mIoU:

```text
lower mIoU only = untargeted
force a specific wrong class = targeted
```

V17 manually selects `person`. V18 selects a target class from clean validation confusion statistics. V18 also selects `person`, but both fail globally:

```text
V17 mIoU_drop: -0.271262
V18 mIoU_drop: -0.291301
```

This suggests that target-class forcing is too difficult for the current trainable generator on V17.

### Per-Image Latent Transformer

Used in Attack V19.

V19 changes the attack mechanism:

```text
Previous trainable attacks:
delta = G(image)

V19:
delta_i = G(image_i, z_i)
```

Here, `z_i` is a learnable latent code for each image. This makes V19 more adaptive than a universal generator while still remaining a trainable adversarial model.

Relationship to PGD:

```text
PGD: optimize delta_i directly
V19: optimize z_i, then generate delta_i through G(image_i, z_i)
```

This is a mechanism-level change, not just a loss-weight or target-class change.

### CosPGD-Inspired Generator Objective

Used in Attack V20.

V20 tries to borrow the CosPGD idea into a trainable Transformer generator. The goal is not to run PGD at inference time, but to make the generator learn perturbations that behave more like a strong per-image optimizer.

Result:

```text
clean mIoU: 79.232563
attack mIoU: 79.260748
mIoU_drop: -0.028184
max delta: 1.896852
mean delta: 0.237785
```

Interpretation:

- The experiment is valid as an objective-level attempt.
- It does not transfer PGD/CosPGD strength into the trainable generator.
- The perturbation is too weak and global mIoU does not fall.

### Target-Mask Distillation Generator

Used in Attack V21.

V21 changes the generator target from a single class to a full segmentation mask. The generator is trained to push the attacked prediction toward a target mask and is evaluated using both degradation and target-similarity metrics.

Main metrics:

```text
clean mIoU: 79.232563
attack mIoU: 79.092610
mIoU_drop: 0.139953
pixel_accuracy_drop: 0.043807
prediction_change_rate: 0.002117
mean_gt_probability_drop: 0.011412
target_match_rate_increase: 0.000509
target_foreground_match_rate_increase: 0.001008
mean_target_probability_increase: 0.000660
mean_delta_pixel: 3.816305
```

Interpretation:

- V21 is closer to a segmentation-native targeted generator than earlier single-class objectives.
- It still gives only a very small global effect on V17.
- This is another reason to use direct per-image PGD-family attacks for the final attack evidence.

### Per-Image CosPGD-Inspired PGD

Used in Attack V22.

V22 separates the CosPGD idea from trainable generators and uses it as a per-image white-box optimizer. It is compared directly against PGD at the same `eps=4/255`, `steps=10` budget.

Results:

| Condition | mIoU | mIoU drop | pixel accuracy drop | prediction change rate | mean GT probability drop |
|---|---:|---:|---:|---:|---:|
| PGD 4/255 steps10 | 5.404918 | 73.827646 | 43.135906 | 0.446279 | 0.432747 |
| CosPGD 4/255 steps10 | 4.639892 | 74.592671 | 68.085676 | 0.699577 | 0.730801 |

Interpretation:

- Both attacks are very strong.
- CosPGD-inspired PGD changes more pixels and lowers GT probability much more than plain PGD under the same budget.

### Momentum PGD and Universal Perturbation

Used in Attack V23.

V23 compares two additional PGD-family ideas:

```text
Momentum PGD: accumulates normalized gradient direction across PGD steps.
Universal Perturbation: learns one shared perturbation for many images.
```

Results:

| Condition | mIoU | mIoU drop | pixel accuracy drop | prediction change rate |
|---|---:|---:|---:|---:|
| PGD 4/255 steps10 | 4.875495 | 74.357068 | 45.072659 | 0.464606 |
| Momentum PGD 4/255 steps10 | 4.569797 | 74.662766 | 48.405085 | 0.499099 |
| Universal PGD 4/255 | 79.573316 | -0.340752 | 0.002639 | 0.002410 |

Interpretation:

- Momentum PGD is a useful stronger baseline.
- Universal Perturbation is not effective in this setup, even though its perturbation magnitude reaches the budget.

### Stronger PGD and Semantic Targeted PGD

Used in Attack V24.

V24 tests whether stronger budgets and foreground-oriented losses improve PGD attacks, and whether target-class segmentation attacks can be made report-friendly.

Important variants:

| Condition | Target mode | eps | steps | mIoU drop | Target match |
|---|---|---:|---:|---:|---:|
| PGD eps6 steps10 | untargeted | 6/255 | 10 | 75.701469 | N/A |
| PGD foreground weight 8 | untargeted | 4/255 | 10 | 75.122680 | N/A |
| all foreground -> person | targeted | 4/255 | 10 | 69.665253 | 0.914434 |
| non-person foreground -> person | targeted | 4/255 | 10 | 69.339849 | 0.919196 |

Interpretation:

- Increasing epsilon from `4/255` to `6/255` improves mIoU degradation.
- Targeting `person` provides a cleaner semantic story than simply erasing objects into background.

### Final Targeted-Person Strengthening

Used in Attack V25.

V25 isolates the targeted semantic attack and adds target-region metrics:

```text
target_region_match_rate
target_region_change_rate
target_region_accuracy_drop
target_region_clean_accuracy
target_region_adv_accuracy
```

Best V25 targeted result:

```text
condition: pgd_target_non_person_fg_to_person_eps6_255_steps20
target_region_match_rate: 0.993359
target_region_change_rate: 0.993568
target_region_accuracy_drop: 0.918506
mIoU_drop: 73.173467
pixel_accuracy_drop: 14.799716
max_delta_pixel: 6.0
mean_delta_pixel: 3.231018
samples: 37
```

Interpretation:

- This is the strongest targeted-person PGD variant so far.
- Full-image pixel accuracy can remain high because background dominates the image.
- For targeted attacks, the report should emphasize target-region match/change/drop rather than only full-image accuracy.

### Final Report Notebook

Used in Attack V26.

V26 is a consolidation notebook, not a new attack mechanism. It contains executable rerun code for the historical comparisons:

```text
V10 FGSM/PGD sanity check
V22 PGD vs CosPGD
V23 Momentum PGD and Universal Perturbation
V24 stronger PGD and semantic targets
V25 targeted-person strengthening sweep
```

It keeps the final visual evidence focused on:

```text
untargeted best: cospgd_untargeted_eps6_255_steps20
targeted best: pgd_target_non_person_fg_to_person_eps6_255_steps20
```

For targeted visualizations, V26 filters examples so the ground truth contains no `person` class and still contains non-person foreground pixels. This makes the task definition closer to the classification attack setup where the target class is absent before the attack.

## Additional Metrics for Attack V19

V19 should not be evaluated only with mIoU. It reports:

| Metric | Meaning |
|---|---|
| `mIoU_drop` | Overall mIoU degradation. |
| `foreground_mIoU_drop` | Foreground-only mIoU degradation. |
| `pixel_accuracy_drop` | Valid-pixel accuracy degradation. |
| `prediction_change_rate` | Fraction of valid pixels whose predicted class changes. |
| `mean_gt_probability_drop` | Drop in model confidence for the GT class on foreground pixels. |
| `per_class_iou_drop` | Class-wise IoU degradation. |
| `max_delta_pixel` / `mean_delta_pixel` | Perturbation magnitude and visibility. |

Actual V19 result:

```text
clean mIoU: 79.232563
attack mIoU: 79.316299
mIoU_drop: -0.083735
foreground_mIoU_drop: -0.087799
pixel_accuracy_drop: 0.002876
prediction_change_rate: 0.002282
mean_gt_probability_drop: 0.009674
max_delta_pixel: 4.0
mean_delta_pixel: 2.454069
```

Interpretation:

- V19 does not reduce global mIoU.
- It does reduce the model's mean GT-class probability, so the perturbation affects confidence before it changes enough argmax pixels.
- The `prediction_change_rate` remains very small, which explains why mIoU does not fall.
- The per-image latent mechanism is valid, but the current latent/generator/loss setting is not strong enough for broad V17 segmentation failure.

## Model Progression
### Attack V1

- Purpose: prove that a trainable adversarial generator can run.
- Limitation: generator capacity was too small.
- Result: weak global mIoU drop.

### Attack V2

- Purpose: provide strong white-box baselines.
- FGSM: one-step gradient-sign perturbation.
- PGD: iterative projected gradient attack.
- Result: PGD is much stronger than all trainable generators, as expected.

### Attack V3

- Purpose: move from tiny generator to U-Net-style generator.
- Improvement: more capacity and targeted-background objective.
- Result: better than V1, still weak globally.

### Attack V4

- Purpose: attack V10 SegMAN-B.
- Status: not completed in Colab.
- Blocker: current Colab Python 3.12 cannot install the required old SegMAN stack cleanly, including the required PyTorch/MMCV/NATTEN combination.

### Attack V5

- Purpose: switch to the strongest Colab-compatible target, V8 SegFormer-B5, and test a ResNet generator.
- Result: first clearly stronger trainable-generator drop.

### Attack V6

- Purpose: test ASPP multi-scale generator.
- Result: clearly stronger than V5.
- Interpretation: multi-scale context helps adversarial perturbation generation.

### Attack V7

- Purpose: test a modern feature-level attack objective.
- Result: strongest trainable generator so far.
- Interpretation: attacking internal representations is more effective than attacking output logits alone.

### Attack V8

- Purpose: test whether the strongest trainable attack family, V7 Feature-ASPP, transfers to the new best target V17 EoMT-DINOv3.
- Change from V7:
  - target model changes from MMSeg SegFormer-B5 to Hugging Face EoMT-DINOv3.
  - checkpoint changes from `.pth` to `model.safetensors`.
  - feature extraction and semantic score computation must follow EoMT's query-based output structure.
- Result:
  - clean mIoU `79.232563`
  - attack mIoU `79.599480`
  - mIoU drop `-0.366917`
- Interpretation:
  - the V7 targeted-background + feature-deviation attack does not transfer successfully to V17.

### Attack V9

- Purpose: test whether a more direct output objective works better on V17 than targeted-background loss.
- Change from V8:
  - replaces targeted-background loss with untargeted CE + correct-vs-wrong margin loss.
  - keeps Feature-ASPP generator to isolate the objective change.
  - keeps feature deviation only as a low-weight auxiliary loss.
- Result:
  - clean mIoU `79.232563`
  - attack mIoU `79.576007`
  - mIoU drop `-0.343444`
- Interpretation:
  - direct CE + margin objective still does not reduce global V17 mIoU.
  - this makes V17 a useful robustness case and suggests that the current trainable generator family is not sufficient for the EoMT target.

### Attack V10

- Purpose: verify whether V17 is attackable under direct white-box optimization.
- Method:
  - FGSM 2/255
  - FGSM 4/255
  - PGD 4/255, 10 steps
- Result:
  - FGSM 4/255 mIoU drop: `33.439740`
  - PGD 4/255 mIoU drop: `74.455294`
- Interpretation:
  - V17 is highly attackable by direct per-image PGD.
  - V8/V9 failure is not because the target is impossible to attack.

### Attack V11

- Purpose: convert strong PGD behavior into a trainable generator.
- Method:
  - Feature-ASPP generator
  - PGD final-delta distillation
  - V17 fixed environment stack
  - V17 `712/37` split
- Result:
  - clean mIoU `79.232563`
  - attack mIoU `79.243165`
  - mIoU drop `-0.010602`
- Interpretation:
  - the final-delta distillation signal is too weak.
  - the generator learns very small perturbations and does not reproduce PGD's destructive effect.

### Attack V12

- Purpose: attack EoMT's query mechanism more directly.
- Method:
  - Feature-ASPP generator
  - dense CE and margin losses
  - query no-object, foreground-class, mask-suppression, and entropy losses
  - V17 tar.gz cache support
- Result:
  - clean mIoU `79.232563`
  - attack mIoU `79.997189`
  - mIoU drop `-0.764625`
- Interpretation:
  - query-level losses damage a few classes locally, but do not reduce global mIoU.
  - the current generator still cannot exploit V17 the way direct PGD can.
>>>>>>> 6dffe60 (Document adversarial attack experiments)
### Attack V13

- Purpose: retry PGD-teacher supervision with a Feature-ASPP generator.
- Result:
  - clean mIoU `79.232563`
  - attack mIoU `79.247728`
  - mIoU drop `-0.015164`
- Interpretation:
  - PGD-teacher training still does not transfer PGD's per-image strength into a reusable generator.

### Attack V14

- Purpose: test a Transformer generator on V17.
- Objective: targeted foreground-to-background.
- Result:
  - clean mIoU `79.232563`
  - attack mIoU `78.974004`
  - mIoU drop `0.258559`
- Interpretation:
  - first positive V17 Transformer-generator drop, but very small.
  - adversarial images showed visible horizontal stripe artifacts.

### Attack V15

- Purpose: condition the Transformer generator on frozen V17/DINO features.
- Result:
  - clean mIoU `79.232563`
  - attack mIoU `79.883066`
  - mIoU drop `-0.650503`
- Interpretation:
  - feature conditioning did not improve the attack.

### Attack V16

- Purpose: keep the Transformer generator but use SEA-inspired segmentation-specific loss.
- Result:
  - clean mIoU `79.232563`
  - attack mIoU `78.949332`
  - mIoU drop `0.283231`
- Interpretation:
  - best V17 targeted-background trainable result so far, but still weak.

### Attack V17

- Purpose: test whether targeting a strong class prior, `person`, is easier than targeting background.
- Result:
  - clean mIoU `79.232563`
  - attack mIoU `79.503825`
  - mIoU drop `-0.271262`
  - non-person foreground to person rate increases only `0.000128`
- Interpretation:
  - target-person attack is ineffective.

### Attack V18

- Purpose: choose target class from clean V17 confusion statistics.
- Result:
  - selected target class: `person`
  - clean mIoU `79.232563`
  - attack mIoU `79.523864`
  - mIoU drop `-0.291301`
  - non-target foreground to selected-target rate increases only `0.000148`
- Interpretation:
  - data-driven selection confirms `person` is the natural confusion target, but targeted training remains ineffective.

### Attack V19

- Purpose: make a true mechanism change from universal generator to per-image latent generator.
- Method:
  - per-image latent Transformer generator
  - untargeted segmentation loss
  - validation-time latent adaptation
  - expanded metrics beyond mIoU
- Result:
  - clean mIoU `79.232563`
  - attack mIoU `79.316299`
  - mIoU drop `-0.083735`
  - foreground mIoU drop `-0.087799`
  - pixel accuracy drop `0.002876`
  - prediction change rate `0.002282`
  - mean GT probability drop `0.009674`
  - max delta `4.0`
  - mean delta `2.454069`
- Interpretation:
  - V19 is the first mechanism-level shift after repeated universal-generator failures.
  - It creates measurable confidence-level attack pressure, but does not reduce global mIoU.

### Attack V20

- Purpose: test whether a CosPGD-inspired objective can make the Transformer generator behave more like direct PGD.
- Result:
  - clean mIoU `79.232563`
  - attack mIoU `79.260748`
  - mIoU drop `-0.028184`
- Interpretation:
  - valid objective-level attempt, but ineffective as a trainable V17 attack.

### Attack V21

- Purpose: train a generator toward a full segmentation target mask instead of a single target class.
- Result:
  - clean mIoU `79.232563`
  - attack mIoU `79.092610`
  - mIoU drop `0.139953`
  - prediction change rate `0.002117`
  - target match increase `0.000509`
- Interpretation:
  - segmentation-native target-mask idea is conceptually useful, but still weak on V17.

### Attack V22

- Purpose: compare per-image PGD and CosPGD-inspired PGD under the same budget.
- Result:
  - PGD 4/255 steps10 mIoU drop `73.827646`
  - CosPGD 4/255 steps10 mIoU drop `74.592671`
  - CosPGD pixel accuracy drop `68.085676`
  - CosPGD mean GT probability drop `0.730801`
- Interpretation:
  - CosPGD-inspired per-image optimization is a stronger untargeted attack candidate than plain PGD.

### Attack V23

- Purpose: compare PGD, Momentum PGD, and Universal Perturbation.
- Result:
  - PGD mIoU drop `74.357068`
  - Momentum PGD mIoU drop `74.662766`
  - Universal Perturbation mIoU drop `-0.340752`
- Interpretation:
  - Momentum PGD is useful; Universal Perturbation is ineffective.

### Attack V24

- Purpose: test stronger PGD budgets, foreground-focused losses, and semantic targeted attacks.
- Result:
  - PGD eps6 steps10 mIoU drop `75.701469`
  - foreground-weight 8 PGD mIoU drop `75.122680`
  - non-person foreground to person target match `0.919196`
- Interpretation:
  - higher epsilon is stronger for untargeted degradation.
  - target-class segmentation attacks need target-region metrics rather than only full-image accuracy.

### Attack V25

- Purpose: strengthen targeted `non-person foreground -> person` PGD with eps/steps sweeps.
- Best result:
  - `pgd_target_non_person_fg_to_person_eps6_255_steps20`
  - target match `0.993359`
  - target change `0.993568`
  - target accuracy drop `0.918506`
  - mIoU drop `73.173467`
- Interpretation:
  - this is the best targeted semantic attack result for the final report.

### Attack V26

- Purpose: consolidate all report evidence into one notebook.
- Includes:
  - historical rerun code for V10/V22/V23/V24/V25;
  - final attack configs for best untargeted and targeted evidence;
  - visualization filtering for no-person targeted examples;
  - report-ready five-panel figures.
- Interpretation:
  - V26 should be the notebook handed to teammates for final integration.
## Reporting Guidance

Recommended report framing:

- Use Attack V2 PGD as the strongest non-trainable white-box baseline.
- Use Attack V7 as the strongest trainable adversarial model.
- Explain that V5 -> V6 -> V7 is not only hyperparameter tuning:
  - V5 changes generator architecture to ResNet.
  - V6 changes generator architecture to multi-scale ASPP.
  - V7 changes attack objective to output plus feature-level attack.
<<<<<<< HEAD
- Include Attack V8, V9, V11, and V12 as negative V17 trainable-generator experiments:- Include Attack V8, V9, and V11-V18 as negative V17 trainable-generator experiments:  - V8 shows that V7's targeted-background + feature objective does not transfer to EoMT-DINOv3.
  - V9 shows that even a direct CE + margin objective does not reduce global mIoU under the same `eps=4/255` budget.
- V11 shows that naive PGD final-delta distillation does not preserve PGD's attack strength.
- V12 shows that query-level EoMT losses are still insufficient for the current generator.
- Include Attack V10 as the V17 white-box diagnostic:
  - it proves V17 is attackable by FGSM/PGD.
  - it justifies the next direction: PGD-guided trainable generator training.=======
- V14-V18 show that Transformer generators, feature conditioning, SEA-inspired targeted loss, target-person attack, and data-driven target selection still do not solve the V17 trainable-generator bottleneck.
- V19 should be framed as the first mechanism-level change after repeated universal-generator failures:
  - it adds per-image latent adaptation;
  - it uses untargeted loss;
  - it reports `foreground_mIoU_drop`, `pixel_accuracy_drop`, `prediction_change_rate`, and `mean_gt_probability_drop`.
  - it still fails to lower global mIoU, but shows a nonzero GT-confidence drop.
- V20-V21 close the trainable-generator exploration on V17:
  - V20: CosPGD-inspired generator objective is ineffective.
  - V21: target-mask distillation is valid but weak.
- V22-V25 are the final per-image white-box attack evidence:
  - V22: CosPGD-inspired PGD is stronger than PGD at the same budget.
  - V23: Momentum PGD is strong; Universal Perturbation fails.
  - V24: eps and foreground-target choices matter.
  - V25: targeted non-person foreground to person is the cleanest targeted segmentation attack.
- V26 is the final notebook to run for reproducible tables and report figures.
- Include Attack V10 as the V17 white-box diagnostic:
  - it proves V17 is attackable by FGSM/PGD.
  - it justifies the next direction: PGD-guided trainable generator training.
## Known Caveats

- V2 PGD is much stronger than trainable attacks because it optimizes perturbations per image.
- Trainable generators are harder because one generator must generalize across validation images.
- V4 was not a model failure; it was an environment compatibility blocker.
<<<<<<< HEAD
- V8, V9, V11, and V12 are evaluated on the V17 37-image validation split, so small per-class drops can be offset by improvements elsewhere.
- Attack V10 completed the V17 PGD/FGSM sanity check and showed V17 is strongly vulnerable to direct per-image PGD.
- The next open question is how to transfer PGD's strength into a reusable trainable generator.>>>>>>> 6dffe60 (Document adversarial attack experiments)
=======
- V8, V9, and V11-V18 are evaluated on the V17 37-image validation split, so small per-class drops can be offset by improvements elsewhere.
- Attack V10 completed the V17 PGD/FGSM sanity check and showed V17 is strongly vulnerable to direct per-image PGD.
- V19 shows that per-image latent adaptation alone is not enough in the current setting. The next open question is whether stronger latent adaptation, more validation steps, or a PGD-guided latent objective can recover more of PGD's strength while remaining a trainable adversarial model.
- V22-V25 are direct per-image attacks; they should not be described as trainable generator methods.
- V25 full-image pixel accuracy can look high even when the targeted region is successfully changed, because background pixels dominate most segmentation images.
- V26 targeted visual filtering improves the task definition for figures, but the full 37-image metrics are still useful as the main validation-set comparison.- Some output CSV files may only exist in Colab/Drive unless explicitly copied back locally. The local notebooks still preserve the displayed results.
