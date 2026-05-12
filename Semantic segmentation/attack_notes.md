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
>>>>>>> 6dffe60 (Document adversarial attack experiments)
## Trainable vs Non-Trainable Attacks

### Trainable Attacks

Trainable attacks learn a generator network. The segmentation model is frozen, and only the adversarial generator is optimized.

Used in:

- Attack V1: tiny trainable perturbation generator
- Attack V3: U-Net-style generator
- Attack V4: SegMAN-B U-Net design, blocked by environment
- Attack V5: ResNet generator
- Attack V6: ASPP generator
- Attack V7: Feature-ASPP generator
<<<<<<< HEAD
- Attack V8: Feature-ASPP generator adapted to V17 EoMT-DINOv3
- Attack V9: CE-margin Feature-ASPP generator adapted to V17 EoMT-DINOv3
- Attack V11: PGD-distilled Feature-ASPP generator adapted to V17 EoMT-DINOv3
- Attack V12: query-level Feature-ASPP generator adapted to V17 EoMT-DINOv3
=======
### Non-Trainable Attacks

Non-trainable attacks directly optimize each input image using gradients, without learning a reusable generator.

Used in:

- Attack V2: FGSM and PGD
- Attack V10: FGSM and PGD on V17 EoMT-DINOv3>>>>>>> 6dffe60 (Document adversarial attack experiments)

FGSM and PGD are very strong white-box baselines, but they are not trainable adversarial models.

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

<<<<<<< HEAD
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

=======## Model Progression

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
## Reporting Guidance

Recommended report framing:

- Use Attack V2 PGD as the strongest non-trainable white-box baseline.
- Use Attack V7 as the strongest trainable adversarial model.
- Explain that V5 -> V6 -> V7 is not only hyperparameter tuning:
  - V5 changes generator architecture to ResNet.
  - V6 changes generator architecture to multi-scale ASPP.
  - V7 changes attack objective to output plus feature-level attack.
<<<<<<< HEAD
- Include Attack V8, V9, V11, and V12 as negative V17 trainable-generator experiments:
  - V8 shows that V7's targeted-background + feature objective does not transfer to EoMT-DINOv3.
  - V9 shows that even a direct CE + margin objective does not reduce global mIoU under the same `eps=4/255` budget.
- V11 shows that naive PGD final-delta distillation does not preserve PGD's attack strength.
- V12 shows that query-level EoMT losses are still insufficient for the current generator.
- Include Attack V10 as the V17 white-box diagnostic:
  - it proves V17 is attackable by FGSM/PGD.
  - it justifies the next direction: PGD-guided trainable generator training.
=======
## Known Caveats

- V2 PGD is much stronger than trainable attacks because it optimizes perturbations per image.
- Trainable generators are harder because one generator must generalize across validation images.
- V4 was not a model failure; it was an environment compatibility blocker.
<<<<<<< HEAD
- V8, V9, V11, and V12 are evaluated on the V17 37-image validation split, so small per-class drops can be offset by improvements elsewhere.
- Attack V10 completed the V17 PGD/FGSM sanity check and showed V17 is strongly vulnerable to direct per-image PGD.
- The next open question is how to transfer PGD's strength into a reusable trainable generator.
=======
>>>>>>> 6dffe60 (Document adversarial attack experiments)
- Some output CSV files may only exist in Colab/Drive unless explicitly copied back locally. The local notebooks still preserve the displayed results.
