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
- Attack V8: Feature-ASPP generator adapted to V17 EoMT-DINOv3
- Attack V9: CE-margin Feature-ASPP generator adapted to V17 EoMT-DINOv3

### Non-Trainable Attacks

Non-trainable attacks directly optimize each input image using gradients, without learning a reusable generator.

Used in:

- Attack V2: FGSM and PGD

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

## Reporting Guidance

Recommended report framing:

- Use Attack V2 PGD as the strongest non-trainable white-box baseline.
- Use Attack V7 as the strongest trainable adversarial model.
- Explain that V5 -> V6 -> V7 is not only hyperparameter tuning:
  - V5 changes generator architecture to ResNet.
  - V6 changes generator architecture to multi-scale ASPP.
  - V7 changes attack objective to output plus feature-level attack.
- Include Attack V8 and V9 as negative V17 experiments:
  - V8 shows that V7's targeted-background + feature objective does not transfer to EoMT-DINOv3.
  - V9 shows that even a direct CE + margin objective does not reduce global mIoU under the same `eps=4/255` budget.
  - These results support the claim that V17 is more robust than the previous SegFormer-B5 target under the tested trainable attacks.

## Known Caveats

- V2 PGD is much stronger than trainable attacks because it optimizes perturbations per image.
- Trainable generators are harder because one generator must generalize across validation images.
- V4 was not a model failure; it was an environment compatibility blocker.
- V8 and V9 are evaluated on the V17 37-image validation split, so small per-class drops can be offset by improvements elsewhere.
- A V17 PGD/FGSM sanity check is needed before concluding that V17 is broadly robust; current evidence only shows the tested trainable generator family is weak against V17.
- Some output CSV files may only exist in Colab/Drive unless explicitly copied back locally. The local notebooks still preserve the displayed results.
