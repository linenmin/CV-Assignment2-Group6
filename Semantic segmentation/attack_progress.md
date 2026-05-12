# Adversarial Attack Progress

This file tracks implementation progress for adversarial-attack notebooks separately from the main semantic-segmentation training progress.

## Current Status

- Best trainable adversarial model: Attack V7 Feature-ASPP Generator.
- Best non-trainable white-box baseline: Attack V2 PGD.
- Main target used for the strongest trainable attacks: V8 SegFormer-B5, `best_mIoU_iter_11000.pth`.
- V4 SegMAN-B attack was designed but not completed because of Colab Python 3.12 dependency incompatibility.
- Latest target tested: V17 EoMT-DINOv3, Hugging Face checkpoint format (`model.safetensors`).
- Attack V8 and V9 did not reduce V17 global mIoU; V17 appears robust to the current trainable Feature-ASPP generator family.

## Attack V1

- Notebook: `attack_v1_trainable_adversary_segformer_b3_colab.ipynb`
- Target: V7 SegFormer-B3
- Checkpoint: `best_mIoU_iter_8000.pth`
- Attack type: tiny trainable adversarial generator
- Trainable: yes
- Perturbation budget: `eps=4/255`
- Output directory:
  - `outputs/attacks/attack_v1_trainable_adversary_segformer_b3`
- Result:
  - clean mIoU: `64.720578`
  - attack mIoU: `64.610856`
  - mIoU drop: `0.109722`
- Interpretation:
  - the setup runs, but the generator is too weak to meaningfully attack the target model.

## Attack V2

- Notebook: `attack_v2_fgsm_pgd_segformer_b3_colab.ipynb`
- Target: V7 SegFormer-B3
- Checkpoint: `best_mIoU_iter_8000.pth`
- Attack type: FGSM and PGD white-box attacks
- Trainable: no
- Output directory:
  - `outputs/attacks/attack_v2_fgsm_pgd_segformer_b3`
- Results:

| Condition | mIoU | mIoU drop | max delta | mean delta |
|---|---:|---:|---:|---:|
| clean | 64.720578 | 0.000000 | 0.0 | 0.000000 |
| fgsm_eps2_255 | 36.305314 | 28.415264 | 2.0 | 1.964038 |
| fgsm_eps4_255 | 33.878626 | 30.841952 | 4.0 | 3.912106 |
| pgd_eps2_255_steps5 | 19.002467 | 45.718110 | 2.0 | 1.418412 |
| pgd_eps4_255_steps10 | 7.917722 | 56.802856 | 4.0 | 2.522398 |
| pgd_eps6_255_steps10 | 7.856018 | 56.864560 | 6.0 | 3.419572 |

- Interpretation:
  - PGD is a very strong non-trainable white-box baseline.
  - It should not be described as the trainable adversarial model requested by the PDF, but it is useful as a robustness reference.

## Attack V3 Smoke

- Notebook: `attack_v3_trainable_unet_adversary_segformer_b3_colab.ipynb`
- Target: V7 SegFormer-B3
- Checkpoint: `best_mIoU_iter_8000.pth`
- Attack type: U-Net-style trainable generator
- Trainable: yes
- Perturbation budget: `eps=4/255`
- Main objective: targeted-background loss with foreground weighting
- Output directory:
  - `outputs/attacks/attack_v3_trainable_unet_adversary_segformer_b3`
- Result:
  - clean mIoU: `64.720578`
  - attack mIoU: `63.947911`
  - mIoU drop: `0.772667`
  - max delta: `4.0`
  - mean delta: `2.960264`
- Interpretation:
  - stronger than V1, but still much weaker than PGD.

## Attack V3 Full

- Notebook: `attack_v3_full_trainable_unet_adversary_segformer_b3_colab.ipynb`
- Target: V7 SegFormer-B3
- Checkpoint: `best_mIoU_iter_8000.pth`
- Attack type: U-Net-style trainable generator, longer run
- Trainable: yes
- Perturbation budget: `eps=4/255`
- Main objective: targeted-background loss with foreground weighting
- Output directory:
  - `outputs/attacks/attack_v3_full_trainable_unet_adversary_segformer_b3`
- Result:
  - clean mIoU: `64.720578`
  - attack mIoU: `63.880034`
  - mIoU drop: `0.840543`
  - max delta: `4.0`
  - mean delta: `3.603717`
- Interpretation:
  - 2000 iterations improve only slightly over the smoke result.

## Attack V4

- Notebook: `attack_v4_trainable_unet_adversary_segman_b_colab.ipynb`
- Target: V10 SegMAN-B
- Checkpoint source:
  - README Google Drive folder
  - `SegMAN_Encoder_b.pth.tar`
  - `best_mIoU_iter_25000.pth`
- Attack type: U-Net-style trainable generator design for SegMAN-B
- Trainable: yes, design only
- Status: blocked
- Blocker:
  - the available Colab runtime uses Python 3.12;
  - the SegMAN dependency stack requires an older PyTorch/MMCV/NATTEN setup that was not installable cleanly in that runtime.
- Interpretation:
  - V4 should be kept as a documented design attempt, not as an evaluated result.

## Attack V5

- Notebook: `attack_v5_resnet_generator_segformer_b5_colab.ipynb`
- Target: V8 SegFormer-B5
- Checkpoint: `best_mIoU_iter_11000.pth`
- Attack type: ResNet image-to-image generator
- Trainable: yes
- Perturbation budget: `eps=4/255`
- Main objective: targeted-background loss with foreground weighting
- Output directory:
  - `outputs/attacks/attack_v5_resnet_generator_segformer_b5`
- Result:
  - clean mIoU: `68.975346`
  - attack mIoU: `66.829134`
  - mIoU drop: `2.146212`
  - max delta: `4.0`
  - mean delta: `3.743160`
- Interpretation:
  - first trainable generator that produces a clear global drop on the stronger SegFormer-B5 target.

## Attack V6

- Notebook: `attack_v6_aspp_generator_segformer_b5_colab.ipynb`
- Target: V8 SegFormer-B5
- Checkpoint: `best_mIoU_iter_11000.pth`
- Attack type: ASPP multi-scale generator
- Trainable: yes
- Perturbation budget: `eps=4/255`
- Main objective: targeted-background loss with foreground weighting
- Output directory:
  - `outputs/attacks/attack_v6_aspp_generator_segformer_b5`
- Result:
  - clean mIoU: `68.975346`
  - attack mIoU: `64.941451`
  - mIoU drop: `4.033895`
  - max delta: `4.0`
  - mean delta: `3.256489`
- Interpretation:
  - multi-scale ASPP context is more effective than the V5 ResNet generator under the same target and perturbation budget.

## Attack V7

- Notebook: `attack_v7_feature_aspp_generator_segformer_b5_colab.ipynb`
- Target: V8 SegFormer-B5
- Checkpoint: `best_mIoU_iter_11000.pth`
- Attack type: Feature-ASPP generator
- Trainable: yes
- Perturbation budget: `eps=4/255`
- Main objective:
  - targeted-background output loss
  - intermediate SegFormer feature deviation loss
- Output directory:
  - `outputs/attacks/attack_v7_feature_aspp_generator_segformer_b5`
- Result:
  - clean mIoU: `68.975346`
  - attack mIoU: `60.354332`
  - mIoU drop: `8.621014`
  - max delta: `4.0`
  - mean delta: `3.390270`
- Interpretation:
  - strongest trainable adversarial model so far.
  - the feature-level objective gives a major gain over V6.

## Attack V8

- Notebook: `attack_v8_feature_aspp_generator_eomt_v17_colab.ipynb`
- Target: V17 EoMT-DINOv3
- Checkpoint format:
  - Hugging Face-style folder
  - `config.json`
  - `preprocessor_config.json`
  - `model.safetensors`
  - `metrics.json`
- Checkpoint source:
  - README Google Drive folder: `https://drive.google.com/drive/u/1/folders/1ImiIyTvxtMVIlrptKXfl3w3qF5CBdPCG`
- Attack type: Feature-ASPP trainable generator adapted from Attack V7
- Trainable: yes
- Perturbation budget: `eps=4/255`
- Main objective:
  - targeted-background output loss
  - EoMT feature deviation loss
- Validation split:
  - V17 `712/37` split
  - `37` validation samples
- Output directory:
  - `outputs/attacks/attack_v8_feature_aspp_generator_eomt_v17`
- Drive backup directory:
  - `MyDrive/CV_Assignment_Outputs/adversarial_attack_v8_feature_aspp_eomt_v17`
- Result:
  - clean mIoU: `79.232563`
  - attack mIoU: `79.599480`
  - mIoU drop: `-0.366917`
  - max delta: `4.0`
  - mean delta: `2.965704`
- Interpretation:
  - the V7 attack idea does not directly transfer to V17.
  - several classes degrade, but the global mIoU slightly increases on the small 37-image split.

## Attack V9

- Notebook: `attack_v9_ce_margin_generator_eomt_v17_colab.ipynb`
- Target: V17 EoMT-DINOv3
- Checkpoint format:
  - Hugging Face-style folder
  - `config.json`
  - `preprocessor_config.json`
  - `model.safetensors`
  - `metrics.json`
- Attack type: CE-margin Feature-ASPP trainable generator
- Trainable: yes
- Perturbation budget: `eps=4/255`
- Main objective:
  - untargeted CE loss
  - correct-vs-wrong margin loss
  - low-weight feature deviation auxiliary loss
- Validation split:
  - V17 `712/37` split
  - `37` validation samples
- Output directory:
  - `outputs/attacks/attack_v9_ce_margin_generator_eomt_v17`
- Drive backup directory:
  - `MyDrive/CV_Assignment_Outputs/adversarial_attack_v9_ce_margin_eomt_v17`
- Result:
  - clean mIoU: `79.232563`
  - attack mIoU: `79.576007`
  - mIoU drop: `-0.343444`
  - max delta: `4.0`
  - mean delta: `2.645145`
- Interpretation:
  - switching from targeted-background to direct CE + margin loss still does not reduce V17 global mIoU.
  - this suggests the issue is not only the V8 objective; V17 may require a different attack family or a stronger diagnostic baseline.

## Next Steps

- Preserve V7 as the current main trainable adversarial-attack result.
- Preserve V8 and V9 as negative but useful V17 robustness experiments.
- Recommended next diagnostic step:
  - run FGSM/PGD directly on V17 EoMT-DINOv3 at `eps=4/255`.
  - If PGD also fails, V17 is genuinely robust under the tested budget.
  - If PGD succeeds, the current trainable generator/objective is the bottleneck.
- Other possible follow-up experiments:
  - evaluate V17 attacks on a larger validation subset instead of only the 37-image split;
  - design an EoMT query-level attack objective;
  - test stronger perturbation budgets only if the assignment allows visible-budget comparison.
