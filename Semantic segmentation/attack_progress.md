# Adversarial Attack Progress

This file tracks implementation progress for adversarial-attack notebooks separately from the main semantic-segmentation training progress.

## Current Status

- Best trainable adversarial model: Attack V7 Feature-ASPP Generator.
- Best non-trainable white-box baseline: Attack V2 PGD.
- Main target used for the strongest trainable attacks: V8 SegFormer-B5, `best_mIoU_iter_11000.pth`.
- V4 SegMAN-B attack was designed but not completed because of Colab Python 3.12 dependency incompatibility.
- Latest target tested: V17 EoMT-DINOv3, Hugging Face checkpoint format (`model.safetensors`).
- Attack V8, V9, V11, and V12 did not reduce V17 global mIoU; V17 appears robust to the current trainable Feature-ASPP generator family.
- Attack V10 confirmed V17 is strongly attackable by direct FGSM/PGD, so the current bottleneck is reusable trainable-generator design rather than target-model immunity.

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

## Attack V10

- Notebook: `attack_v10_fgsm_pgd_eomt_v17_colab.ipynb`
- Target: V17 EoMT-DINOv3
- Checkpoint format:
  - Hugging Face-style folder
  - `config.json`
  - `preprocessor_config.json`
  - `model.safetensors`
  - `metrics.json`
- Attack type: FGSM and PGD white-box attacks
- Trainable: no
- Perturbation budget:
  - FGSM `eps=2/255`
  - FGSM `eps=4/255`
  - PGD `eps=4/255`, `10` steps
- Validation split:
  - V17 `712/37` split
  - `37` validation samples
- Result:

| Condition | mIoU | mIoU drop | max delta | mean delta |
|---|---:|---:|---:|---:|
| clean | 79.232563 | 0.000000 | 0.0 | 0.000000 |
| fgsm_eps2_255 | 49.506473 | 29.726090 | 2.0 | 1.984618 |
| fgsm_eps4_255 | 45.792824 | 33.439740 | 4.0 | 3.957107 |
| pgd_eps4_255_steps10 | 4.777270 | 74.455294 | 4.0 | 2.305559 |

- Interpretation:
  - V17 is strongly attackable under direct per-image white-box optimization.
  - V10 is not a trainable adversarial model, but it is the key diagnostic proving V17 is not immune to adversarial perturbations.

## Attack V11

- Notebook: `attack_v11_pgd_distilled_generator_eomt_v17_colab.ipynb`
- Target: V17 EoMT-DINOv3
- Attack type: PGD-distilled Feature-ASPP generator
- Trainable: yes
- Perturbation budget: `eps=4/255`
- Main objective:
  - imitate PGD final delta
  - preserve image-space perturbation budget
- Environment:
  - V17 fixed Colab stack
  - `numpy==1.26.4`
  - `scipy==1.13.1`
  - `scikit-learn==1.5.2`
  - `pandas==2.2.2`
  - `pillow==11.3.0`
- Data:
  - V17 `712/37` split
  - tar.gz/folder cache support
- Output directory:
  - `outputs/attacks/attack_v11_pgd_distilled_generator_eomt_v17`
- Result:
  - clean mIoU: `79.232563`
  - attack mIoU: `79.243165`
  - mIoU drop: `-0.010602`
  - max delta: `3.809509`
  - mean delta: `0.094721`
- Interpretation:
  - simply distilling PGD's final perturbation is not enough.
  - the learned delta is too small and does not reproduce PGD's destructive effect.

## Attack V12

- Notebook: `attack_v12_query_level_generator_eomt_v17_colab.ipynb`
- Target: V17 EoMT-DINOv3
- Attack type: query-level Feature-ASPP generator
- Trainable: yes
- Perturbation budget: `eps=4/255`
- Main objective:
  - dense CE loss
  - dense margin loss
  - query no-object loss
  - query foreground-class suppression
  - query mask suppression
  - query entropy loss
- Environment:
  - V17 fixed Colab stack
  - explicit uninstall/reinstall for `numpy`, `scipy`, `scikit-learn`, `pandas`, and `pillow`
- Data:
  - V17 `712/37` split
  - tar.gz cache support
- Output directory:
  - `outputs/attacks/attack_v12_query_level_generator_eomt_v17`
- Drive backup directory:
  - `MyDrive/CV_Assignment_Outputs/adversarial_attack_v12_query_level_eomt_v17`
- Result:
  - clean mIoU: `79.232563`
  - attack mIoU: `79.997189`
  - mIoU drop: `-0.764625`
  - max delta: `4.0`
  - mean delta: `1.782573`
- Per-class local degradation:
  - bicycle: `2.444729`
  - bottle: `1.045298`
  - tvmonitor: `0.983434`
  - dog: `0.516119`
  - cow: `0.431439`
- Interpretation:
  - query-level losses affect some classes locally but still fail globally.
  - V12 reinforces that V17 requires a stronger trainable attack strategy than the current Feature-ASPP objective family.

## Next Steps

- Preserve V7 as the current main trainable adversarial-attack result.
- Preserve V8, V9, V11, and V12 as negative but useful V17 trainable-generator experiments.
- Preserve V10 as the direct white-box diagnostic proving V17 can be attacked.
- Recommended next experiment:
  - train a PGD-guided Feature-ASPP generator on V17.
  - Use PGD as teacher supervision instead of only dense/query-level losses.
  - Keep the fixed V17 Colab stack and tar.gz cache workflow.
- Other possible follow-up experiments:
  - evaluate V17 attacks on a larger validation subset instead of only the 37-image split;
  - design an EoMT query-level attack objective;
  - test stronger perturbation budgets only if the assignment allows visible-budget comparison.
