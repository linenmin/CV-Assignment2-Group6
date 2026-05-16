# Adversarial Attack Progress

This file tracks implementation progress for adversarial-attack notebooks separately from the main semantic-segmentation training progress.

## Current Status

- Best trainable adversarial model: Attack V7 Feature-ASPP Generator.
- Best older SegFormer-B3 non-trainable white-box baseline: Attack V2 PGD.
- Best V17 untargeted per-image white-box direction: CosPGD-inspired / Momentum-style PGD family.
- Best V17 targeted semantic attack so far: Attack V25 `pgd_target_non_person_fg_to_person_eps6_255_steps20`.
- Final integration notebook: Attack V26 `attack_v26_final_report_attacks_eomt_v17_colab.ipynb`.
- Main target used for the strongest trainable attacks: V8 SegFormer-B5, `best_mIoU_iter_11000.pth`.
- V4 SegMAN-B attack was designed but not completed because of Colab Python 3.12 dependency incompatibility.
- Latest target tested: V17 EoMT-DINOv3, Hugging Face checkpoint format (`model.safetensors`).
- Attack V8, V9, V11, and V12 did not reduce V17 global mIoU; V17 appears robust to the current trainable Feature-ASPP generator family.
- Attack V10 confirmed V17 is strongly attackable by direct FGSM/PGD, so the current bottleneck is reusable trainable-generator design rather than target-model immunity.>>>>>>> 6dffe60 (Document adversarial attack experiments)- Attack V8, V9, and V11-V18 did not produce a strong V17 global mIoU drop; V17 appears robust to the tested universal trainable generator families.
- Attack V10 confirmed V17 is strongly attackable by direct FGSM/PGD, so the current bottleneck is reusable trainable-generator design rather than target-model immunity.
- Attack V19 completed as the next mechanism-level attempt: per-image latent Transformer generator with untargeted loss and expanded metrics. It lowers GT-class confidence but still does not reduce global mIoU.
- Attack V20 and V21 completed the later trainable-generator attempts:
  - V20 CosPGD-inspired Transformer generator remains ineffective.
  - V21 target-mask distillation generator is valid but weak.
- Attack V22-V25 completed the final per-image white-box comparison branch:
  - PGD vs CosPGD-inspired PGD;
  - PGD vs Momentum PGD vs Universal Perturbation;
  - stronger eps / foreground-focused PGD;
  - targeted `non-person foreground -> person` PGD with target-region metrics.
- Attack V26 consolidates historical rerun code, final best-method configs, no-person targeted visual filtering, and report-ready figures.
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

<<<<<<< HEAD>>>>>>> e078db8 (Add semantic PGD attack report artifacts)
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

<<<<<<< HEAD
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
=======
## Next Steps

- Preserve V7 as the current main trainable adversarial-attack result.
- If more time is available, possible follow-up experiments:
  - tune `LAMBDA_FEATURE` for V7;
  - compare V7 against a boundary-focused generator;
  - evaluate the same attack family on V10 SegMAN-B if a compatible Python 3.10 GPU environment becomes available.
=======
## Attack V13

- Notebook: `attack_v13_pgd_teacher_feature_aspp_eomt_v17_colab.ipynb`
- Target: V17 EoMT-DINOv3
- Attack type: PGD-teacher Feature-ASPP generator
- Trainable: yes
- Perturbation budget: `eps=4/255`
- Main objective:
  - PGD-teacher supervision
  - Feature-ASPP perturbation generator
- Validation split:
  - V17 `712/37` split
  - `37` validation samples
- Result:
  - clean mIoU: `79.232563`
  - attack mIoU: `79.247728`
  - mIoU drop: `-0.015164`
  - max delta: `3.841309`
  - mean delta: `0.086935`
- Interpretation:
  - the PGD-teacher direction still fails to reproduce PGD strength in a reusable trainable generator.

## Attack V14

- Notebook: `attack_v14_transformer_generator_eomt_v17_colab.ipynb`
- Target: V17 EoMT-DINOv3
- Attack type: Patch Transformer generator
- Trainable: yes
- Perturbation budget: `eps=4/255`
- Main objective:
  - targeted foreground-to-background loss
  - EoMT query/object-erasing losses
- Output directory:
  - `outputs/attacks/attack_v14_transformer_generator_eomt_v17`
- Result:
  - clean mIoU: `79.232563`
  - attack mIoU: `78.974004`
  - mIoU drop: `0.258559`
  - foreground-to-background rate increase: `0.010968`
  - max delta: `4.0`
  - mean delta: `3.254454`
- Interpretation:
  - first V17 Transformer-generator result with a small positive mIoU drop.
  - generated adversarial images showed visible horizontal stripe artifacts.

## Attack V15

- Notebook: `attack_v15_dino_feature_conditioned_transformer_eomt_v17_colab.ipynb`
- Target: V17 EoMT-DINOv3
- Attack type: DINO/EoMT feature-conditioned Transformer generator
- Trainable: yes
- Perturbation budget: `eps=4/255`
- Main objective:
  - targeted foreground-to-background loss
  - frozen V17/DINO feature conditioning
  - smoother bilinear decoder
- Output directory:
  - `outputs/attacks/attack_v15_dino_feature_conditioned_transformer_eomt_v17`
- Result:
  - clean mIoU: `79.232563`
  - attack mIoU: `79.883066`
  - mIoU drop: `-0.650503`
  - foreground-to-background rate increase: `0.003256`
  - max delta: `4.0`
  - mean delta: `3.398088`
- Interpretation:
  - conditioning on V17/DINO features did not improve attack effectiveness.

## Attack V16

- Notebook: `attack_v16_sea_loss_transformer_eomt_v17_colab.ipynb`
- Target: V17 EoMT-DINOv3
- Attack type: Transformer generator with SEA-inspired targeted-background loss
- Trainable: yes
- Perturbation budget: `eps=4/255`
- Main objective:
  - targeted foreground-to-background loss
  - class-balanced foreground loss
  - soft IoU suppression
  - prediction divergence
  - query-level object-erasing losses
  - stronger TV regularization
- Output directory:
  - `outputs/attacks/attack_v16_sea_loss_transformer_eomt_v17`
- Result:
  - clean mIoU: `79.232563`
  - attack mIoU: `78.949332`
  - mIoU drop: `0.283231`
  - foreground-to-background rate increase: `0.002704`
  - max delta: `4.0`
  - mean delta: `3.550520`
- Interpretation:
  - best V17 targeted-background trainable result so far.
  - still much weaker than direct PGD.

## Attack V17

- Notebook: `attack_v17_target_person_transformer_eomt_v17_colab.ipynb`
- Target: V17 EoMT-DINOv3
- Attack type: Transformer generator, target-person class attack
- Trainable: yes
- Perturbation budget: `eps=4/255`
- Main objective:
  - push non-person foreground pixels toward `person`
  - exclude GT person pixels from the targeted metric
- Output directory:
  - `outputs/attacks/attack_v17_target_person_transformer_eomt_v17`
- Result:
  - clean mIoU: `79.232563`
  - attack mIoU: `79.503825`
  - mIoU drop: `-0.271262`
  - non-person foreground to person rate increase: `0.000128`
  - max delta: `4.0`
  - mean delta: `3.091201`
- Interpretation:
  - target-person attack is ineffective.
  - it slightly increases mIoU on the small validation split.

## Attack V18

- Notebook: `attack_v18_data_driven_target_transformer_eomt_v17_colab.ipynb`
- Target: V17 EoMT-DINOv3
- Attack type: data-driven target-class Transformer generator
- Trainable: yes
- Perturbation budget: `eps=4/255`
- Main objective:
  - automatically select target class from clean V17 validation confusion
  - train targeted Transformer generator toward selected class
- Target selected:
  - `person`
- Output directory:
  - `outputs/attacks/attack_v18_data_driven_target_transformer_eomt_v17`
- Result:
  - clean mIoU: `79.232563`
  - attack mIoU: `79.523864`
  - mIoU drop: `-0.291301`
  - non-target foreground to selected-target rate increase: `0.000148`
  - max delta: `4.0`
  - mean delta: `3.427649`
- Interpretation:
  - data-driven target selection confirms `person` is the strongest natural confusion target.
  - targeted class forcing still fails to lower V17 global mIoU.

## Attack V19

- Notebook: `attack_v19_per_image_latent_transformer_eomt_v17_colab.ipynb`
- Target: V17 EoMT-DINOv3
- Attack type: per-image latent Transformer generator
- Trainable: yes
- Perturbation budget: `eps=4/255`
- Main objective:
  - untargeted segmentation degradation
  - per-image latent adaptation: `delta_i = G(image_i, z_i)`
  - validation-time latent optimization
- Added metrics:
  - `mIoU_drop`
  - `foreground_mIoU_drop`
  - `pixel_accuracy_drop`
  - `prediction_change_rate`
  - `mean_gt_probability_drop`
  - `per_class_iou_drop`
  - `max_delta_pixel`
  - `mean_delta_pixel`
- Result:
  - clean mIoU: `79.232563`
  - attack mIoU: `79.316299`
  - mIoU drop: `-0.083735`
  - clean foreground mIoU: `78.286862`
  - attack foreground mIoU: `78.374661`
  - foreground mIoU drop: `-0.087799`
  - clean pixel accuracy: `97.574698`
  - attack pixel accuracy: `97.571821`
  - pixel accuracy drop: `0.002876`
  - prediction change rate: `0.002282`
  - clean mean GT probability: `0.753120`
  - attack mean GT probability: `0.743446`
  - mean GT probability drop: `0.009674`
  - max delta: `4.0`
  - mean delta: `2.454069`
  - mean eval latent norm: `4.628114`
- Per-class local degradation:
  - pottedplant: `1.613671`
  - bottle: `1.002132`
  - chair: `0.824402`
  - horse: `0.574746`
  - cow: `0.462599`
  - person: `0.413832`
- Interpretation:
  - V19 is the first mechanism-level shift after repeated universal-generator failures.
  - It is closer to PGD because each image receives its own optimized latent code, while still producing perturbations through a trainable generator.
  - It does not reduce global mIoU, but the nonzero mean GT probability drop and local per-class damage show partial attack pressure.

## Attack V20

- Notebook: `attack_v20_cospgd_transformer_eomt_v17_colab.ipynb`
- Target: V17 EoMT-DINOv3
- Attack type: CosPGD-inspired Transformer generator objective
- Trainable: yes
- Perturbation budget: `eps=4/255`
- Result:
  - clean mIoU: `79.232563`
  - attack mIoU: `79.260748`
  - mIoU drop: `-0.028184`
  - max delta: `1.896852`
  - mean delta: `0.237785`
  - samples: `37`
- Interpretation:
  - objective-level attempt is valid, but it does not transfer PGD/CosPGD strength into a trainable generator.
  - the perturbation is too weak to change global segmentation quality.

## Attack V21

- Notebook: `attack_v21_target_mask_distillation_generator_eomt_v17_colab.ipynb`
- Target: V17 EoMT-DINOv3
- Attack type: target-mask distillation generator
- Trainable: yes
- Perturbation budget: `eps=4/255`
- Main objective:
  - target is a full segmentation mask, not a single class;
  - evaluate mIoU drop, target similarity, and prediction change rate.
- Result:
  - clean mIoU: `79.232563`
  - attack mIoU: `79.092610`
  - mIoU drop: `0.139953`
  - foreground mIoU drop: `0.144938`
  - pixel accuracy drop: `0.043807`
  - prediction change rate: `0.002117`
  - mean GT probability drop: `0.011412`
  - target match rate increase: `0.000509`
  - target foreground match rate increase: `0.001008`
  - mean target probability increase: `0.000660`
  - max delta: `4.0`
  - mean delta: `3.816305`
- Interpretation:
  - the target-mask formulation is more segmentation-native than a single-class target.
  - the global effect remains weak on V17.

## Attack V22

- Notebook: `attack_v22_per_image_cospgd_eomt_v17_colab.ipynb`
- Target: V17 EoMT-DINOv3
- Attack type: per-image PGD and CosPGD-inspired PGD
- Trainable: no
- Main comparison:

| Condition | Target mode | mIoU | mIoU drop | pixel accuracy drop | prediction change rate | mean GT probability drop |
|---|---|---:|---:|---:|---:|---:|
| `pgd_eps4_255_steps10` | untargeted | 5.404918 | 73.827646 | 43.135906 | 0.446279 | 0.432747 |
| `cospgd_eps4_255_steps10` | untargeted | 4.639892 | 74.592671 | 68.085676 | 0.699577 | 0.730801 |
| `pgd_target_fg_to_bg_eps4_255_steps10` | foreground to background | 4.088556 | 75.144007 | 17.901488 | 0.199492 | 0.096481 |
| `cospgd_target_fg_to_bg_eps4_255_steps10` | foreground to background | 4.314037 | 74.918526 | 17.596374 | 0.196435 | 0.090279 |

- Interpretation:
  - CosPGD-inspired untargeted attack is stronger than plain PGD on pixel accuracy drop, prediction change rate, and GT probability drop.
  - Foreground-to-background targeted variants are strong in mIoU drop, but background targeting is less clean as a report story than target-class forcing.

## Attack V23

- Notebook: `attack_v23_momentum_universal_pgd_eomt_v17_colab.ipynb`
- Target: V17 EoMT-DINOv3
- Attack type: PGD, Momentum PGD, Universal Perturbation
- Trainable: no
- Main comparison:

| Condition | mIoU | mIoU drop | pixel accuracy drop | prediction change rate | mean GT probability drop |
|---|---:|---:|---:|---:|---:|
| `pgd_eps4_255_steps10` | 4.875495 | 74.357068 | 45.072659 | 0.464606 | 0.425007 |
| `momentum_pgd_eps4_255_steps10` | 4.569797 | 74.662766 | 48.405085 | 0.499099 | 0.476527 |
| `universal_pgd_eps4_255_train128_epochs2` | 79.573316 | -0.340752 | 0.002639 | 0.002410 | -0.000019 |

- Interpretation:
  - Momentum PGD is a slightly stronger per-image baseline than plain PGD.
  - Universal Perturbation is ineffective under the current training/evaluation setup.

## Attack V24

- Notebook: `attack_v24_stronger_pgd_semantic_targets_eomt_v17_colab.ipynb`
- Target: V17 EoMT-DINOv3
- Attack type: stronger PGD and semantic targeted PGD
- Trainable: no
- Main variants:
  - higher epsilon: `eps=6/255`
  - foreground-focused loss: foreground weights `5` and `8`
  - semantic target masks:
    - `person -> background`
    - `all foreground -> person`
    - `non-person foreground -> person`
- Important results:

| Condition | mIoU | mIoU drop | pixel accuracy drop | target region match |
|---|---:|---:|---:|---:|
| `pgd_eps6_255_steps10` | 3.531095 | 75.701469 | 44.448079 | N/A |
| `pgd_fgweight8_eps4_255_steps10` | 4.109883 | 75.122680 | 34.268106 | N/A |
| `pgd_target_all_fg_to_person_eps4_255_steps10` | 9.567310 | 69.665253 | 13.622665 | 0.914434 |
| `pgd_target_non_person_fg_to_person_eps4_255_steps10` | 9.892714 | 69.339849 | 13.711280 | 0.919196 |

- Interpretation:
  - `eps=6/255` is stronger for untargeted degradation.
  - target-region metrics are necessary for target-class attacks, because full-image pixel accuracy is dominated by background.

## Attack V25

- Notebook: `attack_v25_target_person_stronger_pgd_eomt_v17_colab.ipynb`
- Target: V17 EoMT-DINOv3
- Attack type: targeted semantic PGD sweep
- Trainable: no
- Main target modes:
  - `non_person_foreground_to_person`
  - `all_foreground_to_person`
- Added metrics:
  - `target_region_match_rate`
  - `target_region_change_rate`
  - `target_region_accuracy_drop`
  - target-region clean/adversarial accuracy
- Best targeted result:

| Condition | target match | target change | target acc drop | mIoU drop | pixel accuracy drop | max delta | mean delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pgd_target_non_person_fg_to_person_eps6_255_steps20` | 0.993359 | 0.993568 | 0.918506 | 73.173467 | 14.799716 | 6.0 | 3.231018 |

- Interpretation:
  - this is the strongest targeted semantic attack for the final report.
  - it is closer to the assignment example of making a segmentation model hallucinate a chosen class where it should not.

## Attack V26

- Notebook: `attack_v26_final_report_attacks_eomt_v17_colab.ipynb`
- Target: V17 EoMT-DINOv3
- Attack type: final report integration notebook
- Trainable: no for the final selected attacks
- Purpose:
  - rerun historical comparisons in one place instead of relying only on copied result tables;
  - preserve V10/V22/V23/V24/V25 comparison code;
  - generate final report visualizations for the best untargeted and targeted attacks.
- Historical rerun coverage:
  - V10 FGSM/PGD sanity check;
  - V22 PGD vs CosPGD;
  - V23 Momentum PGD / Universal Perturbation;
  - V24 stronger PGD and semantic targets;
  - V25 targeted-person strengthening.
- Final visual conditions:
  - untargeted best: `cospgd_untargeted_eps6_255_steps20`
  - targeted best: `pgd_target_non_person_fg_to_person_eps6_255_steps20`
- Targeted visualization filter:
  - exclude examples whose ground truth already contains `person`;
  - require non-person foreground pixels so the target task is meaningful.
- Expected output files:
  - `outputs/attacks/attack_v26_final_report_attacks_eomt_v17/attack_summary.csv`
  - `outputs/attacks/attack_v26_final_report_attacks_eomt_v17/final_report_comparison.csv`
  - `outputs/attacks/attack_v26_final_report_attacks_eomt_v17/historical_rerun/historical_attack_summary.csv`
  - `outputs/attacks/attack_v26_final_report_attacks_eomt_v17/visualizations/`
- Interpretation:
  - V26 should be the notebook handed to teammates for final report integration.

## Next Steps

- Run V26 once in Colab with `RUN_HISTORICAL_RERUN = True` to regenerate the historical comparison CSVs.
- After the first full V26 run, set `RUN_HISTORICAL_RERUN = False` if only figures or report text need to be regenerated.
- Use V26 figures for final visual evidence:
  - one untargeted CosPGD-inspired example grid;
  - one targeted non-person foreground to person example grid.
- In the report:
  - preserve V7 as the strongest trainable generator result;
  - preserve V10 as the sanity check proving V17 is attackable;
  - use V22-V25 for the final PGD-family comparison;
  - use V25/V26 targeted-person results for the cleanest semantic targeted attack story.