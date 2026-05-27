# Findings: Kaggle Submission Integration

## Source Material Inventory

### Branch: segmentation (current local branch)

- Best segmentation model: **V17 EoMT-DINOv3** fine-tuned on 712/37
  re-split; Kaggle `0.89139`; checkpoint at
  `Semantic segmentation/outputs/checkpoints/eomt_dinov3_v17_merge_val_lr1e5/best/`
  (1.2 GB, also on Google Drive).
- Best inference recipe: multi-scale TTA `{496, 512, 528}` no flip
  (`scripts/predict_eomt_tta.py`).
- Cityscapes external generalization study: 5-class transferable mIoU
  `0.5088` (V17) vs `0.4168` (V10 SegMAN-B). Artifacts at
  `outputs/cityscapes_generalization/eomt_dinov3_v17_tta_ms496_512_528_noflip/`.
- "Train" class visual concept investigation: 4 paired panels at
  `outputs/figures/train_class_voc_vs_cityscapes/` proving VOC train ≠
  Cityscapes tram.
- Adversarial attack work: 26 versions, final at
  `outputs/attacks/attack_v26_final_report_attacks_eomt_v17/` with
  CosPGD untargeted (mIoU drop 79.16) + targeted non-person->person
  (98.47% match).
- Trainable-vs-PGD side-by-side visualization at
  `outputs/attacks/trainable_vs_pgd_visual/` showing why V17 resists
  trainable generators but collapses under per-image PGD.

### Branch: Image-Classification (teammate)

- Best classification model: **ConvNeXt-Small 320**, val mAP `0.900`,
  Kaggle classification Dice `0.898`.
- Three-stage training: head-warmup -> full fine-tune -> Stage-3 retrain
  on all 749 images.
- Per-class threshold calibration via F1 search on val.
- Horizontal-flip TTA.
- Adversarial attack: targeted aeroplane, FGSM 26.6% / PGD 100% success
  with pixel L_inf mean `0.00687`.

### Branch: main / Adversarial-attacks (placeholder)

- Empty placeholder dirs only; no integration cost.

## Reference Materials

- Reference starter notebook: `ga2_group_6.ipynb` (4-chapter scaffold:
  Overview / Classification / Segmentation / Attack / Discussion).
- Assignment PDF: `CV_GA2.pdf`. Key grading guidance:
  - "a small part of the grade will reflect the actual performance"
  - main signal: correctness, design rationale, insight
  - Section 2.3 explicitly asks for reflection on realism, defenses,
    deployability
  - Section 2.4 asks for lecture links + real-world critique

## Useful Numbers Already Established

| Item | Value | Source |
|---|---|---|
| Classification Kaggle Dice | 0.898 (×2 from 0.449 displayed) | classification README |
| Segmentation Kaggle (with classification) | 0.89139 (V17) | segmentation kaggle_scores.md |
| Cityscapes overlap mIoU (5-class) | 0.5088 (V17) | cityscapes summary |
| Attack on classifier success | 100% PGD targeted | classification README |
| Attack on segmenter untargeted | mIoU 79.23 -> 0.07 | attack_v26 final CSV |
| Attack on segmenter targeted | 98.47% target match | attack_v26 final CSV |

## Constraints / Risks Discovered

- Kaggle notebook runtime is limited and untrusted; we cannot run V17
  inference on 750 test images inside the notebook reliably (V17 weights
  are 1.2 GB, EoMT-DINOv3 requires Transformers >= 5.4 which may not be
  the default Kaggle image). Plan: precomputed predictions go into the
  Kaggle dataset, the notebook only loads them and rebuilds the
  submission CSV via the provided RLE encoder.
- Trainable adversary work covers PDF Figure 3 literally; per-image PGD
  does not. Both should appear in the attack chapter so the TA cannot
  claim we ignored the PDF wording.
- Cityscapes evaluation requires the dataset to be available; we will
  not re-run it inside the notebook. We import the precomputed
  metrics.csv + visualization panels from our dataset.
- Local ipykernel in `gpu_env` cannot start cleanly (pkg_resources
  incompatibility with old `pkgutil.ImpImporter`). nbconvert
  `--execute` fails because the kernel dies before replying to
  kernel_info. The workaround for cell-output injection during
  authoring is `plan/scripts/execute_and_inject.py`; on Kaggle the
  built-in kernel runs the cells natively without this script.
- Picked diverse Chapter 1 sample IDs by filtering for images with
  exactly 1-2 labels: `train_0` (cow), `train_5` (motorbike+person),
  `train_7` (person+train), `train_8` (chair+tvmonitor).
