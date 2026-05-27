# KU Leuven Computer Vision (H02A5a) — Group Assignment 2

**Course:** Computer Vision (H02A5a), KU Leuven, academic year 2025-2026
**Assignment:** Group Assignment 2 — *In the name of deep learning*
**Group:** 6 (Kaixi Yao, Taicheng Liu, Zhenyang Li, Enmin Lin)
**Kaggle leaderboard:** **0.89139** joint Dice (PASCAL VOC 2009 subset, public test)

This repository is the final deliverable of three deep-learning tasks built on the same `749`-image PASCAL VOC 2009 subset distributed for the in-class Kaggle competition:

1. **Multi-label image classification** — 20-class binary labels per image (Sect. 2.1 of the assignment PDF).
2. **Semantic segmentation** — 21-class (20 VOC + background) per-pixel labels (Sect. 2.2).
3. **Adversarial attacks** — white-box attacks on both trained models (Sect. 2.3).
4. **Discussion** — cross-domain Cityscapes evaluation, cross-task observations, limitations, and deployment-readiness verdict (Sect. 2.4).

## How to read this repository

The single source of truth is one Jupyter notebook:

> **`kaggle_submission/notebook/group6_final.ipynb`**

It reads like a written report: every figure, table, and number you see has either a paragraph of interpretation next to it or a CSV under `dataset/tables/` that produced it. The notebook is structured as Chapters 0-6 that follow the PDF section numbering.

If you are reviewing the work, open the notebook directly. Everything else in this repository is supporting material the notebook consumes at runtime.

| You want to... | Look at |
|---|---|
| Read the report end-to-end | `kaggle_submission/notebook/group6_final.ipynb` |
| Reproduce the submission CSV | `kaggle_submission/SUBMISSION_GUIDE.md` |
| Navigate the supporting artefacts | `kaggle_submission/STORY_MAP.md` |
| See per-chapter design decisions | `kaggle_submission/plan/decisions/DM-*.md` |
| See the implementation pipeline that produced the notebook | `kaggle_submission/plan/scripts/` |
| Look at the actual training code per task | branches `Image-Classification` and `segmentation` |

## Headline results

| Track | Metric | Score | Evaluation set |
|---|---|---|---|
| Image classification | Average Precision (mAP) | `0.900` | local 150-image val split |
| Image classification | Kaggle Dice (classification half) | `0.898` | hidden 750-image test |
| Semantic segmentation | mean IoU | `0.793` | local 37-image val split |
| Semantic segmentation | Kaggle Dice (segmentation half) | `0.885` | hidden 750-image test |
| Joint submission | Kaggle Dice (full 1500 rows) | **`0.89139`** | hidden test |
| Adversarial — classifier | PGD targeted success rate | `100%` | 64 val images |
| Adversarial — segmenter | CosPGD untargeted prediction-change rate | `99.58%` | 37 val images |
| Adversarial — segmenter | Targeted PGD target-region match rate | `98.24%` | 25 strict-no-person val images |
| Cross-domain (Cityscapes) | 5-class transferable mIoU on V17 | `0.509` | 500 Cityscapes val, no training |

## Production models

- **Classification:** ConvNeXt-Small (`49.86M` params) initialised from `ConvNeXt_Small_Weights.IMAGENET1K_V1`, fine-tuned with AsymmetricLoss in three stages (head warm-up → full fine-tune → retrain on all 749 images).
- **Segmentation:** **V17** = EoMT-DINOv3-Large (`314.8M` params) with the DINOv3 ViT-L/16 backbone pretrained on `LVD-1689M`, fine-tuned on a `712 / 37` split with multi-scale TTA at `{496, 512, 528}` at inference.
- **Adversarial attacks:** PGD (classifier) and CosPGD + Targeted PGD (segmenter). The segmentation chapter also reports `21` trainable-generator variants that all failed against V17, contrasted against per-image PGD which succeeds completely.

None of the pretrained weights used during this assignment were trained on PASCAL VOC images (the assignment's no-VOC-pretraining clause); the notebook documents the provenance per backbone.

## Repository layout

```
.
├── README.md                            # this file
├── .gitignore
├── CV_GA2.pdf                           # the original assignment specification
├── ga2_group_6.ipynb                    # the lecturer-provided starter notebook
└── kaggle_submission/                   # all final-submission artefacts
    ├── notebook/
    │   └── group6_final.ipynb           # THE report (open this)
    ├── dataset/                         # what the notebook reads at runtime
    │   ├── code/                        # Python modules imported by the notebook
    │   ├── figures/                     # SVG architecture diagrams + attack panels
    │   ├── submissions/                 # pre-merged submission CSVs
    │   └── tables/                      # every CSV cited in the notebook
    │       # dataset/weights/ and dataset/predictions/ are NOT in git;
    │       # they live in the Kaggle dataset to keep the repo small
    ├── outputs/
    │   └── submission.csv               # the 1500-row leaderboard submission
    ├── plan/                            # per-track planning and review docs
    │   ├── task_plan.md
    │   ├── findings.md
    │   ├── progress.md
    │   ├── decisions/                   # per-chapter decision memos (DM-0..6)
    │   └── scripts/                     # repeatable tooling (rendering, audits)
    ├── SUBMISSION_GUIDE.md              # step-by-step Kaggle submission walkthrough
    └── STORY_MAP.md                     # navigation map for the team
```

### What is intentionally not in this branch

- **`Image classification/`, `Semantic segmentation/`, `Adversarial attacks/`** — the per-track development trees, which contain experiment-level scripts, raw checkpoints, ablation outputs, and intermediate notebooks. These are preserved on their dedicated branches:
  - `Image-Classification` branch — ResNet-50, EfficientNet, ConvNeXt-{Tiny, Small, Base, Large}, ConvNeXt-V2, ViT-{B, L}/16 training scripts and per-experiment outputs.
  - `segmentation` branch — V1..V17 segmentation pipeline (MMSegmentation + custom EoMT-DINOv3 fine-tune), the V26 adversarial-attack experiments, the Cityscapes cross-domain evaluation, and the trainable-generator-versus-PGD visualisation.
- **`kul-computer-vision-ga-2-2026/`** — the raw competition data, which is provided directly by Kaggle and is not redistributed in this repository.
- **`kaggle_submission/dataset/weights/`** — `1.2 GB` of safetensors and `190 MB` of `.pth` files. These are distributed through the Kaggle dataset `kaggle.com/datasets/enminlin/kul-cv-ga2-group6-dataset` and re-attached at notebook runtime.

## Reproducing the submission

The full Kaggle workflow is documented in `kaggle_submission/SUBMISSION_GUIDE.md`. The short version is:

1. Upload `kaggle_submission/dataset/` as a Kaggle dataset with slug `kul-cv-ga2-group6-dataset`.
2. Create a new Kaggle notebook in the competition `kul-computer-vision-ga-2-2026`.
3. Attach both the competition data and the uploaded dataset.
4. Import `kaggle_submission/notebook/group6_final.ipynb`.
5. *Run All → Save Version → Save & Run All (Commit)*.
6. Submit `submission.csv` to the competition.

The notebook is non-training by design (training was offline; see the per-track branches). On Kaggle it loads the pre-trained checkpoints, regenerates every figure inline, and writes the leaderboard CSV in about two minutes.

## License

Course coursework. The PASCAL VOC 2009 subset is governed by the original VOC licence; the Cityscapes evaluation uses the Cityscapes Dataset under its non-commercial research licence. No redistribution of either dataset is performed by this repository.
