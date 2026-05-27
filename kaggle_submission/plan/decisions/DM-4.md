# DM-4: Chapter 4 — Adversarial Attacks

**Chapter goal**: deliver PDF Section 2.3 for both the trained
classifier and the trained segmenter, with the trainable-vs-PGD
finding as the central insight. No new training, no model reloading
beyond what is already in the dataset; visual examples come from
pre-rendered PNGs.

## Proposed cell list (13 total: 8 markdown + 5 light code)

| # | Type | Content |
|---|---|---|
| 4.0 | md | Chapter header + setup (white-box, frozen, L∞ budget, two task definitions) |
| 4.1 | md | 4.3.1 Attack on classifier: targeted FGSM + PGD on absent-aeroplane val images |
| 4.2 | code | Load `classification_attack_results.csv`, render styled table |
| 4.3 | md | Classification attack interpretation: 26.6 % FGSM, 100 % PGD, pixel L∞ 0.00687 |
| 4.4 | code | Display pre-rendered `classification_adversarial_aeroplane_examples.png` |
| 4.5 | md | Visual interpretation: perturbation invisible, confidence pushed from ~0 to >0.9 |
| 4.6 | md | 4.3.2 Attack on segmenter: final two configs (CosPGD untargeted + PGD targeted non-person -> person) |
| 4.7 | code | Load `segmentation_attack_results.csv`, render styled table |
| 4.8 | md | Segmentation attack interpretation: mIoU collapse, 98.47 % target-pixel hit rate |
| 4.9 | md | 4.3.3 Trainable adversary vs per-image PGD on the same model (the central insight) |
| 4.10 | code | Side-by-side: trainable generator grid vs PGD grid on V17 |
| 4.11 | md | Mechanism interpretation: V17 EoMT robust to universal generators but vulnerable to per-image gradient |
| 4.12 | md | 4.3.4 Reflection: realism, defenses, human visibility, cross-task symmetry of vulnerability |

## Visual assets to copy into `dataset/figures/`

- `classification_adversarial_aeroplane_examples.png` (already present)
- `segmentation_trainable_attack_grid.png` (from `outputs/attacks/trainable_vs_pgd_visual/trainable_generator_attack_grid.png`)
- `segmentation_pgd_attack_grid.png` (from same directory)
- `segmentation_v26_untargeted_grid.png` (from `outputs/attacks/attack_v26_final_report_attacks_eomt_v17/visualizations/final_untargeted_grid.png`)
- `segmentation_v26_targeted_grid.png` (same directory, `final_targeted_grid.png`)

## CSV tables to author

- `dataset/tables/classification_attack_results.csv` from `weights/convnext_small_320-.../metrics/adversarial_aeroplane.csv` (renamed columns)
- `dataset/tables/segmentation_attack_results.csv` reformatted from V26 `final_report_comparison.csv` (one row per final attack with mIoU drop / prediction change rate / target match rate / pixel L∞)

## How the five hard rules are satisfied

- Pretrained-Weight Provenance: not applicable (frozen pre-trained models, no new pretraining).
- Numerical-Claim Sourcing: every attack number is loaded from a CSV under `dataset/tables/`.
- Train / Val / Test Split Disclosure: 4.0 names the 64 absent-aeroplane val images (classification) and the 37 val images (segmentation).
- Metric-Definition Disambiguation: result tables explicitly tag each metric's scope (target region only / full image / single-class success rate).
- Loss-Ablation Pairing: 4.3.3 is itself a method ablation (universal generator vs per-image gradient).

## Sub-options for user

### A. Trainable-adversary insight as its own subsection
- **A1 (recommended)**: keep as 4.3.3 with side-by-side visual comparison and mechanism interpretation.
- A2: fold into 4.3.2 interpretation paragraph.

### B. Visual example sourcing
- **B1 (recommended)**: load pre-rendered PNGs from the dataset.
- B2: re-render attacks in-notebook (much heavier, requires running models on Kaggle).

### C. Segmentation result table layout
- **C1 (recommended)**: single 2-row table (untargeted, targeted) with 4-5 key metrics.
- C2: two separate small tables.

### D. Trainable-adversary historical evidence depth
- **D1 (recommended)**: one paragraph noting 14+ trainable variants tried in V1-V21, all ineffective on V17 EoMT, plus the visual comparison.
- D2: a representative four-row table (V1 tiny, V7 Feature-ASPP, V8 EoMT version, V19 per-image latent).
- D3: full list of every trainable attempt (too cluttered).

### E. Segmentation visual examples to include
- **E1 (recommended)**: trainable-vs-PGD comparison (anchors 4.3.3) plus one V26 final grid (anchors 4.3.2).
- E2: only V26 grids, drop the trainable-vs-PGD comparison.
- E3: V26 untargeted + V26 targeted + trainable-vs-PGD all three (figure overload).

## Optional insights for user

### Insight #3: Attack budget visualisation
A small paragraph in 4.3.4 reflection making the pixel L∞ numbers
tangible: 0.00687 for classification (~1.7 / 255) and 6 / 255 for
segmentation, both under 3% intensity difference per channel and
completely invisible to a human reviewer. Costs one sentence; no
figure required.

### Insight #4: Cross-task attack-success symmetry
A reflection paragraph noting that two different tasks, two different
models, and two different attack methods all converged on 99-100%
success rate. The redundancy across tasks did not buy any
cross-architecture robustness, which is itself a defense-mechanism
observation.

## Estimated effort

About 35 minutes including figure copy, CSV authoring, helper code,
narrative writing, and Chrome screenshot review.

## Deliverables after approval

1. Five PNGs copied into `dataset/figures/`.
2. Two CSVs in `dataset/tables/`.
3. One helper module in `dataset/code/` (`attack_metrics.py`) for the
   two styled tables and the side-by-side image display.
4. 13 cells appended to `notebook/group6_final.ipynb` after Chapter 3.
5. Updated `STORY_MAP.md` chapter status to draft v1.
6. Discord ping when ready for verification.
