# DM-5: Chapter 5 — Discussion (Section 2.4)

**Chapter goal**: deliver the discussion the PDF explicitly grades as
the most important section. Reflect across all three prior tasks,
link to course lectures, and give an honest deployment-readiness
verdict. Cityscapes external generalisation is the centrepiece; the
train-vs-tram label-mismatch finding gives the chapter its sharpest
critical-thinking moment.

## Proposed cell list (11 cells: 8 markdown + 2 light code + 1 figure display)

| # | Type | Content |
|---|---|---|
| 5.0 | md | Chapter header + roadmap of 5.1 through 5.5 |
| 5.1 | md | 5.1 Lab vs deployment: Cityscapes external evaluation setup |
| 5.2 | code | Cityscapes 5-class transferable mIoU horizontal bar chart |
| 5.3 | md | Per-class interpretation: car / person / bus high, motorbike / bicycle low, train excluded |
| 5.4 | md | The train-vs-tram finding: label-equal-but-concept-different |
| 5.5 | code | Display one representative train-vs-tram paired panel (steam locomotive vs tram) |
| 5.6 | md | Visual interpretation + methodological consequence |
| 5.7 | md | 5.2 Cross-task observations: shared weak classes, shared adversarial fragility |
| 5.8 | md | 5.3 Connections to lectures: five explicit links |
| 5.9 | md | 5.4 Limitations and what we would do differently (four-item structured list) |
| 5.10 | md | 5.5 Deployment readiness verdict across three dimensions |

## Assets to add

- `dataset/tables/cityscapes_transferable_iou.csv` extracted from `Semantic segmentation/outputs/cityscapes_generalization/.../metrics.csv`
- `dataset/figures/cityscapes_train_vs_tram_pair.png` copied from `Semantic segmentation/outputs/figures/train_class_voc_vs_cityscapes/pair_01_voc318_vs_frankfurt_000001_066092_leftImg8bit.png`
- One helper function in a new module (`discussion_metrics.py` or extend `segmentation_metrics.py`) for the Cityscapes per-class chart

## Hard-rule compliance

- Numerical-Claim Sourcing: every number in 5.1-5.3 is loaded from `dataset/tables/cityscapes_transferable_iou.csv`.
- Train / Val / Test Split Disclosure: 5.1 names the 500-image Cityscapes val split and the zero-training condition.
- Metric-Definition Disambiguation: 5.1 contrasts "Cityscapes 5-class transferable mIoU" against the chapter-3 internal val mIoU explicitly.

## Sub-options

### A. Cityscapes chart style
- **A1 (recommended)**: horizontal bar chart, 5 classes sorted by IoU, coloured by category (large-vehicle / small-vehicle), train annotated separately as "excluded from primary metric".
- A2: table only.

### B. Train-vs-tram visual evidence
- **B1 (recommended)**: one most-dramatic paired panel + interpretation paragraph.
- B2: two paired panels side by side.
- B3: text only with caption linking to the panel.

### C. Lecture connections format
- **C1 (recommended)**: bullet list of five named lecture topics each mapped to a concrete project decision.
- C2: prose paragraph.

### D. Limitations format
- **D1 (recommended)**: four structured items (dataset size, val split methodology, preprocessing-locked architecture, TTA scope), each with a one-sentence "what we would do with another week".
- D2: free paragraph.

### E. Deployment readiness verdict layout
- **E1 (recommended)**: three dimensions (in-distribution / cross-domain / adversarial), each with a one-line verdict.
- E2: single narrative paragraph.

## Optional insights for user

### Insight #5: K-fold CV would have caught the val/test mismatch earlier
A single sentence in 5.4 acknowledging that the V4/V5 lab vs Kaggle inversion would have surfaced earlier under 5-fold CV. Costs nothing, demonstrates methodological honesty.

### Insight #6 (strongly recommended): name the failure mode "label-equal-but-concept-different"
Formalise the train-vs-tram observation as a named failure mode distinct from the classical domain shift / covariate shift / label shift taxonomy. We call it **ontological mismatch**: the labels align but the underlying visual concepts do not. This is a small but genuine conceptual contribution and gives the TA something memorable to grade.

## Estimated effort

About 30 minutes including CSV extraction, one helper, one figure copy, eleven cells of writing, and Chrome screenshot review with 1800 px slices.

## Pending user input

- A / B / C / D / E selections.
- Whether to include Insight #5 and #6 (strongly recommend at least #6).
