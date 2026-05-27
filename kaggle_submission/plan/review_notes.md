# Notes: Notebook Review

## Sources Reviewed

- Notebook: `D:\BaiduNetdiskWorkspace\Leuven\8th\Computer Vision\assignment\Group2\kaggle_submission\notebook\group6_final.ipynb`
- Assignment PDF: `D:\BaiduNetdiskWorkspace\Leuven\8th\Computer Vision\assignment\Group2\CV_GA2.pdf`
- Project plan files: `task_plan.md`, `findings.md`, `progress.md`
- Extracted notebook visuals: `plan/review_assets/nb_cell*.png`, HTML tables, and inline SVG screenshots.
- Rendered notebook HTML preview: `plan/review_assets/group6_final_review.html`

## Main Findings

- Overall submission is strong and report-like. It covers all required sections: classification, segmentation, adversarial attacks, discussion, and final submission generation.
- The narrative is mostly polished, but several stale edits remain after iterative revisions.
- The highest-risk issues are factual or structural, not style-only:
  - repeated sentences in Section 4.2,
  - malformed `## 3.6 Reflection` heading,
  - stale subsection references in Chapter 4,
  - stale Table 2 description after columns were removed,
  - incorrect interpretation of the validation-vs-Kaggle scatter,
  - overstatement in cross-task weak-class synthesis.
- Most figures are readable and useful. The weakest visual is the very tall trainable-generator-vs-PGD grid, because it spans multiple screens and loses column context.
- Several interpretations are thoughtful, especially the AsymmetricLoss ablation, classifier attack examples, targeted segmentation attack, and Cityscapes cross-domain discussion.
- Some interpretations need to be tightened where the visual evidence does not support the claim exactly.

## Visual Review Notes

- Class-distribution histogram: clear. Optional improvement: add count labels on bars.
- Four training examples: clear and now avoids the old `train_5` ambiguity.
- BCE vs AsymmetricLoss chart: clear. Add value labels for top positive/negative deltas if space allows.
- From-scratch baseline figure: useful, but right panel labels are dense. Consider splitting trajectory and per-class gaps into separate figures.
- Training curve: readable. Note explicitly that the V17 curve uses a zoomed y-axis so tiny oscillations are not overinterpreted.
- Pivot waterfall: much improved after removing V1 as a delta. Interpretation should be softened from "unambiguous" to "suggests".
- Lab-vs-Kaggle scatter: visually readable, but the `y=x` line is conceptually questionable because x is mIoU and y is Dice.
- Cross-task per-class chart: clear, but the interpretation overstates shared weak classes.
- Attack classifier examples: visually strong. Column titles should say `P(aeroplane)` rather than just `p`.
- Targeted segmentation attack grid: strong, but would benefit from a tiny color legend explaining original class colors vs target `person`.
- Trainable-vs-PGD grid: too tall for report reading. Split into two figures or reduce to three examples.
- SVG architecture diagrams: Ch3 and Ch4 are crisp. Ch2 has a right-side annotation clipped at the canvas edge.
- Attack results table is too wide in rendered HTML; the rightmost column is clipped at normal report width.
- nbconvert warns that 15 image outputs lack alt text. Captions exist, so this is not a grading blocker, but adding alt text or stronger captions would be cleaner.

## Assignment Coverage Notes

- PDF Section 2.1 coverage is strong: from-scratch vs pretrained, layer roles, loss, learning curves, mistakes, and performance are covered.
- PDF Section 2.2 coverage is strong: classification-vs-segmentation differences, architecture, loss ablation, learning curve, performance, and mistakes are covered.
- PDF Section 2.3 coverage is strong: attacks are defined, losses and perturbation budgets are stated, visual examples are included, and realism/defense reflection is present.
- PDF Section 2.4 coverage is strong: Cityscapes cross-domain evaluation, limitations, and deployment readiness are discussed.
- The final submission generation section is appropriate for Kaggle constraints.

