# Notebook Review Suggestions

## Overall Verdict

The notebook is already substantially stronger than a typical course submission. It reads like a report rather than a code dump, covers the assignment PDF's required questions, and uses figures to support most design choices. I would not restructure the whole report.

The main remaining work should be a targeted cleanup pass: remove stale text from previous edits, fix a few evidence-vs-interpretation mismatches, and improve the readability of the widest/tallest figures.

## Must Fix Before Submission

1. Remove repeated text in Section 4.2.
   - The sentence beginning "The two configurations differ in targeting mode..." is repeated three times.
   - The sentence beginning "The figure shows only the targeted run..." is also repeated three times.
   - This is the most visible professionalism issue in the current draft.

2. Fix the malformed segmentation reflection heading.
   - The text currently renders as: `... we did not change them. ## 3.6 Reflection`
   - Put `## 3.6 Reflection` in its own markdown cell or add a blank line before it.

3. Fix stale Chapter 4 references.
   - Chapter 4 intro still says `Section 4.3.1`, `4.3.2`, and `4.3.3`.
   - Current notebook sections are `4.1`, `4.2`, and `4.3`.

4. Fix stale text around Table 2.
   - Section 3.2 says "The joint Kaggle leaderboard score is shown in the rightmost column when applicable", but the rendered Table 2 no longer has a joint-score column.
   - It also says `V13` appears in the table, but the table skips V13.
   - The sentence "For V10-V17 it is the leaderboard score minus the ConvNeXt-Small classification contribution" needs the exact formula or should be simplified.

5. Rewrite the Lab-vs-Kaggle scatter interpretation.
   - The current interpretation uses a `y=x` diagonal, but x is validation mIoU and y is Kaggle Dice. These metrics are not directly comparable.
   - The text also says V11-V16 are on the "opposite side", but visually they are still above the diagonal.
   - Better interpretation: use the plot for rank/model-selection mismatch, not absolute y=x calibration. For example, "V4/V5 improve val mIoU while lowering Kaggle Dice; V17 has the best Kaggle score despite a lower val mIoU than V16."

6. Correct the cross-task weak-class synthesis.
   - Section 3.5 says `pottedplant` and `bottle` are weak in both columns, but bottle has high segmentation IoU.
   - Section 5.2 says `bicycle`, `pottedplant`, and `bottle` are weak in both tasks, but bicycle has classification AP = 1.00.
   - Section 5.2 says "Disagreement only on dog", but bicycle, boat, diningtable, and bottle also show task-asymmetric behavior.
   - Suggested framing: "pottedplant is weak in both; bicycle and dog are segmentation-specific failures; diningtable and bottle are more classification-side failures."

7. Fix the ConvNeXt component table input size.
   - The table says Stem maps `224x224x3 -> 56x56x96`, while the chapter and SVG discuss a `320x320` input.
   - For 320 input, the stem output is `80x80x96`. The diagram is right; the table should match it.

8. Make the Section 4.2 attack table narrower.
   - In the rendered HTML preview, the right side of the table is clipped at normal report width.
   - Keep only the columns the text uses, or split into "budget/setup" and "outcome" tables.

## Recommended Content Edits

1. Express segmentation loss-ablation deltas in the correct unit.
   - Current text says validation mIoU rises by `+1.44` and `+1.16`.
   - The plotted values are `0.625 -> 0.639` and `0.625 -> 0.636`, so write `+0.014` and `+0.012`, or `+1.44` and `+1.16` percentage points.

2. Tone down causal certainty in a few places.
   - "The lesson is unambiguous" should become "The pattern suggests".
   - "That is strong evidence the cause is the dataset rather than the model" should become "This points to dataset and resolution effects, though architecture still matters."

3. Be careful with pretrained-data provenance.
   - "ImageNet-1K is independent of PASCAL VOC" is probably fine but should be phrased as a sourced statement.
   - "`LVD-1689M` contains no PASCAL VOC images" is a strong claim. If you cannot cite a model card/source, soften it to "we use this checkpoint because it is not VOC-supervised and no VOC overlap is documented."

4. Make the Cityscapes comparison metric cleaner.
   - The text compares Cityscapes 5-class mIoU (`0.509`) against VOC validation mIoU (`0.793`), but the VOC number is likely across the full VOC class set, not exactly the same five classes.
   - Best fix: compute VOC validation mIoU restricted to the same five overlap classes and compare against that.
   - If you do not compute it, say "against the model's overall VOC validation mIoU" rather than "on the same underlying classes."

5. Remove or qualify speculative numeric gains.
   - Section 5.3 says a new resolution schedule would "probably" add `0.005` to `0.010` mIoU.
   - Unless this was measured, remove the numeric range or label it explicitly as a hypothesis.

6. Reconsider the "mine PASCAL VOC 2007 and VOC 2008" limitation.
   - The assignment repeatedly says to use the provided training set. Extra VOC splits may be interpreted as outside the intended setup.
   - Safer wording: "If additional labeled data were allowed..." or replace with cross-validation, stronger augmentation, pseudo-labeling, or more careful split design.

## Figure-Specific Suggestions

1. Split the trainable-generator-vs-PGD image grid.
   - The current figure is 1418x2952 and spans multiple screens. It is hard to compare the two halves without scrolling.
   - Use either three examples instead of five, or make two separate figures: one trainable-generator failure, one PGD success.

2. Add larger column headers to the attack grids.
   - Classifier attack: use `Clean image P(aeroplane)=...`, `10x perturbation`, `Adversarial image P(aeroplane)=...`.
   - Segmentation attack: add a small legend for target color and original classes.

3. Fix the clipped Ch2 SVG annotation.
   - The right-side italic annotation is too close to the SVG boundary. Move it left or wrap the "total ≈ 50M params" sentence.

4. Add value labels where the exact number is central to the argument.
   - Good candidates: AsymmetricLoss winners/losers, pivot waterfall bars, cross-task per-class chart for the bottom 4 classes.

5. Add a y-axis zoom note to the V17 training curve.
   - The curve is useful, but the y-axis is tightly zoomed. State this in the caption or title to avoid over-reading small oscillations.

6. Add captions/alt text for exported images.
   - nbconvert reported missing alternative text on 15 images. This is not a core grading issue, but stronger captions improve report quality.

## Language And Flow

- The abstract and opening summary are strong. They set up the story and metrics clearly.
- The chapter ordering is logical: dataset -> classification -> segmentation -> attacks -> discussion -> submission.
- The best writing is where the notebook ties one figure to one design decision, especially in the loss ablation and adversarial sections.
- The main language risk is overconfident interpretation from observational comparisons. Keep strong claims for directly controlled ablations; use more cautious wording for cross-run comparisons, hidden-test comparisons, and domain-shift arguments.
- Remove local-development residue before final Kaggle submission if the rendered outputs still show Windows paths. Kaggle rerun should replace these with `/kaggle/...`, but a saved local-output notebook would look less polished.

## Suggested Edit Order

1. Fix all stale/repeated text and malformed headings.
2. Correct the three evidence mismatches: lab-vs-Kaggle scatter, cross-task weak classes, segmentation loss-ablation units.
3. Narrow the attack table and split/reduce the tallest attack grid.
4. Do a final rendered-HTML pass at roughly 1100px width.
5. Run the Kaggle notebook once from a clean session and confirm `submission.csv` is produced.

