# Task Plan: Kaggle Submission Integration

## Goal

Produce **one** Kaggle notebook that serves as the team's final submission for
Computer Vision GA2 (KUL H02A5a). The notebook is graded by the TAs as a
report, must run end-to-end on Kaggle within resource limits, and must
generate the leaderboard `submission.csv`.

Heavy artifacts (model weights, large CSVs, figures) live in a separate
Kaggle dataset that the notebook imports at runtime. The notebook itself
should read like a written report: discussion + figures + small
demonstrative code snippets only. All production code is in the dataset.

## Hard Requirements From the Assignment PDF

- Cover all three deliverables: 2.1 classification, 2.2 segmentation,
  2.3 adversarial attacks.
- Include the 2.4 discussion with real-world / lecture links.
- Be self-contained: the TA reads the notebook as the report; cannot
  follow links into our private git history (the team uses git for
  internal coordination only, not for grading).
- End with a valid `submission.csv` produced via the provided
  `submit_df_to_competition`-style RLE encoding.

## Non-Goals

- Do not merge git branches just to please grading. The TA never sees git.
- Do not retrain anything from scratch inside the notebook. All training
  was done offline; the notebook only loads the final artifacts.
- Do not include broken / pruned experiments unless they carry a useful
  lesson (negative results that informed our choices are kept; smoke
  runs are dropped).

## Phases

- [ ] Phase 0: align on the notebook outline + interaction protocol (this
      step is collaborative; user approves outline before any drafting)
- [ ] Phase 1: stand up the Kaggle dataset folder structure (model
      weights, submission CSVs, key figures, per-class tables)
- [ ] Phase 2: draft notebook scaffold (markdown headers, empty code
      cells with placeholders)
- [ ] Phase 3: fill each chapter one at a time, following the
      per-chapter decision-memo workflow described in `progress.md`
- [ ] Phase 4: end-to-end dry run of the notebook on a Kaggle-like
      environment to confirm dataset paths and submission.csv generation
- [ ] Phase 5: final QA pass: spelling / figure rendering / table
      alignment / cell ordering / kernel restart-and-run-all

## Per-Chapter Workflow (the human-in-the-loop contract)

For each notebook chapter the agent will:

1. **Propose a decision memo** containing: chapter goal, recommended
   structure, candidate figures, candidate tables, recommended text
   tone, demonstrative code (if any). Provide 2-3 options with
   pros / cons / recommendation when there is a meaningful tradeoff.
2. **Wait for user approval** before writing any cells. While waiting,
   notify the user via Discord (the user is not at the desk by default).
3. **Execute the approved plan**: draft the cells, render figures if
   needed, embed tables, write narrative text.
4. **Self-review with image-reading capability**: open the rendered
   notebook, check figure embedding / markdown formatting / chart
   readability. Iterate until satisfied.
5. **Hand off for user verification**: notify via Discord, ask user to
   acknowledge before moving to the next chapter.

## Decisions Made So Far

- The notebook is one canonical artifact; the team will not try to
  fold all git branches into it.
- pi-planning skill is used for journaling under
  `kaggle_submission/plan/`.
- A navigation `STORY_MAP.md` is maintained at the root of
  `kaggle_submission/` and points to every important folder relevant to
  the main story chain.

## Decisions Resolved (2026-05-17 with user)

- Outline: **Option 3 (hybrid)** — task chapters keep PDF section
  numbering for TA grading, each ending with a mini-reflection. A
  dedicated Discussion chapter at the end consolidates cross-task
  insights and lecture links.
- Demo code level: **light**. Each task chapter contains 1 short demo
  cell with English comments that *actually invokes* the real code
  bundled in the Kaggle dataset (i.e. the code in the dataset must be
  importable and runnable from the notebook). Real heavy logic stays in
  the dataset modules; the notebook surfaces only the call site.
- Cityscapes external generalization: Section 5.1 inside the Discussion
  chapter.
- Per-chapter cadence: each chapter follows decision-memo -> approval
  -> execution -> self-review -> verification, with Discord pings at
  every stop.

## Figure Quality Contract (applies to every figure)

Every figure embedded in the notebook must satisfy:

- **Uniform panel geometry**. When multiple sub-images are gridded,
  resize them all to the same shape first (warp or letterbox). Never
  put raw mixed-aspect-ratio images into the same `subplots` grid:
  taller panels overflow into the captions of the neighbours.
- **Readable typography**. Subplot titles >= 11pt, super-title >= 13pt,
  axis labels >= 10pt. Use `fontsize=` explicitly; do not rely on
  defaults.
- **Notebook-context resolution**. The default Kaggle/Jupyter render
  width is ~960px. Render with `dpi >= 110` and choose `figsize` so
  the natural pixel width is between `1100` and `1600`. Below that the
  text pixelates after browser scaling.
- **No caption / image collision**. Reserve at least one full line of
  blank space between a subplot's image and the next row's title.
  Use `fig.tight_layout(h_pad=1.5)` or manual spacing.
- **Render-in-context check**. After every figure change, regenerate
  the notebook HTML via nbconvert and screenshot the cell with Chrome
  at 1100px window width. Read that screenshot, not the standalone PNG.
  The reviewer must be able to read every label without zooming.
- **Source of truth**. Prefer rendering inline from data in cells over
  pre-rendering to PNG. Pre-render only when the runtime cost would
  exceed a few seconds on Kaggle.

If a figure cannot pass the in-context check, fix the helper, do not
ship the figure.

## Pretrained-Weight Provenance Rule

Every chapter that loads a pretrained backbone must name the exact
weight identifier (e.g. `ConvNeXt_Small_Weights.IMAGENET1K_V1`,
`DINOv3-ViT-L/16 pretrained on LVD-1689M`, `MiT-B5 ImageNet-1K`) in
the same chapter as the model is introduced. It must also state in one
sentence that the pretraining corpus does not contain PASCAL VOC
images. A reviewer should be able to grep the chapter for the
identifier and immediately verify the assignment's no-VOC clause.

## Numerical-Claim Sourcing Rule

Every Dice, mAP, mIoU, success-rate, or pixel-accuracy number that
appears in narrative markdown must satisfy one of: (a) be printed or
plotted by a code cell in the same chapter, or (b) be a row in a CSV
under `dataset/tables/` that is loaded and rendered in the same
chapter. Ranges in narrative text ("about +0.5-1") are not allowed; if
the supporting experiment was not run, the claim is removed rather
than softened with an adverb.

## Train / Val / Test Split Disclosure Rule

Any chapter that reports a validation metric must, before the first
such number, state the split sizes, the random seed, the stratification
scheme (or "none"), and whether the metric is computed on the same
split throughout the chapter. If a stage retrains on the union of
train + val, the chapter must explicitly say which downstream numbers
are still valid and which are not.

## Metric-Definition Disambiguation Rule

Every leaderboard score in the notebook must be tagged with which
slice of the 1500-row submission CSV produced it (classification
half, segmentation half, or both averaged). The Summary of Results
table must include a `scope` column or footnote that makes this
unambiguous, so a reader never has to guess whether two numbers are
comparable.

## Loss-Ablation Pairing Rule

When a chapter introduces a non-default loss (AsymmetricLoss, Dice,
CosPGD, region-rebalance, etc.), the chapter must include at least
one ablation table comparing it against the obvious default (BCE, CE,
PGD) on the same data and the same backbone, reported across the full
class set rather than on a single cherry-picked class. Without this
table the loss choice is treated as undefended.

## Interpretation Discipline (applies to every visual)

Every code cell that produces a visual (chart, table, image grid,
numerical summary) is followed by a short markdown cell that does the
interpretation. The cell does not restate the goal; it tells the
reader what the visual actually shows. Concretely:

- 2-4 sentences, factual, no marketing copy.
- Cite specific numbers or features that appear in the visual; do not
  paraphrase in the abstract ("the table shows trends" is not allowed).
- Link the observation to a downstream design choice or to a finding
  reported later in the report, when there is a connection.
- If the visual contradicts the prior text, flag the contradiction
  honestly. Do not bury it.

A visual without its interpretation cell is not considered done. Apply
this retroactively when reviewing earlier chapters.

## Writing Style Contract (applies to every chapter)

Tone: natural, direct, sincere, authoritative without bragging. Quiet
confidence. No over-enthusiasm. No self-justification. No needless
comparisons. Use plain, precise academic vocabulary. **Banned** unless
the context strictly requires them: `leverage`, `delve into`,
`tapestry`. Default substitutes: `use`, `investigate`, `context`.
Technical jargon only when it actually carries information; no
ornamental complexity.

**Academic register**: this notebook is read by a professor for
grading. No internet slang (`TL;DR`), no marketing copy ("real-world
reality check"), no casual openers ("So we tried..."). Use the words a
peer-reviewed paper would use: `Summary of Results`, `Cross-domain
evaluation`, `Primary metric`. When in doubt, ask which term would
appear in a CVPR / ICCV abstract.

Syntax constraints:

- Mix short and long sentences. Use short ones for the core claim or
  the strong statement; use clear long ones (subordinate / participial
  clauses) when explaining causal chains. Subject and verb should be
  visible on first read.
- Punctuation: minimise the em-dash (`—`). Prefer commas, parentheses,
  or splitting into two sentences.
- If a draft already reads natural and free of AI-smell, leave it
  alone. Better to do nothing than to add filler.

This contract holds for every notebook cell, every decision memo, and
every figure caption.

## Chapter List (frozen unless user revises)

0. Title + Abstract + TL;DR
1. Overview: data, challenges, submission framework
2. Task 2.1 Image Classification
3. Task 2.2 Semantic Segmentation
4. Task 2.3 Adversarial Attacks
5. Task 2.4 Discussion (with Cityscapes as 5.1)
6. Final submission generation
