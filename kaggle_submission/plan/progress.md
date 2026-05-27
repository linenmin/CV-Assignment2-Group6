# Progress: Kaggle Submission Integration

## Session Log

### 2026-05-17 (Bootstrap)

- Read CV_GA2.pdf Sections 2.1-2.4 and Section 3 (submission rules) to
  set the requirements baseline.
- Read reference starter notebook `ga2_group_6.ipynb` to confirm the
  4-chapter scaffold expected by TAs.
- Pulled the Image-Classification branch into the local repo so the
  classification team's README + Discussion.md + attack script are all
  visible without checking the branch out.
- Created `kaggle_submission/plan/` and initialized `task_plan.md`,
  `findings.md`, `progress.md` using the pi-planning skill.
- Confirmed with user: it is acceptable that the canonical submission
  artifact is the notebook + Kaggle dataset, not a merged git branch.
  TAs do not see git, so no need to integrate branches.
- Presented the 3 outline options + interaction protocol + folder
  layout.
- User confirmed (2026-05-17):
  - outline = Option 3 (hybrid)
  - demo code = light (with English comments, must invoke real bundled code)
  - other proposed defaults accepted
- Created folder skeleton under `kaggle_submission/`:
  `plan/decisions/`, `notebook/`, `dataset/{weights,predictions,submissions,figures,tables,code}/`.
- Wrote `kaggle_submission/STORY_MAP.md` as the team-side navigation map.
- Drafted `plan/decisions/DM-0.md` for Chapter 0 (Title + Abstract +
  TL;DR + reading guide). User approved all recommended defaults
  (members listed without roles, B1 5-row table, C1 navigation guide).
- Added the writing-style contract to `task_plan.md`: natural,
  direct, banned vocabulary list, short / long sentence mix,
  minimise em-dash. Applies to every future cell.
- Wrote Chapter 0 into `notebook/group6_final.ipynb` (4 markdown
  cells). Rendered to HTML via nbconvert, then to PNG via headless
  Chrome at `C:\Users\Lem17\AppData\Local\Temp\nb_preview\group6_final.png`.
- Self-review: layout clean, table renders correctly, no banned
  vocabulary. Two style nits flagged to user for confirmation: an
  em-dash in the title and backticks around numerical values.
- Status: Chapter 0 draft v1 ready for user verification.
- User feedback: rejected `TL;DR` as internet slang, not formal enough
  for an academic submission. Em-dash in title kept (option a). Number
  backtick style kept (option a).
- Edited Chapter 0 cell 2: heading `TL;DR` -> `Summary of Results`,
  metric column header `Headline metric` -> `Primary metric`, row
  `Real-world reality check` -> `Cross-domain evaluation`.
- Re-rendered: `C:\Users\Lem17\AppData\Local\Temp\nb_preview\group6_final_v2.png`.
  Self-review: format clean, terminology now consistently academic.
- Chapter 0 status: draft v2 ready for verification.
- User approved Chapter 0 v2 and DM-1 (all recommended defaults).
- Chapter 1 execution:
  - wrote `dataset/code/submission.py` (rle_encode + rle_decode) and
    `dataset/code/data_overview.py` (per-class counts + chart +
    pre-render helper + display helper)
  - picked diverse sample IDs `[0, 5, 7, 8]` covering cow / motorbike+
    person / person+train / chair+tvmonitor; pre-rendered
    `dataset/figures/sample_images_with_masks.png`
  - rewrote `notebook/group6_final.ipynb` adding 7 Chapter 1 cells
    (4 markdown + 3 light code); cell IDs added so future edits are
    safe
  - cell 1.2 path resolution walks up parents to find `kaggle_submission/`,
    so the notebook runs both on Kaggle (under `/kaggle/input/...`) and
    locally
- nbconvert `--execute` failed because the local ipykernel hits an
  old pkg_resources incompatibility. Worked around with
  `plan/scripts/execute_and_inject.py`: walks the notebook, executes
  each code cell in a shared exec context, captures stdout and
  matplotlib figures as base64 PNG, writes outputs back in nbformat v4.
  Re-runnable / idempotent.
- Self-review (Chrome headless screenshot split into top / bottom):
  class-distribution chart shows expected long tail, sample grid
  displays both image and mask rows, RLE round-trip prints expected
  string and `True`. Bullet observations and submission-format
  paragraph render with proper inline code spans.
- Chapter 1 status: draft v1 ready for verification.
- User feedback on Chapter 1 sample grid: panel labels of the form
  `train_5` were ambiguous because `train_` is the training-split file
  prefix, not the VOC class `train`. Updated
  `prerender_sample_images_with_masks` to caption each panel as
  `Image #N\nclasses: ...` and re-rendered
  `dataset/figures/sample_images_with_masks.png`. Re-injected the
  notebook outputs.
- Started Chapter 2 planning. Drafted `plan/decisions/DM-2.md`.
  Awaiting user A/B/C/D selection and the Drive URL for the
  classification weights.
- User feedback on Chapter 1 sample grid: aspect-ratio mismatch caused
  panel #7 to overflow into the row-2 captions, font was too small,
  raster was pixelated. Added Figure Quality Contract to task_plan.md
  and rewrote `render_sample_grid` to (a) resize every panel to
  uniform `256x256` square, (b) raise titles to 12 pt, (c) render
  inline (no pre-render PNG) at `figsize=(11, 6)`, `dpi=130`. Re-injected
  the notebook and verified the cell-sized screenshot in
  Chrome at 1100 px window width before declaring it acceptable.
- Classification weight bundle from Zhenyang reorganised:
  - `weights/convnext_small_320/final_model.pth` (191 MB)
  - `weights/convnext_small_320/best_thresholds.npy`
  - `tables/classification_per_class_ap.csv`
  - `tables/classification_training_history.csv`
  - `tables/classification_evaluation_summary.csv`
  - `tables/classification_adversarial_aeroplane.csv`
  - `tables/classification_test_probabilities.csv`
  - `figures/classification_adversarial_aeroplane_examples.png`
  - `figures/classification_training_history.png`
  - `figures/classification_eval_ap_per_class.png`
  - `submissions/submission_classification_convnext_small_320.csv`
  - original Drive bundle and `.zip` removed after copy.
- Wrote `dataset/code/classification_metrics.py` (load_experiments_table,
  style_experiments_table, load_per_class_ap, plot_per_class_ap) and
  `dataset/code/classification_model.py` (load_convnext_small_weights
  reports param_count + state_dict summary without rebuilding the
  network). Authored `dataset/tables/classification_experiments.csv`
  with 8 backbone rows.
- Upgraded `plan/scripts/execute_and_inject.py` to (a) capture the
  last expression's value via ast rewrite and (b) intercept
  `display(...)` calls, so styled DataFrames are persisted in the
  notebook as `text/html` outputs.
- Added 9 Chapter 2 cells (6 markdown + 3 light code). All cells
  execute cleanly, outputs verified in Chrome screenshot:
  - experiments table renders with ConvNeXt-Small row highlighted
  - per-class AP bar chart shows red bars for AP < 0.80
    (diningtable, pottedplant, sofa, bottle, sheep)
  - weight summary prints 49,857,140 parameters / 344 tensors / 190 layers
- Chapter 2 status: draft v1 ready for verification.
- User flagged that visualizations lacked immediate interpretation.
  Added an "Interpretation Discipline" contract to `task_plan.md`:
  every visual-producing code cell is followed by a 2-4 sentence
  markdown cell that cites specific numbers and links to a design
  choice or downstream finding. Applied retroactively.
- Inserted 5 interpretation cells:
  - after `ch1-class-dist`: long-tail interp citing person/chair/car
    head vs sheep/cow/bus tail
  - after `ch1-sample-grid`: variability interp covering aspect ratio
    and foreground-coverage range
  - after `8cbc7f3c` (experiments table): backbone interp
    contrasting val mAP monotonicity with non-monotonic Kaggle Dice;
    overfitting reading for ConvNeXt-Base; inductive-bias note for ViT
  - after `770e34ef` (per-class chart): five-classes-below-0.80 fact
    + saturation observation for big-silhouette classes
  - after `4f816602` (weight summary): parameter count + head keys
    confirmation tying bundle to the production submission
- Re-injected outputs and screenshot-verified that every visual is
  followed by its interpretation cell.
- Chapter 2 status: draft v2 ready for verification.
- Ran a professor-style review via a fresh sub-agent. Confirmed the
  per-class AP numbers cited in the chapter all match the underlying
  CSV. The critique flagged 8 issues; 5 of them generalised into hard
  rules. Added to `task_plan.md`:
  - Pretrained-Weight Provenance Rule
  - Numerical-Claim Sourcing Rule
  - Train / Val / Test Split Disclosure Rule
  - Metric-Definition Disambiguation Rule
  - Loss-Ablation Pairing Rule
- Applied retroactively to Chapter 2:
  - 2.1.1 now names `ConvNeXt_Small_Weights.IMAGENET1K_V1` and states
    explicitly that ImageNet-1K contains no PASCAL VOC images
  - 2.1.3 now states the 80/20 split, seed 42, no stratification, and
    documents the threshold-sweep / Stage-3 retrain interaction
    honestly (thresholds frozen before Stage 3, not re-swept)
  - Summary table gains a `scope` column so `0.898` and `0.89139`
    cannot be confused; segmentation half row removed because we
    never submitted seg-only
  - dropped the "+0.5-1 Dice" range claim for TTA; replaced with
    "small but consistent improvement on the validation split"
  - `27 training images` removed from 2.1.4 to avoid restating the
    long-tail number; instead points back to Section 1
  - `vit_b_16_224` row in the experiments CSV gets a `notes` field
    explaining "not submitted: pruned at val stage"
  - `style_experiments_table` now passes `na_rep="—"` so a missing
    `kaggle_dice` renders cleanly
- For Issue 7 (Loss-Ablation Pairing Rule) the user opted to actually
  run the BCE baseline rather than soften the claim. Extracted
  `shared.py`, `02_dataset.py`, `03_model.py` from the
  `origin/Image-Classification` branch into
  `plan/scripts/bce_baseline/`. Wrote `train_bce_vs_asl.py` to run
  both losses through Stages 1 + 2 (no Stage 3 leak) and emit
  per-class AP CSVs.
- Full BCE + ASL run completed (about `22` minutes on RTX 4060). Both
  per-class AP CSVs copied to
  `dataset/tables/classification_loss_ablation_{bce,asymmetric}.csv`.
- Aggregate val mAP: BCE `0.9120` vs AsymmetricLoss `0.9193`, delta
  `+0.0073`. The per-class story is much richer than the aggregate:
  - ASL > BCE on `diningtable` (`+0.102`), `sheep` (`+0.083`),
    `sofa` (`+0.063`), `bicycle` (`+0.033`), `cat` (`+0.021`); these
    are the annotation-noisy / long-tail classes the design intent
    targeted.
  - ASL < BCE on `pottedplant` (`-0.065`), `chair` (`-0.056`),
    `dog` (`-0.029`), `bottle` (`-0.025`); honest negative finding,
    documented in the interpretation cell.
- Added `load_loss_ablation` and `plot_loss_ablation` to
  `dataset/code/classification_metrics.py`. Inserted a code cell
  (`a1063a12`) and an interpretation cell (`6960e34b`) into
  Chapter 2 between 2.1.2 (Loss design) and 2.1.3 (Training recipe).
  Fixed an early title typo ("50 epochs" -> "25 epochs: 5 Stage 1 +
  20 Stage 2") on the chart.
- Chapter 2 status: draft v3 ready for verification with full
  Loss-Ablation Pairing Rule compliance.
- Chapter 3 (Semantic Segmentation) draft v1: 18 cells (11 markdown,
  7 light code) covering V17-centric structure as the user
  requested. Each pre-V17 reference names the concrete pivot
  (architecture / pretraining / loss / data) responsible for the
  delta.
- Added 4 segmentation tables under `dataset/tables/`:
  `segmentation_experiments.csv` (V1-V17 with Pivot column,
  val mIoU, Kaggle seg-half Dice, Kaggle joint Dice),
  `segmentation_loss_ablation.csv`,
  `segmentation_per_class_iou.csv` (V17 from
  `tta_v17_ms496_512_528_noflip_val37` summary),
  `segmentation_v17_training_log.csv` (copied from training run).
- Copied the V17 best checkpoint (1.2 GB) into
  `dataset/weights/eomt_dinov3_v17_merge_val_lr1e5/best/`.
- Added two helper modules: `segmentation_metrics.py` (7 functions
  covering experiments table, pivot waterfall, lab-vs-Kaggle
  scatter, loss ablation bars, training curve, per-class IoU,
  cross-task per-class) and `segmentation_model.py` (header-only
  safetensors summary so the checkpoint sanity check does not
  require the matching `transformers` version).
- Implemented all three insight-driven extras the user approved:
  Pivot-attribution waterfall (#1), Lab vs Kaggle scatter (#2), and
  cross-task per-class comparison (D2). Each visual is followed by
  an interpretation cell per the Interpretation Discipline.
- Chapter 3 status: draft v1 ready for verification.
- Chapter 4 (Adversarial Attacks) draft v1: 14 cells (9 markdown +
  5 light code). Structure: 4.0 setup, 4.1 classifier attack
  (table + 4-row visual + interpretation), 4.2 segmenter attack
  (results table + V26 targeted grid + interpretation), 4.3
  trainable-vs-PGD side-by-side (the central insight, with the two
  pre-rendered comparison grids), 4.4 reflection covering all four
  PDF questions (realism, recognisability, reliability,
  defenses).
- Copied 4 visual assets into `dataset/figures/`:
  `segmentation_trainable_attack_grid.png`,
  `segmentation_pgd_attack_grid.png`,
  `segmentation_v26_untargeted_grid.png`,
  `segmentation_v26_targeted_grid.png`. The classifier visual was
  already in `dataset/figures/`.
- Created `dataset/tables/classification_attack_results.csv` and
  `dataset/tables/segmentation_attack_results.csv`.
- Added `dataset/code/attack_metrics.py` with helpers for the two
  styled tables and the side-by-side image display.
- Applied requested insights: #3 attack-budget visualisation
  (rendered as text in the reflection paragraph, 1.7/255 and 6/255
  framing) and #4 cross-task vulnerability symmetry observation
  (also in 4.4 reflection).
- Bug fix: chapter 4 subsections were initially numbered 4.3.1
  through 4.3.4 (mixing chapter index and PDF section index).
  Renumbered to 4.1 / 4.2 / 4.3 / 4.4 for consistency with
  Chapter 3 numbering style.
- Earlier render attempt hit a 2000 px image-dimension limit when
  the full notebook PNG reached 24000 px tall; verification now
  uses 1800 px slices read individually.
- Chapter 4 status: draft v1 ready for verification.
- Chapter 5 (Discussion) draft v1: 8 cells (7 markdown + 1 light
  code). Structure: 5.0 roadmap, 5.1 Cityscapes cross-domain
  evaluation with one bar chart and a two-paragraph interpretation,
  5.2 cross-task observations (shared weak classes, shared
  adversarial fragility, dog asymmetry), 5.3 five named lecture
  connections, 5.4 four-bullet limitations with one-week follow-up
  notes, 5.5 three-dimension deployment-readiness verdict.
- User opted to drop the train-vs-tram visual-concept evidence
  (Insight #6) on the basis that it stems from a self-selected
  Cityscapes evaluation rather than a project-internal model
  failure; the chapter still mentions exclusion of `train` as a
  scoping decision in 5.1 but does not show the paired panel.
- Insight #5 (5-fold CV would have caught the V4 / V5 val-vs-Kaggle
  inversion earlier) is incorporated into the 5.4 validation-split
  bullet.
- Added `dataset/tables/cityscapes_transferable_iou.csv` (copied
  from segmentation-branch metrics output) and extended
  `dataset/code/segmentation_metrics.py` with
  `load_cityscapes_transferable` and `plot_cityscapes_transferable`.
- Self-review via 1800 px slices passed; chart and text render
  cleanly with the `train` bar greyed out and labelled
  "(excluded)".
- Chapter 5 status: draft v1 ready for verification.
- DM-6 (Chapter 6: Final Submission Generation) approved by user
  with all recommended defaults (A1 no extra demo, B1 five-assertion
  sanity check + decoded labels, C1 short reproduction paragraph).
- Copied
  `Semantic segmentation/outputs/submissions/submission_exp_v17_merge_val_tta_ms496_512_528_noflip_with_convnext_small_320.csv`
  into the bundled dataset as
  `dataset/submissions/submission_v17_final_merged.csv`
  (1500 rows; the file that scores `0.89139` on the joint
  leaderboard).
- Chapter 6 draft v1: 5 cells (3 markdown + 2 light code).
  - `ch6-header` (md): explains the assembly step, names the two
    halves (750 cls rows from ConvNeXt-Small, 750 seg rows from
    V17 EoMT-DINOv3 multi-scale TTA), and states upfront that
    end-to-end inference is not run here (~20 GPU-minutes on
    Kaggle would blow the runtime budget).
  - `ch6-write` (code): loads the merged CSV with
    `keep_default_na=False`, picks
    `/kaggle/working/submission.csv` on Kaggle else local
    `outputs/submission.csv`, prints source/output paths plus row
    count and columns.
  - `ch6-sanity-md` (md): explains the five invariants and the
    decoded-classification print.
  - `ch6-sanity-code` (code): asserts 1500 total rows / 750
    classification rows / 750 segmentation rows / no NaN / unique
    Id; prints one classification row, one segmentation row, and
    decodes image 0 back to VOC class names.
  - `ch6-closing` (md): leaderboard score `0.89139`, three-step
    offline reproduction recipe pointing at
    `dataset/weights/eomt_dinov3_v17_merge_val_lr1e5/best/` plus
    `dataset/code/`, and the ~20-minute wall-clock justification.
- Two execution bugs found and fixed:
  - Heredoc injection ate the literal `\n` inside the Python
    source, causing an `unterminated string literal` SyntaxError
    on first run. Rewrote the cell via a `PYEOF` heredoc that
    stored the source as a triple-quoted Python string and then
    called `splitlines(keepends=True)`.
  - First sanity-check run failed with
    `AssertionError: Some Predicted cells are NaN`. Root cause:
    pandas `read_csv` coerces empty strings to `NaN` by default,
    but the Kaggle convention treats empty RLE strings as
    "no foreground predicted", which is valid. Switched to
    `pd.read_csv(merged_path, keep_default_na=False)` and
    changed the assertion to
    `assert not submission_df["Predicted"].isna().any()` so
    empty strings pass and only true NaN fails.
- Re-ran `execute_and_inject.py`: ch6-write printed
  `Total rows : 1500`, columns `['Id', 'Predicted']`;
  ch6-sanity-code printed
  `All five sanity-check assertions passed.` plus
  `Sample classification row: Id = 0_classification`,
  `Predicted (RLE) = 9 1 15 1 20 1`, and
  `Decoded classification for image 0: chair, person, tvmonitor`.
- Local fallback wrote
  `kaggle_submission/outputs/submission.csv` (1500 rows + header
  = 1501 lines, confirmed via `wc -l`).
- STORY_MAP.md updated: Chapter 6 status flipped from
  `not started` to `draft v1, awaiting verification`.
- Chapter 6 status: draft v1 ready for verification. All six
  chapters now have a draft in the notebook.

### 2026-05-18 (Full review pass)

- Ran a three-agent audit of the notebook against (a) the `CV_GA2.pdf`
  hard requirements (Sections 2.1-2.4) and (b) the eleven internal
  contracts in `task_plan.md`. Findings consolidated into a priority
  P0/P1/P2 punch list and shown to the user for batch approval.
- User approved all recommended edits including the harder option of
  measuring a from-scratch ConvNeXt-Tiny baseline rather than
  asserting it.
- Trained `ConvNeXt-Tiny` from random init (`pretrained=False`) for
  `25` epochs on the same `599 / 150` split, with AdamW + cosine
  schedule + the production AsymmetricLoss recipe. Script:
  `plan/scripts/from_scratch_baseline/train_from_scratch.py`. Final
  numbers: `best val mAP 0.2026` at epoch `24`, vs ImageNet-init
  ConvNeXt-Small at `0.9000`. Gap `+0.6974` mAP in favour of
  fine-tuning. Per-class AP from-scratch is below `0.40` on every
  class; uniform, not driven by one or two outliers. CSV outputs
  copied into `dataset/tables/` so the notebook can load them.
- Built `plan/scripts/apply_review_edits.py` to apply 30+
  in-place text replacements plus 10 new cell insertions in one pass.
  Made the script idempotent (skip an edit if the new text is already
  present) so re-runs are safe. Found one corner case where the OLD
  pattern was a substring of the NEW string (e.g. `plt.show()` inside
  a longer block), which caused three code cells to double up; fixed
  by overwriting those three cell sources directly.
- New cells inserted (10 total):
  - `ch2-convnext-components`: ConvNeXt-Small layer inventory table
    (TA Q3 \"What is the use of each layer?\").
  - `ch2-fromscratch-intro / -code / -interp`: from-scratch baseline
    section (TA Section 2.1 \"experiment with two types of models\").
  - `ch2-traincurve-intro / -code / -interp`: Chapter 2 training
    curves (TA Q3 \"Is your model learning as it should?\"). Loaded
    from the existing `classification_training_history.csv` that was
    in the dataset but unused.
  - `ch3-eomt-components`: EoMT-DINOv3 components table + Mermaid
    block diagram of the encoder/queries/heads data flow.
  - `ch4-attack-modes-diagram`: Mermaid diagram contrasting trainable
    generator vs per-image PGD pipelines.
  - `ch-references`: full References section with all nine inline
    citations expanded to author/year/venue/arXiv format (PASCAL VOC,
    Cityscapes, ConvNeXt, AsymmetricLoss, DINOv3, EoMT, FGSM, PGD,
    CosPGD).
- In-place text edits applied (30+): em-dash in title to colon;
  hedging adverbs removed in front of precise numbers; \"robust\"
  marketing-adjective usages replaced; Chapter 5 emoji markers
  replaced with words (Ready / Partial / Not ready); Chapter 5 \"half
  the in-distribution score\" corrected to \"about two-thirds\"
  (`0.509 / 0.793 = 0.64`); Chapter 3 bicycle \"rare class\"
  parenthetical removed (bicycle has ~65 images, not rare); Chapter 4
  \"V21\" vs \"`14+` variants\" reconciled to \"21 variants spanning
  14 architecture families\"; Chapter 4 CosPGD-vs-PGD framing
  rewritten to make clear it is not a clean A/B; ~20 long sentences
  (over 35 words) split; Chapter 5 opener (56 words) restructured
  into a 5-bullet roadmap with a 4-to-5 bridge sentence; Chapter 3.1
  to 3.2 transition added; Chapter 5.3 lecture links upgraded with
  specific slide numbers (L8 slides 26-27, L9 slide 82, L10 slides
  10/14-15/18-20/29-30/37-39/55, L9 slides 24-25, L9 slides 75-77);
  Table 1 and Table 2 captions added; FGSM/CosPGD/Madry citations
  added inline.
- Code-cell edits: cell `ch1-class-dist` now uses `dpi=130`
  (was default 100, below B6 contract); cell `a1063a12` prints top-3
  winners and top-4 losers under AsymmetricLoss for B1 strict
  sourcing; cell `cfd09754` prints 5-class transferable mIoU
  numerically (`0.509`) and 6-class full mIoU (`0.425`); cells
  `8cbc7f3c` and `f2ebd03b` use `.set_caption()` on the styled
  DataFrame.
- Bug fix: cell `cfd09754` was first written using `cityscapes['class']`,
  but the CSV column is `class_name`. The print showed `0.425` instead
  of `0.509` until the column name was corrected and the cell was
  re-injected.
- Re-rendered the full notebook to HTML via nbconvert and to PNG via
  headless Chrome at `1100 x 40000` viewport. Sliced into
  19 x 1800 px panels and reviewed each visually. All A1-A12 and
  B1-B8 items now pass; the Mermaid diagrams render natively in
  nbconvert's HTML output; all table captions render correctly above
  their styled tables.
- Notebook now `83` cells (was `73`). All 6 chapters plus References.
  PDF rubric coverage: from-scratch baseline measured (A2), layer
  inventories present (A3), Chapter 2 training curves present (A5),
  visual triplet, trainable adversary, and defences all present
  (A7-A9), Cityscapes cross-domain measured (A12), explicit
  lecture-slide citations (A10), and the References section makes
  every primary source accessible.
- Status: review v2 complete; notebook ready for Kaggle packaging.

### 2026-05-18 (Diagram v2 + legend audit)

- User reported two issues after the v2 review: (a) several charts
  had legends overlapping the data; (b) the two Chapter 3 / Chapter 4
  diagrams rendered as low-resolution raster PNG and the layout looked
  cramped.
- Legend audit (visual sweep across every figure in the v2 render):
  - Pivot waterfall (`plot_pivot_waterfall`): the 2-column 5-entry
    legend at `upper left` covered the V1 baseline bar entirely.
  - Cross-task per-class (`plot_cross_task_per_class`): the
    `lower right` legend covered the bicycle (`AP = 1.0`) and
    pottedplant orange bars.
  - From-scratch baseline right panel (`ch2-fromscratch-code`): the
    `lower right` legend covered the aeroplane and diningtable
    blue bars.
  - Lab vs Kaggle scatter: V11, V12, V14, V15, V17 text annotations
    collided in the upper-right cluster.
- Fixes applied via `plan/scripts/fix_diagrams_and_legends.py`:
  - All three bar charts: legend moved out of axes to
    `loc='lower center', bbox_to_anchor=(0.5, -0.10..-0.18)`,
    horizontal layout, with `subplots_adjust` to reserve room.
  - Lab vs Kaggle: replaced the static `(6, 4)` annotation offset
    with a hand-tuned per-version override map for V11/V12/V14/V15/V17
    so each label sits in its own quadrant.
- Diagram rewrite: discarded the mermaid + matplotlib-PNG approach and
  rewrote both Chapter 3 and Chapter 4 diagrams as hand-authored SVG.
  Why SVG inline in markdown rather than a referenced image:
  - Jupyter and Kaggle both render `<svg>` tags inside markdown cells.
  - The result is vector and stays sharp at any zoom or DPI.
  - The notebook becomes self-contained: no `dataset/figures/*.png`
    runtime dependency and no mermaid extension required.
- New SVG files at `dataset/figures/ch3_eomt_dataflow.svg` and
  `ch4_attack_modes.svg`. Both designed to:
  - Use a three-colour scientific palette (`#3a6da9` frozen blue,
    `#d97c2e` learnable orange, `#4f9d4f` output green).
  - Set body font to 17 px (titles) and 14 px (subtitles).
  - Use a coloured legend strip near the top.
  - Route every forward arrow horizontally; reserve dashed coloured
    paths for the gradient / update flows.
  - Show Chapter 3 as a single horizontal pipeline with mask queries
    flowing up into the fine-tuned ViT blocks; show Chapter 4 as two
    side-by-side subgraphs (Trainable generator | Per-image PGD) so
    the comparison is visually parallel.
- Embedding script `plan/scripts/embed_svg_diagrams.py`:
  - Loads each SVG, strips per-line indentation (markdown otherwise
    treats 4+ leading spaces as a code block, which dumped the SVG
    source as text rather than rendering it).
  - Removes any prior SVG block and any prior `*Figure N.* ...`
    caption paragraph from the target markdown cell.
  - Appends the new SVG and the new caption.
  - Removes the now-redundant `ch3-eomt-image` and
    `ch4-attack-modes-image` matplotlib-imshow code cells.
- Notebook count back to `83` cells (was `83` before the imshow
  cells were removed; now contains two markdown cells with embedded
  SVG instead of `markdown + imshow code` pairs).
- Re-rendered HTML and screenshotted at `1100 x 40000`. Reviewed
  all `20` slices and the zoom crops of both SVG areas. All four
  legend occlusions are gone, both SVG diagrams render crisply
  with large, readable typography, captions render correctly below
  each diagram.

### 2026-05-18 (Third-pass diagram + table polish)

- User feedback after the v7 review surfaced five more issues. All
  are now fixed.
- **Ch4 SVG asymmetry**: in subgraph B the per-image PGD return
  arrow was a 1-segment vertical stub. Subgraphs A and B were not
  visually parallel. Re-laid both subgraphs on a symmetric grid:
  top-row forward path (`x → param → δ → x+δ`), bottom-row return
  path (`L_adv ← V17`), with a coloured dashed inner-loop arrow
  that sweeps LEFT then UP in both halves. Orange dashed for
  backprop into θ; green dashed for PGD δ update.
- **New Chapter 2 diagram**: added
  `dataset/figures/ch2_convnext_architecture.svg`, a horizontal
  ConvNeXt-Small pipeline (Input → Stem → Stage 1..4 → Pool → Head)
  with a secondary row that zooms into one ConvNeXt block (depthwise
  7×7 → LayerNorm → 1×1 expand 4× → GELU → 1×1 project → +residual).
  Same colour palette as the Chapter 3 and Chapter 4 diagrams: blue
  for ImageNet-1K-pretrained modules, orange for the custom
  multi-label head.
- **Training-curve cell**: the four lines (Stage 1 train / val,
  Stage 2 train / val) previously shared a colour per stage with
  only linestyle distinguishing train vs val. Replaced with four
  distinct colours; epochs now concatenated on a single x-axis
  (S1 epochs 1-5 followed by S2 epochs 6-25) with a vertical dashed
  marker labelled `Stage 2 starts (backbone unfrozen)`. Also
  filtered Stage 3 out of the plot because S3 retrains on
  train + val so its `val_loss` column is undefined.
- **Table 2 cleanup**: dropped the `date`, `val_split`, and
  `kaggle_joint_dice` columns from the segmentation experiments
  table. The reader keeps `version`, `model_family`, `pivot`,
  `val_miou`, `kaggle_seg_dice` — every column carries information
  the narrative directly uses.
- **Pivot waterfall**: V1 was plotted with its absolute Dice
  (`0.726`) as the "delta" against an implicit zero baseline; that
  bar dwarfed every later delta in the `-0.025 to +0.075` range and
  hid the per-pivot story. Filtered V1 out before plotting, passed
  V1's value as an explicit `baseline_value` argument so the title
  still names the reference (`baseline V1 = 0.726`). The y-axis
  now spans the actual per-pivot delta range, and the SegMAN-B
  (V10) and EoMT-DINOv3 (V11) family swaps are visually the tallest
  bars, matching the chapter's narrative.
- **Verbosity pass**: removed `7` sentences that restated points
  already established in the surrounding text. Targets:
  - `ch1-observations` Variable image sizes bullet trimmed.
  - `ch2-fromscratch-interp` lost two trailing sentences that
    duplicated the backbone-scaling and lecture-link discussions.
  - `2fd0c537` (2.1.4) lost the closing paragraph that restated the
    per-class chart for the easy classes.
  - `accfdc57` (3.1 V17 description) lost a second no-VOC clause.
  - `b370583c` (3.2) opening sentence tightened.
  - `b208de22` (4.4) joined two sentences saying the same thing.
  - `5053a761` (5.5) bottom-line paragraph collapsed from three
    rhetorical questions to two sentences.
- Re-rendered v8 HTML and screenshotted. Reviewed all `20` slices
  and zoom crops of the Ch2 ConvNeXt SVG, the four-colour training
  curve, Table 2, the V2-V17 pivot waterfall, and the redesigned
  Ch4 SVG. All fixes verified visually.

## Verification Notes

- Pi-planning skill loaded successfully in this session.
- Three planning files created at
  `D:\BaiduNetdiskWorkspace\Leuven\8th\Computer Vision\assignment\Group2\kaggle_submission\plan\`.
