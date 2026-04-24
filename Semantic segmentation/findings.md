# Findings: Semantic Segmentation V1

## Dataset Findings

### Source: local dataset inspection

- Path: `kul-computer-vision-ga-2-2026/`
- Training examples: `749`
- Image and mask shapes are variable, with major modes around `375x500`, `333x500`, and `500x375`
- Example image arrays are `uint8` RGB with values in `0..255`
- Example mask arrays are `uint8` with class ids in `0..20`
- Pixel-level imbalance is severe:
  - background pixels: about `77.74%`
  - foreground pixels: about `22.26%`
- Image-level labels are long-tailed:
  - common examples: `person`, `chair`, `car`
  - rare examples: `sheep`, `cow`

## Model Research Findings

### Source: SegNeXt official repository

- URL: https://github.com/Visual-Attention-Network/SegNeXt
- The official results table marks `MSCAN-S` as `IN-1K` pretrained.
- The repository README explicitly states that `ImageNet Pre-trained models` are provided.
- The same repository reports leaderboard results on `Pascal VOC`, but those are separate downstream results, not proof that the released `IN-1K` backbone weights touched VOC.

### Source: MMSegmentation SegNeXt configs

- URL: https://github.com/open-mmlab/mmsegmentation/tree/main/configs/segnext
- Official `SegNeXt MSCAN-S` config exists for `ADE20K`.
- The config points to the converted backbone pretrained checkpoint:
  `https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_s_20230227-f33ccdf2.pth`
- MMSeg also publishes the full `SegNeXt-S ADE20K` segmentation checkpoint in the SegNeXt model zoo.

## Policy Interpretation Findings

- The assignment restriction is about avoiding models pretrained on `PASCAL VOC`.
- Current official evidence does not show the `ADE20K` SegNeXt checkpoint being trained on `PASCAL VOC`.
- Therefore, the current working assumption is:
  - `ADE20K segmentation checkpoint` is allowed
  - `ImageNet-1K backbone checkpoint` is also allowed
- This should still be treated as an interpretation of the assignment, not a TA confirmation.

## Engineering Findings

- The base Python environment is broken for `numpy/pandas`.
- A dedicated `seg_gpu_env` is the stable execution environment for this project.
- `seg_gpu_env` requires:
  - `python 3.11`
  - full `mmcv 2.1.0` wheel matching `torch 2.1 + cu121`
  - `numpy<2`
  - `setuptools<81`
- Because the dataset is small and imbalanced, the first implementation milestone should validate preprocessing visually before any long run.
- `MMSegmentation` config loading works with the local project configs and custom imports.
- The repository now contains a shared runtime helper that:
  - registers `mmseg` plus project-specific modules
  - patches the Windows `mmengine collect_env` locale issue
  - gives consistent behavior for tests and CLI scripts
- Windows-specific runtime issues resolved in this session:
  - `mmcv._ext` blocker avoided by moving to `seg_gpu_env` with full `mmcv`
  - `OpenMP` duplicate runtime crash handled by a small compatibility layer
  - `mmengine` compiler-probe decode failure handled by a narrow runtime patch
- Validation/test pipeline should not resize masks before evaluation on this dataset.
  - Keeping original spatial resolution avoids `pred/label` shape mismatches during `IoUMetric`.
- The repository now has a local path to generate a Kaggle-ready `submission.csv`.
  - `classification` rows can be exported with placeholder defaults.
  - `segmentation` rows are encoded from per-image predicted masks using the same RLE scheme as the teacher notebook.
- Host resource probe before the first long run:
  - available RAM was low (`2.29 GB`)
  - GPU memory headroom was moderate (about `5.6 GB` free on the `RTX 4060 Laptop GPU`)
  - this host should prefer `num_workers=0` and conservative launch settings for long training jobs
- The official `ADE20K` SegNeXt-S checkpoint has now been exercised end-to-end in a short sanity run.
  - checkpoint download and initialization succeeded
  - a 2-iteration training run produced a valid best checkpoint
  - this removes the main remaining uncertainty before the first long run
- The project now contains a local delayed early-stopping hook for iterative training.
  - default policy: wait for `5` validations before activation
  - stop only after `6` non-improving `mIoU` validations with `min_delta=0.1`
  - this matches the agreed training strategy better than relying on `save_best` alone
- The first full `ADE20K`-initialized run completed successfully on this host.
  - best validation checkpoint: `best_mIoU_iter_8000.pth`
  - best validation score: `mIoU=62.46`
  - training stopped at `14000` iterations due to the configured delayed early stopping
- The first Kaggle submission from this run scored `0.36281`.
  - submission source: `submission_exp_v1_ade20k_main.csv`
  - this score should not be interpreted as a pure segmentation ceiling because `classification` was still exported as all-zero placeholder values
  - the leaderboard gap is therefore partly a modeling gap and partly an intentionally incomplete submission pipeline
- The second-round rare-focus experiment did not outperform the first baseline.
  - best validation checkpoint: `best_mIoU_iter_4000.pth`
  - best validation score: `mIoU=60.29`
  - delta versus V1: `-2.17 mIoU`
- The second-round tradeoff was uneven across the weak classes when comparing best checkpoints.
  - `bicycle`: `17.83 -> 15.37`
  - `chair`: `41.60 -> 32.05`
  - `cow`: `18.13 -> 10.88`
  - `pottedplant`: `44.53 -> 44.08`
  - `sheep`: `31.98 -> 25.85`
  - this indicates the current weak-class oversampling and focused cropping settings were too strong to improve the targeted classes reliably
- Some non-target classes remained stable or improved, but the overall balance was worse than V1.
  - examples of classes that stayed competitive or improved include `boat`, `car`, and `tvmonitor`
  - the larger loss came from broad regression on several classes rather than a single collapse
- A second-round submission candidate has been exported:
  - submission source: `submission_exp_v2_rare_focus.csv`
  - local Kaggle API calls currently fail with `401 Unauthorized`
  - consequence: leaderboard comparison for V2 still requires a manual upload from a valid Kaggle session
- The manual Kaggle result for V2 is now recorded:
  - submission source: `submission_exp_v2_rare_focus.csv`
  - public score: `0.3583`
  - interpretation: the leaderboard drop matches the local `mIoU` drop, so the second-round regression was real rather than a local-only metric artifact
- The third-round region-rebalance experiment recovered the second-round loss and slightly exceeded the first baseline locally.
  - best validation checkpoint: `best_mIoU_iter_5000.pth`
  - best validation score: `mIoU=62.54`
  - delta versus V1: `+0.08 mIoU`
  - delta versus V2: `+2.25 mIoU`
- The third-round result supports the high-level strategy change away from input-side long-tail correction.
  - V3 kept the original V1 sampling and crop pipeline
  - V3 added a training-only regional auxiliary branch instead of repeating weak-class oversampling and focused cropping
  - this is the first evidence in this project that moving the long-tail intervention from data sampling into the optimization objective is a more stable direction
- Training logs confirm the regional auxiliary branch was genuinely active.
  - `decode.loss_region_rebalance` stayed non-zero across the long run
  - this rules out the trivial failure mode where the extra branch is configured but contributes no optimization signal
- The third-round gain is real but still modest.
  - it is enough to reject the claim that the second-round failure means all long-tail strategies are bad
  - it is not yet strong enough to claim a decisive new leaderboard improvement without the Kaggle result
- A third-round submission candidate has been exported:
  - submission source: `submission_exp_v3_region_rebalance.csv`
  - classification is still placeholder `0`
  - manual Kaggle upload is required to determine whether the small local gain transfers to the leaderboard
- The manual Kaggle result for V3 is now available:
  - submission source: `submission_exp_v3_region_rebalance.csv`
  - public score: `0.36011`
  - versus V1: `0.36281 -> 0.36011`
  - versus V2: `0.3583 -> 0.36011`
- This resolves the main uncertainty around V3:
  - the third-round strategy did recover the V2 failure
  - but the gain stayed too small and too validation-specific to beat V1 on Kaggle
  - the most plausible interpretation is not “method invalid”, but “effect size too small relative to split variance and test-distribution mismatch”
- Why V3 did not beat V1 on Kaggle, despite slightly higher local `mIoU`:
  - local improvement was only `+0.08 mIoU`, which is within the range where validation-split variance can dominate
  - `classification` is still all-zero placeholder output, so the leaderboard still compresses the visible impact of pure segmentation gains
  - V3 improved some weak categories locally, but not uniformly enough to guarantee a global test-set Dice improvement
  - the hidden test distribution is likely not aligned tightly enough with the current validation split for such a small local gain to transfer reliably
- Current strategic conclusion:
  - V2 taught that input-side long-tail correction was too destructive
  - V3 shows optimization-side long-tail correction is more stable
  - but V3 does not yet justify claiming a leaderboard-winning segmentation upgrade over the original V1 baseline
- Storage cleanup is now complete:
  - only `best` and `last` checkpoints are kept for V1, V2, and V3
  - smoke and sanity weight directories were removed
  - checkpoint storage dropped from `6.938 GB` to `0.629 GB`
- Current repository state is good enough for:
  - project structure
  - data analysis and visualization
  - split generation
  - config definition
  - one-iteration training and validation smoke runs
  - second-round rare-focus split generation and focused-crop training
  - third-round region-rebalance training with a training-only auxiliary branch
  - test-time inference over the whole test set
  - `submission.csv` generation for manual Kaggle upload

## New Findings (V4 Prep)

- Teammate analysis correctly identified severe scale variation ("person very large, object very small") and co-occurrence traps ("bus vs bicycle").
- Dataset integrity check confirmed 0 corrupted files and 0 duplicates across train/test splits.
- Purely input-side fixes (like V2) or auxiliary classification losses (like V3) struggled to solve the scale variation fundamentally.
- Modifying the pixel sampler (OHEM) or the base loss function (Dice Loss) is identified as the next strongest theoretical intervention to force the network to learn hard, small objects instead of optimizing for large background/dominant classes.

## V4 OHEM Findings

- The fourth-round OHEM experiment is now the strongest local validation result in the segmentation track.
  - config: `configs/experiments/segnext_s_512x512_adamw_poly_v4_ohem.py`
  - best validation checkpoint: `best_mIoU_iter_13000.pth`
  - best validation score: `mIoU=63.90`
  - delta versus V1: `+1.44 mIoU`
  - delta versus V2: `+3.61 mIoU`
  - delta versus V3: `+1.36 mIoU`
- This result is stronger than the V3 gain and supports the hypothesis that pixel-level hard-example selection is a better intervention than input-side oversampling for this dataset.
- A fourth-round submission candidate has been exported:
  - submission source: `submission_exp_v4_ohem.csv`
  - prediction directory: `outputs/predictions/exp_v4_ohem_test`
  - classification is still placeholder `0`
  - manual Kaggle upload or team merge with the classification output is still required before interpreting leaderboard score
- Training analysis artifacts were generated under `outputs/logs/exp_v4_ohem/analysis`.
- A Colab/Pandas export compatibility issue was found and fixed:
  - error: `AssertionError` from `df.loc[idx, CLASS_NAMES]`
  - resolution: use `df.loc[idx, list(CLASS_NAMES)]`
  - affected file: `src/ga2_seg/submission.py`
- The manual Kaggle result for V4 is now available:
  - submission source: `submission_exp_v4_ohem.csv`
  - public score: `0.35237`
  - versus V1: `0.36281 -> 0.35237`
  - versus V2: `0.3583 -> 0.35237`
  - versus V3: `0.36011 -> 0.35237`
- This creates a local/leaderboard mismatch:
  - local validation improved from V3 `62.54` to V4 `63.90`
  - public score dropped from V3 `0.36011` to V4 `0.35237`
  - the likely explanation is validation split mismatch, OHEM overfitting to validation-hard pixels, altered mask calibration on hidden test images, or the still-placeholder classification output compressing pure segmentation gains
- Strategic conclusion after V4:
  - OHEM is a valuable local experiment but should not replace the best leaderboard-facing baseline yet
  - future segmentation changes need stronger cross-checks than one validation split before assuming Kaggle transfer
  - V1 remains the best public leaderboard segmentation-only submission in the tracker

## V5 CE + Dice Findings

- The fifth-round CE + Dice experiment produced the second-best local validation result so far.
  - config: `configs/experiments/segnext_s_512x512_adamw_poly_v5_ce_dice.py`
  - best validation checkpoint: `best_mIoU_iter_7000.pth`
  - best validation score: `mIoU=63.62`
  - delta versus V1: `+1.16 mIoU`
  - delta versus V3: `+1.08 mIoU`
  - delta versus V4: `-0.28 mIoU`
- A fifth-round submission candidate has been exported:
  - submission source: `submission_exp_v5_ce_dice.csv`
  - analysis artifacts: `outputs/logs/exp_v5_ce_dice/analysis`
  - classification is still placeholder `0`
- The manual Kaggle result for V5 is now available:
  - public score: `0.35553`
  - versus V1: `0.36281 -> 0.35553`
  - versus V4: `0.35237 -> 0.35553`
- Interpretation:
  - CE + Dice is less harmful than OHEM on the public leaderboard, but it still does not recover the original V1 public score.
  - Since both V4 and V5 improve local validation while regressing on Kaggle, the current validation split is not reliable enough to choose leaderboard submissions by local `mIoU` alone.
  - V1 remains the best public segmentation-only submission in the tracker; V5 should be kept as a useful experiment record, not promoted as the final baseline.

## V6 SegFormer-B2 Findings

- The sixth-round experiment changed the model family rather than adding another loss or sampler on top of SegNeXt-S.
  - base comparison target: V1 clean baseline
  - config: `configs/experiments/segformer_b2_512x512_adamw_poly_v6.py`
  - notebook: `train_v6_colab.ipynb`
  - pretrained backbone: SegFormer MiT-B2
  - data split, augmentation style, optimizer family, validation cadence, and early stopping policy stayed aligned with the earlier baseline workflow
- Local validation result:
  - best checkpoint: `best_mIoU_iter_14000.pth`
  - best validation score: `mIoU=63.46`
  - stop step: `20000`
  - versus V1: `+1.00 mIoU`
  - versus V4: `-0.44 mIoU`
  - versus V5: `-0.16 mIoU`
- Leaderboard result:
  - submission source: `submission_exp_v6_segformer_b2.csv`
  - public score: `0.37348`
  - versus V1: `+0.01067`
  - versus V4: `+0.02111`
  - versus V5: `+0.01795`
- Key finding:
  - V6 is the strongest public Kaggle result so far despite not being the strongest local validation run.
  - This strengthens the evidence that the current validation split is imperfect for public-score selection.
  - It also suggests that SegFormer-B2 generalizes better to the hidden/public test distribution than the SegNeXt-S variants tried in V1, V4, and V5.
- Practical conclusion:
  - Promote V6 as the current leaderboard-facing segmentation baseline.
  - Keep V4 and V5 as useful ablation records, but do not prefer them over V6 for public submission unless classification merge or further validation changes the ranking.

## V7 SegFormer-B3 Findings

- The seventh-round experiment kept the successful V6 recipe and only scaled the SegFormer backbone from MiT-B2 to MiT-B3.
  - config: `configs/experiments/segformer_b3_512x512_adamw_poly_v7.py`
  - notebook: `train_v7_colab.ipynb`
  - comparison target: V6 SegFormer-B2 baseline
  - pretrained backbone: SegFormer MiT-B3
- Local validation result:
  - best checkpoint: `best_mIoU_iter_8000.pth`
  - best validation score: `mIoU=64.72`
  - stop step: `14000`
  - versus V1: `+2.26 mIoU`
  - versus V6: `+1.26 mIoU`
  - versus V4: `+0.82 mIoU`
- Leaderboard result:
  - submission source: `submission_exp_v7_segformer_b3.csv`
  - public score: `0.38084`
  - versus V1: `+0.01803`
  - versus V6: `+0.00736`
  - versus V5: `+0.02531`
- Key finding:
  - V7 is the first run in this project that clearly improves both local validation and public Kaggle score over all previous versions.
  - This strongly reduces the chance that the V6 gain was a one-off validation/public mismatch.
  - The current highest-ROI direction is therefore still model-family scaling inside SegFormer, not additional loss engineering on SegNeXt-S.
- Practical conclusion:
  - Promote V7 as the new segmentation baseline for leaderboard-facing submissions.
  - Keep V6 as the direct ablation proving that SegFormer works; keep V4 and V5 as supporting negative/partial results.

## V8 SegFormer-B5 Findings

- The eighth-round experiment kept the V7 training recipe fixed and scaled the SegFormer backbone again, from MiT-B3 to MiT-B5.
  - config: `configs/experiments/segformer_b5_512x512_adamw_poly_v8.py`
  - notebook: `train_v8_colab.ipynb`
  - comparison target: V7 SegFormer-B3 baseline
  - pretrained backbone: SegFormer MiT-B5
- Local validation result:
  - best checkpoint: `best_mIoU_iter_11000.pth`
  - best validation score: `mIoU=68.97`
  - stop step: `17000`
  - versus V1: `+6.51 mIoU`
  - versus V7: `+4.25 mIoU`
  - versus V6: `+5.51 mIoU`
- Leaderboard result:
  - submission source: `submission_exp_v8_segformer_b5.csv`
  - public score: `0.38567`
  - versus V1: `+0.02286`
  - versus V7: `+0.00483`
  - versus V6: `+0.01219`
- Key finding:
  - V8 keeps the SegFormer improvement trend alive on both local validation and public Kaggle.
  - The gain from B3 to B5 is smaller than the earlier B2 -> B3 jump on the public leaderboard, but it is still real and meaningful.
  - This suggests the team is now in a regime where model scaling still helps, but the return per extra unit of compute is starting to diminish.
- Practical conclusion:
  - Promote V8 as the new best single-model segmentation baseline.
  - Use V8 as the reference point for any classification merge, ensemble, or TTA experiments.

## V9 SegFormer Hard-Vote Ensemble Findings

- The repository does not currently contain the V6/V7/V8 `.pth` checkpoints locally, so model-level TTA could not be run from this checkout without retrieving those weights from the training environment.
- A submission-level hard-vote ensemble was generated from the existing V6, V7, and V8 SegFormer submission CSV files.
  - output: `submission_exp_v9_segformer_vote_v6_v7_v8.csv`
  - tie-break baseline: V8 SegFormer-B5
  - mean changed pixels versus V8: `2.14%`
  - mean pairwise disagreement across V6/V7/V8: `5.17%`
- Interpretation:
  - The ensemble is conservative: it keeps most V8 pixels unchanged and only alters regions where the SegFormer models disagree.
  - Because V6, V7, and V8 all improved public Kaggle performance in sequence, their disagreement is a plausible source of useful corrections.
  - This candidate is worth a manual Kaggle upload before spending another multi-hour training run.
- Strategic conclusion:
  - If V9 beats V8, prioritize lightweight ensemble/TTA around the SegFormer family.
  - If V9 does not beat V8, the next expensive experiment should be a model-family upgrade such as SegMAN-B/L rather than another local loss, sampler, crop, or early-stopping tweak.
- Manual Kaggle result:
  - V9 public score: `0.39064`
  - versus V8: `+0.00497`
  - versus V7: `+0.00980`
  - versus V6: `+0.01716`
- Updated conclusion:
  - The SegFormer ensemble direction is now validated, not just hypothesized.
  - The public-score gain from changing only `2.14%` of V8 pixels implies that the model disagreement regions are high-value correction regions.
  - Before running another expensive training job, the team should prioritize retrieving model checkpoints for true TTA, merging real classification outputs, and testing additional diverse SegFormer-style ensemble members.

## V10 SegMAN Preparation Findings

- The team decided to stop the SegFormer ensemble line after V9 and move to the next model-family experiment.
- SegMAN is a credible next candidate because the official CVPR 2025 repository reports stronger ADE20K segmentation results than the SegFormer-B5 reference point used in this project.
  - official SegMAN-B ADE20K result: `52.6 mIoU`
  - official SegMAN-L ADE20K result: `53.2 mIoU`
- Engineering risk is higher than the existing MMSegmentation 1.x workflow:
  - SegMAN is based on MMSegmentation 0.30
  - it requires `natten`
  - it requires the VMamba selective scan CUDA extension
  - therefore it should run in an isolated `segman` environment, not in `seg_gpu_env`
- A project-side adapter is now available to reduce integration risk:
  - it exports GA2 data to a PNG layout compatible with MMSegmentation 0.30
  - it writes `CustomDataset` configs for SegMAN-B/L
  - it keeps background as label `0` and foreground classes as labels `1..20`, matching the existing submission format
- Current blocker before real training:
  - the official ImageNet-pretrained encoder checkpoint is not available locally yet
  - the separate SegMAN environment has not been installed or smoke-tested
