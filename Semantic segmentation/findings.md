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
