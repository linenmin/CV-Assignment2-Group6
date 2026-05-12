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
- Local environment attempt:
  - `gpu_env` remains a valid Torch GPU environment, but it is not compatible with the old SegMAN dependency stack without risking the current working setup
  - `segman_env` successfully supports `torch 2.1.2+cu121`, `mmcv-full 1.7.2`, and `mmsegmentation 0.30.0`
  - Windows remains blocked by NATTEN and selective scan CUDA extension availability
- Practical conclusion:
  - V10 should run on Linux GPU/Colab using `train_v10_segman_colab.ipynb`
  - local Windows can continue to prepare data/configs and export submissions after predictions are available
- WSL2 update:
  - WSL2 can see the NVIDIA GPU and has gcc/cmake, so it is a viable local route for SegMAN
  - `nvcc` is missing inside WSL, which blocks selective scan compilation for now
  - native Windows `segman_env` was removed; future SegMAN local training should use the WSL scripts instead
- WSL2 resolution:
  - CUDA Toolkit was already installed in WSL but was missing from PATH; adding `/usr/local/cuda/bin` exposes `nvcc`
  - Linux WSL can install the official NATTEN wheel for `torch 2.1/cu121`; the upstream SSL issue is handled with `--trusted-host shi-labs.com`
  - `selective_scan` must be built with `--no-build-isolation` because its setup script imports the active environment's PyTorch
  - the WSL `segman` environment now imports all critical runtime dependencies and is ready for SegMAN-B smoke/full training
  - the official SegMAN-B encoder checkpoint is available locally, while larger unused encoder checkpoints were removed to avoid wasting disk space

## Cityscapes External Generalization Findings

- The latest SegMAN-B segmentation model did not use Cityscapes or COCO in the project training pipeline.
  - documented initialization: official ImageNet-pretrained SegMAN encoder
  - project fine-tuning data: GA2/PASCAL VOC subset only
- Cityscapes is the better fit for the report's real-world generalization discussion than COCO.
  - COCO has better VOC-20 category coverage, but it is still a broad natural-image benchmark and is less clearly a real-world domain shift.
  - Cityscapes provides dense pixel-level annotations in urban street scenes from a different acquisition setting, so it better tests whether the model transfers beyond the VOC-style image distribution.
- The main limitation is label-space mismatch.
  - Cityscapes does not contain most VOC categories such as `cat`, `dog`, `cow`, `sheep`, `sofa`, `tvmonitor`, or `diningtable`.
  - The evaluation should therefore use only overlapping classes and report `overlap-class mIoU`.
- Candidate overlap classes:
  - `person`
  - `car`
  - `bus`
  - `bicycle`
  - `motorbike` mapped from Cityscapes `motorcycle`
  - `train`
- The result should not be compared directly against Kaggle segmentation score or VOC validation `mIoU`.
  - It answers a different question: how much the trained model degrades under a realistic street-scene domain shift.
  - A lower score can still strengthen the final report if the failure modes are visualized and interpreted clearly.
- Implementation finding:
  - Cityscapes image data on the external drive is available at `G:\Datasets\leftImg8bit_trainvaltest\leftImg8bit\val`.
  - Current fine-label data at `G:\Datasets\gtFine_trainvaltest\gtFine` only contains `test`; it does not contain the required `val` split.
  - The evaluation script must not run on the planned validation protocol until `gtFine/val` is present.
- Evaluation result after fixing the Cityscapes extraction:
  - `500` Cityscapes validation samples were evaluated.
  - overlap-class `mIoU=0.3484`.
  - strongest transfer classes: `car=0.8249`, `person=0.5241`, `bus=0.5092`.
  - weakest transfer classes: `train=0.0066`, `bicycle=0.0382`, `motorbike=0.1874`.
- Report interpretation:
  - the model learned useful object appearance for large, common street-scene categories
  - the sharp failures on `train`, `bicycle`, and `motorbike` show that the VOC/Kaggle result should not be framed as real-world deployment readiness
  - this Cityscapes result is useful precisely because it exposes domain-shift limitations rather than producing another leaderboard-style number

## EoMT-DINOv3 Research Findings

- EoMT is a CVPR 2025 Highlight model family whose core idea is to use a plain ViT with segmentation queries, avoiding the heavier adapter/decoder stacks used by many earlier high-performing segmentation systems.
- The official EoMT repository now includes DINOv3 support and reports:
  - ADE20K semantic segmentation: EoMT-L, `512x512`, `59.5 mIoU`
  - COCO panoptic segmentation: EoMT-L, `1280x1280`, `58.9 PQ`
  - COCO instance segmentation: EoMT-L, `1280x1280`, `49.9 mAP`
- Hugging Face hosts the ADE20K semantic checkpoint:
  - model id: `tue-mps/eomt-dinov3-ade-semantic-large-512`
  - task: semantic segmentation
  - backbone: DINOv3 ViT-L/16
  - input: `512x512`
  - size: about `0.3B` parameters
- Hugging Face Transformers documentation says EoMT-DINOv3 was released in 2025 and added to Transformers in 2026, with a dedicated `EomtDinov3ForUniversalSegmentation` implementation.
- LightlyTrain independently reports an EoMT+DINOv3 semantic segmentation result on ADE20K around `59.1 mIoU`, close to the official EoMT-DINOv3 `59.5 mIoU` number.
- The most important engineering warning is from the official EoMT DINOv3 model zoo:
  - DINOv3 EoMT weights are deltas with respect to original DINOv3 weights
  - users need access to the original DINOv3 weights before using these models
  - therefore, the first implementation step must be checkpoint loading validation, not full training
- Compared with SegMAN-B:
  - SegMAN-B official ADE20K result: `52.6 mIoU`
  - EoMT-DINOv3-L official ADE20K result: `59.5 mIoU`
  - this is a sufficiently large public benchmark gap to justify a model-family switch if one more expensive experiment is allowed
- Compared with SegMAN-L:
  - SegMAN-L only improves slightly over SegMAN-B in public ADE20K results
  - EoMT-DINOv3 is a better next gamble than scaling SegMAN within the same family
- Practical risk assessment:
  - EoMT-DINOv3-L is larger than SegMAN-B and may be tight on RTX 4060 16GB
  - training must start with `512x512`, mixed precision, small batch size, and gradient accumulation
  - if one-image forward or short smoke training does not fit, stop early rather than spending hours debugging a doomed run
- Recommended next action:
  - set up an isolated WSL2 `eomt` environment and run a strict checkpoint loading plus one-image inference smoke test
  - only after that passes should GA2 dataset adaptation and training be implemented
- Execution update:
  - Windows `gpu_env` was sufficient for EoMT-DINOv3 because it already had Torch CUDA, Transformers with `EomtDinov3ForUniversalSegmentation`, Lightning, timm, safetensors, and pycocotools.
  - WSL `segman` and WSL `torch` environments were not used because both were missing Transformers and would require extra package changes.
  - The machine reports about `8188 MiB` GPU memory, not 16GB, so the training strategy used batch size `1`, mixed precision, and gradient accumulation.
  - The checkpoint downloaded and loaded without manual Hugging Face intervention.
  - Head/mask-only tuning was not enough by itself: validation `mIoU=0.5715`.
  - Unfreezing the last 2 Transformer layers made the experiment competitive: validation `mIoU=0.7926`, above SegMAN-B's `0.7720`.
  - The improvement suggests EoMT-DINOv3 is not just a stronger public ADE20K model; it can also adapt to the GA2/VOC split when a small part of the backbone is fine-tuned.
  - The final judgment still requires Kaggle upload because the project has already seen local-validation gains that did not always transfer to the hidden public split.
- Post-submission debugging finding:
  - the first V11 exports used a preprocessing path that did not match training
  - training manually warped images to `512x512` before the processor, while the original test exporter passed original-aspect-ratio images directly to the processor
  - this made the model look good under the training-script validation path but poor under Kaggle-style original-size inference
  - independent validation confirmed the mismatch: original broken inference foreground `mIoU≈0.3376`, fixed training-matched inference foreground `mIoU≈0.7864`
  - the correct EoMT inference path is now: resize image to `512x512`, run processor/model, decode a `512x512` semantic mask, then resize the discrete mask back to the original image size with nearest-neighbor interpolation
  - the corrected V11 merged submission reached Kaggle public score `0.88342`, above the previous V10 final score `0.85725`
  - the V11 training curve was still improving at the final recorded step, so the model is probably not fully saturated
- Sources checked:
  - https://github.com/tue-mps/eomt
  - https://github.com/tue-mps/eomt/blob/master/model_zoo/dinov3.md
  - https://huggingface.co/tue-mps/eomt-dinov3-ade-semantic-large-512
  - https://huggingface.co/docs/transformers/v5.7.0/en/model_doc/eomt_dinov3
  - https://docs.lightly.ai/train/stable/semantic_segmentation.html

## V16 TTA Findings

- V16 keeps the V15 best checkpoint and only changes the inference path; no new training.
- Implementation: `scripts/predict_eomt_tta.py` performs a Mask2Former-style soft-semantic forward at each requested input side, dynamically rewrites `model.grid_size` so EoMT's token reshape is valid at non-512 inputs, averages soft scores at a common 512×512 reference resolution, then argmaxes and nearest-resizes back to the original image size.
- Architectural constraint that shaped the ablation: EoMT-DINOv3 hard-codes `self.grid_size = (image_size // patch_size, image_size // patch_size) = (32, 32)`. The DINOv3 backbone tolerates variable input via interpolated positional embeddings, but the EoMT decoder reshape and several `F.interpolate(..., size=self.grid_size, ...)` calls assume that exact 32×32 grid. Without patching `model.grid_size`, any non-512 input raises a `RuntimeError` in `conv1`.
- Multi-scale + horizontal-flip ablation on the 112-image validation split (V15 best, reference `mIoU=0.8020`):
  - `512` no flip = `0.8020` (matches reported V15 result, sanity check passes)
  - `512` + hflip = `0.8007`, mostly from a `diningtable` collapse `0.794 -> 0.548`
  - `{448, 512, 576}` no flip = `0.7984`
  - `{448, 512, 576}` + hflip = `0.8013`
  - `{480, 512, 544}` no flip = `0.7978`
  - `{496, 512}` no flip = `0.8027`
  - `{512, 528}` no flip = `0.8018`
  - `{496, 512, 528}` + hflip = `0.8025`
  - `{496, 512, 528}` no flip = `0.8030` ← chosen as the V16 export recipe
- Two robust findings emerged:
  - EoMT-DINOv3 fine-tuned at fixed `512×512` is much more scale-fragile than typical hierarchical encoders. Even a single step out of the patch grid (`448` or `576`) systematically drags multiple foreground classes; only the `±16` window around `512` (one patch on each side) gives a stable gain.
  - Horizontal-flip TTA helps some near-symmetric or right-skewed classes (`bicycle`, `dog`, `train`, `chair`) but tanks `diningtable` so heavily that the overall mIoU drops. This is consistent with `diningtable` images in the training set having strong left-right context priors (chairs, place settings) that flip breaks.
- Test-set behavior:
  - V16 changed `0.73%` of pixels on average versus the V15 best single-scale prediction.
  - This is comparable to the V13->V14 (`0.19%`) and V12->V14 (`0.36%`) pixel-change magnitudes that each gave non-trivial Kaggle gains.
- V16 Kaggle public score: `0.88882`, beating V15 `0.88857` by `+0.00025`.
- The transfer ratio from validation to Kaggle is now stable at about `0.25x`, i.e. each `+0.001` validation mIoU buys roughly `+0.00025` public score. The recent V14 -> V15 -> V16 deltas all sit in the same band.
- This is strong evidence the segmentation track is in a plateau dominated by validation/test variance and by the contribution of the classification component (which makes up roughly half of the public score).

## V17 Re-Split Merge-Val Fine-Tune

- Confirmed that the segmentation `train/val` split is project-defined, not assignment-defined: `scripts/build_splits.py` calls `build_train_val_split(seed=42, val_ratio=0.15)` and writes `data/splits/train.txt` and `data/splits/val.txt`. The assignment only ships `train_set.csv` and `test_set.csv`.
- Compared the segmentation split against the Image-Classification branch's split:
  - segmentation: `random.Random(42).shuffle(sorted_ids)`, `val_ratio=0.15` -> 637/112
  - classification: `sklearn.train_test_split(random_state=42, test_size=0.2)` -> 599/150
  - the two validation sets share only `22` images. 90 images are seg-val but cls-train; 128 are cls-val but seg-train. There is no project-wide held-out subset.
- This means the segmentation 112 val is not a Kaggle-faithful reference, and earlier mini-deltas (V14 -> V15: `+0.0001` local) reflect that fact: such gains may simply be characteristics of this specific 112-image subset rather than true generalization improvements.
- V17 lowers the segmentation `val_ratio` to `0.05`, keeping the same seed, which produces a new validation set that is a strict subset of the old one. 75 images move `validation -> training`, 0 move the other way. Old split is backed up to `training_v15split.txt` / `validation_v15split.txt`.
- The model is then continued from V15 best at `lr=1e-5, unfreeze_last_layers=2, max_steps=2000, early_stop_patience=8`. Best `val_mIoU=0.7923` at step `1000`, early-stopped at step `1800`.
- Fair comparison on the new 37 val with `ms{496,512,528}` no-flip TTA:
  - V15 best: `mIoU=0.7859`
  - V17 best: `mIoU=0.7930`
  - delta `+0.0071` is the strongest training-side gain since V12 -> V14
- V17 test predictions changed `0.41%` of pixels versus V16 on average.
- V17 Kaggle public score: `0.89139`, beating V16 `0.88882` by `+0.00257`. This is the largest single-step Kaggle gain since V11 -> V12 (`+0.00260`).
- The realised local-to-Kaggle transfer ratio is `0.00257 / 0.0071 ≈ 0.36x`, modestly above the recent `~0.25x` band, which is consistent with the gain reflecting genuinely new training signal rather than 37-val noise.
- This is empirical confirmation that the original `val_ratio=0.15` was an over-conservative project-internal choice: 75 of those 112 images were carrying real training value that the model could not access until V17 released them.

## Sliding-Window Negative Finding (internal experiment, no Kaggle submission)

- Motivation: V11 training warps every image to `512x512` before the EoMT processor, so the network has only seen perspective-distorted inputs. Sliding-window inference was meant to feed the model `512x512` crops from an aspect-preserving resize, preserving both the trained resolution and the hard-coded `32x32` token grid for free, while showing the model un-warped content that small / thin objects (`bicycle`, `chair`) need.
- Implementation: `scripts/predict_eomt_sliding_window.py`. Each image is resized to short-side `= 512` keeping aspect ratio, then square `512x512` windows are slid with a configurable stride. Soft Mask2Former semantic scores are averaged in image space with a per-pixel count buffer, then argmaxed and nearest-resized to the original size. Validation is reported both at the `512x512` warped protocol (matching V11..V16) and at the native image size (closer to what Kaggle actually scores).
- Result on the V15 best checkpoint (112 validation images):
  - `window=512, stride=256, no flip`: primary `mIoU=0.7941`, original-size `mIoU=0.7968`
  - `window=512, stride=128, no flip`: primary `mIoU=0.7952`, original-size `mIoU=0.7979`
  - V16 baseline for comparison: primary `mIoU=0.8030`, original-size `mIoU=0.8049`
- Per-class diagnosis at `stride=256, no flip` versus V16:
  - `diningtable` collapsed from `0.779` to `0.575` (`-0.20`)
  - `train` dropped from `0.882` to `0.832`
  - `pottedplant` dropped from `0.771` to `0.758`
  - `dog`, `bus`, `sheep`, and `sofa` improved by `+0.05` to `+0.13`
  - `bicycle` stayed near `0.21` regardless of stride, so the hypothesis that sliding-window would rescue the thin-object class is not supported here
- Interpretation:
  - the model was fine-tuned on full-image-warped `512x512` inputs and has learned strong "this image fits in one 512 square" priors
  - aspect-preserving sliding-window crops are out-of-distribution relative to training, particularly for classes that depend on full-image context (`diningtable`)
  - the same failure pattern appeared earlier with horizontal-flip TTA, where `diningtable` also collapsed by ~`0.24`; both are consistent with the model relying on warped-frame layout priors that aspect-preserving crops and flipped contexts do not satisfy
- Methodological value:
  - this is a clean negative result that strengthens, rather than weakens, the report
  - it shows that the V11 preprocessing choice is effectively locked in once fine-tuning begins, and that recovering the cost would require retraining with aspect-preserving augmentation rather than another inference-time trick
  - the practical implication is that the segmentation track has now exhausted both training-side and inference-side cheap improvements; further gains require a structural change such as retraining at native aspect ratio or improving the classification half of the Kaggle pipeline
- Decision:
  - do not submit the sliding-window candidate to Kaggle; spending a leaderboard slot to confirm a regression would not provide additional information
  - V16 (`outputs/submissions/submission_exp_v16_eomt_dinov3_tta_ms496_512_528_noflip_with_convnext_small_320.csv`, public score `0.88882`) remains the recommended segmentation submission
- Implication: the next inference-side TTA candidate that has a chance of further help is a true sliding-window pass at the native aspect ratio (each window fed at exactly `512×512` so the grid assumption is honoured). Anything else that breaks the trained grid is risky on this checkpoint.
