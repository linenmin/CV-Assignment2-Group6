# Progress: Semantic Segmentation V1

## Session Log

### 2026-04-21

- Created branch `segmentation`.
- Confirmed the user-approved baseline direction: `MMSegmentation + SegNeXt-S`.
- Built and validated a dedicated `seg_gpu_env` for this project.
- Confirmed official `SegNeXt` sources provide:
  - `IN-1K` pretrained `MSCAN-S` backbone weights
  - `ADE20K` full segmentation checkpoints in MMSegmentation
- Recorded the current working decision:
  - main initialization: `ADE20K segmentation checkpoint`
  - control initialization: `ImageNet-1K backbone checkpoint`
- Created persistent planning files in `Semantic segmentation/`:
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
- Added the initial maintainable project skeleton:
  - `src/ga2_seg/`
  - `scripts/`
  - `tests/`
  - `configs/`
- Implemented and verified:
  - dataset path helpers
  - class metadata
  - deterministic split builder
  - dataset statistics generation
  - sample and transform visualization generation
- Installed and verified core Python dependencies used so far:
  - `pytest`
  - `mmengine`
  - `mmsegmentation`
  - `mmcv-lite`
  - `ftfy`
- Added `SegNeXt-S` experiment config using the official `ADE20K` checkpoint as the default initialization.
- Added shared runtime compatibility helpers for:
  - `mmseg` module registration
  - Windows OpenMP duplicate-runtime handling
  - Windows `mmengine collect_env` locale decoding fallback
- Added maintainable entrypoints:
  - `scripts/train.py`
  - `scripts/validate.py`
- Added maintainable test-time utilities:
  - `scripts/predict_test_segmentation.py`
  - `scripts/export_submission.py`
- Added reusable inference and submission modules:
  - `src/ga2_seg/inference.py`
  - `src/ga2_seg/submission.py`
- Fixed the validation pipeline shape mismatch by removing `Resize` from the val/test pipeline.
- Verified in `seg_gpu_env`:
  - full pytest suite passes
  - `analyze_dataset.py`, `visualize_samples.py`, and `build_splits.py` run successfully
  - a one-iteration training smoke run completes and writes checkpoints
  - validation from the same training run reports metrics successfully
  - test-time segmentation prediction runs across all `750` test images
  - `submission.csv` export works with placeholder classification labels
- Probed local resources before the first long run:
  - CPU: `28` logical cores
  - RAM available at probe time: `2.29 GB`
  - GPU: `RTX 4060 Laptop GPU`, `8188 MB` total, about `5592 MB` free
- Decided launch profile for the first long run:
  - initialization: official `ADE20K` checkpoint
  - `work_dir`: `outputs/logs/exp_v1_ade20k_main`
  - `num_workers=0`
  - conservative batch setting for the first full launch to reduce failure risk on this host
- Completed a pretrained sanity run:
  - command: `python .\scripts\train.py --max-iters 2 --val-interval 1 --batch-size 2 --num-workers 0 --work-dir .\outputs\logs\exp_v1_ade20k_sanity`
  - result: training completed and saved `best_mIoU_iter_2.pth`
  - implication: the official `ADE20K` checkpoint loads correctly in the current environment and the project is ready for the first long run
- Implemented conservative delayed early stopping for long runs:
  - hook: `DelayedEarlyStoppingHook`
  - activation: start checking after `5` validation records
  - stop rule: no `mIoU` improvement larger than `0.1` for `6` validations
  - reason: preserve late-improvement headroom while still preventing unproductive full-length runs
- Re-ran the pretrained sanity training with the early-stopping hook enabled:
  - command: `python .\scripts\train.py --max-iters 2 --val-interval 1 --batch-size 2 --num-workers 0 --work-dir .\outputs\logs\exp_v1_ade20k_sanity_earlystop`
  - result: training still completed successfully and saved `best_mIoU_iter_2.pth`
- Completed the first full ADE20K-initialized training run:
  - command: `python .\scripts\train.py --batch-size 2 --num-workers 0 --work-dir .\outputs\logs\exp_v1_ade20k_main`
  - stop condition: delayed early stopping triggered after the validation at `14000` iterations
  - best checkpoint: `outputs/logs/exp_v1_ade20k_main/best_mIoU_iter_8000.pth`
  - best validation metric: `mIoU=62.46`
  - last validation before stop: `mIoU=60.39` at `14000` iterations
- Exported the first leaderboard-facing submission from the best checkpoint:
  - submission file: `outputs/submissions/submission_exp_v1_ade20k_main.csv`
  - classification mode: placeholder `0`
- Recorded the first Kaggle result:
  - public score: `0.36281`
  - tracker file: `kaggle_scores.md`
  - interpretation: this is a usable segmentation baseline, but the score is bottlenecked by placeholder classification outputs
- Implemented the second-round rare-focus experiment without changing backbone or pretrained initialization:
  - duplicated training samples that contain `bicycle`, `chair`, `cow`, `sheep`, or `pottedplant`
  - generated `data/splits/train_rare_focus_v2.txt` with `811` entries from the original `637` train ids
  - replaced plain random crop with a rare-class-focused crop that uses the same crop size but can center on weak-class pixels
  - tightened the single-category crop ratio from `0.75` to `0.65`
- Verified the second-round pipeline before the long run:
  - `python .\scripts\build_rare_focus_split.py`
  - `python -m pytest .\tests -q`
  - `python .\scripts\train.py --config .\configs\experiments\segnext_s_512x512_adamw_poly_v2_rare_focus.py --max-iters 1 --val-interval 1 --batch-size 1 --num-workers 0 --work-dir .\outputs\logs\smoke_v2_rare_focus`
- Completed the second-round rare-focus training run:
  - command: `python .\scripts\train.py --config .\configs\experiments\segnext_s_512x512_adamw_poly_v2_rare_focus.py --num-workers 0 --work-dir .\outputs\logs\exp_v2_rare_focus`
  - stop condition: delayed early stopping triggered after the validation at `11000` iterations
  - best checkpoint: `outputs/logs/exp_v2_rare_focus/best_mIoU_iter_4000.pth`
  - best validation metric: `mIoU=60.29`
  - comparison to V1: `-2.17 mIoU` versus the `62.46` baseline
- Exported the second-round leaderboard-facing submission candidate:
  - prediction directory: `outputs/predictions/exp_v2_rare_focus_test`
  - submission file: `outputs/submissions/submission_exp_v2_rare_focus.csv`
  - classification mode: placeholder `0`
- Attempted to prepare direct Kaggle submission from this host:
  - installed the `kaggle` CLI in `seg_gpu_env`
  - local authentication currently returns `401 Unauthorized`
  - implication: the V2 submission file is ready, but the Kaggle score still needs manual upload from a valid Kaggle session
- Implemented the third-round region-rebalance experiment as a strategy change rather than another input-pipeline tweak:
  - restored the V1 training split and standard random-crop pipeline
  - added a training-only `RegionRebalanceLightHamHead`
  - added class-frequency-aware regional auxiliary classification on top of the decode features
  - kept inference on the standard segmentation output, so test-time complexity stayed effectively unchanged
- Verified the third-round pipeline before the long run:
  - `python -m pytest .\tests -q`
  - `python .\scripts\train.py --config .\configs\experiments\segnext_s_512x512_adamw_poly_v3_region_rebalance.py --max-iters 1 --val-interval 1 --batch-size 1 --num-workers 0 --work-dir .\outputs\logs\smoke_v3_region_rebalance`
  - smoke run confirmed `decode.loss_region_rebalance` appears in the training scalars
- Completed the third-round region-rebalance training run:
  - command: `python .\scripts\train.py --config .\configs\experiments\segnext_s_512x512_adamw_poly_v3_region_rebalance.py --num-workers 0 --work-dir .\outputs\logs\exp_v3_region_rebalance`
  - stop condition: delayed early stopping triggered after the validation at `11000` iterations
  - best checkpoint: `outputs/logs/exp_v3_region_rebalance/best_mIoU_iter_5000.pth`
  - best validation metric: `mIoU=62.54`
  - comparison to V1: `+0.08 mIoU`
  - comparison to V2: `+2.25 mIoU`
- Exported the third-round leaderboard-facing submission candidate:
  - prediction directory: `outputs/predictions/exp_v3_region_rebalance_test`
  - submission file: `outputs/submissions/submission_exp_v3_region_rebalance.csv`
  - classification mode: placeholder `0`
- Generated third-round training analysis artifacts:
  - `outputs/logs/exp_v3_region_rebalance/analysis/training_curves.png`
  - `outputs/logs/exp_v3_region_rebalance/analysis/runtime_diagnostics.png`
  - `outputs/logs/exp_v3_region_rebalance/analysis/validation_metrics.csv`
  - `outputs/logs/exp_v3_region_rebalance/analysis/summary.md`
- Recorded the manual Kaggle result for the third-round submission:
  - public score: `0.36011`
  - comparison to V1: `-0.00270`
  - comparison to V2: `+0.00181`
- Started storage cleanup after the third-round comparison:
  - target policy: keep only `best` and `last` checkpoints for each main experiment
  - remove intermediate `iter_*.pth` files and smoke/sanity weights
- Completed checkpoint cleanup:
  - kept `6` checkpoint files across the three main experiments
  - removed all intermediate checkpoints plus smoke/sanity weight directories
  - weight storage reduced from `6.938 GB` to `0.629 GB`
  - reclaimed disk space: `6.309 GB`

## Current Focus

- Close out the segmentation track around the validated `SegMAN-B` result.
- Keep the repository state reproducible and lightweight:
  - preserve the final tracked submission CSV files
  - preserve only useful SegMAN checkpoints in ignored training outputs
  - keep planning, findings, and score-tracking documents aligned

## Verification Notes

- Latest verification completed in `seg_gpu_env`:
  - `python -m pytest .\\tests -v`
  - `python .\\scripts\\analyze_dataset.py`
  - `python .\\scripts\\visualize_samples.py --num-samples 2`
  - `python .\\scripts\\build_splits.py`
  - `python .\\scripts\\train.py --max-iters 1 --val-interval 1 --batch-size 1 --num-workers 0 --no-pretrained --work-dir .\\outputs\\logs\\smoke_train`
  - `python .\\scripts\\train.py --max-iters 2 --val-interval 1 --batch-size 2 --num-workers 0 --work-dir .\\outputs\\logs\\exp_v1_ade20k_sanity`
  - `python .\\scripts\\train.py --max-iters 2 --val-interval 1 --batch-size 2 --num-workers 0 --work-dir .\\outputs\\logs\\exp_v1_ade20k_sanity_earlystop`
  - `python .\\scripts\\train.py --batch-size 2 --num-workers 0 --work-dir .\\outputs\\logs\\exp_v1_ade20k_main`
  - `python .\\scripts\\build_rare_focus_split.py`
  - `python .\\scripts\\train.py --config .\\configs\\experiments\\segnext_s_512x512_adamw_poly_v2_rare_focus.py --max-iters 1 --val-interval 1 --batch-size 1 --num-workers 0 --work-dir .\\outputs\\logs\\smoke_v2_rare_focus`
  - `python .\\scripts\\train.py --config .\\configs\\experiments\\segnext_s_512x512_adamw_poly_v2_rare_focus.py --num-workers 0 --work-dir .\\outputs\\logs\\exp_v2_rare_focus`
  - `python .\\scripts\\validate.py --checkpoint .\\outputs\\logs\\smoke_train\\best_mIoU_iter_1.pth --num-workers 0`
  - `python .\\scripts\\predict_test_segmentation.py --checkpoint .\\outputs\\logs\\smoke_train\\best_mIoU_iter_1.pth --output-dir .\\outputs\\predictions\\test_smoke_submission`
  - `python .\\scripts\\export_submission.py --prediction-dir .\\outputs\\predictions\\test_smoke_submission --output-path .\\outputs\\submissions\\submission.csv --classification-fill 0`
  - `python .\\scripts\\predict_test_segmentation.py --config .\\configs\\experiments\\segnext_s_512x512_adamw_poly_v2_rare_focus.py --checkpoint .\\outputs\\logs\\exp_v2_rare_focus\\best_mIoU_iter_4000.pth --output-dir .\\outputs\\predictions\\exp_v2_rare_focus_test`
  - `python .\\scripts\\export_submission.py --prediction-dir .\\outputs\\predictions\\exp_v2_rare_focus_test --output-path .\\outputs\\submissions\\submission_exp_v2_rare_focus.csv --classification-fill 0`
  - `python .\\scripts\\train.py --config .\\configs\\experiments\\segnext_s_512x512_adamw_poly_v3_region_rebalance.py --max-iters 1 --val-interval 1 --batch-size 1 --num-workers 0 --work-dir .\\outputs\\logs\\smoke_v3_region_rebalance`
  - `python .\\scripts\\train.py --config .\\configs\\experiments\\segnext_s_512x512_adamw_poly_v3_region_rebalance.py --num-workers 0 --work-dir .\\outputs\\logs\\exp_v3_region_rebalance`
  - `python .\\scripts\\predict_test_segmentation.py --config .\\configs\\experiments\\segnext_s_512x512_adamw_poly_v3_region_rebalance.py --checkpoint .\\outputs\\logs\\exp_v3_region_rebalance\\best_mIoU_iter_5000.pth --output-dir .\\outputs\\predictions\\exp_v3_region_rebalance_test`
  - `python .\\scripts\\export_submission.py --prediction-dir .\\outputs\\predictions\\exp_v3_region_rebalance_test --output-path .\\outputs\\submissions\\submission_exp_v3_region_rebalance.csv --classification-fill 0`
  - `python .\\scripts\\analyze_training_run.py --work-dir .\\outputs\\logs\\exp_v3_region_rebalance --submission-file submission_exp_v3_region_rebalance.csv`

### 2026-04-22 (V4 Experiment)

- Analyzed dataset "traps" (extreme scale variation, multi-label/co-occurrence confusion).
- Confirmed the dataset integrity (no corruptions, no duplicates) via `check_dataset_anomalies.py`.
- Formulated V4 strategy: introduce `OHEMPixelSampler` to force the network to focus on hard, rare, or small examples (e.g., bicycles, sheep).
- Created V4 config `segnext_s_512x512_adamw_poly_v4_ohem.py` inheriting from V1 and overriding `train_cfg`.
- Executed strict SOP before training (updated `task_plan.md`, `progress.md`, `findings.md`, and prepared for `git push`).
- Completed the fourth-round OHEM training run on Colab L4:
  - command: `python scripts/train.py --config configs/experiments/segnext_s_512x512_adamw_poly_v4_ohem.py --num-workers 4 --batch-size 2 --work-dir outputs/logs/exp_v4_ohem`
  - stop condition: delayed early stopping triggered after the monitored `mIoU` did not improve for `6` validation records
  - best checkpoint: `outputs/logs/exp_v4_ohem/best_mIoU_iter_13000.pth`
  - best validation metric: `mIoU=63.90`
  - comparison to V1: `+1.44 mIoU`
  - comparison to V2: `+3.61 mIoU`
  - comparison to V3: `+1.36 mIoU`
- Exported the fourth-round leaderboard-facing submission candidate:
  - prediction directory: `outputs/predictions/exp_v4_ohem_test`
  - predictions written: `750`
  - submission file: `outputs/submissions/submission_exp_v4_ohem.csv`
  - submission rows: `1500`
  - classification mode: placeholder `0`
- Generated fourth-round training analysis artifacts:
  - `outputs/logs/exp_v4_ohem/analysis/training_curves.png`
  - `outputs/logs/exp_v4_ohem/analysis/runtime_diagnostics.png`
  - `outputs/logs/exp_v4_ohem/analysis/validation_metrics.csv`
  - `outputs/logs/exp_v4_ohem/analysis/summary.md`
- Fixed Colab submission export compatibility:
  - `src/ga2_seg/submission.py` now uses `df.loc[idx, list(CLASS_NAMES)]`
  - `train_v4_colab.ipynb` includes a runtime patch before `export_submission.py` for cloned Colab checkouts
- Current V4 status:
  - local validation result is recorded
  - submission file is generated
  - Kaggle score is pending manual upload or merge with the classification teammate output
- Recorded the manual Kaggle result for the fourth-round OHEM submission:
  - public score: `0.35237`
  - comparison to V1: `-0.01044`
  - comparison to V2: `-0.00593`
  - comparison to V3: `-0.00774`
  - interpretation: V4 has the best local validation `mIoU`, but the public score regressed, so it should be kept as an experiment record rather than promoted as the leaderboard baseline

### 2026-04-23 (V5 Experiment)

- Implemented the fifth-round CE + Dice experiment from the V1 baseline:
  - config: `configs/experiments/segnext_s_512x512_adamw_poly_v5_ce_dice.py`
  - notebook: `train_v5_colab.ipynb`
  - objective: `CrossEntropyLoss(loss_weight=1.0) + DiceLoss(loss_weight=0.5)`
- Completed the fifth-round CE + Dice training run on Colab T4:
  - command: `python scripts/train.py --config configs/experiments/segnext_s_512x512_adamw_poly_v5_ce_dice.py --num-workers 4 --batch-size 2 --work-dir outputs/logs/exp_v5_ce_dice`
  - stop condition: delayed early stopping triggered after the validation at `13000` iterations
  - best checkpoint: `outputs/logs/exp_v5_ce_dice/best_mIoU_iter_7000.pth`
  - best validation metric: `mIoU=63.62`
  - comparison to V1: `+1.16 mIoU`
  - comparison to V3: `+1.08 mIoU`
  - comparison to V4: `-0.28 mIoU`
- Exported the fifth-round leaderboard-facing submission candidate:
  - submission file: `outputs/submissions/submission_exp_v5_ce_dice.csv`
  - classification mode: placeholder `0`
- Generated fifth-round training analysis artifacts:
  - `outputs/logs/exp_v5_ce_dice/analysis/training_curves.png`
  - `outputs/logs/exp_v5_ce_dice/analysis/runtime_diagnostics.png`
  - `outputs/logs/exp_v5_ce_dice/analysis/validation_metrics.csv`
  - `outputs/logs/exp_v5_ce_dice/analysis/summary.md`
- Recorded the manual Kaggle result for the fifth-round CE + Dice submission:
  - public score: `0.35553`
  - comparison to V1: `-0.00728`
  - comparison to V4: `+0.00316`
  - interpretation: V5 is locally strong and better than V4 on public Kaggle, but it still does not beat the V1 public baseline

### 2026-04-24 (V6 Experiment)

- Implemented the sixth-round SegFormer-B2 baseline:
  - config: `configs/experiments/segformer_b2_512x512_adamw_poly_v6.py`
  - notebook: `train_v6_colab.ipynb`
  - experiment basis: return to the V1 clean baseline workflow and replace the SegNeXt-S model family with SegFormer-B2 / MiT-B2
  - reason: V4 and V5 improved local validation but failed to beat V1 on Kaggle, so the next meaningful variable was hidden-test generalization from a different model family
- Completed the sixth-round training run on Colab L4:
  - command: `python scripts/train.py --config configs/experiments/segformer_b2_512x512_adamw_poly_v6.py --num-workers 4 --batch-size 2 --work-dir outputs/logs/exp_v6_segformer_b2`
  - stop condition: delayed early stopping after the monitored `mIoU` did not improve for `6` validation records
  - best checkpoint: `outputs/logs/exp_v6_segformer_b2/best_mIoU_iter_14000.pth`
  - best validation metric: `mIoU=63.46`
  - stop step: `20000`
- Exported the sixth-round leaderboard-facing submission candidate:
  - prediction directory: `outputs/predictions/exp_v6_segformer_b2_test`
  - predictions written: `750`
  - submission file: `outputs/submissions/submission_exp_v6_segformer_b2.csv`
  - submission rows: `1500`
  - classification mode: placeholder `0`
- Generated sixth-round training analysis artifacts:
  - `outputs/logs/exp_v6_segformer_b2/analysis/training_curves.png`
  - `outputs/logs/exp_v6_segformer_b2/analysis/runtime_diagnostics.png`
  - `outputs/logs/exp_v6_segformer_b2/analysis/validation_metrics.csv`
  - `outputs/logs/exp_v6_segformer_b2/analysis/summary.md`
- Recorded the manual Kaggle result for the sixth-round SegFormer-B2 submission:
  - public score: `0.37348`
  - comparison to V1: `+0.01067`
  - comparison to V4: `+0.02111`
  - comparison to V5: `+0.01795`
  - interpretation: V6 is the best public leaderboard result so far and should become the current segmentation baseline for the next round
- Repository maintenance note:
  - V6 analysis artifacts are under an ignored `outputs/logs` path, so they were intentionally staged with `git add -f`

### 2026-04-24 (V7 Experiment)

- Implemented the seventh-round SegFormer-B3 baseline:
  - config: `configs/experiments/segformer_b3_512x512_adamw_poly_v7.py`
  - notebook: `train_v7_colab.ipynb`
  - experiment basis: keep the V6 SegFormer-B2 recipe fixed and scale the backbone from MiT-B2 to MiT-B3
  - reason: V6 established that changing the model family improved public generalization, so the next clean step was capacity scaling inside the same family
- Completed the seventh-round training run on Colab L4:
  - command: `python scripts/train.py --config configs/experiments/segformer_b3_512x512_adamw_poly_v7.py --num-workers 4 --batch-size 2 --work-dir outputs/logs/exp_v7_segformer_b3`
  - stop condition: delayed early stopping after the monitored `mIoU` did not improve for `6` validation records
  - best checkpoint: `outputs/logs/exp_v7_segformer_b3/best_mIoU_iter_8000.pth`
  - best validation metric: `mIoU=64.72`
  - stop step: `14000`
- Exported the seventh-round leaderboard-facing submission candidate:
  - submission file: `outputs/submissions/submission_exp_v7_segformer_b3.csv`
  - submission rows: `1500`
  - classification mode: placeholder `0`
- Generated seventh-round training analysis artifacts:
  - `outputs/logs/exp_v7_segformer_b3/analysis/training_curves.png`
  - `outputs/logs/exp_v7_segformer_b3/analysis/runtime_diagnostics.png`
  - `outputs/logs/exp_v7_segformer_b3/analysis/validation_metrics.csv`
  - `outputs/logs/exp_v7_segformer_b3/analysis/summary.md`
- Recorded the manual Kaggle result for the seventh-round SegFormer-B3 submission:
  - public score: `0.38084`
  - comparison to V1: `+0.01803`
  - comparison to V6: `+0.00736`
  - comparison to V5: `+0.02531`
  - interpretation: V7 is the best public leaderboard result so far and becomes the new segmentation baseline for the next round
- Repository maintenance note:
  - V7 analysis artifacts are under an ignored `outputs/logs` path, so they were intentionally staged with `git add -f`
  - `outputs/logs/exp_v7_segformer_b3/analysis/summary.md` had a stale Kaggle score and was corrected to the actual public result `0.38084` before recording

### 2026-04-24 (V8 Experiment)

- Implemented the eighth-round SegFormer-B5 baseline:
  - config: `configs/experiments/segformer_b5_512x512_adamw_poly_v8.py`
  - notebook: `train_v8_colab.ipynb`
  - experiment basis: keep the V7 SegFormer-B3 recipe fixed and scale the backbone from MiT-B3 to MiT-B5
  - reason: V6 and V7 showed a consistent gain from SegFormer scaling, so the next clean step was to test whether a larger backbone would still transfer to Kaggle
- Completed the eighth-round training run on Colab L4:
  - command: `python scripts/train.py --config configs/experiments/segformer_b5_512x512_adamw_poly_v8.py --num-workers 4 --batch-size 2 --work-dir outputs/logs/exp_v8_segformer_b5`
  - stop condition: delayed early stopping after the monitored `mIoU` did not improve for `6` validation records
  - best checkpoint: `outputs/logs/exp_v8_segformer_b5/best_mIoU_iter_11000.pth`
  - best validation metric: `mIoU=68.97`
  - stop step: `17000`
- Exported the eighth-round leaderboard-facing submission candidate:
  - submission file: `outputs/submissions/submission_exp_v8_segformer_b5.csv`
  - submission rows: `1500`
  - classification mode: placeholder `0`
- Generated eighth-round training analysis artifacts:
  - `outputs/logs/exp_v8_segformer_b5/analysis/training_curves.png`
  - `outputs/logs/exp_v8_segformer_b5/analysis/runtime_diagnostics.png`
  - `outputs/logs/exp_v8_segformer_b5/analysis/validation_metrics.csv`
  - `outputs/logs/exp_v8_segformer_b5/analysis/summary.md`
- Recorded the manual Kaggle result for the eighth-round SegFormer-B5 submission:
  - public score: `0.38567`
  - comparison to V1: `+0.02286`
  - comparison to V7: `+0.00483`
  - comparison to V6: `+0.01219`
  - interpretation: V8 is the best public leaderboard result so far and becomes the new segmentation baseline for the next round
- Repository maintenance note:
  - V8 analysis artifacts are under an ignored `outputs/logs` path, so they were intentionally staged with `git add -f`
  - `outputs/logs/exp_v8_segformer_b5/analysis/summary.md` was updated to include the actual public result `0.38567`

### 2026-04-24 (V9 Submission Ensemble Candidate)

- Synchronized the Kaggle score tracker with the SegFormer experiments:
  - V6 SegFormer-B2: `0.37348`
  - V7 SegFormer-B3: `0.38084`
  - V8 SegFormer-B5: `0.38567`
- Added a reusable hard-vote submission ensemble path:
  - module: `src/ga2_seg/submission_ensemble.py`
  - script: `scripts/ensemble_submissions.py`
  - tests: `tests/test_submission_ensemble.py`
- Generated a V9 candidate without additional training:
  - command: `python scripts/ensemble_submissions.py --submission outputs/submissions/submission_exp_v6_segformer_b2.csv --submission outputs/submissions/submission_exp_v7_segformer_b3.csv --submission outputs/submissions/submission_exp_v8_segformer_b5.csv --output-path outputs/submissions/submission_exp_v9_segformer_vote_v6_v7_v8.csv --primary-index 2`
  - output: `outputs/submissions/submission_exp_v9_segformer_vote_v6_v7_v8.csv`
  - images: `750`
  - mean changed pixels versus V8: `2.14%`
  - mean pairwise disagreement across V6/V7/V8: `5.17%`
- Decision:
  - V9 is worth a manual Kaggle upload because it is a conservative test-time ensemble candidate and requires no new training run.
  - If it does not beat V8, the next high-cost experiment should be a model-family upgrade such as SegMAN-B/L rather than another small loss, sampler, or crop modification.

### 2026-04-24 (V9 Kaggle Result)

- Recorded the manual Kaggle result for the V9 SegFormer hard-vote ensemble:
  - submission file: `outputs/submissions/submission_exp_v9_segformer_vote_v6_v7_v8.csv`
  - public score: `0.39064`
  - comparison to V8: `+0.00497`
  - comparison to V7: `+0.00980`
  - comparison to V6: `+0.01716`
- Interpretation:
  - V9 is now the best public leaderboard result in the segmentation track.
  - The gain confirms that V6/V7/V8 carry complementary errors and that a conservative vote over their disagreement regions improves hidden-test performance.
  - Because the ensemble changed only `2.14%` of V8 pixels, the result is strong evidence that the next immediate work should focus on better ensemble/TTA around SegFormer before paying for another full training run.

### 2026-04-24 (V10 SegMAN Preparation)

- Shifted the next high-cost experiment to SegMAN after the team decided to stop the SegFormer ensemble line.
- Verified official SegMAN project facts from the upstream repository:
  - paper: CVPR 2025
  - repository: `https://github.com/yunxiangfu2001/SegMAN`
  - official implementation depends on MMSegmentation 0.30, NATTEN, and VMamba selective scan
  - official ADE20K table reports SegMAN-B `52.6 mIoU` and SegMAN-L `53.2 mIoU`
- Cloned the official repository locally under ignored project content:
  - `Semantic segmentation/external/SegMAN`
- Added a project-side SegMAN adapter:
  - `src/ga2_seg/segman_adapter.py`
  - exports GA2 `.npy` arrays to PNG files for MMSegmentation 0.30
  - writes GA2 SegMAN-B/L configs into the official repository checkout
- Added a CLI wrapper:
  - `scripts/prepare_segman_experiment.py`
- Exported the dataset for SegMAN:
  - train: `637`
  - validation: `112`
  - test: `750`
  - export root: `data/segman_ga2`
- Generated the first V10 config:
  - `external/SegMAN/segmentation/local_configs/segman/ga2/segman_b_ga2.py`
- Validation:
  - `conda run -n seg_gpu_env python -m pytest .\tests\test_segman_adapter.py .\tests\test_submission_ensemble.py -q`
  - result: `5 passed`
- Current blocker:
  - the ImageNet-pretrained encoder file is not present at `external/SegMAN/pretrained/SegMAN_Encoder_b.pth.tar`
  - the official SegMAN environment still needs to be created separately before smoke training

### 2026-04-24 (V10 Local Environment Attempt)

- Checked `gpu_env` first because it is the existing working Torch GPU chain:
  - Python `3.12.3`
  - Torch `2.5.1+cu121`
  - CUDA available on RTX 4060 Laptop GPU
  - `mmseg` was present but unusable because `mmcv` was missing
- Decided not to force old SegMAN dependencies into `gpu_env` because SegMAN targets MMSegmentation 0.30 and `mmcv-full 1.x`; mixing that into Python 3.12/Torch 2.5 would risk breaking the existing working environment.
- Created an isolated `segman_env`:
  - Python `3.10`
  - Torch `2.1.2+cu121`
  - CUDA available on RTX 4060 Laptop GPU
  - `mmcv-full 1.7.2`
  - official SegMAN `mmsegmentation 0.30.0`
- Installed SegMAN requirements except `triton`, which is not available for Windows in this setup.
- Tried NATTEN installation:
  - official wheel command fails because the upstream wheel index only provides Linux wheels for this stack
  - source build with `--no-build-isolation` fails locally and reports missing CMake; it also warns that Windows + CUDA 12 is a known risk for Torch CMake builds
- Added Linux GPU execution path:
  - `train_v10_segman_colab.ipynb`
  - `scripts/predict_segman_test.py`
- Validation:
  - `segman_env` imports `torch`, `mmcv`, and `mmseg` successfully
  - `python -m py_compile scripts/predict_segman_test.py scripts/prepare_segman_experiment.py` passes
  - `conda run -n seg_gpu_env python -m pytest .\tests\test_segman_adapter.py .\tests\test_submission_ensemble.py -q` passes with `5 passed`

### 2026-04-24 (V10 WSL2 Route)

- Checked WSL2 as the preferred local SegMAN route:
  - distro: `Ubuntu-22.04`
  - WSL version: `2`
  - `nvidia-smi` works inside WSL and sees the RTX 4060 Laptop GPU
  - Python `3.10.12` is available
  - gcc `11.4.0` is available
  - cmake `3.22.1` is available
  - `nvcc` is not currently available inside WSL
- Removed the native Windows `segman_env` as requested.
  - confirmed remaining Windows conda envs include `gpu_env` and `seg_gpu_env`, but no `segman_env`
- Added WSL helper scripts:
  - `scripts/wsl_setup_segman.sh`
  - `scripts/wsl_train_segman_b.sh`
- Validation:
  - `bash -n scripts/wsl_setup_segman.sh`
  - `bash -n scripts/wsl_train_segman_b.sh`
- Current blocker:
  - WSL needs CUDA toolkit / `nvcc` before building the SegMAN selective scan extension
  - once `nvcc` is installed, `wsl_setup_segman.sh` should install the isolated Linux `segman` conda env and prepare the GA2 SegMAN config

### 2026-04-24 (V10 WSL2 Environment Ready)

- Confirmed WSL2 already has CUDA Toolkit available under `/usr/local/cuda-12.4`; the initial `nvcc` failure was caused by PATH not including `/usr/local/cuda/bin`.
- Built the isolated WSL `segman` Conda environment:
  - Python `3.10`
  - Torch `2.1.2+cu121`
  - `mmcv-full 1.7.2`
  - official SegMAN `mmsegmentation 0.30.0`
  - `natten 0.17.3+torch210cu121`
  - SegMAN `selective_scan` CUDA extension
- Validation:
  - `torch.cuda.is_available()` returns `True`
  - imports pass for `torch`, `mmcv`, `mmseg`, `natten`, and `selective_scan_cuda_oflex`
- Downloaded the official SegMAN-B encoder checkpoint:
  - `external/SegMAN/pretrained/SegMAN_Encoder_b.pth.tar`
  - size: about `722 MB`
- Re-generated the GA2 SegMAN-B config:
  - train: `637`
  - validation: `112`
  - test: `750`
  - config: `external/SegMAN/segmentation/local_configs/segman/ga2/segman_b_ga2.py`
- Fixed reproducibility issues in the WSL setup script:
  - adds CUDA Toolkit paths before checking `nvcc`
  - installs NATTEN with `--trusted-host shi-labs.com`
  - builds `selective_scan` with `--no-build-isolation`
  - downloads only the SegMAN-B encoder checkpoint when missing

### 2026-04-24 (V10 SegMAN-B Training Started)

- Started SegMAN-B training in WSL and fixed two runtime incompatibilities:
  - MMCV 1.x scatter needed a PyTorch 2.1 device compatibility patch.
  - SegMAN `selective_scan` needed a local AMP patch to force the scan block to full precision and cast back afterward.
- Fixed validation/test preprocessing:
  - added `Pad(size_divisor=32)` because SegMAN decoder uses `pixel_unshuffle` and requires compatible feature dimensions.
- First successful validation:
  - best checkpoint: `best_mIoU_iter_5000.pth`
  - validation: `mIoU=70.75`, `mAcc=84.62`, `aAcc=93.64`
  - this is above the previous V8 local validation reference (`mIoU=68.97`), so the SegMAN direction is currently promising.
- The full run was resumed from `latest.pth` in WSL background mode after foreground command timeout.

### 2026-04-25 (V10 SegMAN-B Submission Results)

- Completed SegMAN-B training to `30000` iterations.
- Best local validation checkpoint:
  - checkpoint: `external/SegMAN/segmentation/outputs/ga2_segman_b/best_mIoU_iter_25000.pth`
  - local validation: `mIoU=77.20`, `mAcc=85.27`, `aAcc=95.47`
- Exported and submitted two V10 candidates:
  - `submission_exp_v10_segman_b_iter25000.csv`: Kaggle `0.42682`, segmentation from SegMAN-B and placeholder classification.
  - `submission_exp_v10_segman_b_iter25000_with_classification.csv`: Kaggle `0.85725`, SegMAN-B segmentation merged with teammate classification from `Image-Classification:output/submission_final.csv`.
- Interpretation:
  - SegMAN-B is a major segmentation improvement over V9 (`0.42682` vs `0.39064`) even before classification.
  - The merged submission validates the end-to-end strategy and is now the best public result.
- Disk cleanup:
  - retain only the best checkpoint and final/latest checkpoint for the SegMAN-B run.

### 2026-05-11 (Cityscapes External Generalization Planning)

- Decided to add a lightweight external generalization experiment for the final discussion.
- Chosen dataset: `Cityscapes`.
  - reason: it has pixel-level semantic labels and a street-scene domain that is clearly different from the GA2/PASCAL VOC training subset
  - tradeoff: it only overlaps with a subset of VOC classes, so the metric must be reported as overlap-class `mIoU`
- Rejected `COCO` as the primary plan for this specific report goal.
  - reason: COCO has better VOC category coverage but is less compelling as a real-world domain-shift test
- Planned evaluation scope:
  - no Cityscapes training or fine-tuning
  - evaluate the existing SegMAN-B checkpoint only
  - use overlapping classes such as `person`, `car`, `bus`, `bicycle`, `motorbike`, and `train`
  - generate quantitative per-class IoU plus qualitative visual examples for the report
- Updated persistent planning files:
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

### 2026-05-11 (Cityscapes Evaluation Script Implementation)

- Added Cityscapes generalization implementation files:
  - `src/ga2_seg/cityscapes_generalization.py`
  - `scripts/evaluate_cityscapes_generalization.py`
  - `tests/test_cityscapes_generalization.py`
- Added support for reading Cityscapes directly from an external drive via:
  - `--left-img-root`
  - `--gt-fine-root`
- Updated `README.md` with the external-drive command using the user's paths:
  - `G:\Datasets\leftImg8bit_trainvaltest`
  - `G:\Datasets\gtFine_trainvaltest`
- Verification:
  - `python .\scripts\evaluate_cityscapes_generalization.py --help` works in the base environment after moving the `mmseg` import into runtime inference.
  - `python -m pytest .\tests\test_cityscapes_generalization.py -q` passed with `3 passed`.
- Current blocker:
  - `leftImg8bit/val` exists on the external drive.
  - `gtFine/val` is missing; the current extracted labels contain only `gtFine/test`.
  - the script correctly raises `FileNotFoundError` and tells the user to extract the correct `gtFine_trainvaltest.zip` so `gtFine/val` exists.

### 2026-05-11 (Cityscapes External Evaluation Completed)

- Rechecked the external drive after re-extraction:
  - `G:\Datasets\leftImg8bit_trainvaltest\leftImg8bit` contains `train`, `val`, and `test`
  - `G:\Datasets\gtFine_trainvaltest\gtFine` contains `train`, `val`, and `test`
  - `gtFine/val` contains `500` `*_gtFine_labelIds.png` files
- WSL2 execution details:
  - mounted `G:` into WSL with `drvfs`
  - used the existing WSL `segman` Conda environment
  - used the current SegMAN-B best checkpoint `best_mIoU_iter_25000.pth`
- Smoke evaluation:
  - `--limit 1` completed successfully and wrote outputs under `outputs/cityscapes_generalization/smoke_segman_b_iter25000`
- Full evaluation:
  - evaluated all `500` Cityscapes validation samples
  - output directory: `outputs/cityscapes_generalization/segman_b_iter25000`
  - overlap-class `mIoU=0.3484`
- Per-class IoU:
  - `car=0.8249`
  - `person=0.5241`
  - `bus=0.5092`
  - `motorbike=0.1874`
  - `bicycle=0.0382`
  - `train=0.0066`
- Interpretation:
  - the result is strong evidence for real-world domain shift
  - large/common road objects transfer much better than small or rare overlap classes
  - the result should be reported as an external-domain robustness check, not as a full VOC-20 metric

### 2026-05-12 (EoMT-DINOv3 Candidate Planning)

- User questioned whether SegMAN is still the best direction and asked for a stronger model-family candidate.
- Researched current EoMT-DINOv3 evidence from official and secondary sources:
  - official EoMT repository
  - official EoMT DINOv3 model zoo
  - Hugging Face model card for `tue-mps/eomt-dinov3-ade-semantic-large-512`
  - Hugging Face Transformers EoMT-DINOv3 documentation
  - LightlyTrain semantic segmentation benchmark notes
- Key planning result:
  - EoMT-DINOv3-L reports ADE20K semantic `59.5 mIoU`, clearly above SegMAN-B's official `52.6 mIoU`
  - this makes it a valid next high-value experiment if another multi-hour training run is acceptable
- Key risk recorded:
  - DINOv3 EoMT official weights are delta weights relative to original DINOv3 weights
  - implementation must begin with a checkpoint loading and one-image inference smoke test
- Updated persistent planning files:
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
- Current status:
  - no environment changes made
  - no code cloned
  - no training started
  - waiting for user confirmation before executing the EoMT-DINOv3 plan

### 2026-05-12 (EoMT-DINOv3 V11 Execution)

- Reused the existing Windows `gpu_env` instead of creating a new environment.
  - Python: `3.12.3`
  - Torch: `2.5.1+cu121`
  - Transformers: `5.4.0`
  - CUDA visible: yes
  - GPU: RTX 4060 Laptop GPU
  - reported memory: about `8188 MiB`
- Added EoMT execution scripts:
  - `scripts/eomt_smoke_test.py`
  - `scripts/train_eomt_dinov3.py`
  - `scripts/predict_eomt_test.py`
- Smoke test result:
  - model: `tue-mps/eomt-dinov3-ade-semantic-large-512`
  - checkpoint loaded successfully from Hugging Face
  - one-image CUDA inference completed
  - peak inference memory: about `1684 MB`
- Training stage 1:
  - output: `outputs/checkpoints/eomt_dinov3_v11_head_mask`
  - trainable parameters: about `3.17M`
  - scope: classification head plus mask head
  - result: validation `mIoU=0.5715`
  - interpretation: insufficient versus SegMAN-B
- Training stage 2:
  - initialized from stage 1 best checkpoint
  - output: `outputs/checkpoints/eomt_dinov3_v11_unfreeze2`
  - trainable parameters: about `28.37M`
  - scope: classification head, mask head, and last 2 Transformer layers
  - max steps: `3000`
  - best step: `3000`
  - best validation `mIoU=0.7926`
  - comparison: above SegMAN-B local validation `mIoU=0.7720`
- Exported submissions:
  - segmentation-only: `outputs/submissions/submission_exp_v11_eomt_dinov3_unfreeze2.csv`
  - initial merged draft used classification rows from the V10 final submission
  - user provided the intended ConvNeXt-small classification file: `outputs/submissions/submission_classification_convnext_small_320.csv`
  - comparison showed `214` of `750` classification rows differ from the V10 classification source
  - regenerated the recommended merged candidate: `outputs/submissions/submission_exp_v11_eomt_dinov3_unfreeze2_with_convnext_small_320.csv`
- Cleanup:
  - removed EoMT smoke and intermediate checkpoint directories
  - retained only the final EoMT run's `best` and `last` directories
- Current recommendation:
  - upload `submission_exp_v11_eomt_dinov3_unfreeze2_with_convnext_small_320.csv` to Kaggle
  - do not treat the local improvement as final proof until public score is known

### 2026-05-12 (V11 Kaggle Drop Root Cause)

- Kaggle results exposed a V11 export bug:
  - `submission_exp_v11_eomt_dinov3_unfreeze2_with_classification.csv`: `0.62898`
  - `submission_exp_v11_eomt_dinov3_unfreeze2_with_convnext_small_320.csv`: `0.64761`
- The classification source was not the main issue.
  - ConvNeXt-small classification differs from the reused V10 classification in `214 / 750` rows.
  - Switching classification improved the score slightly, but the score remained far below V10.
- Root cause:
  - training and validation manually warped every image to `512x512` before the Hugging Face EoMT processor
  - the original V11 test export fed original-aspect-ratio images directly into the processor
  - this created a train/inference preprocessing mismatch
- Evidence:
  - EoMT checkpoint evaluated through the original broken inference path had foreground validation `mIoU≈0.3376`
  - using the training-matched inference path, manual `512x512` warp followed by nearest-neighbor resize back to the original size, restored foreground validation `mIoU≈0.7864`
- Fix:
  - updated `scripts/predict_eomt_test.py` to manually resize images to `512x512` before processor inference
  - resize the discrete predicted mask back to original image size with nearest-neighbor interpolation
  - regenerated predictions under `outputs/predictions/eomt_dinov3_v11_unfreeze2_fixed_preprocess`
  - regenerated segmentation-only CSV: `outputs/submissions/submission_exp_v11_eomt_dinov3_unfreeze2_fixed_preprocess.csv`
  - regenerated merged ConvNeXt-small CSV: `outputs/submissions/submission_exp_v11_eomt_dinov3_unfreeze2_fixed_preprocess_with_convnext_small_320.csv`
- Current recommendation:
  - submit `submission_exp_v11_eomt_dinov3_unfreeze2_fixed_preprocess_with_convnext_small_320.csv`
  - ignore the earlier V11 CSVs for model comparison because they used mismatched preprocessing

### 2026-05-12 (V11 Corrected Kaggle Result)

- Corrected V11 submission result:
  - file: `outputs/submissions/submission_exp_v11_eomt_dinov3_unfreeze2_fixed_preprocess_with_convnext_small_320.csv`
  - Kaggle public score: `0.88342`
- Interpretation:
  - the preprocessing fix resolved the apparent EoMT failure
  - V11 now exceeds the previous V10 final merged score `0.85725`
  - the gain is large enough to keep EoMT-DINOv3 as the current best segmentation direction
- Training-curve observation:
  - V11 unfreeze-2 stage validation mIoU rose from `0.5715` at step 1 to `0.7926` at step 3000
  - the curve did not clearly plateau by step 3000
  - a follow-up run could reasonably continue from the current best checkpoint with a lower learning rate and the same fixed preprocessing
- Generated training-curve artifacts:
  - `outputs/analysis/eomt_dinov3_v11_unfreeze2/training_curve.png`
  - `outputs/analysis/eomt_dinov3_v11_unfreeze2/training_curve_summary.md`

### 2026-05-12 (V12 EoMT Continue Training)

- Continued from V11 best checkpoint:
  - source: `outputs/checkpoints/eomt_dinov3_v11_unfreeze2/best`
  - output: `outputs/checkpoints/eomt_dinov3_v12_continue_unfreeze2_lr1e5`
  - image size: `512x512`
  - trainable scope: classification head, mask head, and last 2 Transformer layers
  - learning rate: `1e-5`
  - max steps: `3000`
  - validation interval: `500`
- Result:
  - best validation mIoU: `0.8002` at step `2000`
  - final validation mIoU: `0.7954` at step `3000`
  - comparison to V11: local validation improved from about `0.7926/0.7864` to `0.8002`
- Exported submissions:
  - segmentation-only: `outputs/submissions/submission_exp_v12_eomt_dinov3_continue_lr1e5_fixed_preprocess.csv`
  - merged with ConvNeXt-small classification: `outputs/submissions/submission_exp_v12_eomt_dinov3_continue_lr1e5_fixed_preprocess_with_convnext_small_320.csv`
- Training-curve artifacts:
  - `outputs/analysis/eomt_dinov3_v12_continue_unfreeze2_lr1e5/training_curve.png`
  - `outputs/analysis/eomt_dinov3_v12_continue_unfreeze2_lr1e5/training_curve_summary.md`
- Current recommendation:
  - upload the V12 merged candidate to Kaggle
  - expected gain is likely modest because local validation improved only slightly

### 2026-05-12 (V12 Kaggle Result and Curve Interpretation)

- V12 Kaggle public score:
  - file: `outputs/submissions/submission_exp_v12_eomt_dinov3_continue_lr1e5_fixed_preprocess_with_convnext_small_320.csv`
  - score: `0.88602`
- Comparison:
  - V10 final: `0.85725`
  - V11 corrected: `0.88342`
  - V12 continued: `0.88602`
  - V12 gain over V11: `+0.00260`
- Training-curve interpretation:
  - V12 starts from V11 best at `0.7926`
  - mIoU rises to `0.7994` at step `1000`
  - it peaks at `0.8002` at step `2000`
  - it slightly drops to `0.7954` by step `3000`
- Conclusion:
  - continued training still helps, but the return is now small
  - the best checkpoint is not the final checkpoint, so future runs should stop earlier or use lower learning rate
  - the next training-only attempt should be smaller and more conservative, not another long run at the same learning rate

### 2026-05-12 (V13 Smoke: Unfreeze-2 vs Unfreeze-4)

- Goal: compare configuration curves at the same validation steps rather than only checking whether a run beats the previous best.
- Shared setup:
  - initialization: `outputs/checkpoints/eomt_dinov3_v12_continue_unfreeze2_lr1e5/best`
  - image size: `512x512`
  - preprocessing: fixed EoMT preprocessing
  - learning rate: `5e-6`
  - max steps: `1000`
  - validation interval: `250`
  - batch size: `1`
  - gradient accumulation: `8`
- Unfreeze-2 result:
  - trainable params: about `28.4M`
  - step 1: `0.8002`
  - step 250: `0.7931`
  - step 500: `0.8011`
  - step 750: `0.8011`
  - step 1000: `0.8012`
- Unfreeze-4 result:
  - trainable params: about `53.6M`
  - step 1: `0.8002`
  - step 250: `0.7900`
  - step 500: `0.7998`
  - step 750: `0.8000`
  - step 1000: `0.8005`
- Interpretation:
  - unfreeze-4 is lower than unfreeze-2 at every matched validation step
  - unfreezing more layers increases trainable parameters substantially but does not improve the curve
  - the current evidence supports staying with last-2-layer unfreezing
- Exported V13 candidate:
  - segmentation-only: `outputs/submissions/submission_exp_v13_eomt_dinov3_unfreeze2_lr5e6_fixed_preprocess.csv`
  - merged with ConvNeXt-small classification: `outputs/submissions/submission_exp_v13_eomt_dinov3_unfreeze2_lr5e6_fixed_preprocess_with_convnext_small_320.csv`
- Recommendation:
  - V13 is safe to submit, but expected gain is very small because local mIoU only improved from `0.8002` to `0.8012`

### 2026-05-12 (V14 Continue V13 Until Early Stop)

- Continued from V13 unfreeze-2 best:
  - source: `outputs/checkpoints/eomt_dinov3_v13_smoke_unfreeze2_lr5e6/best`
  - output: `outputs/checkpoints/eomt_dinov3_v14_continue_unfreeze2_lr5e6_earlystop`
  - learning rate: `5e-6`
  - trainable scope: heads plus last 2 Transformer layers
  - validation interval: `250`
  - early-stop patience: `4`
- Result:
  - step 1: `0.8012`
  - step 500: `0.8017`
  - step 1000: `0.8019`
  - step 1250: `0.7961`
  - step 1500: `0.7962`
  - step 1750: `0.7961`
  - step 2000: `0.7949`
  - early stopped at step `2000`
  - best validation mIoU: `0.8019` at step `1000`
- Exported submissions:
  - segmentation-only: `outputs/submissions/submission_exp_v14_eomt_dinov3_unfreeze2_lr5e6_earlystop_fixed_preprocess.csv`
  - merged with ConvNeXt-small classification: `outputs/submissions/submission_exp_v14_eomt_dinov3_unfreeze2_lr5e6_earlystop_fixed_preprocess_with_convnext_small_320.csv`
- Interpretation:
  - local gain over V13 is only about `+0.0007`
  - this confirms low-learning-rate continuation is nearly saturated
  - after V14, further same-config continuation is not a good use of training budget

### 2026-05-12 (V14 Kaggle Result and Next-Step Analysis)

- V14 Kaggle public score:
  - file: `outputs/submissions/submission_exp_v14_eomt_dinov3_unfreeze2_lr5e6_earlystop_fixed_preprocess_with_convnext_small_320.csv`
  - score: `0.88830`
- Comparison:
  - V11 corrected: `0.88342`
  - V12: `0.88602`
  - V14: `0.88830`
  - V14 gain over V12: `+0.00228`
- Pixel-level submission change:
  - V12 vs V13: about `0.2617%` pixels changed
  - V13 vs V14: about `0.1896%` pixels changed
  - V12 vs V14: about `0.3607%` pixels changed
- Interpretation:
  - the model is still making high-value small corrections, but the total changed area is already tiny
  - repeated same-config continuation is now close to saturation
  - the next meaningful gain is more likely from inference-time robustness or resolution strategy than another low-learning-rate continuation
- Highest-value next candidates:
  - horizontal-flip test-time augmentation with probability/mask averaging if logits can be exported safely
  - controlled `640x640` smoke test for inference and short fine-tuning, only if GPU memory permits
  - checkpoint/submission-level ensemble across V12/V13/V14 only if validation/TTA supports it; naive majority voting may erase V14's small high-value corrections

### 2026-05-12 (V15 Layer-Wise LR + Cosine Scheduler)

- Motivation:
  - V12/V14 improvements showed fixed learning-rate fine-tuning still had room
  - the earlier unfreeze-4 comparison may have been unfair because all unfrozen layers used the same learning rate
- Script update:
  - added parameter groups to `scripts/train_eomt_dinov3.py`
  - supports `--backbone-lr`, `--backbone-lr-decay`, `--scheduler cosine`, `--warmup-steps`, and `--min-lr-ratio`
- V15 configuration:
  - initialization: `outputs/checkpoints/eomt_dinov3_v14_continue_unfreeze2_lr5e6_earlystop/best`
  - output: `outputs/checkpoints/eomt_dinov3_v15_layerlr_cosine_unfreeze4`
  - unfrozen layers: last `4`
  - head/mask learning rate: `5e-6`
  - deepest backbone layer lr: `2e-6`
  - layer-wise decay: `0.5`
  - resulting backbone layer learning rates: `2.5e-7`, `5e-7`, `1e-6`, `2e-6`
  - scheduler: warmup + cosine
  - warmup steps: `150`
  - min lr ratio: `0.1`
  - validation interval: `250`
  - early-stop patience: `5`
- Result:
  - step 1: `0.8019`
  - step 250: `0.7994`
  - step 500: `0.8020`
  - step 750: `0.8014`
  - step 1000: `0.8019`
  - step 1250: `0.8003`
  - step 1500: `0.7962`
  - step 1750: `0.7962`
  - early stopped at step `1750`
  - best validation mIoU: `0.8020` at step `500`
- Interpretation:
  - layer-wise LR fixed the obvious unfreeze-4 instability, but did not produce a meaningful local improvement over V14
  - unfreeze-4 remains only marginal under current data/validation conditions
- Exported submissions:
  - segmentation-only: `outputs/submissions/submission_exp_v15_eomt_dinov3_layerlr_cosine_unfreeze4_fixed_preprocess.csv`
  - merged with ConvNeXt-small classification: `outputs/submissions/submission_exp_v15_eomt_dinov3_layerlr_cosine_unfreeze4_fixed_preprocess_with_convnext_small_320.csv`

### 2026-05-12 (V16 TTA + Multi-Scale Inference)

- Goal: improve the V15 best checkpoint without retraining, using inference-time augmentation only. Ensembling across multiple checkpoints was explicitly out of scope so the final submission stays a single model.
- Implementation: `scripts/predict_eomt_tta.py` averages Mask2Former-style soft semantic score maps across input sides and an optional horizontal flip, with `model.grid_size` rewritten per forward so EoMT's hard-coded `32×32` reshape stays valid at non-512 inputs.
- Sanity check: single-scale `512`, no flip, full 112-image validation set reproduces `mIoU=0.8020`, exactly matching the V15 reported best.
- Ablation (V15 best checkpoint, 112 validation samples):
  - `512` + hflip = `0.8007` (diningtable collapses `0.794 -> 0.548`)
  - `{448, 512, 576}` no flip = `0.7984`
  - `{448, 512, 576}` + hflip = `0.8013`
  - `{480, 512, 544}` no flip = `0.7978`
  - `{496, 512}` no flip = `0.8027`
  - `{512, 528}` no flip = `0.8018`
  - `{496, 512, 528}` + hflip = `0.8025`
  - `{496, 512, 528}` no flip = `0.8030` ← chosen
- Final recipe: narrow multi-scale `{496, 512, 528}` with no horizontal flip. Reason: EoMT-DINOv3 was trained at fixed `512×512` and only a `±16` (one patch) window stays stable; horizontal flip helps small classes but tanks `diningtable` enough to regress the overall score.
- Generated artifacts:
  - test predictions: `outputs/predictions/eomt_dinov3_v16_tta_ms496_512_528_noflip` (`750` `.npy` files)
  - segmentation-only CSV: `outputs/submissions/submission_exp_v16_eomt_dinov3_tta_ms496_512_528_noflip.csv`
  - merged candidate: `outputs/submissions/submission_exp_v16_eomt_dinov3_tta_ms496_512_528_noflip_with_convnext_small_320.csv`
- Pixel-level change versus V15 best: `0.73%` mean, comparable in magnitude to earlier V12->V14 / V14->V15 deltas that produced clear Kaggle gains.
- Kaggle result:
  - file: `outputs/submissions/submission_exp_v16_eomt_dinov3_tta_ms496_512_528_noflip_with_convnext_small_320.csv`
  - public score: `0.88882`
  - versus V15: `0.88857 -> 0.88882`, `+0.00025`
- Interpretation:
  - V16 is the new best public result, but the gain is in the same `~+0.0003` band as V14->V15
  - the local-to-Kaggle transfer ratio is now stable at roughly `0.25x` (every `+0.001` local mIoU buys about `+0.00025` public)
  - to win another `+0.005` Kaggle requires roughly `+0.02` local mIoU, which the current single-checkpoint training plateau cannot deliver without a structural change

### 2026-05-12 (V17 Re-Split Merge-Val Fine-Tune)

- Goal: free up training data by lowering the locally-defined validation ratio. The original `0.15` split was chosen by this project, not by the assignment; the only true held-out split for grading is the Kaggle test set, so the 112 validation images were inflating the "no data leakage" cost beyond what the assignment requires.
- Cross-branch split analysis (computed against the actual `train_set.csv` `Id` order):
  - segmentation branch used `random.Random(42).shuffle()` with `val_ratio=0.15` -> 637 train / 112 val
  - image classification branch used `sklearn.train_test_split(random_state=42)` with `val_split=0.20` -> 599 train / 150 val
  - the two validation sets only overlap on `22` images; `90` segmentation-val images were already in the classification branch's training set, and `128` classification-val images were already in segmentation training
- Re-split:
  - new `val_ratio=0.05` -> 712 train / 37 val
  - script: `scripts/resplit_segman_ga2.py`
  - the new val is a strict prefix of the old `random.Random(42)` shuffle, so 75 images move `validation -> training` and 0 images move the other way
  - old `training.txt` / `validation.txt` backed up to `training_v15split.txt` / `validation_v15split.txt`
- Training:
  - initialization: `outputs/checkpoints/eomt_dinov3_v15_layerlr_cosine_unfreeze4/best`
  - output: `outputs/checkpoints/eomt_dinov3_v17_merge_val_lr1e5`
  - recipe: `lr=1e-5`, `unfreeze_last_layers=2`, constant LR (V12-style, the historically strongest single training jump)
  - `max_steps=2000`, `eval_every=100`, `early_stop_patience=8`, `batch_size=1`, `grad_accum=8`, fixed `512x512` warp preprocessing
  - result: best `val_mIoU=0.7923` at step `1000`, early-stopped at step `1800`
- Fair comparison on the new 37 val with `ms{496,512,528}` no-flip TTA:
  - V15 best (the checkpoint behind V16): `mIoU=0.7859`
  - V17 best: `mIoU=0.7930`
  - delta: `+0.0071`, roughly seven times the V14 -> V15 and V15 -> V16 deltas, so this is the strongest training-side gain since V12 -> V14
- Test predictions: `outputs/predictions/eomt_dinov3_v17_tta_ms496_512_528_noflip` (`750` `.npy` files)
- Segmentation-only CSV: `outputs/submissions/submission_exp_v17_merge_val_tta_ms496_512_528_noflip.csv`
- Merged candidate: `outputs/submissions/submission_exp_v17_merge_val_tta_ms496_512_528_noflip_with_convnext_small_320.csv`
- Pixel-level change vs V16 on the test set: `0.41%` mean, `0.12%` median, `43.5%` max
- Kaggle result:
  - file: `outputs/submissions/submission_exp_v17_merge_val_tta_ms496_512_528_noflip_with_convnext_small_320.csv`
  - public score: `0.89139`
  - versus V16: `0.88882 -> 0.89139`, `+0.00257`
- Interpretation:
  - V17 is the new best public result and the largest single-step Kaggle gain since V11 -> V12 (`+0.00260`)
  - the realised local-to-Kaggle transfer ratio is `0.00257 / 0.0071 ≈ 0.36x`, slightly above the recent `~0.25x` band, which is consistent with the gain being a real training signal rather than 37-val noise
  - this validates the strategic insight that the 112-image val was being held out at the cost of training data, with no assignment-defined reason to do so

### 2026-05-17 (V17 Cityscapes External Generalization Re-Run)

- Goal: refresh the Cityscapes external-domain evaluation (originally done in Phase 27 against V10 SegMAN-B) so that PDF Section 2.4's deployment-readiness discussion is grounded in the V17 EoMT-DINOv3 model that is actually shipped to Kaggle.
- Implementation:
  - new script `scripts/evaluate_cityscapes_generalization_eomt.py`
  - reuses helpers in `src/ga2_seg/cityscapes_generalization.py` (label mapping, IoU bookkeeping, visualisation)
  - plugs in the V16/V17 EoMT inference path: `ms{496,512,528}` no flip, fixed `512x512` warp preprocessing, dynamic `model.grid_size`, Mask2Former-style soft semantic averaging
  - runs on Windows in `gpu_env` directly against `G:\Datasets\leftImg8bit_trainvaltest` / `G:\Datasets\gtFine_trainvaltest`
- Configuration:
  - checkpoint: `outputs/checkpoints/eomt_dinov3_v17_merge_val_lr1e5/best`
  - split: `val`, samples: `500`
  - device: `cuda:0`, AMP enabled
- Result:
  - V17 6-class overlap mIoU: `0.4253`
  - V10 SegMAN-B reference (Phase 27): `0.3484`
  - delta: `+0.0769`
- After visual inspection (see follow-up below), the primary metric was redefined to exclude `train`:
  - V17 5-class transferable mIoU (`person, car, bus, motorbike, bicycle`): **`0.5088`**
  - V10 same metric: `0.4168`
  - delta: **`+0.0920`**
- Per-class IoU comparison:
  - `car`: `0.8249 -> 0.8728` (`+0.0479`)
  - `person`: `0.5241 -> 0.5685` (`+0.0444`)
  - `bus`: `0.5092 -> 0.5264` (`+0.0172`)
  - `motorbike`: `0.1874 -> 0.4389` (`+0.2515`, largest single-class jump)
  - `bicycle`: `0.0382 -> 0.1375` (`+0.0993`)
  - `train`: `0.0066 -> 0.0079` (still effectively zero)
- Interpretation:
  - V17 is materially better than V10 under domain shift, especially on the small / thin classes that were also the weakest on GA2 validation
  - `train` remains a hard failure case: street-scene Cityscapes trains do not match the VOC train concept the model learned
  - V17 narrows but does not close the lab-vs-domain-shift gap: GA2 Kaggle `0.89139` vs Cityscapes overlap mIoU `0.4253`
- Generated artifacts under `outputs/cityscapes_generalization/eomt_dinov3_v17_tta_ms496_512_528_noflip/`:
  - `metrics.csv`
  - `summary.md`
  - 12 qualitative comparison images in `visualizations/`
- Old V10 SegMAN-B Cityscapes results in `outputs/cityscapes_generalization/segman_b_iter25000/` were kept on disk so the V10 vs V17 comparison is reproducible from artifacts.

### 2026-05-17 (Cityscapes train-class visual-concept investigation)

- Trigger: V17's Cityscapes `train` IoU stayed at `0.0079` despite a `+0.077` overall mIoU jump. To check whether this was a real generalization failure or a label-overlap artefact, we extracted the largest `train`-area images from each dataset and built side-by-side panels.
- Script: `scripts/compare_train_class_voc_vs_cityscapes.py` ranks VOC training images and Cityscapes val images by their `train`-class pixel count, then renders 4 paired panels (image + red-overlay of the GT train mask). Output: `outputs/figures/train_class_voc_vs_cityscapes/`.
- What the panels show (direct visual evidence):
  - VOC `train` is dominated by full-frame intercity / steam locomotives photographed as the subject, with railway-specific backgrounds (tracks, overhead wires, smoke).
  - Cityscapes `train` is urban trams (Stadtbahn) captured incidentally from a moving vehicle on a street, typically small in frame, often partially occluded by cars or vans.
  - Out of 4 paired examples there is no visual overlap in vehicle type, scale, viewpoint, or background context.
- Consequence: VOC `train` and Cityscapes `train` are essentially different visual concepts that happen to share the same label. Including `train` in the overlap mIoU was deflating the headline number with a quantity that does not actually measure model generalization.
- Action:
  - Added `TRANSFERABLE_CLASS_NAMES` constant in `src/ga2_seg/cityscapes_generalization.py` (`person, car, bus, motorbike, bicycle`).
  - Added `scripts/recompute_cityscapes_summary.py` so existing `metrics.csv` files can be re-aggregated without paying for another full 500-image inference run.
  - Regenerated `summary.md` for both `eomt_dinov3_v17_tta_ms496_512_528_noflip/` and `segman_b_iter25000/`; each now reports the 5-class transferable mIoU as primary and the 6-class number as secondary.
- New headline figures for the report:
  - V10 SegMAN-B: 5-class transferable mIoU `0.4168` (was `0.3484` under 6-class).
  - V17 EoMT-DINOv3: 5-class transferable mIoU **`0.5088`** (was `0.4253` under 6-class).
  - Delta: **`+0.0920`** (was `+0.0769`).

### 2026-05-12 (Sliding-Window Negative Result, internal only)

- Goal: investigate whether sliding-window inference at the trained `512x512` resolution can recover the cost of the V11 full-image-warp preprocessing, particularly for small or thin classes like `bicycle` and `chair`.
- Script: `scripts/predict_eomt_sliding_window.py`. Resizes the image so its shorter side equals the window size while preserving aspect ratio, then slides square `512x512` windows with a configurable stride. Soft Mask2Former scores are averaged across overlapping windows in image space, then argmax-resized to the original size. Validation reports `mIoU` at both the `512x512` warped protocol (for direct comparison to V16) and at the native image size (Kaggle-relevant).
- Also updated `scripts/predict_eomt_tta.py` to report both `mIoU` protocols so the comparison is apples-to-apples.
- Validation results on the V15 best checkpoint:
  - V16 TTA `ms{496,512,528}` no flip: primary `mIoU=0.8030`, original-size `mIoU=0.8049`
  - Sliding window `w=512, s=256, no flip`: primary `mIoU=0.7941`, original-size `mIoU=0.7968`
  - Sliding window `w=512, s=128, no flip`: primary `mIoU=0.7952`, original-size `mIoU=0.7979`
- Per-class diagnosis at `s=256` vs V16:
  - `diningtable`: `0.779 -> 0.575`, collapse of `-0.20`
  - `train`: `0.882 -> 0.832`
  - `pottedplant`: `0.771 -> 0.758`
  - `dog`, `bus`, `sheep`, `sofa` improved by `+0.05` to `+0.13`
  - `bicycle` stayed near `0.21`, so the original "small-object rescue" hypothesis is not supported here
- Interpretation:
  - the V11 training pipeline warps every image to `512x512`, so the model has learned strong "image fits in one 512 square" priors
  - sliding-window inference shows the model un-warped local crops, which is out-of-distribution for the fine-tuned head
  - the failure pattern matches the earlier horizontal-flip TTA collapse on `diningtable`, suggesting both are symptoms of the same warp-induced layout prior
- Decision:
  - do not generate a sliding-window Kaggle submission; the local regression is large enough that confirming it on Kaggle would not be informative
  - V16 (`submission_exp_v16_eomt_dinov3_tta_ms496_512_528_noflip_with_convnext_small_320.csv`, public score `0.88882`) remains the recommended segmentation submission
  - the segmentation track is now considered closed for further inference-side experiments on this checkpoint; the next meaningful gain would require retraining with aspect-preserving augmentation or improving the classification half of the pipeline

### 2026-05-12 (V15 Kaggle Result)

- V15 Kaggle public score:
  - file: `outputs/submissions/submission_exp_v15_eomt_dinov3_layerlr_cosine_unfreeze4_fixed_preprocess_with_convnext_small_320.csv`
  - score: `0.88857`
- Comparison:
  - V14: `0.88830`
  - V15: `0.88857`
  - gain: `+0.00027`
- Interpretation:
  - layer-wise learning rate and cosine scheduling gave a real but tiny improvement
  - the result supports the user's concern that the previous fixed-learning-rate setup was not optimal
  - however, the marginal gain is now close to leaderboard noise / diminishing returns
  - training-side improvements are likely mostly exhausted unless a more structural change is made
- Recommendation:
  - use V15 as the current best checkpoint/submission
  - prioritize inference-side TTA or controlled resolution/sliding-window testing next
  - avoid more same-resolution fine-tuning unless there is a clearly different hypothesis
