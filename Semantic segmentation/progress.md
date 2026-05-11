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
