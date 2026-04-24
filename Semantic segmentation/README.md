# Semantic Segmentation

Maintainable semantic segmentation workspace for GA2.

## Environment

Use:

```powershell
conda activate seg_gpu_env
```

Core notes:

- keep `numpy<2` to avoid the Windows `torch/mmcv` ABI breakage
- keep `setuptools<81` because `torch 2.1` still imports `pkg_resources`
- install the official OpenMMLab `mmcv` wheel for your CUDA/Torch combo before training
- use `D:\Anaconda3\envs\seg_gpu_env\python.exe` directly on Windows if `conda run` hits encoding issues

## Current scope

- data analysis and preprocessing visualization
- MMSegmentation-based training configs
- train / validate scripts

## Training

Default training entrypoint:

```powershell
conda activate seg_gpu_env
cd "D:\BaiduNetdiskWorkspace\Leuven\8th\Computer Vision\assignment\Group2\Semantic segmentation"
python .\scripts\train.py
```

Useful options:

```powershell
python .\scripts\train.py --config .\configs\experiments\segnext_s_512x512_adamw_poly_v3_region_rebalance.py
python .\scripts\train.py --max-iters 3000 --val-interval 500
python .\scripts\train.py --checkpoint-retention all
```

Checkpoint retention policy:

- default is `best_last`
- after training, the script keeps only:
  - `best_*.pth`
  - the checkpoint referenced by `last_checkpoint`
- use `--checkpoint-retention all` if you explicitly want to keep every intermediate checkpoint

## Validation

```powershell
python .\scripts\validate.py --checkpoint .\outputs\logs\exp_v1_ade20k_main\best_mIoU_iter_8000.pth
```

## Training Analysis

Generate training curves, runtime diagnostics, and a markdown summary:

```powershell
python .\scripts\analyze_training_run.py --work-dir .\outputs\logs\exp_v3_region_rebalance --submission-file submission_exp_v3_region_rebalance.csv
```

This writes:

- `analysis/training_curves.png`
- `analysis/runtime_diagnostics.png`
- `analysis/validation_metrics.csv`
- `analysis/summary.md`

## Test Prediction

Generate predicted segmentation masks for the full test set:

```powershell
python .\scripts\predict_test_segmentation.py --config .\configs\experiments\segnext_s_512x512_adamw_poly_v3_region_rebalance.py --checkpoint .\outputs\logs\exp_v3_region_rebalance\best_mIoU_iter_5000.pth --output-dir .\outputs\predictions\exp_v3_region_rebalance_test
```

## Submission Export

Export a Kaggle-ready `submission.csv` from predicted test masks:

```powershell
python .\scripts\export_submission.py --prediction-dir .\outputs\predictions\exp_v3_region_rebalance_test --output-path .\outputs\submissions\submission_exp_v3_region_rebalance.csv --classification-fill 0
```

Notes:

- `classification-fill 0` is only a placeholder workflow for segmentation-only iteration
- final leaderboard interpretation must consider whether classification is still placeholder output

## Submission Ensemble

Build a hard-vote ensemble directly from existing Kaggle submission CSV files:

```powershell
python .\scripts\ensemble_submissions.py --submission .\outputs\submissions\submission_exp_v6_segformer_b2.csv --submission .\outputs\submissions\submission_exp_v7_segformer_b3.csv --submission .\outputs\submissions\submission_exp_v8_segformer_b5.csv --output-path .\outputs\submissions\submission_exp_v9_segformer_vote_v6_v7_v8.csv --primary-index 2
```

Notes:

- pass submissions from weaker to stronger models when using `--primary-index 2` for the V6/V7/V8 setup
- the primary submission breaks voting ties, so the V9 candidate defaults ties back to V8
- this path requires no model checkpoint and is useful when only submission CSVs are available
