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

## SegMAN V10 Preparation

SegMAN uses the official CVPR 2025 repository and a separate MMSegmentation 0.30 style environment. Keep it isolated from `seg_gpu_env`.

Clone the official repository outside tracked project code:

```powershell
git clone https://github.com/yunxiangfu2001/SegMAN.git .\external\SegMAN
```

Prepare the PNG dataset export and write a GA2 config into the cloned SegMAN tree:

```powershell
python .\scripts\prepare_segman_experiment.py --segman-root .\external\SegMAN --variant b --encoder-checkpoint .\external\SegMAN\pretrained\SegMAN_Encoder_b.pth.tar --batch-size 2 --workers 4
```

Then run training from the SegMAN environment:

```powershell
cd .\external\SegMAN\segmentation
python tools\train.py local_configs\segman\ga2\segman_b_ga2.py --work-dir outputs\ga2_segman_b
```

Notes:

- `data/segman_ga2/` and `external/` are ignored because they are generated or third-party content
- start with `SegMAN-B`; move to `SegMAN-L` only after the environment and smoke training are stable
- official SegMAN requires MMSegmentation 0.30, `natten`, and the VMamba selective scan extension
- on Windows, `segman_env` can install `torch 2.1.2 + cu121`, `mmcv-full 1.7.2`, and `mmseg 0.30.0`, but NATTEN has no official Windows wheel for this stack
- use [train_v10_segman_colab.ipynb](D:/BaiduNetdiskWorkspace/Leuven/8th/Computer%20Vision/assignment/Group2/Semantic%20segmentation/train_v10_segman_colab.ipynb) for the full Linux GPU training workflow

After training, export test predictions and a Kaggle CSV:

```powershell
python .\scripts\predict_segman_test.py --config .\external\SegMAN\segmentation\local_configs\segman\ga2\segman_b_ga2.py --checkpoint .\external\SegMAN\segmentation\outputs\ga2_segman_b\best_mIoU_iter_XXXXX.pth --output-dir .\outputs\predictions\exp_v10_segman_b_test
python .\scripts\export_submission.py --prediction-dir .\outputs\predictions\exp_v10_segman_b_test --output-path .\outputs\submissions\submission_exp_v10_segman_b.csv --classification-fill 0
```
