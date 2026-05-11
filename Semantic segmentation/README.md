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
- iterative segmentation experiments from `SegNeXt` to `SegFormer` and `SegMAN`
- training / validation / inference / submission tooling
- Kaggle-facing submission tracking

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

Current historical note:

- the strongest tracked result in this repository is no longer a `SegNeXt` run
- the best segmentation-only submission is `submission_exp_v10_segman_b_iter25000.csv`
- the best final merged submission is `submission_exp_v10_segman_b_iter25000_with_classification.csv`

Checkpoint retention policy:

- default is `best_last`
- after training, the script keeps only:
  - `best_*.pth`
  - the checkpoint referenced by `last_checkpoint`
- use `--checkpoint-retention all` if you explicitly want to keep every intermediate checkpoint

## Validation

```powershell
python .\scripts\validate.py --checkpoint .\outputs\logs\exp_v8_segformer_b5\best_mIoU_iter_11000.pth
```

## Training Analysis

Generate training curves, runtime diagnostics, and a markdown summary:

```powershell
python .\scripts\analyze_training_run.py --work-dir .\outputs\logs\exp_v8_segformer_b5 --submission-file submission_exp_v8_segformer_b5.csv
```

This writes:

- `analysis/training_curves.png`
- `analysis/runtime_diagnostics.png`
- `analysis/validation_metrics.csv`
- `analysis/summary.md`

## Test Prediction

Generate predicted segmentation masks for the full test set:

```powershell
python .\scripts\predict_test_segmentation.py --config .\configs\experiments\segformer_b5_512x512_adamw_poly_v8.py --checkpoint .\outputs\logs\exp_v8_segformer_b5\best_mIoU_iter_11000.pth --output-dir .\outputs\predictions\exp_v8_segformer_b5_test
```

## Submission Export

Export a Kaggle-ready `submission.csv` from predicted test masks:

```powershell
python .\scripts\export_submission.py --prediction-dir .\outputs\predictions\exp_v8_segformer_b5_test --output-path .\outputs\submissions\submission_exp_v8_segformer_b5.csv --classification-fill 0
```

Notes:

- `classification-fill 0` is only a placeholder workflow for segmentation-only iteration
- final leaderboard interpretation must consider whether classification is still placeholder output
- do not use the legacy generic `outputs/submissions/submission.csv`; prefer experiment-specific filenames

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

### WSL2 Route

WSL2 is preferred over native Windows for SegMAN because NATTEN and selective scan are Linux/CUDA extension builds.

From WSL:

```bash
cd "/mnt/d/BaiduNetdiskWorkspace/Leuven/8th/Computer Vision/assignment/Group2/Semantic segmentation"
bash scripts/wsl_setup_segman.sh
```

The setup script creates the isolated `segman` Conda environment, installs the Linux CUDA extensions, downloads the official SegMAN-B encoder checkpoint when missing, exports the GA2 PNG dataset, and writes the local SegMAN-B config.

Then run:

```bash
bash scripts/wsl_train_segman_b.sh
```

To run in WSL with Discord notification on success or failure:

```bash
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
nohup bash scripts/wsl_run_segman_b_with_notify.sh > outputs/logs/wsl_segman_b_notify_nohup.log 2>&1 &
```

Notes:

- the script prepends `/usr/local/cuda/bin` and `/usr/local/cuda/lib64`; if `nvcc` is still missing, install a Linux CUDA toolkit before running the setup script again
- the expected encoder checkpoint path is `external/SegMAN/pretrained/SegMAN_Encoder_b.pth.tar`
- `wsl_run_segman_b_with_notify.sh` does not store the webhook URL; it only reads `DISCORD_WEBHOOK_URL` from the WSL environment
- for faster long training, consider copying the repo and dataset into the WSL filesystem instead of reading many small files through `/mnt/d`

### Shared SegMAN Weights

The trained SegMAN weights are not stored in Git because the checkpoint files are too large for a normal GitHub repository.

Google Drive folder:

```text
https://drive.google.com/drive/u/1/folders/1orzpwFwAnjh5cne7Odm8prCkF8GfkLvP?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto
```

Required Drive permission:

- general access: `Anyone with the link`
- role: `Viewer`

Download the files from Drive and place them at these paths:

```text
external/SegMAN/pretrained/SegMAN_Encoder_b.pth.tar
external/SegMAN/segmentation/outputs/ga2_segman_b/best_mIoU_iter_25000.pth
```

The second file is the current best trained model used for the V10 submission.

After training, export test predictions and a Kaggle CSV:

```powershell
python .\scripts\predict_segman_test.py --config .\external\SegMAN\segmentation\local_configs\segman\ga2\segman_b_ga2.py --checkpoint .\external\SegMAN\segmentation\outputs\ga2_segman_b\best_mIoU_iter_25000.pth --output-dir .\outputs\predictions\exp_v10_segman_b_iter25000
python .\scripts\export_submission.py --prediction-dir .\outputs\predictions\exp_v10_segman_b_iter25000 --output-path .\outputs\submissions\submission_exp_v10_segman_b_iter25000.csv --classification-fill 0
```

Current best tracked outputs:

- segmentation-only: [submission_exp_v10_segman_b_iter25000.csv](D:/BaiduNetdiskWorkspace/Leuven/8th/Computer%20Vision/assignment/Group2/Semantic%20segmentation/outputs/submissions/submission_exp_v10_segman_b_iter25000.csv)
- merged final submission: [submission_exp_v10_segman_b_iter25000_with_classification.csv](D:/BaiduNetdiskWorkspace/Leuven/8th/Computer%20Vision/assignment/Group2/Semantic%20segmentation/outputs/submissions/submission_exp_v10_segman_b_iter25000_with_classification.csv)

## Cityscapes External Generalization

Cityscapes is planned as a no-training external domain-shift check for the final report. It will evaluate only VOC-overlap classes such as `person`, `car`, `bus`, `bicycle`, `motorbike`, and `train`.

Download Cityscapes manually from:

```text
https://www.cityscapes-dataset.com/downloads/
```

Required files:

```text
leftImg8bit_trainvaltest.zip
gtFine_trainvaltest.zip
```

Extract both archives into:

```text
Semantic segmentation/data/cityscapes/
```

Expected final layout:

```text
data/cityscapes/
  leftImg8bit/val/*/*_leftImg8bit.png
  gtFine/val/*/*_gtFine_labelIds.png
```

The Cityscapes files are ignored by Git. Keep only the placeholder [data/cityscapes/README.md](D:/BaiduNetdiskWorkspace/Leuven/8th/Computer%20Vision/assignment/Group2/Semantic%20segmentation/data/cityscapes/README.md) tracked.
