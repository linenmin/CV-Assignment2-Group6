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
- iterative segmentation experiments from `SegNeXt` to `SegFormer`, `SegMAN`, and `EoMT-DINOv3`
- training / validation / inference / submission tooling
- Kaggle-facing submission tracking

## Current Best (V17)

The current best segmentation model is **V17**, EoMT-DINOv3 fine-tuned on a 712/37 re-split of the GA2 training set, with `ms{496, 512, 528}` no-flip multi-scale TTA at inference.

- checkpoint: `outputs/checkpoints/eomt_dinov3_v17_merge_val_lr1e5/best/` (about `1.2 GB`)
- local fair-comparison validation (new 37 val, V16 TTA recipe): `mIoU=0.7930` versus V15 `0.7859` (`+0.0071`)
- Kaggle public score (segmentation + ConvNeXt-small classification): **`0.89139`** (V16 baseline `0.88882`, `+0.00257`)
- recommended submission: `outputs/submissions/submission_exp_v17_merge_val_tta_ms496_512_528_noflip_with_convnext_small_320.csv`

## Shared Weights (Current)

Checkpoint files are larger than what GitHub will accept, so they are shared via Google Drive instead of Git. The current production share point is V17.

Google Drive folder:

```text
https://drive.google.com/drive/u/1/folders/1ImiIyTvxtMVIlrptKXfl3w3qF5CBdPCG
```

Inside the folder, the V17 weights live under the `eomt_dinov3_v17_checkpoint/` subdirectory.

Required Drive permission:

- general access: `Anyone with the link`
- role: `Viewer`

Download steps:

1. Open the Drive folder above and enter `eomt_dinov3_v17_checkpoint/`
2. Download all four files inside it (`config.json`, `metrics.json`, `model.safetensors`, `preprocessor_config.json`)
3. Place them under the following local path so the inference scripts find them without any `--checkpoint` change:

```text
outputs/checkpoints/eomt_dinov3_v17_merge_val_lr1e5/best/
  config.json
  metrics.json
  model.safetensors          (about 1.2 GB)
  preprocessor_config.json
```

Optional, for ablations against V16 / V15:

```text
outputs/checkpoints/eomt_dinov3_v15_layerlr_cosine_unfreeze4/best/
```

Notes:

- only the `best/` subdirectory is required to reproduce the V17 submission; `last/` and intermediate checkpoints are not needed for inference and are not redistributed
- the ConvNeXt-small classification CSV that V17 is merged with lives at `outputs/submissions/submission_classification_convnext_small_320.csv` and is committed to Git (small file)
- the SegMAN V10 weights described later in this README are no longer the current best model and are not redistributed any more; if you need them, regenerate from the WSL scripts

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

Historical note on prior baselines:

- the strongest tracked result in this repository is no longer a `SegNeXt` or `SegMAN` run
- the V10 SegMAN-B segmentation-only submission was `0.42682`, and merged with classification was `0.85725`
- V11..V16 EoMT-DINOv3 fine-tunes pushed this to `0.88857` -> `0.88882`
- V17 (merge-val fine-tune on `712/37` split) is currently `0.89139` and is the recommended baseline

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

### Shared SegMAN Weights (Historical, V10 only)

The SegMAN V10 checkpoint is no longer the current best model; V17 EoMT-DINOv3 supersedes it. The SegMAN files are kept only so the V10 ablation can still be reproduced.

```text
external/SegMAN/pretrained/SegMAN_Encoder_b.pth.tar
external/SegMAN/segmentation/outputs/ga2_segman_b/best_mIoU_iter_25000.pth
```

These files are not redistributed any more. If you need them, regenerate from the WSL training scripts in this README or contact the original maintainer.

The remainder of this section is preserved as historical V10 reproduction notes only.

```powershell
python .\scripts\predict_segman_test.py --config .\external\SegMAN\segmentation\local_configs\segman\ga2\segman_b_ga2.py --checkpoint .\external\SegMAN\segmentation\outputs\ga2_segman_b\best_mIoU_iter_25000.pth --output-dir .\outputs\predictions\exp_v10_segman_b_iter25000
python .\scripts\export_submission.py --prediction-dir .\outputs\predictions\exp_v10_segman_b_iter25000 --output-path .\outputs\submissions\submission_exp_v10_segman_b_iter25000.csv --classification-fill 0
```

Current best tracked outputs:

- segmentation-only: [submission_exp_v10_segman_b_iter25000.csv](D:/BaiduNetdiskWorkspace/Leuven/8th/Computer%20Vision/assignment/Group2/Semantic%20segmentation/outputs/submissions/submission_exp_v10_segman_b_iter25000.csv)
- merged final submission: [submission_exp_v10_segman_b_iter25000_with_classification.csv](D:/BaiduNetdiskWorkspace/Leuven/8th/Computer%20Vision/assignment/Group2/Semantic%20segmentation/outputs/submissions/submission_exp_v10_segman_b_iter25000_with_classification.csv)

## EoMT-DINOv3 V11 Candidate

EoMT-DINOv3 uses the Hugging Face Transformers implementation instead of the old MMSegmentation 0.30 SegMAN stack. The current local route uses the existing Windows `gpu_env`.

Run the smoke test:

```powershell
$env:PYTHONPATH="src"
conda run -n gpu_env python .\scripts\eomt_smoke_test.py
```

Train the V11 two-stage candidate:

```powershell
$env:PYTHONPATH="src"
conda run -n gpu_env python .\scripts\train_eomt_dinov3.py --output-dir .\outputs\checkpoints\eomt_dinov3_v11_head_mask --max-steps 6000 --eval-every 500 --early-stop-patience 6 --batch-size 1 --grad-accum 8 --lr 1e-4 --unfreeze-last-layers 0
conda run -n gpu_env python .\scripts\train_eomt_dinov3.py --model-id .\outputs\checkpoints\eomt_dinov3_v11_head_mask\best --output-dir .\outputs\checkpoints\eomt_dinov3_v11_unfreeze2 --max-steps 3000 --eval-every 500 --early-stop-patience 4 --batch-size 1 --grad-accum 8 --lr 2e-5 --unfreeze-last-layers 2
```

Export the V11 test predictions and CSV:

```powershell
$env:PYTHONPATH="src"
conda run -n gpu_env python .\scripts\predict_eomt_test.py --checkpoint .\outputs\checkpoints\eomt_dinov3_v11_unfreeze2\best --output-dir .\outputs\predictions\eomt_dinov3_v11_unfreeze2
conda run -n gpu_env python .\scripts\export_submission.py --prediction-dir .\outputs\predictions\eomt_dinov3_v11_unfreeze2 --output-path .\outputs\submissions\submission_exp_v11_eomt_dinov3_unfreeze2.csv
```

Current V11 result:

- model: `tue-mps/eomt-dinov3-ade-semantic-large-512`
- strategy: train head/mask first, then unfreeze the last 2 Transformer layers
- local validation: `mIoU=0.7926`
- comparison: SegMAN-B V10 local validation was `mIoU=0.7720`
- segmentation-only output: [submission_exp_v11_eomt_dinov3_unfreeze2.csv](D:/BaiduNetdiskWorkspace/Leuven/8th/Computer%20Vision/assignment/Group2/Semantic%20segmentation/outputs/submissions/submission_exp_v11_eomt_dinov3_unfreeze2.csv)
- corrected merged candidate output: [submission_exp_v11_eomt_dinov3_unfreeze2_fixed_preprocess_with_convnext_small_320.csv](D:/BaiduNetdiskWorkspace/Leuven/8th/Computer%20Vision/assignment/Group2/Semantic%20segmentation/outputs/submissions/submission_exp_v11_eomt_dinov3_unfreeze2_fixed_preprocess_with_convnext_small_320.csv)
- classification source: [submission_classification_convnext_small_320.csv](D:/BaiduNetdiskWorkspace/Leuven/8th/Computer%20Vision/assignment/Group2/Semantic%20segmentation/outputs/submissions/submission_classification_convnext_small_320.csv)
- Kaggle public score: `0.88342`
- training curve: [training_curve.png](D:/BaiduNetdiskWorkspace/Leuven/8th/Computer%20Vision/assignment/Group2/Semantic%20segmentation/outputs/analysis/eomt_dinov3_v11_unfreeze2/training_curve.png)
- training summary: [training_curve_summary.md](D:/BaiduNetdiskWorkspace/Leuven/8th/Computer%20Vision/assignment/Group2/Semantic%20segmentation/outputs/analysis/eomt_dinov3_v11_unfreeze2/training_curve_summary.md)

Important V11 preprocessing note:

- training warped images to `512x512` before the EoMT processor
- inference must do the same warp before the processor
- after prediction, resize the discrete semantic mask back to the original image size with nearest-neighbor interpolation
- earlier V11 CSVs without `fixed_preprocess` used mismatched preprocessing and should not be used for model comparison

Keep only `best` and `last` under `outputs/checkpoints/eomt_dinov3_v11_unfreeze2/`; smoke and intermediate EoMT checkpoints should be removed after export.

### V12 Continue Training

V12 continues from V11 best at the same `512x512` resolution:

```powershell
$env:PYTHONPATH="src"
conda run -n gpu_env python .\scripts\train_eomt_dinov3.py --model-id .\outputs\checkpoints\eomt_dinov3_v11_unfreeze2\best --output-dir .\outputs\checkpoints\eomt_dinov3_v12_continue_unfreeze2_lr1e5 --max-steps 3000 --eval-every 500 --early-stop-patience 4 --batch-size 1 --grad-accum 8 --lr 1e-5 --unfreeze-last-layers 2
conda run -n gpu_env python .\scripts\predict_eomt_test.py --checkpoint .\outputs\checkpoints\eomt_dinov3_v12_continue_unfreeze2_lr1e5\best --output-dir .\outputs\predictions\eomt_dinov3_v12_continue_unfreeze2_lr1e5_fixed_preprocess
conda run -n gpu_env python .\scripts\export_submission.py --prediction-dir .\outputs\predictions\eomt_dinov3_v12_continue_unfreeze2_lr1e5_fixed_preprocess --output-path .\outputs\submissions\submission_exp_v12_eomt_dinov3_continue_lr1e5_fixed_preprocess.csv
```

Current V12 result:

- local validation: `mIoU=0.8002` at step `2000`
- merged candidate output: [submission_exp_v12_eomt_dinov3_continue_lr1e5_fixed_preprocess_with_convnext_small_320.csv](D:/BaiduNetdiskWorkspace/Leuven/8th/Computer%20Vision/assignment/Group2/Semantic%20segmentation/outputs/submissions/submission_exp_v12_eomt_dinov3_continue_lr1e5_fixed_preprocess_with_convnext_small_320.csv)
- training curve: [training_curve.png](D:/BaiduNetdiskWorkspace/Leuven/8th/Computer%20Vision/assignment/Group2/Semantic%20segmentation/outputs/analysis/eomt_dinov3_v12_continue_unfreeze2_lr1e5/training_curve.png)
- Kaggle public score: `0.88602`

### V15 Layer-Wise LR Candidate

V15 is the current best training-side public result:

- checkpoint: `outputs/checkpoints/eomt_dinov3_v15_layerlr_cosine_unfreeze4/best`
- training change: last-4-layer unfreezing with layer-wise learning rates, warmup, and cosine decay
- local validation: `mIoU=0.8020`
- merged candidate output: [submission_exp_v15_eomt_dinov3_layerlr_cosine_unfreeze4_fixed_preprocess_with_convnext_small_320.csv](D:/BaiduNetdiskWorkspace/Leuven/8th/Computer%20Vision/assignment/Group2/Semantic%20segmentation/outputs/submissions/submission_exp_v15_eomt_dinov3_layerlr_cosine_unfreeze4_fixed_preprocess_with_convnext_small_320.csv)
- Kaggle public score: `0.88857`

### V16 TTA + Multi-Scale Inference

V16 reuses the V15 best checkpoint and only changes the inference path. It stays a single model (no ensembling), which is the configuration that fits the course report.

```powershell
$env:PYTHONPATH="src"
conda run -n gpu_env python .\scripts\predict_eomt_tta.py --checkpoint .\outputs\checkpoints\eomt_dinov3_v15_layerlr_cosine_unfreeze4\best --split test --output-dir .\outputs\predictions\eomt_dinov3_v16_tta_ms496_512_528_noflip --scales 496 512 528 --no-hflip
conda run -n gpu_env python .\scripts\export_submission.py --prediction-dir .\outputs\predictions\eomt_dinov3_v16_tta_ms496_512_528_noflip --output-path .\outputs\submissions\submission_exp_v16_eomt_dinov3_tta_ms496_512_528_noflip.csv --classification-fill 0
```

Then merge classification rows with the ConvNeXt-small classification CSV to produce the leaderboard submission.

Recipe rationale:

- multi-scale averaging at `{496, 512, 528}` improved validation `mIoU` from `0.8020` to `0.8030`
- wider scale windows or horizontal flip both regressed locally, because EoMT-DINOv3 has a hard-coded `32×32` token grid and certain classes (e.g. `diningtable`) have strong left-right context priors that flip TTA breaks

Current V16 result:

- local validation: `mIoU=0.8030` (primary 512x512 protocol) / `0.8049` (original image size)
- merged candidate output: [submission_exp_v16_eomt_dinov3_tta_ms496_512_528_noflip_with_convnext_small_320.csv](D:/BaiduNetdiskWorkspace/Leuven/8th/Computer%20Vision/assignment/Group2/Semantic%20segmentation/outputs/submissions/submission_exp_v16_eomt_dinov3_tta_ms496_512_528_noflip_with_convnext_small_320.csv)
- Kaggle public score: `0.88882` (current best, `+0.00025` over V15)

### V17 Merge-Val Fine-Tune (Smaller Validation Ratio)

Because the segmentation `train/val` split is project-defined (not assignment-defined), V17 lowers the local validation ratio to free up training data for a final fine-tune. The old `0.15` split is preserved in `training_v15split.txt` / `validation_v15split.txt`; the new `0.05` split moves 75 images from validation into training.

```powershell
$env:PYTHONPATH="src"
conda run -n gpu_env python .\scripts\resplit_segman_ga2.py --train-csv .\..\kul-computer-vision-ga-2-2026\train\train_set.csv --val-ratio 0.05
conda run -n gpu_env python .\scripts\train_eomt_dinov3.py --model-id .\outputs\checkpoints\eomt_dinov3_v15_layerlr_cosine_unfreeze4\best --output-dir .\outputs\checkpoints\eomt_dinov3_v17_merge_val_lr1e5 --max-steps 2000 --eval-every 100 --early-stop-patience 8 --batch-size 1 --grad-accum 8 --lr 1e-5 --unfreeze-last-layers 2
conda run -n gpu_env python .\scripts\predict_eomt_tta.py --checkpoint .\outputs\checkpoints\eomt_dinov3_v17_merge_val_lr1e5\best --split test --output-dir .\outputs\predictions\eomt_dinov3_v17_tta_ms496_512_528_noflip --scales 496 512 528 --no-hflip
```

Fair comparison on the new 37 val with `ms{496,512,528}` no-flip TTA:

- V15 best (V16 base): `mIoU=0.7859`
- V17 best: `mIoU=0.7930` (`+0.0071`, the strongest training-side delta since V12 -> V14)

Current V17 artifacts:

- checkpoint: `outputs/checkpoints/eomt_dinov3_v17_merge_val_lr1e5/best`
- merged candidate: [submission_exp_v17_merge_val_tta_ms496_512_528_noflip_with_convnext_small_320.csv](D:/BaiduNetdiskWorkspace/Leuven/8th/Computer%20Vision/assignment/Group2/Semantic%20segmentation/outputs/submissions/submission_exp_v17_merge_val_tta_ms496_512_528_noflip_with_convnext_small_320.csv)
- Kaggle public score: `0.89139` (current best, `+0.00257` over V16, largest single-step Kaggle gain since V11 -> V12)

### Sliding-Window Inference (Negative Result, internal only)

Sliding-window inference was attempted to avoid the perspective distortion that the V11 fine-tuning pipeline introduces by warping every image to `512x512`. Each window is fed at exactly `512x512`, so EoMT's hard-coded `32x32` token grid is satisfied without monkey-patching.

```powershell
$env:PYTHONPATH="src"
conda run -n gpu_env python .\scripts\predict_eomt_sliding_window.py --checkpoint .\outputs\checkpoints\eomt_dinov3_v15_layerlr_cosine_unfreeze4\best --split validation --output-dir .\outputs\predictions\sw_v15_w512_s256_noflip --window-size 512 --stride 256
```

Validation results (V15 best checkpoint):

- V16 TTA reference: primary `mIoU=0.8030`, original-size `mIoU=0.8049`
- sliding window `w=512, s=256`: primary `mIoU=0.7941`, original-size `mIoU=0.7968`
- sliding window `w=512, s=128`: primary `mIoU=0.7952`, original-size `mIoU=0.7979`

The regression is dominated by `diningtable` collapsing from `0.779` to `0.575`, matching the failure mode of horizontal-flip TTA. The interpretation is that the fine-tuned model has learned strong "image fits one 512 square" priors from the V11 warp preprocessing, and aspect-preserving sliding-window crops are out-of-distribution for the segmentation head.

No Kaggle submission was generated. V16 remains the recommended segmentation submission.

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

If the archives are kept on an external drive, pass the extracted roots directly:

```powershell
python .\scripts\evaluate_cityscapes_generalization.py --left-img-root "G:\Datasets\leftImg8bit_trainvaltest" --gt-fine-root "G:\Datasets\gtFine_trainvaltest" --split val --limit 20
```

The full SegMAN-B evaluation command is:

```powershell
python .\scripts\evaluate_cityscapes_generalization.py --left-img-root "G:\Datasets\leftImg8bit_trainvaltest" --gt-fine-root "G:\Datasets\gtFine_trainvaltest" --split val --checkpoint .\external\SegMAN\segmentation\outputs\ga2_segman_b\best_mIoU_iter_25000.pth --output-dir .\outputs\cityscapes_generalization\segman_b_iter25000
```

Expected outputs:

```text
outputs/cityscapes_generalization/segman_b_iter25000/metrics.csv
outputs/cityscapes_generalization/segman_b_iter25000/summary.md
outputs/cityscapes_generalization/segman_b_iter25000/visualizations/
```

Current SegMAN-B result on Cityscapes `val` (Phase 27 reference, V10 model):

- samples: `500`
- metric: overlap-class `mIoU`
- score: `0.3484`
- strongest classes: `car=0.8249`, `person=0.5241`, `bus=0.5092`
- weakest classes: `train=0.0066`, `bicycle=0.0382`, `motorbike=0.1874`

### EoMT-DINOv3 V17 Cityscapes (current model, primary reference)

Re-run with the V17 checkpoint and the V16 multi-scale TTA inference path:

```powershell
$env:PYTHONPATH="src"
conda run -n gpu_env python .\scripts\evaluate_cityscapes_generalization_eomt.py --left-img-root "G:\Datasets\leftImg8bit_trainvaltest" --gt-fine-root "G:\Datasets\gtFine_trainvaltest" --output-dir .\outputs\cityscapes_generalization\eomt_dinov3_v17_tta_ms496_512_528_noflip
```

Output directory: `outputs/cityscapes_generalization/eomt_dinov3_v17_tta_ms496_512_528_noflip/`

#### Primary metric: 5-class transferable mIoU (excludes `train`)

`train` is excluded from the headline because VOC trains (full-frame intercity / steam locomotives) and Cityscapes 'trains' (urban trams in street scenes) are essentially different visual concepts that happen to share the same English word. See `outputs/figures/train_class_voc_vs_cityscapes/` for direct visual evidence and `scripts/compare_train_class_voc_vs_cityscapes.py` for the panel-generator.

- samples: `500`
- transferable classes: `person, car, bus, motorbike, bicycle`
- score: **`0.5088`** (V10 SegMAN-B `0.4168`, delta `+0.0920`)
- per-class IoU (V10 -> V17):
  - `car`: `0.8249 -> 0.8728`
  - `person`: `0.5241 -> 0.5685`
  - `bus`: `0.5092 -> 0.5264`
  - `motorbike`: `0.1874 -> 0.4389` (largest single-class jump)
  - `bicycle`: `0.0382 -> 0.1375`

#### Secondary metric: 6-class overlap mIoU (includes `train`, backward comparison)

- V17: `0.4253`, V10: `0.3484`, delta `+0.0769`
- per-class `train` IoU: `0.0079` (V10 `0.0066`)
- kept only so the original Phase 27 reporting can be reproduced; not recommended as the headline.

To re-aggregate from an existing `metrics.csv` (no re-inference needed):

```powershell
conda run -n gpu_env python .\scripts\recompute_cityscapes_summary.py --metrics-csv .\outputs\cityscapes_generalization\eomt_dinov3_v17_tta_ms496_512_528_noflip\metrics.csv --label "EoMT-DINOv3 V17"
```

Use V17 as the primary external-domain reference in Section 2.4. Keep V10 only as a model-family comparison: changing model family from SegMAN-B to EoMT-DINOv3 also improves cross-domain transfer, especially on small / thin classes. The absolute V17 number (`0.5088`) remains far below the GA2 Kaggle score (`0.89139`), so the deployment-readiness conclusion still stands: high closed-benchmark accuracy is not a proxy for cross-domain robustness.
