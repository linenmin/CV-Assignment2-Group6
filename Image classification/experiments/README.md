# 图像分类实验说明

每个子文件夹对应一个模型实验。文件夹里的 `train.py`、`evaluate.py`、`predict.py` 只是薄封装入口，实际会调用上一级目录里的共享训练、评估和预测代码，同时把不同实验的输出隔离开。

```
experiments/
  efficientnet_b3_320/
    train.py
    evaluate.py
    predict.py
  convnext_tiny_320/
    train.py
    evaluate.py
    predict.py
  convnext_small_320/
    train.py
    evaluate.py
    predict.py
  convnext_small_320_pad_sampler/
    train.py
    evaluate.py
    predict.py
  convnext_base_320/
    train.py
    evaluate.py
    predict.py
  convnext_large_320/
    train.py
    evaluate.py
    predict.py
  convnextv2_tiny_320/
    train.py
    evaluate.py
    predict.py
  convnextv2_base_320/
    train.py
    evaluate.py
    predict.py
  vit_b_16_224/
    train.py
    evaluate.py
    predict.py
  vit_l_16_224/
    train.py
    evaluate.py
    predict.py
  resnet50_224/
    train.py
    evaluate.py
    predict.py
```

输出会写入：

```
output/image_classification/<experiment>/
  checkpoints/
  metrics/
  figures/
    training_history.png
  predictions/
    test_probabilities_<experiment>.csv
    test_binary_predictions_<experiment>.csv
  submissions/
    submission_classification_<experiment>.csv
    submission_final_<experiment>__seg_<segmentation_csv_name>.csv
```

可以直接运行实验文件夹里的入口，也可以使用共享流水线：

```bash
python "Image classification/experiments/efficientnet_b3_320/train.py"
python "Image classification/run_pipeline.py" --experiment efficientnet_b3_320

python "Image classification/experiments/convnext_tiny_320/train.py"
python "Image classification/run_pipeline.py" --experiment convnext_tiny_320

python "Image classification/experiments/convnext_small_320/train.py"
python "Image classification/run_pipeline.py" --experiment convnext_small_320

python "Image classification/experiments/convnext_small_320_pad_sampler/train.py"
python "Image classification/run_pipeline.py" --experiment convnext_small_320_pad_sampler

python "Image classification/run_pipeline.py" --experiment convnext_base_320

python "Image classification/experiments/convnextv2_tiny_320/train.py"
python "Image classification/run_pipeline.py" --experiment convnextv2_tiny_320

python "Image classification/run_pipeline.py" --experiment convnextv2_base_320

python "Image classification/experiments/vit_b_16_224/train.py"
python "Image classification/run_pipeline.py" --experiment vit_b_16_224

python "Image classification/run_pipeline.py" --experiment vit_l_16_224
```

`convnext_small_320_pad_sampler` 保留 ConvNeXt-Small 骨干网络，但将图像预处理改为先方形填充再缩放；训练时对 `bottle`、`diningtable`、`pottedplant`、`sheep`、`sofa` 做温和加权采样；第二阶段使用 patience=4 的提前停止。

`convnextv2_tiny_320` 使用 Hugging Face Transformers 的 `facebook/convnextv2-tiny-1k-224` 作为 backbone，并继续使用本项目的多标签分类头和 AsymmetricLoss。它需要在 `biometrics` 环境安装 `transformers`。`convnextv2_base_320` 是更重的可选对照，建议 Tiny 有希望后再跑。

本地训练结束后会自动根据 `metrics/training_history.csv` 生成 `figures/training_history.png`。如需为已有实验补图，可以运行：

```bash
python "Image classification/08_plot_training_history.py" --all
```

`vit_b_16_224` 是当前推荐先跑的 Vision Transformer 基线。它使用 torchvision 的 ViT-B/16 ImageNet-1K 预训练权重，并保持 224 x 224 输入，避免一开始就引入 positional embedding 插值变量。`vit_l_16_224` 已注册为更重的可选实验，更适合放到 Colab Pro 或更强 GPU 上跑。

如果要在不重新训练的情况下集成已有分类模型：

```bash
python "Image classification/07_ensemble.py" --mode val-search --name ensemble_existing
python "Image classification/07_ensemble.py" --mode predict --name ensemble_existing
```

输出会写入 `output/image_classification/ensemble_existing/`，包括 ensemble 指标、概率文件、二值预测和分类提交 CSV。

当前 Kaggle 反馈显示 `ensemble_existing` 不如 `convnext_small_320`，因此它只作为分析/对照保留，不作为默认最终分类方案。

`convnext_large_320` 也已注册，但它在 8 GB GPU 上批大小只有 2，训练成本较高。除非明确要测试最大 torchvision ConvNeXt 骨干网络，否则建议先跑 Small 和 Base。

如果要在 Colab 或 Colab Pro 的更强 GPU 上训练，使用：

```bash
Image classification/colab_train_convnext.ipynb
```

该 notebook 是自包含版本，支持 ConvNeXt Tiny/Small/Base/Large，输出到 Google Drive，并包含 AMP 显存探测单元。

将分类 CSV 与分割 CSV 合并：

```bash
python "Image classification/merge_submission.py" --experiment convnext_tiny_320 --seg-csv output/submission_exp_v10_segman_b_iter25000.csv
```

如果要合并指定的分类提交或二值预测 CSV，使用 `--clf-csv`。合并后的 CSV 会保存到对应实验的 `submissions/` 文件夹，文件名会包含分类和分割两个来源。
