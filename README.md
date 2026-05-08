# CV 作业 2 — 第 6 组

KUL H02A5a 计算机视觉，组队作业 2。

本仓库为第 6 组的解决方案代码。第 2.1 节为基于 PASCAL VOC 2009 的多标签图像分类任务。

## 任务说明

数据集包含 750 张训练图像和 750 张测试图像，以 `.npy` 数组形式存储。对每张图像，分类器需预测 20 个 PASCAL VOC 类别中哪些类别出现在图像中。

类别：

`aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train,
tvmonitor`

## 实验结果

val mAP 来自各实验目录下的 `evaluation_summary.csv`（使用 `best_model.pth` 评估）。
Kaggle 显示分数需×2 得到真实分类 Dice（提交文件含 1500 行：750 行分类 + 750 行分割占位）。

| 版本 | 实验文件夹 | Backbone | 损失函数 | 输入 | val mAP | Kaggle 显示 | 分类 Dice（×2） |
|------|-----------|----------|---------|------|---------|------------|----------------|
| v1.0 | *(重构前)* | ResNet-50 | NegativeSmoothBCE | 224 | 0.801 | 0.381 | 0.762 |
| v1.1 | `resnet50_224` | ResNet-50 | AsymmetricLoss | 224 + TTA | 0.818 | 0.392 | 0.783 |
| v2 | `efficientnet_b3_320` | EfficientNet-B3 | AsymmetricLoss | 320 + TTA | 0.860 | 0.428 | 0.856 |
| v3 | `convnext_tiny_320` | ConvNeXt-Tiny | AsymmetricLoss | 320 + TTA | **0.893** | **0.437** | **0.873** |

**当前最佳模型：** `convnext_tiny_320`，分类 Dice = **0.873**。

**关于 Kaggle 显示分数：** 完整提交包含分类行和分割行，Kaggle 在 1500 行上统一计算一个 Dice 值；
仅含分类行的提交因缺少分割行得 0，显示分数减半。×2 后的数值才是真实的分类性能。

## 环境配置

所有脚本需要 `biometrics` conda 环境：

```bash
conda activate biometrics
# PyTorch 2.11.0+cu130，torchvision 0.26，scikit-learn，pandas，PIL，tqdm
# timm 未安装 — 所有 backbone 均通过 torchvision.models 加载
```

所有命令从**项目根目录**运行。

## 分类代码结构

共享实现位于 `Image classification/` 目录：

```
Image classification/
  shared.py              # LABELS、路径、load_module()、AsymmetricLoss、ExperimentConfig
  01_explore_data.py     # 数据统计 + 样本网格图
  02_dataset.py          # VOCDataset、数据变换
  03_model.py            # MultiLabelClassifier（EfficientNet-B3 / ConvNeXt-Tiny / ResNet-50）
  04_train.py            # 三阶段训练
  05_evaluate.py         # val mAP + 逐类阈值搜索
  06_predict.py          # TTA 推理 → 提交 CSV
  run_pipeline.py        # 端到端流水线编排
  merge_submission.py    # 合并分类与分割提交文件
  experiments/
    efficientnet_b3_320/ # train.py  evaluate.py  predict.py
    convnext_tiny_320/   # train.py  evaluate.py  predict.py
    resnet50_224/        # train.py  evaluate.py  predict.py
```

每个实验写入各自独立的输出目录：

```
output/image_classification/<experiment>/
  checkpoints/           best_model.pth  last_model.pth  final_model.pth
  metrics/               training_history.csv  ap_per_class.csv
                         best_thresholds.npy   evaluation_summary.csv
  figures/               eval_ap_per_class.png
  predictions/           test_probabilities_<experiment>.csv
                         test_binary_predictions_<experiment>.csv
  submissions/           submission_classification_<experiment>.csv
                         submission_final_<experiment>__seg_<分割文件名>.csv
```

数据探索图表存储在 `output/image_classification/_data/`。

## 实验列表

| 实验文件夹 | Backbone | 输入尺寸 | 损失函数 | 说明 |
|-----------|----------|---------|---------|------|
| `efficientnet_b3_320` | EfficientNet-B3 | 320×320 | AsymmetricLoss | 强模型（v2） |
| `convnext_tiny_320` | ConvNeXt-Tiny | 320×320 | AsymmetricLoss | 最优结果（v3） |
| `resnet50_224` | ResNet-50 | 224×224 | AsymmetricLoss | 基线对比（v1.1） |

## 运行方式

```bash
# 使用最佳模型跑完整流水线
python "Image classification/run_pipeline.py" --experiment convnext_tiny_320

# 跳过数据探索（PNG 已生成）
python "Image classification/run_pipeline.py" --experiment convnext_tiny_320 --skip-explore

# 使用已有 checkpoint，仅重新生成提交文件
python "Image classification/run_pipeline.py" --experiment convnext_tiny_320 --skip-explore --skip-train --ckpt best
```

模型专属入口脚本：

```bash
python "Image classification/experiments/convnext_tiny_320/train.py"
python "Image classification/experiments/convnext_tiny_320/evaluate.py"
python "Image classification/experiments/convnext_tiny_320/predict.py"

python "Image classification/experiments/efficientnet_b3_320/train.py"
python "Image classification/experiments/resnet50_224/train.py"
```

## 关键设计决策

**AsymmetricLoss（ICCV 2021）** — PASCAL VOC 存在系统性的 false-negative 噪声：
对象出现在图像中但未被标注（例如"自行车"图片背景中的行人）。标准 BCE 将这些未标注
的正例当作负例惩罚。ASL 的 `clip=0.05` 使 `P < 0.05` 的负样本 loss 归零，
`gamma_neg=4` 进一步压低 easy 负样本权重，让 rare 正样本主导梯度。
效果：diningtable AP 从 v1.0 的 0.28 提升至 v3 的 0.57。

**三阶段训练** — 冻结 backbone（5 epoch，lr=1e-3）→ 全网络微调（20 epoch，
lr=1e-4，保存 `best_model.pth`）→ 在全部 750 个样本上重训（5 epoch，lr=5e-5，
保存 `final_model.pth`）。第三阶段消除了 20% 验证集 holdout 对训练数据的浪费。

**逐类阈值** — `05_evaluate.py` 在验证集上逐类搜索使 F1 最大化的阈值，
保存为 `best_thresholds.npy`。Kaggle Dice = F1，此操作直接优化提交指标。

**TTA（测试时增强）** — 推理时对原图和水平翻转图分别计算 sigmoid 概率并取均值，
无训练开销。

## Kaggle GPU Notebook

在 Kaggle 上训练时，上传或打开：

```text
Image classification/kaggle_train.ipynb                      # efficientnet_b3_320
Image classification/kaggle_train_convnext_tiny_320.ipynb    # convnext_tiny_320
```

启用 GPU 加速器后运行所有单元格。Notebook 将输出写入
`/kaggle/working/image_classification/<experiment>/`，
生成的 `submission_classification_<experiment>.csv` 可直接下载提交。

## 合并分割提交

分类预测和分割预测都就绪后执行合并：

```bash
python "Image classification/merge_submission.py" \
  --experiment convnext_tiny_320 \
  --seg-csv output/submission_exp_v7_segformer_b3.csv
```

合并文件写入实验的 `submissions/` 目录。如需指定特定分类文件：

```bash
python "Image classification/merge_submission.py" \
  --clf-csv output/image_classification/resnet50_224/predictions/test_binary_predictions_resnet50_224.csv \
  --seg-csv output/submission_exp_v10_segman_b_iter25000.csv
```
