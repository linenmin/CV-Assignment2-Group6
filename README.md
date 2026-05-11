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
如果需要理解 `mAP`、`per-class AP`、阈值、F1、TP/FP/FN 和训练/ensemble 超参数的计算方式，请看
[ensemble_method_explained.md](<Image classification/ensemble_method_explained.md>)。

| 版本 | 实验文件夹 | 骨干网络 | 损失函数 | 输入 | val mAP | Kaggle 显示 | 分类 Dice（×2） |
|------|-----------|----------|---------|------|---------|------------|----------------|
| v1.0 | *(重构前)* | ResNet-50 | NegativeSmoothBCE | 224 | 0.801 | 0.381 | 0.762 |
| v1.1 | `resnet50_224` | ResNet-50 | AsymmetricLoss | 224 + TTA | 0.818 | 0.392 | 0.783 |
| v2 | `efficientnet_b3_320` | EfficientNet-B3 | AsymmetricLoss | 320 + TTA | 0.860 | 0.428 | 0.856 |
| v3 | `convnext_tiny_320` | ConvNeXt-Tiny | AsymmetricLoss | 320 + TTA | **0.893** | **0.437** | **0.873** |
| v4 | `convnext_small_320` | ConvNeXt-Small | AsymmetricLoss | 320 + TTA | **0.900** | **0.449** | **0.898** |
| v5 | `convnext_base_320` | ConvNeXt-Base | AsymmetricLoss | 320 + TTA | **0.911** | —¹ | —¹ |
| v6 | `efficientnet_v2_s_320` | EfficientNet-V2-S | AsymmetricLoss | 320 + TTA | **0.881** | —¹ | —¹ |
| v7 | `ensemble_small_v2`（Small + V2-S） | per-class 加权融合 | — | 320 + TTA | **0.924**² | —¹ | —¹ |

**当前最佳分类模型：** `convnext_small_320`，Kaggle 显示 **0.44905**，
分类 Dice = **0.89810**。当前最佳完整提交总分：**0.87588**。

¹ v5/v6/v7 均未单独测试纯分类提交。完整提交分数（含分割v10）：
`convnext_base_320` **0.86425**（val mAP 最高但大模型过拟合）；
`efficientnet_v2_s_320` **0.86649**；
`ensemble_small_v2` **0.86657**（略优于单模型 V2-S，但低于 Small 单模型 0.87588）。
Single-model 最佳仍为 `convnext_small_320`。

² v7 val mAP 为 kaggle_ensemble notebook 在 150 张验证集上的 per-class 加权融合结果，
  阈值在相同 150 张样本上搜索，存在轻微过拟合风险，Kaggle 实际得分低于 Small 单模型。

**关于 Kaggle 显示分数：** 完整提交包含分类行和分割行，Kaggle 在 1500 行上统一计算一个 Dice 值；
仅含分类行的提交因缺少分割行得 0，显示分数减半。×2 后的数值才是真实的分类性能。

## 环境配置

所有脚本需要 `biometrics` conda 环境：

```bash
conda activate biometrics
# PyTorch 2.11.0+cu130，torchvision 0.26，scikit-learn，pandas，PIL，tqdm
# timm 未安装 — 所有骨干网络均通过 torchvision.models 加载
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
    convnext_small_320/  # train.py  evaluate.py  predict.py
    convnext_base_320/   # train.py  evaluate.py  predict.py
    convnext_large_320/  # train.py  evaluate.py  predict.py
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

| 实验文件夹 | 骨干网络 | 输入尺寸 | 损失函数 | 说明 |
|-----------|----------|---------|---------|------|
| `efficientnet_b3_320` | EfficientNet-B3 | 320×320 | AsymmetricLoss | 强模型（v2） |
| `convnext_tiny_320` | ConvNeXt-Tiny | 320×320 | AsymmetricLoss | 强结果（v3） |
| `convnext_small_320` | ConvNeXt-Small | 320×320 | AsymmetricLoss | 当前最佳分类结果（v4） |
| `convnext_base_320` | ConvNeXt-Base | 320×320 | AsymmetricLoss | 已测（v5），val mAP 0.911，Kaggle 0.86425，小数据集过拟合 |
| `convnext_large_320` | ConvNeXt-Large | 320×320 | AsymmetricLoss | 未测，高显存/过拟合风险更大，不建议尝试 |
| `efficientnet_v2_s_320` | EfficientNet-V2-S | 320×320 | AsymmetricLoss | 已测（v6），val mAP 0.881，Kaggle 0.86649，pottedplant 最强 |
| `vit_b_16_224` | ViT-B/16 | 224×224 | AsymmetricLoss | Vision Transformer 主线基线 |
| `vit_l_16_224` | ViT-L/16 | 224×224 | AsymmetricLoss | 可选重模型，更适合 Colab/强 GPU |
| `resnet50_224` | ResNet-50 | 224×224 | AsymmetricLoss | 基线对比（v1.1） |

## 运行方式

```bash
# 使用最佳分类模型跑完整流水线
python "Image classification/run_pipeline.py" --experiment convnext_small_320

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

python "Image classification/run_pipeline.py" --experiment convnext_small_320 --skip-explore
python "Image classification/run_pipeline.py" --experiment convnext_base_320 --skip-explore
python "Image classification/run_pipeline.py" --experiment vit_b_16_224 --skip-explore

python "Image classification/experiments/efficientnet_b3_320/train.py"
python "Image classification/experiments/resnet50_224/train.py"
```

ConvNeXt 在当前 `torchvision` 环境中可用的系列为 Tiny/Small/Base/Large。
`convnext_small_320` 为当前最佳单模型（Kaggle 0.87588）。`convnext_base_320`
val mAP 更高（0.911）但在 750 张图上发生测试集过拟合，Kaggle 完整得分（0.86425）
反而低于 Small。`convnext_large_320` 参数量约 198M，过拟合风险更大，不建议继续尝试。

ViT 路线从 `vit_b_16_224` 开始。它使用 torchvision 的 ViT-B/16 ImageNet-1K
预训练权重，并保持 224×224 输入，以避免一开始就引入 positional embedding
插值变量。如果这个基线接近或超过 ConvNeXt-Small，再单独做 320 输入或 ViT-L
实验。

## 关键设计决策

**AsymmetricLoss（ICCV 2021）** — PASCAL VOC 存在系统性的 false-negative 噪声：
对象出现在图像中但未被标注（例如"自行车"图片背景中的行人）。标准 BCE 将这些未标注
的正例当作负例惩罚。ASL 的 `clip=0.05` 使 `P < 0.05` 的负样本 loss 归零，
`gamma_neg=4` 进一步压低 easy 负样本权重，让 rare 正样本主导梯度。
效果：diningtable AP 从 v1.0 的 0.28 提升至 v3 的 0.57。

**三阶段训练** — 冻结骨干网络（5 epoch，lr=1e-3）→ 全网络微调（20 epoch，
lr=1e-4，保存 `best_model.pth`）→ 在全部 750 个样本上重训（5 epoch，lr=5e-5，
保存 `final_model.pth`）。第三阶段消除了 20% 验证集 holdout 对训练数据的浪费。

**逐类阈值** — `05_evaluate.py` 在验证集上逐类搜索使 F1 最大化的阈值，
保存为 `best_thresholds.npy`。Kaggle Dice = F1，此操作直接优化提交指标。

**TTA（测试时增强）** — 推理时对原图和水平翻转图分别计算 sigmoid 概率并取均值，
无训练开销。

## Kaggle / Colab Notebooks

`Image classification/` 目录下包含以下 Kaggle/Colab 训练和推理 notebook：

| Notebook | 用途 |
|----------|------|
| `kaggle_train_convnext_base_320.ipynb` | ConvNeXt-Base 320 训练（已完成，结果见 v5） |
| `kaggle_train_efficientnet_v2_s_320.ipynb` | EfficientNet-V2-S 320 训练（已完成，结果见 v6） |
| `kaggle_ensemble.ipynb` | Small + V2-S 概率加权融合（已完成，结果见 v7） |
| `kaggle_kfold_convnext_small_320.ipynb` | 5-fold CV 阈值校准（T4 约 45 分钟） |
| `colab_train_convnext.ipynb` | Colab 训练，支持 Tiny/Small/Base/Large，含 AMP |

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

## Ensemble 与弱类别恢复

概率 ensemble 已实现并可复现：

```bash
python "Image classification/07_ensemble.py" --mode val-search --name ensemble_existing
python "Image classification/07_ensemble.py" --mode predict --name ensemble_existing
python "Image classification/merge_submission.py" \
  --clf-csv output/image_classification/ensemble_existing/submissions/submission_classification_ensemble_existing.csv \
  --seg-csv output/submission_exp_v10_segman_b_iter25000.csv \
  --out-csv output/image_classification/ensemble_existing/submissions/submission_final_ensemble_existing__seg_submission_exp_v10_segman_b_iter25000.csv
```

ensemble 搜索使用与单模型实验相同的验证集划分，先拟合全局权重和逐类别权重，然后保存阈值和权重 JSON。最新本地运行中，由于 `convnext_tiny_320` 的本地 checkpoint 缺失，脚本跳过了它，并集成了 `resnet50_224`、`efficientnet_b3_320` 和 `convnext_small_320`，验证集 mAP 达到 **0.9257**。Kaggle 反馈显示该 ensemble 不如 `convnext_small_320`，因此它只作为分析/对照保留，不作为默认最终方案。

详细方法说明见：

```text
Image classification/ensemble_method_explained.md
```

另外还注册了一个后续训练实验：

```bash
python "Image classification/run_pipeline.py" --experiment convnext_small_320_pad_sampler --skip-explore
```

它保留 ConvNeXt-Small 骨干网络，但在缩放前先做方形填充；训练时对 `bottle`、`diningtable`、`pottedplant`、`sheep`、`sofa` 做温和加权采样；第二阶段使用 patience=4 的提前停止。
