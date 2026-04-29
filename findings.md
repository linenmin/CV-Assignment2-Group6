# Findings & Decisions

## Requirements
- **任务**: CV Assignment 2 — PASCAL VOC 2009 多标签图像分类（20类）
- **提交格式**: 每个样本2行（`_classification` + `_segmentation`），RLE编码
- **Section结构**: 分类(S1) + 语义分割(S2) + 对抗攻击(S3) + 讨论(S4)
- **评分指标**: Kaggle Dice score（非 mAP，阈值敏感）

---

## Project Structure（当前）

```
Image classification/
├── shared.py            # 常量、路径、load_module、AsymmetricLoss
├── 01_explore_data.py   # 数据探索 + 全量PNG转换
├── 02_dataset.py        # VOCDataset, train/val/test transform
├── 03_model.py          # MultiLabelClassifier (resnet50 / efficientnet_b3 / convnext_tiny)
├── 04_train.py          # 3阶段训练 (S1冻结 + S2全微调 + S3全数据重训)
├── 05_evaluate.py       # mAP评估 + per-class阈值优化
├── 06_predict.py        # 测试集推理(含TTA) + RLE编码
├── merge_submission.py  # 合并classification + segmentation
└── run_pipeline.py      # 一键执行全流程
```

---

## Research Findings

### 数据集特性（2026-04-23 观测）
- 训练集：750张，测试集：750张，`.npy` numpy格式（RGB uint8）
- 图像尺寸：高度 min/max 不等，约 333×500（需resize）
- 多标签：每张图平均含 1.8+ 个类别标签
- **标注噪声**：大量 false negative（人物、桌椅等出现在背景但未被标注）
  - 典型案例：被标为bicycle的图包含bus/person；diningtable严重受此影响

### Kaggle实验结果

val mAP 来自各实验目录下的 `evaluation_summary.csv`（使用 `best_model.pth` 评估）。

| 版本 | 实验文件夹 | 模型 | 损失函数 | 输入尺寸 | val mAP | Kaggle显示分数 | 分类Dice（x2） |
|------|-----------|------|----------|----------|---------|---------------|----------------|
| v1.0 | *(重构前)* | ResNet-50 | NegativeSmoothBCE | 224 | 0.801 | 0.38084 | 0.76168 |
| v1.1 | `resnet50_224` | ResNet-50 | AsymmetricLoss | 224+TTA+S3 | 0.8175 | 0.39165 | 0.78330 |
| v2 | `efficientnet_b3_320` | EfficientNet-B3 | AsymmetricLoss | 320+TTA+S3 | **0.8599** | 0.42813 | 0.85626 |
| v3 | `convnext_tiny_320` | ConvNeXt-Tiny | AsymmetricLoss | 320+TTA+S3 | **0.8933** | **0.43673** | **0.87346** |

注：Kaggle 显示分数需×2 得到真实分类 Dice（因提交含 1500 行，750 行为分割占位）。
当前最佳模型：`convnext_tiny_320`，分类 Dice = **0.87346**。

### 逐类 AP 对比（三个当前实验，best_model.pth）

| 类别 | resnet50_224 | efficientnet_b3_320 | convnext_tiny_320 |
|------|-------------|---------------------|-------------------|
| aeroplane | 0.981 | 0.959 | **0.991** |
| bicycle | 0.967 | 0.837 | **0.967** |
| bird | 0.899 | 0.927 | **0.977** |
| boat | 0.981 | 0.992 | **1.000** |
| bottle | **0.785** | 0.800 | 0.764 |
| bus | 0.944 | **0.976** | **0.976** |
| car | 0.886 | 0.866 | **0.897** |
| cat | 0.911 | 0.950 | **0.992** |
| chair | 0.737 | 0.807 | **0.831** |
| cow | 0.792 | **1.000** | **1.000** |
| diningtable | 0.386 | 0.675 | **0.573** |
| dog | 0.723 | 0.774 | **0.856** |
| horse | 0.825 | 0.882 | **0.883** |
| motorbike | 0.877 | 0.922 | **0.944** |
| person | 0.915 | 0.937 | **0.943** |
| pottedplant | 0.629 | 0.603 | **0.792** |
| sheep | 0.651 | 0.738 | **0.841** |
| sofa | 0.709 | 0.667 | **0.758** |
| train | 0.963 | **1.000** | **1.000** |
| tvmonitor | 0.791 | 0.886 | **0.880** |
| **mAP** | **0.8175** | **0.8599** | **0.8933** |

关键观察：
- **diningtable**：所有模型仍是最弱类（ASL 虽有改善，但 false-negative 噪声根本问题未解）
- **pottedplant**：ConvNeXt-Tiny 大幅领先（0.792 vs 0.629/0.603），受益于更好的局部特征提取
- **bottle**：ResNet-50 意外领先（0.785），ConvNeXt 较低（0.764）
- **EfficientNet vs ConvNeXt**：ConvNeXt 在 17/20 类上更强，仅 bottle/diningtable 落后

### Val mAP vs Kaggle Dice 的根本差距
- val mAP = 0.801（AUC指标，阈值无关）
- Kaggle Dice = 0.38（阈值敏感，依赖最优二值化）
- **结论**：mAP高并不直接等于Dice高；需要正确的per-class阈值+正确的损失函数

### Per-class AP 分析（v1 ResNet-50）
| 类别 | AP | 问题原因 |
|------|-----|---------|
| diningtable | **0.28** | false-negative密度最高（桌子大量出现在背景但未标注） |
| pottedplant | ~0.63 | 小目标，224px分辨率损失严重 |
| person | ~0.65 | 背景中人物无处不在但未标注 |
| aeroplane, boat, cow | >0.95 | 外观distinctive，标注干净 |

---

## 技术决策（当前状态）

### 损失函数：AsymmetricLoss（ICCV 2021）
- 替换 `NegativeSmoothBCELoss`
- 参数：`gamma_neg=4, gamma_pos=0, clip=0.05`
- **clip**：当模型对某个负样本预测的 P < 0.05，loss=0，不惩罚潜在false-negative
- **gamma_neg=4**：对easy负样本(p≪0.5)强力下权，让rare正样本主导梯度
- 论文：https://arxiv.org/abs/2009.14119

### 骨干网络：ConvNeXt-Tiny（当前最佳 v3）
- 当前最佳实验为 `convnext_tiny_320`，替代 v2 `efficientnet_b3_320` 作为默认/推荐分类 backbone。
- 通过 `torchvision.models.convnext_tiny` 加载 ImageNet-1K 预训练权重。
- 特征提取路径：`nn.Sequential(model.features, model.avgpool)` + `flatten(1)`。
- 池化特征维度：768（EfficientNet-B3 为 1536，ResNet-50 为 2048）。
- 实测结果：val mAP **0.8933**，Kaggle 显示 **0.43673**，分类 Dice **0.87346**。
- EfficientNet-B3 保留为 v2 对比实验，不再是当前最佳 backbone。

### 输入尺寸：320×320（替代224×224）
- 原始图像约333×500，224缩放损失大量细节
- 320保留更多空间信息，对小目标（bottle, pottedplant）尤其重要
- 内存影响：BATCH_SIZE从32降至16

### 数据增强（训练集）
- RandomHorizontalFlip + RandomRotation(15°)
- ColorJitter(0.3, 0.3, 0.2, 0.05) — 更强的颜色扰动
- RandomErasing(p=0.3, scale=(0.02, 0.2)) — 随机遮挡，增强对局部遮挡的鲁棒性

### 训练策略：3阶段
| 阶段 | Backbone | Epochs | LR | 目的 |
|------|----------|--------|----|------|
| Stage 1 | 冻结 | 5 | 1e-3 | 快速训练分类头 |
| Stage 2 | 解冻 | 20 | 1e-4 | 全网络细调 |
| Stage 3 | 解冻 | 5 | 5e-5 | 全750样本重训（消除val holdout损失） |

### TTA（Test-Time Augmentation）
- 推理时运行 original + horizontal flip，平均sigmoid概率
- 无训练成本，+0.5~1% Dice预期

---

## 关键超参数（当前最佳 v3：convnext_tiny_320）
| 参数 | 值 |
|------|-----|
| EXPERIMENT | convnext_tiny_320 |
| BACKBONE | convnext_tiny |
| IMG_SIZE | 320 |
| BATCH_SIZE | 16 |
| STAGE1_EPOCHS | 5 |
| STAGE1_LR | 1e-3 |
| STAGE2_EPOCHS | 20 |
| STAGE2_LR | 1e-4 |
| STAGE3_EPOCHS | 5 |
| STAGE3_LR | 5e-5 |
| WEIGHT_DECAY | 1e-4 |
| VAL_SPLIT | 0.2 |
| ASL gamma_neg | 4 |
| ASL gamma_pos | 0 |
| ASL clip | 0.05 |

---

## 遇到的问题
| Issue | Resolution |
|-------|------------|
| Windows multiprocessing spawn | `NUM_WORKERS=0` on `os.name=='nt'` (in shared.py) |
| 模块名以数字开头无法import | `load_module()` via importlib.util |
| VOC标注false-negative噪声 | AsymmetricLoss clip参数消除easy负样本惩罚 |
| val mAP与Kaggle Dice差距大 | 优化per-class阈值 + 换ASL + 更大输入尺寸 |

---

## 资源
- `CV_GA2.pdf`：作业详细说明（39页）
- `kul-computer-vision-ga-2-2026/`：数据集目录
- `output/`：所有输出文件（checkpoints、submissions、可视化图表）

## 输出文件状态（已完成）
| 文件 | 说明 | 状态 |
|------|------|------|
| `output/image_classification/*/checkpoints/best_model.pth` | val 最优模型（S2 结束） | 三个实验均已生成 |
| `output/image_classification/*/checkpoints/final_model.pth` | 全数据重训模型（S3 结束） | 三个实验均已生成 |
| `output/image_classification/*/metrics/best_thresholds.npy` | per-class 最优 F1 阈值 | 三个实验均已生成 |
| `output/image_classification/*/submissions/submission_classification_*.csv` | 分类预测 RLE | 三个实验均已生成 |
| `output/image_classification/*/figures/eval_ap_per_class.png` | per-class AP 柱状图 | 三个实验均已生成 |

---

## 未完成部分
1. **Section 3：对抗攻击** — 未实现（FGSM / PGD）
2. **Section 4：讨论** — notebook 未填写
3. **学生姓名** — notebook cell 0 占位符未填写

---

## 2026-04-25 发现：模型专属分类组织结构

### 代码审查发现
- 旧版分类脚本使用硬编码全局变量（`BACKBONE`、`IMG_SIZE`、`CKPT_PATH`、`THRESH_PATH`），
  所有输出写入同一 `output/` 路径，导致不同模型的 checkpoint、阈值、图表和提交 CSV 会互相覆盖。
- 最佳简化方案不是为每个模型复制完整训练代码，而是用共享 `ExperimentConfig`
  统一管理 backbone、输入尺寸、batch size、随机种子和输出路径；各模型文件夹仅作为薄封装入口。
- `run_pipeline.py --ckpt last` 之前指向一个并未被一致保存的文件名；
  现在训练每个 epoch 写入 `last_model.pth`，全数据重训后写入 `final_model.pth`。
- 预测现在同时保存可读 CSV 和 Kaggle 提交 CSV：概率文件、二值预测文件、RLE 提交文件分开保存。

### 新增模型文件夹
- `Image classification/experiments/efficientnet_b3_320/`
- `Image classification/experiments/resnet50_224/`

### 新输出目录约定
每个实验独立拥有：

```
output/image_classification/<experiment>/
  checkpoints/
  metrics/
  figures/
  predictions/
  submissions/
```

### 验证说明
- 所有修改的源文件通过 `py_compile` 检查。
- 在 `biometrics` 环境中，共享脚本和实验封装脚本的帮助输出均正常。
- 用虚拟输入验证两个 backbone 前向传播均返回 `(1, 20)`。

### 2026-04-25 后续更新
- 数据探索步骤不再重写 PNG（数据集已提前转换完毕），
  `01_explore_data.py` 仅读取 `.npy` 文件进行形状统计和样本网格绘图。
- 提交和预测 CSV 的文件名中加入实验名称，便于在模型目录外识别。

### 2026-04-25 Kaggle Notebook
- `kaggle_train.ipynb` 保持自包含，因为 Kaggle 环境默认不含本地的数字前缀 Python 模块。
- Notebook 现在与本地实验命名约定保持一致，所有制品写入
  `/kaggle/working/image_classification/efficientnet_b3_320/`。
- 最终 Kaggle 分类 CSV 为 `submission_classification_efficientnet_b3_320.csv`。

### 2026-04-25 预测索引 Bug
- `test_set.csv` 包含与 `train_set.csv` 相同的 20 个类别列，但值为 `-1` 占位符；
  简单的 `has_labels = all(column in df.columns for column in LABELS)` 检查因此
  将测试集误判为已标注数据。
- 在 `VOCDataset.__getitem__` 中，这导致测试样本返回 `(img, label)` 而非 `(img, idx)`，
  预测代码不能依赖 DataLoader batch 的第二个返回值作为测试 ID。
- 正确做法：
  - 创建 `test_ds = VOCDataset(...)`
  - 迭代时使用 `for imgs, _ in test_loader`
  - 推理完成后用 `all_indices = list(test_ds.indices)` 获取 ID
  - 在构建 DataFrame 前断言 `all_probs.shape[0] == len(all_indices)`
- 此修复同时适用于本地 `06_predict.py` 和 Kaggle notebook 预测单元格。

### 2026-04-25 ConvNeXt-Tiny 规划
- `biometrics` 环境中的 `torchvision==0.26.0+cu130` 已包含 `convnext_tiny`、
  `convnext_small`、`ConvNeXt_Tiny_Weights` 和 `ConvNeXt_Small_Weights`，无需 HuggingFace 依赖。
- ConvNeXt-Tiny 经 `nn.Sequential(model.features, model.avgpool)` + `flatten(1)` 后
  池化特征维度为 768。
- 首个 ConvNeXt 实验选 `convnext_tiny_320` 而非 ConvNeXt-Small，
  以保持与 EfficientNet-B3 相近的计算量和 Kaggle GPU 显存占用。

### 2026-04-25 ConvNeXt-Tiny 结果
- `convnext_tiny_320` 完成三阶段训练。
- 使用 `best_model.pth` 在本地评估，val mAP = **0.8932642162**。
- 最优 val loss：S2 第 4 epoch **0.0293972875**；此后 val loss 开始上升而训练 loss 继续下降。
- S3 全量重训完成，最终训练 loss = **0.0075267962**。
- 最低 AP 类别：diningtable **0.5732**、sofa **0.7582**、bottle **0.7645**、
  pottedplant **0.7916**、chair **0.8308**。
- 最高 AP 类别：train **1.0000**、boat **1.0000**、cow **1.0000**、
  cat **0.9922**、aeroplane **0.9909**。
- 在 `output/image_classification/convnext_tiny_320/` 下生成所有制品：
  checkpoints、metrics、`eval_ap_per_class.png`、概率文件、二值预测和提交 CSV。
- `submission_classification_convnext_tiny_320.csv` 和
  `submission_final_convnext_tiny_320.csv` 均含 1500 行。
- ConvNeXt Kaggle 显示分数 **0.43673**，×2 后分类 Dice = **0.87346**。

### 2026-04-26 Kaggle 分类得分
| 实验文件夹 | Kaggle 显示 | 分类 Dice（×2） | 排名 |
|-----------|------------|----------------|------|
| `convnext_tiny_320` | **0.43673** | **0.87346** | 1 |
| `efficientnet_b3_320` | **0.42813** | **0.85626** | 2 |
| `resnet50_224` | **0.39165** | **0.78330** | 3 |

决策：在 ensemble 或阈值更新超过 **0.87346** 之前，以 `convnext_tiny_320` 作为最强分类提交。
