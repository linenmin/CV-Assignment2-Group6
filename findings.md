# 发现与决策

## 需求
- **任务**: CV Assignment 2 — PASCAL VOC 2009 多标签图像分类（20类）
- **提交格式**: 每个样本2行（`_classification` + `_segmentation`），RLE编码
- **Section结构**: 分类(S1) + 语义分割(S2) + 对抗攻击(S3) + 讨论(S4)
- **评分指标**: Kaggle Dice score（非 mAP，阈值敏感）

---

## 项目结构（当前）

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

## 调研发现

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
| v4 | `convnext_small_320` | ConvNeXt-Small | AsymmetricLoss | 320+TTA+S3 | **0.8995** | **0.44905** | **0.89810** |

注：Kaggle 显示分数需×2 得到真实分类 Dice（因提交含 1500 行，750 行为分割占位）。
当前最佳分类模型：`convnext_small_320`，分类 Dice = **0.89810**。当前最佳完整提交总分：
**0.87588**（`convnext_small_320` + `submission_exp_v10_segman_b_iter25000.csv`）。

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

### 骨干网络：ConvNeXt-Small（当前最佳 v4）
- 当前最佳分类实验为 `convnext_small_320`，替代 v3 `convnext_tiny_320` 作为默认/推荐分类骨干网络。
- 通过 `torchvision.models.convnext_small` 加载 ImageNet-1K 预训练权重。
- 特征提取路径：`nn.Sequential(model.features, model.avgpool)` + `flatten(1)`。
- 池化特征维度：768（ConvNeXt-Tiny 同为 768，EfficientNet-B3 为 1536，ResNet-50 为 2048）。
- 实测结果：val mAP **0.8995**，Kaggle 显示 **0.44905**，分类 Dice **0.89810**。
- EfficientNet-B3 保留为 v2 对比实验，不再是当前最佳骨干网络。

### 输入尺寸：320×320（替代224×224）
- 原始图像约333×500，224缩放损失大量细节
- 320保留更多空间信息，对小目标（bottle, pottedplant）尤其重要
- 内存影响：BATCH_SIZE从32降至16

### 数据增强（训练集）
- RandomHorizontalFlip + RandomRotation(15°)
- ColorJitter(0.3, 0.3, 0.2, 0.05) — 更强的颜色扰动
- RandomErasing(p=0.3, scale=(0.02, 0.2)) — 随机遮挡，增强对局部遮挡的鲁棒性

### 训练策略：3阶段
| 阶段 | 骨干网络 | Epochs | LR | 目的 |
|------|----------|--------|----|------|
| 第一阶段 | 冻结 | 5 | 1e-3 | 快速训练分类头 |
| 第二阶段 | 解冻 | 20 | 1e-4 | 全网络细调 |
| 第三阶段 | 解冻 | 5 | 5e-5 | 全750样本重训（消除val holdout损失） |

### Checkpoint 说明（best / last / final）

训练过程产生三类 `.pth` 文件，保存在 `output/image_classification/<experiment>/checkpoints/` 下：

| 文件 | 保存时机 | 用途 |
|------|---------|------|
| `best_model.pth` | 第一阶段 / 第二阶段的每个 epoch 结束后，若该 epoch 的 val loss **低于历史最优**，则覆盖保存（`04_train.py` line 147）。训练结束时指向整个 S1+S2 阶段 val loss 最低的权重。 | `05_evaluate.py` 用此文件计算 val mAP 和最优阈值；`06_predict.py` 在 `final_model.pth` 不存在时也 fallback 到此文件。 |
| `last_model.pth` | 每个 epoch 结束时**无条件覆盖**保存（line 149）。始终是最新一个 epoch 的权重，不论验证性能好坏。 | 断点续训时的恢复入口；一般不用于最终推理。 |
| `final_model.pth` | Stage 3 训练完毕后保存（line 269）。Stage 3 从 `best_model.pth` 加载权重，在**全量 750 张**训练样本上（无 val holdout）继续微调 5 个 epoch。 | `06_predict.py` 优先加载此文件做测试集推理（含 TTA）。它比 `best_model.pth` 多利用了 20% holdout 数据，通常可带来轻微的泛化提升。 |

> 推理调用顺序：`final_model.pth` → (若不存在) `best_model.pth`。
> 评估（`05_evaluate.py`）固定使用 `best_model.pth`，以确保 val mAP 的可比性。

### TTA（Test-Time Augmentation）
- 推理时运行 original + horizontal flip，平均sigmoid概率
- 无训练成本，+0.5~1% Dice预期

---

## 关键超参数（当前最佳 v4：convnext_small_320）
| 参数 | 值 |
|------|-----|
| EXPERIMENT | convnext_small_320 |
| BACKBONE | convnext_small |
| IMG_SIZE | 320 |
| BATCH_SIZE | 8 |
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
  统一管理骨干网络、输入尺寸、批大小、随机种子和输出路径；各模型文件夹仅作为薄封装入口。
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
- 用虚拟输入验证两个骨干网络前向传播均返回 `(1, 20)`。

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

当时决策：在 ensemble 或阈值更新超过 **0.87346** 之前，以 `convnext_tiny_320`
作为最强分类提交。2026-05-09 后续 Kaggle 测试已由 `convnext_small_320`
超过该分数。

---

## 2026-05-09 ConvNeXt 更大变体发现

- 初始动机：`convnext_tiny_320` 的 Kaggle 显示分数为 **0.43673**，换算后的分类 Dice 为 **0.87346**。后续 Kaggle 测试确认，`convnext_small_320` 提升到显示分数 **0.44905**，换算后的分类 Dice 为 **0.89810**。
- `torchvision` 中没有名为 "Middle" 的 ConvNeXt 版本；可用系列是 Tiny、Small、Base 和 Large。
- 本地 `torchvision==0.26.0+cu130` 暴露了全部四个模型构造函数和四组 ImageNet-1K 权重枚举：
  - `convnext_tiny` / `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`
  - `convnext_small` / `ConvNeXt_Small_Weights.IMAGENET1K_V1`
  - `convnext_base` / `ConvNeXt_Base_Weights.IMAGENET1K_V1`
  - `convnext_large` / `ConvNeXt_Large_Weights.IMAGENET1K_V1`
- 本地参数量：
  - Tiny：28.59M，特征维度 768。
  - Small：50.22M，特征维度 768。
  - Base：88.59M，特征维度 1024。
  - Large：197.77M，特征维度 1536。
- 推荐实验顺序：
  1. `convnext_small_320` 作为最可能有效的下一步。
  2. 如果 Small 表现有提升，再尝试 `convnext_base_320`。
  3. 只有在时间和 GPU 预算允许时才尝试 `convnext_large_320`；它在 8 GB 显存上成本高，也更容易在 750 张训练图上过拟合。
- `CV_GA2.pdf` 确认分类任务允许选择已知架构，并鼓励迁移学习和 fine-tuning；前提是预训练没有使用 PASCAL VOC 图像。ImageNet-1K ConvNeXt 权重符合这个约束。

### `convnext_small_320` 结果
- 已在本地完成完整 train/evaluate/predict 流水线。
- val mAP：**0.8995381256**，在相同验证协议下高于 `convnext_tiny_320`（**0.8932642162**）。
- 最优 val loss：第二阶段第 2 个 epoch 为 **0.02936561396**。后续第二阶段 epoch 出现过拟合：训练 loss 继续下降，但 val loss 上升。
- 第三阶段全量训练最终 loss：**0.0073612132**。
- AP 最弱类别：diningtable **0.5570**、pottedplant **0.7518**、sofa **0.7872**、bottle **0.7914**、sheep **0.8167**。
- 已生成 1500 行提交文件：
  `submission_classification_convnext_small_320.csv`.
- 2026-05-09 收到 Kaggle 结果：
  - 完整提交分数：**0.87588**。
  - 分类 Kaggle 显示分数：**0.44905**。
  - 换算后的分类 Dice：**0.89810**。
  - 决策：`convnext_small_320` 取代 `convnext_tiny_320`，成为当前最佳分类模型。

### Colab 训练决策
- 本地 RTX 4060 Laptop 8 GB 能运行当前全部 ConvNeXt 预设，主要依赖于更大模型使用更小批大小。
- Small 适合本地或 Colab 运行；如果有 Colab Pro、A100、L4 等更强运行时，Base 和 Large 更适合放到那里尝试。
- 已新增 `Image classification/colab_train_convnext.ipynb`，包含：
  - ConvNeXt Tiny/Small/Base/Large 预设。
  - 保守的 Colab 批大小：Tiny 32、Small 16、Base 8、Large 4。
  - AMP 混合精度，用于降低显存占用并提高吞吐。
  - Google Drive 数据集和输出路径。
  - 完整训练前的一步 `memory_probe()` 显存检查。

### Colab 图像输出发现
- `convnext_large_320_colab/figures` 为空，是因为 Colab notebook 只创建了目录并保存 CSV 指标，没有导入 matplotlib，也没有调用 `savefig`。
- 本地评估脚本 `05_evaluate.py` 已经会保存 `figures/eval_ap_per_class.png`；Colab notebook 在自包含改写时遗漏了这一步绘图。
- 已更新 `Image classification/colab_train_convnext.ipynb`，现在会保存：
  - 从 `metrics/training_history.csv` 生成的 `figures/training_history.png`。
  - 从 `metrics/ap_per_class.csv` 生成的 `figures/eval_ap_per_class.png`。
- 新增可选重生成 cell，使已有 Colab 运行可以直接从 CSV 文件补图，无需重新训练。

---

## 2026-05-10 Ensemble 与弱类别恢复

- 在引入 transformer 依赖之前，新增了一个低成本的概率 ensemble 路径。CLI 为 `Image classification/07_ensemble.py`，支持 `--mode val-search` 和 `--mode predict`。
- 最新本地 `val-search` 使用了可用 checkpoint：`resnet50_224`、`efficientnet_b3_320` 和 `convnext_small_320`；由于当前 checkout 中没有 `convnext_tiny_320` 的本地 `best_model.pth`，脚本跳过了它。
- Ensemble 结果：自动选择逐类别加权，val mAP **0.925702**，弱类别平均 F1 **0.790040**。同一套 TTA 验证搜索中的单模型基线为：
  - `resnet50_224`：mAP **0.828671**，弱类别 F1 **0.644744**。
  - `efficientnet_b3_320`：mAP **0.860495**，弱类别 F1 **0.687839**。
  - `convnext_small_320`：mAP **0.902909**，弱类别 F1 **0.740159**。
- 弱类别 ensemble 权重：
  - `diningtable`：0.75 EfficientNet-B3 + 0.25 ConvNeXt-Small。
  - `bottle`：0.50 ResNet-50 + 0.125 EfficientNet-B3 + 0.375 ConvNeXt-Small。
  - `pottedplant`、`sofa`：主要依赖 ConvNeXt-Small。
  - `sheep`：0.333 EfficientNet-B3 + 0.667 ConvNeXt-Small。
- 已生成：
  - `output/image_classification/ensemble_existing/metrics/ensemble_weights.json`
  - `output/image_classification/ensemble_existing/submissions/submission_classification_ensemble_existing.csv`
  - `output/image_classification/ensemble_existing/submissions/submission_final_ensemble_existing__seg_submission_exp_v10_segman_b_iter25000.csv`
- 新增 `convnext_small_320_pad_sampler`：保留 ConvNeXt-Small，但使用方形 padding 后再 resize、弱类别加权采样，以及第二阶段 patience=4 的 early stopping。

## 2026-05-11 Vision Transformer 方向

- 用户反馈 `ensemble_existing` 的 Kaggle 结果不如 `convnext_small_320`，因此 ensemble 不作为最终默认分类方案。
- ViT 主线先从 `vit_b_16_224` 开始：
  - `torchvision.models.vit_b_16`
  - ImageNet-1K 预训练权重
  - 224 x 224 输入
  - batch size 8
  - AsymmetricLoss、三阶段训练、per-class 阈值、TTA 沿用现有分类流水线
- `vit_l_16_224` 作为可选重模型入口，batch size 2；更适合 Colab Pro 或更强 GPU。
- 暂不直接做 320 输入 ViT，因为 torchvision 官方 ViT 权重的 positional embedding 原生对应 224 输入；如果 `vit_b_16_224` 结果有希望，再实现 positional embedding 插值版本。
