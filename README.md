# CV 作业 2 — 第 6 组

KUL H02A5a 计算机视觉，组队作业 2。成员：Kaixi Yao, Enmin Lin, Taicheng Liu, Zhenyang Li。

本仓库负责 **2.1 多标签图像分类** 与 **2.3 对抗攻击（图像分类）** 两个子任务。

---

## 快速结论（给整合人）

| 子任务 | 最终结果 |
|--------|---------|
| **2.1 分类** | ConvNeXt-Small 320，val mAP **0.900**，Kaggle 分类 Dice **0.898**，完整提交总分 **0.876** |
| **2.3 对抗攻击** | 在 convnext_small_320 上对目标类 `aeroplane` 做 targeted white-box 攻击：FGSM 成功率 **26.6%**，PGD（10 步）成功率 **100%** |

---

## 任务说明

数据集：PASCAL VOC 2009 子集，750 张训练图像 + 750 张测试图像，存为 `.npy` 数组。  
每张图像需预测 20 个类别是否出现（多标签二分类）。Kaggle 评分指标为 Dice（等价于 F1）。

类别：`aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor`

---

## 2.1 分类结果

### 全实验对比

> Kaggle 显示分数需 ×2 得到真实分类 Dice，原因见下方说明。

| 版本 | 实验文件夹 | 骨干网络 | 损失函数 | 输入 | val mAP | Kaggle 显示 | 分类 Dice（×2） |
|------|-----------|----------|---------|------|---------|------------|----------------|
| v1.0 | *(重构前)* | ResNet-50 | NegativeSmoothBCE | 224 | 0.801 | 0.381 | 0.762 |
| v1.1 | `resnet50_224` | ResNet-50 | AsymmetricLoss | 224 + TTA | 0.818 | 0.392 | 0.783 |
| v2 | `efficientnet_b3_320` | EfficientNet-B3 | AsymmetricLoss | 320 + TTA | 0.860 | 0.428 | 0.856 |
| v3 | `convnext_tiny_320` | ConvNeXt-Tiny | AsymmetricLoss | 320 + TTA | 0.893 | 0.437 | 0.873 |
| v4 | `convnext_small_320` | ConvNeXt-Small | AsymmetricLoss | 320 + TTA | **0.900** | **0.449** | **0.898** |
| v5 | `convnext_base_320_colab` | ConvNeXt-Base | AsymmetricLoss | 320 + TTA | 0.898 | —¹ | —¹ |
| — | `convnext_large_320` | ConvNeXt-Large | AsymmetricLoss | 320 + TTA | 0.874 | —¹ | —¹ |
| — | `convnextv2_tiny_320` | ConvNeXt V2-Tiny | AsymmetricLoss | 320 + TTA | 0.888 | **0.439** | **0.878** |
| — | `vit_b_16_224` | ViT-B/16 | AsymmetricLoss | 224 + TTA | 0.787 | —¹ | —¹ |
| — | `vit_l_16_224` | ViT-L/16 | AsymmetricLoss | 224 + TTA | 0.797 | —¹ | —¹ |
| v7 | `ensemble_small_v2`（Small + V2-S） | per-class 加权融合 | — | 320 + TTA | 0.924² | —¹ | —¹ |

**当前最佳分类模型：** `convnext_small_320`，Kaggle 显示 **0.44905**，分类 Dice = **0.89810**。  
**最佳完整提交总分（分类 + 分割）：** **0.87588**。

¹ 未单独提交纯分类版本，无法直接读出 Kaggle 分类 Dice。  
² v7 val mAP 在 150 张验证集上搜索 + 融合，存在轻微过拟合风险，Kaggle 实际得分低于 Small 单模型。

**ConvNeXt-Large 说明：** 参数量 ~198M，在 750 张图上过拟合严重，val mAP 反而低于 Small，不推荐继续扩大。  
**ViT 路线说明：** ViT-B/16 和 ViT-L/16 均使用 224 输入（保持 torchvision 预训练 positional embedding 原生尺寸），val mAP 显著低于 ConvNeXt，不作为主线。

### 最佳模型逐类 AP（convnext_small_320）

| 类别 | AP | 最优阈值 |
|------|----|---------|
| aeroplane | 0.991 | 0.60 |
| bicycle | 1.000 | 0.75 |
| bird | 1.000 | 0.75 |
| boat | 1.000 | 0.60 |
| bottle | 0.791 | 0.75 |
| bus | 1.000 | 0.70 |
| car | 0.879 | 0.65 |
| cat | 0.969 | 0.65 |
| chair | 0.832 | 0.75 |
| cow | 1.000 | 0.85 |
| **diningtable** | **0.557** | 0.85 |
| dog | 0.859 | 0.55 |
| horse | 0.883 | 0.70 |
| motorbike | 0.959 | 0.55 |
| person | 0.945 | 0.75 |
| **pottedplant** | **0.752** | 0.75 |
| **sheep** | **0.817** | 0.50 |
| **sofa** | **0.787** | 0.80 |
| train | 1.000 | 0.60 |
| tvmonitor | 0.970 | 0.85 |

最难类别为 `diningtable`（AP=0.557），主要因为该类频繁出现在图像背景中但未被标注（false-negative 噪声）。AsymmetricLoss 的 `clip=0.05` 对此有缓解作用（v1 ResNet-50 的 diningtable AP 仅 0.28）。

---

## 2.3 对抗攻击结果

在**冻结的** `convnext_small_320/final_model.pth` 上执行 targeted white-box attack，目标类别：`aeroplane`。  
攻击样本：64 张验证集图像，均经过筛选确保 ground-truth 中**不含** `aeroplane`（攻击定义更干净）。

### 定量结果

| 攻击方式 | ε | 步数 | clean 目标概率 | adv 目标概率 | 攻击成功率 | pixel L∞（均值） |
|---------|---|------|--------------|------------|----------|-----------------|
| Clean（基线） | — | — | 0.074 | — | 0% | — |
| **FGSM** | 0.03 | 1 | 0.074 | **0.438** | **26.6%** | 0.00687 |
| **PGD** | 0.03（α=0.01）| 10 | 0.074 | **0.983** | **100%** | 0.00687 |

- Clean 激活率 0%：确认这些样本原本不含目标类。
- FGSM 单步即可将目标概率从 0.074 提升至 0.438，但因步长较粗，仅 1/4 样本越过激活阈值。
- PGD 10 步将目标概率推至 0.983，100% 样本攻击成功。
- pixel L∞ 均值仅 **0.00687**（[0,1] 图像尺度），扰动对人眼几乎不可见。

### 攻击方法

```
攻击类型：targeted white-box
目标损失：binary cross-entropy(model(adv), one-hot[aeroplane])
FGSM：adv = x − ε · sign(∇_x L)
PGD ：adv_{t+1} = clip_{ε}(adv_t − α · sign(∇_x L))，并将像素裁剪到合法归一化范围
模型：冻结（model.eval()，所有参数 requires_grad=False）
扰动约束：∥δ∥_∞ ≤ ε=0.03（normalized space），同时保持像素合法
可视化：clean | perturbation×10 | adversarial 三列网格
```

**关键结论：** 模型在 white-box 条件下对小扰动高度敏感。PGD 100% 成功率说明 worst-case 鲁棒性不足，但正常图像上模型表现良好（val mAP 0.900）。提升鲁棒性的方向：adversarial training、更强数据增强、输入预处理检测、ensemble。

### 输出文件

```
output/image_classification/convnext_small_320/
  metrics/adversarial_aeroplane.csv          # 攻击指标表
  figures/adversarial_aeroplane_examples.png # clean/扰动/adversarial 可视化
```

### 运行命令

```bash
# 完整实验（64 样本，10 步 PGD）
/c/Users/31667/.conda/envs/biometrics/python.exe \
  "Image classification/07_adversarial_attack.py" \
  --experiment convnext_small_320 \
  --target-class aeroplane \
  --max-samples 64 \
  --epsilon 0.03 --alpha 0.01 --pgd-steps 10 \
  --ckpt final

# 快速 smoke test（8 样本，3 步 PGD）
/c/Users/31667/.conda/envs/biometrics/python.exe \
  "Image classification/07_adversarial_attack.py" \
  --experiment convnext_small_320 --max-samples 8 --pgd-steps 3
```

---

## 方法说明

### 骨干网络与分类头

`MultiLabelClassifier(backbone, num_classes=20)` 支持以下骨干网络（均来自 `torchvision.models`，ImageNet-1K 预训练）：

| 骨干网络 | 参数量 | 特征维度 |
|---------|--------|---------|
| ResNet-50 | 25M | 2048 |
| EfficientNet-B3 | 12M | 1536 |
| ConvNeXt-Tiny | 29M | 768 |
| ConvNeXt-Small | 50M | 768 |
| ConvNeXt-Base | 89M | 1024 |
| ConvNeXt-Large | 198M | 1536 |
| ConvNeXt V2-Tiny | ~29M | 768 |
| ViT-B/16 | 86M | 768 |
| ViT-L/16 | 307M | 1024 |

分类头：`Linear(feat_dim → 512) → ReLU → Linear(512 → 20)`，输出原始 logits（sigmoid 在 loss/推理时施加，不在 forward 内）。

### AsymmetricLoss（ICCV 2021）

PASCAL VOC 存在系统性 false-negative 噪声（对象出现但未标注）。标准 BCE 将这些未标注正例当负例惩罚。

- `clip=0.05`：P < 0.05 的负样本 loss 归零，不惩罚潜在 false-negative。
- `gamma_neg=4`：easy 负样本梯度被强力压制，rare 正样本主导梯度。
- 效果：diningtable AP 从 v1.0 的 0.28 提升至 v4 的 0.56。

### 三阶段训练

| 阶段 | 骨干 | Epochs | LR | 目的 |
|------|------|--------|----|------|
| Stage 1 | 冻结 | 5 | 1e-3 | 预热分类头 |
| Stage 2 | 解冻 | 20 | 1e-4 | 全网络细调，保存 `best_model.pth` |
| Stage 3 | 解冻 | 5 | 5e-5 | 在全部 750 样本重训，保存 `final_model.pth` |

Stage 3 消除了 20% 验证集 holdout 对训练数据的浪费。

### 逐类阈值

`05_evaluate.py` 在验证集上逐类搜索使 F1 最大化的阈值，保存为 `best_thresholds.npy`。Kaggle Dice = F1，此操作直接优化提交指标。最优阈值范围 0.50–0.85（高阈值类别往往是高精度类别）。

### TTA（测试时增强）

推理时对原图和水平翻转图分别计算 sigmoid 概率并取均值，无训练开销，稳定提升约 0.5–1 Dice。

### Kaggle 分数说明

完整提交包含 **1500 行**（750 分类 + 750 分割）。Kaggle 在全部行上统一计算一个 Dice。
- 纯分类提交（750 行）：分割行得 0，显示分数约为实际分类 Dice 的一半，**需 ×2** 还原。
- 完整提交显示分数 = 分类与分割综合 Dice，不等于二者之和。

---

## 环境配置

```bash
conda activate biometrics
# PyTorch 2.11.0+cu130，torchvision 0.26，scikit-learn，pandas，PIL，tqdm
# timm 未安装 — 所有骨干网络均通过 torchvision.models 加载
```

Python 路径：`/c/Users/31667/.conda/envs/biometrics/python.exe`（Windows）

所有命令从**项目根目录**运行。

---

## 代码结构

```
Image classification/
  shared.py                    # LABELS、路径、load_module()、AsymmetricLoss、ExperimentConfig
  01_explore_data.py           # 数据统计 + 样本网格图
  02_dataset.py                # VOCDataset、数据变换
  03_model.py                  # MultiLabelClassifier（多骨干网络支持）
  04_train.py                  # 三阶段训练
  05_evaluate.py               # val mAP + 逐类阈值搜索
  06_predict.py                # TTA 推理 → 提交 CSV
  07_adversarial_attack.py     # FGSM / PGD targeted attack（任务 2.3）
  07_ensemble.py               # 概率加权 ensemble
  run_pipeline.py              # 端到端流水线编排
  merge_submission.py          # 合并分类与分割提交文件
  experiments/
    resnet50_224/
    efficientnet_b3_320/
    convnext_tiny_320/
    convnext_small_320/        # ← 当前最佳
    convnext_base_320/
    convnext_large_320/
    convnext_small_320_pad_sampler/
    convnextv2_tiny_320/
    convnextv2_base_320/
    vit_b_16_224/
    vit_l_16_224/
```

输出目录结构：

```
output/image_classification/<experiment>/
  checkpoints/    best_model.pth  final_model.pth
  metrics/        training_history.csv  ap_per_class.csv
                  best_thresholds.npy  evaluation_summary.csv
                  adversarial_aeroplane.csv   ← 仅 convnext_small_320
  figures/        eval_ap_per_class.png
                  adversarial_aeroplane_examples.png  ← 仅 convnext_small_320
  predictions/    test_probabilities_<exp>.csv
                  test_binary_predictions_<exp>.csv
  submissions/    submission_classification_<exp>.csv
```

---

## 运行方式

### 分类流水线

```bash
# 使用最佳分类模型跑完整流水线
python "Image classification/run_pipeline.py" --experiment convnext_small_320

# 跳过数据探索（PNG 已生成）
python "Image classification/run_pipeline.py" --experiment convnext_small_320 --skip-explore

# 仅重新生成提交文件（使用已有 checkpoint）
python "Image classification/run_pipeline.py" --experiment convnext_small_320 --skip-explore --skip-train --ckpt best
```

模型专属入口脚本：

```bash
python "Image classification/experiments/convnext_small_320/train.py"
python "Image classification/experiments/convnext_small_320/evaluate.py"
python "Image classification/experiments/convnext_small_320/predict.py"
```

### 对抗攻击（任务 2.3）

```bash
/c/Users/31667/.conda/envs/biometrics/python.exe \
  "Image classification/07_adversarial_attack.py" \
  --experiment convnext_small_320 \
  --target-class aeroplane \
  --max-samples 64 \
  --epsilon 0.03 --alpha 0.01 --pgd-steps 10 \
  --ckpt final
```

输出：`output/image_classification/convnext_small_320/metrics/adversarial_aeroplane.csv` 和 `figures/adversarial_aeroplane_examples.png`。

### 合并分割提交

```bash
python "Image classification/merge_submission.py" \
  --experiment convnext_small_320 \
  --seg-csv output/submission_exp_v10_segman_b_iter25000.csv
```

---

## Kaggle / Colab Notebooks

| Notebook | 用途 |
|----------|------|
| `kaggle_train_convnext_base_320.ipynb` | ConvNeXt-Base 320 训练（已完成） |
| `kaggle_train_efficientnet_v2_s_320.ipynb` | EfficientNet-V2-S 320 训练 |
| `kaggle_ensemble.ipynb` | Small + V2-S 概率加权融合 |
| `kaggle_kfold_convnext_small_320.ipynb` | 5-fold CV 阈值校准 |
| `colab_train_convnext.ipynb` | Colab 训练，支持 Tiny/Small/Base/Large，含 AMP |

---

## Ensemble 与弱类别恢复

```bash
python "Image classification/07_ensemble.py" --mode val-search --name ensemble_existing
python "Image classification/07_ensemble.py" --mode predict --name ensemble_existing
python "Image classification/merge_submission.py" \
  --clf-csv output/image_classification/ensemble_existing/submissions/submission_classification_ensemble_existing.csv \
  --seg-csv output/submission_exp_v10_segman_b_iter25000.csv
```

使用 `resnet50_224`、`efficientnet_b3_320`、`convnext_small_320` 融合，验证集 mAP **0.9257**，但 Kaggle 实际得分低于 `convnext_small_320` 单模型，因此 ensemble 仅作分析/对照保留。

方法详细说明见 [`Image classification/ensemble_method_explained.md`](<Image classification/ensemble_method_explained.md>)。

---

## 关键设计决策汇总

| 决策 | 内容 | 收益 |
|------|------|------|
| AsymmetricLoss | clip=0.05 + gamma_neg=4，压制 easy 负样本，减少 false-negative 惩罚 | diningtable AP +0.28 |
| 三阶段训练 | 头部预热 → 全网络细调 → 全量数据重训 | 消除 holdout 浪费 |
| 逐类阈值 | 验证集 F1 最大化搜索，直接优化 Kaggle Dice | 可提升 1–2 Dice |
| TTA | 原图 + 水平翻转均值，无训练开销 | +0.5–1 Dice |
| 320×320 输入 | 比 224 更多细节，AsymmetricLoss 帮助控制噪声 | +2–3 Dice |
| ConvNeXt-Small | 50M 参数，平衡表现与过拟合风险 | 优于更大的 Base/Large |
