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
├── 03_model.py          # MultiLabelClassifier (efficientnet_b3 / resnet50)
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
| 版本 | 模型 | 损失函数 | 输入尺寸 | val mAP | Kaggle 分类 Dice | Kaggle 完整 Dice |
|------|------|----------|----------|---------|-----------------|-----------------|
| v1（初始）| ResNet-50 | NegativeSmoothBCE | 224 | 0.801 | **0.38084** | — |
| v2（当前）| EfficientNet-B3 | AsymmetricLoss | 320+TTA+S3 | 0.956 | **0.43043** | **0.81610** |

注：0.81610 是含全部 1500 行（分类+分割）的完整提交在 Kaggle 上统一计算的 Dice，**不是**两部分分别得分相加。

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

## Technical Decisions（当前状态）

### 损失函数：AsymmetricLoss（ICCV 2021）
- 替换 `NegativeSmoothBCELoss`
- 参数：`gamma_neg=4, gamma_pos=0, clip=0.05`
- **clip**：当模型对某个负样本预测的 P < 0.05，loss=0，不惩罚潜在false-negative
- **gamma_neg=4**：对easy负样本(p≪0.5)强力下权，让rare正样本主导梯度
- 论文：https://arxiv.org/abs/2009.14119

### 骨干网络：EfficientNet-B3
- 替换 ResNet-50
- ImageNet top-1: 81.6% vs 76.1%（+5.5%）
- 参数量：12M vs 25M（更轻量）
- 通过 `torchvision.models.efficientnet_b3` 加载
- 特征维度：1536（vs ResNet-50的2048）

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

## Key Hyperparameters（v2）
| 参数 | 值 |
|------|-----|
| BACKBONE | efficientnet_b3 |
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

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| Windows multiprocessing spawn | `NUM_WORKERS=0` on `os.name=='nt'` (in shared.py) |
| 模块名以数字开头无法import | `load_module()` via importlib.util |
| VOC标注false-negative噪声 | AsymmetricLoss clip参数消除easy负样本惩罚 |
| val mAP与Kaggle Dice差距大 | 优化per-class阈值 + 换ASL + 更大输入尺寸 |

---

## Resources
- `CV_GA2.pdf`: 作业详细说明（39页）
- `kul-computer-vision-ga-2-2026/`: 数据集目录
- `output/`: 所有输出文件（checkpoints, submissions, 可视化）

## 输出文件状态（v2待生成）
| 文件 | 说明 | 状态 |
|------|------|------|
| output/checkpoints/best_model.pth | val最优模型（S2结束） | 已有v1，需重训v2 |
| output/checkpoints/final_model.pth | 全数据重训模型（S3结束） | 待生成 |
| output/best_thresholds.npy | per-class最优F1阈值 | 已有v1，需重新评估 |
| output/submission_classification.csv | 分类预测RLE | 已有v1，需重新生成 |
| output/eval_ap_per_class.png | per-class AP柱状图 | 已有v1 |

---

## 未完成部分
1. **Section 3: 对抗攻击** — 未实现（FGSM / PGD）
2. **Section 4: 讨论** — notebook未填写
3. **学生姓名** — notebook cell 0 占位符未填写
4. **v2重新训练** — 需重跑04_train.py + 05_evaluate.py + 06_predict.py

---

## 2026-04-25 Findings: Model-Specific Classification Organization

### Code Review Findings
- Previous classification scripts used hard-coded globals (`BACKBONE`,
  `IMG_SIZE`, `CKPT_PATH`, `THRESH_PATH`) and wrote all artifacts into the same
  `output/` paths. This made model comparison fragile because one run could
  overwrite another model's checkpoint, thresholds, plot, or submission CSV.
- The best simplification is not to duplicate full training code per model.
  A shared `ExperimentConfig` keeps backbone, image size, batch size, split seed,
  and output paths together. Model folders are thin entry points.
- `run_pipeline.py --ckpt last` previously pointed at a file name that was not
  consistently saved. Training now writes `last_model.pth` each epoch and
  `final_model.pth` after full-data retraining.
- Prediction now saves both human-readable CSVs and Kaggle submission CSV:
  probabilities, binary predictions, and RLE submission are separated.

### New Model Folders
- `Image classification/experiments/efficientnet_b3_320/`
- `Image classification/experiments/resnet50_224/`

### New Output Convention
For every experiment:

```
output/image_classification/<experiment>/
  checkpoints/
  metrics/
  figures/
  predictions/
  submissions/
```

### Verification Notes
- `py_compile` passed on edited source files.
- In the `biometrics` environment, help output works for shared scripts and
  experiment wrappers.
- Dummy forward passes returned `(1, 20)` for both ResNet-50 and EfficientNet-B3.

### 2026-04-25 Follow-up
- Data exploration should not rewrite PNGs because the dataset has already been
  converted before. `01_explore_data.py` now reads `.npy` files only for shape
  statistics and sample-grid plotting.
- CSV filenames should stay identifiable outside their model folders. Submission
  and prediction CSVs now include the experiment name.

### 2026-04-25 Kaggle Notebook
- `kaggle_train.ipynb` should remain self-contained because Kaggle does not have
  the local numbered Python modules by default.
- The notebook now mirrors the local experiment naming convention and writes all
  artifacts below `/kaggle/working/image_classification/efficientnet_b3_320/`.
- The final Kaggle classification CSV is
  `submission_classification_efficientnet_b3_320.csv`.
