# Task Plan: CV Assignment 2 — Image Classification

## Goal
完成 PASCAL VOC 2009 多标签图像分类任务，在 Kaggle Dice 指标上从 0.38 提升到 0.55+，并最终完成完整 notebook 提交。

## Current Phase
Phase 6: v2 重训与提交

---

## Phases

### Phase 1: 数据探索与准备 ✓
- [x] 数据集结构分析（01_explore_data.py）
- [x] 全量 .npy → PNG 转换（750 train + 750 test）
- [x] VOCDataset 封装（02_dataset.py）
- **Status:** complete

### Phase 2: 模型构建 v1 ✓
- [x] ResNet-50 分类器（03_model.py）
- [x] 共享模块 shared.py（LABELS、路径、NegativeSmoothBCELoss）
- [x] 两阶段训练脚本（04_train.py）
- **Status:** complete

### Phase 3: 评估与预测 v1 ✓
- [x] mAP 评估脚本（05_evaluate.py）— val mAP = 0.801
- [x] per-class 最优阈值搜索
- [x] 测试集推理 + RLE 编码（06_predict.py）
- [x] 合并提交文件（merge_submission.py）
- **Kaggle Score v1: 0.38084**
- **Status:** complete

### Phase 4: 问题分析与改进 ✓
- [x] 诊断 val mAP(0.80) vs Kaggle Dice(0.38) 差距根因
- [x] 调研 PASCAL VOC 多标签最优方案（ASL, EfficientNet, TTA）
- [x] 实施 4 项改进（见下方技术细节）
- **Status:** complete

### Phase 5: 代码重构与共享模块 ✓
- [x] 创建 shared.py（消除重复的 _load_module / LABELS 定义）
- [x] 统一路径常量（DATA_DIR / OUTPUT_DIR）
- [x] Windows NUM_WORKERS=0 修复
- **Status:** complete

### Phase 6: v2 重训与提交 ✓
- [x] 重新运行 `python "Image classification/04_train.py"`
- [x] 运行 `python "Image classification/05_evaluate.py"`（获取新阈值）
- [x] 运行 `python "Image classification/06_predict.py"`（TTA推理）
- [x] 上传到 Kaggle，记录新 Dice score
- **val mAP: 0.956**（diningtable: 0.28→0.68）
- **Kaggle 分类 Dice: 0.43043**（仅含分类行的提交）
- **Kaggle 完整提交 Dice: 0.81610**（含全部 1500 行，即分类+分割）
  - 注：0.81610 **不是** 两部分之和，是 Kaggle 在 1500 行统一计算的 Dice
- **Status:** complete

### Phase 7: Notebook 整合与交付
- [ ] 填写学生姓名（notebook cell 0）
- [ ] Section 3: 实现对抗攻击（FGSM / PGD）
- [ ] Section 4: 填写讨论内容
- [ ] 在 notebook 中展示/引用分类代码结果
- **Status:** pending

---

## 技术细节（当前 v2 配置）

### 核心改动（v1 → v2）

| 改动 | v1 | v2 | 预期收益 |
|------|----|----|---------|
| 损失函数 | NegativeSmoothBCE | **AsymmetricLoss** (γ_neg=4, clip=0.05) | +3~5 Dice |
| Backbone | ResNet-50 (76.1%) | **EfficientNet-B3** (81.6%) | +2~3 Dice |
| 输入尺寸 | 224×224 | **320×320** | +1~2 Dice |
| 推理 | 单次 | **TTA** (原图 + 水平翻转均值) | +0.5~1 Dice |
| 训练数据 | 80% (600样本) | **Stage 3 全量 750 样本重训** | +1~2 Dice |

### AsymmetricLoss 原理
- `clip=0.05`：P(正) < 0.05 的负样本 loss=0，不惩罚潜在 false-negative
- `gamma_neg=4`：easy 负样本梯度被强力压制，rare 正样本主导训练
- 直接解决 diningtable(AP=0.28)、person、pottedplant 的标注噪声问题

### 3阶段训练
| 阶段 | Backbone | Epochs | LR | 目的 |
|------|----------|--------|----|------|
| Stage 1 | 冻结 | 5 | 1e-3 | 热身分类头 |
| Stage 2 | 解冻 | 20 | 1e-4 | 全网络细调 |
| Stage 3 | 解冻 | 5 | 5e-5 | 全量数据，消除 holdout 损失 |

---

## 输出文件
| 文件 | 说明 | 状态 |
|------|------|------|
| output/checkpoints/best_model.pth | val 最优（S2结束） | v1已有 |
| output/checkpoints/final_model.pth | 全数据重训（S3结束） | 待生成 |
| output/best_thresholds.npy | per-class 最优阈值 | v1已有，需更新 |
| output/submission_classification.csv | 分类预测 | v1已有，需更新 |

---

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| Windows spawn multiprocessing | 1 | NUM_WORKERS=0 on Windows |
| 模块名以数字开头无法 import | 1 | importlib.util.spec_from_file_location |
| val mAP 高但 Kaggle Dice 低 | 1 | 换 ASL + 更大输入 + 阈值优化 |

---

## Session 2026-04-25: Classification Experiment Reorganization

### Goal
Organize Section 2.1 image-classification code and outputs by model so that
ResNet-50 and EfficientNet-B3 can be trained, evaluated, and predicted without
overwriting each other's checkpoints, metrics, figures, or CSV files.

### Completed
- [x] Added shared `ExperimentConfig` objects in `Image classification/shared.py`.
- [x] Added two model-specific experiment folders:
  - `Image classification/experiments/efficientnet_b3_320/`
  - `Image classification/experiments/resnet50_224/`
- [x] Refactored training, evaluation, prediction, pipeline, and merge scripts
      to accept `--experiment`.
- [x] Routed outputs to `output/image_classification/<experiment>/`.
- [x] Copied existing classification artifacts into
      `output/image_classification/efficientnet_b3_320/`.
- [x] Updated `README.md` and `Image classification/structure.txt`.

### New Output Layout
`output/image_classification/<experiment>/checkpoints`, `metrics`, `figures`,
`predictions`, and `submissions`.

### Verification
- `python -m py_compile ...` passed for all edited Python files.
- `biometrics` environment CLI checks passed for pipeline, train, evaluate,
  predict, merge, and experiment wrappers.
- Smoke-tested both backbones with dummy input:
  `resnet50 -> torch.Size([1, 20])`,
  `efficientnet_b3 -> torch.Size([1, 20])`.

### Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| `git status` dubious ownership | Read status | Used one-command `git -c safe.directory=...` without global config change |
| `pdftotext` unavailable and no local PDF Python parser | PDF extraction | Continued using code and existing project docs; no internet lookup needed |
| Default `python` missing `torch` | CLI smoke test | Used `C:\Users\31667\.conda\envs\biometrics\python.exe` |
| `ruff` not installed in biometrics env | Lint attempt | Recorded as unavailable; relied on py_compile and smoke tests |
| `py_compile` touched `__pycache__` | Verification cleanup | Restored/removed generated cache files so source changes stay clean |

### Follow-up 2026-04-25
- [x] Removed `.npy -> PNG` conversion from the exploration step.
- [x] Kept exploration statistics and sample-grid plotting from `.npy` files.
- [x] Added experiment names to generated prediction/submission CSV filenames.
- [x] Updated `Image classification/kaggle_train.ipynb` for Kaggle GPU training
      with experiment-specific output folders and filenames.
