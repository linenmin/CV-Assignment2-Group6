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

## 会话 2026-04-25：分类实验重构

### 目标
将 2.1 分类代码和输出按模型隔离，使 ResNet-50 与 EfficientNet-B3 的训练、评估、预测互不覆盖各自的 checkpoint、指标、图表和提交 CSV。

### 已完成
- [x] 在 `shared.py` 中添加共享 `ExperimentConfig` 对象
- [x] 新增两个模型专属实验目录：
  - `Image classification/experiments/efficientnet_b3_320/`
  - `Image classification/experiments/resnet50_224/`
- [x] 重构训练、评估、预测、流水线、合并脚本，统一支持 `--experiment` 参数
- [x] 输出路径统一路由至 `output/image_classification/<experiment>/`
- [x] 将已有 EfficientNet-B3 制品复制到 `output/image_classification/efficientnet_b3_320/`
- [x] 更新 `README.md` 和 `Image classification/structure.txt`

### 新输出结构
`output/image_classification/<experiment>/checkpoints`、`metrics`、`figures`、`predictions`、`submissions`

### 验证
- 所有修改的 Python 文件通过 `python -m py_compile` 检查
- 在 `biometrics` 环境中确认 pipeline、train、evaluate、predict、merge 及实验封装脚本的帮助输出正常
- 用虚拟输入验证两个 backbone：`resnet50 → torch.Size([1, 20])`，`efficientnet_b3 → torch.Size([1, 20])`

### 遇到的错误
| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| `git status` 报 dubious ownership | 1 | 使用单次 `git -c safe.directory=...` 而不修改全局配置 |
| `pdftotext` 不可用且无本地 PDF 解析器 | 1 | 直接使用代码和已有项目文档，无需提取 PDF |
| 默认 `python` 不含 `torch` | 1 | 改用 `C:\Users\31667\.conda\envs\biometrics\python.exe` |
| `ruff` 未安装在 biometrics 环境 | 1 | 仅依赖 py_compile 和 smoke test |
| `py_compile` 生成 `__pycache__` | 1 | 验证完成后清理缓存文件，保持源码目录干净 |

### 后续更新（2026-04-25）
- [x] 从探索步骤中移除 `.npy → PNG` 转换（PNG 已存在，不重复生成）
- [x] 保留从 `.npy` 读取进行统计分析和样本网格绘图
- [x] 预测和提交 CSV 文件名中加入实验名称，便于在文件夹外识别
- [x] 更新 `kaggle_train.ipynb`，使用实验专属输出目录和文件名
- [x] 修复 `06_predict.py` 和 `kaggle_train.ipynb` 中的预测索引 bug：测试 ID 来自 `test_ds.indices` 而非 DataLoader batch 的第二个返回值
- [x] 验证 `resnet50_224` 预测写入 1500 行提交文件

### 后续更新（2026-04-25）：ConvNeXt-Tiny 实验
- [x] 规划第三个分类实验 `convnext_tiny_320`（CVPR 2022 ConvNeXt "现代 CNN" backbone）
- [x] 选用 torchvision `convnext_tiny`（ImageNet-1K 预训练），320×320 输入，AsymmetricLoss，沿用三阶段训练、per-class 阈值和 TTA
- [x] 完成 `convnext_tiny_320` 的训练、评估、预测和提交文件生成
- [x] 记录本地 ConvNeXt-Tiny 结果：val mAP = **0.8933**（来自 `best_model.pth`）
- [x] 生成 `submission_classification_convnext_tiny_320.csv` 和 `submission_final_convnext_tiny_320.csv`，均为 1500 行
- [x] 记录三个实验的 Kaggle 得分，并说明显示分数需×2 才是真实分类 Dice

### ConvNeXt-Tiny 本地结果汇总
| 项目 | 值 |
|------|---|
| 实验 | `convnext_tiny_320` |
| Backbone | ConvNeXt-Tiny |
| 输入 | 320×320 + TTA + Stage 3 全量重训 |
| 评估 checkpoint | `output/image_classification/convnext_tiny_320/checkpoints/best_model.pth` |
| val mAP | **0.8932642162** |
| 最优 val loss | S2 第 4 epoch **0.0293972875** |
| 最弱 AP 类别 | diningtable 0.5732、sofa 0.7582、bottle 0.7645 |
| 最强 AP 类别 | train 1.0000、boat 1.0000、cow 1.0000 |
| S3 最终训练 loss | 第 5 epoch **0.0075267962** |
| Kaggle 显示分数 | **0.43673** |
| 分类 Dice（×2） | **0.87346** |
| 提交文件 | `submission_classification_convnext_tiny_320.csv`、`submission_final_convnext_tiny_320.csv` |

### Kaggle 分类模型排名
| 排名 | 版本 | 实验文件夹 | Kaggle 显示 | 分类 Dice（×2） |
|------|------|-----------|------------|----------------|
| 1 | v3 | `convnext_tiny_320` | **0.43673** | **0.87346** |
| 2 | v2 | `efficientnet_b3_320` | **0.42813** | **0.85626** |
| 3 | v1.1 | `resnet50_224` | **0.39165** | **0.78330** |
| — | v1.0 | *(重构前 ResNet-50 + NegativeSmoothBCE)* | 0.38084 | 0.76168 |

### 遇到的错误（后续）
| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| `06_predict.py` 中 `ValueError: Shape of passed values is (750, 20), indices imply (20, 20)` | 1 | `test_set.csv` 含 `-1` 占位标签列，导致 DataLoader 返回标签而非 ID；改为使用 `test_ds.indices` 并校验行数 |
| `kaggle_train.ipynb` 存在相同的预测索引 bug | 1 | 同样应用 `test_ds.indices` 修复方案 |
