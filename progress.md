# Progress Log

---

## Session: 2026-04-23 (Session 1)

### Phase 1: Requirements & Discovery
- **Status:** complete
- **Started:** 2026-04-23 ~19:30
- Actions taken:
  - 读取 AGENTS.md，了解可用技能
  - 分析项目结构和已有输出文件
  - 阅读所有 Python 源代码和 notebook
  - 确认 submission 格式（1500行 = 750×2）
- Files created:
  - task_plan.md, findings.md, progress.md（初始版）

### Phase 2~4: Pipeline 构建 v1
- **Status:** complete
- Actions taken:
  - 创建 01_explore_data.py → 09/explore_data.py
  - 创建 02_dataset.py（VOCDataset + transforms）
  - 创建 03_model.py（ResNet50Classifier）
  - 创建 04_train.py（2阶段训练）
  - 创建 05_evaluate.py（mAP + 阈值优化）
  - 创建 06_predict.py（推理 + RLE）
  - 创建 shared.py（共享常量 + NegativeSmoothBCELoss）
- Key results:
  - val mAP = **0.801**
  - Kaggle Score = **0.38084**

---

## Session: 2026-04-23 (Session 2 — 改进)

### Phase 4: 问题分析与改进 v2
- **Status:** complete
- **Started:** 2026-04-23 ~21:00

**根因分析：**
- val mAP(0.80) vs Kaggle Dice(0.38) 差距 = 指标错位 + 噪声损失函数 + 输入分辨率不足
- diningtable AP=0.28 = false-negative 密度最高（桌子大量出现在背景但未标注）

**实施改动：**

1. **shared.py** — 用 `AsymmetricLoss`（ICCV 2021）替换 `NegativeSmoothBCELoss`
   - gamma_neg=4, gamma_pos=0, clip=0.05
   - clip 参数：P < 0.05 的负样本贡献零梯度，解决 false-negative 惩罚

2. **03_model.py** — 新增 `MultiLabelClassifier`，支持 `efficientnet_b3` / `resnet50`
   - EfficientNet-B3: 81.6% ImageNet top-1 vs ResNet-50 76.1%
   - 统一 freeze_backbone / unfreeze_backbone API

3. **02_dataset.py** — 输入尺寸 224→320，新增 RandomErasing(p=0.3)
   - 320×320 保留更多空间细节
   - BATCH_SIZE 对应降至 16

4. **04_train.py** — 新增 Stage 3（全量数据重训，5 epochs，lr=5e-5）
   - 消除 20% val holdout 带来的训练样本浪费
   - 输出 final_model.pth（供 06_predict.py 使用）

5. **06_predict.py** — 新增 TTA（原图 + 水平翻转均值）
   - `_predict_tta()` 函数
   - 自动优先使用 final_model.pth，fallback 到 best_model.pth

- Files modified:
  - shared.py, 03_model.py, 02_dataset.py, 04_train.py, 06_predict.py

---

## Session: 2026-04-23 (Session 3 — 规划文件更新)

### Phase: 文档整理
- **Status:** complete
- Actions taken:
  - 读取现有 task_plan.md / findings.md / progress.md
  - 用 /pi-planning-with-files 全量更新三个规划文件
  - 反映 v2 改动、Kaggle 实验结果、待做事项

---

---

## Session: 2026-04-24（v2 结果分析 + 文档更新）

### 阶段：v2 Kaggle 验证 + 文档整理
- **Status:** complete

**v2 实验结果：**
- val mAP: **0.956**（diningtable: 0.28→0.68，ASL clip 显著改善）
- Kaggle 分类 Dice: **0.43043**（仅提交分类行时的得分）
- Kaggle 完整提交 Dice: **0.81610**（含全部 1500 行时的得分）
  - 注：0.81610 是 Kaggle 在全部行上统一计算的单一 Dice 值，**不是**两部分分别得分之和

**差距分析（val mAP 0.956 vs Kaggle 0.43）：**
- val set 仅 150 张，阈值在其上优化存在过拟合
- mAP 是阈值无关的 AUC 指标；Dice 是阈值敏感指标
- val→test 泛化差距从 v1 的 0.42 扩大到 v2 的 0.53

**本次更新的文件：**
- `CLAUDE.md` — 新增 Experiment Results 表，修正 0.81610 说明，更新 diningtable AP 数字
- `task_plan.md` — Phase 6 标记为 complete，填入实际 Kaggle 得分
- `findings.md` — 更新 Kaggle 实验结果表（v2 行）
- `memory/project_image_classification.md` — 修正 0.81610 的错误描述

---

## Session: 2026-04-29（三实验结果整理）

### 阶段：查看并记录 resnet50_224 / efficientnet_b3_320 / convnext_tiny_320 结果
- **Status:** complete

**数据来源：** 各实验目录下的 `evaluation_summary.csv` + `ap_per_class.csv`（均使用 `best_model.pth` 评估）

| 实验 | val mAP | Kaggle 显示分数 | 分类 Dice (×2) |
|------|---------|----------------|----------------|
| `resnet50_224` (V1) | 0.8175 | 0.39165 | 0.78330 |
| `efficientnet_b3_320` (V2) | 0.8599 | 0.42813 | 0.85626 |
| `convnext_tiny_320` (V3) | **0.8933** | **0.43673** | **0.87346** |

**与旧记录差异：**
- EfficientNet-B3 val mAP 由早期记录的 0.956 修正为 **0.8599**（代码重构后重新评估的实测值）
- ResNet-50 ASL 的 val mAP 此前为缺失值，现补全为 **0.8175**

**本次更新的文件：**
- `CLAUDE.md` — 修正 v2 mAP（0.956→0.8599），补全 v4 mAP（0.8175），加入实验文件夹列
- `findings.md` — 更新 Kaggle 结果表，新增逐类 AP 三实验对比表
- `progress.md` — 本条记录

## Next Steps（下一 Session 需做）

1. 若需进一步提升分类分数（目标 0.50+）：
   - K-fold 交叉验证（val 仅 150 张，折叠训练更稳健）
   - MixUp / CutMix 数据增强（小数据集正则化）
   - 更大 backbone（EfficientNet-B4/B5）
   - 多模型 ensemble（不同 seed 或 backbone 平均概率）
2. Section 3：实现对抗攻击（FGSM / PGD）
3. Section 4：填写 notebook 讨论部分
4. 填写学生姓名（notebook cell 0）

---

## 测试结果

| 测试 | 输入 | 预期 | 实际 | 状态 |
|------|------|------|------|------|
| v1 Kaggle 分数 | submission_v1 | >0.38 基线 | 0.38084 | 已记录 |
| v1 val mAP | 05_evaluate.py | >0.5 | 0.801 | ✓ |
| ASL 前向传播 | 单元测试 | tensor 标量 | 0.1761 | ✓ |
| EfficientNet-B3 输出 | 单元测试 | (2,20) | (2,20) | ✓ |
| ResNet-50 输出 | 单元测试 | (2,20) | (2,20) | ✓ |
| v2 Kaggle 分数 | `efficientnet_b3_320` | >0.50 调整后 | Kaggle 0.42813，调整后 0.85626 | 已记录 |

---

## 错误日志

| 时间戳 | 错误 | 尝试次数 | 解决方案 |
|--------|------|---------|---------|
| 2026-04-23 | Windows 多进程 spawn 问题 | 1 | NUM_WORKERS=0 |
| 2026-04-23 | 模块名以数字开头无法直接 import | 1 | importlib.util |
| 2026-04-23 | val mAP 0.80 但 Kaggle Dice 0.38 | 1 | ASL + 320px + TTA + S3 |

---

## 五问重启检查

| 问题 | 答案 |
|------|------|
| 我在哪里？ | 第 6 阶段：v2 重训与提交 |
| 我要去哪里？ | Kaggle Dice > 0.50 |
| 目标是什么？ | 提升 PASCAL VOC 多标签分类 Dice |
| 我学到了什么？ | v1=0.38；改进: ASL+EffNetB3+320px+TTA+S3全量 |
| 我做了什么？ | v2 代码已就绪，待重新训练并提交 |

---

*Update after completing each phase or encountering errors*

---

## 会话：2026-04-25 - 分类代码重构

### 状态
已完成。

### 已执行操作
- 读取 `.agent/skills/pi-planning-with-files/SKILL.md`。
- 读取 `task_plan.md`、`findings.md` 和 `progress.md`。
- 比对 Claude 已启用插件与 Codex 配置；插件集已对齐，无需修改。
- 审查所有 2.1 分类脚本。
- 将共享配置重构为 `ExperimentConfig`。
- 为 EfficientNet-B3 和 ResNet-50 添加实验专属文件夹。
- 更新 train/evaluate/predict/pipeline/merge 脚本，支持实验名称参数并使用模型专属输出目录。
- 更新 README 和结构文档。
- 将现有分类制品复制到新的 `efficientnet_b3_320` 输出文件夹。

### 验证
- 已修改的 Python 文件通过 `python -m py_compile`。
- 使用 `C:\Users\31667\.conda\envs\biometrics\python.exe` 验证以下 CLI 帮助输出：
  - `run_pipeline.py --help`
  - `04_train.py --help`
  - `05_evaluate.py --help`
  - `06_predict.py --help`
  - 实验封装脚本 `--help`
  - `merge_submission.py --help`
- 冒烟测试：
  - ResNet-50 虚拟前向：`torch.Size([1, 20])`
  - EfficientNet-B3 虚拟前向：`torch.Size([1, 20])`

### 未执行
- 完整训练因 GPU/时间成本未运行。
- 完整预测因任务为结构/代码整理未运行，冒烟测试已验证模型加载与 CLI 路由。

### 备注
- 默认 `python` 不含 `torch`，需使用 `biometrics` conda 环境。
- `ruff` 未安装在 `biometrics` 环境中。
- 现有 git 状态已显示删除的分割提交文件和未跟踪的 `.agent/`，本任务未修改这些内容。

---

## 会话：2026-04-25 - 探索脚本与 CSV 命名跟进

### 状态
已完成。

### 已执行操作
- 从 `Image classification/01_explore_data.py` 中移除 PNG 转换逻辑。
- 保留形状统计和样本网格生成，直接从原始 `.npy` 文件读取图像。
- 更新提交和预测 CSV 路径，加入实验名称：
  - `submission_classification_<experiment>.csv`
  - `submission_final_<experiment>.csv`
  - `test_probabilities_<experiment>.csv`
  - `test_binary_predictions_<experiment>.csv`
- 更新 README、实验 README、流水线说明和结构文档。
- 重命名新输出文件夹中已复制的 EfficientNet 提交制品。

### 验证
- 修改的脚本通过 `python -m py_compile`。
- `biometrics` 环境中 `06_predict.py` 和 `merge_submission.py` 的帮助输出验证通过。

---

## 会话：2026-04-25 - Kaggle Notebook 更新

### 状态
已完成。

### 已执行操作
- 将 `Image classification/kaggle_train.ipynb` 重写为简洁的自包含 Kaggle GPU Notebook。
- 在 `/kaggle/working/image_classification/efficientnet_b3_320/` 下添加实验专属 Kaggle 输出目录。
- 更新 Kaggle 输出文件名，加入实验名称：
  - `best_model_efficientnet_b3_320.pth`
  - `last_model_efficientnet_b3_320.pth`
  - `final_model_efficientnet_b3_320.pth`
  - `best_thresholds_efficientnet_b3_320.npy`
  - `test_probabilities_efficientnet_b3_320.csv`
  - `test_binary_predictions_efficientnet_b3_320.csv`
  - `submission_classification_efficientnet_b3_320.csv`
- 更新 `README.md` 和 `Image classification/structure.txt`，添加 Kaggle Notebook 工作流说明。

### 验证
- Notebook JSON 解析成功。
- 用 Python `compile(...)` 编译所有代码单元格成功。

---

## 会话：2026-04-25 - 预测索引 Bug 修复与 Notebook 审查

### 状态
已完成。

### 已执行操作
- 修复 `Image classification/06_predict.py`，原因为 `resnet50_224` 预测时报错：
  `ValueError: Shape of passed values is (750, 20), indices imply (20, 20)`。
- 根本原因：`test_set.csv` 包含所有 20 个类别列（填充 `-1` 占位符），导致 `VOCDataset` 将测试行视为有标签数据，返回 `(img, label)` 而非 `(img, idx)`。预测循环因此错误地将标签张量当作索引收集。
- 更新预测代码，忽略 batch 第二个返回值，改用 `test_ds.indices` 作为权威测试图像 ID。
- 在构建概率和二值预测 DataFrame 之前，添加预测数量合理性检查。
- 审查 `Image classification/kaggle_train.ipynb`，对 Kaggle 预测单元格应用相同修复。
- 将 Notebook Stage 3 消息从硬编码的 `750 training samples` 改为动态 `len(df)`，因为本地训练 CSV 有 749 行。

### 验证
- 执行：
  `C:\Users\31667\.conda\envs\biometrics\python.exe "Image classification\06_predict.py" --experiment resnet50_224`
- 预测完成并写入：
  - `output/image_classification/resnet50_224/submissions/submission_classification_resnet50_224.csv`
  - `output/image_classification/resnet50_224/predictions/test_probabilities_resnet50_224.csv`
  - `output/image_classification/resnet50_224/predictions/test_binary_predictions_resnet50_224.csv`
- 提交文件行数：1500 行（分类行 + 空分割行）。
- `kaggle_train.ipynb` JSON 解析成功。
- 用 Python `compile(...)` 编译所有 Notebook 代码单元格成功。

### 备注
- 默认 `python` 仍不含 `torch`，模型推理和训练需使用 `biometrics` conda 环境。
- PowerShell profile 执行策略警告在 shell 命令中出现，但不影响项目脚本运行。

---

## 会话：2026-04-25 - 模型专属合并提交

### 状态
已完成。

### 已执行操作
- 扩展 `Image classification/merge_submission.py`，支持将指定分类模型与指定分割 CSV 合并。
- 添加 `--clf-csv` 参数，支持指定特定分类文件，而非始终使用默认实验提交。
- 同时支持含 `*_classification` 行的分类提交 CSV 和含 20 个二值标签列的 `test_binary_predictions_<experiment>.csv`。
- 防止概率 CSV 被静默当作二值提交处理。
- 从分类 CSV 文件夹或文件名推断实验名称，例如 `submission_classification_convnext_tiny_320.csv` 在可能时路由到 ConvNeXt 输出目录。
- 将默认合并文件名改为包含两个来源名称，例如：
  `submission_final_convnext_tiny_320__seg_submission_exp_v10_segman_b_iter25000.csv`。
- 更新 README、实验 README 和结构文档，添加新命名规则和 `--clf-csv` 用法说明。

### 验证
- `merge_submission.py` 通过 `py_compile`。
- `merge_submission.py --help` 显示 `--clf-csv`、`--seg-csv` 和 `--out-csv`。
- 验证使用默认 ConvNeXt 分类提交合并：
  `submission_final_convnext_tiny_320.csv` 共 1500 行。
- 验证从 ResNet 二值预测合并：
  `submission_final_resnet50_224__clf_test_binary_predictions_resnet50_224__seg_submission_exp_v10_segman_b_iter25000.csv` 共 1500 行。
- 验证合并行顺序以 `0_classification`、`0_segmentation`、`1_classification`、`1_segmentation` 开头。

---

## 会话：2026-04-25 - 添加 ConvNeXt-Tiny 实验

### 状态
已完成。

### 已执行操作
- 比对 Claude 已启用插件与 Codex 配置；插件集已对齐，无需修改。
- 确认本地 torchvision 具有 ConvNeXt-Tiny 和 ConvNeXt-Small 工厂函数及 ImageNet-1K 权重枚举。
- 规划 `convnext_tiny_320` 作为第三个分类实验，复用现有共享 train/evaluate/predict 流水线。
- 将 `convnext_tiny` 添加到 `MultiLabelClassifier`。
- 在共享实验配置中注册 `convnext_tiny_320`。
- 添加本地 train/evaluate/predict 封装脚本和实验 README。
- 添加 `kaggle_train_convnext_tiny_320.ipynb`。
- 更新 README、实验文档、结构文档、CLAUDE.md、task_plan.md 和 findings.md。

### 验证
- 修改的脚本和新封装脚本通过 `python -m py_compile`。
- `MultiLabelClassifier(backbone="convnext_tiny", pretrained=False)` 对 `(1, 3, 320, 320)` 虚拟张量返回 `torch.Size([1, 20])`。
- `run_pipeline.py --help` 和新训练封装脚本显示 `convnext_tiny_320` 为有效实验。
- 编译 `kaggle_train_convnext_tiny_320.ipynb` 所有 9 个代码单元格成功。
- 验证后清理生成的 `__pycache__` 变更，使 diff 聚焦于源码、文档和 Notebook 改动。

### 未执行
- 完整 ConvNeXt 训练/评估/预测因 GPU/时间成本未运行。

---

## 会话：2026-04-26 - ConvNeXt-Tiny 结果记录

### 状态
已完成。

### 已执行操作
- 读取 `.agent/skills/pi-planning-with-files/SKILL.md`。
- 比对 Claude 已启用插件与 Codex 配置；插件集已对齐，无需修改。
- 审查 `output/image_classification/convnext_tiny_320/` 下的 ConvNeXt-Tiny 输出制品。
- 更新 `task_plan.md`、`findings.md`、`progress.md` 和 `CLAUDE.md`，记录已完成的本地 ConvNeXt 结果。

### 记录结果
- 实验：`convnext_tiny_320`。
- 评估 checkpoint：`best_model.pth`。
- val mAP：**0.8932642162**。
- 最优验证 loss：Stage 2 第 4 epoch **0.0293972875**。
- Stage 3 最终训练 loss：**0.0075267962**。
- 最弱 AP 类别：diningtable **0.5732**、sofa **0.7582**、bottle **0.7645**、pottedplant **0.7916**、chair **0.8308**。
- 最强 AP 类别：train **1.0000**、boat **1.0000**、cow **1.0000**、cat **0.9922**、aeroplane **0.9909**。
- 生成的 CSV：
  - `submission_classification_convnext_tiny_320.csv`，共 1500 行。
  - `submission_final_convnext_tiny_320.csv`，共 1500 行。
  - `test_probabilities_convnext_tiny_320.csv`，共 750 行。
  - `test_binary_predictions_convnext_tiny_320.csv`，共 750 行。

### 备注
- 本次记录时 ConvNeXt Kaggle 分数尚未获取，在后续分数更新中记录。

---

## 会话：2026-04-26 - Kaggle 分类分数更新

### 状态
已完成。

### 已执行操作
- 记录用户提供的三个分类实验的 Kaggle 显示分数。
- 更新 `task_plan.md`、`findings.md`、`progress.md` 和 `CLAUDE.md`。
- 按照约定同时存储原始 Kaggle 显示分数和调整后分类 Dice 分数（真实分类分数 = Kaggle 显示分数 × 2）。

### 记录分数
| 实验 | Kaggle 显示 | 分类 Dice（×2） |
|------|------------|----------------|
| `convnext_tiny_320` | **0.43673** | **0.87346** |
| `efficientnet_b3_320` | **0.42813** | **0.85626** |
| `resnet50_224` | **0.39165** | **0.78330** |

### 决策
- 按调整后 Kaggle Dice 排名，`convnext_tiny_320` 为当前最佳分类模型。

---

## 会话：2026-04-29 - ConvNeXt 默认与 findings 同步

### 状态
已完成。

### 已执行操作
- 使用 `planning-with-files` 上下文（`task_plan.md`、`findings.md`、`progress.md`）并读取本地技能说明。
- 核对 `Image classification/shared.py`：`convnext_tiny_320` 已映射到 `backbone="convnext_tiny"`。
- 核对 `Image classification/03_model.py`：已通过 `torchvision.models.convnext_tiny` 支持 ConvNeXt-Tiny，池化特征维度为 768。
- 核对 `Image classification/experiments/convnext_tiny_320/{train,evaluate,predict}.py`：三个入口均设置 `EXPERIMENT = "convnext_tiny_320"`。
- 更新 `findings.md`，将当前技术决策和关键超参数从旧 v2 EfficientNet-B3 口径同步为当前最佳 v3 ConvNeXt-Tiny。
- 更新 `Image classification/shared.py`，将 `DEFAULT_EXPERIMENT` 从 `efficientnet_b3_320` 改为 `convnext_tiny_320`。

### 验证
- 搜索了项目中的 `EfficientNet`、`convnext_tiny_320`、`backbone` 和 `BACKBONE` 引用。
- `python -m py_compile "Image classification/shared.py"` 通过；已清理本次生成的 `shared.cpython-312.pyc` 缓存文件。
- 历史结果表中的 EfficientNet-B3 条目保留为 v2 对比记录。
