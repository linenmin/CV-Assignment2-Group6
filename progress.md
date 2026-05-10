# 进度记录

---

## 会话：2026-04-23（第 1 次）

### 第 1 阶段：需求与探索
- **状态：** 已完成
- **开始时间：** 2026-04-23 ~19:30
- 已执行操作：
  - 读取 AGENTS.md，了解可用技能
  - 分析项目结构和已有输出文件
  - 阅读所有 Python 源代码和 notebook
  - 确认 submission 格式（1500行 = 750×2）
- 新建文件：
  - task_plan.md, findings.md, progress.md（初始版）

### 第 2~4 阶段：Pipeline 构建 v1
- **状态：** 已完成
- 已执行操作：
  - 创建 01_explore_data.py → 09/explore_data.py
  - 创建 02_dataset.py（VOCDataset + transforms）
  - 创建 03_model.py（ResNet50Classifier）
  - 创建 04_train.py（2阶段训练）
  - 创建 05_evaluate.py（mAP + 阈值优化）
  - 创建 06_predict.py（推理 + RLE）
  - 创建 shared.py（共享常量 + NegativeSmoothBCELoss）
- 关键结果：
  - val mAP = **0.801**
  - Kaggle Score = **0.38084**

---

## 会话：2026-04-23（第 2 次：改进）

### 第 4 阶段：问题分析与改进 v2
- **状态：** 已完成
- **开始时间：** 2026-04-23 ~21:00

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

- 修改文件：
  - shared.py, 03_model.py, 02_dataset.py, 04_train.py, 06_predict.py

---

## 会话：2026-04-23（第 3 次：规划文件更新）

### 阶段：文档整理
- **状态：** 已完成
- 已执行操作：
  - 读取现有 task_plan.md / findings.md / progress.md
  - 用 /pi-planning-with-files 全量更新三个规划文件
  - 反映 v2 改动、Kaggle 实验结果、待做事项

---

---

## Session: 2026-04-24（v2 结果分析 + 文档更新）

### 阶段：v2 Kaggle 验证 + 文档整理
- **状态：** 已完成

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
- **状态：** 已完成

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

## 下一步（下一次会话需做）

1. 若需进一步提升分类分数（目标 0.50+）：
   - K-fold 交叉验证（val 仅 150 张，折叠训练更稳健）
   - MixUp / CutMix 数据增强（小数据集正则化）
   - 更大骨干网络（EfficientNet-B4/B5）
   - 多模型 ensemble（不同 seed 或骨干网络平均概率）
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
- 最优验证 loss：第二阶段第 4 个 epoch **0.0293972875**。
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

## 会话：2026-04-29 - ConvNeXt 默认设置与发现文档同步

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

---

## 会话：2026-05-09 - ConvNeXt 更大变体

### 状态
实现已完成，当时尚未运行训练。

### 已执行操作
- 阅读现有 planning 文件、CLAUDE.md、README.md、实验 README、共享配置和模型实现。
- 比较 Claude 启用插件与 Codex 插件配置。Codex 保留原生 GitHub/Superpowers 等价插件，并禁用重复 Claude 变体，符合仓库策略。
- 在 `biometrics` 环境确认 torchvision 提供 ConvNeXt Tiny/Small/Base/Large，并且每个都有 ImageNet-1K 权重。
- 检查 `CV_GA2.pdf` 第 2.1 节和 Kaggle 实际要求；作业允许使用库内预训练模型做迁移学习/fine-tuning，同时提醒最终 notebook 需要自包含并注意资源消耗。
- 将 `convnext_small`、`convnext_base` 和 `convnext_large` 加入 `Image classification/03_model.py`。
- 新增实验配置：
  - `convnext_small_320`，批大小 8。
  - `convnext_base_320`，批大小 4。
  - `convnext_large_320`，批大小 2。
- 为三个新实验添加 train/evaluate/predict 封装脚本和 README 文件。
- 更新 README.md、CLAUDE.md、实验 README、structure.txt、task_plan.md、findings.md 和 progress.md。

### 验证
- `03_model.py`、`shared.py` 和全部新增实验封装脚本通过 `py_compile`。
- 使用 `pretrained=False` 的 smoke 前向传播均通过：
  - `convnext_small_320`：批大小 8，替换分类头后参数量 49.86M，输出 `(1, 20)`。
  - `convnext_base_320`：批大小 4，替换分类头后参数量 88.10M，输出 `(1, 20)`。
  - `convnext_large_320`：批大小 2，替换分类头后参数量 197.02M，输出 `(1, 20)`。
- `run_pipeline.py --help` 会在 `--experiment` 中列出所有新实验。
- 由于 GPU 时间成本，当时尚未运行完整模型训练/评估/预测。

### 错误记录
| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| PowerShell 拒绝 Bash heredoc 语法（`<<'PY'`） | 1 | 改用 PowerShell here-string 管道传给 Python |
| `Start-Process` 在路径空格处拆开了 `Image classification/run_pipeline.py` | 1 | 在参数字符串内对脚本路径加引号后重启 |

### Stop hook 后续
- 规划 hook 报告整体作业计划未完成，因为第 7 阶段（notebook 集成/对抗攻击/讨论）仍待完成。
- 对当前用户请求而言，直接相关的未完成事项是更大的 ConvNeXt 实验本身。因此先继续训练 `convnext_small_320`，没有切换到无关的 notebook 阶段。

### `convnext_small_320` 已完成
- 完整流水线成功完成。
- 最优 val loss：第二阶段第 2 个 epoch 为 **0.02936561396**。
- val mAP：**0.8995381256**，略高于 Tiny 的 **0.8932642162**。
- 第三阶段全量训练最终 loss：**0.0073612132**。
- 已生成 1500 行提交文件：`output/image_classification/convnext_small_320/submissions/submission_classification_convnext_small_320.csv`。
- 由于 Small 本地 mAP 有提升，继续尝试 `convnext_base_320`。

### `convnext_base_320` 已中断
- Small 本地提升后启动了 `convnext_base_320` 完整流水线。
- 已下载权重并运行到第一阶段第 3 个 epoch：
  - S1 epoch 1：val loss **0.0428**
  - S1 epoch 2：val loss **0.0388**
  - S1 epoch 3：val loss **0.0331**
- 随后用户转向讨论 GPU 效率/容量，因此手动停止 PID 4888 的后台进程，避免意外占用 GPU。
- 没有产出完整 Base 评估或提交。

### GPU 容量检查
- 停止 Base 后当前 GPU 无运行进程，`nvidia-smi` 报告 0 MiB 占用。
- 在 RTX 4060 Laptop GPU 上使用当前实验批大小和 `pretrained=False` 做一步训练 benchmark：
  - `convnext_tiny_320`：batch 16，峰值分配 **3.51 GB**，**1.83 s/step**。
  - `convnext_small_320`：batch 8，峰值分配 **2.88 GB**，**1.40 s/step**。
  - `convnext_base_320`：batch 4，峰值分配 **2.19 GB**，**1.29 s/step**。
  - `convnext_large_320`：batch 2，峰值分配 **3.71 GB**，**1.40 s/step**。
- 解读：所有配置变体都能容纳一步训练，但随着模型变大，批大小必须缩小，所以按样本计的吞吐明显下降。

### Stop hook 后续 2
- Stop hook 再次触发，因为全局作业计划仍有第 7 阶段待完成。
- 重新阅读 `task_plan.md`，并将其当前阶段更新为第 7 阶段。
- 清理 ConvNeXt 更大变体 checklist：`convnext_small_320` 已完成，`convnext_base_320` 已启动并主动停止，Base/Large 后续是否完成改为取决于 GPU 时间收益。
- 当前没有后台训练进程。

### 面向更强 GPU 训练的 Colab notebook
- 创建 `Image classification/colab_train_convnext.ipynb`。
- 该 notebook 面向 Colab 自包含，并支持：
  - `convnext_tiny_320_colab`
  - `convnext_small_320_colab`
  - `convnext_base_320_colab`
  - `convnext_large_320_colab`
- 它会挂载 Google Drive、验证/解压数据集、使用 AMP 混合精度、包含一步显存探测、按相同三阶段协议训练、评估 mAP/阈值，并写出概率、二值预测和 1500 行分类提交。
- 已更新 README、实验 README、structure.txt、task_plan.md 和 findings.md，记录 Colab 工作流。

### 第 7 阶段讨论草稿
- Stop hook 再次触发，因此重新阅读 `task_plan.md`，并检查 `Discussion.md` 和顶层 notebook 结构。
- 将 `Discussion.md` 从笔记式草稿改写成完整讨论章节。
- 新讨论使用项目中的具体证据：
  - 749 张训练图，平均每图 1.43 个标签。
  - 类别不平衡，例如 person 207 vs sheep 27 和 cow 30。
  - 模型对比：ResNet-50、EfficientNet-B3、ConvNeXt-Tiny 和 ConvNeXt-Small。
  - ConvNeXt-Small 本地 mAP **0.8995**，ConvNeXt-Tiny Kaggle 显示 **0.43673** / 换算分类 Dice **0.87346**。
  - Small 的最弱 AP 类别：diningtable、pottedplant、sofa、bottle、sheep。
- 已在 `task_plan.md` 标记第 4 节讨论内容完成，同时保留 notebook 插入、学生姓名和第 3 节对抗攻击为待办。

### Kaggle 结果更新：`convnext_small_320`
- 用户报告最新 Kaggle 测试结果：
  - 完整提交分数：**0.87588**。
  - `convnext_small_320` 分类 Kaggle 显示分数：**0.44905**。
  - 换算后的分类 Dice：**0.89810**。
- 已更新 README.md、CLAUDE.md、findings.md、task_plan.md 和 progress.md。
- 决策：`convnext_small_320` 取代 `convnext_tiny_320`，成为当前最佳分类模型。
- 已将 `Image classification/shared.py` 中的 `DEFAULT_EXPERIMENT` 更新为 `convnext_small_320`。

### Stop hook 后的 Notebook 集成
- Stop hook 再次触发。重新阅读 `task_plan.md` 并检查 notebook section cell。
- 更新 `ga2_group_6.ipynb`：
  - 在第 1 节下插入最终分类结果；
  - 将第 4 节模板讨论替换为完成后的 `Discussion.md` 内容。
- 验证：
  - notebook JSON 可成功加载；
  - 所有 notebook 代码 cell 可按 Python 解析。
- 已在 `task_plan.md` 标记 “在 notebook 中展示/引用最终分类结果” 完成。
- 仍待完成：学生姓名和第 3 节对抗攻击。

### Colab notebook 解释与超参数建议
- 更新 `Image classification/colab_train_convnext.ipynb`，在每个代码 cell 前加入解释性 markdown。
- 新增 “Recommended Settings for 749 Training Images” 小节，覆盖：
  - 本地 RTX 4060 8 GB、Colab T4、L4/V100 和 A100 的批大小范围；
  - 图像尺寸、epoch、学习率、weight decay、验证集划分和 AsymmetricLoss 参数的推荐范围；
  - 当前建议优先跑 `convnext_small_320_colab`，只有在 GPU 预算足够时再尝试 `convnext_base_320_colab`。
- 验证：
  - notebook JSON 可成功加载；
  - 12 个代码 cell 可按 Python 解析；
  - 13 个生成的解释/建议 cell 已存在。

### 第 7 阶段对抗攻击实现
- 新增 `Image classification/07_adversarial_attack.py`。
- 针对当前分类器实现 targeted FGSM 和 PGD。
- 在 `convnext_small_320` 上运行冒烟测试，目标类别为 `aeroplane`，参数为 `max_samples=8`、`epsilon=0.03`、`alpha=0.01`、`pgd_steps=3`。
- 冒烟测试结果：
  - clean 目标概率均值 **0.3257**；
  - FGSM 目标概率均值 **0.7112**；
  - PGD 目标概率均值 **0.9272**；
  - PGD 目标激活率 **1.000**。
- 摘要保存到 `output/image_classification/convnext_small_320/metrics/adversarial_aeroplane.csv`。
- 更新 `ga2_group_6.ipynb` 第 3 节，加入解释、命令和结果表。
- 验证：
  - `07_adversarial_attack.py` 通过 `py_compile`；
  - notebook JSON 可成功加载；
  - notebook 代码 cell 可按 Python 解析。
- 已在 `task_plan.md` 中标记第 3 节完成；剩余全局阻塞是 notebook cell 0 中的学生姓名。

### Colab figures 目录修复
- 调查为什么 `I:\我的云端硬盘\CV-Assignment2\image_classification\convnext_large_320_colab\figures` 为空。
- 根因：`Image classification/colab_train_convnext.ipynb` 创建了 `FIGURES_DIR`，但没有导入 matplotlib 或调用 `savefig`。本地 `05_evaluate.py` 会保存 `eval_ap_per_class.png`，而自包含 Colab notebook 只写了 CSV 指标。
- 更新 Colab notebook，使其保存：
  - `figures/training_history.png`
  - `figures/eval_ap_per_class.png`
- 新增可选重生成 cell，可从已有 `metrics/training_history.csv` 和 `metrics/ap_per_class.csv` 重建图像，无需重新训练。
- 已用现有 Google Drive 指标 CSV 立即生成缺失 PNG：
  - `I:\我的云端硬盘\CV-Assignment2\image_classification\convnext_large_320_colab\figures\training_history.png`
  - `I:\我的云端硬盘\CV-Assignment2\image_classification\convnext_large_320_colab\figures\eval_ap_per_class.png`
- 验证：
  - notebook JSON 可成功加载；
  - 在考虑 Colab `!nvidia-smi` 语法后，所有代码 cell 可按 Python 解析。

### Stop hook 后续：第 7 阶段剩余工作
- Stop hook 报告全局规划任务仍未完成。
- 按要求先更新 progress，再重新阅读 `task_plan.md`，定位剩余第 7 阶段事项。
- 重新阅读 `task_plan.md` 后确认，唯一未完成的第 7 阶段事项是 notebook cell 0 的学生姓名。
- 检查 `ga2_group_6.ipynb` cell 0，发现模板占位符 `name1, name2, ...` 以及旧的 “replace X with group number” 提示。
- 已将 cell 0 更新为 **Group 6**，并移除过时的 notebook 名称 TODO，因为仓库和 notebook 文件名已经标识 Group 6。
- 剩余阻塞：仓库任何位置都没有实际最终组员姓名，因此需要用户输入后才能标记第 7 阶段完成。
- 验证：`ga2_group_6.ipynb` JSON 可成功加载，所有代码 cell 可按 Python 解析。

### Colab AP 图风格对齐
- 用户指出期望的 AP 图风格是本地 `05_evaluate.py` 图：x 轴按 VOC 类别顺序，红色虚线表示 mAP。
- 更新 `Image classification/colab_train_convnext.ipynb`，使 `plot_ap_per_class()` 匹配本地评估风格，而不是之前的横向排序柱状图。
- 修改评估 cell，使其在保存 `ap_per_class.csv` 后立即调用 `plot_ap_per_class(ap_df, AP_PLOT, map_score=map_score)`。
- 修改可选重生成 cell，使其读取 `evaluation_summary.csv`，并用其中保存的 `mAP` 重绘图像。
- 已重新生成 Drive 图像：`I:\我的云端硬盘\CV-Assignment2\image_classification\convnext_large_320_colab\figures\eval_ap_per_class.png`，尺寸 **1200x400**，与用户给的示例布局一致。
- 验证：`Image classification/colab_train_convnext.ipynb` 代码 cell 可成功解析。

### Stop hook 后续：重新检查剩余任务
- Colab AP 图风格修复后 Stop hook 再次触发。
- 按要求先更新 `progress.md`，再重新阅读 `task_plan.md`。
- 预期剩余阻塞仍相同：仓库文件中没有 `ga2_group_6.ipynb` cell 0 所需的最终学生姓名。

### Stop hook 后续：第 7 阶段最终阻塞
- Stop hook `stop:4` 要求更新 `progress.md`，然后重新阅读 `task_plan.md` 并继续剩余阶段。
- 重新阅读 `task_plan.md`；第 7 阶段已完成 6/7。
- 剩余事项：`ga2_group_6.ipynb` cell 0 中的最终学生姓名。
- 下一步：在询问用户前，先检查仓库文件中是否有可靠的组员姓名来源。
- 仓库文本搜索没有找到最终成员名单。
- Git 历史中有可能的作者线索：
  - `Enmin Lin <enmin.lin@student.kuleuven.be>`
  - `121philip <ykx20010330@gmail.com>`
  - `hapgab <zhenyang.linguistics@outlook.com>`
- 这些线索不足以安全替换 notebook TODO 为正式学生姓名；需要用户确认。

### 会话：2026-05-10 - Ensemble 与弱类别恢复实现
- 实现 `Image classification/07_ensemble.py`：
  - `--mode val-search` 用于搜索全局和逐类别 ensemble 权重；
  - `--mode predict` 用于生成测试集概率、二值预测和提交 CSV；
  - 默认跳过本地 checkpoint 不存在的实验。
- 扩展 `ExperimentConfig`，新增 `transform_mode`、`sampler` 和 `early_stop_patience`。
- 在 `02_dataset.py` 中新增方形 padding 图像变换。
- 在 `04_train.py` 中新增弱类别加权采样和第二阶段 early stopping。
- 注册 `convnext_small_320_pad_sampler`，并添加实验入口脚本。
- 已完成验证：
  - 修改后的脚本和新增入口均通过 `py_compile`。
  - `07_ensemble.py --help` 和 `run_pipeline.py --help` 能显示新接口。
  - 16 张验证样本的 ensemble 冒烟测试完成。
  - `convnext_small_320_pad_sampler` 前向/反向冒烟测试完成：输入 `(8, 3, 320, 320)`，输出 `(8, 20)`。
- 已运行完整 existing-model ensemble：
  - 包含 `resnet50_224`、`efficientnet_b3_320`、`convnext_small_320`。
  - 跳过缺失本地 checkpoint 的 `convnext_tiny_320`。
  - 选择模式：逐类别加权。
  - val mAP：**0.925702**。
  - 弱类别平均 F1：**0.790040**。
- 已生成：
  - `output/image_classification/ensemble_existing/submissions/submission_classification_ensemble_existing.csv`
  - `output/image_classification/ensemble_existing/submissions/submission_final_ensemble_existing__seg_submission_exp_v10_segman_b_iter25000.csv`

### 会话：2026-05-10 - 中文文档整理
- 新增 `Image classification/ensemble_method_explained.md`，用中文详细解释 ensemble 方法、`ensemble_smoke` 与 `ensemble_existing` 的区别、输出文件和复现命令。
- 将本次新增/相关的 Markdown 文档改为中文表述，包括根 README 新增段落、实验 README、`findings.md`、`progress.md` 和 `task_plan.md` 的本次记录。
- 记录偏好：以后项目工作记录、方案说明和新增 Markdown 默认使用中文；代码标识符、命令、文件名和实验名保持原文。

### Stop hook 后续：规划状态解析误报
- Stop hook `stop:4` 报告 planning-with-files 任务为 0/7 阶段完成。
- 按要求先更新 `progress.md`，然后重新阅读 `task_plan.md`。
- 重新核对后确认：前 6 个阶段均已完成；第 7 阶段除 notebook cell 0 的正式学生姓名外，其余项目也已完成。
- 判断误报原因：我们按用户要求将 Markdown 记录中文化后，hook 可能无法识别原先的英文 `Status: complete` 状态字段。
- 下一步：在保持中文记录为主的前提下，为 `task_plan.md` 补充机器可读的阶段状态标记，并再次检查是否还有可从仓库可靠推断的学生姓名。
- 已在 `task_plan.md` 中恢复每个阶段的窄范围 `Status: complete/pending` 标记，同时保留中文状态说明，供 hook 解析。
- 重新检查 `ga2_group_6.ipynb` cell 0：当前仍为 `Student names: TODO: add final group member names`。
- 仓库文本搜索没有发现正式组员名单；git 历史只显示 `Enmin Lin`、`121philip`、`hapgab` 等作者线索，不足以作为正式学生姓名填写。
- 当前唯一剩余阻塞仍是用户提供正式组员姓名。

### 会话：2026-05-10 - 评估指标与超参数讲解补充
- 用户希望补充解释 `mAP`、`per-class AP` 等评估指标如何计算，以及训练/评估/ensemble 超参数分别是什么意思。
- 已扩展 `Image classification/ensemble_method_explained.md`：
  - 解释验证集划分、概率矩阵和标签矩阵。
  - 解释 AP、per-class AP、mAP、阈值、F1、TP、FP、FN 的计算方式。
  - 说明为什么 mAP 高不一定等于 Kaggle Dice 高。
  - 补充 `ExperimentConfig`、三阶段训练、AsymmetricLoss、图像增强、弱类别采样、ensemble 参数和 checkpoint 参数。
- 已在 `README.md` 的实验结果段落加入该讲解文档链接，方便后续查阅。

### Stop hook 后续：6/7 阶段完成
- Stop hook `stop:4` 再次触发，这次识别为 6/7 阶段完成，说明 `task_plan.md` 中恢复的 `Status: complete/pending` 标记已被 hook 识别。
- 按要求先更新 `progress.md`，然后重新阅读 `task_plan.md`。
- 预计唯一剩余阶段仍是第 7 阶段中的 notebook cell 0 正式学生姓名。
- 已重新核查 `ga2_group_6.ipynb`、`task_plan.md`、`progress.md` 和 `findings.md` 中的 TODO/姓名相关记录。
- 当前 notebook cell 0 仍为 `Student names: TODO: add final group member names`。
- Git 历史中的作者线索仍只有 `Enmin Lin`、`121philip`、`hapgab` 和 bot 作者，无法等同于正式学生姓名。
- 结论：不能安全自动完成剩余 1/7；需要用户提供正式组员姓名后才能填写 notebook 并将第 7 阶段标记为完成。
