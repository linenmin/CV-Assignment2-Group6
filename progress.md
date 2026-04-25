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

## Test Results

| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| v1 Kaggle Score | submission_v1 | >0.38 基线 | 0.38084 | 已记录 |
| v1 val mAP | 05_evaluate.py | >0.5 | 0.801 | ✓ |
| ASL forward pass | unit test | tensor scalar | 0.1761 | ✓ |
| EfficientNet-B3 出力 | unit test | (2,20) | (2,20) | ✓ |
| ResNet-50 出力 | unit test | (2,20) | (2,20) | ✓ |
| v2 Kaggle Score | 待提交 | >0.50 | — | pending |

---

## Error Log

| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-04-23 | Windows multiprocessing spawn | 1 | NUM_WORKERS=0 |
| 2026-04-23 | 模块名以数字开头无法直接 import | 1 | importlib.util |
| 2026-04-23 | val mAP 0.80 但 Kaggle Dice 0.38 | 1 | ASL + 320px + TTA + S3 |

---

## 5-Question Reboot Check

| Question | Answer |
|----------|--------|
| Where am I? | Phase 6: v2 重训与提交 |
| Where am I going? | Kaggle Dice > 0.50 |
| What's the goal? | PASCAL VOC 多标签分类 Dice 提升 |
| What have I learned? | v1=0.38；改进: ASL+EffNetB3+320px+TTA+S3全量 |
| What have I done? | v2代码已就绪，待重新训练并提交 |

---

*Update after completing each phase or encountering errors*

---

## Session: 2026-04-25 - Classification Reorganization

### Status
Complete.

### Actions Taken
- Read `.agent/skills/pi-planning-with-files/SKILL.md`.
- Read `task_plan.md`, `findings.md`, and `progress.md`.
- Compared Claude enabled plugins with Codex config; enabled plugin set was
  already aligned.
- Reviewed all Section 2.1 classification scripts.
- Refactored shared configuration into `ExperimentConfig`.
- Added experiment-specific folders for EfficientNet-B3 and ResNet-50.
- Updated train/evaluate/predict/pipeline/merge scripts to accept experiment
  names and use model-specific output folders.
- Updated README and structure documentation.
- Copied existing classification artifacts into the new
  `efficientnet_b3_320` output folder.

### Verification
- `python -m py_compile` passed for edited Python files.
- `C:\Users\31667\.conda\envs\biometrics\python.exe` CLI checks:
  - `run_pipeline.py --help`
  - `04_train.py --help`
  - `05_evaluate.py --help`
  - `06_predict.py --help`
  - experiment wrapper `--help`
  - `merge_submission.py --help`
- Smoke test:
  - ResNet-50 dummy forward: `torch.Size([1, 20])`
  - EfficientNet-B3 dummy forward: `torch.Size([1, 20])`

### Not Run
- Full training was not run because it is GPU/time intensive.
- Full prediction was not run because the request was structure/code cleanup,
  and the smoke tests already verified model loading and CLI routing.

### Notes
- Default `python` lacks `torch`; use the `biometrics` conda environment.
- `ruff` is not installed in the `biometrics` environment.
- Existing git status already showed deleted segmentation submission files and
  untracked `.agent/`; these were not changed by this task.

---

## Session: 2026-04-25 - Explore Script and CSV Naming Follow-up

### Status
Complete.

### Actions Taken
- Removed PNG conversion from `Image classification/01_explore_data.py`.
- Kept shape statistics and sample-grid generation, reading images directly
  from the original `.npy` files.
- Updated submission and prediction CSV paths to include the experiment name:
  - `submission_classification_<experiment>.csv`
  - `submission_final_<experiment>.csv`
  - `test_probabilities_<experiment>.csv`
  - `test_binary_predictions_<experiment>.csv`
- Updated README, experiment README, run pipeline text, and structure docs.
- Renamed the copied EfficientNet submission artifact in the new output folder.

### Verification
- `python -m py_compile` passed for the touched scripts.
- `biometrics` environment help checks passed for `06_predict.py` and
  `merge_submission.py`.

---

## Session: 2026-04-25 - Kaggle Notebook Update

### Status
Complete.

### Actions Taken
- Rewrote `Image classification/kaggle_train.ipynb` as a clean, self-contained
  Kaggle GPU notebook.
- Added experiment-aware Kaggle output folders under
  `/kaggle/working/image_classification/efficientnet_b3_320/`.
- Updated Kaggle output filenames to include the experiment name:
  - `best_model_efficientnet_b3_320.pth`
  - `last_model_efficientnet_b3_320.pth`
  - `final_model_efficientnet_b3_320.pth`
  - `best_thresholds_efficientnet_b3_320.npy`
  - `test_probabilities_efficientnet_b3_320.csv`
  - `test_binary_predictions_efficientnet_b3_320.csv`
  - `submission_classification_efficientnet_b3_320.csv`
- Updated `README.md` and `Image classification/structure.txt` with the Kaggle
  notebook workflow.

### Verification
- Parsed notebook JSON successfully.
- Compiled every code cell with Python `compile(...)`.
