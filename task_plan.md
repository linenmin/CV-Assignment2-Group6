# 任务计划：CV Assignment 2 图像分类

## 目标
完成 PASCAL VOC 2009 多标签图像分类任务，在 Kaggle Dice 指标上从 0.38 提升到 0.55+，并最终完成完整 notebook 提交。

## 当前阶段
第 7 阶段：Notebook 整合与交付（待完成）

---

## 阶段

### Phase 1: 数据探索与准备 ✓
- [x] 数据集结构分析（01_explore_data.py）
- [x] 全量 .npy → PNG 转换（750 train + 750 test）
- [x] VOCDataset 封装（02_dataset.py）
- **状态：** 已完成
- **Status:** complete

### Phase 2: 模型构建 v1 ✓
- [x] ResNet-50 分类器（03_model.py）
- [x] 共享模块 shared.py（LABELS、路径、NegativeSmoothBCELoss）
- [x] 两阶段训练脚本（04_train.py）
- **状态：** 已完成
- **Status:** complete

### Phase 3: 评估与预测 v1 ✓
- [x] mAP 评估脚本（05_evaluate.py）— val mAP = 0.801
- [x] per-class 最优阈值搜索
- [x] 测试集推理 + RLE 编码（06_predict.py）
- [x] 合并提交文件（merge_submission.py）
- **Kaggle Score v1: 0.38084**
- **状态：** 已完成
- **Status:** complete

### Phase 4: 问题分析与改进 ✓
- [x] 诊断 val mAP(0.80) vs Kaggle Dice(0.38) 差距根因
- [x] 调研 PASCAL VOC 多标签最优方案（ASL, EfficientNet, TTA）
- [x] 实施 4 项改进（见下方技术细节）
- **状态：** 已完成
- **Status:** complete

### Phase 5: 代码重构与共享模块 ✓
- [x] 创建 shared.py（消除重复的 _load_module / LABELS 定义）
- [x] 统一路径常量（DATA_DIR / OUTPUT_DIR）
- [x] Windows NUM_WORKERS=0 修复
- **状态：** 已完成
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
- **状态：** 已完成
- **Status:** complete

### Phase 7: Notebook 整合与交付
- [x] 填写学生姓名（notebook cell 0）：Kaixi Yao, Enmin Lin, Taicheng Liu, Zhenyang Li
- [x] Section 3: 实现对抗攻击（FGSM / PGD）
- [x] Section 4: 填写讨论内容（已在 `Discussion.md` 中完成草稿）
- [x] 在 notebook 中展示/引用分类代码结果
- **状态：** 已完成
- **Status:** complete

---

## 技术细节（当前 v2 配置）

### 核心改动（v1 → v2）

| 改动 | v1 | v2 | 预期收益 |
|------|----|----|---------|
| 损失函数 | NegativeSmoothBCE | **AsymmetricLoss** (γ_neg=4, clip=0.05) | +3~5 Dice |
| 骨干网络 | ResNet-50 (76.1%) | **EfficientNet-B3** (81.6%) | +2~3 Dice |
| 输入尺寸 | 224×224 | **320×320** | +1~2 Dice |
| 推理 | 单次 | **TTA** (原图 + 水平翻转均值) | +0.5~1 Dice |
| 训练数据 | 80% (600样本) | **Stage 3 全量 750 样本重训** | +1~2 Dice |

### AsymmetricLoss 原理
- `clip=0.05`：P(正) < 0.05 的负样本 loss=0，不惩罚潜在 false-negative
- `gamma_neg=4`：easy 负样本梯度被强力压制，rare 正样本主导训练
- 直接解决 diningtable(AP=0.28)、person、pottedplant 的标注噪声问题

### 3阶段训练
| 阶段 | 骨干网络 | Epochs | LR | 目的 |
|------|----------|--------|----|------|
| 第一阶段 | 冻结 | 5 | 1e-3 | 热身分类头 |
| 第二阶段 | 解冻 | 20 | 1e-4 | 全网络细调 |
| 第三阶段 | 解冻 | 5 | 5e-5 | 全量数据，消除 holdout 损失 |

---

## 输出文件
| 文件 | 说明 | 状态 |
|------|------|------|
| output/checkpoints/best_model.pth | val 最优（S2结束） | v1已有 |
| output/checkpoints/final_model.pth | 全数据重训（S3结束） | 待生成 |
| output/best_thresholds.npy | per-class 最优阈值 | v1已有，需更新 |
| output/submission_classification.csv | 分类预测 | v1已有，需更新 |

---

## 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|-------|---------|------------|
| Windows spawn multiprocessing | 1 | Windows 下设置 `NUM_WORKERS=0` |
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
- 用虚拟输入验证两个骨干网络：`resnet50 → torch.Size([1, 20])`，`efficientnet_b3 → torch.Size([1, 20])`

### 遇到的错误
| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| `git status` 报“所有权可疑”错误 | 1 | 使用单次 `git -c safe.directory=...` 而不修改全局配置 |
| `pdftotext` 不可用且无本地 PDF 解析器 | 1 | 直接使用代码和已有项目文档，无需提取 PDF |
| 默认 `python` 不含 `torch` | 1 | 改用 `C:\Users\31667\.conda\envs\biometrics\python.exe` |
| `ruff` 未安装在 biometrics 环境 | 1 | 仅依赖 py_compile 和冒烟测试 |
| `py_compile` 生成 `__pycache__` | 1 | 验证完成后清理缓存文件，保持源码目录干净 |

### 后续更新（2026-04-25）
- [x] 从探索步骤中移除 `.npy → PNG` 转换（PNG 已存在，不重复生成）
- [x] 保留从 `.npy` 读取进行统计分析和样本网格绘图
- [x] 预测和提交 CSV 文件名中加入实验名称，便于在文件夹外识别
- [x] 更新 `kaggle_train.ipynb`，使用实验专属输出目录和文件名
- [x] 修复 `06_predict.py` 和 `kaggle_train.ipynb` 中的预测索引 bug：测试 ID 来自 `test_ds.indices` 而非 DataLoader batch 的第二个返回值
- [x] 验证 `resnet50_224` 预测写入 1500 行提交文件

### 后续更新（2026-04-25）：ConvNeXt-Tiny 实验
- [x] 规划第三个分类实验 `convnext_tiny_320`（CVPR 2022 ConvNeXt "现代 CNN" 骨干网络）
- [x] 选用 torchvision `convnext_tiny`（ImageNet-1K 预训练），320×320 输入，AsymmetricLoss，沿用三阶段训练、per-class 阈值和 TTA
- [x] 完成 `convnext_tiny_320` 的训练、评估、预测和提交文件生成
- [x] 记录本地 ConvNeXt-Tiny 结果：val mAP = **0.8933**（来自 `best_model.pth`）
- [x] 生成 `submission_classification_convnext_tiny_320.csv` 和 `submission_final_convnext_tiny_320.csv`，均为 1500 行
- [x] 记录三个实验的 Kaggle 得分，并说明显示分数需×2 才是真实分类 Dice

### ConvNeXt-Tiny 本地结果汇总
| 项目 | 值 |
|------|---|
| 实验 | `convnext_tiny_320` |
| 骨干网络 | ConvNeXt-Tiny |
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
| 1 | v4 | `convnext_small_320` | **0.44905** | **0.89810** |
| 2 | v3 | `convnext_tiny_320` | **0.43673** | **0.87346** |
| 3 | v2 | `efficientnet_b3_320` | **0.42813** | **0.85626** |
| 4 | v1.1 | `resnet50_224` | **0.39165** | **0.78330** |
| — | v1.0 | *(重构前 ResNet-50 + NegativeSmoothBCE)* | 0.38084 | 0.76168 |

### 遇到的错误（后续）
| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| `06_predict.py` 中 `ValueError: Shape of passed values is (750, 20), indices imply (20, 20)` | 1 | `test_set.csv` 含 `-1` 占位标签列，导致 DataLoader 返回标签而非 ID；改为使用 `test_ds.indices` 并校验行数 |
| `kaggle_train.ipynb` 存在相同的预测索引 bug | 1 | 同样应用 `test_ds.indices` 修复方案 |

---

## 会话 2026-05-09：ConvNeXt 更大变体

### 目标
评估更大的 ConvNeXt 骨干网络能否超过当前最佳 `convnext_tiny_320` 分类 Dice **0.87346**。

### 发现
- 本地 `biometrics` 环境包含 `torch==2.11.0+cu130` 和 `torchvision==0.26.0+cu130`。
- `torchvision.models` 提供 `convnext_tiny`、`convnext_small`、`convnext_base` 和 `convnext_large`，且均有 ImageNet-1K 权重。
- 本地 GPU：NVIDIA GeForce RTX 4060 Laptop GPU，8 GB 显存。
- 本地检查得到的参数量：
  - ConvNeXt-Tiny：28.59M，特征维度 768。
  - ConvNeXt-Small：50.22M，特征维度 768。
  - ConvNeXt-Base：88.59M，特征维度 1024。
  - ConvNeXt-Large：197.77M，特征维度 1536。

### 实施计划
- [x] 在 `Image classification/03_model.py` 中注册 `convnext_small`、`convnext_base` 和 `convnext_large`。
- [x] 新增实验配置：
  - `convnext_small_320`，批大小 8。
  - `convnext_base_320`，批大小 4。
  - `convnext_large_320`，批大小 2。
- [x] 在 `Image classification/experiments/` 下新增轻量 train/evaluate/predict 入口。
- [x] 更新 README/CLAUDE/structure 文档。
- [x] 完整训练 `convnext_small_320`。
- [x] 如果 Small 提升或接近最佳，则启动 `convnext_base_320`。
- [ ] 仅在 GPU 时间值得时完成 `convnext_base_320`，因为吞吐会更低。
- [ ] 将 `convnext_large_320` 视为可选项，因为 750 张训练图和 8 GB 显存会让它更慢、更容易过拟合。

### ConvNeXt-Small 结果
- `convnext_small_320` 已完成完整流水线。
- 最优 checkpoint：第二阶段第 2 个 epoch，val loss **0.02936561396**。
- val mAP：**0.8995381256**，略高于 `convnext_tiny_320` 的 mAP **0.8932642162**。
- AP 最弱类别：diningtable **0.5570**、pottedplant **0.7518**、sofa **0.7872**、bottle **0.7914**、sheep **0.8167**。
- 第三阶段全量训练最终 loss：**0.0073612132**。
- 已生成 1500 行提交文件：
  `output/image_classification/convnext_small_320/submissions/submission_classification_convnext_small_320.csv`.
- Kaggle 结果：完整提交分数 **0.87588**；分类显示分数 **0.44905**；换算后的分类 Dice **0.89810**。这使 `convnext_small_320` 成为当前最佳分类模型。
- `convnext_base_320` 曾启动并运行到第一阶段第 3 个 epoch；后来因用户询问 GPU 效率/容量而主动停止。当前还没有完整的 Base 评估或提交。
- 新增 `Image classification/colab_train_convnext.ipynb` 供更强 Colab GPU 使用。它支持 ConvNeXt Tiny/Small/Base/Large 预设、AMP 混合精度、Google Drive 输出，以及用于批大小检查的 `memory_probe()` cell。

---

## 会话 2026-05-09：第 7 阶段讨论草稿

- Stop hook 要求继续剩余的全局第 7 阶段工作。
- 阅读了 `Discussion.md` 和 `ga2_group_6.ipynb` 结构。
- 将 `Discussion.md` 改写为完整的第 4 节讨论草稿，覆盖：
  - 迁移学习和骨干网络选择；
  - 数据增强和类别不平衡；
  - AsymmetricLoss 与噪声负标签；
  - mAP、Kaggle Dice 和阈值选择；
  - 逐类别失败分析；
  - 真实世界限制、数据集偏差和部署风险。
- 第 7 阶段剩余事项：
  - notebook cell 0 中的学生姓名；
  - 第 3 节对抗攻击实现。

### Notebook 集成更新
- 在 `ga2_group_6.ipynb` 的 `# 1. Image classification` 后插入 “Final classification model and results” markdown cell。
- 将模板 `# 4. Discussion` markdown cell 替换为完成后的 `Discussion.md` 内容。
- Notebook JSON 可成功加载，所有代码 cell 可按 Python 解析。
- 向 `Image classification/colab_train_convnext.ipynb` 添加解释性 markdown 和推荐超参数范围。
- 第 7 阶段剩余阻塞：
  - 学生姓名仍未知；
  - 最终 notebook 元数据/cell 0 仍需要真实学生姓名。

### 对抗攻击更新
- 新增 `Image classification/07_adversarial_attack.py`。
- 针对当前分类器实现 targeted FGSM 和 PGD 攻击。
- 更新 `ga2_group_6.ipynb` 第 3 节，加入：
  - 方法说明；
  - 复现冒烟测试的命令；
  - FGSM/PGD 结果表。
- 在 8 张验证图上以 `aeroplane` 为目标运行冒烟测试：
  - clean 目标概率均值：**0.3257**；
  - FGSM 目标概率均值：**0.7112**；
  - PGD 目标概率均值：**0.9272**；
  - PGD 目标激活率：**1.000**。
- 验证：`07_adversarial_attack.py` 可编译，冒烟测试成功运行，notebook JSON 和代码 cell 均可正确解析。

### 遇到的错误
| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| 在 PowerShell 中探测 torchvision 模型时误用了 Bash heredoc 语法 | 1 | 改用 PowerShell here-string 管道传给 Python |
| Colab `convnext_large_320_colab/figures` 为空 | 1 | notebook 创建了 `FIGURES_DIR` 但没有调用 `savefig`；补充 matplotlib 绘图 helper，并新增可选 CSV 到 PNG 重生成 cell |

### Stop Hook 后续：Notebook Cell 0
- Stop hook 后重新阅读本计划。
- 确认第 7 阶段唯一剩余阻塞是 `ga2_group_6.ipynb` cell 0 中的学生姓名。
- 已将 cell 0 更新为 **Group 6**，并移除关于替换 notebook 名称中 `X` 的旧模板 TODO。
- 剩余阻塞：最终组员姓名未知，且无法从仓库文件中可靠发现。
- 后续检查：仓库文本搜索仍未发现最终成员名单。Git 历史中有作者线索（`Enmin Lin`、`121philip`、`hapgab`），但不是可靠的正式组员名单，因此不应在没有用户确认时填写 notebook TODO。

---

## 会话 2026-05-10：分类 Ensemble 与弱类别恢复

### 目标
在不立即引入 ViT/Swin/DINOv2/CLIP 依赖的前提下提升分类分数，重点处理 `diningtable`、`bottle`、`pottedplant`、`sheep`、`sofa` 等弱类别。

### 已实现
- [x] 新增 `Image classification/07_ensemble.py`。
- [x] 在现有验证集划分上新增全局和逐类别概率权重搜索。
- [x] 新增 ensemble 测试集预测和提交文件生成。
- [x] 扩展实验配置，新增 `transform_mode`、`sampler` 和 `early_stop_patience`。
- [x] 新增保留长宽比的 `square_pad` 图像变换。
- [x] 新增弱类别加权采样。
- [x] 新增第二阶段 early stopping。
- [x] 注册 `convnext_small_320_pad_sampler`，并添加入口脚本和文档。

### 结果
- 完整 ensemble val-search 使用了 `resnet50_224`、`efficientnet_b3_320` 和 `convnext_small_320`。
- `convnext_tiny_320` 因本地 checkpoint 缺失被跳过。
- 选择的 ensemble 模式：逐类别加权。
- val mAP：**0.925702**。
- 弱类别平均 F1：**0.790040**。
- 与 segmentation v10 合并后的完整提交文件：
  `output/image_classification/ensemble_existing/submissions/submission_final_ensemble_existing__seg_submission_exp_v10_segman_b_iter25000.csv`.

### 下一步
- 用户反馈 ensemble 结果不如 `convnext_small_320`，因此不采用 `ensemble_existing` 作为最终默认分类方案。
- 下一步重点转向 Vision Transformer。

---

## 会话 2026-05-11：ViT 实验启动

- 用户确认正式学生姓名：Kaixi Yao, Enmin Lin, Taicheng Liu, Zhenyang Li；Phase 7 已完成。
- 用户反馈 `ensemble_existing` 的结果不如 `convnext_small_320`，因此 ensemble 只作为分析/对照保留。
- 下一步重点转向 Vision Transformer：
  - `vit_b_16_224`：主线实验，torchvision ViT-B/16 ImageNet-1K 预训练权重，224 x 224 输入，batch size 8。
  - `vit_l_16_224`：可选重模型实验，batch size 2，更适合 Colab Pro 或更强 GPU。
- ViT 先使用 224 输入，是为了保持 torchvision 预训练 positional embedding 的原生尺寸；如果 B/16 有希望，再单独实现 320 输入的 positional embedding 插值实验。
---

## Phase 8: From-scratch classification baseline
- **Status:** in_progress

### Goal
Satisfy the classification-task requirement in `CV_GA2.pdf` by adding a model trained from random initialization and comparing it with the current best transfer-learning model, `convnext_small_320`.

### Plan
- [x] Confirm PDF requirement and choose a fair baseline design.
- [x] Add configuration support for pretrained vs scratch initialization.
- [x] Add `convnext_small_320_scratch` experiment config.
- [x] Update training so scratch models do not freeze a random backbone in Stage 1.
- [x] Add train/evaluate/predict wrappers and README for the scratch experiment.
- [x] Smoke test config, CLI registration, and model forward pass.
- [ ] Run full training:
  `C:\Users\31667\.conda\envs\biometrics\python.exe "Image classification/run_pipeline.py" --experiment convnext_small_320_scratch --skip-explore`
- [ ] Run/confirm evaluation outputs:
  `output/image_classification/convnext_small_320_scratch/metrics/evaluation_summary.csv`
- [ ] Generate prediction/submission CSV.
- [ ] Record val mAP and, if submitted, Kaggle score in `README.md`, `findings.md`, and `progress.md`.
- [ ] Add a short report/notebook discussion comparing scratch vs transfer learning.

### Current Implementation
- `Image classification/shared.py`: added `pretrained`, `freeze_backbone_stage1`, and `convnext_small_320_scratch`.
- `Image classification/04_train.py`: uses `config.pretrained`; Stage 1 can be full-network warmup.
- `Image classification/experiments/convnext_small_320_scratch/`: new wrappers and experiment README.

### Verification
- `py_compile` passed.
- `04_train.py --help` and `run_pipeline.py --help` list `convnext_small_320_scratch`.
- Random-initialized ConvNeXt-Small forward pass returns `(1, 20)`.

### Risks
- Scratch ConvNeXt-Small has about 49.9M trainable parameters and may overfit or converge slowly on 750 images.
- If local GPU time is too high, use this implementation as the code baseline and consider a lighter `resnet50_224_scratch` follow-up only if needed.
