# Task Plan: Semantic Segmentation V1

## Goal

Build a maintainable `Semantic segmentation` project around `MMSegmentation + SegNeXt-S` with validated data analysis, preprocessing visualization, pretrained initialization, and iterative leaderboard experiments.

## Phases

- [x] Phase 1: Research candidate models and verify official pretrained support
- [x] Phase 2: Inspect dataset structure and identify preprocessing risks
- [x] Phase 3: Review design and directory structure
- [x] Phase 4: Implement data analysis and visualization scripts
- [x] Phase 5: Implement custom dataset conversion / config / training entrypoints
- [x] Phase 6: Run baseline training and validate outputs
- [x] Phase 7: Commit, push branch, and notify via Discord
- [x] Phase 8: Launch the first full ADE20K-initialized training run and capture the result
- [x] Phase 9: Track Kaggle leaderboard submissions and iterate from the first baseline
- [x] Phase 10: Execute the second-round rare-focus sampling and cropping experiment
- [x] Phase 11: 执行第三轮“区域再平衡辅助分支”实验
- [x] Phase 12: 上传第三轮提交并记录 Kaggle 分数
- [x] Phase 13: 清理历史冗余权重，仅保留 `best` 与 `last`

## Key Questions

1. Which first-pass model offers the best tradeoff between recency, stability, and maintainability?
2. Which preprocessing choices are required by the dataset shape and class imbalance?
3. Which pretrained checkpoints are allowed under the assignment constraints?
4. Which initialization should be the default first-run leaderboard baseline?

## Decisions Made

- Model family: `SegNeXt-S`
  Rationale: recent enough, strong segmentation backbone, lower engineering risk than heavier alternatives.
- Framework: `MMSegmentation`
  Rationale: mature config system, standard training scripts, easier long-term iteration.
- Main pretraining strategy: `ADE20K` segmentation checkpoint initialization
  Rationale: official checkpoint is not trained on `PASCAL VOC`, aligns with the current reading of the assignment restriction, and should transfer better than backbone-only initialization on a 749-image training set.
- Control experiment: `ImageNet-1K` pretrained MSCAN-S backbone only
  Rationale: keeps a compliant lower-assumption baseline for comparison if ADE20K initialization underperforms or raises policy questions later.
- Training control: best-checkpoint selection plus conservative early stopping
  Rationale: segmentation curves often improve late; aggressive stopping is counterproductive.
  Default policy: begin checking after `5` validations, then stop after `6` non-improving `mIoU` validations with `min_delta=0.1`.
- Git workflow: work on branch `segmentation`
  Rationale: isolates semantic segmentation work from the rest of the repository.

## Errors Encountered

- Base Python environment has a broken `numpy/pandas` binary combination.
  Resolution: use the dedicated `seg_gpu_env` instead of the base environment.
- Windows training hit three framework/runtime issues:
  - missing `mmcv._ext` in the old environment
  - `OpenMP` duplicate runtime crashes in plotting
  - `mmengine` locale decode failure while collecting compiler info
  Resolution: use `seg_gpu_env` with full `mmcv`, add a Windows compatibility layer, and patch `mmengine collect_env` narrowly in project runtime code.
- Validation originally resized masks before evaluation, which broke `IoUMetric` shape alignment.
  Resolution: keep val/test samples at original resolution.
- Resource probing before the first full run showed only `2.29 GB` free RAM on the host.
  Resolution: use a conservative launch profile for the first long run with `num_workers=0`, dedicated `work_dir`, and a short pretrained sanity check before the main run.

## Status

**Currently in Phase 13** - 第三轮 Kaggle 分数已经记录，当前进入磁盘清理阶段：删除中间 checkpoint 和无用烟雾/预热权重，只保留主实验的 `best` 与 `last`。

## Current Diagnosis

- 从接近最优的验证轮次看，当前最差的类别持续集中在 `bicycle`、`chair`、`cow`、`sheep`、`pottedplant`。
  这个问题是通过验证日志中的分类别交并比表定位出来的：这些类别在多次验证中长期低于主流类别，且波动明显更大。
- 这说明当前主瓶颈更像是低频类别或小目标类别学习不足，而不是主干网络整体失效。
- 第二轮“弱类重采样 + 聚焦裁剪”没有超过第一轮基线。
  最佳 `mIoU` 从 V1 的 `62.46` 降到 V2 的 `60.29`，说明输入侧强行改分布的副作用已经超过收益。
- 第三轮已经把“长尾修正”从输入采样侧转移到了训练目标侧。
  当前结果显示：区域再平衡辅助分支至少没有像第二轮那样明显破坏整体分布，本地最佳 `mIoU` 提升到 `62.54`，略高于第一轮 `62.46`，明显高于第二轮 `60.29`。
- 训练日志确认辅助损失持续参与优化。
  在第三轮正式训练中，`decode.loss_region_rebalance` 在多个训练步上持续非零，说明区域再平衡分支不是空转配置，而是真正参与了梯度更新。

## Phase 11 Plan

- 目标：在保持第一轮数据管线不变的前提下，只引入训练期区域再平衡辅助分支，做一次高层策略替换实验。
- 代码改动：
  - 新增区域再平衡辅助头
  - 新增第三轮实验配置
  - 将训练集类别频次接入辅助分支
- 验证标准：
  - 配置与单元测试通过
  - 烟雾训练可跑通，并在日志中出现区域再平衡辅助损失
  - 完成一轮正式训练，并与 V1/V2 比较 `mIoU` 与 Kaggle 分数

## Phase 11 Result

- 正式训练已完成：
  - 最佳 checkpoint：`outputs/logs/exp_v3_region_rebalance/best_mIoU_iter_5000.pth`
  - 最佳验证 `mIoU`：`62.54`
  - 早停轮次：`11000`
- 与前两轮对比：
  - 相比 V1：`62.46 -> 62.54`，本地 `mIoU` 小幅上升 `0.08`
  - 相比 V2：`60.29 -> 62.54`，本地 `mIoU` 明显回升 `2.25`
- 提交文件已导出：
  - `outputs/submissions/submission_exp_v3_region_rebalance.csv`

## Phase 12 Result

- 第三轮 Kaggle 分数：`0.36011`
- 与前两轮对比：
  - 相比 V1：`0.36281 -> 0.36011`，下降 `0.00270`
  - 相比 V2：`0.3583 -> 0.36011`，回升 `0.00181`
- 当前结论：
  - 第三轮确实修复了第二轮的明显退化
  - 但本地 `mIoU` 的微小提升没有稳定转化为 Kaggle 提升
  - 因此区域再平衡方向“可行但收益有限”，还不足以单独带来榜单突破

## Phase 13 Plan

- 保留三轮主实验的：
  - `best_mIoU_iter_*.pth`
  - 最后一轮 `iter_*.pth`
- 删除三轮主实验的其余中间 `iter_*.pth`
- 删除烟雾训练与预热训练目录中的权重文件
- 更新文档记录清理前后的空间变化
- 将“只保留 `best` 和 `last`”的逻辑集成到训练脚本默认行为中
- 更新 README，补充训练分析图与 `submission.csv` 导出流程

## Phase 13 Result

- 已保留的主实验权重：
  - V1：`best_mIoU_iter_8000.pth`，`iter_14000.pth`
  - V2：`best_mIoU_iter_4000.pth`，`iter_11000.pth`
  - V3：`best_mIoU_iter_5000.pth`，`iter_11000.pth`
- 已删除内容：
  - 三轮主实验的所有中间 `iter_*.pth`
  - `exp_v1_ade20k_sanity`
  - `exp_v1_ade20k_sanity_earlystop`
  - `smoke_train`
  - `smoke_v2_rare_focus`
  - `smoke_v3_region_rebalance`
- 空间变化：
  - 清理前权重总量：`6.938 GB`
  - 清理后权重总量：`0.629 GB`
  - 释放空间：`6.309 GB`
- 训练脚本默认行为已更新：
  - `scripts/train.py` 训练完成后默认执行 checkpoint 清理
  - 默认策略为仅保留 `best` 与 `last`
  - 如需保留全部中间权重，可显式传入 `--checkpoint-retention all`
