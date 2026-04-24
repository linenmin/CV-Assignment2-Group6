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
- [x] Phase 14: 執行第四輪 OHEM (Online Hard Example Mining) 實驗
- [x] Phase 15: 上傳第四輪提交並紀錄 Kaggle 分數
- [x] Phase 16: 執行第五輪 CE + Dice loss 實驗並紀錄 Kaggle 分數

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

**Currently after Phase 16** - V5 CE + Dice has completed locally and its public Kaggle score has been recorded. The next iteration should prioritize validation/test mismatch analysis, classification merge, or conservative ensembling rather than simply chasing higher local `mIoU`.

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

## Phase 14 Plan

- 目標：引入 OHEM (Online Hard Example Mining) 機制，解決模型對小物件或困難陷阱辨識率不佳的問題。
- 代碼改動：
  - 新增 V4 實驗配置：`configs/experiments/segnext_s_512x512_adamw_poly_v4_ohem.py`
  - 在 `train_cfg` 中覆寫 `sampler` 為 `OHEMPixelSampler`
- 驗證標準：
  - 順利跑通 Smoke Test 並成功生成 `best_mIoU_iter_*.pth`
  - 本地 Validation 中針對特定弱勢類別（如 bicycle, sheep）的 IoU 應有顯著提升
  - 產生 V4 的 `submission.csv` 與訓練曲線圖表

## Phase 14 Result

- V4 OHEM training completed successfully on Colab L4.
  - config: `configs/experiments/segnext_s_512x512_adamw_poly_v4_ohem.py`
  - training command: `python scripts/train.py --config configs/experiments/segnext_s_512x512_adamw_poly_v4_ohem.py --num-workers 4 --batch-size 2 --work-dir outputs/logs/exp_v4_ohem`
  - stop condition: delayed early stopping after the monitored `mIoU` stopped improving for `6` validation records
  - best checkpoint: `outputs/logs/exp_v4_ohem/best_mIoU_iter_13000.pth`
  - best validation metric: `mIoU=63.90`
- Comparison against previous rounds:
  - V1: `62.46 -> 63.90`, `+1.44 mIoU`
  - V2: `60.29 -> 63.90`, `+3.61 mIoU`
  - V3: `62.54 -> 63.90`, `+1.36 mIoU`
- Leaderboard-facing artifacts were generated:
  - prediction directory: `outputs/predictions/exp_v4_ohem_test`
  - submission file: `outputs/submissions/submission_exp_v4_ohem.csv`
  - classification mode: placeholder `0`, pending merge with the classification teammate output
  - analysis directory: `outputs/logs/exp_v4_ohem/analysis`
- Notebook/runtime issue resolved:
  - Colab pandas raised an `AssertionError` when indexing classification columns with the `CLASS_NAMES` tuple.
  - `src/ga2_seg/submission.py` now converts `CLASS_NAMES` to `list(CLASS_NAMES)` before DataFrame selection.

## Phase 15 Plan

- Manually upload or merge `submission_exp_v4_ohem.csv` with the classification teammate output.
- Record the Kaggle public score once available.
- Use V4 as the current segmentation baseline for the next iteration because it is the strongest local validation result so far.

## Phase 15 Result

- V4 OHEM Kaggle public score was recorded:
  - submission file: `outputs/submissions/submission_exp_v4_ohem.csv`
  - public score: `0.35237`
  - classification mode: placeholder `0`
- Comparison against previous public submissions:
  - V1: `0.36281 -> 0.35237`, `-0.01044`
  - V2: `0.3583 -> 0.35237`, `-0.00593`
  - V3: `0.36011 -> 0.35237`, `-0.00774`
- Interpretation:
  - V4 is the strongest local validation run (`mIoU=63.90`) but the weakest public leaderboard submission so far.
  - This suggests the OHEM improvement is validation-specific, changes the mask distribution in a way the hidden test score dislikes, or is being masked by the still-placeholder classification output.
  - V4 should remain useful as an experiment record, but it should not replace V1 as the leaderboard baseline until a merged classification submission or additional validation confirms transfer.

## Phase 16 Result

- V5 CE + Dice training completed successfully on Colab T4.
  - config: `configs/experiments/segnext_s_512x512_adamw_poly_v5_ce_dice.py`
  - method: V1 baseline with `CrossEntropyLoss` plus `DiceLoss(loss_weight=0.5)`
  - best checkpoint: `outputs/logs/exp_v5_ce_dice/best_mIoU_iter_7000.pth`
  - best validation metric: `mIoU=63.62`
  - stop step: `13000`
- Comparison against previous local validation runs:
  - V1: `62.46 -> 63.62`, `+1.16 mIoU`
  - V3: `62.54 -> 63.62`, `+1.08 mIoU`
  - V4: `63.90 -> 63.62`, `-0.28 mIoU`
- Leaderboard-facing artifacts were generated:
  - submission file: `outputs/submissions/submission_exp_v5_ce_dice.csv`
  - analysis directory: `outputs/logs/exp_v5_ce_dice/analysis`
  - classification mode: placeholder `0`
- V5 Kaggle public score was recorded:
  - public score: `0.35553`
  - versus V1: `0.36281 -> 0.35553`, `-0.00728`
  - versus V4: `0.35237 -> 0.35553`, `+0.00316`
- Interpretation:
  - CE + Dice is more stable than V4 OHEM on the public leaderboard, but it still does not beat V1.
  - The repeated local/Kaggle mismatch indicates that future work should focus on better validation, classification merge, or ensembling rather than only increasing local `mIoU`.

## Phase 17 Result

- V6 SegFormer-B2 baseline training completed successfully on Colab L4.
  - config: `configs/experiments/segformer_b2_512x512_adamw_poly_v6.py`
  - notebook: `train_v6_colab.ipynb`
  - method: return to the V1 clean baseline workflow, then replace the SegNeXt-S model family with SegFormer-B2 / MiT-B2
  - training command: `python scripts/train.py --config configs/experiments/segformer_b2_512x512_adamw_poly_v6.py --num-workers 4 --batch-size 2 --work-dir outputs/logs/exp_v6_segformer_b2`
  - stop condition: delayed early stopping after the monitored `mIoU` did not improve for `6` validation records
  - best checkpoint: `outputs/logs/exp_v6_segformer_b2/best_mIoU_iter_14000.pth`
  - best validation metric: `mIoU=63.46`
  - stop step: `20000`
- Comparison against previous local validation runs:
  - V1: `62.46 -> 63.46`, `+1.00 mIoU`
  - V4: `63.90 -> 63.46`, `-0.44 mIoU`
  - V5: `63.62 -> 63.46`, `-0.16 mIoU`
- Leaderboard-facing artifacts were generated:
  - submission file: `outputs/submissions/submission_exp_v6_segformer_b2.csv`
  - submission rows: `1500`
  - analysis directory: `outputs/logs/exp_v6_segformer_b2/analysis`
  - classification mode: placeholder `0`, pending merge with the classification teammate output
- V6 Kaggle public score was recorded:
  - public score: `0.37348`
  - versus V1: `0.36281 -> 0.37348`, `+0.01067`
  - versus V4: `0.35237 -> 0.37348`, `+0.02111`
  - versus V5: `0.35553 -> 0.37348`, `+0.01795`
- Interpretation:
  - V6 is not the best local validation run, but it is the strongest public leaderboard result so far.
  - This supports the hypothesis that changing the model family improved hidden-test generalization more than further tuning loss or hard-pixel sampling on SegNeXt-S.
  - SegFormer-B2 should become the next leaderboard-facing segmentation baseline.

## Phase 18 Plan

- Treat V6 SegFormer-B2 as the current public-score baseline.
- Before the next round, update the three planning files again and commit/push the repository state.
- Candidate next experiments:
  - SegFormer-B2 with stronger regularization only after preserving the clean V6 baseline
  - SegFormer-B3 if compute budget allows
  - test-time augmentation or ensemble with the best SegNeXt-S runs
  - merge segmentation predictions with the classification teammate output before final Kaggle interpretation

## Phase 18 Result

- V7 SegFormer-B3 baseline training completed successfully on Colab L4.
  - config: `configs/experiments/segformer_b3_512x512_adamw_poly_v7.py`
  - notebook: `train_v7_colab.ipynb`
  - method: keep the V6 SegFormer training recipe fixed and scale the model family from MiT-B2 to MiT-B3
  - training command: `python scripts/train.py --config configs/experiments/segformer_b3_512x512_adamw_poly_v7.py --num-workers 4 --batch-size 2 --work-dir outputs/logs/exp_v7_segformer_b3`
  - stop condition: delayed early stopping after the monitored `mIoU` did not improve for `6` validation records
  - best checkpoint: `outputs/logs/exp_v7_segformer_b3/best_mIoU_iter_8000.pth`
  - best validation metric: `mIoU=64.72`
  - stop step: `14000`
- Comparison against previous local validation runs:
  - V1: `62.46 -> 64.72`, `+2.26 mIoU`
  - V6: `63.46 -> 64.72`, `+1.26 mIoU`
  - V4: `63.90 -> 64.72`, `+0.82 mIoU`
- Leaderboard-facing artifacts were generated:
  - submission file: `outputs/submissions/submission_exp_v7_segformer_b3.csv`
  - submission rows: `1500`
  - analysis directory: `outputs/logs/exp_v7_segformer_b3/analysis`
  - classification mode: placeholder `0`
- V7 Kaggle public score was recorded:
  - public score: `0.38084`
  - versus V1: `0.36281 -> 0.38084`, `+0.01803`
  - versus V6: `0.37348 -> 0.38084`, `+0.00736`
  - versus V5: `0.35553 -> 0.38084`, `+0.02531`
- Interpretation:
  - V7 is now the strongest local validation run and the strongest public leaderboard result in the segmentation track.
  - The V6 -> V7 transition is a clean confirmation that the SegFormer family is the most productive direction so far.
  - V7 should replace V6 as the current leaderboard-facing segmentation baseline.

## Phase 19 Plan

- Preserve V7 as the new baseline by updating repository records before the next experiment.
- Candidate next experiments:
  - merge V7 segmentation with the best classification submission before final leaderboard interpretation
  - SegFormer-B4/B5 only if compute budget and Colab stability are acceptable
  - B2/B3 or V6/V7 ensemble and test-time augmentation after the single-model baseline is fully recorded

## Phase 19 Result

- V8 SegFormer-B5 baseline training completed successfully on Colab L4.
  - config: `configs/experiments/segformer_b5_512x512_adamw_poly_v8.py`
  - notebook: `train_v8_colab.ipynb`
  - method: keep the V7 SegFormer recipe fixed and scale the backbone from MiT-B3 to MiT-B5
  - training command: `python scripts/train.py --config configs/experiments/segformer_b5_512x512_adamw_poly_v8.py --num-workers 4 --batch-size 2 --work-dir outputs/logs/exp_v8_segformer_b5`
  - stop condition: delayed early stopping after the monitored `mIoU` did not improve for `6` validation records
  - best checkpoint: `outputs/logs/exp_v8_segformer_b5/best_mIoU_iter_11000.pth`
  - best validation metric: `mIoU=68.97`
  - stop step: `17000`
- Comparison against previous local validation runs:
  - V1: `62.46 -> 68.97`, `+6.51 mIoU`
  - V7: `64.72 -> 68.97`, `+4.25 mIoU`
  - V6: `63.46 -> 68.97`, `+5.51 mIoU`
- Leaderboard-facing artifacts were generated:
  - submission file: `outputs/submissions/submission_exp_v8_segformer_b5.csv`
  - submission rows: `1500`
  - analysis directory: `outputs/logs/exp_v8_segformer_b5/analysis`
  - classification mode: placeholder `0`
- V8 Kaggle public score was recorded:
  - public score: `0.38567`
  - versus V1: `0.36281 -> 0.38567`, `+0.02286`
  - versus V7: `0.38084 -> 0.38567`, `+0.00483`
  - versus V6: `0.37348 -> 0.38567`, `+0.01219`
- Interpretation:
  - V8 is now the strongest local validation run and the strongest public leaderboard result in the segmentation track.
  - The SegFormer scaling trend remains positive from B2 -> B3 -> B5, so the core strategy is still working.
  - V8 should replace V7 as the current leaderboard-facing segmentation baseline.

## Phase 20 Plan

- Preserve V8 as the new single-model baseline by updating repository records before any further experiments.
- Candidate next experiments:
  - merge V8 segmentation with the best classification submission and measure the combined Kaggle gain
  - lightweight B3/B5 ensemble or test-time augmentation before attempting even heavier single models
  - only consider further model scaling if the extra compute still gives a worthwhile public-score return

## Phase 20 Result

- V6/V7/V8 public scores were added to `kaggle_scores.md` so the leaderboard tracker is now aligned with `task_plan.md`, `progress.md`, and `findings.md`.
- A low-cost V9 hard-vote ensemble candidate was generated from the existing SegFormer submission files:
  - inputs:
    - `outputs/submissions/submission_exp_v6_segformer_b2.csv`
    - `outputs/submissions/submission_exp_v7_segformer_b3.csv`
    - `outputs/submissions/submission_exp_v8_segformer_b5.csv`
  - output: `outputs/submissions/submission_exp_v9_segformer_vote_v6_v7_v8.csv`
  - tie-break baseline: V8 SegFormer-B5
  - mean changed pixels versus V8: `2.14%`
  - mean pairwise disagreement across V6/V7/V8: `5.17%`
- Interpretation:
  - V9 is a conservative ensemble candidate because it only changes a small fraction of V8 pixels.
  - It is worth a manual Kaggle upload because it costs no additional training and tests whether the SegFormer family disagreement contains useful corrections.
  - If V9 does not beat V8, the next expensive training experiment should remain a model-family upgrade such as SegMAN-B/L rather than another local loss or sampler tweak.
