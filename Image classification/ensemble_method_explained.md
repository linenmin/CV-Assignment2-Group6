# 现有分类模型 Ensemble 方法详解

这份文档解释 `Image classification/07_ensemble.py` 的工作方式，以及为什么它能帮助 `diningtable`、`bottle`、`pottedplant`、`sheep`、`sofa` 这些弱类别。

## 为什么要做 ensemble

当前最好的单模型是 `convnext_small_320`，它的整体验证 mAP 已经接近 0.90，但弱类别仍然拖后腿。问题不是简单的“模型不够大”，而是：

- 训练集只有 749 张图，部分类别正样本很少，例如 `sheep` 只有 27 张、`bottle` 42 张、`pottedplant` 43 张、`sofa` 44 张、`diningtable` 48 张。
- 验证集只有约 150 张图，`diningtable` 在当前 split 中只有 6 个正样本。少数几个样本就能明显改变阈值和 F1。
- 不同模型对不同类别的偏好不一样。例如 EfficientNet-B3 的 `diningtable` AP 比 ConvNeXt-Small 更好，而 ConvNeXt-Small 在 `pottedplant`、`sheep`、`sofa` 上更强。

所以 ensemble 的核心思想是：不要让一个模型独自决定所有类别，而是让每个类别选择更可靠的模型组合。

## 输入是什么

`07_ensemble.py` 使用已经训练好的模型 checkpoint，不重新训练。默认候选模型是：

```text
resnet50_224
efficientnet_b3_320
convnext_tiny_320
convnext_small_320
```

如果某个模型的本地 checkpoint 不存在，脚本默认跳过它。本次实际可用的是：

```text
resnet50_224
efficientnet_b3_320
convnext_small_320
```

`convnext_tiny_320` 被跳过，因为当前本地输出目录里没有它的 `best_model.pth`。

## 第一步：在同一个验证集上收集概率

每个模型都对同一批验证图像输出 20 个类别的概率：

```text
模型 A -> 每张图 20 个概率
模型 B -> 每张图 20 个概率
模型 C -> 每张图 20 个概率
```

注意这里使用的是概率，不是 0/1 标签。比如某张图对 `diningtable` 的预测可能是：

```text
ResNet-50        0.31
EfficientNet-B3  0.82
ConvNeXt-Small   0.55
```

这比直接投票更细，因为模型的置信度也参与计算。

## 评估指标是怎么算出来的

这一节解释你在日志、图表和 CSV 里看到的 `AP`、`mAP`、`F1`、`TP/FP/FN` 是怎么来的。当前项目里，单模型评估由 `05_evaluate.py` 计算，ensemble 评估由 `07_ensemble.py` 计算，两者使用同一套核心逻辑。

### 验证集划分

评估不是在测试集上做的，因为测试集没有真实标签。脚本会先把训练集拆成训练集和验证集：

```python
train_test_split(
    range(len(df)),
    test_size=config.val_split,
    random_state=config.random_seed,
)
```

当前默认：

```text
val_split = 0.2
random_seed = 42
```

也就是说，大约 80% 训练图用于训练，20% 训练图作为验证集。所有模型只要使用同一个实验配置，就会得到同一个验证集划分，因此 AP/mAP 才能比较。

### 概率矩阵和标签矩阵

模型评估时会得到两个矩阵：

```text
probs  shape = (验证图片数, 20)
labels shape = (验证图片数, 20)
```

例如验证集有 150 张图，就大致是：

```text
probs  = 150 x 20 的概率矩阵，范围 0~1
labels = 150 x 20 的真实标签矩阵，只含 0/1
```

`probs[i, j]` 表示第 `i` 张验证图被模型认为包含第 `j` 个类别的概率。`labels[i, j]` 表示这张图真实是否标注了该类别。

模型本身输出的是 logits，不是概率。脚本用 sigmoid 转成概率：

```python
probs = 1 / (1 + np.exp(-logits))
```

### AP：某一个类别的排序质量

`AP` 是 Average Precision，中文可以理解为“某一类的平均精度”。它衡量的是：如果按某个类别的预测概率从高到低排序，正样本是否尽量排在前面。

以 `diningtable` 为例，脚本取出：

```text
y_true  = labels[:, diningtable]
y_score = probs[:, diningtable]
```

然后调用：

```python
average_precision_score(y_true, y_score)
```

直观理解：

- 如果真正有 `diningtable` 的图片大多被排在高概率位置，AP 就高。
- 如果很多没桌子的图片也被排到前面，或者真桌子图片排得很低，AP 就低。
- AP 不需要先选择阈值，它评价的是“排序好不好”。

这就是为什么 AP 和最终 Kaggle Dice 不完全一样：AP 看排序，Kaggle 看最终 0/1 预测。

### per-class AP：20 个类别各自的 AP

`per-class AP` 就是对 20 个类别分别计算 AP。代码逻辑等价于：

```python
for i, label in enumerate(LABELS):
    ap = average_precision_score(labels[:, i], probs[:, i])
```

输出文件：

```text
output/image_classification/<experiment>/metrics/ap_per_class.csv
```

里面每一行对应一个类别，例如：

```text
class,ap,best_threshold
diningtable,0.5570,0.85
pottedplant,0.7518,0.65
...
```

图 `eval_ap_per_class.png` 里的每根柱子就是对应类别的 AP。

### mAP：20 个 AP 的平均值

`mAP` 是 mean Average Precision，也就是 20 个类别 AP 的简单平均：

```python
mAP = mean(ap_1, ap_2, ..., ap_20)
```

代码里是：

```python
map_score = float(ap_df["ap"].mean())
```

举例，如果某模型 20 类 AP 的平均值是 `0.8995`，日志就会显示：

```text
mAP = 0.8995
```

注意：mAP 是验证集指标，不是 Kaggle 最终分数。它很适合比较模型排序能力，但最终提交还要经过阈值二值化。

### 阈值：把概率变成 0/1 预测

Kaggle 需要的是每个类别是否出现，也就是 0/1。概率必须经过阈值：

```text
如果 prob > threshold，预测为 1
否则预测为 0
```

例如 `diningtable` 阈值为 `0.85`：

```text
prob = 0.82 -> 预测 0
prob = 0.91 -> 预测 1
```

这里每个类别有自己的阈值，因为不同类别的概率分布不一样。`person` 可能经常高概率，`diningtable` 可能更保守，统一用 0.5 往往不是最佳。

### F1、TP、FP、FN：阈值后的实际命中情况

阈值确定后，每个类别会得到一串 0/1 预测。脚本会和真实标签比较，得到：

```text
TP = True Positive，真实有，模型也预测有
FP = False Positive，真实没有，模型预测有
FN = False Negative，真实有，模型预测没有
```

F1 的公式是：

```text
F1 = 2 * TP / (2 * TP + FP + FN)
```

也可以理解为 precision 和 recall 的折中：

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
```

例如 ensemble 的 `diningtable`：

```text
TP = 4
FP = 1
FN = 2
F1 = 2 * 4 / (2 * 4 + 1 + 2) = 8 / 11 = 0.7273
```

这就是 `threshold_report.csv` 里 F1、TP、FP、FN 的来源。

### 最优阈值是怎么搜的

脚本不会直接使用 0.5，而是对每个类别尝试一组阈值：

```python
thresholds = np.arange(0.1, 0.91, 0.05)
```

也就是：

```text
0.10, 0.15, 0.20, ..., 0.90
```

每个类别单独选择 F1 最高的阈值：

```python
for class_idx in range(20):
    for threshold in thresholds:
        preds = probs[:, class_idx] > threshold
        f1 = f1_score(labels[:, class_idx], preds)
```

输出文件：

```text
best_thresholds.npy
ap_per_class.csv
threshold_report.csv
```

### 为什么 mAP 高但 Kaggle 不一定高

原因是二者评价的东西不同：

| 指标 | 看什么 | 是否依赖阈值 | 用在哪里 |
|---|---|---:|---|
| AP | 单个类别的概率排序质量 | 否 | `ap_per_class.csv` |
| mAP | 20 个类别 AP 的平均 | 否 | `evaluation_summary.csv` |
| F1 / Dice | 二值预测是否命中 | 是 | 阈值搜索、Kaggle 分类分数 |

所以一个模型可能 mAP 很高，但如果阈值选得不好，Kaggle Dice 仍然低。当前项目必须同时关注：

- mAP：模型有没有把真样本排到前面。
- per-class AP：哪些类别仍然弱。
- per-class threshold：每类概率切在哪里最合适。
- F1 / Dice：最终 0/1 提交质量。

## 第二步：搜索全局权重

全局权重表示所有类别都用同一组模型比例。例如：

```text
0.25 * ResNet-50 + 0.25 * EfficientNet-B3 + 0.50 * ConvNeXt-Small
```

脚本会在一个简单网格里搜索这些权重，例如步长为 `0.25` 时，会尝试：

```text
1.00 / 0.00 / 0.00
0.75 / 0.25 / 0.00
0.50 / 0.25 / 0.25
...
```

每组权重都会生成一份 ensemble 概率，然后计算验证集 mAP。mAP 最高的一组就是最佳全局权重。

本次结果：

```text
全局 ensemble mAP = 0.9113
```

这已经高于 `convnext_small_320` 单模型。

## 第三步：搜索逐类别权重

逐类别权重是这次最重要的部分。

它不要求所有类别共享同一组模型比例，而是给每个类别单独找最优组合。例如：

```text
diningtable = 0.00 * ResNet-50 + 0.75 * EfficientNet-B3 + 0.25 * ConvNeXt-Small
pottedplant = 0.25 * ResNet-50 + 0.00 * EfficientNet-B3 + 0.75 * ConvNeXt-Small
sheep = 0.00 * ResNet-50 + 0.333 * EfficientNet-B3 + 0.667 * ConvNeXt-Small
```

这正好符合我们的观察：`diningtable` 更适合借 EfficientNet-B3 的判断，而 `pottedplant`、`sofa` 更依赖 ConvNeXt-Small。

本次结果：

```text
逐类别 ensemble mAP = 0.925702
```

脚本比较全局权重和逐类别权重后，自动选择了逐类别权重。

## 第四步：重新搜索每个类别的阈值

Kaggle 的分类分数本质上依赖二值预测，所以概率还需要变成 0/1。

单模型里我们已经做过 per-class threshold search。ensemble 后概率分布变了，所以阈值必须重新搜。脚本会对每个类别尝试：

```text
0.10, 0.15, 0.20, ..., 0.90
```

然后选择让该类别 F1 最高的阈值。

本次弱类别结果如下：

| 类别 | AP | 阈值 | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|---:|---:|
| bottle | 0.8080 | 0.75 | 8 | 0 | 4 | 0.8000 |
| diningtable | 0.6870 | 0.85 | 4 | 1 | 2 | 0.7273 |
| pottedplant | 0.7863 | 0.65 | 8 | 3 | 4 | 0.6957 |
| sheep | 1.0000 | 0.60 | 4 | 0 | 0 | 1.0000 |
| sofa | 0.8094 | 0.75 | 8 | 4 | 2 | 0.7273 |

弱类别平均 F1 从 `convnext_small_320` 的 0.7402 提升到 0.7900。

## 超参数说明

这里列出当前分类训练、评估和 ensemble 中最常见的超参数。你可以把它们理解成“不是模型自己学出来的，而是我们在训练前设定的控制旋钮”。

### 实验配置参数

这些参数定义在 `shared.py` 的 `ExperimentConfig` 中。

| 参数 | 当前例子 | 含义 | 调大/调小的影响 |
|---|---:|---|---|
| `name` | `convnext_small_320` | 实验名称，也决定输出目录名。 | 不影响模型性能，只影响文件组织。 |
| `backbone` | `convnext_small` | 使用的骨干网络，用于提取图像特征。 | 更大模型可能更强，但更慢、更容易过拟合。 |
| `img_size` | `320` | 输入图像缩放后的尺寸。 | 更大保留更多细节，但显存和训练时间增加。 |
| `batch_size` | `8` | 训练时每一步送入模型的图片数。 | 更大更稳定但更占显存；太小会让梯度更噪。 |
| `eval_batch_size` | `16` | 验证/推理时每一步送入模型的图片数。 | 只影响速度和显存，不影响结果。 |
| `val_split` | `0.2` | 训练集中留作验证集的比例。 | 更大验证更稳，但训练样本更少。 |
| `random_seed` | `42` | 随机种子，控制训练/验证划分和采样随机性。 | 固定后结果更可复现；换 seed 可做多 seed ensemble。 |
| `transform_mode` | `resize` / `square_pad` | 图像预处理方式。 | `resize` 直接拉伸；`square_pad` 先补成方形，保留长宽比。 |
| `sampler` | `none` / `weak_class_weighted` | 训练样本抽样方式。 | 弱类加权采样会让含弱类图片更常被训练到。 |
| `early_stop_patience` | `4` | 验证集若连续多少个 epoch 不提升就停止。 | 可减少过拟合；太小可能提前停。 |

#### `batch_size` 调大/调小会发生什么

`batch_size` 表示训练时每一步送进模型的图片数量。它不是越大越好，也不是越小越稳，主要是在显存、速度和梯度噪声之间取平衡。

调小 `batch_size` 的影响：
- 显存占用更低，可以跑更大的 backbone 或更大的输入尺寸。例如 `convnextv2_base_320` 比 Tiny 更重，所以默认 batch size 更小。
- 每个 step 的样本更少，梯度更有噪声。对本项目这种只有约 750 张训练图的小数据集，这种噪声有时反而像正则化，不一定是坏事。
- 每个 epoch 的 step 数更多，训练墙钟时间可能变长。
- 如果小到 `2` 或 `4`，loss 曲线可能更抖，学习率不宜再调大。

调大 `batch_size` 的影响：
- 显存占用更高，最直接的风险是 CUDA out of memory。
- 每个 step 的梯度更稳定，loss 曲线通常更平滑。
- 每个 epoch 的 step 数更少，GPU 吞吐通常更好，训练可能更快。
- 在小数据集上，过大的 batch 可能让模型更快记住训练集，泛化不一定提升。

当前建议：
- `convnextv2_tiny_320` 先用 `batch_size=8`。如果显存还有明显余量，可以试 `16`。
- `convnextv2_base_320` 先用 `batch_size=4`。只有确认显存足够时再试 `6` 或 `8`。
- batch size 从 `8` 调到 `16` 时，先保持 `stage2_lr=1e-4` 不变；如果 val loss 稳定，再小幅试 `1.5e-4`。
- batch size 从 `8` 调到 `4` 或 `2` 时，不要提高学习率；必要时把 `stage2_lr` 降到 `5e-5` 到 `8e-5`。

### 三阶段训练参数

当前训练脚本是三阶段：

| 参数 | 默认值 | 所属阶段 | 含义 |
|---|---:|---|---|
| `stage1_epochs` | 5 | 第一阶段 | 冻结骨干网络，只训练分类头多少轮。 |
| `stage1_lr` | `1e-3` | 第一阶段 | 分类头学习率。分类头是新初始化的，所以学习率可以较大。 |
| `stage2_epochs` | 20 | 第二阶段 | 解冻全模型，整体 fine-tuning 多少轮。 |
| `stage2_lr` | `1e-4` | 第二阶段 | 全模型学习率。预训练权重已经有用，所以学习率要小。 |
| `stage3_epochs` | 5 | 第三阶段 | 从最佳 checkpoint 出发，在全部训练图上继续训练多少轮。 |
| `stage3_lr` | `5e-5` | 第三阶段 | 全量数据重训学习率，比第二阶段更小，避免破坏已学到的权重。 |
| `weight_decay` | `1e-4` | 全阶段 | AdamW 的权重衰减，起正则化作用。 |

三阶段的目的：

```text
第一阶段：先让新分类头适应 VOC 20 类。
第二阶段：微调整个预训练模型，让特征更适合当前数据集。
第三阶段：用全部 750 张训练图补回验证集 holdout 的数据损失。
```

### 损失函数参数：AsymmetricLoss

当前使用 `AsymmetricLoss`，参数在 `shared.py` 中：

| 参数 | 默认值 | 含义 | 为什么重要 |
|---|---:|---|---|
| `gamma_neg` | `4` | 对容易的负样本降低权重。 | PASCAL VOC 有很多未标注但实际存在的物体，不能让负样本主导训练。 |
| `gamma_pos` | `0` | 对正样本的 focal 衰减强度。 | 保持正样本梯度，不额外压低稀有类别正样本。 |
| `clip` | `0.05` | 对负样本概率做截断，低概率负样本 loss 变小或归零。 | 缓解 false-negative 标签噪声。 |
| `eps` | `1e-8` | 数值稳定项，避免 `log(0)`。 | 防止计算出现 NaN。 |

直观解释：

- 普通 BCE 会认真惩罚每个负标签。
- 但 VOC 里很多“负标签”其实只是没标出来，不一定真的不存在。
- ASL 会降低这些容易负样本的影响，让模型更关注真正有信息的正样本和困难样本。

### 图像增强参数

训练变换在 `02_dataset.py` 中。常见增强包括：

| 增强 | 当前设置 | 含义 |
|---|---|---|
| `RandomHorizontalFlip` | 默认概率 0.5 | 随机水平翻转，增加左右方向变化。 |
| `RandomRotation(15)` | 最大旋转 15 度 | 增加轻微视角变化。 |
| `ColorJitter(0.3, 0.3, 0.2, 0.05)` | 亮度/对比度/饱和度/色调扰动 | 增加颜色鲁棒性。 |
| `RandomErasing(p=0.3, scale=(0.02, 0.2))` | 30% 概率随机遮挡一块区域 | 让模型不要过度依赖某个局部纹理。 |
| `square_pad` | 新实验可选 | 先补边再缩放，避免把长方形图像硬拉成正方形。 |

### 采样参数：弱类别加权

`convnext_small_320_pad_sampler` 使用：

```text
sampler = "weak_class_weighted"
WEAK_CLASSES = bottle, diningtable, pottedplant, sheep, sofa
```

每张训练图的采样权重是：

```python
weight = 1.0 + 0.75 * weak_count
weight = clip(weight, 1.0, 3.0)
```

其中 `weak_count` 表示这张图含有多少个弱类别。例子：

```text
不含弱类别：weight = 1.00
含 1 个弱类别：weight = 1.75
含 2 个弱类别：weight = 2.50
含 3 个或更多弱类别：weight = 3.00
```

这不是改标签，而是让弱类别相关图片更常出现在训练 batch 中。

### Ensemble 参数

`07_ensemble.py` 的主要参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--mode val-search` | 必填 | 在验证集上搜索模型权重和阈值。 |
| `--mode predict` | 必填 | 使用已经保存的权重和阈值预测测试集。 |
| `--experiments` | 四个默认实验 | 候选模型列表；本地 checkpoint 缺失时默认跳过。 |
| `--ckpt` | `best` | 使用哪个 checkpoint。`best` 用验证集 loss 最佳模型，`final` 用全量重训模型，`auto` 优先 final。 |
| `--name` | `ensemble_selected` | ensemble 输出目录名。 |
| `--grid-step` | `0.25` | 权重搜索网格步长。越小越细，但搜索更慢。 |
| `--val-limit` | 无 | 只取前 N 张验证图，主要用于冒烟测试。正式实验不要设置。 |
| `--no-tta` | 关闭时默认使用 TTA | 是否禁用水平翻转 TTA。正式实验默认使用 TTA。 |

`grid-step` 的含义：

```text
step = 0.25 -> 权重只能是 0, 0.25, 0.50, 0.75, 1.00
step = 0.10 -> 搜索更细，但组合更多
```

本次正式运行用的是：

```bash
python "Image classification/07_ensemble.py" --mode val-search --name ensemble_existing --grid-step 0.25
```

### Checkpoint 参数

训练会保存三种 checkpoint：

| 文件 | 含义 | 适合用途 |
|---|---|---|
| `best_model.pth` | 第一/第二阶段中验证 loss 最低的模型。 | 本地评估、阈值搜索、模型比较。 |
| `last_model.pth` | 最近一个 epoch 的模型。 | 断点续训或排查训练过程。 |
| `final_model.pth` | 从 best 出发，在全部训练集上第三阶段重训后的模型。 | 测试集推理和提交。 |

当前 ensemble 默认使用 `best`，原因是它要在验证集上比较 AP/mAP 和搜索阈值，使用 `best_model.pth` 更可比。

## 输出文件说明

正式 ensemble 名称是 `ensemble_existing`，输出目录为：

```text
output/image_classification/ensemble_existing/
```

关键文件：

```text
metrics/ensemble_weights.json
```

保存模型列表、全局权重、逐类别权重、阈值和验证 mAP。

```text
metrics/ap_per_class.csv
metrics/threshold_report.csv
```

保存每个类别的 AP、阈值、TP、FP、FN、F1。

```text
predictions/test_probabilities_ensemble_existing.csv
predictions/test_binary_predictions_ensemble_existing.csv
```

保存测试集概率和二值预测。

```text
submissions/submission_classification_ensemble_existing.csv
```

只包含分类结果和空 segmentation 占位行。

```text
submissions/submission_final_ensemble_existing__seg_submission_exp_v10_segman_b_iter25000.csv
```

这是分类 ensemble 与现有 v10 分割结果合并后的完整 1500 行提交文件。这个才是建议上传 Kaggle 的文件。

## `ensemble_smoke` 和 `ensemble_existing` 的区别

`ensemble_smoke` 是测试脚本能不能跑通的临时实验：

- 只用了 `resnet50_224` 和 `convnext_small_320`。
- 只取前 16 张验证图。
- 关闭 TTA。
- 权重搜索步长较粗。
- 结果不能代表真实性能，不应该提交。

`ensemble_existing` 是正式实验：

- 使用完整验证集。
- 使用 TTA。
- 使用可用的已训练模型：`resnet50_224`、`efficientnet_b3_320`、`convnext_small_320`。
- 搜索全局权重和逐类别权重。
- 生成正式测试集提交文件。

## 如何重新运行

搜索 ensemble 权重：

```bash
python "Image classification/07_ensemble.py" --mode val-search --name ensemble_existing
```

生成测试集分类提交：

```bash
python "Image classification/07_ensemble.py" --mode predict --name ensemble_existing
```

合并分割结果：

```bash
python "Image classification/merge_submission.py" \
  --clf-csv output/image_classification/ensemble_existing/submissions/submission_classification_ensemble_existing.csv \
  --seg-csv output/submission_exp_v10_segman_b_iter25000.csv \
  --out-csv output/image_classification/ensemble_existing/submissions/submission_final_ensemble_existing__seg_submission_exp_v10_segman_b_iter25000.csv
```

## 风险和注意事项

这个方法仍然使用同一个验证集划分搜索权重和阈值，所以可能对验证集有一定过拟合。它的好处是成本低，不需要重新训练模型，能快速验证不同骨干网络的互补性。

如果 Kaggle 分数提升，就可以把 `ensemble_existing` 作为最终分类提交方案。如果 Kaggle 不提升，下一步再训练 `convnext_small_320_pad_sampler`，用方形填充和弱类采样来改善单模型本身。
