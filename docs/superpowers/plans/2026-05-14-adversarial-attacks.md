# 对抗攻击实现计划

> **给 agentic workers 的说明：** 必须使用子技能：推荐使用 `superpowers:subagent-driven-development`，也可以使用 `superpowers:executing-plans`，按任务逐项执行本计划。步骤使用 checkbox（`- [ ]`）语法来跟踪进度。

**目标：** 完成作业 Sect. 2.3：针对冻结的 `convnext_small_320` 图像分类模型，演示并解释一次 targeted white-box adversarial attack。

**架构：** 保持已经训练好的分类器不变，在验证集图像上用 FGSM 和 PGD 生成有界扰动。报告目标类别概率变化、攻击成功率、扰动大小，并给出 clean image、perturbation、adversarial image 的可视化例子。

**技术栈：** PyTorch、torchvision transforms、pandas、matplotlib，以及现有的 `Image classification` 流水线。

---

## 文件结构

- 修改：`Image classification/07_adversarial_attack.py`
  - 保留现有 targeted FGSM/PGD 实现。
  - 将评估图像过滤为“目标类别不存在”的样本，让任务定义更干净。
  - 将 adversarial images 限制在合法的 normalized pixel bounds 内。
  - 保存视觉例子和更丰富的指标。
- 修改：`ga2_group_6.ipynb`
  - 用最终 Sect. 2.3 的结果和图替换当前 smoke-test 描述。
- 输出：`output/image_classification/convnext_small_320/metrics/adversarial_aeroplane.csv`
  - FGSM 和 PGD 的摘要表。
- 输出：`output/image_classification/convnext_small_320/figures/adversarial_aeroplane_examples.png`
  - Clean image、perturbation、adversarial image 的图像网格。

## Task 1：收紧攻击任务定义

**文件：**
- 修改：`Image classification/07_adversarial_attack.py`

- [ ] **Step 1：只选择目标类别不存在的图像**

添加 CLI option：

```python
parser.add_argument(
    "--target-absent-only",
    action="store_true",
    default=True,
    help="Evaluate only validation images whose ground-truth target label is 0.",
)
```

更新 `load_val_subset`：先选择 `df.iloc[val_idx]`，再过滤 `target_class == 0` 的行，最后取前 `max_samples` 个样本。

- [ ] **Step 2：保持分类器冻结**

加载 checkpoint 后设置：

```python
model.eval()
for parameter in model.parameters():
    parameter.requires_grad_(False)
```

FGSM/PGD 内部仍然会让输入 tensor 使用 `requires_grad_(True)`，所以梯度会流向图像输入，但不会更新模型权重。

- [ ] **Step 3：将 adversarial images 限制到合法 normalized 范围**

添加与 `02_dataset.py` 一致的常量：

```python
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
```

添加 helper：

```python
def clamp_normalized(imgs: torch.Tensor) -> torch.Tensor:
    mean = MEAN.to(imgs.device)
    std = STD.to(imgs.device)
    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std
    return torch.max(torch.min(imgs, upper), lower)
```

在 FGSM 之后，以及 PGD 每一步之后调用这个 helper。

## Task 2：生成更有说服力的定量结果

**文件：**
- 修改：`Image classification/07_adversarial_attack.py`

- [ ] **Step 1：添加指标**

对 clean、FGSM、PGD 的预测，计算：

```python
target_prob_mean
target_activation_rate
attack_success_rate = target_activation_rate_adv
mean_linf = np.abs(adv_imgs - clean_imgs).max(axis=(1, 2, 3)).mean()
mean_l2 = np.sqrt(((adv_imgs - clean_imgs) ** 2).sum(axis=(1, 2, 3))).mean()
```

报告里的核心结论应该是：clean 图像的目标类别 activation 很低，FGSM 会提高它，PGD 会进一步提高它。

- [ ] **Step 2：运行最终实验**

从项目根目录运行：

```bash
C:\Users\31667\.conda\envs\biometrics\python.exe "Image classification/07_adversarial_attack.py" --experiment convnext_small_320 --target-class aeroplane --max-samples 64 --epsilon 0.03 --alpha 0.01 --pgd-steps 10 --ckpt final
```

预期输出：

```text
Experiment: convnext_small_320 | backbone=convnext_small | img_size=320 | batch_size=8
Checkpoint: ...\final_model.pth
Target class: aeroplane
Saved adversarial summary: ...\adversarial_aeroplane.csv
```

## Task 3：保存视觉证据

**文件：**
- 修改：`Image classification/07_adversarial_attack.py`

- [ ] **Step 1：添加 denormalization helper**

```python
def denormalize(imgs: torch.Tensor) -> torch.Tensor:
    mean = MEAN.to(imgs.device)
    std = STD.to(imgs.device)
    return (imgs * std + mean).clamp(0, 1)
```

- [ ] **Step 2：保存图像网格**

创建 `plot_attack_examples(clean, adv, target_probs_clean, target_probs_adv, out_path)`，并保存包含以下列的图像网格：

```text
clean image | perturbation amplified 10x | adversarial image
```

使用 4 个例子。在 subplot title 中写出 clean 和 adversarial 的目标类别概率。

- [ ] **Step 3：确认图像已生成**

运行实验后，确认：

```powershell
Get-ChildItem output\image_classification\convnext_small_320\figures\adversarial_aeroplane_examples.png
```

预期：存在一个非空 PNG 文件。

## Task 4：更新 Notebook 报告

**文件：**
- 修改：`ga2_group_6.ipynb`

- [ ] **Step 1：替换 smoke-test 说明**

替换当前那句“如果加入可视化，分析会更完整”的描述。最终文字应该说明：攻击是在目标类别为负的验证样本上运行的，分类器保持冻结，扰动由 `epsilon=0.03` 约束。

- [ ] **Step 2：插入最终表格**

使用下面文件里的值：

```text
output/image_classification/convnext_small_320/metrics/adversarial_aeroplane.csv
```

报告以下内容：

```text
Attack, target class, epsilon, PGD steps, clean target probability, adversarial target probability, clean activation, adversarial activation, mean perturbation norm.
```

- [ ] **Step 3：插入视觉例子**

引用：

```markdown
![Adversarial aeroplane examples](output/image_classification/convnext_small_320/figures/adversarial_aeroplane_examples.png)
```

- [ ] **Step 4：加入反思**

覆盖 PDF 里的提示问题：

```text
This is a realistic white-box diagnostic but not a realistic remote black-box attack, because it assumes full access to model weights and gradients. It shows the model is locally sensitive to small, targeted perturbations, not that the entire classifier is useless. Robustness could be improved with adversarial training, stronger data augmentation, input preprocessing checks, model ensembles, and evaluation under both white-box and black-box attacks.
```

可以在 notebook 中改写成中文或英文；建议最终报告中用英文，风格与作业 PDF 和 notebook 其他部分保持一致。

## Task 5：验证

**文件：**
- 测试：`Image classification/07_adversarial_attack.py`
- 测试：`ga2_group_6.ipynb`

- [ ] **Step 1：语法检查脚本**

```bash
C:\Users\31667\.conda\envs\biometrics\python.exe -m py_compile "Image classification/07_adversarial_attack.py"
```

预期：没有输出，exit code 为 0。

- [ ] **Step 2：运行快速 smoke test**

```bash
C:\Users\31667\.conda\envs\biometrics\python.exe "Image classification/07_adversarial_attack.py" --experiment convnext_small_320 --target-class aeroplane --max-samples 8 --epsilon 0.03 --alpha 0.01 --pgd-steps 3 --ckpt final
```

预期：CSV 和 PNG 会重新生成。

- [ ] **Step 3：运行最终实验**

```bash
C:\Users\31667\.conda\envs\biometrics\python.exe "Image classification/07_adversarial_attack.py" --experiment convnext_small_320 --target-class aeroplane --max-samples 64 --epsilon 0.03 --alpha 0.01 --pgd-steps 10 --ckpt final
```

预期：PGD 的目标类别 activation 高于 FGSM，并且二者都高于 clean activation。

- [ ] **Step 4：检查 notebook section**

确认 `ga2_group_6.ipynb` Sect. 3 包含：

```text
task definition
losses and regularization
quantitative results
visual examples
reflection on realism and robustness
```

