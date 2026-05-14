# 对抗攻击流程说明

可以把对抗攻击理解成一句话：

**不重新训练原模型，只对输入图片加一点很小的“有方向的噪声”，让模型做出我们想要的错误判断。**

在本项目里，我们做的是：

> 让 `convnext_small_320` 在一些原本不含 `aeroplane` 的图片上，也预测出 `aeroplane`。

## 1. 选定攻击目标

我们定义的是 targeted attack：

```text
目标：让模型更相信图片里有 aeroplane
```

所以这不是随便让模型出错，而是指定模型往 `aeroplane` 这个类别出错。

## 2. 固定原分类模型

加载表现最好的模型：

```text
convnext_small_320 / final_model.pth
```

然后冻结它：

```text
模型参数不更新
只对输入图片求梯度
```

这点很重要。作业 PDF 里也要求原模型保持 unchanged，攻击者只负责产生扰动。

## 3. 选择被攻击图片

我们从 validation split 里选图片，而且只选：

```text
ground truth 里没有 aeroplane 的图片
```

这样攻击成功的定义很清楚：

```text
原本不该有 aeroplane
攻击后模型却认为有 aeroplane
```

## 4. 得到 clean prediction

先把原图喂给模型，看模型对 `aeroplane` 的原始概率。

最终结果里：

```text
clean aeroplane probability mean = 0.074
clean activation rate = 0%
```

这说明这些图原本确实不会被模型判断为 `aeroplane`。

## 5. 定义攻击 loss

我们希望模型输出更像：

```text
aeroplane = 1
其他类别 = 0
```

所以用 binary cross entropy loss 来衡量：

```text
模型输出和目标 aeroplane label 之间的差距
```

然后关键来了：我们对输入图片求梯度。

```text
梯度告诉我们：每个像素往哪个方向改，最能让模型相信 aeroplane 存在
```

## 6. 生成扰动

FGSM 是一步攻击：

```text
算一次梯度
沿着目标方向改一点点图片
```

PGD 是多步攻击：

```text
每次改一点点
改完检查有没有超过 epsilon 限制
重复多次
```

所以 PGD 通常更强。

本实验使用：

```text
epsilon = 0.03
alpha = 0.01
PGD steps = 10
```

## 7. 限制扰动大小

攻击不能随便把图片改坏，否则人一眼就能看出来。

所以我们限制：

```text
扰动不能超过 epsilon
图片像素仍然必须在合法范围内
```

在最终实验中，pixel-space 的平均 `L_inf` 只有：

```text
0.00687 on [0, 1] image scale
```

也就是每个像素最多只变化很小一点。

## 8. 再次预测 adversarial image

把扰动后的图片重新输入模型，看 `aeroplane` 概率有没有上升。

最终结果：

```text
FGSM: 0.074 -> 0.438, success 26.56%
PGD : 0.074 -> 0.983, success 100%
```

意思是：PGD 几乎把所有被攻击图片都推过了 `aeroplane` 的判断阈值。

## 9. 可视化

报告里展示三列：

```text
clean image
perturbation x10
adversarial image
```

中间的 perturbation 放大 10 倍是为了让人能看见扰动结构。真实 adversarial image 看起来应该和 clean image 很像。

## 10. 写反思

最后要解释这个攻击意味着什么。

它说明模型在白盒条件下对小扰动很敏感。但这不等于模型完全没用，因为正常图片上它仍然表现很好。它说明的是：

```text
模型对 worst-case perturbation 不够鲁棒
```

可以提高鲁棒性的方法包括：

```text
adversarial training
更强数据增强
输入检测/预处理
ensemble
同时做 white-box 和 black-box robustness evaluation
```

## 一句话总结

```text
选攻击目标
→ 冻结模型
→ 选目标负样本
→ 算 clean 预测
→ 对输入求梯度
→ 生成 FGSM/PGD 扰动
→ 限制扰动大小
→ 重新预测
→ 统计成功率
→ 可视化和反思
```

