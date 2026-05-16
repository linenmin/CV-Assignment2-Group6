# 語意分割對抗攻擊 README

本文件是語意分割 adversarial attack 的入口說明，給整合報告的隊友快速了解：我們做了什麼攻擊、結果如何、為什麼最後 V26 選用兩種攻擊方式。

重點聚焦在 **PGD-family white-box attacks**，也就是 V10、V22、V23、V24、V25 到 V26 的主線。

---

## 快速結論

| 項目 | 最終選擇 |
|---|---|
| Target model | V17 EoMT-DINOv3 semantic segmentation model |
| Clean mIoU | `79.232563` |
| Clean pixel accuracy | `97.574698` |
| 最終 untargeted attack | `cospgd_untargeted_eps6_255_steps20` |
| 最終 targeted attack | `pgd_target_non_person_fg_to_person_eps6_255_steps20` |
| 最終整合 notebook | `attack_v26_final_report_attacks_eomt_v17_colab.ipynb` |

我們最後保留兩種攻擊：

1. **Untargeted degradation**
   - 目標：盡量降低 segmentation 品質。
   - 使用：CosPGD-inspired PGD，`eps=6/255`，`steps=20`。

2. **Targeted semantic attack**
   - 目標：讓非 person 的前景物體被分割成 `person`。
   - 使用：PGD targeted `non-person foreground -> person`，`eps=6/255`，`steps=20`。
   - V26 的 targeted visualization 會優先篩選 ground truth 中原本沒有 `person` 的圖片，讓攻擊定義更乾淨，類似 classification README 中「原本沒有 aeroplane，攻擊後讓模型預測 aeroplane」的設定。

---

## 任務定義

我們攻擊的是 frozen **V17 EoMT-DINOv3** 語意分割模型。對抗圖片由原圖加上一個小擾動得到：

```text
adversarial_image = clip(clean_image + perturbation, valid_image_range)
```

擾動受 `L_inf` budget 限制：

```text
eps = 4/255 或 6/255
```

意思是每個 pixel 的每個 RGB channel 最多只能改變 4 或 6 個 0-255 pixel value。人眼看到的 adversarial image 通常仍然很像原圖；圖中的 perturbation 會放大成 `Perturbation x10`，只是為了讓擾動紋理看得見。

---

## 為什麼主線改用 PGD-family？

我們之前嘗試過多個 trainable generator attack，但在 V17 EoMT-DINOv3 上效果普遍很弱。相反地，直接對每張圖片做 per-image gradient optimization 的 PGD 類攻擊非常強。

因此最後報告的 PGD 主線是：

```text
V10: FGSM / PGD sanity check，確認 V17 可以被直接攻擊
V22: PGD vs CosPGD-inspired PGD
V23: PGD vs Momentum PGD vs Universal Perturbation
V24: 提高 eps、加強 foreground loss、測 semantic targeted attacks
V25: 強化 targeted non-person foreground -> person
V26: 整合 historical rerun + final visualization
```

這條線的邏輯是：先證明模型可被攻擊，再比較不同 PGD-family 方法，最後選出一個最強的 untargeted 攻擊和一個最乾淨的 targeted semantic 攻擊。

---

## Notebook 對照表

| Notebook | 目的 | 主要結論 |
|---|---|---|
| `attack_v10_fgsm_pgd_eomt_v17_colab.ipynb` | FGSM / PGD sanity check | V17 對 direct PGD 非常脆弱。 |
| `attack_v22_per_image_cospgd_eomt_v17_colab.ipynb` | 比較 PGD 和 CosPGD-inspired PGD | CosPGD-inspired PGD 在 change rate 和 GT probability drop 上更強。 |
| `attack_v23_momentum_universal_pgd_eomt_v17_colab.ipynb` | 比較 Momentum PGD 和 Universal Perturbation | Momentum PGD 有幫助；Universal Perturbation 有少量效果但遠弱於 per-image PGD。 |
| `attack_v24_stronger_pgd_semantic_targets_eomt_v17_colab.ipynb` | 測 eps、foreground weight、semantic target | 提高 eps 或加強 foreground weight 都能提升 PGD 強度；targeted attack 需要看 target-region metrics。 |
| `attack_v25_target_person_stronger_pgd_eomt_v17_colab.ipynb` | 強化 targeted person attack | steps20 對 targeted attack 很重要；non-person foreground -> person 是最乾淨的 target story。 |
| `attack_v26_final_report_attacks_eomt_v17_colab.ipynb` | 最終整合 notebook | 重跑歷史比較，輸出 final figures 和 CSV。 |

---

## 攻擊方法

### FGSM

FGSM 是一步 gradient-sign attack：

```text
delta = eps * sign(gradient)
```

它很快，但通常比 iterative PGD 弱。

### PGD

PGD 是 iterative projected gradient attack：

```text
initialize delta within [-eps, eps]
repeat steps times:
    compute gradient of attack loss
    update delta by alpha * sign(gradient)
    project delta back to [-eps, eps]
    clamp image to valid range
```

Untargeted PGD 的目標是讓 segmentation prediction 變差，因此會最大化 ground-truth segmentation loss。

Targeted PGD 的目標是讓 prediction 靠近指定 target mask，因此會最小化 target-mask loss。

### CosPGD-inspired PGD

CosPGD-inspired PGD 是 PGD 的變體，目標是更有效地降低模型對正確 segmentation 的信心。我們在 V22 中發現，它在 untargeted attack 上比 plain PGD 更強，尤其是：

```text
prediction_change_rate
mean_gt_probability_drop
pixel_accuracy_drop
```

### Momentum PGD

Momentum PGD 會累積多步 gradient direction：

```text
momentum = decay * momentum + normalized_gradient
delta = delta + alpha * sign(momentum)
```

V23 顯示 Momentum PGD 比 plain PGD 稍強。

### Universal Perturbation

Universal Perturbation 嘗試學一個所有圖片共用的擾動。V26 historical rerun 顯示它有少量 mIoU drop，但遠弱於 per-image PGD / CosPGD / Momentum PGD，因此沒有作為 final attack。

---

## Loss 與 regularization

### Untargeted loss

Untargeted PGD 使用 foreground-weighted cross entropy：

```text
loss = CE(prediction, ground_truth)
```

攻擊時最大化這個 loss。Foreground pixels 可以給更高權重，避免攻擊只被大量 background pixels 主導。

### Targeted loss

Targeted attack 會建立一張 target segmentation mask。最終 targeted 設定是：

```text
target_mask = ground_truth_mask
target_mask[non-person foreground pixels] = person
loss = CE(prediction, target_mask)
```

也就是：原本不是 person 的前景物體，攻擊後希望被模型分割成 `person`。

Final targeted config：

```text
target_mode = non_person_foreground_to_person
eps = 6/255
steps = 20
foreground_weight = 5
```

### Regularization

PGD 的主要 regularization 是 `L_inf` projection：

```text
delta clipped to [-eps, eps]
adversarial image clipped to valid image range
```

這確保 adversarial image 和 clean image 在 pixel space 中仍然非常接近。

---

## 主要結果

### V10：FGSM / PGD sanity check

| Condition | mIoU | mIoU drop | max delta | mean delta |
|---|---:|---:|---:|---:|
| clean | 79.232563 | 0.000000 | 0.0 | 0.000000 |
| FGSM 2/255 | 49.507389 | 29.725174 | 2.0 | 1.984618 |
| FGSM 4/255 | 45.791120 | 33.441443 | 4.0 | 3.957107 |
| PGD 4/255 steps10 | 5.342937 | 73.889627 | 4.0 | 2.305545 |
| PGD target fg->bg 4/255 steps10 | 4.422456 | 74.810107 | 4.0 | 2.299249 |

**結論：** V17 並不是不能被攻擊。直接 per-image PGD 可以把 mIoU 從 `79.23` 大幅降到約 `5`。

---

### V22：PGD vs CosPGD-inspired PGD

| Condition | mIoU | mIoU drop | pixel accuracy drop | prediction change rate | mean GT probability drop |
|---|---:|---:|---:|---:|---:|
| PGD 4/255 steps10 | 4.557810 | 74.674753 | 44.159039 | 0.456625 | N/A |
| CosPGD 4/255 steps10 | 3.311241 | 75.921323 | 67.669935 | 0.698096 | N/A |
| PGD target fg->bg 4/255 steps10 | 3.985366 | 75.247198 | 17.905344 | 0.199530 | N/A |
| CosPGD target fg->bg 4/255 steps10 | 3.895775 | 75.336788 | 17.859134 | 0.199069 | N/A |

**結論：** CosPGD-inspired PGD 在 untargeted attack 上比 plain PGD 更有破壞力，尤其是 prediction change 和 GT probability drop。

---

### V23：Momentum PGD / Universal Perturbation

| Condition | mIoU | mIoU drop | pixel accuracy drop | prediction change rate |
|---|---:|---:|---:|---:|
| PGD 4/255 steps10 | 5.660835 | 73.571728 | 45.307067 | 0.468187 |
| Momentum PGD 4/255 steps10 | 4.511180 | 74.721383 | 48.547394 | 0.500784 |
| Universal PGD 4/255 | 69.983980 | 9.248584 | 1.662950 | 0.020352 |

**結論：** Momentum PGD 稍微強於 PGD；Universal Perturbation 有少量效果，但與 per-image PGD-family 的 `70+` mIoU drop 相比仍然很弱。

---

### V24：更強 PGD 與 semantic targets

| Condition | Target mode | eps | steps | mIoU drop | target region match |
|---|---|---:|---:|---:|---:|
| PGD eps4 steps10 | untargeted | 4/255 | 10 | 74.784221 | N/A |
| PGD eps6 steps10 | untargeted | 6/255 | 10 | 75.251972 | N/A |
| PGD fgweight5 eps4 steps10 | untargeted | 4/255 | 10 | 75.217065 | N/A |
| PGD fgweight8 eps4 steps10 | untargeted | 4/255 | 10 | 75.437186 | N/A |
| person -> background | targeted | 4/255 | 10 | -5.563859 | 0.991639 |
| all foreground -> person | targeted | 4/255 | 10 | 69.341453 | 0.892918 |
| non-person foreground -> person | targeted | 4/255 | 10 | 69.950390 | 0.893831 |

**結論：**

- 提高 `eps` 或提高 foreground weight 都能讓 untargeted PGD 更強；這次 rerun 中 `fgweight8 eps4` 的 mIoU drop 甚至略高於 `eps6 steps10`。
- Targeted attack 不能只看 full-image pixel accuracy，因為 background pixels 太多，會稀釋前景區域的攻擊效果。
- `non-person foreground -> person` 是比較貼近題目範例的 semantic targeted attack。

---

### V25：最終 targeted-person 強化版

| Condition | eps | steps | target match | target change | target acc drop | mIoU drop |
|---|---:|---:|---:|---:|---:|---:|
| non-person fg -> person | 4/255 | 20 | 0.981320 | 0.985583 | 0.917411 | 72.675208 |
| non-person fg -> person | 6/255 | 10 | 0.930965 | 0.966553 | 0.911874 | 71.722274 |
| non-person fg -> person | 6/255 | 20 | 0.968717 | 0.981878 | 0.913942 | 72.068882 |
| all fg -> person | 6/255 | 10 | 0.925209 | 0.744861 | 0.676960 | 70.425610 |
| all fg -> person | 6/255 | 20 | 0.981241 | 0.758834 | 0.686631 | 72.604668 |

**Final targeted attack：**

```text
pgd_target_non_person_fg_to_person_eps6_255_steps20
```

V25 historical rerun 顯示，`steps=20` 對 targeted attack 很重要；`eps4 steps20` 和 `eps6 steps20` 都很強。V26 最後採用 `eps6 steps20` 作為 stronger-budget final targeted setting。Final report table 中，這個設定的 target-region match rate 為 `0.984675`，表示約 `98.47%` 的 target-region pixels 被模型預測成 `person`。

---

## 為什麼 V26 選這兩種 final attacks？

| Final role | Method | 原因 |
|---|---|---|
| Untargeted final | `cospgd_untargeted_eps6_255_steps20` | V22 顯示 CosPGD-inspired PGD 比 PGD 更強；V24 顯示提高 eps/foreground pressure 能提升攻擊強度；steps20 作為 final stronger setting。 |
| Targeted final | `pgd_target_non_person_fg_to_person_eps6_255_steps20` | V25 顯示 steps20 的 target-region attack 很強；V26 採用 eps6 steps20 作為 stronger-budget final targeted setting，並搭配 no-person visualization filtering。 |

V26 的角色不是發明新攻擊，而是把之前的 comparison 和 final evidence 整合到同一份 notebook：

```text
historical rerun code
final selected attack configs
summary CSV export
per-class IoU export
five-panel visualizations
no-person targeted visualization filtering
```

---

## 指標說明

| 指標 | 意義 |
|---|---|
| `mIoU` | Mean Intersection over Union，語意分割常用指標，越低代表攻擊越成功。 |
| `mIoU_drop` | clean mIoU - adversarial mIoU，越大代表攻擊越強。 |
| `pixel_accuracy` | valid pixels 中預測正確的比例。 |
| `pixel_accuracy_drop` | clean pixel accuracy - adversarial pixel accuracy。 |
| `prediction_change_rate` | 有多少 valid pixels 的預測類別被改變。 |
| `mean_gt_probability` | 模型對 ground-truth class 的平均信心。 |
| `mean_gt_probability_drop` | ground-truth probability 降低多少。 |
| `target_region_match_rate` | targeted attack 中，target region 有多少比例被預測成目標類別。 |
| `target_region_change_rate` | target region 有多少比例的 prediction 被改變。 |
| `target_region_accuracy_drop` | target region 對原始 ground truth 的 accuracy 下降多少。 |
| `max_delta_pixel` | 0-255 pixel scale 中最大擾動值。 |
| `mean_delta_pixel` | 0-255 pixel scale 中平均擾動值。 |

對 untargeted attack，主要看：

```text
mIoU_drop
pixel_accuracy_drop
prediction_change_rate
mean_gt_probability_drop
```

對 targeted attack，主要看：

```text
target_region_match_rate
target_region_change_rate
target_region_accuracy_drop
```

因為 full-image pixel accuracy 很容易被大量 background pixels 稀釋。

### 如何判斷攻擊成功？

Untargeted attack 的目標是讓模型整體分割變差，因此判斷方向是：

```text
mIoU 越低越好
mIoU_drop 越大越好
pixel_accuracy_drop 越大越好
prediction_change_rate 越高越好
mean_gt_probability_drop 越大越好
```

也就是說，如果 attack 後 mIoU 從 `79.23` 掉到 `4` 或 `5`，代表模型的 segmentation 幾乎被破壞。

Targeted attack 的目標不是單純讓模型變差，而是讓指定區域變成指定類別。因此判斷方向是：

```text
target_region_match_rate 越接近 1 越好
target_region_change_rate 越接近 1 越好
target_region_accuracy_drop 越大越好
```

以 final targeted attack 為例：

```text
pgd_target_non_person_fg_to_person_eps6_255_steps20
target_region_match_rate = 0.984675
```

這表示在 target region 中，約 `98.47%` 的 pixels 被模型預測成我們指定的目標類別 `person`。因此雖然 full-image pixel accuracy 可能看起來沒有降到很低，targeted attack 仍然是成功的，因為它完成了「把非 person 前景分割成 person」這個指定任務。

---

## 視覺化內容

Final visualization 建議使用 V26 輸出的五欄圖：

```text
Original Image
Original Prediction
Perturbation x10
Adversarial Image
Adversarial Prediction
```

Targeted examples 會盡量篩選：

```text
ground truth 不含 person
但有 non-person foreground pixels
```

這樣可以清楚描述：

```text
我們對原本沒有 person 的 foreground object 加上微小擾動，
使 segmentation model 將它們分割成 person。
```

---

## 輸出文件

V26 預期輸出位置：

```text
outputs/attacks/attack_v26_final_report_attacks_eomt_v17/
```

主要文件：

```text
attack_summary.csv
final_report_comparison.csv
per_class_iou.csv
historical_rerun/historical_attack_summary.csv
historical_rerun/historical_per_class_iou.csv
visualizations/
```

---

## 運行方式

使用最終 notebook：

```text
Semantic segmentation/attack_v26_final_report_attacks_eomt_v17_colab.ipynb
```

第一次完整重跑建議：

```python
RUN_HISTORICAL_RERUN = True
RUN_UNIVERSAL_ABLATION = True
```

這會重跑 V10 / V22 / V23 / V24 / V25 的 historical comparison。

如果已經跑完一次，只想重畫圖或重輸出 final table，可以改成：

```python
RUN_HISTORICAL_RERUN = False
```

---

## 報告可用總結

我們的語意分割攻擊可以這樣寫：

1. V10 先確認 V17 EoMT-DINOv3 在 white-box 條件下可以被 direct PGD 攻擊。
2. V22 比較 PGD 和 CosPGD-inspired PGD，發現 CosPGD-inspired 在 untargeted degradation 上更強。
3. V23 測 Momentum PGD 和 Universal Perturbation，Momentum PGD 有幫助，Universal Perturbation 有少量效果但遠弱於 per-image PGD-family。
4. V24 測更高 `eps`、foreground-focused loss、以及 semantic target masks，發現提高 eps 或 foreground weight 都能提升攻擊強度，且 targeted attack 應使用 target-region metrics。
5. V25 找到最乾淨的 targeted semantic attack family：`non-person foreground -> person`；`steps=20` 對 target-region 成功率很重要。
6. V26 將這些比較整合成 final report notebook，並輸出最終表格與圖。

最後使用：

```text
Untargeted final: cospgd_untargeted_eps6_255_steps20
Targeted final:   pgd_target_non_person_fg_to_person_eps6_255_steps20
```

這兩個 final attacks 分別對應：

```text
降低整體 segmentation 品質
讓模型在指定前景區域 hallucinate 成 person
```
