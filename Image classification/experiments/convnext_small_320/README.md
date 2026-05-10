# convnext_small_320

当前最佳单模型 ConvNeXt-Small 分类实验。

- 骨干网络：ConvNeXt-Small
- 图像尺寸：320 x 320
- 批大小：8
- 损失函数：AsymmetricLoss
- 输出目录：`output/image_classification/convnext_small_320/`

```bash
python "Image classification/experiments/convnext_small_320/train.py"
python "Image classification/experiments/convnext_small_320/evaluate.py"
python "Image classification/experiments/convnext_small_320/predict.py"
```
