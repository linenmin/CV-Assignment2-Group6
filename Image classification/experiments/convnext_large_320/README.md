# convnext_large_320

ConvNeXt-Large 分类实验。

- 骨干网络：ConvNeXt-Large
- 图像尺寸：320 x 320
- 批大小：2
- 损失函数：AsymmetricLoss
- 输出目录：`output/image_classification/convnext_large_320/`

这是 8 GB GPU 上成本较高的实验。除非明确要测试 torchvision 中最大的 ConvNeXt 骨干网络，否则建议先运行 `convnext_small_320` 和 `convnext_base_320`。

```bash
python "Image classification/experiments/convnext_large_320/train.py"
python "Image classification/experiments/convnext_large_320/evaluate.py"
python "Image classification/experiments/convnext_large_320/predict.py"
```
