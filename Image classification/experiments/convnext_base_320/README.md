# convnext_base_320

ConvNeXt-Base 分类实验。

- 骨干网络：ConvNeXt-Base
- 图像尺寸：320 x 320
- 批大小：4
- 损失函数：AsymmetricLoss
- 第二阶段提前停止：patience 4
- 输出目录：`output/image_classification/convnext_base_320/`

```bash
python "Image classification/experiments/convnext_base_320/train.py"
python "Image classification/experiments/convnext_base_320/evaluate.py"
python "Image classification/experiments/convnext_base_320/predict.py"
```
