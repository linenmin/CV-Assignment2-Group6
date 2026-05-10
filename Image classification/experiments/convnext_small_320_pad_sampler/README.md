# convnext_small_320_pad_sampler

用于弱类别恢复的 ConvNeXt-Small 实验。

- 骨干网络：ConvNeXt-Small
- 图像尺寸：320 x 320
- 图像预处理：先方形 padding，再 resize
- 采样方式：对 bottle、diningtable、pottedplant、sheep、sofa 做弱类加权采样
- 第二阶段提前停止：patience 4
- 损失函数：AsymmetricLoss
- 输出目录：`output/image_classification/convnext_small_320_pad_sampler/`

```bash
python "Image classification/experiments/convnext_small_320_pad_sampler/train.py"
python "Image classification/experiments/convnext_small_320_pad_sampler/evaluate.py"
python "Image classification/experiments/convnext_small_320_pad_sampler/predict.py"
```
