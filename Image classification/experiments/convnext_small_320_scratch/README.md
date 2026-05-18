# convnext_small_320_scratch

From-scratch ConvNeXt-Small classification baseline for the assignment's comparison between random initialization and transfer learning.

- Backbone: ConvNeXt-Small
- Initialization: random, no ImageNet weights
- Image size: 320 x 320
- Batch size: 8
- Loss: AsymmetricLoss
- Stage 1: full-network warmup, not frozen
- Output directory: `output/image_classification/convnext_small_320_scratch/`

```bash
python "Image classification/experiments/convnext_small_320_scratch/train.py"
python "Image classification/experiments/convnext_small_320_scratch/evaluate.py"
python "Image classification/experiments/convnext_small_320_scratch/predict.py"
```

This experiment is meant as the fair architectural baseline for `convnext_small_320`: same backbone, input size, loss, thresholds, and TTA, but without pretrained weights.
