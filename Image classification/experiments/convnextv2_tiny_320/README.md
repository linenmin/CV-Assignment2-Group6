# convnextv2_tiny_320

HF Transformers ConvNeXt V2 Tiny baseline for the VOC multi-label classifier.

- Backbone: `facebook/convnextv2-tiny-1k-224`
- Input size: 320 x 320
- Batch size: 8
- Loss: AsymmetricLoss
- Output directory: `output/image_classification/convnextv2_tiny_320/`

```bash
python "Image classification/experiments/convnextv2_tiny_320/train.py"
python "Image classification/experiments/convnextv2_tiny_320/evaluate.py"
python "Image classification/experiments/convnextv2_tiny_320/predict.py"
```

This experiment requires the optional `transformers` package in the `biometrics`
environment.
