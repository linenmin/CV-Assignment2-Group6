# convnextv2_base_320

Heavier optional HF Transformers ConvNeXt V2 Base experiment.

- Backbone: `facebook/convnextv2-base-1k-224`
- Input size: 320 x 320
- Batch size: 4
- Loss: AsymmetricLoss
- Output directory: `output/image_classification/convnextv2_base_320/`

```bash
python "Image classification/experiments/convnextv2_base_320/train.py"
python "Image classification/experiments/convnextv2_base_320/evaluate.py"
python "Image classification/experiments/convnextv2_base_320/predict.py"
```

This experiment requires the optional `transformers` package in the `biometrics`
environment. Try Tiny first unless you have GPU time to spend.
