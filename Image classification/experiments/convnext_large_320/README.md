# convnext_large_320

ConvNeXt-Large classification experiment.

- Backbone: ConvNeXt-Large
- Image size: 320 x 320
- Batch size: 2
- Loss: AsymmetricLoss
- Output: `output/image_classification/convnext_large_320/`

This is a high-cost experiment on an 8 GB GPU. Run `convnext_small_320` and
`convnext_base_320` first unless you specifically want to test the largest
torchvision ConvNeXt backbone.

```bash
python "Image classification/experiments/convnext_large_320/train.py"
python "Image classification/experiments/convnext_large_320/evaluate.py"
python "Image classification/experiments/convnext_large_320/predict.py"
```
