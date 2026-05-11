# vit_l_16_224

Optional heavier Vision Transformer experiment using torchvision ViT-L/16
ImageNet-1K weights.

- Backbone: ViT-L/16
- Image size: 224 x 224
- Batch size: 2
- Loss: AsymmetricLoss
- Output directory: `output/image_classification/vit_l_16_224/`

Use this after `vit_b_16_224` is worth pursuing. On the local 8 GB RTX 4060
Laptop GPU it will be slow, so Colab Pro or a stronger GPU is the better home.

```bash
python "Image classification/experiments/vit_l_16_224/train.py"
python "Image classification/experiments/vit_l_16_224/evaluate.py"
python "Image classification/experiments/vit_l_16_224/predict.py"
```
