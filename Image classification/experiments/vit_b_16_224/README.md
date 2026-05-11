# vit_b_16_224

Vision Transformer baseline using torchvision ViT-B/16 ImageNet-1K weights.

- Backbone: ViT-B/16
- Image size: 224 x 224
- Batch size: 8
- Loss: AsymmetricLoss
- Output directory: `output/image_classification/vit_b_16_224/`

This is the first ViT experiment to run. It keeps the pretrained model's native
224-pixel input size so the positional embeddings match the released weights.

```bash
python "Image classification/experiments/vit_b_16_224/train.py"
python "Image classification/experiments/vit_b_16_224/evaluate.py"
python "Image classification/experiments/vit_b_16_224/predict.py"
```
