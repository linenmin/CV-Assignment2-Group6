# Image Classification Experiments

Each folder is one model experiment. The files inside are thin entry points:
they call the shared training/evaluation/prediction code in the parent
directory, while keeping outputs separated by experiment.

```
experiments/
  efficientnet_b3_320/
    train.py
    evaluate.py
    predict.py
  convnext_tiny_320/
    train.py
    evaluate.py
    predict.py
  convnext_small_320/
    train.py
    evaluate.py
    predict.py
  convnext_base_320/
    train.py
    evaluate.py
    predict.py
  convnext_large_320/
    train.py
    evaluate.py
    predict.py
  resnet50_224/
    train.py
    evaluate.py
    predict.py
```

Outputs are written to:

```
output/image_classification/<experiment>/
  checkpoints/
  metrics/
  figures/
  predictions/
    test_probabilities_<experiment>.csv
    test_binary_predictions_<experiment>.csv
  submissions/
    submission_classification_<experiment>.csv
    submission_final_<experiment>__seg_<segmentation_csv_name>.csv
```

You can run either from these folders or from the shared pipeline:

```bash
python "Image classification/experiments/efficientnet_b3_320/train.py"
python "Image classification/run_pipeline.py" --experiment efficientnet_b3_320

python "Image classification/experiments/convnext_tiny_320/train.py"
python "Image classification/run_pipeline.py" --experiment convnext_tiny_320

python "Image classification/experiments/convnext_small_320/train.py"
python "Image classification/run_pipeline.py" --experiment convnext_small_320

python "Image classification/run_pipeline.py" --experiment convnext_base_320
```

`convnext_large_320` is registered too, but it uses batch size 2 and is the
least efficient first experiment on an 8 GB GPU. Prefer Small and Base before
spending time on Large.

For Colab or Colab Pro runs with stronger GPUs, use:

```bash
Image classification/colab_train_convnext.ipynb
```

It is self-contained, supports ConvNeXt Tiny/Small/Base/Large presets, writes to
Google Drive, and includes an AMP memory probe before training.

Merge a trained model's classification CSV with a segmentation CSV using:

```bash
python "Image classification/merge_submission.py" --experiment convnext_tiny_320 --seg-csv output/submission_exp_v10_segman_b_iter25000.csv
```

Use `--clf-csv` when you want to merge a specific classification submission or
binary prediction CSV. The merged CSV is saved in that experiment's
`submissions/` folder with both source names in the filename.
