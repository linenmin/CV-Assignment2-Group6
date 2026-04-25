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
    submission_final_<experiment>.csv
```

You can run either from these folders or from the shared pipeline:

```bash
python "Image classification/experiments/efficientnet_b3_320/train.py"
python "Image classification/run_pipeline.py" --experiment efficientnet_b3_320
```
