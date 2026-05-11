# Cityscapes Data Placeholder

This directory is reserved for the Cityscapes external generalization check.

Download the official Cityscapes files manually from:

```text
https://www.cityscapes-dataset.com/downloads/
```

Required files:

```text
leftImg8bit_trainvaltest.zip
gtFine_trainvaltest.zip
```

After download, extract both archives into this directory so the final layout is:

```text
data/cityscapes/
  leftImg8bit/
    train/
    val/
    test/
  gtFine/
    train/
    val/
    test/
```

Only the `val` split is needed for the planned external generalization evaluation.

The extracted data and zip files are intentionally ignored by Git.
