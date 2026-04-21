# Semantic Segmentation

Maintainable semantic segmentation workspace for GA2.

## Environment

Use:

```powershell
conda activate seg_gpu_env
```

Core notes:

- keep `numpy<2` to avoid the Windows `torch/mmcv` ABI breakage
- keep `setuptools<81` because `torch 2.1` still imports `pkg_resources`
- install the official OpenMMLab `mmcv` wheel for your CUDA/Torch combo before training
- use `D:\Anaconda3\envs\seg_gpu_env\python.exe` directly on Windows if `conda run` hits encoding issues

## Current scope

- data analysis and preprocessing visualization
- MMSegmentation-based training configs
- train / validate scripts
