@echo off
setlocal

cd /d "%~dp0"

"C:\Users\31667\.conda\envs\biometrics\python.exe" "Image classification\merge_submission.py" ^
  --clf-csv "output\image_classification\convnext_small_320\submissions\submission_classification_convnext_small_320.csv" ^
  --seg-csv "output\submission_exp_v10_segman_b_iter25000.csv"
