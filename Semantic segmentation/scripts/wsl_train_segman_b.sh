#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-segman}"
REPO_ROOT="${REPO_ROOT:-/mnt/d/BaiduNetdiskWorkspace/Leuven/8th/Computer Vision/assignment/Group2}"
SEG_ROOT="${REPO_ROOT}/Semantic segmentation"
SEG_MAN_ROOT="${SEG_ROOT}/external/SegMAN"
ENCODER_CKPT="${SEG_MAN_ROOT}/pretrained/SegMAN_Encoder_b.pth.tar"

if [ -d /usr/local/cuda/bin ]; then
  export PATH="/usr/local/cuda/bin:${PATH}"
fi
if [ -d /usr/local/cuda/lib64 ]; then
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "${ENV_NAME}"

if [ ! -f "${ENCODER_CKPT}" ]; then
  echo "ERROR: missing pretrained encoder checkpoint: ${ENCODER_CKPT}" >&2
  echo "Download SegMAN_Encoder_b.pth.tar from the official SegMAN README and place it there first." >&2
  exit 1
fi

cd "${SEG_ROOT}"
python scripts/patch_mmcv_torch21.py
python scripts/patch_segman_amp.py
python scripts/prepare_segman_experiment.py \
  --segman-root "${SEG_MAN_ROOT}" \
  --variant b \
  --encoder-checkpoint "${ENCODER_CKPT}" \
  --batch-size 2 \
  --workers 4

cd "${SEG_MAN_ROOT}/segmentation"
TRAIN_ARGS=(local_configs/segman/ga2/segman_b_ga2.py --work-dir outputs/ga2_segman_b)
if [ -f outputs/ga2_segman_b/latest.pth ]; then
  TRAIN_ARGS+=(--resume-from outputs/ga2_segman_b/latest.pth)
fi
python tools/train.py "${TRAIN_ARGS[@]}"

cd "${SEG_ROOT}"
BEST_CKPT="$(ls -1 "${SEG_MAN_ROOT}/segmentation/outputs/ga2_segman_b"/best_mIoU_*.pth | tail -n 1)"
python scripts/predict_segman_test.py \
  --config "${SEG_MAN_ROOT}/segmentation/local_configs/segman/ga2/segman_b_ga2.py" \
  --checkpoint "${BEST_CKPT}" \
  --output-dir outputs/predictions/exp_v10_segman_b_test
python scripts/export_submission.py \
  --prediction-dir outputs/predictions/exp_v10_segman_b_test \
  --output-path outputs/submissions/submission_exp_v10_segman_b.csv \
  --classification-fill 0

echo "best_checkpoint=${BEST_CKPT}"
echo "submission=outputs/submissions/submission_exp_v10_segman_b.csv"
