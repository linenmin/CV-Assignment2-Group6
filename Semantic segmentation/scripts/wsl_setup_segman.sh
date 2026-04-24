#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-segman}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
REPO_ROOT="${REPO_ROOT:-/mnt/d/BaiduNetdiskWorkspace/Leuven/8th/Computer Vision/assignment/Group2}"
SEG_ROOT="${REPO_ROOT}/Semantic segmentation"
SEG_MAN_ROOT="${SEG_ROOT}/external/SegMAN"
MINICONDA_DIR="${HOME}/miniconda3"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is not available inside WSL. Install/update NVIDIA WSL support first." >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  if [ ! -x "${MINICONDA_DIR}/bin/conda" ]; then
    echo "Installing Miniconda into ${MINICONDA_DIR}"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "${MINICONDA_DIR}"
  fi
  # shellcheck source=/dev/null
  source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
else
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip

python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install "numpy<2" "setuptools<81" "opencv-python<4.10" -U
python -m pip install -U openmim
mim install mmcv-full==1.7.2
python -m pip install "numpy<2" "opencv-python<4.10" "setuptools<81" -U

if [ ! -d "${SEG_MAN_ROOT}/.git" ]; then
  git clone --depth 1 https://github.com/yunxiangfu2001/SegMAN.git "${SEG_MAN_ROOT}"
fi

python -m pip install -v -e "${SEG_MAN_ROOT}/segmentation"
grep -v '^triton==' "${SEG_MAN_ROOT}/requirements.txt" > /tmp/segman_requirements_linux.txt
python -m pip install -r /tmp/segman_requirements_linux.txt

python -m pip install natten==0.17.3+torch210cu121 -f https://shi-labs.com/natten/wheels/

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc is not available inside WSL." >&2
  echo "Install CUDA toolkit for WSL/Linux before building SegMAN selective_scan." >&2
  echo "Suggested Ubuntu package route: sudo apt-get update && sudo apt-get install -y nvidia-cuda-toolkit" >&2
  exit 2
fi

python -m pip install "${SEG_MAN_ROOT}/kernels/selective_scan"

cd "${SEG_ROOT}"
python -m pip install -e .
python scripts/prepare_segman_experiment.py \
  --segman-root "${SEG_MAN_ROOT}" \
  --variant b \
  --encoder-checkpoint "${SEG_MAN_ROOT}/pretrained/SegMAN_Encoder_b.pth.tar" \
  --batch-size 2 \
  --workers 4

python - <<'PY'
import torch, mmcv, mmseg, natten
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("mmcv", mmcv.__version__)
print("mmseg", mmseg.__version__)
print("natten", natten.__version__)
PY

echo "SegMAN WSL environment is ready."
