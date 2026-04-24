#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/d/BaiduNetdiskWorkspace/Leuven/8th/Computer Vision/assignment/Group2}"
SEG_ROOT="${REPO_ROOT}/Semantic segmentation"
LOG_DIR="${SEG_ROOT}/outputs/logs"
mkdir -p "${LOG_DIR}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/wsl_segman_b_train_notify_${RUN_ID}.log"
PID_FILE="${LOG_DIR}/wsl_segman_b_train_notify.pid"
START_TS="$(date +%s)"

send_discord() {
  local status="$1"
  local details="$2"

  if [ -z "${DISCORD_WEBHOOK_URL:-}" ]; then
    echo "DISCORD_WEBHOOK_URL is not set; skip Discord notification." | tee -a "${LOG_FILE}"
    return 0
  fi

  python - "$status" "$details" <<'PY'
import json
import os
import sys
from urllib import request

status = sys.argv[1]
details = sys.argv[2]
webhook = os.environ.get("DISCORD_WEBHOOK_URL")
payload = {"content": f"SegMAN-B WSL training {status}.\n{details}"}
req = request.Request(
    webhook,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with request.urlopen(req, timeout=20) as resp:
    print(f"discord_status={resp.status}")
PY
}

cd "${SEG_ROOT}" || exit 1
echo $$ > "${PID_FILE}"
echo "run_id=${RUN_ID}" | tee -a "${LOG_FILE}"
echo "log=${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "pid=$$" | tee -a "${LOG_FILE}"

bash scripts/wsl_train_segman_b.sh 2>&1 | tee -a "${LOG_FILE}"
STATUS=${PIPESTATUS[0]}
END_TS="$(date +%s)"
DURATION_SEC=$((END_TS - START_TS))

BEST_CKPT="$(find "${SEG_ROOT}/external/SegMAN/segmentation/outputs/ga2_segman_b" -maxdepth 1 -name 'best_mIoU_*.pth' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
SUBMISSION="${SEG_ROOT}/outputs/submissions/submission_exp_v10_segman_b.csv"

if [ "${STATUS}" -eq 0 ]; then
  DETAILS="Status: success
Duration: ${DURATION_SEC}s
Best checkpoint: ${BEST_CKPT:-not found}
Submission: ${SUBMISSION}
Log: ${LOG_FILE}"
  send_discord "finished successfully" "${DETAILS}" | tee -a "${LOG_FILE}"
else
  DETAILS="Status: failed
Exit code: ${STATUS}
Duration: ${DURATION_SEC}s
Best checkpoint: ${BEST_CKPT:-not found}
Log: ${LOG_FILE}"
  send_discord "failed" "${DETAILS}" | tee -a "${LOG_FILE}"
fi

rm -f "${PID_FILE}"
exit "${STATUS}"
