#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_DIR="${HOME}/.local/etc/basalt"
APRILGRID_PATH="${DATA_DIR}/aprilgrid_6x6.json"
T265_DATASET_ROOT="/data/t265_calib_data"
RESULT_ROOT="${REPO_ROOT}/realsense_eval_results"
SUMMARY_PATH="${RESULT_ROOT}/summary.txt"
PYTHON_BIN="${BASALT_VENV_PYTHON:-python3}"
CALIB_ROOT="${RESULT_ROOT}/calib"

rm -rf "${RESULT_ROOT}"
mkdir -p "${RESULT_ROOT}"
: > "${SUMMARY_PATH}"

log() {
  echo "$*" | tee -a "${SUMMARY_PATH}"
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || {
    echo "Required file not found: ${path}" >&2
    exit 1
  }
}

log "Running Realsense response calibration"
"${PYTHON_BIN}" "${HOME}/.local/bin/basalt_response_calib.py" \
  -d "${T265_DATASET_ROOT}/response_calib" \
  --no-gui \
  --save-plot "${RESULT_ROOT}/response_calib.png" | tee -a "${SUMMARY_PATH}"
require_file "${RESULT_ROOT}/response_calib.png"

log ""
log "Running Realsense camera calibration"
mkdir -p "${CALIB_ROOT}"
basalt_calibrate --no-gui \
  --dataset-path "${T265_DATASET_ROOT}/cam_calib" \
  --dataset-type euroc \
  --result-path "${CALIB_ROOT}/" \
  --aprilgrid "${APRILGRID_PATH}" \
  --cam-types kb4 kb4 | tee -a "${SUMMARY_PATH}"
require_file "${CALIB_ROOT}/calibration.json"
mkdir -p "${RESULT_ROOT}/cam_calib"
cp -f "${CALIB_ROOT}/calibration.json" "${RESULT_ROOT}/cam_calib/calibration.json"
find "${CALIB_ROOT}" -maxdepth 1 -type f \( -name 'vignette*.png' -o -name 'vignette*.jpg' \) -exec cp -f {} "${RESULT_ROOT}/cam_calib/" \;

log ""
log "Running Realsense IMU calibration"
basalt_calibrate_imu --no-gui \
  --dataset-path "${T265_DATASET_ROOT}/imu_calib" \
  --dataset-type euroc \
  --result-path "${CALIB_ROOT}/" \
  --aprilgrid "${APRILGRID_PATH}" \
  --accel-noise-std 0.00818 \
  --gyro-noise-std 0.00226 \
  --accel-bias-std 0.01 \
  --gyro-bias-std 0.0007 | tee -a "${SUMMARY_PATH}"
require_file "${CALIB_ROOT}/calibration.json"
require_file "${CALIB_ROOT}/mocap_calibration.json"
mkdir -p "${RESULT_ROOT}/imu_calib"
cp -f "${CALIB_ROOT}/calibration.json" "${RESULT_ROOT}/imu_calib/calibration.json"
cp -f "${CALIB_ROOT}/mocap_calibration.json" "${RESULT_ROOT}/imu_calib/mocap_calibration.json"

log ""
log "Checking Realsense calibration against reference"
"${PYTHON_BIN}" "${SCRIPT_DIR}/check_calibration.py" \
  --generated "${CALIB_ROOT}/calibration.json" \
  --reference "${DATA_DIR}/t265_kb4_calib.json" \
  --label "t265_kb4" | tee -a "${SUMMARY_PATH}"

log ""
log "Running Realsense time alignment"
mkdir -p "${RESULT_ROOT}/time_alignment"
basalt_time_alignment --no-gui \
  --dataset-path "${T265_DATASET_ROOT}/sequence0" \
  --dataset-type euroc \
  --calibration "${CALIB_ROOT}/calibration.json" \
  --mocap-calibration "${CALIB_ROOT}/mocap_calibration.json" \
  --output "${RESULT_ROOT}/time_alignment/time_alignment.json" \
  --output-error "${RESULT_ROOT}/time_alignment/error.csv" \
  --output-gyro "${RESULT_ROOT}/time_alignment/gyro.csv" \
  --output-mocap "${RESULT_ROOT}/time_alignment/mocap.csv" \
  --output-gt "${RESULT_ROOT}/time_alignment/gt_data.csv" | tee -a "${SUMMARY_PATH}"
require_file "${RESULT_ROOT}/time_alignment/time_alignment.json"
require_file "${RESULT_ROOT}/time_alignment/error.csv"
require_file "${RESULT_ROOT}/time_alignment/gyro.csv"
require_file "${RESULT_ROOT}/time_alignment/mocap.csv"
require_file "${RESULT_ROOT}/time_alignment/gt_data.csv"

log ""
log "Running Realsense VIO on sequence0"
basalt_vio \
  --dataset-path "${T265_DATASET_ROOT}/sequence0" \
  --cam-calib "${CALIB_ROOT}/calibration.json" \
  --dataset-type euroc \
  --config-path "${DATA_DIR}/euroc_config.json" \
  --show-gui 0 \
  --result-path "${RESULT_ROOT}/vio_rmse_ate.txt" | tee -a "${SUMMARY_PATH}"
require_file "${RESULT_ROOT}/vio_rmse_ate.txt"

log ""
log "Running Realsense device smoke tests"
basalt_rs_t265_record --help >/dev/null
basalt_rs_t265_vio --help >/dev/null
log "Device smoke tests: OK"

log ""
log "Realsense evaluation finished successfully."
