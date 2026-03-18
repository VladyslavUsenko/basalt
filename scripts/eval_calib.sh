#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_DIR="${HOME}/.local/etc/basalt"
APRILGRID_PATH="${DATA_DIR}/aprilgrid_6x6.json"
RESULT_ROOT="${REPO_ROOT}/calib_eval_results"
SUMMARY_PATH="${RESULT_ROOT}/summary.txt"

mkdir -p "${RESULT_ROOT}"
: > "${SUMMARY_PATH}"

run_case() {
  local label="$1"
  local camera_dataset="$2"
  local imu_dataset="$3"
  local cam_model="$4"
  local reference="$5"
  local check_mocap_identity="${6:-0}"

  local result_dir="${RESULT_ROOT}/${label}"
  rm -rf "${result_dir}"
  mkdir -p "${result_dir}"

  echo "Running ${label} (${cam_model})"

  basalt_calibrate --no-gui \
    --dataset-path "${camera_dataset}" \
    --dataset-type bag \
    --aprilgrid "${APRILGRID_PATH}" \
    --result-path "${result_dir}/" \
    --cam-types "${cam_model}" "${cam_model}"

  basalt_calibrate_imu --no-gui \
    --dataset-path "${imu_dataset}" \
    --dataset-type bag \
    --aprilgrid "${APRILGRID_PATH}" \
    --result-path "${result_dir}/" \
    --gyro-noise-std 0.000282 \
    --accel-noise-std 0.016 \
    --gyro-bias-std 0.0001 \
    --accel-bias-std 0.001

  local args=(
    --generated "${result_dir}/calibration.json"
    --reference "${reference}"
    --label "${label}"
  )
  if [[ "${check_mocap_identity}" == "1" ]]; then
    args+=(--mocap "${result_dir}/mocap_calibration.json" --check-mocap-identity)
  fi

  python3 "${SCRIPT_DIR}/check_calibration.py" "${args[@]}" | tee -a "${SUMMARY_PATH}"
  echo "" | tee -a "${SUMMARY_PATH}"
}

run_case "tumvi_ds" \
  "/data/tumvi_calib_data/dataset-calib-cam3_512_16.bag" \
  "/data/tumvi_calib_data/dataset-calib-imu1_512_16.bag" \
  "ds" \
  "${DATA_DIR}/tumvi_512_ds_calib.json" \
  1

run_case "tumvi_eucm" \
  "/data/tumvi_calib_data/dataset-calib-cam3_512_16.bag" \
  "/data/tumvi_calib_data/dataset-calib-imu1_512_16.bag" \
  "eucm" \
  "${DATA_DIR}/tumvi_512_eucm_calib.json" \
  1

run_case "euroc_ds" \
  "/data/euroc_calib_data/cam_april.bag" \
  "/data/euroc_calib_data/imu_april.bag" \
  "ds" \
  "${DATA_DIR}/euroc_ds_calib.json"

run_case "euroc_eucm" \
  "/data/euroc_calib_data/cam_april.bag" \
  "/data/euroc_calib_data/imu_april.bag" \
  "eucm" \
  "${DATA_DIR}/euroc_eucm_calib.json"

echo "Calibration evaluation finished successfully."
