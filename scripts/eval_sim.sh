#!/bin/bash
set -euo pipefail

RESULT_ROOT="${PWD}/sim_eval_results"
MARG_DATA_DIR="${RESULT_ROOT}/marg"
VIO_RESULT_PATH="${RESULT_ROOT}/vio_rmse.txt"
MAPPER_RESULT_PATH="${RESULT_ROOT}/mapper_rmse.txt"

VIO_RMSE_LIMIT=0.05
MAPPER_RMSE_LIMIT=0.05

CAM_CALIB="${HOME}/.local/etc/basalt/euroc_ds_calib.json"

rm -rf "${RESULT_ROOT}"
mkdir -p "${RESULT_ROOT}"

echo "Running basalt_vio_sim in headless mode..."
basalt_vio_sim \
  --show-gui 0 \
  --cam-calib "${CAM_CALIB}" \
  --marg-data "${MARG_DATA_DIR}" \
  --result-path "${VIO_RESULT_PATH}"

if [[ ! -f "${VIO_RESULT_PATH}" ]]; then
  echo "Missing VIO simulation result file: ${VIO_RESULT_PATH}" >&2
  exit 1
fi

echo "Running basalt_mapper_sim in headless mode..."
basalt_mapper_sim \
  --show-gui 0 \
  --cam-calib "${CAM_CALIB}" \
  --marg-data "${MARG_DATA_DIR}" \
  --result-path "${MAPPER_RESULT_PATH}"

if [[ ! -f "${MAPPER_RESULT_PATH}" ]]; then
  echo "Missing mapper simulation result file: ${MAPPER_RESULT_PATH}" >&2
  exit 1
fi

vio_rmse="$(cat "${VIO_RESULT_PATH}")"
mapper_rmse="$(cat "${MAPPER_RESULT_PATH}")"

python3 - "$vio_rmse" "$mapper_rmse" "$VIO_RMSE_LIMIT" "$MAPPER_RMSE_LIMIT" <<'PY'
import sys

vio_rmse = float(sys.argv[1])
mapper_rmse = float(sys.argv[2])
vio_limit = float(sys.argv[3])
mapper_limit = float(sys.argv[4])

print(f"basalt_vio_sim RMSE: {vio_rmse:.7f} (limit {vio_limit:.7f})")
print(f"basalt_mapper_sim RMSE: {mapper_rmse:.7f} (limit {mapper_limit:.7f})")

failed = False
if vio_rmse > vio_limit:
    print("VIO simulation RMSE exceeds threshold", file=sys.stderr)
    failed = True
if mapper_rmse > mapper_limit:
    print("Mapper simulation RMSE exceeds threshold", file=sys.stderr)
    failed = True

sys.exit(1 if failed else 0)
PY
