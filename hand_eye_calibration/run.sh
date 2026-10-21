#!/usr/bin/env bash
set -euo pipefail
calib_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${HAND_EYE_PYTHON:-/home/cjh/anaconda3/envs/tpkmp/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
if [[ $# -lt 1 ]]; then
  echo "用法: bash hand_eye_calibration/run.sh doctor.py [参数]"
  exit 2
fi
script_name="$1"
shift
exec "$python_bin" -B "$calib_dir/$script_name" "$@"
