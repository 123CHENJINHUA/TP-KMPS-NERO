#!/usr/bin/env bash
set -euo pipefail
calib_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${HAND_EYE_PYTHON:-/home/cjh/anaconda3/envs/tpkmp/bin/python}"
mkdir -p "$calib_dir/.vendor" "$calib_dir/.tmp"
export TMPDIR="$calib_dir/.tmp"
export PYTHONDONTWRITEBYTECODE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
# Scientific packages remain in the existing tpkmp environment; do not replace numpy.
"$python_bin" -B -c 'import numpy, scipy, matplotlib'
"$python_bin" -B -m pip install --no-cache-dir --only-binary=:all: --no-deps \
  --target "$calib_dir/.vendor" -r "$calib_dir/requirements.txt"
bash "$calib_dir/run.sh" doctor.py
