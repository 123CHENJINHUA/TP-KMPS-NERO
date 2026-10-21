#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${TPKMP_PYTHON:-/home/cjh/anaconda3/envs/tpkmp/bin/python}"
mkdir -p "$project_dir/hardware_checks/.vendor" "$project_dir/hardware_checks/.tmp"
export TMPDIR="$project_dir/hardware_checks/.tmp"
export PYTHONDONTWRITEBYTECODE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
"$python_bin" -B -m pip install --no-cache-dir --target "$project_dir/hardware_checks/.vendor" 'python-can==4.6.1'
