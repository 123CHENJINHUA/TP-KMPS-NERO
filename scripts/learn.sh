#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-$project_dir/output/.matplotlib}"
export OPENBLAS_NUM_THREADS="${TPKMP_THREADS:-1}"
export OMP_NUM_THREADS="${TPKMP_THREADS:-1}"
exec "${TPKMP_PYTHON:-/home/cjh/anaconda3/envs/tpkmp/bin/python}" -m sim_bridge.learning "$@"
