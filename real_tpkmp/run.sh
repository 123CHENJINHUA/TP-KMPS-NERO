#!/usr/bin/env bash
set -euo pipefail
real_tpkmp_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
real_tpkmp_python="${TPKMP_PYTHON:-/home/cjh/anaconda3/envs/tpkmp/bin/python}"
if [[ ! -x "$real_tpkmp_python" ]]; then
  printf '找不到 Python：%s；用 TPKMP_PYTHON 指定已有环境。\n' "$real_tpkmp_python" >&2
  exit 1
fi
if [[ $# -lt 1 ]]; then
  printf '用法：bash real_tpkmp/run.sh collect.py|train.py|validate.py|frames.py|test_real.py|self_test.py [参数]\n' >&2
  exit 1
fi
real_tpkmp_module="${1%.py}"
shift
case "$real_tpkmp_module" in
  collect|train|validate|frames|test_real|self_test|make_synthetic|vision.cli|vision.meshy.cli) ;;
  *) printf '不支持的入口：%s\n' "$real_tpkmp_module" >&2; exit 1 ;;
esac
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export MPLCONFIGDIR="$real_tpkmp_dir/.cache/matplotlib"
export XDG_CACHE_HOME="$real_tpkmp_dir/.cache"
export HF_HOME="$real_tpkmp_dir/vision/.cache/huggingface"
export TORCH_HOME="$real_tpkmp_dir/vision/.cache/torch"
export TORCH_EXTENSIONS_DIR="$real_tpkmp_dir/vision/.cache/torch_extensions"
export PIP_CACHE_DIR="$real_tpkmp_dir/vision/.cache/pip"
export CUDA_CACHE_PATH="$real_tpkmp_dir/vision/.cache/cuda"
export TRITON_CACHE_DIR="$real_tpkmp_dir/vision/.cache/triton"
export NUMBA_CACHE_DIR="$real_tpkmp_dir/vision/.cache/numba"
cd -- "$real_tpkmp_dir/.."
exec "$real_tpkmp_python" -B -m "real_tpkmp.$real_tpkmp_module" "$@"
