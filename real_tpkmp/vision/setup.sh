#!/usr/bin/env bash
set -euo pipefail
vision_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
vision_action="${1:-help}"
export CONDA_PKGS_DIRS="$vision_root/.cache/conda_pkgs"
export CONDA_ENVS_PATH="$vision_root/envs"
export CONDA_REGISTER_ENVS=false
export PIP_CACHE_DIR="$vision_root/.cache/pip"
export XDG_CACHE_HOME="$vision_root/.cache"
export HF_HOME="$vision_root/.cache/huggingface"
export TORCH_HOME="$vision_root/.cache/torch"
export TORCH_EXTENSIONS_DIR="$vision_root/.cache/torch_extensions"
export TMPDIR="$vision_root/.cache/tmp"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$vision_root/envs" "$vision_root/third_party" "$vision_root/weights"
vision_conda="${VISION_CONDA:-/home/cjh/anaconda3/bin/conda}"
vision_base_python="${VISION_BASE_PYTHON:-/home/cjh/anaconda3/envs/qwen-sam3/bin/python}"
vision_pose_env="$vision_root/envs/foundationpose"
vision_sam_env="$vision_root/envs/sam3"
vision_pose_src="$vision_root/third_party/FoundationPose"
vision_cuda="$vision_root/envs/cuda_toolchain"
case "$vision_action" in
  sources)
    if [[ ! -f "$vision_pose_src/estimater.py" ]]; then
      git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60 clone --depth 1 https://github.com/NVlabs/FoundationPose.git "$vision_pose_src"
    fi
    if [[ ! -f "$vision_root/third_party/sam3/sam3/model_builder.py" ]]; then
      git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60 clone --depth 1 https://github.com/facebookresearch/sam3.git "$vision_root/third_party/sam3"
    fi
    ;;
  sam3-env)
    [[ -x "$vision_base_python" ]] || { printf '用 VISION_BASE_PYTHON 指定已有 PyTorch+SAM3 依赖的 Python。\n' >&2; exit 1; }
    if [[ ! -x "$vision_sam_env/bin/python" ]]; then
      "$vision_base_python" -m venv --system-site-packages "$vision_sam_env"
    fi
    "$vision_sam_env/bin/python" -m pip install -r "$vision_root/requirements-sam3.txt"
    "$vision_sam_env/bin/python" -B -c 'import torch, PIL, timm; print("Reused PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())'
    printf 'SAM3 使用独立 venv，复用已有环境依赖；源码与权重位于本视觉目录。\n'
    ;;
  pose-env)
    [[ -f "$vision_pose_src/estimater.py" ]] || { printf '先运行 setup.sh sources。\n' >&2; exit 1; }
    [[ -x "$vision_base_python" ]] || { printf '用 VISION_BASE_PYTHON 指定已有 CUDA PyTorch+torchvision 的 Python。\n' >&2; exit 1; }
    "$vision_base_python" -B -c 'import torch, torchvision; assert torch.cuda.is_available(), "CUDA GPU 不可见"; print(torch.__version__, torch.version.cuda)'
    if [[ ! -x "$vision_pose_env/bin/python" ]]; then
      "$vision_base_python" -m venv --system-site-packages "$vision_pose_env"
    fi
    if [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
      vision_cuda="$CUDA_HOME"
    else
      vision_cuda_version="${VISION_CUDA_VERSION:-$("$vision_base_python" -B -c 'import torch; print(torch.version.cuda)')}"
      if [[ ! -x "$vision_cuda/bin/nvcc" ]]; then
        "$vision_conda" create --yes --prefix "$vision_cuda" --override-channels -c nvidia -c conda-forge \
          "cuda-nvcc=$vision_cuda_version" "cuda-cudart-dev=$vision_cuda_version" eigen boost-cpp ninja cmake
      fi
    fi
    export CUDA_HOME="$vision_cuda"
    export PATH="$vision_pose_env/bin:$vision_cuda/bin:$PATH"
    export CPATH="$vision_cuda/targets/x86_64-linux/include:${CPATH:-}"
    export LIBRARY_PATH="$vision_cuda/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$vision_cuda/lib:$vision_cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
    export CMAKE_PREFIX_PATH="$vision_cuda:${CMAKE_PREFIX_PATH:-}"
    vision_nvidia_includes="$("$vision_pose_env/bin/python" -B -c 'import torch; from pathlib import Path; print(":".join(str(p) for p in (Path(torch.__file__).resolve().parents[1]/"nvidia").glob("*/include")))')"
    export CPATH="$CPATH:$vision_nvidia_includes"
    export MAX_JOBS="${VISION_BUILD_JOBS:-2}"
    export TORCH_CUDA_ARCH_LIST
    TORCH_CUDA_ARCH_LIST="$("$vision_pose_env/bin/python" -B -c 'import torch; a,b=torch.cuda.get_device_capability(); print(f"{a}.{b}")')"
    "$vision_pose_env/bin/python" -m pip install -r "$vision_root/requirements-foundationpose.txt"
    if [[ ! -f "$vision_root/third_party/pytorch3d/setup.py" ]]; then
      git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60 clone --depth 1 https://github.com/facebookresearch/pytorch3d.git "$vision_root/third_party/pytorch3d"
    fi
    if [[ ! -f "$vision_root/third_party/nvdiffrast/pyproject.toml" && ! -f "$vision_root/third_party/nvdiffrast/setup.py" ]]; then
      git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60 clone --depth 1 https://github.com/NVlabs/nvdiffrast.git "$vision_root/third_party/nvdiffrast"
    fi
    if ! "$vision_pose_env/bin/python" -B -c 'import pytorch3d._C' >/dev/null 2>&1; then
      "$vision_pose_env/bin/python" -m pip install --use-pep517 --no-build-isolation "$vision_root/third_party/pytorch3d"
    fi
    if ! "$vision_pose_env/bin/python" -B -c 'import nvdiffrast.torch' >/dev/null 2>&1; then
      "$vision_pose_env/bin/python" -m pip install --no-build-isolation "$vision_root/third_party/nvdiffrast"
    fi
    vision_pybind_dir="$("$vision_pose_env/bin/python" -m pybind11 --cmakedir)"
    cmake -S "$vision_pose_src/mycpp" -B "$vision_pose_src/mycpp/build" \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$vision_cuda" \
      -Dpybind11_DIR="$vision_pybind_dir" -DPYTHON_EXECUTABLE="$vision_pose_env/bin/python"
    cmake --build "$vision_pose_src/mycpp/build" --parallel "$MAX_JOBS"
    printf 'FoundationPose 环境已构建。接着运行 setup.sh weights 和 doctor.sh。\n'
    ;;
  weights)
    vision_weights_ready=1
    for vision_run in 2023-10-28-18-33-37 2024-01-11-20-02-45; do
      if [[ ! -s "$vision_pose_src/weights/$vision_run/model_best.pth" || ! -s "$vision_pose_src/weights/$vision_run/config.yml" ]]; then
        vision_weights_ready=0
      fi
    done
    if [[ "$vision_weights_ready" == 1 ]]; then
      printf 'FoundationPose 两组权重已齐全，跳过下载。\n'
      exit 0
    fi
    [[ -x "$vision_pose_env/bin/python" ]] || { printf '先运行 setup.sh pose-env。\n' >&2; exit 1; }
    mkdir -p "$vision_pose_src/weights"
    "$vision_pose_env/bin/python" -m gdown --folder 'https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i' -O "$vision_pose_src/weights/" --continue --no-cookies
    for vision_run in 2023-10-28-18-33-37 2024-01-11-20-02-45; do
      if [[ ! -e "$vision_pose_src/weights/$vision_run" && -d "$vision_pose_src/weights/no_diffusion/$vision_run" ]]; then
        ln -s "no_diffusion/$vision_run" "$vision_pose_src/weights/$vision_run"
      fi
    done
    ;;
  *)
    printf '用法：bash real_tpkmp/vision/setup.sh sources|sam3-env|pose-env|weights\n'
    printf '所有新增环境、CUDA 工具链、下载、编译输出均位于 real_tpkmp/vision。\n'
    printf '默认复用 /home/cjh/anaconda3/envs/qwen-sam3 中已有的 PyTorch，不修改该环境。\n'
    ;;
esac
