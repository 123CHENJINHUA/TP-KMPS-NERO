"""Report dependencies without loading model weights, opening devices or sending CAN."""
from .common import ROOT, object_config, write_json, output_path
from .geometry import load_mesh
import importlib.util
import json
import os
from pathlib import Path
import subprocess


def environment(python, modules):
    if not Path(python).is_file():
        return dict(python=python, exists=False, error='Python environment missing')
    script = ('import json,importlib.util; '
              f'r={{m:bool(importlib.util.find_spec(m)) for m in {modules!r}}}; '
              '\ntry:\n import torch\n r.update(torch_version=torch.__version__,cuda_available=torch.cuda.is_available(),cuda_build=torch.version.cuda)'
              '\nexcept Exception as e:\n r["torch_error"]=str(e)\nprint(json.dumps(r))')
    try:
        result = subprocess.run([python, '-B', '-c', script], capture_output=True, text=True, timeout=30)
        value = json.loads(result.stdout.strip().splitlines()[-1]) if result.returncode == 0 else dict(error=result.stderr[-1200:])
    except Exception as exc:
        value = dict(error=str(exc))
    return dict(python=python, exists=True, **value)


def doctor(args):
    pose_python = args.pose_python or os.environ.get('TPKMP_POSE_PYTHON', str(ROOT/'envs/foundationpose/bin/python'))
    sam_python = args.sam3_python or os.environ.get('SAM3_PYTHON', str(ROOT/'envs/sam3/bin/python'))
    root = Path(args.foundationpose_root or ROOT/'third_party/FoundationPose')
    if not (root/'estimater.py').exists() and (root/'FoundationPose').is_dir():
        root /= 'FoundationPose'
    files = [root/'weights'/run/name for run in ('2023-10-28-18-33-37', '2024-01-11-20-02-45')
             for name in ('model_best.pth', 'config.yml')]
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=10)
        gpu_info = (gpu.stdout or gpu.stderr).strip()
    except Exception as exc:
        gpu_info = str(exc)
    cfg = object_config()
    mesh = load_mesh(cfg)
    result = dict(parent_modules={name: bool(importlib.util.find_spec(name)) for name in ('cv2', 'pyrealsense2', 'numpy', 'scipy')},
                  object_id=cfg['object_id'], mesh_extents_m=(mesh.reshape(-1, 3).max(axis=0)-mesh.reshape(-1, 3).min(axis=0)).tolist(),
                  gpu=gpu_info, foundationpose_source=(root/'estimater.py').is_file(),
                  mycpp_built=any((root/'mycpp/build').glob('mycpp*.so')),
                  cuda_compiler=(ROOT/'envs/cuda_toolchain/bin/nvcc').is_file(),
                  foundationpose_missing_weights=[str(f) for f in files if not f.is_file()],
                  foundationpose=environment(pose_python, ['torch', 'trimesh', 'cv2', 'nvdiffrast', 'pytorch3d', 'open3d', 'warp']),
                  sam3_checkpoint=Path(args.sam3_checkpoint or ROOT/'weights/sam3.pt').is_file(),
                  sam3=environment(sam_python, ['torch', 'sam3', 'PIL']),
                  opens_camera=False, opens_can=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        write_json(output_path(args.out), result)
