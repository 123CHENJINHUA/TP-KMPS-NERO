"""GPU worker using FoundationPose's published API, including ++ vendored backend.

Input/output poses always refer to the original CAD frame, scaled to meters.
The estimator internally centers its mesh and restores that frame in register().
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request', required=True)
    args = p.parse_args()
    request = json.loads(Path(args.request).read_text())
    root = Path(request['foundationpose_root']).resolve()
    if not (root/'estimater.py').is_file() and (root/'FoundationPose/estimater.py').is_file():
        root = root/'FoundationPose'
    if not (root/'estimater.py').is_file():
        raise FileNotFoundError('缺少 FoundationPose estimater.py；先运行 setup.sh sources')
    sys.path.insert(0, str(root))
    os.chdir(root)
    import numpy as np
    import torch
    import trimesh
    includes = list((Path(torch.__file__).resolve().parents[1]/'nvidia').glob('*/include'))
    os.environ['CPATH'] = os.pathsep.join(map(str, includes))+os.pathsep+os.environ.get('CPATH', '')
    if not torch.cuda.is_available():
        raise RuntimeError('FoundationPose 需要 CUDA，当前 GPU/驱动不可见；不会返回假定位结果')
    import warp
    warp.config.kernel_cache_dir = str(Path(__file__).resolve().parent/'.cache/warp')
    if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}'
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor, dr
    from Utils import trimesh_add_pure_colored_texture
    mesh = trimesh.load(request['mesh_path'], force='mesh')
    mesh.apply_scale(request['mesh_scale_to_m'])
    if request.get('mesh_rgb') is not None:
        mesh = trimesh_add_pure_colored_texture(mesh, color=np.asarray(request['mesh_rgb']), resolution=10)
    scorer, refiner = ScorePredictor(), PoseRefinePredictor()
    estimator = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                              scorer=scorer, refiner=refiner, glctx=dr.RasterizeCudaContext(),
                              debug=0, debug_dir=request['debug_dir'])
    with np.load(request['input'], allow_pickle=False) as arrays:
        rgb, depth, K, masks = arrays['rgb'], arrays['depth_m'], arrays['K'], arrays['masks']
    poses = []
    for color, distance, mask in zip(rgb, depth, masks):
        # Independent registration in each static frame, not a temporally copied pose.
        pose = estimator.register(K=K, rgb=color, depth=distance, ob_mask=mask.astype(bool),
                                  iteration=request['iterations'])
        value = np.asarray(pose, float).reshape(4, 4)
        if not np.isfinite(value).all():
            raise ValueError('FoundationPose 返回无效位姿')
        poses.append(value)
    with Path(request['output']).open('xb') as stream:
        np.savez_compressed(stream, T_color_object=np.array(poses))


if __name__ == '__main__':
    main()
