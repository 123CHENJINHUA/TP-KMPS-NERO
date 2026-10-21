"""RGB-D + mask + CAD -> camera/base object pose; failed quality remains diagnostic."""
from .common import ROOT, read_capture, object_config, output_path, new_directory, write_json, digest, utc_now, checked, pose_average
from .geometry import load_mesh, quality, bounds, corners
from .segmentation import get_masks
from .coordinates import base_color_transforms
from ..bootstrap import read_json
import os
from pathlib import Path
import subprocess
import numpy as np


def overlay(rgb, K, pose, box, label):
    import cv2
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    points = corners(box)
    camera = points@pose[:3, :3].T+pose[:3, 3]
    if np.min(camera[:, 2]) > 0:
        uvw = camera@K.T
        uv = np.rint(uvw[:, :2]/uvw[:, 2:]).astype(int)
        for i in range(8):
            for j in range(i+1, 8):
                if np.count_nonzero(points[i] != points[j]) == 1:
                    cv2.line(image, tuple(uv[i]), tuple(uv[j]), (0, 255, 255), 2)
        origin = np.zeros(3)
        length = min(.04, float(np.max(box[1]-box[0]))/2)
        for axis, color in zip(np.eye(3)*length, ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
            xyz = np.array([origin, axis])@pose[:3, :3].T+pose[:3, 3]
            if np.min(xyz[:, 2]) > 0:
                pixels = xyz@K.T
                pixels = np.rint(pixels[:, :2]/pixels[:, 2:]).astype(int)
                cv2.arrowedLine(image, tuple(pixels[0]), tuple(pixels[1]), color, 2, tipLength=.2)
    cv2.putText(image, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0) if label == 'PASS' else (0, 0, 255), 2)
    return image


def process_poses(capture_meta, arrays, masks, poses, cfg, hand_eye=None, backend='foundationpose'):
    poses = np.array([checked(p) for p in poses])
    if poses.shape != (len(arrays['rgb']), 4, 4) or masks.shape != arrays['depth_m'].shape:
        raise ValueError('位姿或 mask 数量与采集帧不一致')
    triangles = load_mesh(cfg)
    reports = [quality(triangles, p, depth, mask, arrays['K'], cfg)
               for p, depth, mask in zip(poses, arrays['depth_m'], masks)]
    camera_pose, camera_errors = pose_average(poses)
    base_poses, hand_eye_meta = None, None
    if hand_eye:
        base_color, hand_eye_meta = base_color_transforms(hand_eye, capture_meta)
        base_poses = np.array([p@q for p, q in zip(base_color, poses)])
    average, deviations = pose_average(base_poses if base_poses is not None else poses)
    reasons = []
    if not all(r['passed'] for r in reports):
        reasons.append('rgbd_geometry_quality_failed')
    if (deviations[:, 0].max() > cfg['max_static_translation_m']
            or deviations[:, 1].max() > cfg['max_static_rotation_rad']):
        reasons.append('multi_frame_pose_unstable')
    if len(poses) < 3:
        reasons.append('need_at_least_three_independent_frames')
    if capture_meta.get('synthetic'):
        reasons.append('synthetic_capture')
    if backend != 'foundationpose':
        reasons.append('diagnostic_pose_backend')
    if hand_eye_meta is None:
        reasons.append('hand_eye_missing')
    else:
        reasons.extend(hand_eye_meta['reasons'])
    output = dict(schema_version=1, kind='object_observation', object_id=cfg['object_id'],
                  created_utc=utc_now(), observed_unix_s=float(max(f['host_unix_s'] for f in capture_meta['frames'])),
                  synthetic=bool(capture_meta['synthetic'] or (hand_eye_meta is not None and 'synthetic_hand_eye' in hand_eye_meta['reasons'])),
                  object_frame='original_mesh_scaled_to_m', mesh_sha256=digest(cfg['mesh_path']),
                  T_color_object=camera_pose.tolist(), T_color_object_per_frame=poses.tolist(),
                  T_base_object=average.tolist() if base_poses is not None else None,
                  config=capture_meta['config'], camera=capture_meta['camera'], hand_eye=hand_eye_meta,
                  backend=backend, quality_per_frame=reports,
                  max_translation_deviation_m=float(deviations[:, 0].max()),
                  max_rotation_deviation_rad=float(deviations[:, 1].max()),
                  usable_for_robot=not reasons, execution_blocked_reasons=sorted(set(reasons)),
                  max_observation_age_s=cfg['max_observation_age_s'])
    return output


def estimate(args):
    import cv2
    directory, capture_meta, arrays = read_capture(args.capture)
    cfg = object_config(args.object)
    load_mesh(cfg)  # Fail on unit/geometry errors before running large neural networks.
    output = new_directory(output_path(args.out))
    masks, mask_meta = get_masks(arrays['rgb'], cfg, output, method=args.mask_method,
                                python=args.sam3_python, checkpoint=args.sam3_checkpoint, source=args.sam3_source,
                                prompt=args.prompt, instance=args.instance, mask_file=args.mask_file,
                                box=getattr(args, 'box', None))
    for mask, depth in zip(masks, arrays['depth_m']):
        if np.count_nonzero(mask) < cfg['min_mask_pixels']:
            raise ValueError('目标 mask 太小')
        valid = mask & (depth >= cfg['min_depth_m']) & (depth <= cfg['max_depth_m'])
        if np.count_nonzero(valid)/np.count_nonzero(mask) < cfg['min_mask_depth_fraction']:
            raise ValueError('目标内有效深度太少；检查遮挡、反光、距离和深度配准')
    cv2.imwrite(str(output/'mask.png'), masks[0].astype(np.uint8)*255)
    rgb = arrays['rgb'][0]
    view = rgb.copy()
    view[masks[0]] = (view[masks[0]]*.5+np.array([0, 255, 0])*.5).astype(np.uint8)
    cv2.imwrite(str(output/'segmentation.png'), cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
    if args.diagnostic_poses:
        # An import path for other estimators / offline fixtures, never executable.
        with np.load(args.diagnostic_poses, allow_pickle=False) as data:
            poses = data['T_color_object']
        backend = 'diagnostic_import'
    else:
        python = args.pose_python or os.environ.get('TPKMP_POSE_PYTHON', str(ROOT/'envs/foundationpose/bin/python'))
        root = Path(args.foundationpose_root or ROOT/'third_party/FoundationPose').resolve()
        if not Path(python).is_file():
            raise ValueError('FoundationPose Python 尚未安装；运行 setup.sh pose-env 或设置 TPKMP_POSE_PYTHON')
        with (output/'pose_input.npz').open('xb') as stream:
            np.savez_compressed(stream, rgb=arrays['rgb'], depth_m=arrays['depth_m'], K=arrays['K'], masks=masks)
        request = dict(foundationpose_root=str(root), mesh_path=cfg['mesh_path'],
                       mesh_scale_to_m=.001 if cfg['mesh_units'] == 'mm' else 1., mesh_rgb=cfg.get('mesh_rgb'),
                       input=str(output/'pose_input.npz'), output=str(output/'poses.npz'),
                       iterations=args.iterations, debug_dir=str(output/'foundationpose_debug'))
        write_json(output/'pose_request.json', request)
        print(f'FoundationPose 配准 {len(masks)} 帧；首次运行可能需要编译 CUDA 内核。', flush=True)
        environment = os.environ.copy()
        toolchain = ROOT/'envs/cuda_toolchain'
        if 'CUDA_HOME' not in environment and (toolchain/'bin/nvcc').is_file():
            environment['CUDA_HOME'] = str(toolchain)
        if environment.get('CUDA_HOME'):
            toolchain = Path(environment['CUDA_HOME'])
            environment['PATH'] = str(toolchain/'bin')+os.pathsep+environment.get('PATH', '')
            for key, directories in (('CPATH', [toolchain/'targets/x86_64-linux/include']),
                                     ('LIBRARY_PATH', [toolchain/'targets/x86_64-linux/lib']),
                                     ('LD_LIBRARY_PATH', [toolchain/'lib', toolchain/'targets/x86_64-linux/lib'])):
                environment[key] = os.pathsep.join(map(str, directories))+os.pathsep+environment.get(key, '')
        environment.setdefault('MAX_JOBS', '2')
        with (output/'foundationpose.log').open('x') as log:
            try:
                subprocess.run([python, '-B', str(ROOT/'foundationpose_worker.py'), '--request', str(output/'pose_request.json')],
                               check=True, timeout=900, stdout=log, stderr=subprocess.STDOUT, env=environment)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError(f'FoundationPose 运行失败，详见 {output / "foundationpose.log"}') from exc
        with np.load(output/'poses.npz', allow_pickle=False) as data:
            poses = data['T_color_object']
        backend = 'foundationpose'
    result = process_poses(capture_meta, arrays, masks, poses, cfg, args.hand_eye, backend)
    result.update(capture_path=str(directory), capture_sha256=digest(directory/'capture.json'), segmentation=mask_meta)
    write_json(output/'observation.json', result)
    view = overlay(rgb, arrays['K'], checked(result['T_color_object']), bounds(load_mesh(cfg)),
                   'PASS' if result['usable_for_robot'] else 'DIAGNOSTIC / BLOCKED')
    cv2.imwrite(str(output/'overlay.png'), view)
    print(f'相机中的物体位姿：\n{np.array(result["T_color_object"])}')
    if result['T_base_object'] is not None:
        print(f'基座中的物体位姿：\n{np.array(result["T_base_object"])}')
    print(f'可用于真机目标={result["usable_for_robot"]}；阻止原因={result["execution_blocked_reasons"]}')
    print(f'结果：{output / "observation.json"}\n叠加图：{output / "overlay.png"}')
