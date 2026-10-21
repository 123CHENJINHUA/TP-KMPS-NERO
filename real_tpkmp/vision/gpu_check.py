"""Actual neural inference on synthetic CAD images, including a translated object.

No known poses are passed into either network. All persistent outputs stay synthetic.
Repeated synthetic frames are not a real-camera noise or accuracy benchmark.
"""
from .common import read_capture, output_path, new_directory, write_json, digest, read_json, utc_now
from .geometry import load_mesh, render_depth
from .smoke import fixture
from .estimate import estimate
from .bindings import bind, export_frames
from ..data import read_record
from ..kinematics import pose_matrix
from ..make_synthetic import generate
from ..bootstrap import PROJECT
import argparse
import copy
import subprocess
import sys
import time
import numpy as np


def shifted_capture(source, target, cfg):
    source, meta, data = read_capture(source)
    output = new_directory(target)
    with np.load(source/'known_poses.npz', allow_pickle=False) as archive:
        pose = archive['T_color_object'][0].copy()
    pose[0, 3] += .008
    depth = render_depth(load_mesh(cfg), pose, data['K'], data['depth_m'].shape[1:]).astype(np.float32)
    rgb = np.full(data['rgb'].shape[1:], 100, np.uint8)
    rgb[depth > 0] = [60, 150, 210]
    with (output/'rgbd.npz').open('xb') as stream:
        np.savez_compressed(stream, rgb=np.repeat(rgb[None], 3, axis=0), depth_m=np.repeat(depth[None], 3, axis=0), K=data['K'])
    meta = copy.deepcopy(meta)
    meta['created_utc'] = utc_now()
    for frame in meta['frames']:
        frame['host_unix_s'] = time.time()
    meta['rgbd_sha256'] = digest(output/'rgbd.npz')
    write_json(output/'capture.json', meta)
    # Ground truth is stored for developer inspection only, not provided to inference.
    with (output/'known_poses.npz').open('xb') as stream:
        np.savez_compressed(stream, T_color_object=np.repeat(pose[None], 3, axis=0))
    return output


def gpu_check(args):
    from .common import object_config
    output = new_directory(output_path(args.out))
    dataset = generate(output/'dataset')
    _, demo = read_record(dataset/'demo_001.npz')
    reference, _, _ = fixture(output/'reference_capture', pose_matrix(demo['frames'][1]), synthetic=True)
    options = dict(object=None, mask_method='sam3', mask_file=None, prompt='blue object', instance=None,
                   sam3_python=None, sam3_checkpoint=None, sam3_source=None, pose_python=None,
                   foundationpose_root=None, iterations=5, diagnostic_poses=None, hand_eye=str(reference/'hand_eye.json'))
    estimate(argparse.Namespace(capture=str(reference), out=str(output/'reference_pose'), **options))
    bind(argparse.Namespace(observation=str(output/'reference_pose/observation.json'), demo=str(dataset/'demo_001.npz'),
                            config=None, can=None, out=str(output/'grasp.json')))
    shifted = shifted_capture(reference, output/'new_capture', object_config())
    estimate(argparse.Namespace(capture=str(shifted), out=str(output/'new_pose'), **options))
    export_frames(argparse.Namespace(observation=str(output/'new_pose/observation.json'), grasp=str(output/'grasp.json'),
                                     template_frames=str(dataset/'demo_001_frames.json'), config=None, out=str(output/'frames.json')))
    subprocess.run([sys.executable, '-B', '-m', 'real_tpkmp.train', '--dataset', str(dataset), '--out', str(output/'model')], cwd=PROJECT, check=True)
    subprocess.run([sys.executable, '-B', '-m', 'real_tpkmp.test_real', '--model', str(output/'model/model.npz'),
                    '--frames', str(output/'frames.json'), '--out', str(output/'preview')], cwd=PROJECT, check=True)
    current = read_json(output/'new_pose/observation.json')
    previous = read_json(output/'reference_pose/observation.json')
    plan_meta, _ = read_record(output/'preview/plan.npz', 'plan')
    if not plan_meta['synthetic'] or plan_meta['executable']:
        raise AssertionError('GPU 合成联调计划未保留合成/不可执行标记')
    passed = all(q['passed'] for r in (previous, current) for q in r['quality_per_frame'])
    write_json(output/'checks.json', dict(passed=passed, synthetic=True, neural_network_inference_tested=True,
                 ground_truth_passed_to_network=False, object_translation_color_m=[.008, 0, 0],
                 opens_camera=False, opens_can=False, plan_executable=plan_meta['executable'],
                 reference_quality=previous['quality_per_frame'], new_quality=current['quality_per_frame'],
                 notes='Repeated rendered frames test software integration, not real-camera precision or temporal noise.'))
    if not passed:
        raise ValueError(f'网络几何检查未全部通过，见 {output}/checks.json')
    print(f'GPU 视觉新位置联调通过：{output}/checks.json；未连接相机/CAN，合成计划不能真机执行。')
