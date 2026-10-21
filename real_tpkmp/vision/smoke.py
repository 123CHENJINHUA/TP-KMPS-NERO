"""Known-pose synthetic RGB-D tests the plumbing, not neural-network accuracy."""
from .common import ROOT, output_path, new_directory, object_config, write_json, digest, utc_now
from .geometry import load_mesh, render_depth, bounds
from .estimate import estimate
from .bindings import bind, export_frames
from ..bootstrap import PROJECT
from ..data import config, physical_signature, read_record
from ..kinematics import fk_matrix, pose_matrix
from ..make_synthetic import generate
import argparse
import subprocess
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation


def fixture(directory, base_tcp=None, synthetic=True):
    import cv2
    cfg, obj = config(), object_config()
    triangles = load_mesh(obj)
    output = new_directory(directory)
    height, width = 240, 320
    K = np.array([[450., 0, width/2], [0, 450., height/2], [0, 0, 1.]])
    T_color_object = np.eye(4)
    T_color_object[:3, :3] = Rotation.from_euler('xyz', [np.pi, 0, .15]).as_matrix()
    T_color_object[:3, 3] = np.array([0, 0, .55])-T_color_object[:3, :3]@bounds(triangles).mean(axis=0)
    depth = render_depth(triangles, T_color_object, K, (height, width)).astype(np.float32)
    mask = depth > 0
    rgb = np.full((height, width, 3), 100, np.uint8)
    rgb[mask] = [60, 150, 210]
    camera = dict(serial='synthetic_d435i', width=width, height=height, fps=30, frame='color_optical',
                  K=K.tolist(), distortion_model='none', distortion_coefficients=[0.]*5, rectified=True, registered_depth=True)
    q = np.array([-.3, -.3, .3, 1., .1, .1, 0.])
    flange = fk_matrix(q)
    base_tcp = flange if base_tcp is None else base_tcp
    T_object_tcp = np.eye(4)
    T_object_tcp[:3, 3] = [.075, .049, .15]
    T_base_object = base_tcp@np.linalg.inv(T_object_tcp)
    T_flange_color = np.linalg.inv(flange)@T_base_object@np.linalg.inv(T_color_object)
    frames = [dict(host_unix_s=time.time()+i*.0001, frame_number=i, robot=dict(
              joints_rad=q.tolist(), T_base_flange=flange.tolist(), joint_source='measured', signature=physical_signature(cfg))) for i in range(3)]
    with (output/'rgbd.npz').open('xb') as stream:
        np.savez_compressed(stream, rgb=np.repeat(rgb[None], 3, axis=0), depth_m=np.repeat(depth[None], 3, axis=0), K=K)
    meta = dict(schema_version=1, kind='rgbd_capture', synthetic=synthetic, config=cfg, frames=frames,
                camera=camera, source_camera=camera, static_capture=True, created_utc=utc_now(), rgbd_sha256=digest(output/'rgbd.npz'))
    write_json(output/'capture.json', meta)
    robot = dict(arm_id=cfg['arm_id'], firmware=cfg['firmware'], parent_frame=cfg['base_frame'], joint_source='measured')
    hand_eye = dict(schema_version=1, mode='eye-in-hand', transform_name='T_flange_color',
                    T_flange_color=dict(matrix=T_flange_color.tolist()), passed=True, synthetic=synthetic,
                    signature=dict(robot=robot, camera=camera, synthetic=synthetic))
    write_json(output/'hand_eye.json', hand_eye)
    with (output/'known_poses.npz').open('xb') as stream:
        np.savez_compressed(stream, T_color_object=np.repeat(T_color_object[None], 3, axis=0))
    cv2.imwrite(str(output/'rgb.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return output, T_color_object, T_base_object


def smoke(args):
    output = new_directory(output_path(args.out))
    dataset = generate(output/'dataset')
    _, demo = read_record(dataset/'demo_001.npz')
    capture, _, _ = fixture(output/'capture', pose_matrix(demo['frames'][1]), synthetic=True)
    options = argparse.Namespace(capture=str(capture), object=None, out=str(output/'estimate'), hand_eye=str(capture/'hand_eye.json'),
                  mask_method='hsv', mask_file=None, prompt=None, instance=None, sam3_python=None, sam3_checkpoint=None,
                  sam3_source=None, pose_python=None, foundationpose_root=None, iterations=5,
                  diagnostic_poses=str(capture/'known_poses.npz'))
    estimate(options)
    bind(argparse.Namespace(observation=str(output/'estimate/observation.json'), demo=str(dataset/'demo_001.npz'),
                            config=None, can=None, out=str(output/'grasp.json')))
    export_frames(argparse.Namespace(observation=str(output/'estimate/observation.json'), grasp=str(output/'grasp.json'),
                                     template_frames=str(dataset/'demo_001_frames.json'), config=None, out=str(output/'frames.json')))
    subprocess.run([sys.executable, '-B', '-m', 'real_tpkmp.train', '--dataset', str(dataset), '--out', str(output/'model')],
                   cwd=PROJECT, check=True)
    subprocess.run([sys.executable, '-B', '-m', 'real_tpkmp.test_real', '--model', str(output/'model/model.npz'),
                    '--frames', str(output/'frames.json'), '--out', str(output/'preview')], cwd=PROJECT, check=True)
    meta, data = read_record(output/'preview/plan.npz')
    from ..planning import verify_plan
    try:
        verify_plan(meta, data, config())
    except ValueError:
        pass
    else:
        raise AssertionError('Synthetic visual plan unexpectedly executable')
    write_json(output/'checks.json', dict(passed=True, synthetic=True, opens_can=False, opens_camera=False,
                neural_network_inference_tested=False, stages=['RGBD geometry', 'automatic HSV mask', 'hand-eye transforms',
                'object-to-TCP binding', 'visual task frames', 'TP-KMP training', 'bounded IK plan', 'execution rejection']))
    print(f'视觉离线链路通过：{output}/checks.json；已知位姿用于接口测试，不代表网络识别精度。')
