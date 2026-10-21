"""Explicit eye-in/eye-to-hand transforms and provenance checks."""
from ..bootstrap import read_json, digest
from ..data import physical_signature
from ..kinematics import fk_matrix
from .common import checked
import numpy as np


def base_color_transforms(hand_eye_path, capture):
    result = read_json(hand_eye_path)
    mode = result.get('mode')
    name = {'eye-in-hand': 'T_flange_color', 'eye-to-hand': 'T_base_color'}.get(mode)
    if result.get('schema_version') != 1 or name is None or result.get('transform_name') != name:
        raise ValueError('需要本项目手眼标定 result.json，且变换名称与标定模式一致')
    signature = result['signature']
    model, expected = capture['source_camera'], signature['camera']
    for key in ('serial', 'width', 'height', 'fps', 'frame'):
        if model.get(key) != expected.get(key):
            raise ValueError(f'手眼标定与当前相机的 {key} 不一致')
    if (not np.allclose(model['K'], expected['K'], atol=1e-5, rtol=1e-7)
            or model['distortion_model'] != expected['distortion_model']
            or not np.allclose(model['distortion_coefficients'], expected['distortion_coefficients'], atol=1e-8)):
        raise ValueError('手眼标定使用的内参/畸变与本次拍照不同；需使用同组内参重新求解手眼')
    robot = signature['robot']
    cfg = capture['config']
    reasons = []
    if result.get('passed') is not True:
        reasons.append('hand_eye_validation_failed')
    if result.get('synthetic') or signature.get('synthetic'):
        reasons.append('synthetic_hand_eye')
    if robot.get('joint_source') != 'measured':
        reasons.append('hand_eye_used_leader_instead_of_follower')
    if robot.get('arm_id') != cfg['arm_id'] or robot.get('parent_frame') != cfg['base_frame'] or robot.get('firmware') != cfg['firmware']:
        reasons.append('hand_eye_robot_identity_mismatch')
    if model.get('passed') is False:
        reasons.append('intrinsics_validation_failed')
    value = result[name]
    transform = checked(value['matrix'] if isinstance(value, dict) else value)
    matrices = []
    for frame in capture['frames']:
        if mode == 'eye-in-hand':
            sample = frame.get('robot')
            if sample is None or sample.get('joint_source') != 'measured':
                raise ValueError('眼在手上需要每帧配对的从臂 measured 位姿；不能用采样后的当前位置代替')
            if sample.get('signature') != physical_signature(cfg):
                raise ValueError('拍照时从臂/TCP 配置不一致')
            flange = checked(sample['T_base_flange'])
            if not np.allclose(fk_matrix(sample['joints_rad']), flange, atol=1e-7):
                raise ValueError('拍照时关节角与法兰 FK 不一致')
            matrices.append(flange@transform)
        else:
            matrices.append(transform.copy())
    return np.array(matrices), dict(path=str(hand_eye_path), sha256=digest(hand_eye_path), mode=mode,
                                    passed=not reasons, reasons=reasons)
