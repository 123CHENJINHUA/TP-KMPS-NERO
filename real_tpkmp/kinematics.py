"""Nero SDK MDH geometry; offline, bounded IK and task-space transforms."""
from . import bootstrap
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from pyAgxArm.api.constants import ROBOT_MDH_PRESET

MDH = np.asarray(ROBOT_MDH_PRESET['nero'], float)


def pose_matrix(pose):
    pose = np.asarray(pose, float)
    if pose.shape != (7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:]) - 1) > .01:
        raise ValueError('位姿需要 [x,y,z,qw,qx,qy,qz]，单位 m，四元数必须归一化')
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def matrix_pose(matrix):
    matrix = np.asarray(matrix, float)
    return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]]


def fk_matrix(joints):
    joints = np.asarray(joints, float)
    if joints.shape != (7,) or not np.isfinite(joints).all():
        raise ValueError('需要七个有限的关节角，单位 rad')
    matrix = np.eye(4)
    for q, (d, a, alpha, offset) in zip(joints, MDH):
        c, s = np.cos(q + offset), np.sin(q + offset)
        ca, sa = np.cos(alpha), np.sin(alpha)
        matrix = matrix @ np.array([[c, -s, 0, a], [ca*s, ca*c, -sa, -sa*d],
                                     [sa*s, sa*c, ca, ca*d], [0, 0, 0, 1]])
    return matrix


def tcp_pose(joints, cfg):
    return matrix_pose(fk_matrix(joints) @ pose_matrix(cfg['T_flange_tcp_pose_wxyz']))


def pose_error(first, second):
    a, b = pose_matrix(first), pose_matrix(second)
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), float(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())


def inverse_path(poses, seed, reference, cfg):
    low, high = np.array(cfg['joint_min_rad']), np.array(cfg['joint_max_rad'])
    tool_inverse = np.linalg.inv(pose_matrix(cfg['T_flange_tcp_pose_wxyz']))
    previous = np.clip(np.asarray(seed, float), low + 1e-8, high - 1e-8)
    path = []
    for index, (pose, preferred) in enumerate(zip(poses, reference)):
        target = pose_matrix(pose) @ tool_inverse
        def residual(q):
            current = fk_matrix(q)
            return np.r_[(current[:3, 3] - target[:3, 3]) / .002,
                         Rotation.from_matrix(target[:3, :3].T @ current[:3, :3]).as_rotvec() / .01,
                         .015 * (q - preferred)]
        solution = least_squares(residual, previous, bounds=(low + 1e-8, high - 1e-8),
                                 max_nfev=120, ftol=1e-9, xtol=1e-9, gtol=1e-8)
        actual = tcp_pose(solution.x, cfg)
        position, rotation = pose_error(actual, pose)
        if (not solution.success or position > cfg['ik_position_tolerance_m']
                or rotation > cfg['ik_rotation_tolerance_rad']):
            raise ValueError(f'轨迹点 {index} IK 不通过：{position*1000:.2f} mm / {np.rad2deg(rotation):.2f}°')
        if path and np.max(np.abs(solution.x - previous)) > .25:
            raise ValueError(f'轨迹点 {index} IK 关节分支跳变；需缩小泛化范围或调整示教')
        previous = solution.x
        path.append(previous.copy())
    return np.asarray(path)
