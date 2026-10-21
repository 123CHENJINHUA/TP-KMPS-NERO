"""T_parent_child maps child coordinates into parent coordinates; metres throughout."""
import bootstrap  # noqa: F401
import numpy as np
from scipy.spatial.transform import Rotation


def checked(value):
    T = np.asarray(value, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError('位姿必须是有限的 4×4 矩阵')
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError('齐次矩阵最后一行必须为 [0,0,0,1]')
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1, atol=1e-5):
        raise ValueError('旋转矩阵不是有效的 SO(3) 旋转')
    return T.copy()


def inverse(T):
    T = checked(T)
    result = np.eye(4)
    result[:3, :3] = T[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ T[:3, 3]
    return result


def from_pose6(pose):
    """SDK xyz in m; roll,pitch,yaw in rad; R=Rz(yaw)Ry(pitch)Rx(roll)."""
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError('pose6 需要 6 个有限数值，单位 m/rad')
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', pose[3:]).as_matrix()
    T[:3, 3] = pose[:3]
    return T


def pack(T):
    T = checked(T)
    return np.r_[Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]


def unpack(value):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(value[:3]).as_matrix()
    T[:3, 3] = value[3:]
    return T


def mean_pose(poses):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_matrix(np.array([p[:3, :3] for p in poses])).mean().as_matrix()
    result[:3, 3] = np.mean([p[:3, 3] for p in poses], axis=0)
    return result


def distance(first, second):
    delta = inverse(first) @ checked(second)
    return (float(np.linalg.norm(delta[:3, 3])),
            float(Rotation.from_matrix(delta[:3, :3]).magnitude()))


def describe(T):
    T = checked(T)
    return {'matrix': T.tolist(), 'translation_m': T[:3, 3].tolist(),
            'quaternion_xyzw': Rotation.from_matrix(T[:3, :3]).as_quat().tolist(),
            'rpy_xyz_deg': Rotation.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True).tolist()}
