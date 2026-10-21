"""Shared file format: meters, quaternion [w, x, y, z], Nero base frame."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def poses(values):
    result = np.asarray(values, dtype=float).copy()
    if result.ndim < 1 or result.shape[-1] != 7 or not np.isfinite(result).all():
        raise ValueError("Poses must be finite [..., 7] arrays: x y z qw qx qy qz (meters)")
    norms = np.linalg.norm(result[..., 3:], axis=-1)
    if np.any(np.abs(norms - 1.0) > 0.01):
        raise ValueError("Quaternion norm must be within 0.01 of 1")
    result[..., 3:] /= norms[..., None]
    return result


def read_frames(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not data.get("base_frame") or not data.get("tcp_frame") or not data.get("frames"):
        raise ValueError("Frames JSON needs base_frame, tcp_frame and a nonempty frames mapping")
    names = list(data["frames"])
    values = poses([data["frames"][name] for name in names])
    if "robot_base" in names and not np.allclose(values[names.index("robot_base")], [0, 0, 0, 1, 0, 0, 0]):
        raise ValueError("robot_base must be the identity pose in base_frame")
    return data["base_frame"], data["tcp_frame"], names, values


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects earlier recordings and reviewed trajectories.
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def metadata(data):
    if str(data["robot"]) != "nero" or str(data["units"]) != "m,wxyz":
        raise ValueError("Expected a Nero file in m,wxyz format; re-record old robot skills")
    base, tcp = str(data["base_frame"]), str(data["tcp_frame"])
    if not base or not tcp:
        raise ValueError("Missing base/TCP frame")
    if np.asarray(data["synthetic"]).shape != () or data["synthetic"].dtype.kind != "b":
        raise ValueError("synthetic must be a scalar boolean")
    return base, tcp


def common_metadata(base, tcp, synthetic=False):
    return dict(robot="nero", units="m,wxyz", base_frame=base, tcp_frame=tcp, synthetic=synthetic)


def resample_demo(demo, count):
    demo = np.asarray(demo, dtype=float)
    if demo.ndim != 2 or demo.shape[1] != 8 or len(demo) < 3 or not np.isfinite(demo).all():
        raise ValueError("Demonstration must contain at least three finite [t, pose7] rows")
    time = demo[:, 0]
    if np.any(np.diff(time) <= 0) or count < 3:
        raise ValueError("Demonstration times must strictly increase and samples must be >= 3")
    pose = poses(demo[:, 1:])
    phase = (time - time[0]) / (time[-1] - time[0])
    target = np.linspace(0, 1, count)
    # ponytail: phase interpolation assumes matching action stages; add DTW for unequal pauses.
    xyz = np.column_stack([np.interp(target, phase, pose[:, i]) for i in range(3)])
    quat = Slerp(phase, Rotation.from_quat(pose[:, [4, 5, 6, 3]]))(target).as_quat()
    return np.column_stack([target, xyz, quat[:, [3, 0, 1, 2]]])


def validate_path(trajectory, config):
    trajectory = poses(trajectory)
    if trajectory.ndim != 2 or len(trajectory) < 2:
        raise ValueError("Trajectory must have at least two pose7 waypoints")
    low, high = np.asarray(config["workspace_min"]), np.asarray(config["workspace_max"])
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or np.any(low >= high):
        raise ValueError("Invalid workspace bounds")
    if np.any(trajectory[:, :3] < low) or np.any(trajectory[:, :3] > high):
        raise ValueError("Trajectory leaves the configured workspace")
    if np.max(np.linalg.norm(np.diff(trajectory[:, :3], axis=0), axis=1)) > config["max_cartesian_step"]:
        raise ValueError("Trajectory contains an excessive Cartesian step")
    rotation = Rotation.from_quat(trajectory[:, [4, 5, 6, 3]])
    if np.max((rotation[1:] * rotation[:-1].inv()).magnitude()) > config["max_rotation_step"]:
        raise ValueError("Trajectory contains an excessive orientation step")
    return trajectory


def check_start(current, target, config):
    current, target = poses(current), poses(target)
    distance = np.linalg.norm(current[:3] - target[:3])
    rotations = Rotation.from_quat(np.asarray([current, target])[:, [4, 5, 6, 3]])
    angle = (rotations[0].inv() * rotations[1]).magnitude()
    if distance > config["start_position_tolerance"] or angle > config["start_angle_tolerance"]:
        raise ValueError(f"Move near the start first: position error {distance:.4f} m, angle error {angle:.4f} rad")


def read_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for name in (
        "max_cartesian_step", "max_rotation_step", "start_position_tolerance", "start_angle_tolerance",
        "max_joint_step", "max_joint_speed", "max_joint_acceleration", "state_max_age", "timeout", "cartesian_resolution",
    ):
        value = config[name]
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    return config
