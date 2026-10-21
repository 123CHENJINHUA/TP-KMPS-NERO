"""Simulation records use the same TP-KMP arrays, with an explicit simulation tag."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from nero_bridge.data import common_metadata, metadata, poses, save_npz


def demo_arrays(rows, joints, target):
    rows, joints, target = np.asarray(rows), np.asarray(joints), poses(target)
    if rows.ndim != 2 or rows.shape[1] != 8 or len(rows) < 3 or not np.isfinite(rows).all():
        raise ValueError("Record at least three finite samples")
    if joints.shape != (len(rows), 7) or not np.isfinite(joints).all():
        raise ValueError("Expected seven actual joint positions per sample")
    if np.any(np.diff(rows[:, 0]) <= 0):
        raise ValueError("Simulation sample times must increase")
    poses(rows[:, 1:])
    return dict(demo=rows, joint_positions=joints, frames=target[None], frame_names=["target"],
                source="isaacsim-5.1", **common_metadata("base_link", "gripper_base", True))


def save_demo(path, rows, joints, target):
    save_npz(path, **demo_arrays(rows, joints, target))


def require_sim(data):
    if metadata(data) != ("base_link", "gripper_base") or not bool(data["synthetic"]):
        raise ValueError("Use a simulation dataset in base_link / gripper_base")
    return data


def shifted_frames(data, offset):
    require_sim(data)
    if data["frame_names"].tolist() != ["target"]:
        raise ValueError("This scene uses one task frame named target")
    target = poses(data["frames"])
    if target.shape != (1, 7):
        raise ValueError("Expected one target pose")
    offset = np.asarray(offset, dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("Offset must contain three finite meters")
    target[0, :3] += offset
    return dict(base_frame="base_link", tcp_frame="gripper_base", frames={"target": target[0].tolist()})


def metrics(desired, actual, position_tolerance=.01, angle_tolerance=.1):
    desired, actual = poses(desired), poses(actual)
    if desired.shape != actual.shape or desired.ndim != 2 or len(desired) < 2:
        raise ValueError("Need matching desired and actual trajectory arrays")
    distance = np.linalg.norm(actual[:, :3] - desired[:, :3], axis=1)
    angle = (Rotation.from_quat(actual[:, [4, 5, 6, 3]])
             * Rotation.from_quat(desired[:, [4, 5, 6, 3]]).inv()).magnitude()
    return dict(samples=len(desired), position_rmse_m=float(np.sqrt(np.mean(distance**2))),
                position_max_m=float(distance.max()), position_final_m=float(distance[-1]),
                angle_rmse_rad=float(np.sqrt(np.mean(angle**2))), angle_max_rad=float(angle.max()),
                angle_final_rad=float(angle[-1]), position_tolerance_m=position_tolerance,
                angle_tolerance_rad=angle_tolerance,
                passed=bool(distance.max() <= position_tolerance and angle.max() <= angle_tolerance))


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
