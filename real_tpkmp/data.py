"""Versioned recording format and six-stage phase alignment."""
from .bootstrap import HERE, read_json, inside
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from nero_bridge.data import save_npz, load_npz
from .kinematics import pose_matrix

STAGES = ('approach', 'grasp', 'lift', 'transfer', 'release', 'retreat')
FRAME_NAMES = ('start', 'pick', 'place')


def config(path=None, channel=None):
    cfg = read_json(path or HERE / 'config.json')
    if channel:
        cfg['channel'] = channel
    low, high = np.array(cfg['joint_min_rad']), np.array(cfg['joint_max_rad'])
    if low.shape != (7,) or high.shape != (7,) or not np.isfinite([low, high]).all() or np.any(low >= high):
        raise ValueError('关节范围配置无效')
    bounds = np.array([cfg['workspace_min_m'], cfg['workspace_max_m']], float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
        raise ValueError('工作空间配置无效')
    pose_matrix(cfg['T_flange_tcp_pose_wxyz'])
    for key, value in cfg.items():
        if isinstance(value, (int, float)) and (not np.isfinite(value) or value <= 0):
            raise ValueError(f'{key} 必须为正数')
    if not 5 <= cfg['control_hz'] <= 100 or not 1 <= cfg['sdk_speed_percent'] <= 20:
        raise ValueError('首版 control_hz 范围 5..100，SDK 速度比例范围 1..20')
    for key in ('gripper_width_limits_m', 'gripper_angle_limits_deg'):
        values = np.asarray(cfg[key], float)
        if values.shape != (2,) or not np.isfinite(values).all() or values[0] >= values[1]:
            raise ValueError(f'{key} 无效')
    boxes = np.asarray(cfg['keepout_aabbs_m'], float)
    if boxes.size and (boxes.ndim != 3 or boxes.shape[1:] != (2, 3)
                       or not np.isfinite(boxes).all() or np.any(boxes[:, 0] >= boxes[:, 1])):
        raise ValueError('禁入区需要 [[min_xyz,max_xyz], ...]')
    return cfg


def physical_signature(cfg):
    return {key: cfg[key] for key in ('arm_id', 'firmware', 'base_frame', 'tcp_frame', 'T_flange_tcp_pose_wxyz')}


def save_record(path, meta, **arrays):
    save_npz(inside(path), meta_json=json.dumps(meta, ensure_ascii=False, allow_nan=False), **arrays)


def read_record(path, kind=None):
    data = load_npz(path)
    meta = json.loads(str(data.pop('meta_json')))
    if meta.get('schema_version') != 1 or meta.get('robot') != 'nero' or not isinstance(meta.get('synthetic'), bool):
        raise ValueError('不支持的真机数据格式')
    if kind and meta.get('kind') != kind:
        raise ValueError(f'需要 {kind} 文件')
    return meta, data


def read_frames(path, cfg):
    data = read_json(path)
    if (data.get('base_frame'), data.get('tcp_frame')) != (cfg['base_frame'], cfg['tcp_frame']):
        raise ValueError('参考系文件 base_frame / tcp_frame 不匹配')
    if data.get('signature', physical_signature(cfg)) != physical_signature(cfg):
        raise ValueError('参考系文件的机械臂/TCP 安装配置不匹配')
    if data.get('units') != 'm,wxyz':
        raise ValueError('参考系文件单位需要 m,wxyz')
    if set(data.get('frames', {})) != set(FRAME_NAMES):
        raise ValueError('参考系需要 start、pick、place 三个位姿')
    result = np.array([data['frames'][name] for name in FRAME_NAMES], float)
    for pose in result:
        pose_matrix(pose)
    return data, result


def frame_document(frames, cfg, synthetic=False, **extra):
    return dict(base_frame=cfg['base_frame'], tcp_frame=cfg['tcp_frame'], units='m,wxyz', signature=physical_signature(cfg),
                synthetic=bool(synthetic), frames={name: np.asarray(pose).tolist() for name, pose in zip(FRAME_NAMES, frames)}, **extra)


def align(data, count, cfg):
    if count < 24 or count % len(STAGES):
        raise ValueError('samples 需为至少 24 的 6 倍数')
    t, stage, poses = np.asarray(data['time_s']), np.asarray(data['stage']), np.asarray(data['tcp'])
    n = len(t)
    for key, shape in (('tcp', (n, 7)), ('joints_rad', (n, 7)), ('gripper_measured', (n,)),
                       ('gripper_command', (n,)), ('gripper_force_N', (n,))):
        if np.asarray(data[key]).shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f'示教字段 {key} 无效')
    if (n < 18 or stage.shape != (n,) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0)
            or np.max(np.diff(t)) > cfg['max_sample_gap_s'] or np.any(np.diff(stage) < 0)
            or set(stage.tolist()) != set(range(6))):
        raise ValueError('示教需要六个依次递增的阶段、递增时间且无反馈断档')
    if not np.issubdtype(stage.dtype, np.integer):
        raise ValueError('stage 必须为整数')
    if np.any(np.asarray(data['joints_rad']) < cfg['joint_min_rad']) or np.any(np.asarray(data['joints_rad']) > cfg['joint_max_rad']):
        raise ValueError('采集关节超出当前配置范围')
    phase, out, joints, command, measured, force, durations = [], [], [], [], [], [], []
    for i in range(6):
        indices = np.flatnonzero(stage == i)
        if len(indices) < 3 or t[indices[-1]] - t[indices[0]] < cfg['minimum_stage_duration_s']:
            raise ValueError(f'阶段 {i+1} {STAGES[i]} 太短；需至少 3 个样本和 {cfg["minimum_stage_duration_s"]} 秒')
        start = indices[0]
        end = indices[-1] + (1 if i < 5 else 0)
        source = np.arange(start, end + 1)
        grid = np.linspace(0, 1, count)
        part = grid[(grid >= i/6) & ((grid < (i+1)/6) if i < 5 else (grid <= 1))]
        target = t[start] + (part*6-i) * (t[end]-t[start])
        source_poses = poses[source]
        rotations = Slerp(t[source], Rotation.from_quat(source_poses[:, [4, 5, 6, 3]]))(target).as_quat()
        xyz = np.column_stack([np.interp(target, t[source], source_poses[:, k]) for k in range(3)])
        phase.extend(part)
        out.extend(np.c_[xyz, rotations[:, [3, 0, 1, 2]]])
        joints.extend(np.column_stack([np.interp(target, t[source], data['joints_rad'][source, k]) for k in range(7)]))
        held = np.clip(np.searchsorted(t, target, side='right') - 1, indices[0], indices[-1])
        command.extend(data['gripper_command'][held])
        measured.extend(data['gripper_measured'][held])
        force.extend(data['gripper_force_N'][held])
        durations.append(t[end] - t[start])
    return dict(phase=np.array(phase), tcp=np.array(out), joints=np.array(joints),
                command=np.array(command), measured=np.array(measured), force=np.array(force), durations=np.array(durations))
