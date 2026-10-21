"""Task-frame prediction, contact-phase holds, bounded IK and conservative timing."""
from .bootstrap import digest
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp
from .data import read_record, read_frames, save_record, physical_signature
from .kinematics import pose_error, pose_matrix, tcp_pose, inverse_path
from .learning import predict


def interpolate_poses(phase, poses, targets):
    xyz = np.column_stack([np.interp(targets, phase, poses[:, j]) for j in range(3)])
    quat = Slerp(phase, Rotation.from_quat(poses[:, [4, 5, 6, 3]]))(targets).as_quat()
    return np.c_[xyz, quat[:, [3, 0, 1, 2]]]


def anchored(poses, frames):
    """Pin TCP to manually taught grasp/release references; never fit holdout data."""
    phase = np.linspace(0, 1, len(poses))
    knots = np.array([0., 1/6, 2/6, 4/6, 5/6, 1.])
    source = interpolate_poses(phase, poses, knots)
    target = np.array([frames[0], frames[1], frames[1], frames[2], frames[2], source[-1]])
    discrepancy = np.array([pose_error(a, b) for a, b in zip(source, target)])
    if discrepancy[:, 0].max() > .025 or discrepancy[:, 1].max() > .20:
        raise ValueError('学习轨迹在抓取/放置参考点偏差过大；需要改善示教和训练，不能强行修正')
    result = poses.copy()
    for j in range(3):
        result[:, j] += np.interp(phase, knots, target[:, j]-source[:, j])
    delta = Rotation.from_quat(target[:, [4, 5, 6, 3]]) * Rotation.from_quat(source[:, [4, 5, 6, 3]]).inv()
    rotations = Slerp(knots, delta)(phase) * Rotation.from_quat(poses[:, [4, 5, 6, 3]])
    result[:, 3:] = rotations.as_quat()[:, [3, 0, 1, 2]]
    for begin, end, pose in ((1/6, 2/6, frames[1]), (4/6, 5/6, frames[2])):
        result[(phase >= begin) & (phase <= end)] = pose
    # Include exact contact boundaries in the IK knot sequence.
    grid = np.unique(np.r_[phase, knots])
    result = interpolate_poses(phase, result, grid)
    for value, pose in zip(knots, target):
        result[np.argmin(abs(grid-value))] = pose
    return grid, result, discrepancy


def gripper_profile(phase, demonstrated):
    """Learn native command endpoints; perform closing/opening during contact holds."""
    source = np.linspace(0, 1, len(demonstrated))
    initial = float(demonstrated[0])
    closed = float(np.median(demonstrated[(source >= .28) & (source < 1/3)]))
    opened = float(np.median(demonstrated[(source >= .78) & (source < 5/6)]))
    def ramp(begin, end):
        u = np.clip((phase-begin)/(end-begin), 0, 1)
        return 10*u**3-15*u**4+6*u**5
    result = initial+(closed-initial)*ramp(1/6, 1/6+.6/6)+(opened-closed)*ramp(4/6, 4/6+.6/6)
    return np.clip(result, min(initial, closed, opened), max(initial, closed, opened))


def check_path(time_s, joints, tcp, gripper, mode, cfg):
    time_s, joints, tcp, gripper = map(np.asarray, (time_s, joints, tcp, gripper))
    n = len(time_s)
    if (n < 3 or time_s.shape != (n,) or joints.shape != (n, 7) or tcp.shape != (n, 7)
            or gripper.shape != (n,) or not all(np.isfinite(a).all() for a in (time_s, joints, tcp, gripper))
            or abs(time_s[0]) > 1e-9 or np.any(np.diff(time_s) <= 0)):
        raise ValueError('轨迹数组形状、有限值或时间轴无效')
    if time_s[-1] > cfg['max_duration_s']:
        raise ValueError(f'轨迹时长 {time_s[-1]:.1f}s 超出限制；需改善轨迹或调整配置后重新规划')
    if np.any(joints < cfg['joint_min_rad']) or np.any(joints > cfg['joint_max_rad']):
        raise ValueError('轨迹关节越界')
    if np.any(tcp[:, :3] < cfg['workspace_min_m']) or np.any(tcp[:, :3] > cfg['workspace_max_m']):
        raise ValueError('TCP 轨迹超出工作空间')
    for low, high in cfg['keepout_aabbs_m']:
        if np.any(np.all((tcp[:, :3] >= low) & (tcp[:, :3] <= high), axis=1)):
            raise ValueError('TCP 轨迹穿过配置的禁入区')
    if mode not in ('width', 'angle'):
        raise ValueError('夹爪模式无效')
    limits = cfg['gripper_width_limits_m' if mode == 'width' else 'gripper_angle_limits_deg']
    if np.any(gripper < limits[0]) or np.any(gripper > limits[1]):
        raise ValueError('夹爪目标越界')
    velocities = np.gradient(joints, time_s, axis=0)
    accelerations = np.gradient(velocities, time_s, axis=0)
    differences = np.array([pose_error(a, b) for a, b in zip(tcp[:-1], tcp[1:])])
    dt = np.diff(time_s)
    metrics = dict(duration_s=float(time_s[-1]), max_joint_speed_rad_s=float(np.max(abs(velocities))),
                   max_joint_acceleration_rad_s2=float(np.max(abs(accelerations))),
                   max_tcp_speed_m_s=float(np.max(differences[:, 0]/dt)),
                   max_tcp_angular_speed_rad_s=float(np.max(differences[:, 1]/dt)),
                   max_gripper_speed_native_s=float(np.max(abs(np.diff(gripper))/dt)))
    ratios = [metrics['max_joint_speed_rad_s']/cfg['max_joint_speed_rad_s'],
              np.sqrt(metrics['max_joint_acceleration_rad_s2']/cfg['max_joint_acceleration_rad_s2']),
              metrics['max_tcp_speed_m_s']/cfg['max_tcp_speed_m_s'],
              metrics['max_tcp_angular_speed_rad_s']/cfg['max_tcp_angular_speed_rad_s'],
              metrics['max_gripper_speed_native_s']/cfg['gripper_width_speed_m_s' if mode == 'width' else 'gripper_angle_speed_deg_s']]
    return metrics, max(ratios)


def make_plan(model_path, frames_path, cfg):
    meta, data = read_record(model_path, 'model')
    frame_meta, frames = read_frames(frames_path, cfg)
    from .vision.bindings import frame_reasons
    blocked_reasons = frame_reasons(frame_meta)
    if meta['signature'] != physical_signature(cfg):
        raise ValueError('模型与当前机械臂/TCP 配置不匹配')
    distances = [[pose_error(a, b) for a, b in zip(frames, old)] for old in data['frames']]
    valid = [all(p <= cfg['max_frame_translation_m'] and r <= cfg['max_frame_rotation_rad'] for p, r in row)
             for row in distances]
    if not any(valid):
        raise ValueError('新参考系离示教范围过远；先采集附近示教，或审核配置后扩大泛化范围')
    score = [sum(p+.1*r for p, r in row) if ok else np.inf for row, ok in zip(distances, valid)]
    nearest = int(np.argmin(score))
    raw = predict(meta, data, frames)
    phase, target, correction = anchored(raw, frames)
    source_phase = np.linspace(0, 1, meta['samples'])
    reference = np.column_stack([np.interp(phase, source_phase, data['reference_joints'][nearest, :, j]) for j in range(7)])
    q = inverse_path(target, reference[0], reference, cfg)
    joints_curve = PchipInterpolator(phase, q)
    commands = np.median(data['gripper_commands'], axis=0)
    forces = np.minimum(np.median(data['gripper_forces_N'], axis=0), cfg['gripper_max_force_N'])
    force_curve = PchipInterpolator(source_phase, forces)
    durations = np.median(data['stage_durations_s'], axis=0)
    stage_time = np.r_[0, np.cumsum(durations)]
    phase_curve = PchipInterpolator(stage_time/stage_time[-1], np.linspace(0, 1, 7))
    duration = float(stage_time[-1])
    for _ in range(6):
        t = np.linspace(0, duration, int(np.ceil(duration*cfg['control_hz']))+1)
        progress = t/duration
        smooth = 10*progress**3-15*progress**4+6*progress**5
        progress_phase = np.clip(phase_curve(smooth), 0, 1)
        dense_q = joints_curve(progress_phase)
        dense_tcp = np.array([tcp_pose(row, cfg) for row in dense_q])
        gripper = gripper_profile(progress_phase, commands)
        metrics, ratio = check_path(t, dense_q, dense_tcp, gripper, meta['gripper_mode'], cfg)
        if ratio <= 1.0001:
            break
        duration *= ratio*1.05
    else:
        raise ValueError('时间缩放未收敛')
    expected = interpolate_poses(phase, target, progress_phase)
    deviation = np.array([pose_error(a, b) for a, b in zip(dense_tcp, expected)])
    if deviation[:, 0].max() > .003 or deviation[:, 1].max() > .03:
        raise ValueError('关节插值偏离 TCP 轨迹；需要增加 --samples 后重新训练')
    plan_meta = dict(schema_version=1, robot='nero', kind='plan',
                     synthetic=bool(meta['synthetic'] or frame_meta.get('synthetic', True)),
                     config=cfg, signature=physical_signature(cfg), gripper_mode=meta['gripper_mode'],
                     model_path=str(Path(model_path).resolve()), model_sha256=digest(model_path),
                     frames_path=str(Path(frames_path).resolve()),
                     frames_sha256=digest(frames_path), execution_blocked_reasons=blocked_reasons,
                     validation_passed=bool(meta['validation_passed']),
                     executable=bool(meta['validation_passed'] and not meta['synthetic'] and not frame_meta.get('synthetic', True) and not blocked_reasons),
                     physical_task_success='not_tested')
    report = dict(**metrics, heldout_passed=meta['validation_passed'], synthetic=plan_meta['synthetic'],
                  executable=plan_meta['executable'], execution_blocked_reasons=blocked_reasons, nearest_training_demo=nearest,
                  anchor_correction_max_m=float(correction[:, 0].max()),
                  anchor_correction_max_rad=float(correction[:, 1].max()),
                  interpolation_max_position_error_m=float(deviation[:, 0].max()),
                  start_joints_deg=np.rad2deg(dense_q[0]).tolist(), start_tcp_pose_wxyz=dense_tcp[0].tolist(),
                  gripper_mode=meta['gripper_mode'], start_gripper_native=float(gripper[0]),
                  collision_check='TCP workspace and configured AABBs only; no full robot/scene collision model',
                  gripper_force_cap_N=cfg['gripper_max_force_N'])
    return plan_meta, dict(time_s=t, phase=progress_phase, joints_rad=dense_q, tcp=dense_tcp,
                           gripper_command=gripper, gripper_force_N=force_curve(progress_phase),
                           frames=frames, learned_tcp=raw), report


def verify_plan(meta, data, cfg):
    if meta.get('synthetic') or not meta.get('executable') or not meta.get('validation_passed'):
        raise ValueError('合成数据或未通过留出验证的轨迹禁止真机执行')
    if meta['signature'] != physical_signature(cfg):
        raise ValueError('计划与当前机械臂/TCP 不匹配')
    if meta.get('frames_sha256'):
        if digest(meta['frames_path']) != meta['frames_sha256']:
            raise ValueError('生成计划后参考系文件发生变化；需重新规划')
        from .vision.bindings import frame_reasons
        frame_meta, _ = read_frames(meta['frames_path'], cfg)
        reasons = frame_reasons(frame_meta)
        if reasons:
            raise ValueError(f'视觉参考系不能执行：{reasons}')
    if digest(meta['model_path']) != meta['model_sha256']:
        raise ValueError('生成计划后模型文件发生变化；需重新规划')
    model_meta, _ = read_record(meta['model_path'], 'model')
    if model_meta['synthetic'] or not model_meta['validation_passed']:
        raise ValueError('模型本身未通过真机执行检查')
    computed = np.array([tcp_pose(row, cfg) for row in data['joints_rad']])
    if np.max(abs(computed-data['tcp'])) > 1e-7:
        # quaternion sign must be identical to FK saved by this planner.
        raise ValueError('计划 TCP 和关节 FK 不一致')
    metrics, ratio = check_path(data['time_s'], data['joints_rad'], computed,
                                data['gripper_command'], meta['gripper_mode'], cfg)
    force = np.asarray(data['gripper_force_N'])
    if (force.shape != data['time_s'].shape or not np.isfinite(force).all()
            or np.any(force < 0) or np.any(force > cfg['gripper_max_force_N']) or ratio > 1.001):
        raise ValueError('轨迹速度、加速度或夹爪力检查不通过')
    if np.max(np.diff(data['time_s'])) > 1.1/cfg['control_hz']:
        raise ValueError('计划采样频率不足')
    return metrics
