"""Original task-parameterized KMP, with whole-demonstration holdout."""
from .bootstrap import digest
from pathlib import Path
import numpy as np
from .data import align, read_record, physical_signature
from .kinematics import pose_error, pose_matrix
from nero_bridge.model import build_model, fuse


def errors(actual, target):
    values = np.array([pose_error(a, b) for a, b in zip(actual, target)])
    return dict(position_rms_m=float(np.sqrt(np.mean(values[:, 0]**2))),
                rotation_rms_rad=float(np.sqrt(np.mean(values[:, 1]**2))),
                position_max_m=float(values[:, 0].max()), rotation_max_rad=float(values[:, 1].max()))


def load_demos(directory, cfg, samples):
    paths = sorted(Path(directory).glob('demo_*.npz'))
    if len(paths) < 4:
        raise ValueError('至少需要 4 组成功示教；建议 8–15 组、覆盖不同抓取和放置位置')
    rows, mode = [], None
    for path in paths:
        meta, data = read_record(path, 'demonstration')
        if meta.get('signature') != physical_signature(cfg) or meta.get('joint_source') != 'measured':
            raise ValueError(f'{path.name} 机械臂/TCP 配置或反馈来源不匹配')
        if meta.get('success') is not True:
            raise ValueError(f'{path.name} 不是标注成功的示教')
        task_source = meta.get('task_frame_source', {})
        if task_source.get('synthetic') or task_source.get('execution_blocked_reasons'):
            raise ValueError(f'{path.name} 的视觉/任务参考系来自合成或未通过检查的数据')
        current = meta.get('gripper_mode')
        if current not in ('width', 'angle') or (mode is not None and current != mode):
            raise ValueError('所有示教必须使用相同的夹爪 width 或 angle 模式')
        mode = current
        frames = np.asarray(data['frames'], float)
        if frames.shape != (3, 7):
            raise ValueError('示教参考系形状需要 (3,7)')
        for frame in frames:
            pose_matrix(frame)
        result = align(data, samples, cfg)
        limits = cfg['gripper_width_limits_m' if mode == 'width' else 'gripper_angle_limits_deg']
        if np.any(result['command'] < limits[0]) or np.any(result['command'] > limits[1]) or np.any(result['force'] < 0):
            raise ValueError(f'{path.name} 夹爪目标/力越界')
        if pose_error(result['tcp'][0], frames[0])[0] > .001:
            raise ValueError(f'{path.name} start 参考系不是本次示教起点')
        rows.append(dict(path=str(path.resolve()), sha256=digest(path), meta=meta, frames=frames, **result))
    return rows, mode


def rebuild(meta, data):
    return build_model(data['demonstrations'], data['frames'], meta['samples'], meta['components'])


def predict(meta, data, frames, model=None):
    return fuse(model or rebuild(meta, data), frames, meta['samples'])


def report_rows(model, rows, cfg, samples):
    reports, predictions = [], []
    for row in rows:
        predicted = fuse(model, row['frames'], samples)
        metric = errors(predicted, row['tcp'])
        metric['passed'] = (metric['position_rms_m'] <= cfg['validation_position_rms_m']
                            and metric['rotation_rms_rad'] <= cfg['validation_rotation_rms_rad'])
        reports.append(dict(file=row['path'], **metric))
        predictions.append(predicted)
    return reports, predictions


def plot_poses(path, reference, prediction, title):
    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=(10, 4))
    ax = figure.add_subplot(121, projection='3d')
    ax.plot(*np.asarray(reference)[:, :3].T, label='demonstration')
    ax.plot(*np.asarray(prediction)[:, :3].T, label='TP-KMP')
    ax.set(xlabel='base X (m)', ylabel='base Y (m)', zlabel='base Z (m)', title=title)
    ax.legend()
    ax = figure.add_subplot(122)
    values = np.array([pose_error(a, b) for a, b in zip(prediction, reference)])
    ax.plot(np.linspace(0, 1, len(values)), values[:, 0]*1000, label='position (mm)')
    ax.plot(np.linspace(0, 1, len(values)), np.rad2deg(values[:, 1]), label='rotation (deg)')
    ax.set(xlabel='phase', title='Prediction error')
    ax.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
