"""Solve and validate both modes without ambiguous transform directions."""
from bootstrap import read_json
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from transforms import checked, inverse, mean_pose, pack, unpack, distance

METHODS = {name: getattr(cv2, f'CALIB_HAND_EYE_{name}') for name in ('PARK', 'TSAI', 'HORAUD', 'ANDREFF', 'DANIILIDIS')}


def signature(metadata):
    return {key: metadata[key] for key in ('mode', 'board', 'camera', 'robot', 'synthetic')}


def load_dataset(path):
    path = Path(path).resolve()
    metadata = read_json(path / 'metadata.json')
    if metadata.get('schema_version') != 1:
        raise ValueError('不支持的数据格式')
    files = sorted((path / 'samples').glob('*.json'))
    samples = [read_json(p) for p in files]
    if not samples:
        raise ValueError('数据集没有样本')
    if len(set(s['id'] for s in samples)) != len(samples):
        raise ValueError('样本 ID 重复')
    if metadata['mode'] != 'intrinsics':
        if metadata['mode'] not in ('eye-in-hand', 'eye-to-hand'):
            raise ValueError('手眼模式无效')
        for sample in samples:
            checked(sample['robot']['T_base_flange'])
            checked(sample['T_color_board'])
    encoded = json.dumps({'signature': signature(metadata), 'samples': samples}, sort_keys=True, allow_nan=False).encode()
    return metadata, samples, hashlib.sha256(encoded).hexdigest()


def distinct_count(poses):
    unique = []
    for pose in poses:
        duplicate = False
        for previous in unique:
            t, r = distance(previous, pose)
            if t < .005 and r < np.deg2rad(3):
                duplicate = True
                break
        if not duplicate:
            unique.append(pose)
    return len(unique)


def excitation(poses):
    count = distinct_count(poses)
    if count < 12:
        raise ValueError('训练至少需要 12 个不同位姿；建议总共采集 30–40 张')
    rows, axes, angles = [], [], []
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            R = (inverse(poses[i]) @ poses[j])[:3, :3]
            vector = Rotation.from_matrix(R).as_rotvec()
            angle = np.linalg.norm(vector)
            if angle > np.deg2rad(3):
                rows.append(np.eye(3) - R)
                axes.append(vector / angle)
                angles.append(angle)
    if not rows:
        raise ValueError('只有平移或转角太小，无法解算手眼外参；需绕至少两个不同轴旋转')
    values = np.linalg.svd(np.vstack(rows), compute_uv=False)
    axis_values = np.linalg.svd(np.array(axes), compute_uv=False)
    ratio = float(values[-1] / values[0])
    report = {'relative_rotation_max_deg': float(np.rad2deg(max(angles))),
              'distinct_poses': count,
              'translation_system_singular_values': values.tolist(), 'condition_ratio': ratio,
              'rotation_axis_singular_values': axis_values.tolist()}
    if max(angles) < np.deg2rad(15) or ratio < .02 or axis_values[1] / axis_values[0] < .05:
        raise ValueError(f'运动激励不足：{report}；绕不同轴明显倾斜，同时改变位置')
    return report


def pose_arrays(samples):
    return ([checked(s['robot']['T_base_flange']) for s in samples],
            [checked(s['T_color_board']) for s in samples])


def constants(mode, P, Q, X):
    if mode == 'eye-in-hand':
        return [p @ X @ q for p, q in zip(P, Q)]  # fixed base<-board
    return [inverse(p) @ X @ q for p, q in zip(P, Q)]  # rigid flange<-board


def residual_pair(mode, p, q, X, F):
    if mode == 'eye-in-hand':
        return F, p @ X @ q
    return p @ F, X @ q


def fit(samples, mode, method='PARK', refine=True):
    P, Q = pose_arrays(samples)
    observability = excitation(P)
    # Eye-to-hand: substituting inverse(base<-flange) into OpenCV's first
    # argument makes its result base<-color. This is NOT an inversion of the
    # eye-in-hand result: the entire set of input robot poses is inverted.
    inputs = P if mode == 'eye-in-hand' else [inverse(p) for p in P]
    R, t = cv2.calibrateHandEye([p[:3, :3] for p in inputs], [p[:3, 3] for p in inputs],
                               [q[:3, :3] for q in Q], [q[:3, 3] for q in Q], method=METHODS[method])
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = R, np.asarray(t).ravel()
    X = checked(X)
    F = mean_pose(constants(mode, P, Q, X))
    refinement = {'enabled': refine}
    if refine:
        def residual(parameters):
            x, f = unpack(parameters[:6]), unpack(parameters[6:])
            values = []
            for p, q in zip(P, Q):
                first, second = residual_pair(mode, p, q, x, f)
                delta = inverse(first) @ second
                values.extend(delta[:3, 3] / .001)
                values.extend(Rotation.from_matrix(delta[:3, :3]).as_rotvec() / .005)
            return values
        solution = least_squares(residual, np.r_[pack(X), pack(F)], loss='soft_l1', f_scale=1., max_nfev=300)
        if not solution.success or not np.isfinite(solution.x).all():
            raise ValueError(f'非线性细化未收敛：{solution.message}')
        X, F = checked(unpack(solution.x[:6])), checked(unpack(solution.x[6:]))
        refinement.update(evaluations=int(solution.nfev), cost=float(solution.cost),
                          translation_residual_scale_m=.001, rotation_residual_scale_rad=.005,
                          loss='soft_l1', nuisance_frame='base_board' if mode == 'eye-in-hand' else 'flange_board')
    return X, F, observability, refinement


def evaluate(samples, mode, X, F, board=None, camera=None):
    rows = []
    for sample in samples:
        p, q = checked(sample['robot']['T_base_flange']), checked(sample['T_color_board'])
        first, second = residual_pair(mode, p, q, X, F)
        t, r = distance(first, second)
        row = {'id': sample['id'], 'translation_mm': t * 1000., 'rotation_deg': float(np.rad2deg(r)),
               'pnp_reprojection_rms_px': sample.get('reprojection_rms_px')}
        if board is not None and camera is not None and 'corners_px' in sample:
            from vision import camera_matrix
            K, dist = camera_matrix(camera)
            predicted = inverse(X) @ (inverse(p) @ F if mode == 'eye-in-hand' else p @ F)
            obj = board.getChessboardCorners()[np.array(sample['corner_ids'], np.int32)].astype(np.float64)
            if np.min((obj @ predicted[:3, :3].T + predicted[:3, 3])[:, 2]) <= 0:
                raise ValueError('标定预测把板放到了相机后方；检查坐标系和数据配对')
            uv = cv2.projectPoints(obj, cv2.Rodrigues(predicted[:3, :3])[0], predicted[:3, 3], K, dist)[0].reshape(-1, 2)
            row['loop_reprojection_rms_px'] = float(np.sqrt(np.mean(np.sum((uv - sample['corners_px']) ** 2, axis=1))))
        rows.append(row)
    def metrics(name):
        values = np.array([r[name] for r in rows], dtype=float)
        return {'rms': float(np.sqrt(np.mean(values ** 2))), 'median': float(np.median(values)), 'max': float(np.max(values))}
    result = {'count': len(rows), 'translation_mm': metrics('translation_mm'), 'rotation_deg': metrics('rotation_deg'), 'samples': rows}
    if all('loop_reprojection_rms_px' in r for r in rows):
        result['loop_reprojection_px'] = metrics('loop_reprojection_rms_px')
    return result


def passed(report, limits):
    good = (report['translation_mm']['rms'] <= limits['translation_rms_mm'] and
            report['rotation_deg']['rms'] <= limits['rotation_rms_deg'] and
            report['translation_mm']['max'] <= limits['translation_rms_mm'] * 3 and
            report['rotation_deg']['max'] <= limits['rotation_rms_deg'] * 3)
    if 'loop_reprojection_px' in report:
        good = good and report['loop_reprojection_px']['rms'] <= limits['loop_reprojection_rms_px']
    return bool(good)


def print_report(name, report):
    print(f'{name}: n={report["count"]}, 平移 RMS={report["translation_mm"]["rms"]:.3f} mm, '
          f'旋转 RMS={report["rotation_deg"]["rms"]:.3f} deg')
    if 'loop_reprojection_px' in report:
        print(f'  闭环重投影 RMS={report["loop_reprojection_px"]["rms"]:.3f} px')
