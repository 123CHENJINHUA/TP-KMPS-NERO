"""Camera-only tracking acceptance and explicitly requested measurement windows."""
from .common import checked, pose_average
from scipy.spatial.transform import Rotation
import time
import numpy as np


def center_position(pose, center):
    pose = checked(pose)
    return pose[:3, :3]@np.asarray(center)+pose[:3, 3]


class TrackingGuard:
    def __init__(self, center, max_step_m=.08, max_step_rad=np.pi/4, max_gap_s=10.):
        self.center = np.asarray(center, float)
        self.max_step_m, self.max_step_rad, self.max_gap_s = max_step_m, max_step_rad, max_gap_s
        self.pose, self.timestamp, self.epoch = None, None, 0

    def reset(self):
        self.pose, self.timestamp = None, None

    def needs_registration(self, timestamp):
        return self.pose is None or timestamp-self.timestamp > self.max_gap_s

    def assess(self, pose, quality, timestamp, registered):
        pose = checked(pose)
        reasons = list(quality['reasons']) if not quality['passed'] else []
        delta_m = delta_rad = None
        if self.pose is not None:
            if timestamp <= self.timestamp:
                reasons.append('non_increasing_frame_time')
            if not registered:
                if timestamp-self.timestamp > self.max_gap_s:
                    reasons.append('tracking_frame_gap')
                delta_m = float(np.linalg.norm(center_position(pose, self.center)-center_position(self.pose, self.center)))
                delta_rad = float(Rotation.from_matrix(pose[:3, :3]@self.pose[:3, :3].T).magnitude())
                if delta_m > self.max_step_m:
                    reasons.append('translation_jump')
                if delta_rad > self.max_step_rad:
                    reasons.append('rotation_jump')
        accepted = not reasons
        if accepted:
            if registered:
                self.epoch += 1
            self.pose, self.timestamp = pose.copy(), timestamp
        else:
            self.reset()
        return dict(accepted=accepted, reasons=sorted(set(reasons)), epoch=self.epoch,
                    center_step_m=delta_m, rotation_step_rad=delta_rad)


def static_statistics(poses, center):
    poses = np.array([checked(pose) for pose in poses])
    if len(poses) < 3:
        raise ValueError('静止统计至少需要 3 个有效位姿')
    centers = np.array([center_position(pose, center) for pose in poses])
    mean_pose, _ = pose_average(poses)
    distances = np.linalg.norm(centers-centers.mean(axis=0), axis=1)
    angles = Rotation.from_matrix(poses[:, :3, :3]@mean_pose[:3, :3].T).magnitude()
    return dict(samples=len(poses), T_color_object_mean=mean_pose.tolist(),
                center_mean_m=centers.mean(axis=0).tolist(),
                center_std_xyz_mm=(centers.std(axis=0)*1000).tolist(),
                center_rms_jitter_mm=float(np.sqrt(np.mean(distances**2))*1000),
                center_max_deviation_mm=float(distances.max()*1000),
                rotation_rms_jitter_deg=float(np.rad2deg(np.sqrt(np.mean(angles**2)))),
                rotation_max_deviation_deg=float(np.rad2deg(angles.max())),
                absolute_pose_accuracy_tested=False)


class Measurements:
    """S: static jitter; A/B: averaged centers before/after a measured translation."""
    def __init__(self, center, samples=10, expected_distance_mm=None):
        self.center, self.samples, self.expected = center, samples, expected_distance_mm
        self.active, self.anchor = None, None
        self.message = 'S: static jitter | A/B: displacement | R: reinitialize | P: snapshot | Q: quit'

    def start(self, kind):
        if kind not in ('static', 'A', 'B'):
            raise ValueError('未知测量窗口')
        if kind == 'B' and self.anchor is None:
            self.message = 'Record A first, then translate the object with CAMERA FIXED.'
            return
        if kind == 'A':
            self.anchor = None
        self.active = dict(kind=kind, poses=[], epoch=None)
        self.message = f'{kind}: keep CAMERA and OBJECT still for {self.samples} accepted frames.'

    def cancel(self, reason, invalidate_anchor=False):
        if self.active or (invalidate_anchor and self.anchor is not None):
            self.message = f'Measurement canceled: {reason}. Press S/A/B again.'
        self.active = None
        if invalidate_anchor:
            self.anchor = None

    def consume(self, result):
        if not result['accepted']:
            self.cancel('tracking lost', invalidate_anchor=True)
            return None
        epoch = result['epoch']
        if self.anchor is not None and self.anchor['epoch'] != epoch:
            self.anchor = None
            if self.active and self.active['kind'] == 'B':
                self.cancel('reinitialized since A; record A again')
        if self.active is None:
            return None
        active = self.active
        if active['epoch'] is None:
            active['epoch'] = epoch
        elif active['epoch'] != epoch:
            self.cancel('reinitialized during measurement', invalidate_anchor=True)
            return None
        active['poses'].append(result['T_color_object'])
        n = len(active['poses'])
        self.message = f'{active["kind"]}: {n}/{self.samples} accepted frames; keep still.'
        if n < self.samples:
            return None
        stats = static_statistics(active['poses'], self.center)
        # These only reject visible motion. A small jitter is not proof of accuracy.
        steady = stats['center_max_deviation_mm'] <= 5. and stats['rotation_max_deviation_deg'] <= 3.
        report = dict(kind=active['kind'], epoch=epoch, unix_s=time.time(), steady=steady,
                      measurement_valid=steady, **stats)
        if not steady:
            report['measurement_valid'] = False
            self.message = 'Object/camera was not steady. Repeat this measurement.'
        elif active['kind'] == 'A':
            self.anchor = report
            self.message = 'A saved. Translate object; keep CAMERA fixed, then press B.'
        elif active['kind'] == 'B':
            start = self.anchor
            delta = np.array(report['center_mean_m'])-np.array(start['center_mean_m'])
            angle = float(np.rad2deg(Rotation.from_matrix(np.array(report['T_color_object_mean'])[:3, :3]
                              @np.array(start['T_color_object_mean'])[:3, :3].T).magnitude()))
            measured = float(np.linalg.norm(delta)*1000)
            report.update(kind='displacement', anchor_A=start, center_delta_xyz_mm=(delta*1000).tolist(),
                          measured_distance_mm=measured, rotation_change_deg=angle,
                          expected_distance_mm=self.expected, measurement_valid=angle <= 3.,
                          required_assumption='Camera fixed, same object, pure translation; manually verify with ruler.',
                          distance_error_mm=(measured-self.expected) if self.expected is not None and angle <= 3. else None)
            self.message = (f'Distance {measured:.2f} mm; rotation {angle:.2f} deg'
                            +(f'; error {report["distance_error_mm"]:+.2f} mm' if report['distance_error_mm'] is not None else ''))
        else:
            report['measurement_valid'] = True
            self.message = f'Static RMS: {stats["center_rms_jitter_mm"]:.2f} mm / {stats["rotation_rms_jitter_deg"]:.2f} deg'
        self.active = None
        return report
