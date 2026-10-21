"""Explain duplicate poses and detect disagreement between robot and board rotation."""
import bootstrap
import numpy as np
from transforms import checked, distance


class MotionCaptureError(ValueError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def compare_capture(packet, detection, previous):
    t, r = distance(checked(previous['robot']['T_base_flange']), checked(packet['T_base_flange']))
    row = {'reference_id': previous['id'], 'flange_translation_mm': t * 1000.,
           'flange_rotation_deg': float(np.rad2deg(r)), 'max_joint_change_deg': None,
           'board_translation_mm': None, 'board_rotation_deg': None, 'rotation_disagreement_deg': None}
    if packet.get('joints_rad') is not None and previous['robot'].get('joints_rad') is not None:
        delta = np.array(packet['joints_rad']) - np.array(previous['robot']['joints_rad'])
        delta = (delta + np.pi) % (2 * np.pi) - np.pi
        row['max_joint_change_deg'] = float(np.rad2deg(np.abs(delta).max()))
    if detection and 'T_color_board' in detection and 'T_color_board' in previous:
        bt, br = distance(checked(previous['T_color_board']), checked(detection['T_color_board']))
        row.update(board_translation_mm=bt * 1000., board_rotation_deg=float(np.rad2deg(br)),
                   rotation_disagreement_deg=abs(float(np.rad2deg(br - r))))
    return row


def check_capture(packet, detection, accepted_samples, max_rotation_disagreement_deg=8.):
    rows = [compare_capture(packet, detection, previous) for previous in accepted_samples]
    # For BOTH hand-eye configurations, robot and visual-board relative rotations
    # are conjugate (possibly inverted). Their angles must agree, independent of
    # the unknown hand-eye/board offset. Translational magnitudes need not agree.
    for row in rows:
        mismatch = row['rotation_disagreement_deg']
        if mismatch is not None and mismatch > max_rotation_disagreement_deg:
            msg = (f'图像与机械臂旋转不一致（相对样本 {row["reference_id"]}）：'
                   f'板转角 {row["board_rotation_deg"]:.2f}°，CAN/FK 法兰转角 {row["flange_rotation_deg"]:.2f}°。'
                   '请核对 CAN 通道、位姿来源是否对应装相机或标定板的实体臂，以及相机/板的固定安装。'
                   '从臂需要其 measured 实际反馈，不能用主臂 leader 指令代替。')
            if row['max_joint_change_deg'] is not None:
                msg += f' CAN 最大关节变化只有 {row["max_joint_change_deg"]:.3f}°；报文时间戳更新不等于关节数值更新。'
            raise MotionCaptureError(msg, {'code': 'vision_robot_rotation_mismatch', 'comparisons': rows})
    for row in rows:
        if row['flange_translation_mm'] < 5. and row['flange_rotation_deg'] < 3.:
            msg = (f'与样本 {row["reference_id"]} 的法兰位姿太近：'
                   f'位置差 {row["flange_translation_mm"]:.3f} mm，转角差 {row["flange_rotation_deg"]:.3f}°'
                   '（需至少 5 mm 或 3°）。')
            change = row['max_joint_change_deg']
            if change is not None:
                msg += f' CAN 最大关节变化 {change:.3f}°。'
                if change < .1:
                    msg += ' 若实体臂确实明显移动，请核对通道/反馈；当前读到的关节数值几乎未变。'
                elif change > 5:
                    msg += ' 关节变化大而末端位姿接近，可能是七轴冗余运动；请改变法兰/板的实际位姿。'
            if row['board_rotation_deg'] is not None:
                msg += f' 图像中板的位置差 {row["board_translation_mm"]:.2f} mm、转角差 {row["board_rotation_deg"]:.2f}°。'
            raise MotionCaptureError(msg, {'code': 'duplicate_flange_pose', 'comparisons': rows})
    return rows
