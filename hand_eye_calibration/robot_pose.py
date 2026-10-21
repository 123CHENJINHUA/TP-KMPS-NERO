"""Receive-only SocketCAN + offline SDK decoding/FK. Never connects SDK transport."""
from bootstrap import PROJECT, read_json
from collections import Counter, deque
import sys
import threading
import time
import numpy as np
from transforms import from_pose6, checked, distance

MEASURED_PARTS = ('joint_12', 'joint_34', 'joint_56', 'joint_7')


def load_sdk():
    sys.path.insert(0, str(PROJECT / 'pyAgxArm'))
    import pyAgxArm
    return pyAgxArm


def complete_feedback(robot, source, firmware, now, max_age):
    parts = MEASURED_PARTS
    getter = 'get_joint_angles'
    if source == 'leader':
        getter = 'get_leader_joint_angles'
        parts = (tuple(f'leader_joint_{i}' for i in range(1, 8)) if firmware in ('default', 'v111') else
                 ('leader_joint_12', 'leader_joint_34', 'leader_joint_56', 'leader_joint_7'))
    messages = [getattr(robot._parser, name, None) for name in parts]
    if any(m is None for m in messages):
        raise ValueError(f'{source} 关节反馈不完整；需要 {parts}。请确认板/相机装在哪个实体臂上')
    stamps = np.array([float(m.timestamp) for m in messages])
    if not np.isfinite(stamps).all() or np.any(now - stamps > max_age) or np.any(stamps - now > .05):
        raise ValueError('关节反馈已过期或时间戳异常；没有用零值、旧值或另一种来源替代')
    if np.ptp(stamps) > .05:
        raise ValueError('各关节分帧时间相差过大')
    message = getattr(robot, getter)()
    if message is None:
        raise ValueError('SDK 没有有效关节反馈')
    joints = np.array(list(message.msg), dtype=float)
    if joints.shape != (7,) or not np.isfinite(joints).all():
        raise ValueError('Nero 关节数据无效')
    return joints, stamps


class CanPoseReader:
    def __init__(self, channel='can0', firmware='v121', source='measured', max_age=.15):
        if source not in ('measured', 'leader'):
            raise ValueError('采集必须明确选择 measured 或 leader；诊断 both 不会自动选择来源')
        import can
        from pathlib import Path
        flags = Path('/sys/class/net') / channel / 'flags'
        if not flags.exists() or not int(flags.read_text().strip(), 16) & 1:
            raise ValueError(f'{channel} 不存在或 DOWN；请先配置主机 CAN 接口')
        sdk = load_sdk()
        cfg = sdk.create_agx_arm_config(robot=sdk.ArmModel.NERO, firmeware_version=firmware,
                                       interface='socketcan', channel=channel, auto_connect=False,
                                       enable_check_can=False)
        self.robot = sdk.AgxArmFactory.create_arm(cfg)
        # Only python-can's receive socket is opened. SDK is used offline; no connect,
        # no SDK monitor, no effector, no modes, no send, no robot movement.
        self.bus = can.Bus(interface='socketcan', channel=channel, receive_own_messages=False)
        self.source, self.firmware, self.max_age = source, firmware, max_age
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.error = None
        self.rx_counts = Counter()
        self.rx_ids = Counter()
        self.last_frame_unix_s = None
        self.history = deque(maxlen=500)
        self.thread = threading.Thread(target=self._receive, daemon=True)
        self.thread.start()

    def _receive(self):
        try:
            while not self.stop.is_set():
                message = self.bus.recv(.05)
                if message is not None:
                    with self.lock:
                        self.rx_counts['frames'] += 1
                        self.rx_counts['error_frames'] += int(message.is_error_frame)
                        self.rx_counts['remote_frames'] += int(message.is_remote_frame)
                        self.rx_ids[f'0x{message.arbitration_id:03X}'] += 1
                        self.last_frame_unix_s = time.time()
                        if not message.is_error_frame and not message.is_remote_frame:
                            self.robot._parser.parse_packet(message)
        except Exception as exc:
            self.error = str(exc)

    def transport_status(self):
        """Raw receive counts distinguish a silent bus from incomplete decoding."""
        with self.lock:
            return {'frames': self.rx_counts['frames'],
                    'error_frames': self.rx_counts['error_frames'],
                    'remote_frames': self.rx_counts['remote_frames'],
                    'ids': dict(self.rx_ids), 'receiver_error': self.error,
                    'last_frame_age_s': (None if self.last_frame_unix_s is None else
                                         max(0., time.time() - self.last_frame_unix_s))}

    def _packet(self, source, now):
        """Caller holds lock so getter/FK and every CAN part form one snapshot."""
        joints, stamps = complete_feedback(self.robot, source, self.firmware, now, self.max_age)
        T = from_pose6(self.robot.fk(joints.tolist()))
        return {'T_base_flange': T.tolist(), 'joints_rad': joints.tolist(),
                  'joint_frame_unix_s': stamps.tolist(), 'host_unix_s': now,
                  'host_monotonic_s': time.monotonic(), 'source': source}

    def inspect_sources(self):
        """Inspect both feedback types; this never changes the configured source."""
        result = {}
        with self.lock:
            for source in ('measured', 'leader'):
                try:
                    if self.error:
                        raise ValueError(f'CAN 接收失败：{self.error}')
                    packet = self._packet(source, time.time())
                    result[source] = {'available': True, 'packet': packet, 'error': None}
                except ValueError as exc:
                    result[source] = {'available': False, 'packet': None, 'error': str(exc)}
        return result

    def sample(self):
        if self.error:
            raise ValueError(f'CAN 接收失败：{self.error}')
        with self.lock:
            now = time.time()
            try:
                packet = self._packet(self.source, now)
            except ValueError as exc:
                other = 'leader' if self.source == 'measured' else 'measured'
                try:
                    self._packet(other, now)
                except ValueError:
                    raise exc
                hint = (f' 已检测到完整且新鲜的 {other} 反馈。'
                        f'若板/相机确实安装在这组角度对应的实体臂上，请重启并明确使用 --joint-source {other}；'
                        '程序没有自动切换或保存另一只臂的位姿。')
                raise ValueError(str(exc) + hint) from exc
        self.history.append(packet)
        return packet

    def stationary_window(self, seconds=.5):
        packets = []
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            packets.append(self.sample())
            time.sleep(.02)
        require_stationary(packets)
        return packets

    def wait_ready(self, seconds=2.):
        """Fail before creating a session if the explicitly selected source is absent."""
        deadline = time.monotonic() + seconds
        last_error = None
        while time.monotonic() < deadline:
            try:
                return self.sample()
            except ValueError as exc:
                last_error = exc
            time.sleep(.02)
        raise ValueError(f'等待 {self.source} 反馈超时：{last_error}')

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1)
        self.bus.shutdown()
        self.robot.disconnect()


def require_stationary(packets, joint_range_rad=.003, translation_m=.001, rotation_rad=np.deg2rad(.25)):
    if len(packets) < 5:
        raise ValueError('静止检查样本不足')
    joints = np.array([p['joints_rad'] for p in packets])
    if np.max(np.ptp(joints, axis=0)) > joint_range_rad:
        raise ValueError('机械臂还在移动：请停稳后重新按空格')
    reference = checked(packets[0]['T_base_flange'])
    for packet in packets[1:]:
        t, r = distance(reference, checked(packet['T_base_flange']))
        if t > translation_m or r > rotation_rad:
            raise ValueError('采样窗口内法兰位姿变化过大')


def manual_pose(path=None):
    if path:
        value = read_json(path)
        if value.get('translation_unit') != 'm' or value.get('parent_frame') != 'base' or value.get('child_frame') != 'flange':
            raise ValueError('位姿文件需要 translation_unit=m, parent_frame=base, child_frame=flange')
        T = checked(value['T_base_flange'])
    else:
        print('保持机械臂和板静止；输入实际法兰 x y z roll pitch yaw（m、rad），ZYX RPY：', flush=True)
        numbers = [float(x) for x in input().replace(',', ' ').split()]
        T = from_pose6(numbers)
    return {'T_base_flange': T.tolist(), 'host_unix_s': time.time(),
            'host_monotonic_s': time.monotonic(), 'source': 'manual', 'stability_checked': False}
