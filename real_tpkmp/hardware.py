"""Passive follower feedback; explicit, separately attached command transport.

One SocketCAN socket excludes its own TX echo, so foreign controllers remain visible.
No SDK connect/monitor, enable, homing, leader/follower switching or zeroing is used.
"""
from . import bootstrap
import struct
import threading
import time
from pathlib import Path
import numpy as np
from .kinematics import tcp_pose

JOINT_IDS = {0x2A5: (0, 1), 0x2A6: (2, 3), 0x2A7: (4, 5), 0x2A9: (6,)}
CONTROL_IDS = {0x150, 0x151, 0x152, 0x153, 0x154, 0x155, 0x156, 0x157,
               0x159, 0x170, 0x470, 0x471} | set(range(0x15A, 0x161))


class Feedback:
    """Pure decoder, also usable in offline wire-format tests."""
    def __init__(self):
        self.q = np.zeros(7)
        self.stamps = np.full(4, -np.inf)
        self.measured = self.command = None
        self.low_speed = np.full(7, -np.inf)
        self.foreign_time = -np.inf
        self.error_frames = 0

    def feed(self, identifier, payload, now, error=False):
        if error:
            self.error_frames += 1
            return
        if identifier in CONTROL_IDS:
            self.foreign_time = now
        if identifier in JOINT_IDS and len(payload) == 8:
            indices = JOINT_IDS[identifier]
            self.q[list(indices)] = np.array(struct.unpack('>ii', payload)[:len(indices)]) * np.pi / 180000
            self.stamps[list(JOINT_IDS).index(identifier)] = now
        elif identifier in (0x2A8, 0x159) and len(payload) == 8:
            value, force, status, mode_byte = struct.unpack('>ihBB', payload)
            angle = mode_byte == 1 if identifier == 0x2A8 else bool(status & 4)
            item = dict(value=value * (.001 if angle else .000001), force=force*.001,
                        mode='angle' if angle else 'width', time=now, status=status)
            if identifier == 0x2A8:
                self.measured = item
            elif mode_byte == 0 and status in (1, 5):
                self.command = item
            else:
                self.command = None  # Never learn disabled/zeroing commands.
        elif 0x261 <= identifier <= 0x267 and len(payload) == 8:
            self.low_speed[identifier - 0x261] = now

    def snapshot(self, now, cfg, need_command=True):
        if (np.any(now-self.stamps > cfg['state_max_age_s'])
                or np.ptp(self.stamps) > cfg['max_joint_packet_skew_s']):
            raise ValueError('从臂 measured 反馈缺失、过期或分帧不同步；不会替换成 leader')
        if self.measured is None or now-self.measured['time'] > cfg['state_max_age_s']:
            raise ValueError('夹爪实际反馈 0x2A8 缺失或过期')
        if need_command and (self.command is None or now-self.command['time'] > cfg['state_max_age_s']
                             or self.command['mode'] != self.measured['mode']):
            raise ValueError('夹爪目标 0x159 缺失、过期、禁用或模式与实际反馈不一致')
        result = dict(joints_rad=self.q.copy(), tcp=tcp_pose(self.q, cfg),
                      gripper_measured=self.measured['value'], gripper_mode=self.measured['mode'],
                      gripper_feedback_force_N=self.measured['force'], host_monotonic_s=now)
        if need_command:
            result.update(gripper_command=self.command['value'], gripper_force_N=self.command['force'])
        return result


class Reader:
    def __init__(self, cfg):
        import can
        import pyAgxArm as sdk
        flags = Path('/sys/class/net') / cfg['channel'] / 'flags'
        if not flags.exists() or not int(flags.read_text().strip(), 16) & 1:
            raise ValueError(f'{cfg["channel"]} 不存在或 DOWN；请先启动 CAN 接口')
        sdk_config = sdk.create_agx_arm_config(robot=sdk.ArmModel.NERO, firmeware_version=cfg['firmware'],
                        interface='socketcan', channel=cfg['channel'], auto_connect=False,
                        enable_check_can=False, log_level='ERROR')
        self.robot = sdk.AgxArmFactory.create_arm(sdk_config)
        self.gripper = self.robot.init_effector(self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self.bus = can.Bus(interface='socketcan', channel=cfg['channel'], receive_own_messages=False)
        self.cfg, self.feedback = cfg, Feedback()
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.error, self.writer_attached = None, False
        self.thread = threading.Thread(target=self._receive, daemon=True)
        self.thread.start()

    def _receive(self):
        try:
            while not self.stop.is_set():
                msg = self.bus.recv(.02)
                if msg is None:
                    continue
                with self.lock:
                    if not (msg.is_remote_frame or msg.is_extended_id):
                        self.feedback.feed(msg.arbitration_id, msg.data, time.monotonic(), msg.is_error_frame)
                    if not (msg.is_error_frame or msg.is_remote_frame or msg.is_extended_id):
                        self.robot._ctx._run_parser_packet_funs(msg)
        except Exception as exc:
            self.error = str(exc)

    def sample(self, need_command=True):
        with self.lock:
            if self.error or self.feedback.error_frames:
                raise ValueError(f'CAN 接收错误：{self.error or "收到 CAN error frame"}')
            return self.feedback.snapshot(time.monotonic(), self.cfg, need_command)

    def wait_ready(self, need_command=True):
        deadline, error = time.monotonic()+3, None
        while time.monotonic() < deadline:
            try:
                return self.sample(need_command)
            except ValueError as exc:
                error = exc
            time.sleep(.02)
        raise ValueError(f'等待反馈超时：{error}')

    def require_exclusive(self, quiet_s=1.):
        with self.lock:
            if time.monotonic()-self.feedback.foreign_time < quiet_s:
                raise ValueError('CAN 上仍有其他控制源（主臂/其他进程）；停止主臂发令后才能执行')

    def foreign_stamp(self):
        with self.lock:
            return self.feedback.foreign_time

    def require_no_new_controller(self, stamp):
        # Latch conflicts even if a process stall lasts longer than quiet_s.
        with self.lock:
            if self.feedback.foreign_time > stamp:
                raise ValueError('执行期间检测到另一控制源，立即停止回放')

    def require_enabled(self):
        with self.lock:
            if np.any(time.monotonic()-self.feedback.low_speed > .5) or not self.robot.get_joint_enable_status(255):
                raise ValueError('七轴使能反馈不完整、过期或从臂未使能；脚本不会自动使能')

    def attach_writer(self):
        self.require_exclusive()
        self.require_enabled()
        self.robot._ctx.comm = _Transport(self.bus, self.cfg['channel'])
        self.robot.set_auto_set_motion_mode_enabled(False)
        self.robot.set_joint_limits_enabled(True)
        self.writer_attached = True

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1)
        self.robot._ctx.comm = None
        self.bus.shutdown()


class _Transport:
    def __init__(self, bus, channel):
        self.bus, self.channel = bus, channel

    def send(self, message):
        self.bus.send(message, timeout=.02)

    def get_channel(self):
        return self.channel
