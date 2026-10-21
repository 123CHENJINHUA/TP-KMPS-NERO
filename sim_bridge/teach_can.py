"""Passive SocketCAN reader for two Nero teaching arms (firmware >= 1.12).

Only receives 0x155/0x156/0x157/0x170 joint commands and 0x159
AGX gripper command echoes. It never transmits CAN frames.
"""

import math
from pathlib import Path
import socket
import struct
import time

CAN_FRAME = struct.Struct('=IB3x8s')
CAN_ERR_FLAG = 0x20000000
CAN_RTR_FLAG = 0x40000000
CAN_EFF_FLAG = 0x80000000
CAN_SFF_MASK = 0x7ff
JOINT_IDS = {0x155: 0, 0x156: 2, 0x157: 4, 0x170: 6}
JOINT_SCALE = math.pi / 180000.0


def decode_teach_frame(raw):
    """Return (kind, slot, value) or None for unrelated/invalid frames."""
    if len(raw) < CAN_FRAME.size:
        return None
    can_id, length, data = CAN_FRAME.unpack_from(raw)
    if can_id & (CAN_ERR_FLAG | CAN_RTR_FLAG | CAN_EFF_FLAG):
        return None
    identity = can_id & CAN_SFF_MASK
    if identity in JOINT_IDS:
        count = 1 if identity == 0x170 else 2
        if length < count * 4:
            return None
        values = struct.unpack_from('>i' if count == 1 else '>ii', data)
        angles = tuple(value * JOINT_SCALE for value in values)
        if not all(math.isfinite(value) and abs(value) <= 2 * math.pi for value in angles):
            return None
        return 'joint', JOINT_IDS[identity], angles
    if identity == 0x159 and length >= 8:
        raw_value, force_mn, status_code, set_zero = struct.unpack('>ihBB', data)
        if status_code > 7:
            return None
        mode = 'angle' if status_code & 4 else 'width'
        value = raw_value * (1e-3 if mode == 'angle' else 1e-6)
        return 'gripper', None, {'value': value, 'mode': mode,
                                  'force_N': force_mn * 1e-3,
                                  'status_code': status_code, 'set_zero': set_zero}
    return None


class TeachCanReader:
    """Keep the newest complete joint and gripper samples from both buses."""

    def __init__(self, channels=('can0', 'can1'), stale_after=.25):
        if len(channels) != 2 or channels[0] == channels[1]:
            raise ValueError('Two distinct CAN channels are required')
        if not .05 <= stale_after <= 2.:
            raise ValueError('stale_after must be in [0.05, 2.0] seconds')
        self.channels = tuple(channels)
        self.stale_after = stale_after
        self.sockets = {}
        self.states = {ch: {'parts': {}, 'gripper': None} for ch in self.channels}
        try:
            for channel in self.channels:
                flags_path = Path('/sys/class/net') / channel / 'flags'
                try:
                    flags = int(flags_path.read_text().strip(), 16)
                except OSError as exc:
                    raise RuntimeError(f'{channel} is missing') from exc
                if not flags & 1:
                    raise RuntimeError(f'{channel} is DOWN; enable it at 1 Mbps first')
                sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
                try:
                    sock.setblocking(False)
                    sock.bind((channel,))
                except BaseException:
                    sock.close()
                    raise
                self.sockets[channel] = sock
        except BaseException:
            self.close()
            raise

    def poll(self):
        """Drain pending frames without blocking Isaac Sim's main thread."""
        for channel, sock in self.sockets.items():
            state = self.states[channel]
            for _ in range(4096):
                try:
                    raw = sock.recv(CAN_FRAME.size)
                except BlockingIOError:
                    break
                except OSError as exc:
                    raise RuntimeError(f'{channel} CAN receive failed: {exc}') from exc
                decoded = decode_teach_frame(raw)
                if decoded is None:
                    continue
                kind, slot, value = decoded
                stamp = time.monotonic()
                if kind == 'joint':
                    state['parts'][slot] = (value, stamp)
                else:
                    state['gripper'] = (value, stamp)

    def snapshot(self, channel):
        """Fresh joints and width, or None for an incomplete/stale field."""
        now = time.monotonic()
        state = self.states[channel]
        parts = state['parts']
        if all(slot in parts and now - parts[slot][1] <= self.stale_after
               for slot in (0, 2, 4, 6)):
            joints = [angle for slot in (0, 2, 4, 6) for angle in parts[slot][0]]
        else:
            joints = None
        gripper = state['gripper']
        if gripper is not None and now - gripper[1] <= self.stale_after:
            gripper_data = dict(gripper[0])
            opening = gripper_data['value'] if gripper_data['mode'] == 'width' else None
        else:
            gripper_data = None
            opening = None
        return {'joints': joints, 'opening': opening, 'gripper': gripper_data}

    def close(self):
        for sock in self.sockets.values():
            sock.close()
        self.sockets.clear()
