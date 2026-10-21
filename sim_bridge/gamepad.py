"""Linux Xbox input and joy_pkg-compatible TCP control, without ROS dependencies."""

import array
import fcntl
import os
from pathlib import Path
import struct
import time

import numpy as np
from scipy.spatial.transform import Rotation

EVENT = struct.Struct('IhBB')
# Linux input-event-codes.h: BTN_X/BTN_NORTH=0x133, BTN_Y/BTN_WEST=0x134.
# The connected joydev device reports physical X at index 2 and Y at index 3.
A, B, X, Y, START = 0x130, 0x131, 0x133, 0x134, 0x13b


def deadzone(value, threshold):
    value = float(np.clip(value, -1., 1.))
    return np.sign(value) * max(0., abs(value) - threshold) / (1. - threshold)


def gamepad_pose(pose, axes, buttons, dt, speed=.03, angular_speed=np.deg2rad(15), deadband=.12):
    """Integrate joy2robot's twist in the TCP frame (input uses ROS axis signs)."""
    lx, ly, rx, ry = [deadzone(axes.get(key, 0.), deadband) for key in ('lx', 'ly', 'rx', 'ry')]
    lt, rt = [max(0., deadzone(axes.get(key, 0.), deadband)) for key in ('lt', 'rt')]
    linear = np.array([ly, -lx, lt - rt]) * speed
    angular = np.array([rx, ry, float(A in buttons) - float(B in buttons)]) * angular_speed
    result = np.asarray(pose, dtype=float).copy()
    rotation = Rotation.from_quat(result[[4, 5, 6, 3]])
    result[:3] += rotation.apply(linear) * dt
    result[3:] = (rotation * Rotation.from_rotvec(angular * dt)).as_quat()[[3, 0, 1, 2]]
    return result


class XboxJoystick:
    """Nonblocking joydev reader; discovers semantic axis/button codes via ioctl.

    Supports standard xpad (RX/RY sticks, Z/RZ triggers) and Bluetooth layouts
    (Z/RZ sticks, BRAKE/GAS triggers). No event-device exclusivity or key capture.
    """

    def __init__(self, device='auto'):
        self.device = device
        self.fd = None
        self.axes = {}
        self.buttons = set()
        self.pressed = set()
        self.message = 'Xbox not connected; connect controller (USB recommended).'
        self.next_connect = 0.
        self.axis_map = []
        self.button_map = []
        self.codes = {}

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
        self.fd = None
        self.axes.clear()
        self.buttons.clear()
        self.pressed.clear()

    def _open(self, path):
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        try:
            na, nb = bytearray(1), bytearray(1)
            fcntl.ioctl(fd, 0x80016a11, na, True)
            fcntl.ioctl(fd, 0x80016a12, nb, True)
            axes = array.array('B', [0] * 64)
            buttons = array.array('H', [0] * 512)
            fcntl.ioctl(fd, 0x80406a32, axes, True)
            fcntl.ioctl(fd, 0x84006a34, buttons, True)
            axis_map, button_map = list(axes[:na[0]]), list(buttons[:nb[0]])
            if {0, 1, 2, 3, 4, 5, 16, 17}.issubset(axis_map):
                codes = dict(lx=0, ly=1, rx=3, ry=4, lt=2, rt=5)
            elif {0, 1, 2, 5, 9, 10, 16, 17}.issubset(axis_map):
                codes = dict(lx=0, ly=1, rx=2, ry=5, lt=10, rt=9)
            else:
                raise ValueError(f'Unsupported Xbox axis map: {axis_map}; try USB connection')
            if not {A, B, Y, START}.issubset(button_map):
                raise ValueError('Device does not expose Xbox A/B/Y/Menu buttons')
        except BaseException:
            os.close(fd)
            raise
        self.fd = fd
        self.axis_map, self.button_map, self.codes = axis_map, button_map, codes
        self.message = f'Xbox connected: {path}'

    def poll(self, now=None):
        now = time.monotonic() if now is None else now
        self.pressed.clear()
        if self.fd is None:
            if now < self.next_connect:
                return False
            self.next_connect = now + 1.
            paths = sorted(Path('/dev/input').glob('js*')) if self.device == 'auto' else [Path(self.device)]
            for path in paths:
                try:
                    self._open(path)
                    break
                except (OSError, ValueError) as exc:
                    self.message = f'Xbox unavailable: {path}: {exc}'
            if self.fd is None:
                return False
        try:
            while True:
                try:
                    data = os.read(self.fd, EVENT.size)
                except BlockingIOError:
                    break
                if len(data) != EVENT.size:
                    raise OSError('Joystick disconnected or incomplete event')
                _, value, kind, number = EVENT.unpack(data)
                initial = bool(kind & 0x80)
                kind &= ~0x80
                if kind == 2 and number < len(self.axis_map):
                    self.axes[self.axis_map[number]] = float(np.clip(value / 32767., -1., 1.))
                elif kind == 1 and number < len(self.button_map):
                    code = self.button_map[number]
                    if value:
                        if code not in self.buttons and not initial:
                            self.pressed.add(code)
                        self.buttons.add(code)
                    else:
                        self.buttons.discard(code)
        except OSError as exc:
            self.close()
            self.message = f'Xbox disconnected: {exc}; motion stopped.'
            return False
        return set(self.codes.values()).union({16, 17}).issubset(self.axes)

    def controls(self):
        # Linux joystick axes increase right/down, opposite to joy_node's signs.
        result = {key: -self.axes.get(self.codes[key], 0.) for key in ('lx', 'ly', 'rx', 'ry')}
        result.update({key: (self.axes.get(self.codes[key], -1.) + 1.) / 2. for key in ('lt', 'rt')})
        result.update(dx=-self.axes.get(16, 0.), dy=-self.axes.get(17, 0.))
        return result


class XboxControl:
    """Require neutral input after pause/disconnect; repeat D-pad every 0.2 s."""

    def __init__(self, deadband=.12):
        self.deadband = deadband
        self.armed = False
        self.next_grip = 0.

    def reset(self):
        self.armed = False
        self.next_grip = 0.

    def update(self, connected, enabled, axes, buttons, opening, now):
        if not connected or not enabled:
            self.reset()
            return False, opening
        if not self.armed:
            neutral = all(abs(value) <= self.deadband for value in axes.values()) and not buttons
            self.armed = neutral
            return False, opening
        dx, dy = axes.get('dx', 0.), axes.get('dy', 0.)
        if not dx and not dy:
            self.next_grip = now
        elif now >= self.next_grip:
            # Reference gripper_forwarder: 100% is CLOSED; 0% is OPEN.
            percent = 100. * (1. - opening / .1)
            percent += 3. * np.sign(dx)
            if dy > 0:
                percent = 100.
            elif dy < 0:
                percent = 65.
            opening = .1 * (1. - np.clip(percent, 0., 100.) / 100.)
            self.next_grip = now + .2
        return True, float(opening)
