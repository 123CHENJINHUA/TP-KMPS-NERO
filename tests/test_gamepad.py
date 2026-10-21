import os
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from sim_bridge.gamepad import A, B, X, Y, START, EVENT, XboxJoystick, XboxControl, gamepad_pose


class GamepadTests(unittest.TestCase):
    def test_tcp_frame_translation_and_analog_speed(self):
        rotation = Rotation.from_euler('z', 90, degrees=True)
        pose = np.r_[[.3, 0, .3], rotation.as_quat()[[3, 0, 1, 2]]]
        moved = gamepad_pose(pose, {'lx': -1}, set(), .1)
        np.testing.assert_allclose(moved[:3], pose[:3] + [-.003, 0, 0], atol=1e-12)
        neutral = gamepad_pose(pose, {'lx': .1, 'ry': -.1, 'lt': .05}, set(), .1)
        np.testing.assert_allclose(neutral, pose)
        half = gamepad_pose(pose, {'lx': -.56}, set(), .1)
        np.testing.assert_allclose(half[:3] - pose[:3], (moved[:3] - pose[:3]) / 2, atol=1e-12)

    def test_trigger_signs_and_body_rotations(self):
        pose = np.array([.3, 0, .3, 1, 0, 0, 0.])
        for axes, z in [({'lt': 1}, .003), ({'rt': 1}, -.003), ({'lt': 1, 'rt': 1}, 0.)]:
            moved = gamepad_pose(pose, axes, set(), .1)
            self.assertAlmostEqual(moved[2] - pose[2], z)
        initial = Rotation.from_euler('y', 75, degrees=True)
        pose[3:] = initial.as_quat()[[3, 0, 1, 2]]
        for axes, buttons, vector in [({'rx': 1}, set(), [1, 0, 0]),
                                       ({'ry': 1}, set(), [0, 1, 0]),
                                       ({}, {A}, [0, 0, 1]), ({}, {B}, [0, 0, -1])]:
            moved = gamepad_pose(pose, axes, buttons, .1)
            expected = initial * Rotation.from_rotvec(np.array(vector) * np.deg2rad(1.5))
            np.testing.assert_allclose(Rotation.from_quat(moved[[4, 5, 6, 3]]).as_matrix(), expected.as_matrix(), atol=1e-12)
        np.testing.assert_allclose(gamepad_pose(pose, {}, {A, B}, .1), pose, atol=1e-12)

    def test_neutral_required_after_disconnect_or_pause(self):
        control = XboxControl()
        def update(connected=True, enabled=True, axes=None, buttons=None):
            return control.update(connected, enabled, axes or {}, buttons or set(), .05, 0)[0]
        self.assertFalse(update(axes={'lt': 1}))
        self.assertFalse(update(buttons={A}))
        self.assertFalse(update())  # Arm, without motion on this sample.
        self.assertTrue(update(axes={'lt': 1}))
        self.assertFalse(update(connected=False))
        self.assertFalse(update(axes={'lt': 1}))
        self.assertFalse(update())
        self.assertTrue(update())
        self.assertFalse(update(enabled=False))
        self.assertFalse(update(axes={'dx': 1}))

    def test_gripper_reference_polarity_repeat_and_limits(self):
        control = XboxControl()
        control.update(True, True, {}, set(), .05, 0)
        def grip(axes, opening, now):
            return control.update(True, True, axes, set(), opening, now)[1]
        self.assertAlmostEqual(grip({'dx': 1}, .05, 0), .047)
        self.assertAlmostEqual(grip({'dx': 1}, .047, .1), .047)
        self.assertAlmostEqual(grip({'dx': -1}, .047, .21), .05)
        self.assertAlmostEqual(grip({'dy': 1}, .05, .5), 0)
        self.assertAlmostEqual(grip({'dy': -1}, 0, .8), .035)
        self.assertAlmostEqual(grip({'dx': -1}, .1, 1.1), .1)
        self.assertAlmostEqual(grip({'dx': 1}, 0, 1.4), 0)

    def test_linux_events_initialization_held_state_and_disconnect(self):
        read_fd, write_fd = os.pipe()
        os.set_blocking(read_fd, False)
        joy = XboxJoystick()
        joy.fd = read_fd
        joy.axis_map = [0, 1, 2, 3, 4, 5, 16, 17]
        joy.button_map = [A, B, X, Y, START]
        joy.codes = dict(lx=0, ly=1, rx=3, ry=4, lt=2, rt=5)
        try:
            initial = b''.join(EVENT.pack(0, -32767 if code in (2, 5) else 0, 0x82, index)
                               for index, code in enumerate(joy.axis_map))
            os.write(write_fd, initial + EVENT.pack(0, 1, 0x81, 4))
            self.assertTrue(joy.poll(0))
            self.assertFalse(joy.pressed)  # Reconnect with Menu held cannot toggle controls.
            self.assertEqual(joy.controls()['lt'], 0)
            os.write(write_fd, EVENT.pack(1, -32767, 2, 0) + EVENT.pack(1, 32767, 2, 2)
                     + EVENT.pack(1, 0, 1, 4) + EVENT.pack(2, 1, 1, 4))
            self.assertTrue(joy.poll(.1))
            self.assertEqual(joy.controls()['lx'], 1)
            self.assertEqual(joy.controls()['lt'], 1)
            self.assertEqual(joy.pressed, {START})
            self.assertTrue(joy.poll(.2))  # Held controls need no repeated OS event.
            self.assertEqual(joy.controls()['lt'], 1)
            self.assertFalse(joy.pressed)
            os.write(write_fd, EVENT.pack(3, 1, 1, 2))
            self.assertTrue(joy.poll(.24))
            self.assertEqual(joy.pressed, {X})
            os.write(write_fd, EVENT.pack(3, 1, 1, 3))
            self.assertTrue(joy.poll(.25))
            self.assertEqual(joy.pressed, {Y})
            self.assertTrue(joy.poll(.26))
            self.assertFalse(joy.pressed)
            os.close(write_fd)
            write_fd = None
            self.assertFalse(joy.poll(.3))
            self.assertEqual(joy.axes, {})
            self.assertEqual(joy.buttons, set())
            self.assertIsNone(joy.fd)
        finally:
            if write_fd is not None:
                os.close(write_fd)
            joy.close()

    def test_layout_discovery_and_permission_error(self):
        for layout, expected in [([0, 1, 2, 3, 4, 5, 16, 17], (3, 4, 2, 5)),
                                 ([0, 1, 2, 5, 9, 10, 16, 17], (2, 5, 10, 9))]:
            def ioctl(fd, command, data, mutate):
                if command == 0x80016a11:
                    data[0] = len(layout)
                elif command == 0x80016a12:
                    data[0] = 5
                elif command == 0x80406a32:
                    for index, value in enumerate(layout):
                        data[index] = value
                elif command == 0x84006a34:
                    for index, value in enumerate([A, B, X, Y, START]):
                        data[index] = value
            joy = XboxJoystick('/dev/input/js0')
            with patch('sim_bridge.gamepad.os.open', return_value=99), patch('sim_bridge.gamepad.fcntl.ioctl', side_effect=ioctl), patch('sim_bridge.gamepad.os.close'):
                joy._open('/dev/input/js0')
                self.assertEqual(tuple(joy.codes[key] for key in ('rx', 'ry', 'lt', 'rt')), expected)
                joy.close()
        joy = XboxJoystick('/dev/input/js0')
        with patch('sim_bridge.gamepad.os.open', side_effect=PermissionError('Permission denied')):
            self.assertFalse(joy.poll(0))
            self.assertIn('Permission denied', joy.message)


if __name__ == '__main__':
    unittest.main()
