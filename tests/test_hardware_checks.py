import struct
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'hardware_checks'))
from probe_can import CAN_ERR_FLAG, CAN_FRAME, inside_project, parse_frame
from read_dual_nero import (JOINT_PARTS, read_joint_feedback,
                            read_gripper_feedback, read_leader_gripper_feedback)


class HardwareCheckTests(unittest.TestCase):
    def test_can_frame_parsing(self):
        payload = CAN_FRAME.pack(0x2A5, 8, b'12345678')
        self.assertEqual(parse_frame(payload), (0x2A5, 8, False))
        self.assertEqual(parse_frame(CAN_FRAME.pack(CAN_ERR_FLAG | 0x40, 0, b'12345678')),
                         (0x40, 0, True))
        with self.assertRaises(ValueError):
            parse_frame(payload[:5])

    def test_feedback_needs_all_joint_parts(self):
        parser = SimpleNamespace(**{name: object() for name in JOINT_PARTS[:-1]})
        message = SimpleNamespace(msg=[.1] * 7, timestamp=123., hz=50.)
        robot = SimpleNamespace(_parser=parser, get_joint_angles=lambda: message)
        self.assertIsNone(read_joint_feedback(robot, 'get_joint_angles', JOINT_PARTS))
        setattr(parser, JOINT_PARTS[-1], object())
        self.assertEqual(read_joint_feedback(robot, 'get_joint_angles', JOINT_PARTS)['rad'], [.1] * 7)
        message.msg = [float('nan')] * 7
        self.assertIsNone(read_joint_feedback(robot, 'get_joint_angles', JOINT_PARTS))

    def test_gripper_feedback_width_and_force(self):
        state = SimpleNamespace(value=.047, force=1.2, mode='width')
        wrapped = SimpleNamespace(msg=state, timestamp=123., hz=30.)
        effector = SimpleNamespace(get_gripper_status=lambda: wrapped)
        data = read_gripper_feedback(effector)
        self.assertEqual((data['value'], data['mode'], data['force_N']), (.047, 'width', 1.2))

    def test_leader_gripper_echo_width_and_angle_units(self):
        state = SimpleNamespace(value=.047, force=1.2, status_code=1, set_zero=0)
        wrapped = SimpleNamespace(msg=state, timestamp=123., hz=200.)
        effector = SimpleNamespace(get_gripper_ctrl_states=lambda: wrapped)
        width = read_leader_gripper_feedback(effector)
        self.assertEqual((width['value'], width['mode'], width['status_code']),
                         (.047, 'width', 1))
        state.value = .09  # Vendored 0x159 decoder scales angle millidegrees as metres.
        state.status_code = 5
        angle = read_leader_gripper_feedback(effector)
        self.assertEqual((angle['value'], angle['mode']), (90., 'angle'))

    def test_report_must_stay_in_project(self):
        self.assertTrue(inside_project(PROJECT / 'hardware_checks' / 'reports' / 'check.json').is_relative_to(PROJECT))
        with self.assertRaises(ValueError):
            inside_project('/tmp/outside-hardware-check.json')


if __name__ == '__main__':
    unittest.main()
