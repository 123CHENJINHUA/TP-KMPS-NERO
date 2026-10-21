"""Offline safety/format tests. No SocketCAN socket is opened."""
from .bootstrap import HERE
import struct
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from .data import config, align, read_record
from .hardware import Feedback, _Transport, Reader
from .kinematics import tcp_pose, fk_matrix, pose_matrix, inverse_path
from .test_real import check_start, check_tracking, execute
from .planning import verify_plan, check_path, gripper_profile
from .make_synthetic import generate
from .test_visual_collection import VisualCollectionTests


def ready_feedback():
    fb = Feedback()
    for identifier in (0x2A5, 0x2A6, 0x2A7, 0x2A9):
        fb.feed(identifier, struct.pack('>ii', 1000, 2000), 10.)
    fb.feed(0x2A8, struct.pack('>ihBB', 20000, 1000, 0, 0), 10.)
    fb.feed(0x159, struct.pack('>ihBB', 0, 4000, 1, 0), 10.)
    return fb


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.cfg = config()

    def test_measured_not_leader_and_incomplete_rejected(self):
        fb = Feedback()
        for identifier in (0x155, 0x156, 0x157, 0x170):
            fb.feed(identifier, bytes(8), 10.)
        with self.assertRaises(ValueError):
            fb.snapshot(10., self.cfg)
        state = ready_feedback().snapshot(10.01, self.cfg)
        self.assertAlmostEqual(state['joints_rad'][0], np.deg2rad(1))
        self.assertAlmostEqual(state['gripper_measured'], .02)
        self.assertEqual(state['gripper_command'], 0)

    def test_stale_skew_zeroing_and_mixed_modes_rejected(self):
        fb = ready_feedback()
        with self.assertRaises(ValueError):
            fb.snapshot(10.2, self.cfg)
        fb.stamps[0] = 9.94
        with self.assertRaises(ValueError):
            fb.snapshot(10.01, self.cfg)
        fb = ready_feedback()
        fb.feed(0x159, struct.pack('>ihBB', 0, 4000, 1, 1), 10.)
        with self.assertRaises(ValueError):
            fb.snapshot(10.01, self.cfg)
        fb.feed(0x159, struct.pack('>ihBB', 45000, 4000, 5, 0), 10.)
        with self.assertRaises(ValueError):
            fb.snapshot(10.01, self.cfg)
        fb.feed(0x2A8, struct.pack('>ihBB', 40000, 1000, 0, 1), 10.)
        state = fb.snapshot(10.01, self.cfg)
        self.assertEqual(state['gripper_mode'], 'angle')
        self.assertEqual(state['gripper_command'], 45.)

    def test_foreign_control_check(self):
        reader = Reader.__new__(Reader)
        import threading
        reader.lock, reader.feedback = threading.Lock(), ready_feedback()
        with patch('real_tpkmp.hardware.time.monotonic', return_value=10.2):
            with self.assertRaises(ValueError):
                reader.require_exclusive()
        with patch('real_tpkmp.hardware.time.monotonic', return_value=12.):
            reader.require_exclusive()
        stamp = reader.foreign_stamp()
        reader.feedback.feed(0x155, bytes(8), 12.)
        with self.assertRaises(ValueError):
            reader.require_no_new_controller(stamp)

    def test_sdk_geometry_and_wire_commands_without_can(self):
        import pyAgxArm as sdk
        from scipy.spatial.transform import Rotation
        cfg = sdk.create_agx_arm_config(robot=sdk.ArmModel.NERO, firmeware_version='v121',
                                       auto_connect=False, log_level='ERROR')
        robot = sdk.AgxArmFactory.create_arm(cfg)
        tool = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        class Bus:
            def __init__(self):
                self.messages = []
            def send(self, msg, timeout=None):
                self.messages.append(msg)
        bus = Bus()
        self.assertIsNone(robot._ctx.comm)
        self.assertEqual(len(bus.messages), 0)
        for q in (np.zeros(7), np.array([-.3, -.3, .3, 1, .1, .1, 0])):
            official = robot.fk(q.tolist())
            transform = fk_matrix(q)
            np.testing.assert_allclose(transform[:3, 3], official[:3], atol=1e-9)
            np.testing.assert_allclose(transform[:3, :3], Rotation.from_euler('xyz', official[3:]).as_matrix(), atol=1e-9)
        robot._ctx.comm = _Transport(bus, 'offline')
        robot.set_auto_set_motion_mode_enabled(False)
        robot.set_joint_limits_enabled(True)
        robot.set_speed_percent(10)
        robot.set_motion_mode('j')
        robot.move_j([0.]*7)
        tool.move_gripper_m(.03, 4.)
        tool.move_gripper_deg(45., 4.)
        robot.electronic_emergency_stop()
        ids = {m.arbitration_id for m in bus.messages}
        self.assertTrue({0x151, 0x155, 0x156, 0x157, 0x170, 0x159, 0x150} <= ids)
        self.assertTrue(ids <= {0x151, 0x155, 0x156, 0x157, 0x170, 0x159, 0x150})
        grip = [m for m in bus.messages if m.arbitration_id == 0x159]
        self.assertEqual(struct.unpack('>ihBB', grip[0].data), (30000, 4000, 1, 0))
        self.assertEqual(struct.unpack('>ihBB', grip[1].data), (45000, 4000, 5, 0))
        robot._ctx.comm = None

    def test_ik_unreachable_and_round_trip(self):
        q = np.array([-.3, -.3, .3, 1., .1, .1, 0.])
        target = tcp_pose(q, self.cfg)
        result = inverse_path([target], q, [q], self.cfg)
        np.testing.assert_allclose(result[0], q, atol=1e-6)
        target[0] = 3.
        with self.assertRaises(ValueError):
            inverse_path([target], q, [q], self.cfg)

    def test_synthetic_execution_rejected_before_hardware(self):
        with patch('real_tpkmp.hardware.Reader', side_effect=AssertionError('hardware opened')):
            with self.assertRaises(ValueError):
                execute('unused', dict(synthetic=True), {}, self.cfg, HERE)

    def test_execution_feedback_fault_sends_stop_and_saves_report(self):
        from unittest.mock import MagicMock
        import json
        q = np.array([-.3, -.3, .3, 1., .1, .1, 0.])
        state = dict(joints_rad=q, tcp=tcp_pose(q, self.cfg), gripper_mode='width', gripper_measured=.08)
        data = dict(time_s=np.array([0., .02, .04]), joints_rad=np.tile(q, (3, 1)),
                    tcp=np.tile(state['tcp'], (3, 1)), gripper_command=np.full(3, .08),
                    gripper_force_N=np.full(3, 4.), phase=np.linspace(0, 1, 3))
        fake = MagicMock()
        fake.writer_attached = False
        def sample(*args):
            if fake.writer_attached:
                raise ValueError('simulated stale measured feedback')
            return state
        fake.sample.side_effect = sample
        fake.attach_writer.side_effect = lambda: setattr(fake, 'writer_attached', True)
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            from pathlib import Path
            with (patch('real_tpkmp.test_real.verify_plan'),
                  patch('real_tpkmp.hardware.Reader', return_value=fake),
                  patch('real_tpkmp.test_real.digest', return_value='test'),
                  patch('real_tpkmp.test_real.time.sleep'),
                  patch('real_tpkmp.test_real.time.monotonic', side_effect=np.arange(1000)*.03)):
                with self.assertRaises(SystemExit):
                    execute('unused', dict(gripper_mode='width'), data, self.cfg, Path(temp))
            fake.robot.electronic_emergency_stop.assert_called_once()
            fake.close.assert_called_once()
            report = json.loads((Path(temp)/'execution.json').read_text())
            self.assertIn('stale', report['failure'])
            self.assertFalse(report['completed'])

    def test_start_tracking_and_workspace_checks(self):
        q = np.array([-.3, -.3, .3, 1., .1, .1, 0.])
        pose = tcp_pose(q, self.cfg)
        state = dict(joints_rad=q.copy(), tcp=pose.copy(), gripper_mode='width', gripper_measured=.08)
        data = dict(joints_rad=np.array([q]), tcp=np.array([pose]), gripper_command=np.array([.08]))
        check_start(state, data, self.cfg, 'width')
        state['joints_rad'][0] += .3
        with self.assertRaises(ValueError):
            check_start(state, data, self.cfg, 'width')
        with self.assertRaises(ValueError):
            check_tracking(state, q, pose, self.cfg)
        bad = np.tile(pose, (3, 1))
        bad[:, 0] = 2
        with self.assertRaises(ValueError):
            check_path(np.arange(3), np.tile(q, (3, 1)), bad, np.full(3, .08), 'width', self.cfg)

    def test_gripper_closes_before_lift_and_opens_before_retreat(self):
        phase = np.linspace(0, 1, 96)
        for opened, closed in ((.08, 0.), (95., 10.)):
            commands = np.where((phase >= 1/6) & (phase < 4/6), closed, opened)
            query = np.array([0., 1/6, .28, 1/3, .5, .78, 5/6, 1.])
            values = gripper_profile(query, commands)
            np.testing.assert_allclose(values, [opened, opened, closed, closed, closed, opened, opened, opened], atol=1e-12)

    def test_alignment_and_no_gap_acceptance(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            directory = generate(f'{temp}/fixtures', demos=1)
            _, data = read_record(directory/'demo_001.npz')
            result = align(data, 96, self.cfg)
            np.testing.assert_allclose(result['phase'], np.linspace(0, 1, 96))
            self.assertEqual(result['command'].min(), 0.)
            self.assertAlmostEqual(result['measured'].min(), .02)
            data['time_s'][50:] += 1.
            with self.assertRaises(ValueError):
                align(data, 96, self.cfg)


if __name__ == '__main__':
    unittest.main(verbosity=2)
