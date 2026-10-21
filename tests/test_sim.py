import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np

from nero_bridge.data import load_npz
from sim_bridge.data import metrics, require_sim, save_demo, shifted_frames
from sim_bridge.app import apply_ik_target, gripper_joint_targets, keyboard_key_name, keyboard_pose


class SimulationDataTests(unittest.TestCase):
    def test_gripper_targets_respect_urdf_limits(self):
        robot = ET.parse(Path(__file__).resolve().parents[1] / "assets/nero/nero.urdf").getroot()
        for opening, expected in [(0., [0., 0.]), (.05, [.025, -.025]),
                                  (.1, [.05, -.05]), (-1., [0., 0.]), (1., [.05, -.05])]:
            result = gripper_joint_targets(opening)
            np.testing.assert_allclose(result, expected)
            for index, value in enumerate(result, 1):
                limit = robot.find(f"joint[@name='gripper_joint{index}']/limit")
                self.assertGreaterEqual(value, float(limit.attrib["lower"]))
                self.assertLessEqual(value, float(limit.attrib["upper"]))
        for invalid in [float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                gripper_joint_targets(invalid)

    def test_keyboard_string_and_enum_inputs(self):
        for name in ("I", "K", "J", "L", "U", "O", "ESCAPE",
                     "KEY_1", "KEY_2", "KEY_3", "KEY_4", "KEY_5", "KEY_6"):
            with self.subTest(name=name):
                self.assertEqual(keyboard_key_name(name), name)
                self.assertEqual(keyboard_key_name(name.lower()), name)
                self.assertEqual(keyboard_key_name(SimpleNamespace(name=name)), name)
        for digit in "123456":
            self.assertEqual(keyboard_key_name(digit), "KEY_" + digit)
        self.assertEqual(keyboard_key_name(None), "")
        self.assertEqual(keyboard_key_name("F12"), "F12")

    def test_keyboard_translation_and_inverse(self):
        pose = np.array([.3, .1, .4, 1., 0, 0, 0])
        for axis, positive, negative in [(0, "I", "K"), (1, "J", "L"), (2, "U", "O")]:
            moved = keyboard_pose(pose, positive, .001, .01)
            expected = pose.copy()
            expected[axis] += .001
            np.testing.assert_allclose(moved, expected)
            np.testing.assert_allclose(keyboard_pose(moved, negative, .001, .01), pose)

    def test_keyboard_rotation_uses_world_axes(self):
        from scipy.spatial.transform import Rotation
        initial = Rotation.from_euler("z", 90, degrees=True)
        pose = np.r_[[.3, .1, .4], initial.as_quat()[[3, 0, 1, 2]]]
        for axis in range(3):
            moved = keyboard_pose(pose, f"KEY_{axis * 2 + 1}", .001, .1)
            delta = np.zeros(3)
            delta[axis] = .1
            expected = Rotation.from_rotvec(delta) * initial
            np.testing.assert_allclose(Rotation.from_quat(moved[[4, 5, 6, 3]]).as_matrix(), expected.as_matrix(), atol=1e-12)
            np.testing.assert_allclose(moved[:3], pose[:3])
            np.testing.assert_allclose(keyboard_pose(moved, f"KEY_{axis * 2 + 2}", .001, .1), pose, atol=1e-12)

    def test_joint_command_keeps_advancing_when_measured_joint_lags(self):
        state = SimpleNamespace(joint_positions=np.zeros(2))
        controller = SimpleNamespace(get_applied_action=lambda: state)
        robot = SimpleNamespace(get_articulation_controller=lambda: controller,
                                get_joint_positions=lambda **kw: np.zeros(1),
                                apply_action=lambda action: state.joint_positions.__setitem__(
                                    action.joint_indices, action.joint_positions))
        for _ in range(20):
            old = state.joint_positions.copy()
            action = SimpleNamespace(joint_indices=np.array([1]), joint_positions=np.array([1.]))
            apply_ik_target(robot, action, .05)
            self.assertLessEqual(abs(state.joint_positions[1] - old[1]), .05 + 1e-12)
        np.testing.assert_allclose(state.joint_positions, [0, 1])

    def test_recording_metadata_and_frames(self):
        pose = np.array([.3, 0, .3, 1., 0, 0, 0])
        rows = np.column_stack([np.arange(3) / 30, np.tile(pose, (3, 1))])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.npz"
            save_demo(path, rows, np.zeros((3, 7)), pose)
            data = require_sim(load_npz(path))
            self.assertTrue(data["synthetic"])
            self.assertEqual(str(data["source"]), "isaacsim-5.1")
            frame = shifted_frames(data, [.1, 0, 0])["frames"]["target"]
            self.assertAlmostEqual(frame[0], .4)
            with self.assertRaises(FileExistsError):
                save_demo(path, rows, np.zeros((3, 7)), pose)
            with self.assertRaises(ValueError):
                save_demo(Path(directory) / "bad.npz", rows[[0, 0, 2]], np.zeros((3, 7)), pose)
            data["synthetic"] = np.asarray(False)
            with self.assertRaises(ValueError):
                require_sim(data)

    def test_metrics_and_antipodal_quaternions(self):
        wanted = np.tile([.3, 0, .3, 1., 0, 0, 0], (10, 1))
        actual = wanted.copy()
        actual[:, 0] += .002
        actual[:, 3] *= -1
        report = metrics(wanted, actual)
        self.assertTrue(report["passed"])
        self.assertAlmostEqual(report["position_rmse_m"], .002)
        self.assertAlmostEqual(report["angle_max_rad"], 0)
        actual[-1, 0] += .02
        self.assertFalse(metrics(wanted, actual)["passed"])

    def test_portable_robot_asset(self):
        directory = Path(__file__).resolve().parents[1] / "assets/nero"
        robot = ET.parse(directory / "nero.urdf").getroot()
        self.assertEqual(len(robot.findall("joint[@type='revolute']")), 7)
        self.assertIsNotNone(robot.find("link[@name='gripper_base']"))
        self.assertIsNone(robot.find("link[@name='world']"))
        for mesh in robot.findall(".//mesh"):
            self.assertTrue((directory / mesh.attrib["filename"]).is_file())


if __name__ == "__main__":
    unittest.main()
