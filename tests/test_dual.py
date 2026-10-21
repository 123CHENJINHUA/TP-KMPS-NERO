import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from sim_bridge.dual import (gripper_command_step, mount_poses, selected_after_y,
                             physics_view_ready, robot_handles_ready)
from sim_bridge.gamepad import X, Y


class DualArmTests(unittest.TestCase):
    def test_fixed_bases_face_positive_x_after_opposite_90_degree_turns(self):
        left, right = mount_poses(.9, .5)
        np.testing.assert_allclose(left[0], [0, -.45, .5])
        np.testing.assert_allclose(right[0], [0, .45, .5])
        for _, orientation in (left, right):
            heading = Rotation.from_quat(orientation[[1, 2, 3, 0]]).apply([1, 0, 0])
            np.testing.assert_allclose(heading, [1., 0., 0.], atol=1e-8)
            self.assertAlmostEqual(np.linalg.norm(orientation), 1.)

    def test_y_switches_once_and_only_when_enabled(self):
        self.assertEqual(selected_after_y(0, {Y}, True, True, True), 1)
        self.assertEqual(selected_after_y(1, {Y}, True, True, True), 0)
        for connected, enabled, playing in ((False, True, True), (True, False, True),
                                            (True, True, False)):
            self.assertEqual(selected_after_y(0, {Y}, connected, enabled, playing), 0)
        self.assertEqual(selected_after_y(1, set(), True, True, True), 1)
        self.assertEqual(selected_after_y(0, {X}, True, True, True), 0)
        self.assertNotEqual(X, Y)

    def test_live_teaching_gripper_target_has_no_artificial_ramp(self):
        current = np.array([0., 0.])
        np.testing.assert_allclose(gripper_command_step(current, .1, 120, True),
                                   [.05, -.05])
        normal = gripper_command_step(current, .1, 120, False)
        self.assertLess(abs(normal[0]), .001)
        self.assertLess(abs(normal[1]), .001)

    def test_disposed_and_stopped_views_are_detected_without_crashing(self):
        class View:
            def __init__(self, state):
                self.state = state
            def is_valid(self):
                return self.state != "disposed"
            def is_physics_handle_valid(self):
                if self.state == "missing_attribute":
                    raise AttributeError("'Articulation' object has no attribute '_physics_view'")
                return self.state == "ready"
        for state in ("disposed", "missing_attribute", "stopped"):
            self.assertFalse(physics_view_ready(View(state)))
        self.assertTrue(physics_view_ready(View("ready")))
        self.assertFalse(physics_view_ready(None))

    def test_robot_handle_guard_covers_missing_physics_view(self):
        class Robot:
            def is_valid(self):
                return True
            @property
            def handles_initialized(self):
                raise AttributeError("missing _physics_view")
        self.assertFalse(robot_handles_ready(Robot()))

    def test_second_robot_uses_same_meshes_and_distinct_root_name(self):
        asset = Path(__file__).resolve().parents[1] / 'assets/nero'
        first = ET.parse(asset / 'nero.urdf').getroot()
        second = ET.parse(asset / 'nero_dual_2.urdf').getroot()
        self.assertEqual((first.attrib['name'], second.attrib['name']), ('nero', 'nero2'))
        self.assertEqual([mesh.attrib['filename'] for mesh in first.findall('.//mesh')],
                         [mesh.attrib['filename'] for mesh in second.findall('.//mesh')])
        self.assertEqual([joint.attrib['name'] for joint in first.findall('joint')],
                         [joint.attrib['name'] for joint in second.findall('joint')])


if __name__ == '__main__':
    unittest.main()
