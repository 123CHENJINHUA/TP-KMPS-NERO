import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from nero_bridge.data import load_npz, save_npz
from sim_bridge.data import demo_arrays, require_sim
from sim_bridge.grasp import (GRASP_OFFSET, GRASP_SEED, TASK, GraspConfig, assess_grasp,
                              grasp_pose, grasp_prediction_fields, grasp_recording_fields,
                              grasp_skill_fields, grasp_waypoint, held_commands)
from sim_bridge.grasp import grasp_position_bounds, sample_grasp_positions


class GraspTests(unittest.TestCase):
    def test_stratified_positions_cover_each_axis_and_repeat_by_seed(self):
        bounds = grasp_position_bounds(x_range=(.28, .34), y_range=(-.05, .05))
        points = sample_grasp_positions(12, 42, bounds, "stratified")
        self.assertEqual(points.shape, (12, 2))
        unit = (points - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
        self.assertTrue(np.all((unit >= 0) & (unit < 1)))
        for axis in range(2):
            np.testing.assert_array_equal(np.sort(np.floor(unit[:, axis] * 12)), np.arange(12))
        np.testing.assert_array_equal(points, sample_grasp_positions(12, 42, bounds, "stratified"))
        self.assertFalse(np.array_equal(points, sample_grasp_positions(12, 43, bounds, "stratified")))
        # The 30 mm cube stays on the existing 180 mm table, including a 10 mm margin.
        self.assertTrue(np.all(np.abs(points - [.3, 0.]) + .015 + .01 <= .09))

    def test_sampling_preserves_old_random_and_fixed_position_modes(self):
        bounds = grasp_position_bounds(.012)
        expected = np.array([.3, 0.]) + np.random.default_rng(0).uniform(-.012, .012, (5, 2))
        np.testing.assert_allclose(sample_grasp_positions(5, 0, bounds), expected)
        fixed = sample_grasp_positions(3, 0, grasp_position_bounds(0), "stratified")
        np.testing.assert_allclose(fixed, np.tile([.3, 0.], (3, 1)))

    def test_rejects_outside_or_reversed_sampling_bounds(self):
        for x, y in [((.25, .34), (-.05, .05)), ((.28, .35), (-.05, .05)),
                     ((.28, .34), (-.06, .05)), ((.34, .28), (-.05, .05)),
                     ((float("nan"), .34), (-.05, .05))]:
            with self.subTest(x=x, y=y), self.assertRaises(ValueError):
                grasp_position_bounds(x_range=x, y_range=y)

    def setUp(self):
        self.config = GraspConfig()
        self.target = np.array([.3, 0., .165, 1., 0., 0., 0.])
        self.times = np.linspace(0, 8, 241)
        plan = [grasp_waypoint(self.target, t / 8, self.config) for t in self.times]
        self.tcp = np.array([p[0] for p in plan])
        self.commands = np.array([p[1] for p in plan])
        self.stages = np.array([p[2] for p in plan])
        self.objects = np.tile(self.target, (len(plan), 1))
        lifting = self.times >= 8 * .65
        self.objects[lifting, 2] += self.tcp[lifting, 2] - grasp_pose(self.target)[2]
        self.fingers = np.tile([.015, -.015], (len(plan), 1))
        self.rows = np.column_stack([self.times, self.tcp])

    def assess(self):
        return assess_grasp(self.times, self.tcp, self.objects, self.fingers,
                            self.target, self.config, 8.)

    def fields(self):
        return grasp_recording_fields(self.rows, self.fingers, self.commands, self.objects,
                                      self.stages, self.config, 8., True)

    def test_grasp_requires_sustained_lift_not_just_arm_motion(self):
        self.assertTrue(self.assess()["passed"])
        self.objects[:-1, 2] = self.target[2]
        report = self.assess()
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["lifted"])

    def test_diagnoses_object_slipping_after_initial_grip(self):
        # Match the observed failure: 30 mm gap before lift, small lift, then empty closure.
        lift = self.times >= 8 * .65
        self.objects[lift, 2] = self.target[2]
        self.objects[np.argmin(abs(self.times - 5.8)), 2] += .008
        self.fingers[self.times >= 6.4] = 0.
        report = self.assess()
        self.assertFalse(report["passed"])
        self.assertIn("slip during lift", report["reason"])
        self.assertAlmostEqual(report["pre_lift_finger_gap_m"], .03)
        self.assertAlmostEqual(report["max_cube_lift_m"], .008)

    def test_diagnoses_missed_grip_separately(self):
        self.fingers[:] = 0.
        report = self.assess()
        self.assertFalse(report["passed"])
        self.assertIn("did not close around", report["reason"])

    def test_rejects_cube_away_from_fingers_and_empty_closed_gripper(self):
        self.objects[:, 0] += .1
        self.assertFalse(self.assess()["checks"]["between_fingers"])
        self.objects[:, 0] -= .1
        self.fingers[:] = 0.
        self.assertFalse(self.assess()["checks"]["fingers_blocked"])

    def test_slip_and_incomplete_hold_fail(self):
        self.objects[-1, 0] += .01
        self.assertFalse(self.assess()["checks"]["stable"])
        report = assess_grasp(self.times[:-30], self.tcp[:-30], self.objects[:-30], self.fingers[:-30],
                              self.target, self.config, 8.)
        self.assertFalse(report["passed"])

    def test_top_down_geometry_and_phase_boundaries(self):
        pose = grasp_pose(self.target)
        rotation = Rotation.from_quat(pose[[4, 5, 6, 3]])
        np.testing.assert_allclose(pose[:3] + rotation.apply(GRASP_OFFSET), self.target[:3])
        np.testing.assert_allclose(rotation.apply([0, 0, 1]), [0, 0, -1], atol=1e-12)
        # Furthest fingertip is 135.8 mm from gripper_base; it must clear the table.
        self.assertGreater(pose[2] - .1358 - self.config.table_height, .003)
        self.assertEqual(grasp_waypoint(self.target, .4, self.config)[1:], (0., 2))
        for boundary in [.1, .4, .65, .9]:
            a = grasp_waypoint(self.target, boundary - 1e-8, self.config)[0]
            b = grasp_waypoint(self.target, boundary, self.config)[0]
            np.testing.assert_allclose(a, b, atol=1e-7)
        tipped = self.target.copy()
        tipped[3:] = Rotation.from_euler("x", .4).as_quat()[[3, 0, 1, 2]]
        with self.assertRaises(ValueError):
            grasp_pose(tipped)

    def test_recording_roundtrip_and_training_profile(self):
        arrays = demo_arrays(self.rows, np.zeros((len(self.rows), 7)), self.target)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo_001.npz"
            save_npz(path, **arrays, **self.fields())
            recording = require_sim(load_npz(path))
            self.assertEqual(str(recording["task"]), TASK)
            np.testing.assert_allclose(recording["object_poses"], self.objects)
            skill = grasp_skill_fields([recording, recording], 100)
            prediction = grasp_prediction_fields(skill)
            self.assertEqual(prediction["grasp_duration"], 8.)
            phase = prediction["gripper_phase"]
            self.assertTrue(np.all(prediction["gripper_commands"][phase < .4] == .08))
            self.assertTrue(np.all(prediction["gripper_commands"][phase > .42] == 0.))
            with self.assertRaises(ValueError):
                grasp_skill_fields([recording, arrays], 100)
            recording["grasp_success"] = np.array(False)
            with self.assertRaises(ValueError):
                grasp_skill_fields([recording, recording], 100)

    def test_discrete_gripper_events_and_input_validation(self):
        np.testing.assert_allclose(held_commands([0, .4, 1], [.08, 0, 0], [.39, .4, .8]), [.08, 0, 0])
        self.assertEqual(grasp_skill_fields([{}, {}], 100), {})
        with self.assertRaises(ValueError):
            held_commands([0, .4, .4], [.08, 0, 0], .2)
        with self.assertRaises(ValueError):
            grasp_recording_fields(self.rows, self.fingers[:-1], self.commands, self.objects,
                                   self.stages, self.config, 8., True)
        with self.assertRaises(ValueError):
            GraspConfig(cube_size=.2)

    def test_train_predict_files_keep_gripper_schedule(self):
        # Exercise the real file pipeline without requiring the ML environment.
        # Numerical TP-KMP fitting itself is deliberately not simulated by this test.
        from nero_bridge import model
        arrays = demo_arrays(self.rows, np.zeros((len(self.rows), 7)), self.target)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            paths = [directory / f"demo_{index}.npz" for index in range(2)]
            for path in paths:
                save_npz(path, **arrays, **self.fields())
            skill, prediction = directory / "skill.npz", directory / "prediction.npz"
            with patch.object(model, "build_model", return_value=object()), patch.object(model, "fuse", return_value=self.tcp):
                model.train(paths, skill, samples=len(self.tcp), components=4)
                model.predict(skill, prediction)
            fitted, predicted = load_npz(skill), require_sim(load_npz(prediction))
            self.assertEqual(str(predicted["task"]), TASK)
            np.testing.assert_array_equal(fitted["gripper_commands"], predicted["gripper_commands"])
            self.assertEqual(GraspConfig.from_json(predicted["grasp_config"]), self.config)
            self.assertEqual(float(predicted["grasp_duration"]), 8.)

    def test_default_pick_paths_are_reachable_with_urdf_limits(self):
        # Independent CPU FK/IK check; this does not replace a PhysX grasp test.
        urdf = ET.parse(Path(__file__).resolve().parents[1] / "assets/nero/nero.urdf").getroot()
        chain, limits = [], []
        for joint in urdf.findall("joint"):
            if joint.attrib["name"].startswith("gripper_joint"):
                continue
            origin = joint.find("origin")
            transform = np.eye(4)
            transform[:3, 3] = np.fromstring(origin.attrib["xyz"], sep=" ")
            transform[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.attrib["rpy"], sep=" ")).as_matrix()
            axis = joint.find("axis")
            chain.append((transform, None if axis is None else np.fromstring(axis.attrib["xyz"], sep=" ")))
            if axis is not None:
                limits.append([float(joint.find("limit").attrib[key]) for key in ("lower", "upper")])

        def fk(q):
            result, index = np.eye(4), 0
            for origin, axis in chain:
                result = result @ origin
                if axis is not None:
                    delta = np.eye(4)
                    delta[:3, :3] = Rotation.from_rotvec(axis * q[index]).as_matrix()
                    result = result @ delta
                    index += 1
            return result

        lower, upper = np.array(limits).T
        for dx, dy in [(0, 0), (-.02, -.02), (-.02, .02), (.02, -.02), (.02, .02),
                       (-.02, -.05), (-.02, .05), (.04, -.05), (.04, .05),
                       (-.02, 0), (.04, 0), (.01, -.05), (.01, .05)]:
            target = self.target.copy()
            target[:2] += [dx, dy]
            q = GRASP_SEED.copy()
            for phase in [0, .25, .4, .75, 1]:
                wanted = grasp_waypoint(target, phase, self.config)[0]
                orientation = Rotation.from_quat(wanted[[4, 5, 6, 3]])

                def residual(joints):
                    actual = fk(joints)
                    return np.r_[5 * (actual[:3, 3] - wanted[:3]),
                                 (orientation.inv() * Rotation.from_matrix(actual[:3, :3])).as_rotvec()]

                result = least_squares(residual, q, bounds=(lower, upper), max_nfev=150)
                self.assertLess(np.linalg.norm(result.fun), .0025, (dx, dy, phase))
                q = result.x


if __name__ == "__main__":
    unittest.main()
