"""Meaningful geometry/provenance tests without camera, CAN or neural models."""
from .common import ROOT, object_config, checked, read_capture, read_json, write_json
from .geometry import load_mesh, quality
from .capture import project_depth_points
from .coordinates import base_color_transforms
from .estimate import process_poses
from .segmentation import hsv_masks
from .bindings import build_frames, frame_reasons
from .smoke import fixture
from .settings_test import SettingsTests
from ..data import config, frame_document
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np


class VisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.path, self.pose, self.base_pose = fixture(Path(self.temp.name)/'capture', synthetic=False)
        _, self.meta, self.data = read_capture(self.path)
        self.obj = object_config()
        self.mask = self.data['depth_m'][0] > 0

    def tearDown(self):
        self.temp.cleanup()

    def test_mesh_units_and_original_origin(self):
        triangles = load_mesh(self.obj)
        np.testing.assert_allclose(np.ptp(triangles.reshape(-1, 3), axis=0), [.15, .098, .023], atol=1e-6)
        bad = dict(self.obj, mesh_units='m')
        with self.assertRaises(ValueError):
            load_mesh(bad)

    def test_rigid_transform_rejects_reflection_scale_nan(self):
        for value in (-1., 2., np.nan):
            matrix = np.eye(4)
            matrix[0, 0] = value
            with self.assertRaises(ValueError):
                checked(matrix)

    def test_depth_registration_zbuffer_uses_optical_z(self):
        points = np.array([[0, 0, 1.], [0, 0, .5], [10, 0, 1], [0, 0, 0]])
        depth = project_depth_points(points, np.array([[10, 0, 3], [0, 10, 3], [0, 0, 1]]), (7, 7))
        self.assertAlmostEqual(depth[3, 3], .5)

    def test_geometry_passes_true_pose_rejects_shift_and_missing_depth(self):
        triangles = load_mesh(self.obj)
        report = quality(triangles, self.pose, self.data['depth_m'][0], self.mask, self.data['K'], self.obj)
        self.assertTrue(report['passed'], report)
        shifted = self.pose.copy()
        shifted[2, 3] += .04
        self.assertFalse(quality(triangles, shifted, self.data['depth_m'][0], self.mask, self.data['K'], self.obj)['passed'])
        self.assertFalse(quality(triangles, self.pose, np.zeros_like(self.data['depth_m'][0]), self.mask, self.data['K'], self.obj)['passed'])

    def test_automatic_mask_and_ambiguity_rejection(self):
        masks, _ = hsv_masks(self.data['rgb'], self.obj)
        self.assertGreater(np.count_nonzero(masks[0] & self.mask)/np.count_nonzero(self.mask), .9)
        rgb = np.full((1, 100, 160, 3), 100, np.uint8)
        rgb[0, 10:40, 10:40] = [60, 150, 210]
        rgb[0, 50:80, 100:130] = [60, 150, 210]
        with self.assertRaises(ValueError):
            hsv_masks(rgb, self.obj)

    def test_eye_in_hand_and_eye_to_hand_equations(self):
        transforms, report = base_color_transforms(self.path/'hand_eye.json', self.meta)
        self.assertTrue(report['passed'])
        np.testing.assert_allclose(transforms[0]@self.pose, self.base_pose, atol=1e-8)
        result = read_json(self.path/'hand_eye.json')
        result.update(mode='eye-to-hand', transform_name='T_base_color', T_base_color=dict(matrix=transforms[0].tolist()))
        outside = self.path/'outside.json'
        write_json(outside, result)
        transforms2, report = base_color_transforms(outside, self.meta)
        np.testing.assert_allclose(transforms2, transforms, atol=1e-8)
        bad = copy.deepcopy(self.meta)
        bad['frames'][0]['robot']['joint_source'] = 'leader'
        with self.assertRaises(ValueError):
            base_color_transforms(self.path/'hand_eye.json', bad)

    def test_intrinsics_mismatch_and_failed_calibration(self):
        bad = copy.deepcopy(self.meta)
        bad['source_camera']['K'][0][0] += 10
        with self.assertRaises(ValueError):
            base_color_transforms(self.path/'hand_eye.json', bad)
        result = read_json(self.path/'hand_eye.json')
        result['passed'] = False
        filename = self.path/'failed.json'
        write_json(filename, result)
        _, report = base_color_transforms(filename, self.meta)
        self.assertFalse(report['passed'])

    def test_independent_pose_stability_and_diagnostic_gate(self):
        masks = np.repeat(self.mask[None], 3, axis=0)
        poses = np.repeat(self.pose[None], 3, axis=0)
        good = process_poses(self.meta, self.data, masks, poses, self.obj, self.path/'hand_eye.json')
        self.assertTrue(good['usable_for_robot'], good['execution_blocked_reasons'])
        diagnostic = process_poses(self.meta, self.data, masks, poses, self.obj, self.path/'hand_eye.json', backend='diagnostic_import')
        self.assertFalse(diagnostic['usable_for_robot'])
        poses[2, 0, 3] += .02
        bad = process_poses(self.meta, self.data, masks, poses, self.obj, self.path/'hand_eye.json')
        self.assertIn('multi_frame_pose_unstable', bad['execution_blocked_reasons'])

    def test_visual_frames_compose_grasp_and_keep_place(self):
        from ..kinematics import matrix_pose
        from ..bootstrap import digest
        masks = np.repeat(self.mask[None], 3, axis=0)
        obs = process_poses(self.meta, self.data, masks, np.repeat(self.pose[None], 3, axis=0), self.obj, self.path/'hand_eye.json')
        obs_path = self.path/'observation.json'
        write_json(obs_path, obs)
        grasp = dict(schema_version=1, kind='object_grasp', signature={k: self.meta['config'][k] for k in ('arm_id', 'firmware', 'base_frame', 'tcp_frame', 'T_flange_tcp_pose_wxyz')},
                     object_id=obs['object_id'], mesh_sha256=obs['mesh_sha256'], object_frame=obs['object_frame'],
                     hand_eye_sha256=obs['hand_eye']['sha256'], synthetic=False, usable_for_robot=True,
                     execution_blocked_reasons=[], T_object_tcp=np.eye(4).tolist())
        grasp_path = self.path/'grasp.json'
        write_json(grasp_path, grasp)
        frame_path = self.path/'template.json'
        initial = np.array([[0, 0, .4, 1, 0, 0, 0], [0, 0, .3, 1, 0, 0, 0], [.1, .1, .3, 1, 0, 0, 0]])
        write_json(frame_path, frame_document(initial, config()))
        result = build_frames(obs_path, grasp_path, frame_path, config())
        np.testing.assert_allclose(result['frames']['pick'], matrix_pose(self.base_pose), atol=1e-8)
        np.testing.assert_allclose(result['frames']['place'], initial[2])
        self.assertEqual(frame_reasons(result), [])
        with patch('real_tpkmp.vision.bindings.time.time', return_value=obs['observed_unix_s']+1000):
            self.assertIn('observation_expired', frame_reasons(result))
        result['vision']['grasp_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            frame_reasons(result)


def run():
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(case) for case in (VisionTests, SettingsTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
