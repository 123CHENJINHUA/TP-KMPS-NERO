import copy
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from sim_bridge.wrist_cameras import camera_definitions, load_camera_config, validate_camera_config


class WristCameraConfigTests(unittest.TestCase):
    def test_default_optical_axis_points_toward_finger_contact_region(self):
        config = load_camera_config()
        for side in ('left', 'right'):
            mount = config[side]
            ray = Rotation.from_euler('xyz', mount['rotation_xyz_deg'], degrees=True).apply([0, 0, -1])
            wanted = np.array([0., 0., .125]) - np.asarray(mount['translation_m'])
            wanted /= np.linalg.norm(wanted)
            self.assertGreater(float(np.dot(ray, wanted)), .99)

    def test_bad_pose_and_inverted_clipping_range_are_rejected(self):
        config = load_camera_config()
        for section, key, bad in [('left', 'translation_m', [0, float('nan'), 0]),
                                  ('right', 'rotation_xyz_deg', [180, 0])]:
            invalid = copy.deepcopy(config)
            invalid[section][key] = bad
            with self.assertRaises(ValueError):
                validate_camera_config(invalid)
        invalid = copy.deepcopy(config)
        invalid['clipping_range_m'] = [10, .01]
        with self.assertRaises(ValueError):
            validate_camera_config(invalid)

    def test_extra_camera_paths_and_default_cameras_are_preserved(self):
        config = load_camera_config('assets/cameras/wrist_cameras_extra_example.json')
        definitions = camera_definitions(config, ['/World/nero', '/World/nero2'])
        self.assertEqual([item['path'] for item in definitions], [
            '/World/nero/gripper_base/WristCamera',
            '/World/nero2/gripper_base/WristCamera',
            '/World/nero/gripper_base/CameraSide',
            '/World/UserScene/Environment/CameraTop'])
        self.assertEqual([item['arm'] for item in definitions], [1, 2, 1, None])

    def test_extra_camera_can_override_lens_without_changing_other_cameras(self):
        config = load_camera_config('assets/cameras/wrist_cameras_extra_example.json')
        config['extra_cameras'][0].update(resolution=[320, 240], focal_length_mm=12., show_view=False)
        definitions = camera_definitions(config, ['/World/nero', '/World/nero2'])
        self.assertEqual(definitions[2]['resolution'], [320, 240])
        self.assertEqual(definitions[2]['focal_length_mm'], 12.)
        self.assertFalse(definitions[2]['show_view'])
        for index in (0, 1, 3):
            self.assertEqual(definitions[index]['resolution'], [640, 480])
            self.assertEqual(definitions[index]['focal_length_mm'], 18.)

    def test_invalid_names_parents_and_duplicate_paths_are_rejected(self):
        config = load_camera_config('assets/cameras/wrist_cameras_extra_example.json')
        for field, value in [('name', '../Camera'), ('parent', '/World'),
                             ('name', 'WristCamera'), ('show_view', 'false')]:
            invalid = copy.deepcopy(config)
            invalid['extra_cameras'][0][field] = value
            with self.assertRaises(ValueError):
                validate_camera_config(invalid)
        invalid = copy.deepcopy(config)
        invalid['extra_cameras'].append(copy.deepcopy(invalid['extra_cameras'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate camera path'):
            validate_camera_config(invalid)

    def test_visualization_options_inherit_and_override_per_camera(self):
        config = load_camera_config('assets/cameras/wrist_cameras_extra_example.json')
        config['visualization']['show_axes'] = False
        config['extra_cameras'][0]['visualization'] = {'show_axes': True, 'axis_length_m': .12}
        definitions = camera_definitions(config, ['/World/nero', '/World/nero2'])
        self.assertTrue(definitions[2]['visualization']['show_axes'])
        self.assertEqual(definitions[2]['visualization']['axis_length_m'], .12)
        for index in (0, 1, 3):
            self.assertFalse(definitions[index]['visualization']['show_axes'])
            self.assertEqual(definitions[index]['visualization']['axis_length_m'], .08)
        self.assertTrue(all(item['visualization']['show_body'] for item in definitions))

    def test_bad_visualization_options_are_rejected(self):
        config = load_camera_config()
        for bad in [{'show_axes': 'false'}, {'axis_length_m': float('nan')},
                    {'axis_length_m': 0}, {'show_body': 1}, []]:
            invalid = copy.deepcopy(config)
            invalid['left']['visualization'] = bad
            with self.assertRaises(ValueError):
                validate_camera_config(invalid)

    def test_configuration_cannot_be_loaded_outside_project(self):
        with self.assertRaisesRegex(ValueError, 'inside this project'):
            load_camera_config('/tmp/wrist_cameras.json')


if __name__ == '__main__':
    unittest.main()
