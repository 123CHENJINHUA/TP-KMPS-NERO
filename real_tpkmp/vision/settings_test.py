"""Shared JSON defaults and CLI overrides without opening camera/CAN/GPU."""
from ..bootstrap import PROJECT, write_json, read_json, digest
from .common import ROOT
from .settings import read_settings, materialize_settings
from pathlib import Path
from unittest.mock import patch
import io
import copy
import json
import os
import sys
import tempfile
import unittest


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.root = Path(self.temp.name)
        self.path = self.root/'camera.json'
        self.values = dict(serial='test_camera', width=848, height=480, fps=15,
                           intrinsics='hand_eye_calibration/test_intrinsics.json',
                           hand_eye='hand_eye_calibration/test_result.json',
                           grasp='real_tpkmp/vision/test_grasp.json', mask_method='hsv')
        write_json(self.path, self.values)
        self.capture_dir = self.root/'capture'
        write_json(self.capture_dir/'capture.json', dict(config=read_json(PROJECT/'real_tpkmp/config.json')))
        self.snapshots = patch('real_tpkmp.vision.settings.SNAPSHOT_ROOT', self.root/'snapshots')
        self.snapshots.start()

    def tearDown(self):
        self.snapshots.stop()
        self.temp.cleanup()

    def intrinsics(self):
        return dict(serial='test_camera', width=848, height=480, fps=15, frame='color_optical',
                    K=[[500., 0, 424.], [0, 501., 240.], [0, 0, 1]],
                    distortion_model='none', distortion_coefficients=[0.]*5)

    def dispatch(self, arguments, backend, *, default=False):
        from .cli import main
        arguments = list(arguments)
        if not default:
            arguments += ['--camera-config', str(self.path)]
        with (patch.object(sys, 'argv', ['vision', *arguments]), patch(backend) as run,
              patch('real_tpkmp.vision.cli.DEFAULT_CONFIG', self.path),
              patch('sys.stdout', new_callable=io.StringIO)):
            main()
        run.assert_called_once()
        return run.call_args.args[0]

    def test_paths_use_project_root_and_null_calibration_is_allowed(self):
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            settings = read_settings(self.path)
        finally:
            os.chdir(previous)
        self.assertEqual(settings['intrinsics'], str(PROJECT/self.values['intrinsics']))
        self.values.update(intrinsics=None, hand_eye=None, grasp=None)
        self.path.write_text(json.dumps(self.values))
        settings = read_settings(self.path)
        self.assertIsNone(settings['hand_eye'])
        self.assertIsNone(settings['intrinsics'])

    def test_capture_reads_default_file_at_startup(self):
        args = self.dispatch(['capture', '--out', 'unused'], 'real_tpkmp.vision.capture.capture', default=True)
        self.assertEqual((args.serial, args.width, args.height, args.fps), ('test_camera', 848, 480, 15))
        self.assertEqual(args.intrinsics, str(PROJECT/self.values['intrinsics']))
        self.assertEqual(args.frames, 3)

    def test_explicit_camera_options_override_json_and_factory_intrinsics(self):
        args = self.dispatch(['capture', '--out', 'unused', '--serial', 'override', '--width', '640',
                              '--fps', '30', '--intrinsics', 'custom.json'], 'real_tpkmp.vision.capture.capture')
        self.assertEqual((args.serial, args.width, args.fps), ('override', 640, 30))
        self.assertEqual(args.intrinsics, 'custom.json')
        args = self.dispatch(['track', '--out', 'unused', '--factory-intrinsics'], 'real_tpkmp.vision.tracking.track')
        self.assertEqual(args.serial, 'test_camera')
        self.assertIsNone(args.intrinsics)
        self.assertEqual(args.mask_method, 'hsv')

    def test_estimate_uses_shared_hand_eye_and_supports_camera_only(self):
        args = self.dispatch(['estimate', '--capture', str(self.capture_dir), '--out', 'unused'], 'real_tpkmp.vision.estimate.estimate')
        self.assertEqual(args.hand_eye, str(PROJECT/self.values['hand_eye']))
        self.assertEqual(args.iterations, 5)
        args = self.dispatch(['estimate', '--capture', 'unused', '--out', 'unused', '--camera-only'],
                             'real_tpkmp.vision.estimate.estimate')
        self.assertIsNone(args.hand_eye)
        args = self.dispatch(['estimate', '--capture', 'unused', '--out', 'unused', '--hand-eye', 'another.json'],
                             'real_tpkmp.vision.estimate.estimate')
        self.assertEqual(args.hand_eye, 'another.json')

    def test_frames_uses_shared_grasp_binding(self):
        args = self.dispatch(['frames', '--observation', 'unused', '--template-frames', 'unused', '--out', 'unused'],
                             'real_tpkmp.vision.bindings.export_frames')
        self.assertEqual(args.grasp, str(PROJECT/self.values['grasp']))

    def test_replay_does_not_load_live_camera_config(self):
        self.path.write_text('invalid JSON intentionally')
        with patch('real_tpkmp.vision.settings.read_settings', side_effect=AssertionError('live config read')):
            args = self.dispatch(['track', '--replay', 'recorded', '--out', 'unused', '--mask-method', 'hsv'],
                                 'real_tpkmp.vision.tracking.track')
        self.assertEqual(args.replay, 'recorded')
        self.assertIsNone(args.intrinsics)

    def test_invalid_config_fails_before_camera_is_opened(self):
        for key, value in (('serial', 12345), ('width', 0), ('hand_eye', []), ('unknown_field', 1)):
            with self.subTest(key=key):
                self.path.write_text(json.dumps(dict(self.values, **{key: value})))
                with self.assertRaises(ValueError):
                    read_settings(self.path)
        with (patch('real_tpkmp.vision.capture.capture', side_effect=AssertionError('camera opened')),
              patch('sys.stderr', new_callable=io.StringIO)):
            from .cli import main
            with patch.object(sys, 'argv', ['vision', 'capture', '--camera-config', str(self.path), '--out', 'unused']):
                with self.assertRaises(SystemExit) as error:
                    main()
            self.assertEqual(error.exception.code, 2)

    def test_inline_intrinsics_are_used_without_parameter_paths(self):
        self.values.update(intrinsics=self.intrinsics(), hand_eye={'draft': True}, grasp=None, object_id='keel')
        self.path.write_text(json.dumps(self.values))
        args = self.dispatch(['track', '--out', 'unused'], 'real_tpkmp.vision.tracking.track')
        model = read_json(args.intrinsics)
        self.assertEqual(model['K'], self.intrinsics()['K'])
        self.assertEqual(model['source'], 'inline_parameters')
        self.assertEqual(read_json(args.object)['object_id'], 'keel')
        # Track never tries to turn the incomplete hand-eye block into a usable transform.
        self.assertFalse(hasattr(args, 'hand_eye'))

    def test_inline_intrinsics_mismatch_or_invalid_matrix_rejected_before_capture(self):
        for change in ({'serial': 'another_camera'}, {'K': [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]}):
            self.values['intrinsics'] = dict(self.intrinsics(), **change)
            self.path.write_text(json.dumps(self.values))
            with (patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit) as error):
                self.dispatch(['capture', '--out', 'unused'], 'real_tpkmp.vision.capture.capture')
            self.assertEqual(error.exception.code, 2)

    def test_inline_calibration_snapshot_is_stable_and_does_not_promote_passed_flag(self):
        calibration = dict(schema_version=1, mode='eye-in-hand', transform_name='T_flange_color',
                           T_flange_color=dict(matrix=[[1, 0, 0, .02], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                               translation_unit='m'), passed=False, synthetic=False,
                           signature=dict(camera=self.intrinsics(), robot=dict(arm_id='test_arm', firmware='v121',
                                          parent_frame='base', joint_source='measured')))
        first = materialize_settings(dict(hand_eye=calibration))['hand_eye']
        second = materialize_settings(dict(hand_eye=calibration))['hand_eye']
        self.assertEqual(first, second)
        self.assertFalse(read_json(first)['passed'])
        before = digest(first)
        altered = json.loads(json.dumps(calibration))
        altered['T_flange_color']['matrix'][0][3] += .01
        third = materialize_settings(dict(hand_eye=altered))['hand_eye']
        self.assertNotEqual(first, third)
        self.assertEqual(digest(first), before)
        Path(first).write_text('{}')
        with self.assertRaisesRegex(ValueError, '快照被修改'):
            materialize_settings(dict(hand_eye=calibration))

    def test_object_selector_rejects_path_traversal_and_unknown_models(self):
        for name in ('../keel', 'missing_test_model_123'):
            self.values['object_id'] = name
            self.path.write_text(json.dumps(self.values))
            with self.assertRaises(ValueError):
                read_settings(self.path)

    def compact_parameters(self):
        return dict(serial='test_camera', width=848, height=480, fps=15, object_id='keel', mask_method='hsv',
                    intrinsics={key: self.intrinsics()[key] for key in ('K', 'distortion_model', 'distortion_coefficients')},
                    hand_eye=dict(mode='eye-in-hand', matrix=[[1, 0, 0, .02], [0, 1, 0, 0],
                                                            [0, 0, 1, 0], [0, 0, 0, 1]], passed=True))

    def test_compact_parameters_expand_shared_camera_without_rewriting_user_file(self):
        raw = self.compact_parameters()
        raw['intrinsics']['passed'] = False
        self.path.write_text(json.dumps(raw))
        before = digest(self.path)
        args = self.dispatch(['estimate', '--capture', str(self.capture_dir), '--out', 'unused'], 'real_tpkmp.vision.estimate.estimate')
        calibration = read_json(args.hand_eye)
        self.assertEqual(calibration['signature']['camera'], self.intrinsics())
        self.assertEqual(calibration['signature']['robot']['joint_source'], 'measured')
        self.assertEqual(calibration['T_flange_color']['matrix'], raw['hand_eye']['matrix'])
        args = self.dispatch(['track', '--out', 'unused'], 'real_tpkmp.vision.tracking.track')
        self.assertEqual(read_json(args.intrinsics)['serial'], raw['serial'])
        self.assertFalse(read_json(args.intrinsics)['passed'])
        self.assertEqual(digest(self.path), before)

    def test_compact_calibration_rejects_reuse_after_shared_camera_changes(self):
        raw = self.compact_parameters()
        self.path.write_text(json.dumps(raw))
        first = materialize_settings(read_settings(self.path))['hand_eye']
        for name in ('serial', 'width', 'K'):
            changed = copy.deepcopy(raw)
            if name == 'K':
                changed['intrinsics']['K'][0][0] += 10
            else:
                changed[name] = 'another_camera' if name == 'serial' else 640
            self.path.write_text(json.dumps(changed))
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, '相机/内参.*变化'):
                materialize_settings(read_settings(self.path))
        changed['hand_eye']['matrix'][0][3] += .005
        self.path.write_text(json.dumps(changed))
        self.assertNotEqual(materialize_settings(read_settings(self.path))['hand_eye'], first)

    def test_compact_eye_to_hand_respects_robot_config_and_failed_validation(self):
        raw = self.compact_parameters()
        raw['hand_eye'].update(mode='eye-to-hand', passed=False)
        self.path.write_text(json.dumps(raw))
        robot = read_json(PROJECT/'real_tpkmp/config.json')
        robot['arm_id'] = 'another_follower'
        (self.capture_dir/'capture.json').write_text(json.dumps(dict(config=robot)))
        args = self.dispatch(['estimate', '--capture', str(self.capture_dir), '--out', 'unused'],
                             'real_tpkmp.vision.estimate.estimate')
        result = read_json(args.hand_eye)
        self.assertEqual(result['transform_name'], 'T_base_color')
        self.assertEqual(result['signature']['robot']['arm_id'], robot['arm_id'])
        self.assertFalse(result['passed'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
