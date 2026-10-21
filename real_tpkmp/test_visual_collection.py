"""Visual collection lifecycle regressions; all hardware/inference is replaced."""
from .bootstrap import read_json, write_json, digest
from .data import config, frame_document, physical_signature, read_record
from .kinematics import matrix_pose, pose_matrix
from .visual_collection import load_settings, VisualCollection
from .vision.common import ROOT, object_config, read_capture
from .vision.estimate import process_poses
from .vision.bindings import grasp_target
from .vision.smoke import fixture
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch, MagicMock
import io
import json
import sys
import tempfile
import time
import unittest
import numpy as np


class VisualCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.root, self.cfg = Path(self.temp.name), config()
        self.capture, self.color_object, self.base_object = fixture(self.root/'capture', synthetic=False)
        _, metadata, arrays = read_capture(self.capture)
        obj = object_config()
        masks = arrays['depth_m'] > 0
        self.obs = process_poses(metadata, arrays, masks, np.repeat(self.color_object[None], 3, axis=0),
                                 obj, self.capture/'hand_eye.json')
        # Include a nonidentity grasp rotation to exercise the full SE(3) composition.
        transform = np.eye(4)
        transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        transform[:3, 3] = [.075, .049, .15]
        grasp = dict(schema_version=1, kind='object_grasp', synthetic=False, usable_for_robot=True,
                     signature=physical_signature(self.cfg), object_id=obj['object_id'], mesh_sha256=digest(obj['mesh_path']),
                     object_frame='original_mesh_scaled_to_m', hand_eye_sha256=digest(self.capture/'hand_eye.json'),
                     execution_blocked_reasons=[], T_object_tcp=transform.tolist())
        self.grasp = self.root/'grasp.json'
        write_json(self.grasp, grasp)
        self.settings_path = self.root/'settings.json'
        write_json(self.settings_path, dict(object=str(ROOT/'objects/keel.json'),
                   hand_eye=str(self.capture/'hand_eye.json'), grasp=str(self.grasp)))
        self.settings = load_settings(self.settings_path, self.cfg)
        self.root_patch = patch('real_tpkmp.visual_collection.ROOT', self.root/'visual_runs')
        self.root_patch.start()
        self.output = self.root/'dataset'
        self.output.mkdir()
        self.visual = VisualCollection(self.settings, self.cfg, self.output)
        self.state = dict(tcp=matrix_pose(self.base_object @ transform), joints_rad=np.zeros(7),
                          gripper_mode='width', gripper_measured=.08, gripper_command=.08,
                          gripper_force_N=4., gripper_feedback_force_N=0.)
        self.reader = MagicMock()
        self.reader.sample.side_effect = lambda *args: dict(self.state, host_monotonic_s=time.monotonic())

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def estimate(self, args):
        write_json(Path(args.out)/'observation.json', self.obs)

    def localize(self, accepted=True):
        with (patch('real_tpkmp.vision.capture.capture', return_value=self.capture) as capture,
              patch('real_tpkmp.vision.estimate.estimate', side_effect=self.estimate),
              patch('real_tpkmp.visual_collection.preview_target', return_value=accepted)):
            result = self.visual.localize(self.reader)
        self.assertIs(capture.call_args.kwargs['reader'], self.reader)
        self.assertFalse(capture.call_args.args[0].no_robot)
        return result

    def place(self):
        self.visual.set_place(self.state['tcp'], dict(type='live_measured'))

    def test_settings_derive_camera_and_reject_mismatches_before_hardware(self):
        self.assertEqual(self.settings['serial'], 'synthetic_d435i')
        bad = read_json(self.settings_path)
        bad['serial'] = 'another_camera'
        path = self.root/'bad_settings.json'
        write_json(path, bad)
        with (patch('real_tpkmp.collect.Reader', side_effect=AssertionError('hardware opened')),
              patch('sys.stderr', new_callable=io.StringIO)):
            from .collect import main
            with patch.object(sys, 'argv', ['collect', '--vision-config', str(path), '--out', str(self.root/'bad')]):
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)
        bad.pop('serial')
        bad['frames'] = 2
        path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, '至少采集 3 帧'):
            load_settings(path, self.cfg)

    def test_full_grasp_transform_fixed_place_and_one_observation_per_start(self):
        self.place()
        self.assertTrue(self.localize())
        expected = self.base_object @ np.array(read_json(self.grasp)['T_object_tcp'])
        np.testing.assert_allclose(pose_matrix(self.visual.target['pick']), expected, atol=1e-8)
        frames, document, path, state = self.visual.start(self.reader)
        np.testing.assert_allclose(frames[0], state['tcp'])
        np.testing.assert_allclose(frames[2], self.state['tcp'])
        self.assertEqual(document['vision'], read_json(path)['vision'])
        self.assertTrue(document['place_reference']['sha256'])
        with self.assertRaisesRegex(ValueError, '先按 V'):
            self.visual.start(self.reader)
        self.assertIsNotNone(self.visual.place)
        self.assertTrue(self.localize())
        self.visual.start(self.reader)

    def test_cancel_and_failed_retry_remove_previous_target(self):
        self.place()
        self.localize()
        self.assertFalse(self.localize(accepted=False))
        self.assertIsNone(self.visual.target)
        with self.assertRaisesRegex(ValueError, '先按 V'):
            self.visual.start(self.reader)
        self.localize()
        with (patch('real_tpkmp.vision.capture.capture', side_effect=RuntimeError('camera busy')),
              self.assertRaisesRegex(RuntimeError, 'camera busy')):
            self.visual.localize(self.reader)
        self.assertIsNone(self.visual.target)

    def test_expiry_or_changed_source_prevents_start(self):
        self.place()
        self.localize()
        with patch('real_tpkmp.vision.bindings.time.time', return_value=self.obs['observed_unix_s']+1000):
            with self.assertRaisesRegex(ValueError, 'observation_expired'):
                self.visual.start(self.reader)
        self.reader.sample.assert_not_called()
        self.localize()
        self.grasp.write_text(self.grasp.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, '来源文件已改变'):
            self.visual.start(self.reader)
        self.reader.sample.assert_not_called()

    def test_blocked_geometry_cannot_be_accepted(self):
        self.obs.update(usable_for_robot=False, execution_blocked_reasons=['rgbd_geometry_quality_failed'])
        with (patch('real_tpkmp.vision.capture.capture', return_value=self.capture),
              patch('real_tpkmp.vision.estimate.estimate', side_effect=self.estimate),
              patch('real_tpkmp.visual_collection.preview_target') as preview):
            with self.assertRaisesRegex(ValueError, 'rgbd_geometry_quality_failed'):
                self.visual.localize(self.reader)
        preview.assert_not_called()
        self.assertIsNone(self.visual.target)

    def test_place_template_real_only_and_can_be_updated(self):
        initial = np.tile(self.state['tcp'], (3, 1))
        template = self.root/'template.json'
        write_json(template, frame_document(initial, self.cfg))
        other = self.root/'other'
        other.mkdir()
        collection = VisualCollection(self.settings, self.cfg, other, template)
        np.testing.assert_allclose(collection.place, initial[2])
        changed = initial[2].copy()
        changed[0] += .005
        collection.set_place(changed, dict(type='live_measured'))
        np.testing.assert_allclose(collection.place, changed)
        synthetic = self.root/'synthetic.json'
        write_json(synthetic, frame_document(initial, self.cfg, synthetic=True))
        another = self.root/'another'
        another.mkdir()
        with self.assertRaisesRegex(ValueError, '真实记录'):
            VisualCollection(self.settings, self.cfg, another, synthetic)

    def test_capture_reuses_passive_reader_and_keeps_caller_ownership(self):
        from argparse import Namespace
        from .vision.capture import capture
        _, metadata, arrays = read_capture(self.capture)
        camera = MagicMock()
        camera.model = camera.source_model = metadata['camera']
        camera.K, camera.T_color_depth = arrays['K'], np.eye(4)
        camera.depth_scale_m, camera.device = .001, {}
        camera.intrinsics_sha256 = None
        camera.frame.side_effect = [(rgb, depth, dict(meta)) for rgb, depth, meta in
                                   zip(arrays['rgb'], arrays['depth_m'], metadata['frames'])]
        args = Namespace(config=str(self.visual.robot_config), can=self.cfg['channel'], no_robot=False,
                         frames=3, serial=metadata['camera']['serial'], width=320, height=240, fps=30,
                         intrinsics=None, out=str(self.root/'reused_capture'))
        with (patch('real_tpkmp.hardware.Reader', side_effect=AssertionError('second reader opened')),
              patch('real_tpkmp.vision.capture.RGBDCamera', return_value=camera)):
            path = capture(args, reader=self.reader)
        self.reader.close.assert_not_called()
        camera.close.assert_called_once()
        _, saved, _ = read_capture(path)
        self.assertEqual([f['robot']['joint_source'] for f in saved['frames']], ['measured']*3)
        camera.reset_mock()
        camera.frame.side_effect = ValueError('camera disconnected')
        args.out = str(self.root/'failed_capture')
        with (patch('real_tpkmp.vision.capture.RGBDCamera', return_value=camera),
              self.assertRaisesRegex(ValueError, 'camera disconnected')):
            capture(args, reader=self.reader)
        self.reader.close.assert_not_called()
        camera.close.assert_called_once()

    def test_preview_projects_grasp_tcp_and_supports_accept_cancel(self):
        import cv2
        from .visual_collection import preview_target
        from .vision.coordinates import base_color_transforms
        _, metadata, arrays = read_capture(self.capture)
        path = self.root/'preview_pose/observation.json'
        write_json(path, self.obs)
        cv2.imwrite(str(path.parent/'overlay.png'), arrays['rgb'][0])
        target = grasp_target(path, self.grasp, self.cfg)
        base_color, _ = base_color_transforms(self.capture/'hand_eye.json', metadata)
        point = (np.linalg.inv(base_color[0]) @ pose_matrix(target['pick']))[:3, 3] @ arrays['K'].T
        expected = tuple(np.rint(point[:2]/point[2]).astype(int))
        with (patch.dict('os.environ', {'DISPLAY': ':test'}), patch.object(cv2, 'namedWindow'),
              patch.object(cv2, 'imshow'), patch.object(cv2, 'resizeWindow'),
              patch.object(cv2, 'getWindowProperty', return_value=1), patch.object(cv2, 'destroyAllWindows'),
              patch.object(cv2, 'waitKey', return_value=13) as key,
              patch.object(cv2, 'circle', wraps=cv2.circle) as circle):
            filename = self.root/'pick_preview.png'
            self.assertTrue(preview_target(self.capture, path, target, filename))
            self.assertEqual(circle.call_args.args[1], expected)
            self.assertIsNotNone(cv2.imread(str(filename)))
            key.return_value = 27
            self.assertFalse(preview_target(self.capture, path, target, filename))

    def run_loop(self, sequence, extra, output):
        from .collect import main
        @contextmanager
        def fake_keys():
            iterator = iter(sequence)
            yield lambda: next(iterator)  # Exhaustion fails instead of hanging.
        with (patch.object(sys, 'argv', ['collect', '--out', str(output), '--demos', '2', *extra]),
              patch('real_tpkmp.collect.Reader', return_value=self.reader),
              patch('real_tpkmp.collect.keys', fake_keys), patch('real_tpkmp.collect.flush_keys'),
              patch('real_tpkmp.collect.time.sleep'), patch('real_tpkmp.collect.align'),
              patch('real_tpkmp.vision.capture.capture', return_value=self.capture) as capture,
              patch('real_tpkmp.vision.estimate.estimate', side_effect=self.estimate),
              patch('real_tpkmp.visual_collection.preview_target', return_value=True),
              patch('sys.stdout', new_callable=io.StringIO)):
            main()
        self.reader.close.assert_called_once()
        self.reader.attach_writer.assert_not_called()
        return capture.call_count

    def test_collection_loop_discard_relocalize_and_preserve_place_provenance(self):
        output = self.root/'loop'
        sequence = ['r', 'l', 'v', 'r', 'v', 'x', 'r', 'v', 'r'] + ['n']*5 + ['s', 'r', 'v', 'r'] + ['n']*5 + ['s']
        calls = self.run_loop(sequence, ['--vision-config', str(self.settings_path)], output)
        self.assertEqual(calls, 3)
        first, data = read_record(output/'demo_001.npz')
        second, _ = read_record(output/'demo_002.npz')
        self.assertEqual(set(data['stage']), set(range(6)))
        a, b = first['task_frame_source'], second['task_frame_source']
        self.assertNotEqual(a['vision']['observation_path'], b['vision']['observation_path'])
        self.assertEqual(a['place_reference'], b['place_reference'])
        self.assertEqual(digest(a['path']), a['sha256'])
        self.assertEqual(read_json(output/'demo_001_frames.json')['vision'], a['vision'])
        self.assertEqual(read_json(output/'session.json')['collection_mode'], 'visual')

    def test_collection_uses_default_camera_config_without_path_argument(self):
        output = self.root/'default_config_loop'
        sequence = ['l'] + (['v', 'r'] + ['n']*5 + ['s'])*2
        with patch('real_tpkmp.collect.DEFAULT_CONFIG', self.settings_path):
            self.assertEqual(self.run_loop(sequence, [], output), 2)
        self.assertEqual(read_json(output/'session.json')['collection_mode'], 'visual')

    def inline_parameters(self):
        calibration = read_json(self.capture/'hand_eye.json')
        camera = calibration['signature']['camera']
        return dict(**{key: camera[key] for key in ('serial', 'width', 'height', 'fps')},
                    intrinsics=camera, hand_eye=calibration, grasp=read_json(self.grasp),
                    object_id='keel', mask_method='hsv')

    def test_inline_collection_without_intrinsics_hand_eye_or_grasp_paths(self):
        from .vision.settings import read_settings, materialize_settings
        inline_path = self.root/'inline_camera.json'
        raw = self.inline_parameters()
        raw['intrinsics'] = {key: raw['intrinsics'][key] for key in ('K', 'distortion_model', 'distortion_coefficients')}
        matrix = raw['hand_eye']['T_flange_color']['matrix']
        raw['hand_eye'] = dict(mode='eye-in-hand', matrix=matrix, passed=True)
        write_json(inline_path, raw)
        output = self.root/'inline_loop'
        sequence = ['l'] + (['v', 'r'] + ['n']*5 + ['s'])*2
        with patch('real_tpkmp.vision.settings.SNAPSHOT_ROOT', self.root/'snapshots'):
            calibration_path = materialize_settings(read_settings(inline_path), names=('hand_eye',))['hand_eye']
            raw['grasp']['hand_eye_sha256'] = digest(calibration_path)
            inline_path.write_text(json.dumps(raw))
            self.obs['hand_eye'].update(path=calibration_path, sha256=digest(calibration_path))
            self.assertEqual(self.run_loop(sequence, ['--camera-config', str(inline_path)], output), 2)
        meta, _ = read_record(output/'demo_001.npz')
        self.assertEqual(meta['task_frame_source']['vision']['object_id'], 'keel')
        normalized = read_json(output/'vision_config.json')
        self.assertEqual(read_json(normalized['intrinsics'])['K'], self.obs['camera']['K'])
        self.assertTrue(read_json(normalized['hand_eye'])['passed'])

    def test_grasp_can_be_written_as_parameters_and_wrong_calibration_is_rejected(self):
        from .vision.settings import update_grasp_parameters
        inline_path = self.root/'writable_camera.json'
        initial = self.inline_parameters()
        initial['grasp'] = None
        write_json(inline_path, initial)
        grasp = read_json(self.grasp)
        with patch('real_tpkmp.vision.settings.SNAPSHOT_ROOT', self.root/'snapshots'):
            from .vision.cli import main
            with (patch.object(sys, 'argv', ['vision', 'bind-grasp', '--observation', 'unused', '--out', 'unused',
                                             '--camera-config', str(inline_path), '--update-config']),
                  patch('real_tpkmp.vision.bindings.bind', return_value=grasp)):
                main()
            updated = read_json(inline_path)
            self.assertEqual(updated['grasp']['T_object_tcp'], grasp['T_object_tcp'])
            self.assertNotIn('source', updated['grasp'])
            self.assertEqual(load_settings(inline_path, self.cfg)['serial'], initial['serial'])
            saved_hash = digest(inline_path)
            grasp['hand_eye_sha256'] = 'different_calibration'
            with self.assertRaisesRegex(ValueError, '手眼参数.*不一致'):
                update_grasp_parameters(inline_path, grasp)
        self.assertEqual(digest(inline_path), saved_hash)

    def test_incomplete_inline_hand_eye_matrix_is_rejected(self):
        inline_path = self.root/'incomplete_matrix.json'
        initial = self.inline_parameters()
        initial['hand_eye']['T_flange_color']['matrix'][0][0] = None
        write_json(inline_path, initial)
        with (patch('real_tpkmp.vision.settings.SNAPSHOT_ROOT', self.root/'snapshots'),
              self.assertRaisesRegex(ValueError, '补齐.*hand_eye')):
            load_settings(inline_path, self.cfg)

    def test_manual_collection_still_requires_new_references_each_demo(self):
        output = self.root/'manual'
        sequence = (['p', 'l', 'r'] + ['n']*5 + ['s'])*2
        self.assertEqual(self.run_loop(sequence, ['--manual'], output), 0)
        meta, data = read_record(output/'demo_002.npz')
        self.assertNotIn('task_frame_source', meta)
        self.assertEqual(data['frames'].shape, (3, 7))
        self.assertEqual(read_json(output/'session.json')['collection_mode'], 'manual')

    def test_file_reference_mode_remains_compatible(self):
        template = self.root/'manual_frames.json'
        write_json(template, frame_document(np.tile(self.state['tcp'], (3, 1)), self.cfg))
        output = self.root/'file_mode'
        sequence = (['r'] + ['n']*5 + ['s'])*2
        self.assertEqual(self.run_loop(sequence, ['--task-frames', str(template)], output), 0)
        meta, _ = read_record(output/'demo_002.npz')
        self.assertEqual(meta['task_frame_source']['sha256'], digest(template))


if __name__ == '__main__':
    unittest.main(verbosity=2)
