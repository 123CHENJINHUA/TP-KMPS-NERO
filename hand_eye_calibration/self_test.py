"""Known-transform, vision, data integrity and receive-only decoder regression tests."""
from bootstrap import HERE, inside_folder, write_json, read_json
import argparse
import tempfile
import threading
import unittest
from unittest.mock import patch
from collections import Counter, deque
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from calibration import fit, evaluate, excitation, load_dataset, signature, passed
from transforms import from_pose6, inverse, distance, checked, unpack
from vision import load_board, BoardDetector
from realsense_camera import opencv_intrinsics, extrinsic_matrix, RealSenseCamera
from robot_pose import CanPoseReader, complete_feedback, require_stationary, load_sdk
from board_diagnostics import diagnose_layout
from motion_checks import check_capture, MotionCaptureError


CAMERA = {'serial': 'SYNTHETIC_NOT_A_CAMERA', 'width': 1280, 'height': 720, 'fps': 30,
          'frame': 'color_optical', 'K': [[920., 0, 640], [0, 915., 360], [0, 0, 1]],
          'distortion_model': 'opencv_brown', 'distortion_coefficients': [.01, -.02, .0005, -.0008, 0.]}


def synthetic(mode, count=36, noise_px=0., seed=3):
    rng = np.random.default_rng(seed)
    board, config = load_board(dictionary='DICT_4X4_50')
    X = from_pose6([.035, -.025, .065, .25, -.18, .42])
    F = from_pose6([.45, -.12, .25, .2, -.3, .7] if mode == 'eye-in-hand' else [.012, -.015, .12, -.2, .1, -.4])
    detector = BoardDetector(board, CAMERA)
    samples = []
    ids = np.arange(len(board.getChessboardCorners()))
    obj = board.getChessboardCorners().astype(np.float64)
    for i in range(count):
        q = from_pose6([rng.uniform(-.14, -.05), rng.uniform(-.18, -.08), rng.uniform(.5, .85),
                       rng.uniform(-.55, .55), rng.uniform(-.55, .55), rng.uniform(-.65, .65)])
        # Generate independent world/base poses from known fixture mounting, then
        # reproject geometry. The solver must recover X, not just fit a loop.
        p = F @ inverse(q) @ inverse(X) if mode == 'eye-in-hand' else X @ q @ inverse(F)
        uv = cv2.projectPoints(obj, cv2.Rodrigues(q[:3, :3])[0], q[:3, 3], np.array(CAMERA['K']), np.array(CAMERA['distortion_coefficients']))[0].reshape(-1, 2)
        if noise_px:
            uv += rng.normal(0, noise_px, uv.shape)
            q = np.array(detector.pose_from_corners(ids, uv)['T_color_board'])
            jitter = unpack(np.r_[rng.normal(0, .00015, 3), rng.normal(0, .00008, 3)])
            p = p @ jitter
        samples.append({'id': f'{i:04d}', 'image': f'images/{i:04d}.png',
                        'robot': {'T_base_flange': p.tolist(), 'source': 'synthetic'},
                        'T_color_board': q.tolist(), 'corner_ids': ids.tolist(), 'corners_px': uv.tolist(),
                        'reprojection_rms_px': float(noise_px)})
    metadata = {'schema_version': 1, 'synthetic': True, 'mode': mode, 'board': config, 'camera': CAMERA,
                'camera_raw': CAMERA, 'robot': {'arm_id': 'SYNTHETIC', 'pose_source': 'synthetic',
                'parent_frame': 'base', 'child_frame': 'flange'}}
    return metadata, samples, X, F


class CalibrationTests(unittest.TestCase):
    def reprocess_fixture(self, folder):
        metadata, samples, _, _ = synthetic('eye-in-hand', count=2)
        source = Path(folder) / 'source'
        write_json(source / 'metadata.json', metadata)
        (source / 'images').mkdir()
        for sample in samples:
            cv2.imwrite(str(source / sample['image']), np.zeros((720, 1280, 3), np.uint8))
            write_json(source / 'samples' / f'{sample["id"]}.json', sample)
        camera = dict(CAMERA, passed=True, source='test_independent_rgb_intrinsics')
        intrinsic = Path(folder) / 'intrinsics.json'
        write_json(intrinsic, camera)
        detector = SimpleNamespace()
        from unittest.mock import Mock
        detector.detect = Mock(side_effect=[{k: s[k] for k in (
            'corner_ids', 'corners_px', 'T_color_board', 'reprojection_rms_px')} for s in samples])
        detector.pose_from_corners = Mock(side_effect=[{k: s[k] for k in (
            'T_color_board', 'reprojection_rms_px')} for s in samples])
        return source, intrinsic, detector

    def test_reprocess_preserves_source_robot_poses_and_images(self):
        from reprocess import reprocess_dataset
        (HERE / '.tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=HERE / '.tmp') as temporary:
            source, intrinsic, detector = self.reprocess_fixture(temporary)
            metadata, original, digest = load_dataset(source)
            output = Path(temporary) / 'derived'
            with patch('reprocess.BoardDetector', return_value=detector):
                self.assertEqual(reprocess_dataset(source, intrinsic, output), 2)
            new_metadata, new_samples, _ = load_dataset(output)
            self.assertEqual(load_dataset(source)[2], digest)
            self.assertEqual(new_metadata['camera'], read_json(intrinsic))
            self.assertEqual(new_metadata['board'], metadata['board'])
            self.assertEqual(new_metadata['reprocessing']['source_dataset_sha256'], digest)
            for old, new in zip(original, new_samples):
                self.assertEqual(old['robot'], new['robot'])
                self.assertEqual(old['id'], new['id'])
                self.assertEqual(old['corner_ids'], new['corner_ids'])
                self.assertEqual((source / old['image']).read_bytes(), (output / new['image']).read_bytes())
            with self.assertRaisesRegex(ValueError, '已存在'):
                reprocess_dataset(source, intrinsic, output)

    def test_reprocess_rejects_intrinsic_camera_mismatch(self):
        from reprocess import reprocess_dataset
        (HERE / '.tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=HERE / '.tmp') as temporary:
            source, _, _ = self.reprocess_fixture(temporary)
            wrong = Path(temporary) / 'other_camera.json'
            write_json(wrong, dict(CAMERA, serial='OTHER_CAMERA', passed=True))
            output = Path(temporary) / 'derived'
            with self.assertRaisesRegex(ValueError, '必须与原始 RGB 图像一致'):
                reprocess_dataset(source, wrong, output)
            self.assertFalse(output.exists())

    def test_reprocess_aborts_changed_corner_set_without_partial_output(self):
        from reprocess import reprocess_dataset
        (HERE / '.tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=HERE / '.tmp') as temporary:
            source, intrinsic, detector = self.reprocess_fixture(temporary)
            _, samples, digest = load_dataset(source)
            detections = [{k: s[k] for k in ('corner_ids', 'corners_px', 'T_color_board',
                                            'reprojection_rms_px')} for s in samples]
            detections[-1]['corner_ids'] = detections[-1]['corner_ids'][:-1]
            detector.detect.side_effect = detections
            output = Path(temporary) / 'derived'
            with patch('reprocess.BoardDetector', return_value=detector), \
                    self.assertRaisesRegex(ValueError, '编号集合'):
                reprocess_dataset(source, intrinsic, output)
            self.assertFalse(output.exists())
            self.assertFalse(list(Path(temporary).glob('.reprocess_*')))
            self.assertEqual(load_dataset(source)[2], digest)

    def test_large_5x5_board_geometry_and_detection(self):
        board, config = load_board(HERE / 'board_configs' / 'charuco_rows9_cols14_5x5_20mm.json')
        self.assertEqual((config['squares_x'], config['squares_y']), (14, 9))
        np.testing.assert_array_equal(board.getIds().ravel(), np.arange(63))
        self.assertEqual(len(board.getChessboardCorners()), 104)
        np.testing.assert_allclose(board.getChessboardCorners().max(axis=0), [.26, .16, 0.], atol=1e-7)
        image = board.generateImage((1008, 648), marginSize=0, borderBits=1)
        image = cv2.copyMakeBorder(image, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
        detection = BoardDetector(board, min_corners=40).detect(image)
        self.assertEqual(len(detection['corner_ids']), 104)
        self.assertEqual(set(detection['marker_ids']), set(range(63)))
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        for name in ('DICT_5X5_250', 'DICT_5X5_1000'):
            other = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
            np.testing.assert_array_equal(dictionary.bytesList[:63], other.bytesList[:63])

    def test_dictionary_capacity_rejects_large_board_before_rendering(self):
        with self.assertRaisesRegex(ValueError, '63.*DICT_5X5_50.*0–49'):
            load_board(HERE / 'board_configs' / 'charuco_rows9_cols14_5x5_20mm.json',
                       dictionary='DICT_5X5_50')

    def test_large_board_intrinsic_calibration_recovers_known_camera(self):
        import calibrate_intrinsics
        import io
        rng = np.random.default_rng(85)
        board, config = load_board(HERE / 'board_configs' / 'charuco_rows9_cols14_5x5_20mm.json')
        objects = board.getChessboardCorners().astype(float)
        metadata = {'schema_version': 1, 'synthetic': True, 'mode': 'intrinsics',
                    'board': config, 'camera': None, 'camera_raw': CAMERA, 'robot': None}
        (HERE / '.tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=HERE / '.tmp') as temporary:
            folder = Path(temporary)
            write_json(folder / 'metadata.json', metadata)
            for i in range(25):
                pose = from_pose6([rng.uniform(-.17, -.03), rng.uniform(-.12, -.03), rng.uniform(.5, .85),
                                   rng.uniform(.2, .65), rng.uniform(-.5, .5), rng.uniform(-.7, .7)])
                uv = cv2.projectPoints(objects, cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3],
                                       np.array(CAMERA['K']), np.array(CAMERA['distortion_coefficients']))[0].reshape(-1, 2)
                uv += rng.normal(0., .03, uv.shape)
                write_json(folder / 'samples' / f'{i:04d}.json',
                           {'id': f'{i:04d}', 'corner_ids': list(range(104)), 'corners_px': uv.tolist()})
            with patch('sys.argv', ['calibrate_intrinsics.py', '--dataset', str(folder)]), \
                    patch('sys.stdout', new=io.StringIO()):
                self.assertEqual(calibrate_intrinsics.main(), 0)
            camera = read_json(folder / 'intrinsics.json')
            self.assertTrue(camera['passed'])
            self.assertEqual(len(camera['train_ids']), 20)
            self.assertEqual(len(camera['holdout_ids']), 5)
            self.assertLess(camera['holdout_rms_px'], .1)
            np.testing.assert_allclose(np.array(camera['K'])[[0, 1], [0, 1]],
                                       np.array(CAMERA['K'])[[0, 1], [0, 1]], atol=2.)

    def test_capture_corner_threshold_uses_selected_board(self):
        import collect
        import io
        (HERE / '.tmp').mkdir(exist_ok=True)
        large = HERE / 'board_configs' / 'charuco_rows9_cols14_5x5_20mm.json'
        with tempfile.TemporaryDirectory(dir=HERE / '.tmp') as temporary:
            for config, threshold, valid in ((large, 40, True), (large, 105, False),
                                              (HERE / 'board_config.json', 40, False)):
                with self.subTest(config=config.name, threshold=threshold), \
                        patch('sys.argv', ['collect.py', '--mode', 'intrinsics', '--board-config', str(config),
                                           '--min-corners', str(threshold), '--out', str(Path(temporary) / 'capture')]), \
                        patch('collect.RealSenseCamera', side_effect=RuntimeError('test camera reached')) as camera, \
                        patch('collect.cv2.destroyAllWindows'), patch('sys.stderr', new=io.StringIO()):
                    if valid:
                        with self.assertRaisesRegex(RuntimeError, 'test camera reached'):
                            collect.main()
                        camera.assert_called_once()
                    else:
                        with self.assertRaises(SystemExit) as caught:
                            collect.main()
                        self.assertEqual(caught.exception.code, 2)
                        camera.assert_not_called()

    def camera_device(self, serial='USB2_CAMERA', usb='2.1', color_fps=15, depth_fps=30):
        import pyrealsense2 as rs
        profiles = []
        for stream, pixel_format, width, height, fps in (
                (rs.stream.color, rs.format.bgr8, 1280, 720, color_fps),
                (rs.stream.depth, rs.format.z16, 640, 480, depth_fps)):
            profile = SimpleNamespace(stream_type=lambda value=stream: value,
                                      format=lambda value=pixel_format: value,
                                      width=lambda value=width: value, height=lambda value=height: value,
                                      fps=lambda value=fps: value, is_video_stream_profile=lambda: True)
            profile.as_video_stream_profile = lambda value=profile: value
            profiles.append(profile)
        info = {rs.camera_info.name: 'D435i', rs.camera_info.serial_number: serial,
                rs.camera_info.firmware_version: 'TEST', rs.camera_info.usb_type_descriptor: usb,
                rs.camera_info.physical_port: '/FAKE_USB_PORT'}
        sensor = SimpleNamespace(get_stream_profiles=lambda: profiles)
        return SimpleNamespace(supports=lambda key: key in info, get_info=info.__getitem__,
                               query_sensors=lambda: [sensor])

    def test_usb2_rejects_unsupported_default_before_opening_camera(self):
        device = self.camera_device()
        with patch('realsense_camera.rs.context', return_value=SimpleNamespace(query_devices=lambda: [device])), \
                patch('realsense_camera.rs.pipeline') as pipeline:
            with self.assertRaisesRegex(ValueError, 'USB 2.1.*1280x720@30') as caught:
                RealSenseCamera()
            self.assertIn('--fps 15', str(caught.exception))
            pipeline.assert_not_called()

    def test_stream_modes_belong_to_selected_camera(self):
        devices = [self.camera_device('USB3_CAMERA', '3.2', 30), self.camera_device()]
        with patch('realsense_camera.rs.context', return_value=SimpleNamespace(query_devices=lambda: devices)), \
                patch('realsense_camera.rs.pipeline') as pipeline:
            with self.assertRaisesRegex(ValueError, 'USB2_CAMERA'):
                RealSenseCamera(serial='USB2_CAMERA')
            pipeline.assert_not_called()

    def test_optional_depth_must_support_requested_frame_rate(self):
        device = self.camera_device()
        with patch('realsense_camera.rs.context', return_value=SimpleNamespace(query_devices=lambda: [device])), \
                patch('realsense_camera.rs.pipeline') as pipeline:
            with self.assertRaisesRegex(ValueError, '深度 Z16') as caught:
                RealSenseCamera(fps=15, depth=True)
            self.assertIn('--with-depth', str(caught.exception))
            pipeline.assert_not_called()

    def test_unresolvable_combination_does_not_start_or_change_modes(self):
        for device, fps in ((self.camera_device(), 15), (self.camera_device('USB3_CAMERA', '3.2', 30), 30)):
            with self.subTest(fps=fps), \
                    patch('realsense_camera.rs.context', return_value=SimpleNamespace(query_devices=lambda: [device])), \
                    patch('realsense_camera.rs.pipeline') as pipeline, \
                    patch('realsense_camera.rs.pipeline_wrapper'), patch('realsense_camera.rs.config') as config:
                config.return_value.can_resolve.return_value = False
                with self.assertRaisesRegex(RuntimeError, '请求的流组合无法解析'):
                    RealSenseCamera(fps=fps)
                pipeline.return_value.start.assert_not_called()
                pipeline.return_value.stop.assert_not_called()
                config.return_value.enable_stream.assert_called_once()

    def test_camera_start_error_keeps_sdk_reason_and_requested_mode(self):
        device = self.camera_device()
        with patch('realsense_camera.rs.context', return_value=SimpleNamespace(query_devices=lambda: [device])), \
                patch('realsense_camera.rs.pipeline') as pipeline, \
                patch('realsense_camera.rs.pipeline_wrapper'), patch('realsense_camera.rs.config') as config:
            config.return_value.can_resolve.return_value = True
            pipeline.return_value.start.side_effect = RuntimeError('device busy')
            with self.assertRaisesRegex(RuntimeError, '启动失败') as caught:
                RealSenseCamera(fps=15)
            self.assertIn('1280x720@15', str(caught.exception))
            self.assertIn('device busy', str(caught.exception))
            pipeline.return_value.stop.assert_not_called()

    def test_raw_receive_counts_include_unparsed_frames_without_sending(self):
        reader = CanPoseReader.__new__(CanPoseReader)
        reader.stop, reader.lock = threading.Event(), threading.Lock()
        reader.rx_counts, reader.rx_ids = Counter(), Counter()
        reader.last_frame_unix_s, reader.error = None, None
        decoded = []
        reader.robot = SimpleNamespace(_parser=SimpleNamespace(parse_packet=decoded.append))
        frames = [SimpleNamespace(arbitration_id=identity, is_error_frame=error, is_remote_frame=remote)
                  for identity, error, remote in ((0x155, False, False), (0x123, True, False),
                                                  (0x456, False, True), (0x155, False, False))]
        pending = iter([None, *frames])
        def recv(timeout):
            message = next(pending)
            if message is frames[-1]:
                reader.stop.set()
            return message
        reader.bus = SimpleNamespace(recv=recv)
        self.assertEqual(reader.transport_status()['frames'], 0)
        self.assertIsNone(reader.transport_status()['last_frame_age_s'])
        reader._receive()
        status = reader.transport_status()
        self.assertEqual(status['frames'], 4)
        self.assertEqual(status['error_frames'], 1)
        self.assertEqual(status['remote_frames'], 1)
        self.assertEqual(status['ids']['0x155'], 2)
        self.assertEqual(decoded, [frames[0], frames[3]])
        self.assertIsNone(status['receiver_error'])
        self.assertGreaterEqual(status['last_frame_age_s'], 0.)

    def test_redundant_joint_motion_does_not_bypass_duplicate_pose(self):
        P = np.eye(4)
        Q = from_pose6([-.1, -.1, .6, .2, .3, .1])
        old = {'id': '0000', 'robot': {'T_base_flange': P.tolist(), 'joints_rad': [0.] * 7}, 'T_color_board': Q.tolist()}
        packet = {'T_base_flange': P.tolist(), 'joints_rad': [np.deg2rad(30)] * 7}
        with self.assertRaises(MotionCaptureError) as caught:
            check_capture(packet, {'T_color_board': Q.tolist()}, [old])
        self.assertEqual(caught.exception.report['code'], 'duplicate_flange_pose')
        self.assertIn('冗余', str(caught.exception))

    def test_changed_board_with_constant_can_is_rejected(self):
        P = np.eye(4)
        Q = from_pose6([-.1, -.1, .6, 0., 0., 0.])
        rotated = from_pose6([-.1, -.1, .6, .4, 0., 0.])
        old = {'id': '0000', 'robot': {'T_base_flange': P.tolist(), 'joints_rad': [0.] * 7}, 'T_color_board': Q.tolist()}
        with self.assertRaises(MotionCaptureError) as caught:
            check_capture(old['robot'], {'T_color_board': rotated.tolist()}, [old])
        self.assertEqual(caught.exception.report['code'], 'vision_robot_rotation_mismatch')
        self.assertGreater(caught.exception.report['comparisons'][0]['board_rotation_deg'], 20.)

    def test_same_orientation_distinct_translation_is_accepted(self):
        P, next_P = np.eye(4), np.eye(4)
        next_P[0, 3] = .04
        Q = from_pose6([-.1, -.1, .6, .2, .1, .3])
        old = {'id': '0000', 'robot': {'T_base_flange': P.tolist()}, 'T_color_board': Q.tolist()}
        rows = check_capture({'T_base_flange': next_P.tolist()}, {'T_color_board': Q.tolist()}, [old])
        self.assertAlmostEqual(rows[0]['flange_translation_mm'], 40.)

    def test_motion_angle_invariant_in_both_hand_eye_modes(self):
        for mode in ('eye-in-hand', 'eye-to-hand'):
            with self.subTest(mode=mode):
                _, samples, _, _ = synthetic(mode, count=2, seed=7)
                rows = check_capture(samples[1]['robot'], samples[1], samples[:1])
                self.assertLess(rows[0]['rotation_disagreement_deg'], 1e-9)
                # Tool/board offsets change translation magnitudes, so translation
                # equality must not be a hand-eye invariant or a rejection criterion.
                self.assertGreater(abs(rows[0]['flange_translation_mm'] - rows[0]['board_translation_mm']), .01)

    def layout_fixture(self, dims=(5, 7), legacy=False, order='configured'):
        config = {'squares_x': 5, 'squares_y': 7, 'square_length_m': .04, 'marker_length_m': .03,
                  'dictionary': 'DICT_6X6_50', 'legacy_pattern': False, 'marker_ids': None}
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        board = cv2.aruco.CharucoBoard(dims, .04, .03, dictionary)
        board.setLegacyPattern(legacy)
        if order != 'configured':
            positions = np.array(board.getObjPoints()).mean(axis=1)
            ids = np.arange(len(positions), dtype=np.int32)
            if order == 'column-major':
                ids[np.lexsort((positions[:, 1], positions[:, 0]))] = np.arange(len(ids))
            else:
                ids = np.random.default_rng(31).permutation(ids)
            board = cv2.aruco.CharucoBoard(dims, .04, .03, dictionary, ids)
            board.setLegacyPattern(legacy)
        image = cv2.copyMakeBorder(board.generateImage((dims[0] * 100, dims[1] * 100)), 40, 40, 40, 40,
                                  cv2.BORDER_CONSTANT, value=255)
        corners, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(image)
        return diagnose_layout(image, config, dictionary, corners, ids)

    def test_diagnostic_standard_6x6(self):
        layouts, suggestion = self.layout_fixture()
        self.assertIsNotNone(suggestion)
        self.assertEqual((suggestion['squares_x'], suggestion['squares_y']), (5, 7))
        self.assertEqual(layouts[0]['charuco_corners'], 24)

    def test_diagnostic_swapped_rows_columns(self):
        layouts, suggestion = self.layout_fixture((7, 5))
        self.assertIsNotNone(suggestion)
        self.assertEqual((suggestion['squares_x'], suggestion['squares_y']), (7, 5))
        self.assertEqual(layouts[0]['charuco_corners'], 24)

    def test_diagnostic_column_major_ids(self):
        layouts, suggestion = self.layout_fixture(order='column-major')
        self.assertIsNotNone(suggestion)
        self.assertEqual(layouts[0]['marker_order'], 'column-major')
        self.assertEqual(layouts[0]['charuco_corners'], 24)
        configured = next(r for r in layouts if r['layout'] == '5x7, legacy=False, ids=configured')
        self.assertEqual(configured['charuco_corners'], 0)
        self.assertEqual(configured['marker_geometry']['matched_markers'], 17)

    def test_diagnostic_even_row_legacy(self):
        layouts, suggestion = self.layout_fixture((6, 8), legacy=True)
        self.assertIsNotNone(suggestion)
        self.assertTrue(suggestion['legacy_pattern'])
        self.assertEqual(layouts[0]['charuco_corners'], 35)

    def test_diagnostic_rejects_unmodelled_id_order(self):
        layouts, suggestion = self.layout_fixture(order='random')
        self.assertIsNone(suggestion)
        self.assertFalse(any(r['plausible_full_board'] for r in layouts))

    def test_exact_both_modes(self):
        for mode in ('eye-in-hand', 'eye-to-hand'):
            with self.subTest(mode=mode):
                _, samples, truth, fixture = synthetic(mode)
                x, f, _, _ = fit(samples[:28], mode)
                t, r = distance(truth, x)
                self.assertLess(t, 1e-8)
                self.assertLess(r, 1e-8)
                self.assertLess(distance(fixture, f)[0], 1e-8)
                report = evaluate(samples[28:], mode, x, f)
                self.assertLess(report['translation_mm']['rms'], 1e-5)

    def test_noisy_pnp_both_modes(self):
        for mode in ('eye-in-hand', 'eye-to-hand'):
            with self.subTest(mode=mode):
                metadata, samples, truth, fixture = synthetic(mode, noise_px=.12)
                x, f, _, _ = fit(samples[:28], mode)
                t, r = distance(truth, x)
                self.assertLess(t, .003)
                self.assertLess(r, np.deg2rad(.4))
                board, _ = load_board(dictionary='DICT_4X4_50')
                report = evaluate(samples[28:], mode, x, f, board, CAMERA)
                self.assertTrue(passed(report, {'translation_rms_mm': 5., 'rotation_rms_deg': 2., 'loop_reprojection_rms_px': 5.}))

    def test_translation_only_rejected(self):
        poses = [from_pose6([i * .01, 0, .4, 0, 0, 0]) for i in range(20)]
        with self.assertRaisesRegex(ValueError, '只有平移'):
            excitation(poses)

    def test_single_axis_rejected(self):
        poses = [from_pose6([i * .01, 0, .4, 0, 0, i * .05]) for i in range(20)]
        with self.assertRaisesRegex(ValueError, '激励不足'):
            excitation(poses)

    def test_holdout_does_not_refit_fixture(self):
        _, samples, _, _ = synthetic('eye-in-hand')
        x, f, _, _ = fit(samples[:28], 'eye-in-hand')
        heldout = samples[28:]
        for s in heldout:
            s['robot']['T_base_flange'][0][3] += .04
        report = evaluate(heldout, 'eye-in-hand', x, f)
        self.assertGreater(report['translation_mm']['rms'], 39.)

    def test_inverse_robot_required_eye_to_hand(self):
        _, samples, _, _ = synthetic('eye-to-hand')
        try:
            x, f, _, _ = fit(samples[:28], 'eye-in-hand')
        except ValueError:
            return  # A wrong-mode input may be rejected before it produces an SO(3) result.
        report = evaluate(samples[28:], 'eye-in-hand', x, f)
        self.assertGreater(report['translation_mm']['rms'], 10.)

    def test_planar_pnp(self):
        board, _ = load_board(dictionary='DICT_4X4_50')
        detector = BoardDetector(board, CAMERA)
        truth = from_pose6([-.11, -.14, .65, .32, -.26, .12])
        obj = board.getChessboardCorners().astype(np.float64)
        uv = cv2.projectPoints(obj, cv2.Rodrigues(truth[:3, :3])[0], truth[:3, 3], np.array(CAMERA['K']), np.array(CAMERA['distortion_coefficients']))[0].reshape(-1, 2)
        recovered = detector.pose_from_corners(np.arange(24), uv)
        t, r = distance(truth, np.array(recovered['T_color_board']))
        self.assertLess(t, 1e-7)
        self.assertLess(r, 1e-7)

    def test_actual_board_detection(self):
        board, _ = load_board(dictionary='DICT_4X4_50')
        image = cv2.copyMakeBorder(board.generateImage((1000, 1400)), 80, 80, 80, 80, cv2.BORDER_CONSTANT, value=255)
        detected = BoardDetector(board).detect(image)
        self.assertEqual(len(detected['corner_ids']), 24)
        self.assertEqual(len(detected['marker_ids']), 17)

    def test_dictionary_is_not_guessed(self):
        config = read_json(HERE / 'board_config.json')
        config['dictionary'] = None
        (HERE / '.tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=HERE / '.tmp') as temp:
            path = Path(temp) / 'unknown_board.json'
            write_json(path, config)
            with self.assertRaisesRegex(ValueError, '必须指定'):
                load_board(path)

    def test_realsense_distortion_models(self):
        base = dict(CAMERA)
        base['distortion_model'] = 'distortion.brown_conrady'
        self.assertEqual(opencv_intrinsics(base)['distortion_model'], 'opencv_brown')
        for model in ('inverse_brown_conrady', 'modified_brown_conrady'):
            base['distortion_model'] = f'distortion.{model}'
            with self.assertRaises(ValueError):
                opencv_intrinsics(base)
        base['distortion_coefficients'] = [0.] * 5
        self.assertEqual(opencv_intrinsics(base)['distortion_model'], 'none')

    def test_realsense_column_major_extrinsics(self):
        import pyrealsense2 as rs
        truth = from_pose6([.01, -.02, .03, .4, -.2, .3])
        ext = rs.extrinsics()
        ext.rotation = truth[:3, :3].flatten(order='F').tolist()
        ext.translation = truth[:3, 3].tolist()
        recovered = extrinsic_matrix(ext)
        point = [.1, .2, .7]
        # Independent SDK point-transform checks storage order, not a mirrored reshape.
        actual = rs.rs2_transform_point_to_point(ext, point)
        np.testing.assert_allclose((recovered @ np.r_[point, 1])[:3], actual, atol=1e-7)

    def test_incomplete_or_stale_feedback(self):
        from robot_pose import MEASURED_PARTS
        parser = SimpleNamespace()
        robot = SimpleNamespace(_parser=parser, get_joint_angles=lambda: SimpleNamespace(msg=[.1] * 7))
        with self.assertRaisesRegex(ValueError, '不完整'):
            complete_feedback(robot, 'measured', 'v121', 100, .15)
        for name in MEASURED_PARTS:
            setattr(parser, name, SimpleNamespace(timestamp=99.99))
        joints, _ = complete_feedback(robot, 'measured', 'v121', 100, .15)
        self.assertEqual(len(joints), 7)
        parser.joint_12.timestamp = 98
        with self.assertRaisesRegex(ValueError, '过期'):
            complete_feedback(robot, 'measured', 'v121', 100, .15)

    def feedback_reader_fixture(self):
        parser = SimpleNamespace()
        for name in ('leader_joint_12', 'leader_joint_34', 'leader_joint_56', 'leader_joint_7'):
            setattr(parser, name, SimpleNamespace(timestamp=99.99))
        robot = SimpleNamespace(_parser=parser,
            get_joint_angles=lambda: SimpleNamespace(msg=[.1] * 7),
            get_leader_joint_angles=lambda: SimpleNamespace(msg=[.2] * 7),
            fk=lambda q: [.3, 0., .4, 0., 0., 0.])
        reader = CanPoseReader.__new__(CanPoseReader)
        reader.robot, reader.firmware, reader.source = robot, 'v121', 'measured'
        reader.max_age, reader.error, reader.lock = .15, None, threading.Lock()
        reader.history = deque()
        return reader

    def test_diagnostic_sources_never_switch_collection(self):
        reader = self.feedback_reader_fixture()
        with patch('robot_pose.time.time', return_value=100.):
            status = reader.inspect_sources()
            self.assertFalse(status['measured']['available'])
            self.assertTrue(status['leader']['available'])
            self.assertEqual(reader.source, 'measured')
            with self.assertRaisesRegex(ValueError, '--joint-source leader'):
                reader.sample()
            self.assertEqual(len(reader.history), 0)
            reader.source = 'leader'
            sample = reader.sample()
            self.assertEqual(sample['source'], 'leader')
            self.assertEqual(sample['joints_rad'], [.2] * 7)

    def test_stale_alternate_feedback_is_not_suggested(self):
        reader = self.feedback_reader_fixture()
        reader.robot._parser.leader_joint_12.timestamp = 98.
        with patch('robot_pose.time.time', return_value=100.):
            status = reader.inspect_sources()
            self.assertFalse(status['leader']['available'])
            with self.assertRaises(ValueError) as caught:
                reader.sample()
            self.assertNotIn('--joint-source leader', str(caught.exception))

    def test_stationary_window_rejects_motion(self):
        poses = [{'T_base_flange': np.eye(4).tolist(), 'joints_rad': [0.] * 7} for _ in range(5)]
        require_stationary(poses)
        poses[-1]['joints_rad'][0] = .02
        with self.assertRaisesRegex(ValueError, '还在移动'):
            require_stationary(poses)

    def test_offline_sdk_decoder_and_fk(self):
        import can
        sdk = load_sdk()
        cfg = sdk.create_agx_arm_config(robot=sdk.ArmModel.NERO, firmeware_version='v121', auto_connect=False, enable_check_can=False)
        robot = sdk.AgxArmFactory.create_arm(cfg)
        try:
            # Verify all parser names & unit conversions against real SDK without opening CAN.
            now = 100.
            # IDs are read from parser table rather than assumed, as firmware may differ.
            for name, numbers in zip(('joint_12', 'joint_34', 'joint_56', 'joint_7'), ([100, -200], [300, -400], [500, -600], [700, 0])):
                ids = [i for i, spec in robot._parser._rx_map.items() if spec[0] == name]
                self.assertEqual(len(ids), 1, name)
                data = b''.join(int(n).to_bytes(4, 'big', signed=True) for n in numbers)
                robot._parser.parse_packet(can.Message(arbitration_id=ids[0], data=data, is_extended_id=False, timestamp=now))
            joints, _ = complete_feedback(robot, 'measured', 'v121', now, .15)
            np.testing.assert_allclose(joints, np.deg2rad([.1, -.2, .3, -.4, .5, -.6, .7]), atol=1e-12)
            pose = from_pose6(robot.fk(joints.tolist()))
            checked(pose)
            from pyAgxArm.utiles.tf import pose6_to_T16
            np.testing.assert_allclose(pose, np.array(pose6_to_T16(robot.fk(joints.tolist()))).reshape(4, 4), atol=1e-12)
        finally:
            robot.disconnect()

    def test_real_sdk_leader_payload_changes_update_joint_and_fk(self):
        import can
        sdk = load_sdk()
        cfg = sdk.create_agx_arm_config(robot=sdk.ArmModel.NERO, firmeware_version='v121', auto_connect=False, enable_check_can=False)
        robot = sdk.AgxArmFactory.create_arm(cfg)
        try:
            names = ('leader_joint_12', 'leader_joint_34', 'leader_joint_56', 'leader_joint_7')
            ids = {name: next(i for i, spec in robot._parser._rx_map.items() if spec[0] == name) for name in names}
            for name in names:
                robot._parser.parse_packet(can.Message(arbitration_id=ids[name], data=b'\0' * 8,
                    is_extended_id=False, timestamp=100.))
            before, _ = complete_feedback(robot, 'leader', 'v121', 100., .15)
            payload = int(45000).to_bytes(4, 'big', signed=True) + b'\0' * 4
            robot._parser.parse_packet(can.Message(arbitration_id=ids['leader_joint_12'], data=payload,
                is_extended_id=False, timestamp=100.))
            after, _ = complete_feedback(robot, 'leader', 'v121', 100., .15)
            np.testing.assert_allclose(after, [np.pi / 4] + [0.] * 6, atol=1e-12)
            t, r = distance(from_pose6(robot.fk(before.tolist())), from_pose6(robot.fk(after.tolist())))
            self.assertAlmostEqual(np.rad2deg(r), 45., places=8)
        finally:
            robot.disconnect()

    def test_invalid_rotation_and_output_escape(self):
        bad = np.eye(4)
        bad[0, 0] = -1
        with self.assertRaises(ValueError):
            checked(bad)
        with self.assertRaisesRegex(ValueError, '输出必须'):
            inside_folder('/tmp/handeye-outside.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write-demo', help='另存两套明确标记 synthetic=true 的测试数据，不是真机标定')
    args = p.parse_args()
    if args.write_demo:
        root = inside_folder(args.write_demo)
        if root.exists():
            raise ValueError('合成输出目录已存在')
        for mode in ('eye-in-hand', 'eye-to-hand'):
            metadata, samples, x, f = synthetic(mode, noise_px=.12)
            folder = root / mode
            write_json(folder / 'metadata.json', metadata)
            for sample in samples:
                write_json(folder / 'samples' / f'{sample["id"]}.json', sample)
            write_json(folder / 'ground_truth.json', {'X': x.tolist(), 'F': f.tolist(), 'synthetic': True})
        print(f'合成数据：{root}（只能验证程序，不可用于实际设备）')
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CalibrationTests))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
