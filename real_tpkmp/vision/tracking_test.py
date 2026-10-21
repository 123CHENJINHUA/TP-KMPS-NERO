"""Tracking and measurement tests without camera, CAN, CUDA or neural inference."""
from .common import ROOT, object_config, read_capture
from .tracking_core import TrackingGuard, Measurements, static_statistics, center_position
from .tracking import Engine, resize_frame, dashboard, plot_history
from .smoke import fixture
from .box_prompt import MovingBox, checked_box, select_box_candidate, select_box
from .segmentation import get_masks
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation


def pose(x=0., angle=0.):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler('z', angle).as_matrix()
    result[:3, 3] = [x, 0, .5]
    return result


GOOD = dict(passed=True, reasons=[])


def result(matrix, epoch=1, accepted=True):
    return dict(accepted=accepted, epoch=epoch, T_color_object=matrix.tolist())


class TrackingTests(unittest.TestCase):
    def test_registration_tracking_jump_recovery(self):
        guard = TrackingGuard([0, 0, 0])
        self.assertTrue(guard.needs_registration(1))
        self.assertTrue(guard.assess(pose(), GOOD, 1., True)['accepted'])
        self.assertEqual(guard.epoch, 1)
        self.assertFalse(guard.needs_registration(2))
        self.assertTrue(guard.assess(pose(.01), GOOD, 2., False)['accepted'])
        bad = guard.assess(pose(.3), GOOD, 3., False)
        self.assertFalse(bad['accepted'])
        self.assertIn('translation_jump', bad['reasons'])
        self.assertTrue(guard.needs_registration(4))
        self.assertTrue(guard.assess(pose(.3), GOOD, 4., True)['accepted'])
        self.assertEqual(guard.epoch, 2)

    def test_quality_loss_and_temporal_gap_force_registration(self):
        guard = TrackingGuard([0, 0, 0])
        guard.assess(pose(), GOOD, 1., True)
        self.assertTrue(guard.needs_registration(12.))
        bad = guard.assess(pose(), GOOD, 12., False)
        self.assertIn('tracking_frame_gap', bad['reasons'])
        guard.assess(pose(), GOOD, 13., True)
        bad = guard.assess(pose(), dict(passed=False, reasons=['depth_median_error']), 14., False)
        self.assertFalse(bad['accepted'])
        self.assertIsNone(guard.pose)
        self.assertTrue(guard.needs_registration(15.))

    def test_rotation_flip_and_old_frame_rejected(self):
        guard = TrackingGuard([0, 0, 0])
        guard.assess(pose(), GOOD, 1., True)
        self.assertIn('rotation_jump', guard.assess(pose(angle=np.pi), GOOD, 2., False)['reasons'])
        guard.assess(pose(), GOOD, 3., True)
        self.assertIn('non_increasing_frame_time', guard.assess(pose(), GOOD, 3., False)['reasons'])

    def test_center_step_independent_of_original_mesh_origin(self):
        center = np.array([.1, .05, .02])
        guard = TrackingGuard(center, max_step_m=.001, max_step_rad=np.pi)
        first, second = pose(), pose(angle=np.pi/2)
        second[:3, 3] = center_position(first, center)-second[:3, :3]@center
        guard.assess(first, GOOD, 1., True)
        check = guard.assess(second, GOOD, 2., False)
        self.assertTrue(check['accepted'])
        self.assertLess(check['center_step_m'], 1e-10)

    def test_registered_depth_resize_keeps_metric_z_and_camera_projection(self):
        rgb = np.full((480, 640, 3), 100, np.uint8)
        depth = np.full((480, 640), .55, np.float32)
        K = np.array([[600., 0, 320], [0, 600, 240], [0, 0, 1]])
        small_rgb, small_depth, small_K = resize_frame(rgb, depth, K, 320)
        self.assertEqual(small_rgb.shape, (240, 320, 3))
        np.testing.assert_allclose(small_depth, .55)
        point = np.array([.05, .04, .55])
        old, new = K@point, small_K@point
        np.testing.assert_allclose(new[:2]/new[2], (old[:2]/old[2])*.5)

    def test_static_jitter_has_no_absolute_accuracy_claim(self):
        stats = static_statistics([pose(-.001), pose(), pose(.001)], [0, 0, 0])
        self.assertAlmostEqual(stats['center_rms_jitter_mm'], np.sqrt(2/3))
        self.assertAlmostEqual(stats['rotation_rms_jitter_deg'], 0.)
        self.assertFalse(stats['absolute_pose_accuracy_tested'])

    def measure(self, measurements, kind, matrix, epoch=1):
        measurements.start(kind)
        report = None
        for _ in range(3):
            report = measurements.consume(result(matrix, epoch))
        return report

    def test_known_translation_and_rotation_rejection(self):
        measures = Measurements([0, 0, 0], samples=3, expected_distance_mm=50.)
        self.measure(measures, 'A', pose())
        report = self.measure(measures, 'B', pose(.05))
        self.assertTrue(report['measurement_valid'])
        self.assertAlmostEqual(report['measured_distance_mm'], 50.)
        self.assertAlmostEqual(report['distance_error_mm'], 0.)
        report = self.measure(measures, 'B', pose(.05, .2))
        self.assertFalse(report['measurement_valid'])
        self.assertIsNone(report['distance_error_mm'])

    def test_lost_or_reregistered_pose_invalidates_measurement(self):
        measures = Measurements([0, 0, 0], samples=3)
        self.measure(measures, 'A', pose())
        measures.start('B')
        self.assertIsNone(measures.consume(result(pose(.05), epoch=2)))
        self.assertIsNone(measures.anchor)
        self.assertIsNone(measures.active)
        measures.start('static')
        measures.consume(result(pose()))
        measures.consume(result(pose(), accepted=False))
        self.assertIsNone(measures.active)

    def test_moving_object_not_reported_as_static(self):
        measures = Measurements([0, 0, 0], samples=3)
        measures.start('static')
        for x in (0., .02, .04):
            report = measures.consume(result(pose(x)))
        self.assertFalse(report['steady'])
        self.assertFalse(report['measurement_valid'])

    def test_box_validation_and_spatial_candidate_selection(self):
        for bad in ([0, 0, 0, 1], [0, 0, 2, 1], [0, float('nan'), .5, .5], [1, 0, .5, 1]):
            with self.assertRaises(ValueError):
                checked_box(bad)
        masks = np.zeros((2, 100, 200), bool)
        masks[0, 20:60, 20:80] = True
        masks[1, 20:60, 130:190] = True
        self.assertEqual(select_box_candidate(masks, [.7, .99], [.05, .1, .5, .75]), 0)
        with self.assertRaises(ValueError):
            select_box_candidate(masks, [.1, .99], [.05, .1, .5, .75])

    def test_box_selection_coordinates_match_frozen_resized_image_and_cancel(self):
        import cv2, os
        image = np.zeros((1080, 1920, 3), np.uint8)
        with patch.dict(os.environ, {'DISPLAY': ':fake'}), patch.object(cv2, 'destroyWindow'), \
                patch.object(cv2, 'selectROI', return_value=(100, 50, 200, 150)) as dialog:
            box = select_box(image)
        np.testing.assert_allclose(box, [100/960, 50/540, 300/960, 200/540])
        self.assertEqual(dialog.call_args.args[1].shape, (540, 960, 3))
        with patch.dict(os.environ, {'DISPLAY': ':fake'}), patch.object(cv2, 'destroyWindow'), \
                patch.object(cv2, 'selectROI', return_value=(0, 0, 0, 0)):
            self.assertIsNone(select_box(image))

    def test_box_prompt_follows_target_motion_and_requires_selection_after_clear(self):
        import cv2
        image = np.zeros((120, 160, 3), np.uint8)
        texture = np.random.default_rng(42).integers(30, 230, (50, 70, 1), dtype=np.uint8)
        image[25:75, 20:90] = np.repeat(texture, 3, axis=2)
        mask = np.zeros((120, 160), bool)
        mask[25:75, 20:90] = True
        tracker = MovingBox()
        tracker.accept(image, mask)
        original = tracker.box.copy()
        following = cv2.warpAffine(image, np.array([[1, 0, 12], [0, 1, 8]], np.float32), (160, 120))
        predicted = tracker.predict(following)
        np.testing.assert_allclose((predicted[:2]+predicted[2:]-original[:2]-original[2:])/2,
                                   [12/160, 8/120], atol=.01)
        # A few distant mask speckles must not grow the user's prompt box.
        noisy = mask.copy()
        noisy[0, 0] = True
        tracker.accept(image, noisy, original)
        np.testing.assert_allclose(tracker.box, original)
        tracker.clear()
        with self.assertRaisesRegex(ValueError, '重新框选'):
            tracker.predict(following)

    def test_static_box_segmentation_passes_boxes_without_text(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            output = Path(temp)
            frames = np.zeros((3, 40, 60, 3), np.uint8)
            box = [.1, .2, .8, .9]
            def fake_worker(command, **kwargs):
                self.assertNotIn('--prompt', command)
                with np.load(command[command.index('--input')+1], allow_pickle=False) as payload:
                    np.testing.assert_allclose(payload['boxes_xyxy'], np.repeat([box], 3, axis=0))
                np.savez_compressed(command[command.index('--output')+1], masks=np.ones((3, 40, 60), bool), scores=[.9]*3)
            with patch('real_tpkmp.vision.segmentation.subprocess.run', side_effect=fake_worker):
                masks, metadata = get_masks(frames, object_config(), output, method='box', box=box)
            self.assertIsNone(metadata['prompt'])
            self.assertEqual(metadata['method'], 'box')
            self.assertEqual(masks.shape, (3, 40, 60))

    def test_box_loss_waits_for_explicit_reselection(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            directory = Path(temp)
            capture, known, _ = fixture(directory/'input')
            _, _, data = read_capture(capture)
            output = directory/'output'
            output.mkdir()
            args = Namespace(mask_method='box', box=[.2, .2, .8, .8], max_step_mm=80., max_step_deg=45.,
                     max_gap_s=10., foundationpose_root=None, register_iterations=5, track_iterations=2,
                     sam3_source=None, sam3_checkpoint=None, prompt=None, instance=None)
            engine = Engine(args, object_config(), output)
            class FakePose:
                def __init__(self): self.resets = []
                def request(self, request):
                    self.resets.append(request['reset'])
                    return dict(ok=True, T_color_object=known.tolist(), registered=request['reset'], elapsed_ms=1.)
            class FakeSegmentation:
                def __init__(self): self.calls = 0
                def request(self, request):
                    self.calls += 1
                    self_box = checked_box(request['box_xyxy'])
                    with np.load(request['input'], allow_pickle=False) as archive: rgb = archive['rgb']
                    if not np.any(rgb[:, :, 2] > rgb[:, :, 0]):
                        return dict(ok=False, error='目标消失')
                    np.save(request['output'], data['depth_m'][0] > 0, allow_pickle=False)
                    return dict(ok=True, score=.9)
            engine.pose, engine.sam3 = FakePose(), FakeSegmentation()
            for i in range(5):
                rgb = data['rgb'][0].copy()
                if i == 2: rgb[:] = 100
                frame = dict(rgb=rgb, depth_m=data['depth_m'][0], K=data['K'], revision=0,
                             meta=dict(host_monotonic_s=i+1., frame_number=i), received_monotonic_s=0.)
                if i == 4: frame['box_xyxy'] = args.box
                report = engine.process(frame)
                self.assertEqual(report['accepted'], i in (0, 1, 4))
                if i in (2, 3): self.assertTrue(report['needs_box_selection'])
            self.assertEqual(engine.sam3.calls, 4)
            self.assertEqual(engine.pose.resets, [True, False, True])

    def test_gui_keeps_live_pixels_separate_from_processed_overlay(self):
        import cv2
        live = np.full((240, 320, 3), [255, 0, 0], np.uint8)
        processed = np.full((240, 320, 3), [0, 255, 0], np.uint8)
        last = dict(_view=processed, accepted=False, mode='lost', frame_number=1,
                    result_monotonic_s=0., frame_meta=dict(host_monotonic_s=0.), processing_ms=200.,
                    input_to_result_ms=210., quality=None, reasons=['test'])
        session = Namespace(last=last, measurements=Namespace(message='test'))
        with patch.object(cv2, 'imshow') as show, patch.object(cv2, 'waitKey', return_value=-1), \
                patch.object(cv2, 'getWindowProperty', return_value=1):
            dashboard(live, session, Namespace(ready=True))
        canvas = show.call_args.args[1]
        np.testing.assert_array_equal(canvas[100, 100], [0, 0, 255])
        np.testing.assert_array_equal(canvas[100, 660], [0, 255, 0])

    def test_plot_preserves_lost_frame_gaps(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            history = [dict(time_s=1., xyz=[0, 0, .5], quality=None, accepted=True),
                       dict(time_s=2., xyz=None, quality=None, accepted=False)]
            path = Path(temp)/'tracking.png'
            plot_history(history, path)
            self.assertGreater(path.stat().st_size, 1000)

    def test_pipeline_reinitializes_after_missing_mask_and_never_exports_robot_target(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            directory = Path(temp)
            capture, known, _ = fixture(directory/'input')
            _, meta, data = read_capture(capture)
            output = directory/'output'
            output.mkdir()
            args = Namespace(max_step_mm=80., max_step_deg=45., max_gap_s=10., foundationpose_root=None,
                     register_iterations=5, track_iterations=2, sam3_source=None, sam3_checkpoint=None,
                     prompt=None, instance=None)
            engine = Engine(args, object_config(), output)
            class FakePose:
                def __init__(self):
                    self.resets = []
                def request(self, request):
                    self.resets.append(request['reset'])
                    return dict(ok=True, T_color_object=known.tolist(), registered=request['reset'], elapsed_ms=1.)
            engine.pose, engine.sam3 = FakePose(), None
            for i in range(4):
                rgb = data['rgb'][0].copy()
                if i == 2:
                    rgb[:] = 100  # Current-frame segmentation lost, no stale mask reuse.
                frame = dict(rgb=rgb, depth_m=data['depth_m'][0], K=data['K'], revision=0,
                             meta=dict(host_monotonic_s=i+1., frame_number=i), received_monotonic_s=0.)
                output = engine.process(frame)
                self.assertFalse(output['usable_for_robot'])
                self.assertTrue(output['diagnostic_only'])
                self.assertEqual(output['accepted'], i != 2)
                if i == 2:
                    self.assertIsNone(output['T_color_object'])
            self.assertEqual(engine.pose.resets, [True, False, True])
            self.assertEqual(engine.guard.epoch, 2)


def run():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TrackingTests)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)
