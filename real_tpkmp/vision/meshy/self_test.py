"""No network, API key, GPU inference, camera or CAN. Test conversions via CPU worker."""
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ..common import ROOT, object_config, read_json
from ..geometry import load_mesh
from . import api
from .models import import_model
import cv2
import numpy as np


class MeshyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.root = Path(self.temp.name)
        self.images = []
        for i in range(4):
            path = self.root/f'view_{i}.png'
            cv2.imwrite(str(path), np.full((32, 40, 3), 30+i*40, np.uint8))
            self.images.append(str(path))

    def tearDown(self):
        self.temp.cleanup()

    def options(self, count=1):
        return Namespace(images=self.images[:count], out=str(self.root/f'job_{count}'), ai_model='meshy-7.1',
                         geometry_resolution='standard', texture=False, polycount=None,
                         dry_run=True, wait_seconds=0., poll_seconds=1.)

    def test_single_multi_payload_and_local_snapshots(self):
        for count in (1, 4):
            args = self.options(count)
            directory, manifest = api.prepare(args)
            body = api.request_body(manifest, directory)
            self.assertEqual(manifest['endpoint'], api.ENDPOINTS[0 if count == 1 else 1])
            self.assertIn('image_url' if count == 1 else 'image_urls', body)
            self.assertNotIn('Authorization', (directory/'request.json').read_text())
            self.assertNotIn('base64', (directory/'request.json').read_text())
            self.assertFalse(body['auto_size'])
            self.assertFalse(body['image_enhancement'])
            self.assertFalse(body['should_remesh'])
            snapshot = directory/manifest['images'][0]['snapshot']
            snapshot.write_bytes(b'modified')
            with self.assertRaisesRegex(ValueError, '被修改'):
                api.request_body(manifest, directory)

    def test_input_limits_duplicates_and_no_network_dry_run(self):
        for paths in ([], self.images+[self.images[0]], [self.images[0]]*2):
            with self.assertRaises(ValueError):
                api.validate_images(paths)
        args = self.options()
        with patch.object(api, 'Client', side_effect=AssertionError('must not create a client')):
            api.generate(args)
        self.assertFalse((Path(args.out)/'submission.json').exists())

    def test_failed_post_is_not_retried_or_charged_twice(self):
        args = self.options()
        directory, manifest = api.prepare(args)
        class Client:
            def request(self, *args, **kwargs):
                raise RuntimeError('simulated network failure after POST')
        with self.assertRaises(RuntimeError):
            api.submit(directory, manifest, args, Client())
        self.assertTrue((directory/'submission.json').exists())
        with self.assertRaisesRegex(ValueError, '已有提交尝试'):
            api.submit(directory, manifest, args, Client())

    def test_api_host_auth_and_safe_http_error(self):
        from urllib.error import HTTPError
        from unittest.mock import Mock
        with patch.dict(api.os.environ, {'MESHY_API_KEY': 'fake-key-for-offline-test'}):
            client = api.Client()
        opener = Mock()
        opener.open.side_effect = HTTPError(api.API, 402, 'Payment Required', {}, None)
        client.opener = opener
        with self.assertRaisesRegex(RuntimeError, '账户积分不足') as error:
            client.request('image-to-3d', payload={'image_url': 'data:fixture'})
        self.assertNotIn('fake-key', str(error.exception))
        self.assertEqual(opener.open.call_count, 1)
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, api.API+'/image-to-3d')
        self.assertEqual(request.get_header('Authorization'), 'Bearer fake-key-for-offline-test')
        with self.assertRaises(ValueError):
            client.request('https://another-host.invalid')

    def test_resume_download_is_idempotent_and_checks_files(self):
        args = self.options(4)
        directory, manifest = api.prepare(args)
        task = dict(id='test-job', status='SUCCEEDED', progress=100,
                    model_urls={'stl': 'https://assets.meshy.ai/test.stl'})
        class Client:
            def __init__(self):
                self.calls = []
            def request(self, endpoint, task_id=None, payload=None):
                self.calls.append((endpoint, task_id, payload))
                return {'result': 'test-job'} if payload else task
        client = Client()
        def local_download(url, path):
            path.write_bytes(b'offline-model-fixture')
            return dict(path=path.name, sha256=api.sha256(path), bytes=path.stat().st_size)
        with patch.object(api, 'download', side_effect=local_download) as mocked:
            api.submit(directory, manifest, args, client)
            api.poll(directory, manifest, client, 0, 1)
            self.assertEqual(mocked.call_count, 1)
        self.assertEqual(sum(payload is not None for _, _, payload in client.calls), 1)
        (directory/'model.stl').write_bytes(b'edited')
        with self.assertRaisesRegex(ValueError, '不会覆盖'):
            api.poll(directory, manifest, client, 0, 1)

    def import_options(self, mismatch=False):
        return Namespace(mesh=str(ROOT/'models/keel/keel.STL'), object_id='keel_meshy_fixture',
                         measured_extents_mm=[150., 98., 50. if mismatch else 23.],
                         rotate_xyz_deg=[0., 0., 90.], prompt='blue metal bracket',
                         out=str(self.root/'import'), pose_python=None)

    def test_import_real_units_rotated_frame_and_compatible_config(self):
        args = self.import_options()
        original_sha = api.sha256(args.mesh)
        output = import_model(args)
        cfg = object_config(output/'object.json')
        triangles = load_mesh(cfg)
        np.testing.assert_allclose(np.ptp(triangles.reshape(-1, 3), axis=0), [.098, .15, .023], atol=1e-7)
        report = read_json(output/'import.json')
        self.assertAlmostEqual(report['uniform_scale_to_m'], .001, places=8)
        self.assertEqual(api.sha256(args.mesh), original_sha)
        self.assertNotIn('hsv_lower', cfg)
        self.assertTrue((output/'preview.png').is_file())
        self.assertTrue((output/'model.glb').is_file())
        self.assertFalse(np.allclose(np.array(report['source_to_object_similarity'])[:3, :3], np.eye(3)))
        with self.assertRaises(FileExistsError):
            import_model(args)
        # GLB scene/node transforms must round-trip to the same meter-scale STL.
        args.mesh = str(output/'model.glb')
        args.out = str(self.root/'glb_import')
        args.rotate_xyz_deg = [0., 0., 0.]
        second = import_model(args)
        np.testing.assert_allclose(np.ptp(load_mesh(object_config(second/'object.json')).reshape(-1, 3), axis=0),
                                   [.098, .15, .023], atol=1e-7)

    def test_bad_proportions_do_not_become_a_usable_object(self):
        args = self.import_options(mismatch=True)
        with self.assertRaisesRegex(ValueError, '不会强行拉伸'):
            import_model(args)
        output = Path(args.out)
        self.assertFalse(read_json(output/'import.json')['dimensions_passed'])
        self.assertFalse((output/'object.json').exists())
        self.assertTrue((output/'model.stl').exists())


def run():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MeshyTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
