import math
from pathlib import Path
import struct
import time
import unittest

import numpy as np

from sim_bridge.app import resolve_scene_usd
from sim_bridge.dual import clip_teach_joints, teach_gripper_opening, urdf_joint_limits
from sim_bridge.teach_can import CAN_FRAME, TeachCanReader, decode_teach_frame


PROJECT = Path(__file__).resolve().parents[1]


class TeachSyncTests(unittest.TestCase):
    def test_decode_live_nero_joint_and_gripper_frames(self):
        # Passive frames captured from can0, checked against pyAgxArm's scale.
        joint = CAN_FRAME.pack(0x155, 8, bytes.fromhex('000013e6fffe7099'))
        kind, slot, angles = decode_teach_frame(joint)
        self.assertEqual((kind, slot), ('joint', 0))
        self.assertAlmostEqual(angles[0], 5.094 * math.pi / 180)
        self.assertAlmostEqual(angles[1], -102.247 * math.pi / 180)
        grip = CAN_FRAME.pack(0x159, 8, bytes.fromhex('0000b9970bb80100'))
        kind, slot, state = decode_teach_frame(grip)
        self.assertEqual((kind, slot), ('gripper', None))
        self.assertEqual((state['value'], state['mode'], state['force_N']),
                         (.047511, 'width', 3.0))
        self.assertIsNone(decode_teach_frame(CAN_FRAME.pack(0x155, 2, b'12345678')))

    def test_stale_or_incomplete_stream_is_not_applied(self):
        reader = TeachCanReader.__new__(TeachCanReader)
        reader.stale_after = .25
        now = time.monotonic()
        reader.states = {'can0': {
            'parts': {0: ((0., 0.), now), 2: ((0., 0.), now),
                      4: ((0., 0.), now), 6: ((0.,), now - 1.)},
            'gripper': ({'value': .05, 'mode': 'width'}, now),
        }}
        self.assertIsNone(reader.snapshot('can0')['joints'])
        reader.states['can0']['parts'][6] = ((0.,), now)
        self.assertEqual(reader.snapshot('can0')['joints'], [0.] * 7)

    def test_both_teaching_grippers_use_direct_opening_mapping(self):
        self.assertAlmostEqual(teach_gripper_opening(.02, 0), .02)
        self.assertAlmostEqual(teach_gripper_opening(.08, 0), .08)
        self.assertAlmostEqual(teach_gripper_opening(.02, 1), .02)
        self.assertAlmostEqual(teach_gripper_opening(.08, 1), .08)

    def test_scene_template_is_local_and_external_files_are_rejected(self):
        path = resolve_scene_usd('assets/scenes/workcell_template.usda')
        self.assertEqual(path, PROJECT / 'assets/scenes/workcell_template.usda')
        with self.assertRaises(ValueError):
            resolve_scene_usd('/tmp/other_scene.usda')

    def test_urdf_limits_are_applied_to_teaching_angles(self):
        limits = urdf_joint_limits(PROJECT / 'assets/nero/nero.urdf')
        source = [0.095, -1.785, .1, 2.189, -.059, .273, 1.691]
        clipped, limited = clip_teach_joints(source, limits)
        self.assertEqual(limited, ('joint2', 'joint4', 'joint7'))
        np.testing.assert_array_less(clipped, limits[:, 1] + 1e-9)
        np.testing.assert_array_less(limits[:, 0] - 1e-9, clipped)


if __name__ == '__main__':
    unittest.main()
