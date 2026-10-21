import hashlib
import json
import struct
import unittest
from pathlib import Path
import numpy as np

from sim_bridge.grasp import GraspConfig, TASK, grasp_pose, grasp_skill_fields
from sim_bridge.objects import OBJECTS, load_object_profile


class ObjectModelTests(unittest.TestCase):
    def test_keel_sources_and_metric_geometry(self):
        profile = load_object_profile("keel")
        for filename, digest in profile["source_sha256"].items():
            self.assertEqual(hashlib.sha256((OBJECTS / "keel" / filename).read_bytes()).hexdigest(), digest)
        data = (OBJECTS / "keel/keel.STL").read_bytes()
        count = struct.unpack_from("<I", data, 80)[0]
        vertices = np.array([struct.unpack_from("<9f", data, 84 + i * 50 + 12) for i in range(count)]).reshape(-1, 3)
        local = (vertices - profile["source_grasp_point"]) * profile["source_scale_to_m"]
        np.testing.assert_allclose(np.ptp(local, axis=0), [.15, .098, .023])
        self.assertAlmostEqual(local[:, 2].min(), -profile["origin_height_m"])
        self.assertTrue(Path(profile["asset_path"]).is_file())
        self.assertTrue(profile["units_confirmed_by_user"])

    def test_object_config_roundtrip_and_table_clearance(self):
        config = GraspConfig.for_object("keel")
        self.assertEqual(config, GraspConfig.from_json(config.to_json()))
        self.assertEqual(config.object_name, "keel")
        self.assertAlmostEqual(config.cube_size, .038)
        self.assertAlmostEqual(config.cube_mass, .08)
        target = np.array([.3, 0., config.table_height + config.support_height, 1., 0., 0., 0.])
        pose = grasp_pose(target)
        # The finger's furthest tip clears the 10 mm-high base protrusions.
        self.assertGreater(pose[2] - .1358 - config.table_height, .010)
        old = json.loads(GraspConfig().to_json())
        old.pop("object_name")
        old.pop("object_origin_height")
        self.assertEqual(GraspConfig.from_json(json.dumps(old)), GraspConfig())
        self.assertAlmostEqual(GraspConfig().support_height, .015)

    def test_object_names_and_mixed_object_training(self):
        self.assertIsNone(load_object_profile("cube"))
        with self.assertRaises(ValueError):
            load_object_profile("../keel")
        with self.assertRaises(ValueError):
            load_object_profile("missing_model")
        recordings = []
        for name in ("cube", "keel"):
            rows = np.zeros((3, 8)); rows[:, 0] = [0, 5, 10]; rows[:, 4] = 1
            recordings.append(dict(task=TASK, grasp_config=GraspConfig.for_object(name).to_json(),
                grasp_duration=10., grasp_success=True, demo=rows, gripper_commands=np.array([.08, 0, 0])))
        with self.assertRaisesRegex(ValueError, "share geometry"):
            grasp_skill_fields(recordings, 10)


if __name__ == "__main__":
    unittest.main()
