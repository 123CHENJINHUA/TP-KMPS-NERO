"""Run with the tpkmp environment: python -m unittest assembly_collection.test_geometry."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from assembly_collection.geometry import compose, inverse, pose, rotation, wall_targets, HOLES, insertion_metrics, keel_hole_metrics, keel_grasp, FINGER_CENTER, screw_order, canonical_screw_target, axial_alignment_target


class GeometryTests(unittest.TestCase):
    def test_contact_alignment_keeps_bolt_roll_free(self):
        wanted=pose([-.5,.1,1.2],Rotation.from_euler('y',np.pi/2))
        for roll in (-2.,0.,.5,2.5):
            for tilt in (-.02,.01):
                actual=pose([-.49,.11,1.19],rotation(wanted)*Rotation.from_euler('z',roll)*Rotation.from_euler('y',tilt))
                aligned=axial_alignment_target(wanted,actual)
                np.testing.assert_allclose(aligned[:3],wanted[:3],atol=1e-12)
                expected=rotation(wanted)*Rotation.from_euler('z',roll)
                self.assertLess((rotation(aligned)*expected.inv()).magnitude(),1e-10)
        opposite=pose([0,0,0],rotation(wanted)*Rotation.from_euler('x',np.pi))
        np.testing.assert_allclose(rotation(axial_alignment_target(wanted,opposite)).apply([0,0,1]),
            rotation(wanted).apply([0,0,1]),atol=1e-12)

    def test_bearing_neck_uses_its_actual_hole_clearance(self):
        wall=pose([-.5,0,1.],Rotation.from_euler('z',np.pi/2))
        keel,_,bolts,_=wall_targets(wall)
        self.assertAlmostEqual(keel_hole_metrics(bolts[0],keel,0,.005)['keel_hole_clearance_m'],.001)
        self.assertAlmostEqual(keel_hole_metrics(bolts[3],keel,3,.005)['keel_hole_clearance_m'],.00025)
        keel[2]-=.0005
        shaft=keel_hole_metrics(bolts[3],keel,3)
        neck=keel_hole_metrics(bolts[3],keel,3,.005)
        self.assertLess(shaft['keel_hole_radial_error_m'],shaft['keel_hole_clearance_m'])
        self.assertGreater(neck['keel_hole_radial_error_m'],neck['keel_hole_clearance_m'])

    def test_rack_spin_does_not_clock_gripper_at_wall(self):
        wall=pose([-.5,0,1.],Rotation.from_euler('z',np.pi/2))
        _,_,bolts,normal=wall_targets(wall)
        tool=pose([0,0,.16],Rotation.from_euler('z',np.pi/2)*Rotation.from_euler('x',np.pi))
        for spin in (-2.,-.3,0.,.7,2.5):
            relative=compose(inverse(pose([0,0,0],Rotation.from_euler('z',spin))),tool)
            target=canonical_screw_target(bolts[3],relative)
            np.testing.assert_allclose(rotation(target).apply([0,0,1]),normal,atol=1e-12)
            np.testing.assert_allclose(rotation(compose(target,relative)).apply([0,1,0]),[0,0,-1],atol=1e-12)

    def test_right_column_is_complete_before_left_column(self):
        self.assertEqual(screw_order(),[4,3,1,2])
        self.assertEqual(screw_order(3,2),[3,4])
        for first in (3,4):
            order=screw_order(first)
            self.assertTrue(np.all(HOLES[np.array(order[:2])-1,0]>0))
            self.assertTrue(np.all(HOLES[np.array(order[2:])-1,0]<0))
        with self.assertRaises(ValueError):screw_order(1)

    def test_pose_round_trip(self):
        a=pose([.3,-.2,.7],Rotation.from_euler('xyz',[.4,-.3,1.2]))
        np.testing.assert_allclose(compose(a,inverse(a))[:3],0,atol=1e-12)
        self.assertLess(rotation(compose(a,inverse(a))).magnitude(),1e-12)

    def test_tilted_grasp_tracks_rotated_keel(self):
        for yaw in (0.,np.pi/2,-.5):
            obj=pose([-.4,.02,1.1],Rotation.from_euler('z',yaw))
            for tilt in (0.,np.deg2rad(-65)):
                tool=keel_grasp(obj,.015,tilt)
                np.testing.assert_allclose(tool[:3]+rotation(tool).apply(FINGER_CENTER),
                    obj[:3]+rotation(obj).apply([.015,0,0]),atol=1e-12)
                np.testing.assert_allclose(rotation(tool).apply([0,1,0]),
                    rotation(obj).apply([0,-1,0]),atol=1e-12)

    def test_keel_holes_match_rotated_wall(self):
        for yaw in (0.,np.pi/2,-.3):
            wall=pose([-.62,.04,.98574],Rotation.from_euler('z',yaw))
            keel,holes,bolts,normal=wall_targets(wall)
            actual=keel[:3]+rotation(keel).apply(HOLES)
            np.testing.assert_allclose(actual-holes,np.tile(normal*.0019,(4,1)),atol=1e-12)
            self.assertAlmostEqual(np.linalg.norm(holes[0]-holes[2]),.123)
            self.assertAlmostEqual(np.linalg.norm(holes[0]-holes[1]),.068)
            for screw,hole in zip(bolts,holes):
                m=insertion_metrics(screw,hole,normal)
                self.assertAlmostEqual(m['depth_m'],.0243)
                self.assertLess(m['radial_error_m'],1e-12)
                self.assertLess(m['axis_error_rad'],1e-7)

    def test_wrong_depth_and_lateral_offset_detected(self):
        wall=pose([-.62,0,1.],Rotation.from_euler('z',np.pi/2))
        _,holes,bolts,normal=wall_targets(wall)
        bad=bolts[0].copy();bad[1]+=.002
        self.assertAlmostEqual(insertion_metrics(bad,holes[0],normal)['radial_error_m'],.002)
        bad=bolts[0].copy();bad[:3]+=.04*normal
        self.assertLess(insertion_metrics(bad,holes[0],normal)['depth_m'],0)

    def test_actual_keel_must_stay_around_shaft(self):
        wall=pose([-.62,0,1.],Rotation.from_euler('z',np.pi/2))
        keel,_,bolts,_=wall_targets(wall)
        for index in range(4):
            m=keel_hole_metrics(bolts[index],keel,index)
            self.assertTrue(m['keel_shaft_crosses_plate'])
            self.assertLess(m['keel_hole_radial_error_m'],1e-12)
        displaced=keel.copy();displaced[1]+=.003
        m=keel_hole_metrics(bolts[3],displaced,3)
        self.assertGreater(m['keel_hole_radial_error_m'],m['keel_hole_clearance_m'])

if __name__=='__main__':unittest.main()
