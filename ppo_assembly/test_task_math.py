import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from assembly_collection.geometry import pose,wall_targets,insertion_metrics,keel_hole_metrics
from ppo_assembly.task_math import bolt_metrics,centered,retained,axis_quat,qmul,apply
from ppo_assembly.position_force import axial_velocity,force_costs


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.wall=pose(np.array([-.47,0.,1.]),Rotation.from_euler('z',np.pi/2))
        self.keel,self.holes,self.bolts,self.normal=wall_targets(self.wall)

    def metrics(self,bolt,keel=None):
        t=lambda a:torch.tensor(np.array(a)[None],dtype=torch.float64)
        return bolt_metrics(t(bolt),t(self.keel if keel is None else keel),t(self.holes[3]),t(self.normal))

    def test_batched_metrics_match_independent_geometry(self):
        for displacement in (0.,.00015,.0008):
            bolt=self.bolts[3].copy();bolt[2]+=displacement
            m=self.metrics(bolt)
            ref=insertion_metrics(bolt,self.holes[3],self.normal)
            k=keel_hole_metrics(bolt,self.keel,3,.005)
            for name,key in [('depth','depth_m'),('radial','radial_error_m'),('entry','entry_radial_error_m'),('axis_error','axis_error_rad')]:
                self.assertAlmostEqual(m[name].item(),ref[key],places=7)
            self.assertAlmostEqual(m['keel_radial'].item(),k['keel_hole_radial_error_m'],places=7)
            self.assertEqual(m['crosses'].item(),k['keel_shaft_crosses_plate'])

    def test_success_requires_actual_keel_hole(self):
        self.assertTrue(centered(self.metrics(self.bolts[3])).item())
        wrong=self.keel.copy();wrong[2]+=.005
        self.assertFalse(centered(self.metrics(self.bolts[3],wrong)).item())
        self.assertFalse(retained(self.metrics(self.bolts[3],wrong)).item())

    def test_not_inserted_cannot_pass(self):
        bolt=self.bolts[3].copy();bolt[:3]+=.04*self.normal
        self.assertFalse(centered(self.metrics(bolt)).item())
        self.assertFalse(retained(self.metrics(bolt)).item())

    def test_small_rotation_and_composition(self):
        a=axis_quat(torch.tensor([[0.,0.,.03]],dtype=torch.float64))
        b=axis_quat(torch.tensor([[.02,0.,0.]],dtype=torch.float64))
        v=torch.tensor([[1.,2.,3.]],dtype=torch.float64)
        expected=(Rotation.from_rotvec([0,0,.03])*Rotation.from_rotvec([.02,0,0])).apply([1,2,3])
        np.testing.assert_allclose(apply(qmul(a,b),v).numpy()[0],expected,atol=1e-12)


class PositionForceTests(unittest.TestCase):
    def velocity(self,requested,force,contact):
        return axial_velocity(torch.tensor([requested]),torch.tensor([force]),torch.tensor([7.5]),
            torch.tensor([contact]),.008,.004,2.).item()

    def test_no_force_chasing_before_contact(self):
        self.assertEqual(self.velocity(0.,[0.,0.,0.],False),0.)
        self.assertAlmostEqual(self.velocity(.004,[0.,0.,0.],False),.004)

    def test_high_axial_force_causes_withdrawal(self):
        self.assertLess(self.velocity(.004,[9.,0.,0.],True),0.)
        self.assertGreater(self.velocity(.004,[6.,0.,0.],True),0.)
        self.assertLess(self.velocity(-.004,[1.,0.,0.],True),0.)

    def test_large_lateral_force_prevents_forward_push(self):
        self.assertLess(self.velocity(.004,[1.,6.,0.],True),0.)

    def test_force_cost_is_not_contact_bonus(self):
        force=torch.tensor([[0.,0.,0.],[7.5,0.,0.],[10.,3.,0.]])
        values=force_costs(force,force,torch.full((3,),7.5),torch.tensor([False,True,True]),
            torch.ones(3,dtype=torch.bool),8.,5.,2.)
        self.assertEqual(values['tracking'][0].item(),0.)
        self.assertEqual(values['tracking'][1].item(),0.)
        self.assertGreater(values['overload'][2].item(),values['overload'][1].item())
        self.assertGreater(values['lateral'][2].item(),values['lateral'][1].item())
        self.assertTrue(all((v>=0.).all().item() for v in values.values()))


if __name__=='__main__':unittest.main()
