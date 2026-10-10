import unittest
import numpy as np
from coupled_hand_targets_v1 import CoupledHand,Relation
class CouplingTests(unittest.TestCase):
    def test_negative_offset_forces_feasible_motor_opening(self):
        h=CoupledHand(['motor','distal'],[[0,1.34],[0,1.59]],[Relation('distal','motor',1.1169,-.15)])
        q,_=h.reference_endpoints([.03,.03],.03)
        self.assertAlmostEqual(q[0],.18/1.1169);self.assertAlmostEqual(q[1],.03)
    def test_chained_thumb_reserve_and_upper_limit(self):
        h=CoupledHand(['m','s','d'],[[0,.7854],[0,.567232],[0,.3957]],[Relation('s','m',.7222,0),Relation('d','s',.69754,0)])
        lo,hi=h.reference_endpoints([.03]*3,.03)
        self.assertAlmostEqual(lo[2],.03);self.assertTrue((lo>=.03-1e-12).all());self.assertTrue((hi<=h.limits[:,1]-.03+1e-12).all())
    def test_negative_multiplier_and_nonzero_offset(self):
        h=CoupledHand(['m','s'],[[0,1],[-1,1]],[Relation('s','m',-2,.5)])
        bounds=h.feasible_motor_intervals();np.testing.assert_allclose(bounds,[[0,.75]])
        np.testing.assert_allclose(h.expand([.5]),[.5,-.5])
    def test_random_motor_support_and_interpolation_preserve_every_relation(self):
        h=CoupledHand(['m','s','d'],[[0,.523477],[0,.606844],[0,.430866]],[Relation('s','m',1.1425,0),Relation('d','s',.7508,.1)])
        rng=np.random.default_rng(20261011);bounds=h.feasible_motor_intervals()
        q=h.expand(rng.uniform(bounds[:,0],bounds[:,1],(1000,1)))
        self.assertTrue((q>=h.limits[:,0]-1e-12).all() and (q<=h.limits[:,1]+1e-12).all())
        np.testing.assert_allclose(q[:,1],q[:,0]*1.1425,atol=1e-15);np.testing.assert_allclose(q[:,2],q[:,1]*.7508+.1,atol=1e-15)
        opened,far=h.reference_endpoints([.03]*3);q=opened+np.linspace(0,1,101)[:,None]*(far-opened)
        np.testing.assert_allclose(q[:,2],q[:,1]*.7508+.1,atol=1e-15)
    def test_cycles_and_infeasible_offset_are_rejected(self):
        with self.assertRaises(AssertionError):CoupledHand(['a','b'],[[0,1]]*2,[Relation('a','b',1,0),Relation('b','a',1,0)])
        h=CoupledHand(['a','b'],[[0,1]]*2,[Relation('b','a',1,2)])
        with self.assertRaises(AssertionError):h.feasible_motor_intervals()
if __name__=='__main__':unittest.main()
