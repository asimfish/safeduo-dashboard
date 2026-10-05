import unittest
import numpy as np
from analyze import Metric,miss_flags,row_classes
class Tests(unittest.TestCase):
    def test_physical_sign_and_changing_exemptions(self):
        actual=np.array([-.01,-.01,0.,-.01,-.01])
        pred=np.array([0.,-.001,.01,.01,.01])
        pre=np.array([False,False,False,True,False])
        post=np.array([False,False,False,False,True])
        miss,stable=miss_flags(pred,actual,pre,post)
        np.testing.assert_equal(miss,[True,False,False,True,False])
        np.testing.assert_equal(stable,[True,False,False,False,False])
    def test_weighted_streaming_signed_and_absolute(self):
        m=Metric();m.add([.001,-.003]);m.add([.002])
        r=m.result();self.assertEqual(r['values'],3)
        self.assertAlmostEqual(r['mae_m'],.002)
        self.assertAlmostEqual(r['signed_mean_m'],0)
        self.assertAlmostEqual(r['rmse_m'],np.sqrt(14e-6/3))
        self.assertEqual(r['max_m'],.003)
        self.assertGreaterEqual(r['p99_upper_m'],.003)
    def test_histogram_overflow_is_not_an_error_bound(self):
        m=Metric();m.add([.1,2.]);r=m.result()
        self.assertIsNone(r['p95_upper_m']);self.assertEqual(r['histogram_overflow'],1)
        self.assertEqual(r['max_m'],2.)
    def test_four_geometry_channels(self):
        np.testing.assert_equal(row_classes(dict(class_id=[0,1,1,2],pair_sphere_idx=[[0,1],[0,0],[1,1],[1,0]],sphere_arm_id=[0,2])),[0,1,2,3])
if __name__=='__main__':unittest.main()
