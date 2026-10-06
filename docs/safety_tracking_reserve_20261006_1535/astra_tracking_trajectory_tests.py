"""Synthetic tests for exact byte identity; no campaign inputs."""
import unittest
import numpy as np
from astra_tracking_trajectory_identity import compare

class Tests(unittest.TestCase):
    def test_signed_zero_changes_bits_despite_equal_numbers(self):
        a=np.zeros((2,3,4),np.float32);b=a.copy();b[0,1,2]=-0.
        self.assertTrue(np.array_equal(a,b))
        r=compare(a,b,1)
        self.assertFalse(r['all_bits_equal']);self.assertEqual(r['differing_env_ids'],[1])
    def test_fifo_env_axis_and_adjacent_float_detected(self):
        a=np.ones((2,6,3,4),np.float32);b=a.copy()
        self.assertTrue(compare(a,b,2)['all_bits_equal'])
        b[-1,-1,2,-1]=np.nextafter(np.float32(1),np.float32(2))
        self.assertEqual(compare(a,b,2)['differing_env_ids'],[2])
    def test_nonfinite_dtypes_and_absent_class_handling(self):
        a=np.ones((2,3,4),np.float32)
        for value in [np.nan,-np.inf,np.inf]:
            b=a.copy();b[0,0,0]=value
            with self.assertRaises(ValueError):compare(b,b,1)
        a[0,0,0]=np.inf
        self.assertTrue(compare(a,a,1,True)['all_bits_equal'])
        with self.assertRaises(ValueError):compare(a,a.astype(np.float64),1,True)
        with self.assertRaises(ValueError):compare(a,a[:1],1,True)
    def test_boolean_deep_flags(self):
        a=np.zeros((2,3,4),bool);b=a.copy();b[1,2,0]=True
        self.assertEqual(compare(a,b,1)['differing_env_ids'],[2])

if __name__=='__main__':unittest.main(verbosity=2)
