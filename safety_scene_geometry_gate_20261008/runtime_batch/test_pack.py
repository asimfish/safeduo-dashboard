import unittest
import numpy as np
from batch_contacts import pack

class NativePackingContract(unittest.TestCase):
    def test_reordered_sensor_rows_preserve_signed_native_points(self):
        force=np.array([[99],[2],[-2],[np.nan],[7],[-5]],np.float32);normal=np.arange(18,dtype=np.float32).reshape(6,3);count=np.array([[2,0],[0,2]]);start=np.array([[1,999],[999,4]])
        packed=pack([force,normal,count,start],[1,0],8)
        np.testing.assert_array_equal(packed[-2],count[[1,0]])
        for i,src in enumerate([1,0]):
            for j in range(2):
                n=count[src,j]
                if n:
                    a=packed[-1][i,j];b=start[src,j];np.testing.assert_array_equal(packed[0][a:a+n],force[b:b+n]);np.testing.assert_array_equal(packed[1][a:a+n],normal[b:b+n])
        self.assertEqual(float(abs(packed[0]).sum()),16)
        self.assertTrue(np.isfinite(packed[0]).all())
    def test_capacity_and_invalid_live_offsets_refuse(self):
        force=np.zeros((4,1));count=np.array([[4]]);start=np.array([[0]])
        with self.assertRaises(AssertionError):pack([force,count,start],[0],4)
        with self.assertRaises(AssertionError):pack([force,np.array([[1]]),np.array([[5]])],[0],4)

if __name__=='__main__':unittest.main()
