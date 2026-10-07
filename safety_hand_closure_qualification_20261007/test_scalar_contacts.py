import unittest,torch,numpy as np
from contact_points import scalar_contacts
class View:
 def __init__(self,forces,count,start):self.values=(torch.tensor(forces,dtype=torch.float32)[:,None],None,None,None,torch.tensor(count),torch.tensor(start))
 def get_contact_data(self,dt):return self.values
class ScalarContract(unittest.TestCase):
 def test_opposed_points_do_not_cancel(self):
  # Opposed normals would cancel their vector resultant, while total normal is20N.
  v=View([10,10,999],[[2]],[[0]]);scalar,count,mx,force,start=scalar_contacts(v,.008333,3);self.assertEqual(scalar[0,0],20);self.assertEqual(mx,10)
 def test_unused_buffer_nonfinite_ignored(self):
  scalar,count,mx,force,start=scalar_contacts(View([np.nan,999],[[0]],[[-1]]),.008333,2);self.assertEqual(scalar[0,0],0);self.assertEqual(mx,0)
 def test_live_force_nonfinite_rejected(self):
  with self.assertRaises(AssertionError):scalar_contacts(View([np.nan,0],[[1]],[[0]]),.008333,2)
 def test_count_outside_capacity_rejected(self):
  with self.assertRaises(AssertionError):scalar_contacts(View([1,2],[[1]],[[2]]),.008333,2)
if __name__=='__main__':unittest.main(verbosity=2)
