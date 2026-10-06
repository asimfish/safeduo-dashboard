import unittest
import numpy as np
import torch
from object_binding import pack_task_input,pickup_weight,FIELDS

class InputContract(unittest.TestCase):
    def setUp(self):
        self.state=torch.zeros(2,2,13);self.state[:,:,3]=1
        self.origins=torch.tensor([[10.,-5.,0.],[-7.,4.,0.]])
        self.state[:,:,:3]=self.origins[:,None,:]+torch.tensor([.5,.1,.84])
        self.sizes=torch.tensor([[.04,.7,.08],[.04,.3,.08]])
        self.goals=torch.tensor([[.37,0,.84],[-.5,0,.84]])
        self.valid=torch.ones(2,2,dtype=torch.bool)
    def pack(self,**kw):
        args=dict(native_states=self.state,origins=self.origins,sizes=self.sizes,goals=self.goals,state_time_s=1.,now_s=1.,valid=self.valid,max_age_s=.016667);args.update(kw)
        return pack_task_input(**args)
    def test_frame_and_data_binding(self):
        x=self.pack();self.assertEqual(x.shape,(2,2,len(FIELDS)))
        torch.testing.assert_close(x[:,:,:3],torch.tensor([.5,.1,.84]).expand(2,2,3),atol=1e-6,rtol=0)
        torch.testing.assert_close(x[0,:,13:16],self.sizes)
        torch.testing.assert_close(x[1,:,16:19],self.goals)
        changed=self.state.clone();changed[0,0,0]+=.02
        self.assertGreater(float((self.pack(native_states=changed)-x).abs().max()),.019)
    def test_invalid_states_fail_closed(self):
        for kw in [dict(now_s=1.1),dict(now_s=.9),dict(valid=torch.zeros_like(self.valid)),dict(sizes=self.sizes*0)]:
            with self.assertRaises(ValueError):self.pack(**kw)
        bad=self.state.clone();bad[0,0,3]=0
        with self.assertRaises(ValueError):self.pack(native_states=bad)
        bad=self.state.clone();bad[0,0,0]=float('nan')
        with self.assertRaises(ValueError):self.pack(native_states=bad)
    def test_pickup_only_correction_preserves_start_and_goal(self):
        for end in (10.2,11.7):
            w=pickup_weight(np.array([0.,.5,3.5,7.2,end,26.2]),end)
            np.testing.assert_allclose(w,[0,0,1,1,0,0],atol=1e-14)

if __name__=='__main__':unittest.main()
