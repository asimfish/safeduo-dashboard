import unittest,json
from pathlib import Path
import numpy as np
import torch
from object_binding_v3 import bind_reference

class ReferenceInvariants(unittest.TestCase):
    def fixture(self):
        path=Path('/home/liyufeng/safeduo/artifacts/forensics/20261004_randomized_safety_campaign_v4/block_00/task_two_pair_pdz_uax07.npz')
        with np.load(path,allow_pickle=False) as z:q={k[2:]:z[k][[0,210,630]].copy() for k in z.files if k.startswith('q_')}
        x=torch.zeros(1,2,21);x[0,0,:3]=torch.tensor([.55,.02,.84]);x[0,0,3]=1
        x[0,1,:3]=torch.tensor([-.42,-.02,.84]);x[0,1,3]=x[0,1,6]=2**-.5
        return q,x
    def test_original_q0_and_final_pose_are_bitexact(self):
        q,x=self.fixture();bound,audit=bind_reference(q,7.,x,torch.tensor([True]),[[.53,0,.84],[-.4,0,.84]],[0,np.pi/2])
        for arm in q:
            self.assertTrue(np.array_equal(bound[arm][0,0],q[arm][0]))
            self.assertTrue(np.array_equal(bound[arm][0,-1],q[arm][-1]))
            self.assertGreater(float(np.abs(bound[arm][0,1]-q[arm][1]).max()),1e-4)

if __name__=='__main__':unittest.main()
