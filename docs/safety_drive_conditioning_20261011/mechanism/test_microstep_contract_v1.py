import unittest
import numpy as np
from microstep_contract_v1 import microsteps,constant_control_targets,complete_event_layout,control_fifo_targets

class ClockTests(unittest.TestCase):
    def test_original_and_refined_complete_clocks(self):
        for n in [2,4,8]:
            r=dict(control_dt_s=.016666,physics_dt_s=.016666/n,physics_substeps_per_control=n)
            self.assertEqual(microsteps(r),n)
            complete_event_layout(np.repeat(np.arange(3),n),np.tile(np.arange(n),3),3,n)
    def test_original_default(self):
        self.assertEqual(microsteps(dict(control_dt_s=.016666,physics_dt_s=.008333)),2)
    def test_invalid_clock_or_count(self):
        for n in [True,0,-1,8.0]:
            with self.assertRaises(AssertionError):microsteps(dict(control_dt_s=.016666,physics_dt_s=.00208325,physics_substeps_per_control=n))
        with self.assertRaises(AssertionError):microsteps(dict(control_dt_s=.016666,physics_dt_s=.008333,physics_substeps_per_control=8))
    def test_drop_duplicate_and_invalid_microstep_rejected(self):
        f=np.repeat(np.arange(2),8);s=np.tile(np.arange(8),2)
        for ff,ss in [(f[:-1],s[:-1]),(np.append(f,f[-1]),np.append(s,s[-1])),(f,np.where(np.arange(16)==7,8,s))]:
            with self.assertRaises(AssertionError):complete_event_layout(ff,ss,2,8)
    def test_constant_targets_and_last_microstep_tamper(self):
        for n in [2,8]:
            target=np.repeat(np.arange(6).reshape(3,1,2),n,axis=0)
            constant_control_targets(target,n)
            target[-1,0,0]+=1
            with self.assertRaises(AssertionError):constant_control_targets(target,n)

    def test_fifo_retains_six_control_cycles_under_refinement(self):
        for n in [2,8]:
            initial=np.array([[0.,1.]])
            control_targets=np.arange(40).reshape(20,1,2)
            issued=np.repeat(control_targets,n,axis=0)
            applied=np.concatenate([np.broadcast_to(initial,(6*n,1,2)),issued[:-6*n]])
            control_fifo_targets(applied,issued,initial,n,6)
            with self.assertRaises(AssertionError):control_fifo_targets(applied,issued,initial,n,5)
            applied[-1,0,0]+=1
            with self.assertRaises(AssertionError):control_fifo_targets(applied,issued,initial,n,6)

if __name__=='__main__':unittest.main()
