import unittest,json
from pathlib import Path
from types import SimpleNamespace
import torch
import numpy as np
from safeduo.delta.skill_replay import SkillTrajectory,SkillReplayDelta,SkillNoiseParams
from clearance_gate import ClearanceGate,zero_held_command
P=dict(release_start_s=15.2,clearance_start_s=18.7,support_normal_min_n=.1,bottom_abs_max_m=.006,settled_speed_max_m_s=.03,settled_angular_max_rad_s=.3,tilt_max_deg=10.,hand_normal_max_n=.1,open_error_max_rad=.2,stable_steps=6,max_wait_s=2.,reaction_speed_max_m_s=.1,reaction_angular_max_rad_s=1.,reaction_steps=3)
def good(n=1):
    return dict(table_n=torch.ones(n,2),bottom_delta_m=torch.zeros(n,2),speed_m_s=torch.zeros(n,2),angular_rad_s=torch.zeros(n,2),tilt_deg=torch.zeros(n,2),hand_n=torch.zeros(n,2,2),hand_open_max_error_rad=torch.zeros(n,4))
class GateTests(unittest.TestCase):
    def gate(self):
        g=ClearanceGate(['clearance_feedback'],.016666,P);g.phase[:]=18.69;return g
    def test_freezing_clock_alone_is_red(self):
        tr=SkillTrajectory.load('/home/liyufeng/safeduo/artifacts/safety_object_binding_20261006/task_qualified_two_pair.npz')
        b=SkillReplayDelta(2,[tr],params=SkillNoiseParams.tier(0),auto_advance=False,device='cpu')
        b.set_time(19.2);state=SimpleNamespace(dt=.016666)
        first=b.sample(state);second=b.sample(state)
        self.assertGreater(max(v.abs().max().item() for v in second.delta_q.values()),0)
        for a in first.delta_q:torch.testing.assert_close(first.delta_q[a],second.delta_q[a],rtol=0,atol=0)
        second._r37_nominal={a:v.clone() for a,v in second.delta_q.items()}
        zero_held_command(second,torch.tensor([True,False]))
        for a,v in second.delta_q.items():
            self.assertEqual(v[0].abs().max().item(),0);torch.testing.assert_close(v[1],first.delta_q[a][1],atol=0,rtol=0)
            self.assertEqual(second._r37_nominal[a][0].abs().max().item(),0)
    def test_six_fresh_stable_states_required(self):
        g=self.gate();x=good()
        for _ in range(5):self.assertTrue(g.step(x).item());self.assertEqual(g.stage.item(),1)
        g.step(x);self.assertEqual(g.stage.item(),2)
        old=g.phase.clone();self.assertFalse(g.step(x).item());self.assertGreater(g.phase.item(),old.item())
    def test_timeout_is_abort_not_success(self):
        for key in ('table_n','bottom_delta_m','speed_m_s','angular_rad_s','tilt_deg','hand_n','hand_open_max_error_rad'):
            g=self.gate();x=good();x[key][:]=20 if key!='table_n' else 0
            for _ in range(122):held=g.step(x)
            self.assertEqual(g.stage.item(),3,key);self.assertEqual(g.reason.item(),1,key);self.assertTrue(held.item());self.assertAlmostEqual(g.phase.item(),18.7)
    def test_recontact_after_admission_stops(self):
        g=self.gate();x=good()
        for _ in range(6):g.step(x)
        x['hand_n'][0,1,0]=.11
        for _ in range(2):self.assertFalse(g.step(x).item())
        self.assertTrue(g.step(x).item());self.assertEqual(g.reason.item(),2)
    def test_nonfinite_is_fail_closed(self):
        g=self.gate();x=good();x['speed_m_s'][0,0]=float('nan');self.assertTrue(g.step(x).item());self.assertEqual(g.reason.item(),3)
    def test_clone_independence_and_stationary_control(self):
        g=ClearanceGate(['scheduled_debt','stationary_release','clearance_feedback'],.016666,P);g.phase[:]=15.3
        self.assertEqual(g.step(good(3)).tolist(),[False,True,True]);self.assertTrue((g.phase>15.3).all())
if __name__=='__main__':unittest.main(verbosity=2)
