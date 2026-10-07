"""Independent failure contract tests. No native physical-safety claim."""
import unittest,numpy as np
from hand_admission import HandAdmission
P=dict(open_rad={'U_L':[.35,0.]},limits_rad={'U_L':[[0,1.31],[0,.524]]},allowed_goals_rad={'U_L':[[.6,.2]]},max_speed_rad_s=3.,max_self_normal_n=.1,max_start_error_rad=.02,ramp_s=1.,registered_reference_times_s=[0.,2.],context_exclusions=[dict(arm='U_L',reference_time_s=2.,goal_rad=[.6,.2])])
def state(**kw):return dict(q=[.35,0.],qd=[0.,0.],self_normal_n=[0.],unintended_normal_n=[0.],step=9,**kw) if not kw else {**state(),**kw}
class Contract(unittest.TestCase):
 def gate(self):return HandAdmission(P,'U_L')
 def req(self,g,goal=[.6,.2],s=None,ramp=1.,ref=0.):return g.request(goal,ramp,ref,s or state(),10)
 def test_external_contact_rejects_even_without_self_contact(self):self.assertEqual(self.req(self.gate(),s=state(unintended_normal_n=[.11])).reason,'unintended_contact')
 def test_context_rejection_does_not_globally_delete_goal(self):
  self.assertEqual(self.req(self.gate(),ref=2.).reason,'known_unsafe_context');self.assertTrue(self.req(self.gate(),ref=0.).admitted)
 def test_external_nonfinite(self):self.assertEqual(self.req(self.gate(),s=state(unintended_normal_n=[np.nan])).reason,'nonfinite_state')
 def test_qualified_exact_target(self):self.assertTrue(self.req(self.gate()).admitted)
 def test_unknown_no_interpolation(self):self.assertEqual(self.req(self.gate(),[.61,.2]).reason,'unqualified_exact_target')
 def test_target_bounds(self):self.assertEqual(self.req(self.gate(),[1.4,.2]).reason,'target_bounds')
 def test_target_nonfinite(self):self.assertEqual(self.req(self.gate(),[np.nan,.2]).reason,'invalid_target')
 def test_different_duration(self):self.assertEqual(self.req(self.gate(),ramp=.5).reason,'unqualified_path_duration')
 def test_unregistered_arm_pose(self):self.assertEqual(self.req(self.gate(),ref=6.).reason,'unregistered_arm_reference')
 def test_non_neutral_start(self):self.assertEqual(self.req(self.gate(),s=state(q=[.5,0.])).reason,'unqualified_start')
 def test_state_stale(self):self.assertEqual(self.req(self.gate(),s=state(step=8)).reason,'stale_state')
 def test_state_nonfinite(self):self.assertEqual(self.req(self.gate(),s=state(qd=[0.,np.inf])).reason,'nonfinite_state')
 def test_speed(self):self.assertEqual(self.req(self.gate(),s=state(qd=[0.,3.1])).reason,'speed_limit')
 def test_contact(self):self.assertEqual(self.req(self.gate(),s=state(self_normal_n=[.11])).reason,'self_contact')
 def test_missing_state(self):self.assertEqual(self.req(self.gate(),s={'step':9}).reason,'missing_or_invalid_state')
 def test_abort_latches_after_admission(self):
  g=self.gate();self.assertTrue(self.req(g).admitted)
  self.assertTrue(g.observe(state(self_normal_n=[1.]),10).aborted)
  self.assertEqual(self.req(g).reason,'abort_latched')
 def test_boundary_contact_passes(self):self.assertTrue(self.req(self.gate(),s=state(self_normal_n=[.1])).admitted)
 def test_float_perturbation_rejected(self):self.assertFalse(self.req(self.gate(),[float(np.nextafter(np.float32(.6),np.float32(1))),.2]).admitted)
if __name__=='__main__':unittest.main(verbosity=2)
