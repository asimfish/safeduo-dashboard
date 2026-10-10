"""Software regression; no physics or safety acceptance is implied."""
import unittest
import numpy as np
from feedback_grip_v1 import FeedbackGrip


class FeedbackGripTests(unittest.TestCase):
    def setUp(self):
        self.dt=.016666;self.g=FeedbackGrip(self.dt,{'hand':(.55,.55)})
        self.q=np.full(12,.2);self.v=np.zeros(12);self.lim=np.tile([0.,1.],(12,1));self.mask=np.arange(12)>=8

    def update(self,n,force=0.,f=.08,t=.325,q=None,v=None,**kw):
        return self.g.update('hand',n,n*self.dt,n*self.dt,self.q if q is None else q,self.v if v is None else v,self.lim,self.mask,f,t,force,**kw)

    def test_seek_is_bounded_and_continuous(self):
        d=self.update(0);self.assertEqual(d.mode,'SEEK')
        self.assertAlmostEqual(d.finger_fraction,.08+.15*self.dt);self.assertAlmostEqual(d.thumb_fraction,.325+.15*self.dt)
        self.assertEqual(self.update(1,f=.55,t=.55).finger_fraction,.55)

    def test_force_hysteresis_and_resume(self):
        self.assertEqual(self.update(0,4.1).mode,'HOLD')
        d=self.update(1,3.);self.assertEqual((d.finger_fraction,d.thumb_fraction),(.08,.325))
        self.assertEqual(self.update(2,2.4).mode,'SEEK')

    def test_thumb_guard_keeps_other_group_from_advancing(self):
        q=self.q.copy();v=self.v.copy();q[9]=.08;v[9]=-.75
        d=self.update(0,q=q,v=v);self.assertTrue(d.thumb_guard);self.assertFalse(d.finger_guard)
        self.assertEqual(d.finger_fraction,.08);self.assertLess(d.thumb_fraction,.325)

    def test_actual_position_drift_overrides_positive_velocity_snapshot(self):
        self.update(0,q=np.full(12,.16))
        q=np.full(12,.16);q[9]=.13;v=np.full(12,.1)
        d=self.update(1,q=q,v=v);self.assertTrue(d.thumb_guard);self.assertEqual(d.mode,'JOINT_GUARD')

    def test_stationary_open_joints_are_allowed_and_high_force_relaxes(self):
        q=np.full(12,.03);d=self.update(0,q=q);self.assertEqual(d.mode,'SEEK')
        d=self.update(1,7.,q=q);self.assertEqual(d.mode,'RELAX');self.assertLess(d.finger_fraction,.08)

    def test_invalid_feedback_cannot_update_history(self):
        self.update(0)
        with self.assertRaises(ValueError):self.update(0)
        with self.assertRaises(ValueError):self.update(4)
        q=self.q.copy();q[2]=np.nan
        with self.assertRaises(ValueError):self.update(1,q=q)
        self.assertEqual(self.g.previous['hand'][0],0)

    def test_random_software_transitions_never_jump_or_exceed_stroke(self):
        rng=np.random.default_rng(20261011);f,t=.08,.325
        for i in range(400):
            q=rng.uniform(.03,.9,12);v=rng.uniform(-1,1,12);force=float(rng.uniform(0,8))
            d=self.update(i,force,f,t,q,v)
            self.assertTrue(0<=d.finger_fraction<=.55 and 0<=d.thumb_fraction<=.55)
            self.assertLessEqual(abs(d.finger_fraction-f),.75*self.dt+1e-12)
            self.assertLessEqual(abs(d.thumb_fraction-t),.75*self.dt+1e-12)
            f,t=d.finger_fraction,d.thumb_fraction


if __name__=='__main__':unittest.main()
