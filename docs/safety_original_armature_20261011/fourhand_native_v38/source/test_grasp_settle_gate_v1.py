import unittest
from grasp_settle_gate_v1 import GraspSettleGate


class NativePhaseAdmissionTests(unittest.TestCase):
    def sample(self, g, t, **kw):
        args=dict(time_s=t, elapsed_s=1.5+t, nominal_done=True,
                  position_error=.001, speed=.001, applied_error=0.,
                  physical_safe=True, fresh=True)
        args.update(kw)
        return g.observe(**args)

    def test_requires_continuous_dwell_not_elapsed_motion(self):
        g=GraspSettleGate(physics_dt_s=.01)
        self.assertFalse(self.sample(g,0.)['ready'])
        for i in range(1,30): self.assertFalse(self.sample(g,i*.01)['ready'])
        self.assertTrue(self.sample(g,.3)['ready'])

    def test_measured_moving_delayed_or_nominal_incomplete_reset(self):
        for kw in [dict(speed=.281),dict(position_error=.056),
                   dict(applied_error=2e-6),dict(nominal_done=False),
                   dict(physical_safe=False),dict(fresh=False)]:
            g=GraspSettleGate(physics_dt_s=.01)
            for i in range(20):self.sample(g,i*.01)
            self.assertFalse(self.sample(g,.2,**kw)['valid'])
            self.assertFalse(self.sample(g,.21)['ready'])

    def test_gap_duplicate_and_nonfinite_never_count_as_dwell(self):
        for t in [.19,.25,float('nan')]:
            g=GraspSettleGate(physics_dt_s=.01)
            for i in range(20):self.sample(g,i*.01)
            self.assertFalse(self.sample(g,t)['valid'])
            self.assertIsNone(g.since)

    def test_timeout_without_relaxing_threshold(self):
        g=GraspSettleGate()
        r=self.sample(g,0.,elapsed_s=8.,position_error=.006)
        self.assertTrue(r['timed_out']);self.assertFalse(r['ready'])


if __name__=='__main__':unittest.main()
