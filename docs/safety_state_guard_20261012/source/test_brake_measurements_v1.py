import unittest
import numpy as np
from brake_measurements_v1 import settling_window


class BrakeObserverTests(unittest.TestCase):
    def test_temporary_stop_then_motion_does_not_pass(self):
        t=np.arange(11)*.05
        self.assertFalse(settling_window(t,[.01]*5+[.1]+[.01]*5,0)['settled'])

    def test_speed_equal_cap_is_allowed_but_full_dwell_required(self):
        t=np.arange(10)*.05
        result=settling_window(t,[.1]*3+[.02]*7,.1)
        self.assertTrue(result['settled'])
        self.assertAlmostEqual(result['confirmation_latency_s'],.35)

    def test_pretrigger_low_speed_cannot_count(self):
        t=np.arange(10)*.05
        self.assertFalse(settling_window(t,[0.]*10,.2)['settled'])

    def test_nonfinite_or_reset_physical_clock_rejected(self):
        for t,v in [([0,.1,0],[0,0,0]),([0,.1],[0,float('nan')])]:
            with self.assertRaises(ValueError):settling_window(t,v,0)


if __name__=='__main__':unittest.main()
