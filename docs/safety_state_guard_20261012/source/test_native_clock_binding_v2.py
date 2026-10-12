import unittest
from native_clock_binding_v2 import validate_native_clock


class ClockBindingContracts(unittest.TestCase):
    def registration(self,n=16):
        return dict(physics_dt_s=1/(60*n),control_dt_s=1/60,physics_substeps_per_control=n,position_target_FIFO_controls=6)

    def test_both_registered_native_profiles_bind_exact_fifo_duration(self):
        for n in (16,32):
            r=validate_native_clock(self.registration(n),60*n,1/(60*n))
            self.assertEqual(r['FIFO_s'],.1);self.assertEqual(r['physics_substeps_per_control'],n)

    def test_observed_old_cfg_quantization_is_still_rejected(self):
        r=self.registration();r.update(physics_dt_s=.001041625,control_dt_s=.016666)
        with self.assertRaisesRegex(AssertionError,'Registered dt'):validate_native_clock(r,960,1/960)

    def test_getter_scene_disagreement_still_rejected(self):
        with self.assertRaisesRegex(AssertionError,'getter and USD'):validate_native_clock(self.registration(32),960,1/1920)

    def test_new_metadata_cannot_claim_old_runtime_rate(self):
        with self.assertRaisesRegex(AssertionError,'Registered dt'):validate_native_clock(self.registration(32),960,1/960)

    def test_control_alias_and_wrong_fifo_rejected(self):
        r=self.registration(32);r['control_dt_s']=.016666
        with self.assertRaisesRegex(AssertionError,'Control clock'):validate_native_clock(r,1920,1/1920)
        r=self.registration(32);r['position_target_FIFO_controls']=5
        with self.assertRaises(AssertionError):validate_native_clock(r,1920,1/1920)

    def test_unsupported_microstep_profile_rejected(self):
        with self.assertRaises(AssertionError):validate_native_clock(self.registration(64),3840,1/3840)


if __name__=='__main__':unittest.main(verbosity=2)
