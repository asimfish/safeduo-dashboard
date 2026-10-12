import unittest
from emergency_brake_latch_v1 import EmergencyBrakeLatch


class BrakeContracts(unittest.TestCase):
    def test_unqualified_candidate_latches_until_episode_end(self):
        b=EmergencyBrakeLatch();self.assertFalse(b.update(14,'SCALED'))
        self.assertTrue(b.update(15,'UNQUALIFIED_BRAKE'))
        self.assertTrue(b.update(24,'REQUEST'));self.assertEqual(b.trigger_control,15)

    def test_valid_sampled_brake_is_not_confused_with_unqualified_fallback(self):
        b=EmergencyBrakeLatch();self.assertFalse(b.update(12,'BRAKE'))

    def test_invalid_event_does_not_advance_mode(self):
        b=EmergencyBrakeLatch()
        with self.assertRaises(ValueError):b.update(-1,'UNQUALIFIED_BRAKE')
        self.assertIsNone(b.trigger_control)


if __name__=='__main__':unittest.main(verbosity=2)
