"""Prevent silent positive size corruption and relabelled old native samples."""
from dataclasses import replace
import unittest
import numpy as np
from object_input_contract_v2 import RegisteredNativeObjectInput, SCHEMA
from test_object_input_contract_v1 import frame


class RegisteredInputTest(unittest.TestCase):
    def gate(self):
        return RegisteredNativeObjectInput(('beam',), [[.04, .7, .08]], physics_dt_s=.01)

    def test_positive_wrong_sizes_are_rejected_then_correct_packet_recovers(self):
        gate = self.gate();packet = frame()
        for size in [packet.size_m * .5, packet.size_m * 1000, packet.size_m + .001]:
            decision = gate.assess(replace(packet, size_m=size), expected_event=0, now_s=.01)
            self.assertEqual(decision.reason, 'REGISTERED_OBJECT_SIZE_CHANGED')
            self.assertEqual(decision.lock_request, (True,) * 4)
        self.assertIsNone(gate.assess(packet, expected_event=0, now_s=.01).reason)

    def test_native_float32_and_registered_float64_extent_encodings_are_valid(self):
        for sizes in [frame().size_m, frame().size_m.astype(np.float32)]:
            decision = self.gate().assess(replace(frame(), size_m=sizes), expected_event=0, now_s=.01)
            self.assertIsNone(decision.reason)
            self.assertEqual(decision.schema, SCHEMA)

    def test_current_event_label_cannot_hide_wrong_timestamp_inside_age_limit(self):
        packet = frame(1, .015);gate = self.gate()
        decision = gate.assess(packet, expected_event=1, now_s=.02)
        self.assertEqual(decision.reason, 'OBJECT_TIMESTAMP_EVENT_MISMATCH')
        self.assertIsNone(gate.assess(frame(1, .02), expected_event=1, now_s=.02).reason)

    def test_consumer_future_clock_still_denies_physical_stop_claim(self):
        decision = self.gate().assess(frame(), expected_event=0, now_s=.009)
        self.assertEqual(decision.reason, 'FUTURE_OBJECT_STATE')
        self.assertFalse(decision.physical_stop_verified)
        self.assertFalse(decision.actor_input_modified)


if __name__ == '__main__':
    unittest.main()
