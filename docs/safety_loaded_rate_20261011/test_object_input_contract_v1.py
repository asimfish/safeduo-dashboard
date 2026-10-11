"""Meaningful input faults; no simulator or full-task acceptance claim."""
from dataclasses import replace
import unittest
import numpy as np
from object_input_contract_v1 import FreshNativeObjectInput, ObjectFrame, FEATURE_ORDER


def frame(event=0, time=.01, names=('beam',)):
    k = len(names)
    state = np.zeros((k, 13), dtype=np.float32)
    state[:, :3] = [.2, .1, 1.02]
    state[:, 3] = 1
    state[:, 7:] = [.01, .02, .03, .04, .05, .06]
    return ObjectFrame(names, event, time, state, np.tile([.04, .7, .08], (k, 1)))


class InputContractTest(unittest.TestCase):
    def test_all16_features_preserve_position_size_and_both_velocities(self):
        packet = frame()
        result = FreshNativeObjectInput(('beam',)).assess(packet, expected_event=0, now_s=.01)
        self.assertEqual(FEATURE_ORDER[7:10], ('size_x', 'size_y', 'size_z'))
        self.assertIsNone(result.reason)
        self.assertEqual(result.features.shape, (1, 16))
        np.testing.assert_array_equal(result.features[:, :7], packet.root_state[:, :7])
        np.testing.assert_array_equal(result.features[:, 7:10], packet.size_m)
        np.testing.assert_array_equal(result.features[:, 10:], packet.root_state[:, 7:])
        packet.root_state.fill(0)
        self.assertEqual(result.features[0, 3], 1)
        self.assertFalse(result.features.flags.writeable)
        self.assertFalse(result.actor_input_modified)
        self.assertFalse(result.physical_stop_verified)

    def test_missing_future_stale_and_event_faults_request_all_four_locks(self):
        packet = frame()
        faults = [(None, 'MISSING_OBJECT_FRAME'),
                  (replace(packet, state_time_s=.011), 'FUTURE_OBJECT_STATE'),
                  (replace(packet, state_time_s=-.1), 'STALE_OBJECT_STATE'),
                  (replace(packet, physics_event=1), 'OBJECT_EVENT_MISMATCH'),
                  (replace(packet, physics_event=True), 'OBJECT_EVENT_MISMATCH'),
                  (replace(packet, state_time_s=float('nan')), 'NONFINITE_OBJECT_TIMESTAMP'),
                  (replace(packet, state_time_s=True), 'NONFINITE_OBJECT_TIMESTAMP')]
        gate = FreshNativeObjectInput(('beam',))
        for damaged, reason in faults:
            with self.subTest(reason=reason):
                decision = gate.assess(damaged, expected_event=0, now_s=.01)
                self.assertEqual(decision.reason, reason)
                self.assertIsNone(decision.features)
                self.assertEqual(decision.lock_request, (True,) * 4)
        self.assertIsNone(gate.assess(packet, expected_event=0, now_s=.01).reason)

    def test_inventory_order_coordinate_and_scale_are_explicit(self):
        packet = frame(names=('beam', 'post'))
        faults = [replace(packet, names=('post', 'beam')),
                  replace(packet, names=('beam',)), replace(packet, coordinate_frame='local'),
                  replace(packet, quaternion_order='xyzw'), replace(packet, length_unit='millimetre'),
                  replace(packet, extent_convention='half'), replace(packet, source='camera')]
        for damaged in faults:
            decision = FreshNativeObjectInput(('beam', 'post')).assess(damaged, expected_event=0, now_s=.01)
            self.assertIsNotNone(decision.reason)
            self.assertEqual(decision.lock_request, (True,) * 4)

    def test_nonfinite_shape_dtype_and_extent_faults_do_not_advance_history(self):
        packet = frame()
        faults = []
        for index in range(13):
            state = packet.root_state.copy();state[0, index] = np.nan
            faults.append(replace(packet, root_state=state))
        for index in range(3):
            size = packet.size_m.copy();size[0, index] = -1
            faults.append(replace(packet, size_m=size))
        zero_q = packet.root_state.copy();zero_q[:, 3:7] = 0
        faults.extend([replace(packet, root_state=zero_q),
                       replace(packet, root_state=packet.root_state[0]),
                       replace(packet, root_state=packet.root_state.astype(np.int32)),
                       replace(packet, size_m=packet.size_m[0])])
        gate = FreshNativeObjectInput(('beam',))
        for damaged in faults:
            self.assertEqual(gate.assess(damaged, expected_event=0, now_s=.01).lock_request, (True,) * 4)
        self.assertIsNone(gate.assess(packet, expected_event=0, now_s=.01).reason)

    def test_repeat_and_reorder_rejected_then_fresh_state_recovers(self):
        gate = FreshNativeObjectInput(('beam',))
        self.assertIsNone(gate.assess(frame(), expected_event=0, now_s=.01).reason)
        repeated = gate.assess(frame(), expected_event=0, now_s=.01)
        self.assertEqual(repeated.reason, 'REORDERED_OR_REPEATED_OBJECT_EVENT')
        clock_stuck = gate.assess(frame(1, .01), expected_event=1, now_s=.02)
        self.assertEqual(clock_stuck.reason, 'OBJECT_CLOCK_NOT_ADVANCING')
        self.assertIsNone(gate.assess(frame(1, .02), expected_event=1, now_s=.02).reason)


if __name__ == '__main__':
    unittest.main()
