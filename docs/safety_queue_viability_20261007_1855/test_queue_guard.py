"""Focused counterexamples and rejected malformed input before random outcomes."""
import unittest
import numpy as np
from queue_guard import diagnose_set, queue_linear_probe, issuance_status


class Contract(unittest.TestCase):
    def test_each_row_feasible_but_intersection_empty(self):
        r = diagnose_set([[1], [-1]], [-.2, -.2], [-1], [1])
        self.assertEqual(r['individual_impossible_rows'], 0)
        self.assertEqual(r['highs_status'], 2)

    def test_zero_in_box_does_not_satisfy_safety(self):
        r = diagnose_set([[1]], [-.2], [-1], [1])
        self.assertTrue(r['zero_in_box'])
        self.assertFalse(r['zero_satisfies_current_rows'])
        self.assertEqual(r['highs_status'], 0)
        self.assertFalse(r['physical_safety_certified'])

    def test_command_rate_cannot_bound_actual_velocity(self):
        r = queue_linear_probe([.03], [.0], [[1]], [0], [-3.2], np.zeros((6,1)), 1/60, np.array([True]))
        self.assertLess(r['cv6_min_margin_m'], 0)
        self.assertEqual(r['pending_min_margin_m'], [.03]*6)

    def test_future_remains_unknown_for_feasible_set(self):
        self.assertTrue(diagnose_set([[1]], [1], [-1], [1])['future_safety'].startswith('UNKNOWN'))

    def test_zero_j_contradiction_retained(self):
        self.assertEqual(diagnose_set([[0]], [-.2], [-1], [1])['highs_status'], 2)

    def test_original_projection_tolerance_not_geometry_threshold(self):
        r = issuance_status(np.array([0, 1e-6, 1.01e-6]), np.zeros(3), np.zeros(3), np.zeros(3))
        np.testing.assert_array_equal(r['unmet_selected_constraints'], [False, False, True])
        self.assertFalse(r['modifies_command'])

    def test_inputs_not_changed(self):
        a=np.array([[1.,-2.]]); h=np.array([1.]); before=a.copy()
        diagnose_set(a,h,[-1,-1],[1,1]); np.testing.assert_array_equal(a,before)

    def test_rejects_nonfinite_and_inverted_bounds(self):
        for args in [([[np.nan]],[0],[-1],[1]), ([[1]],[0],[2],[1]), ([[1]],[0],[-1],[np.inf])]:
            with self.assertRaises(ValueError): diagnose_set(*args)

    def test_exact_six_immutable_targets_required(self):
        with self.assertRaises(ValueError):
            queue_linear_probe([1],[0],[[1]],[0],[0],np.zeros((5,1)),1/60,np.array([True]))

    def test_no_relevant_rows_is_unknown(self):
        r=queue_linear_probe([1],[0],[[1]],[0],[0],np.zeros((6,1)),1/60,np.array([False]))
        self.assertEqual(r['status'],'UNKNOWN_NO_RELEVANT_ROWS')


if __name__ == '__main__': unittest.main(verbosity=2)
