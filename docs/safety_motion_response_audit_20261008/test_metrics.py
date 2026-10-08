"""Independent small oracles for the passive report's counting and masks."""
import unittest

import numpy as np

from audit_motion import merge, summarize_error


class MetricOracleTests(unittest.TestCase):
    def test_crossings_zeros_and_distance_bands(self):
        # Five observations: two misses (including a forecast exactly zero),
        # one false alarm, one correct warning, one correctly clear result.
        d = np.array([.004, .01, .049, .05, .09])
        p = np.array([.001, -.001, -.003, 0., .04])
        a = np.array([-.002, .002, -.005, -.001, .03])
        m = summarize_error(p, a, np.ones(5, bool), d)
        self.assertEqual(m['all']['observations'], 5)
        self.assertEqual(m['all']['future_negative'], 3)
        self.assertEqual(m['all']['predicted_negative'], 2)
        self.assertEqual(m['all']['missed_negative'], 2)
        self.assertEqual(m['all']['false_alarms'], 1)
        self.assertAlmostEqual(m['all']['absolute_error_sum_m'], .019)
        self.assertAlmostEqual(m['all']['max_gap_overestimate_m'], .01)
        self.assertAlmostEqual(m['all']['max_gap_underestimate_m'], .003)
        self.assertEqual([m[k]['observations'] for k in ['below10mm', '10to50mm', 'above50mm']], [1, 2, 2])

    def test_exempt_or_already_negative_origins_are_excluded(self):
        d = np.array([-.001, .001, .002, .003])
        origin_exempt = np.array([False, True, False, False])
        future_exempt = np.array([False, False, True, False])
        eligible = (d >= 0) & ~origin_exempt & ~future_exempt
        report = summarize_error(np.ones(4), -np.ones(4), eligible, d)
        self.assertEqual(report['all']['observations'], 1)
        self.assertEqual(report['all']['missed_negative'], 1)
        self.assertEqual(report['all']['max_gap_overestimate_m'], 2.)

    def test_streaming_aggregate_preserves_maximum_and_total(self):
        report = {}
        merge(report, summarize_error(np.array([.001]), np.array([-.002]), np.array([True]), np.array([.004])))
        merge(report, summarize_error(np.array([.04, .02]), np.array([.03, .01]), np.array([True, True]), np.array([.06, .08])))
        self.assertEqual(report['all']['observations'], 3)
        self.assertEqual(report['all']['future_negative'], 1)
        self.assertEqual(report['all']['missed_negative'], 1)
        self.assertAlmostEqual(report['all']['absolute_error_sum_m'], .023)
        self.assertAlmostEqual(report['all']['max_gap_overestimate_m'], .01)

    def test_empty_class_has_no_invented_observations(self):
        report = summarize_error(np.array([1.]), np.array([-1.]), np.array([False]), np.array([.001]))
        for bin_report in report.values():
            self.assertTrue(all(value == 0 for value in bin_report.values()))


if __name__ == '__main__':
    unittest.main()
