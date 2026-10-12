"""CPU counterexamples. No fixture is native physics or System0 training data."""
import unittest
import numpy as np
from long_motion_stream import Domain, FAMILIES, MotionStream, load_domain
from motion_episode_judge import judge, GATES


def toy_domain():
    return Domain(np.asarray(['motor'+str(i) for i in range(50)]),
                  np.zeros(50), np.ones(50), np.full(50, .6))


def collect(family, seed=931, duration=4., chunk=1024, domain=None):
    parts = list(MotionStream(domain or toy_domain(), family, seed, duration).chunks(chunk))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def fixture():
    n, duration = 97, .1
    t = np.arange(n)/960.
    q = np.repeat((.4+.1*t)[:, None], 50, axis=1)
    mount = np.tile((.01*t)[:, None, None], (1, 4, 3))
    trace = {'sample_index': np.arange(n), 'time_s': t, 'actual_q': q,
             'actual_qd': np.full((n, 50), .1), 'mount_xyz': mount,
             'issued_q': q.copy(), 'applied_q': q.copy(), 'held': np.zeros(n, bool)}
    trace.update({k: np.full(n, .01 if k == 'minimum_proxy_gap_m' else 0.) for k in GATES})
    contract = {'minimum_span_fraction': [.009]*50, 'minimum_mount_path_m': [.001]*4,
                'predeclared_contract_sha256': 'CPU-fixture-only',
                'qualified_reachable_domain_sha256': 'CPU-fixture-only'}
    return trace, duration, contract


class InputContracts(unittest.TestCase):
    def test_chunking_does_not_change_random_process(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                a, b = collect(family, chunk=7), collect(family, chunk=4096)
                for k in a: np.testing.assert_array_equal(a[k], b[k])

    def test_continuous_full50_and_bounds_all_families(self):
        domain = load_domain()
        for family in FAMILIES:
            with self.subTest(family=family):
                a = collect(family, domain=domain, duration=3.1)
                np.testing.assert_array_equal(a['sample_index'], np.arange(187))
                np.testing.assert_allclose(a['time_s'], np.arange(187)/60., atol=0., rtol=0.)
                self.assertEqual(a['requested_q'].shape, (187, 50))
                self.assertTrue(np.isfinite(a['requested_velocity']).all())
                self.assertTrue((a['requested_q'] >= domain.lower-1e-12).all())
                self.assertTrue((a['requested_q'] <= domain.upper+1e-12).all())

    def test_rate_bounded_processes_obey_requested_speed_caps(self):
        domain = load_domain()
        for family in ('rate_bounded_waypoints', 'persistent_velocity', 'stop_restart',
                       'staggered_arms', 'bursts_reversals', 'boundary_traversal'):
            with self.subTest(family=family):
                a = collect(family, domain=domain, duration=10.)
                speed = np.abs(np.diff(a['requested_q'], axis=0)*60.)
                self.assertTrue((speed <= domain.cap+1e-10).all())

    def test_iid_has_wide_unselected_hand_and_arm_requests(self):
        a = collect('iid_every_control', duration=30.)
        self.assertTrue((np.ptp(a['requested_q'], axis=0) > .99).all())
        self.assertTrue((np.abs(np.diff(a['requested_q'], axis=0))*60. > .6).any())
        # Wide discontinuous intentions deliberately exceed feasible command slew.

    def test_whole_episode_seed_reproducibility_and_independence(self):
        a, b, c = collect('iid_every_control', seed=123), collect('iid_every_control', seed=123), collect('iid_every_control', seed=124)
        np.testing.assert_array_equal(a['requested_q'], b['requested_q'])
        self.assertFalse(np.array_equal(a['requested_q'], c['requested_q']))

    def test_long_clock_survives_8192_physics_step_boundary(self):
        a = collect('iid_every_control', duration=30., chunk=512)
        self.assertEqual(a['time_s'][-1], 30.)
        self.assertGreater(30*960, 8192)
        self.assertEqual(len(np.unique(a['sample_index'])), 1801)
        self.assertTrue((np.diff(a['time_s']) > 0).all())
        # Generator continuity is separate from native ledger capacity qualification.

    def test_stop_restart_has_real_stationary_and_resumed_requests(self):
        a = collect('stop_restart', duration=20.)
        delta = np.diff(a['requested_q'], axis=0)
        self.assertTrue((delta[480:659] == 0).all())
        self.assertGreater(np.abs(delta[660:]).sum(), 0.)

    def test_stress_and_cartesian_claims_do_not_follow_from_joint_correlations(self):
        a = collect('correlated_opposed')
        np.testing.assert_allclose(a['requested_q'][:, 13:20], 1.-a['requested_q'][:, :7], atol=1e-15)
        # There is deliberately no derived Cartesian-safe/collision label.

    def test_invalid_clock_and_domain_rejected(self):
        for duration in (0., np.nan, .001):
            with self.assertRaises(ValueError): MotionStream(toy_domain(), FAMILIES[0], 1, duration)
        with self.assertRaises(ValueError): MotionStream(toy_domain(), 'unknown', 1, 30.)
        with self.assertRaises(ValueError): Domain(np.asarray(['same']*50), np.zeros(50), np.ones(50), np.ones(50))


class MeasuredEpisodeContracts(unittest.TestCase):
    def evaluate(self, trace, duration=.1, contract=None):
        return judge(trace, toy_domain(), duration, contract)

    def test_finite_fixture_candidate_never_certifies_native_or_hardware(self):
        trace, duration, contract = fixture()
        r = self.evaluate(trace, duration, contract)
        self.assertTrue(r['finite_recorded_long_motion_candidate_pass'])
        self.assertFalse(r['native_capture_custody_qualified_here'])
        self.assertFalse(r['all_arbitrary_motion_safety_proven'])

    def test_all_hold_can_pass_safety_but_fails_motion(self):
        trace, duration, contract = fixture()
        trace['actual_q'][:] = .4; trace['actual_qd'][:] = 0.
        trace['mount_xyz'][:] = 0.; trace['held'][:] = True
        r = self.evaluate(trace, duration, contract)
        self.assertTrue(r['recorded_original_gates_pass'])
        self.assertFalse(r['actual_coverage_contract_pass'])
        self.assertGreater(min(r['issued_span_fraction_per_motor']), 0.)

    def test_only_arm_motion_does_not_count_as_full50(self):
        trace, duration, contract = fixture()
        for a,b in ((7,13),(20,26),(32,38),(44,50)):
            trace['actual_q'][:, a:b] = .4; trace['actual_qd'][:, a:b] = 0.
        r = self.evaluate(trace, duration, contract)
        self.assertFalse(r['actual_coverage_contract_pass'])

    def test_early_failure_is_not_full_duration_success(self):
        trace, _, contract = fixture()
        trace = {k: v[:32] for k,v in trace.items()}
        trace['speed_excess_rad_s'][-1] = 1.
        r = self.evaluate(trace, .1, contract)
        self.assertFalse(r['complete_episode']); self.assertEqual(r['first_bad_index'], 31)
        self.assertFalse(r['finite_recorded_long_motion_candidate_pass'])

    def test_safe_short_prefix_is_not_long_completion(self):
        trace, _, contract = fixture()
        r = self.evaluate(trace, 600., contract)
        self.assertTrue(r['recorded_original_gates_pass']); self.assertFalse(r['complete_episode'])

    def test_reset_clock_cannot_be_concatenated_to_long_episode(self):
        trace, duration, contract = fixture()
        trace['time_s'][48:] -= trace['time_s'][48]
        r = self.evaluate(trace, duration, contract)
        self.assertFalse(r['continuous_clock']); self.assertFalse(r['complete_episode'])

    def test_missing_microstep_cannot_be_hidden(self):
        trace, duration, contract = fixture()
        trace = {k: np.delete(v, 40, axis=0) for k,v in trace.items()}
        self.assertFalse(self.evaluate(trace, duration, contract)['continuous_clock'])

    def test_same_length_clock_gap_cannot_be_hidden(self):
        trace, duration, contract = fixture()
        trace['time_s'][41:] += 1./960.
        self.assertFalse(self.evaluate(trace, duration, contract)['complete_episode'])

    def test_negative_proxy_gap_is_preserved_without_collision_overclaim(self):
        trace, duration, contract = fixture()
        trace['minimum_proxy_gap_m'][40] = -1.2e-5
        r = self.evaluate(trace, duration, contract)
        self.assertEqual(r['first_bad_index'], 40); self.assertFalse(r['recorded_original_gates_pass'])

    def test_observed_speed_is_checked_even_if_reported_summary_is_zero(self):
        trace, duration, contract = fixture()
        trace['actual_qd'][20, 49] = 6.
        r = self.evaluate(trace, duration, contract)
        self.assertFalse(r['recorded_original_gates_pass']); self.assertEqual(r['first_bad_index'], 20)

    def test_observed_bound_is_checked_even_if_reported_summary_is_zero(self):
        trace, duration, contract = fixture()
        trace['actual_q'][20, 0] = 2.
        self.assertFalse(self.evaluate(trace, duration, contract)['recorded_original_gates_pass'])

    def test_without_predeclared_reachable_coverage_acceptance_is_pending(self):
        trace, duration, _ = fixture()
        r = self.evaluate(trace, duration)
        self.assertIsNone(r['actual_coverage_contract_pass'])
        self.assertFalse(r['finite_recorded_long_motion_candidate_pass'])

    def test_missing_nonfinite_and_negative_fields_fail_closed(self):
        trace, duration, contract = fixture()
        del trace['actual_qd']
        with self.assertRaises(ValueError): self.evaluate(trace, duration, contract)
        trace, duration, contract = fixture(); trace['payload_hand_force_N'][0] = np.nan
        with self.assertRaises(ValueError): self.evaluate(trace, duration, contract)
        trace, duration, contract = fixture(); trace['hard_excess_rad'][0] = -1.
        with self.assertRaises(ValueError): self.evaluate(trace, duration, contract)


if __name__ == '__main__': unittest.main(verbosity=2)
