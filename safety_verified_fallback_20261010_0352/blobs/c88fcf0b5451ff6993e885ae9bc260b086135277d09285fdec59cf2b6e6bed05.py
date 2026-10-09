"""Analytical positive oracles, deliberate negative inputs, real witness regressions."""
from dataclasses import replace
import csv
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

import native_teacherforced_cpu_v1 as h
from native_teacherforced_cpu_v1 import Macro74, Variant, future_target_pairing, teacher_forced_two_micro
from run_native_calibration_v1 import Inputs, sha

D = Path(__file__).resolve().parent


def fixture():
    z = np.zeros(74, np.float32)
    return Macro74(q=z.copy(), qd=z.copy(), actual_targets=np.full((2,74), .5, np.float32),
        native_q=np.zeros((2,74), np.float32), native_qd=np.zeros((2,74), np.float32),
        mass=2*np.eye(74), armature=z.copy(), kp=np.full(74, 4.), kd=np.full(74, 3.),
        gravity=np.full(74, .7), coriolis=z.copy(), gravity_disabled=np.zeros(74, bool),
        velocity_target=np.full(74, .1), explicit_actuation=np.full(74, .2), effort_limits=np.full(74, 10.),
        dt_model=.1, pre_native_step=8, native_steps=np.array([9,10]),
        pre_native_time=.8, native_times=np.array([.9,1.]), native_tick=.1)


class AnalyticalTests(unittest.TestCase):
    def test_two_micro_matches_closed_form_diagonal_recurrence(self):
        sample = fixture()
        result = teacher_forced_two_micro(sample, Variant('test'))
        q = v = 0.
        expected = []
        for _ in range(2):
            v = (2*v + .1*(4*(.5-q) + 3*.1 + .2 - .7))/(2+.1*(3+.1*4))
            q += .1*v
            expected.append((q,v))
        np.testing.assert_allclose(result['predicted_q'], np.array(expected)[:,0,None]*np.ones((2,74)), atol=2e-16)
        np.testing.assert_allclose(result['predicted_qd'], np.array(expected)[:,1,None]*np.ones((2,74)), atol=2e-16)
        np.testing.assert_array_equal(result['q_error'], result['predicted_q'])

    def test_positive_and_negative_torque_saturation_analytic(self):
        for sign in (1,-1):
            sample = replace(fixture(), actual_targets=np.full((2,74), sign*100, np.float32),
                kp=np.full(74,100.), kd=np.zeros(74), gravity=np.zeros(74),
                velocity_target=np.zeros(74), explicit_actuation=np.zeros(74), effort_limits=np.full(74,.25))
            result = teacher_forced_two_micro(sample, Variant('test'))
            np.testing.assert_allclose(result['model_torque'], sign*.25, atol=1e-12)
            np.testing.assert_allclose(result['predicted_qd'], np.array([.0125,.025])[:,None]*np.full((2,74),sign), atol=1e-10)
            np.testing.assert_allclose(result['predicted_q'], np.array([.00125,.00375])[:,None]*np.full((2,74),sign), atol=1e-10)
            self.assertTrue(all(np.all(x['saturation_mask']) for x in result['numerical']))

    def test_coupled_mass_satisfies_implicit_balance(self):
        rng = np.random.default_rng(582)
        b = rng.normal(size=(74,74)) * .05
        m = b.T@b + np.eye(74)*.3
        sample = replace(fixture(), mass=m, effort_limits=np.linspace(.05,2.,74),
                         actual_targets=np.full((2,74),10.,np.float32))
        r = teacher_forced_two_micro(sample, Variant('test'))
        q, v = sample.q.astype(float), sample.qd.astype(float)
        for sub in range(2):
            q1, v1 = r['predicted_q'][sub], r['predicted_qd'][sub]
            tau = np.clip(sample.kp*(sample.actual_targets[sub]-q1) + sample.kd*(sample.velocity_target-v1)
                          + sample.explicit_actuation, -sample.effort_limits, sample.effort_limits)
            np.testing.assert_allclose(m@(v1-v), .1*(tau-sample.gravity), atol=1e-9, rtol=0)
            np.testing.assert_allclose(q1, q+.1*v1, atol=1e-12)
            q,v = q1,v1

    def test_second_micro_does_not_teacher_force_to_native_first_micro(self):
        sample = fixture()
        first = teacher_forced_two_micro(sample, Variant('test'))
        changed = replace(sample, native_q=np.full((2,74),100.,np.float32), native_qd=np.full((2,74),-9.,np.float32))
        second = teacher_forced_two_micro(changed, Variant('test'))
        np.testing.assert_array_equal(first['predicted_q'], second['predicted_q'])
        np.testing.assert_array_equal(first['predicted_qd'], second['predicted_qd'])
        np.testing.assert_allclose(second['q_error'], first['predicted_q']-100.)

    def test_new_macro_resets_from_supplied_native_state(self):
        sample = replace(fixture(), q=np.full(74,3.,np.float32), qd=np.full(74,-.2,np.float32))
        r = teacher_forced_two_micro(sample, Variant('test'))
        expected_v = (2*float(np.float32(-.2))+.1*(4*(.5-3)+.3+.2-.7))/(2+.1*3.4)
        np.testing.assert_allclose(r['predicted_q'][0], 3+.1*expected_v)

    def test_armature_gravity_coriolis_are_explicit_sensitivities(self):
        sample = replace(fixture(), armature=np.ones(74), coriolis=np.full(74,.4))
        base = teacher_forced_two_micro(sample, Variant('base'))
        lighter = teacher_forced_two_micro(sample, Variant('as_recorded', armature_mode='as_recorded'))
        no_c = teacher_forced_two_micro(sample, Variant('no_c', include_coriolis=False))
        no_g = teacher_forced_two_micro(sample, Variant('no_g', gravity_mode='off'))
        for other in (lighter,no_c,no_g):
            self.assertGreater(other['predicted_qd'][0,0], base['predicted_qd'][0,0])
        disabled = teacher_forced_two_micro(replace(sample, gravity_disabled=np.ones(74,bool)), Variant('disabled'))
        np.testing.assert_array_equal(disabled['predicted_q'], no_g['predicted_q'])

    def test_initial_dynamics_require_explicit_values(self):
        for variant in (Variant('stale_m',mass_epoch='initial_macro'), Variant('stale_b',bias_epoch='initial_macro')):
            with self.assertRaises(ValueError):
                teacher_forced_two_micro(fixture(), variant)
        base = teacher_forced_two_micro(fixture(), Variant('base'))
        stale = teacher_forced_two_micro(fixture(), Variant('stale',mass_epoch='initial_macro',bias_epoch='initial_macro'),
                                        initial_mass=fixture().mass, initial_bias=fixture().gravity)
        np.testing.assert_array_equal(base['predicted_q'], stale['predicted_q'])

    def test_input_arrays_are_not_mutated(self):
        sample = fixture()
        before = {k:h.array_sha(v) for k,v in vars(sample).items() if isinstance(v,np.ndarray)}
        teacher_forced_two_micro(sample, Variant('test'))
        self.assertEqual(before,{k:h.array_sha(v) for k,v in vars(sample).items() if isinstance(v,np.ndarray)})


class NegativeTests(unittest.TestCase):
    def test_changed_hand_target_within_macro_rejected(self):
        sample = fixture(); target=sample.actual_targets.copy(); target[1,62] += .001
        with self.assertRaisesRegex(ValueError,'full74 targets differ'):
            teacher_forced_two_micro(replace(sample, actual_targets=target), Variant('test'))

    def test_target26_or_float64_is_not_native_full74(self):
        for targets in (np.zeros((2,26),np.float32), np.zeros((2,74),np.float64)):
            with self.assertRaises(ValueError):
                teacher_forced_two_micro(replace(fixture(),actual_targets=targets), Variant('test'))

    def test_signed_zero_is_a_byte_mismatch(self):
        target=np.zeros((2,74),np.float32); target[1,73]=-0.
        with self.assertRaisesRegex(ValueError,'full74 targets differ'):
            teacher_forced_two_micro(replace(fixture(),actual_targets=target),Variant('test'))

    def test_nonfinite_inputs_rejected(self):
        sample=fixture(); q=sample.q.copy(); q[0]=np.nan
        with self.assertRaises(ValueError):
            teacher_forced_two_micro(replace(sample,q=q),Variant('test'))

    def test_asymmetric_indefinite_mass_and_negative_gains_rejected(self):
        asymmetric=fixture().mass.copy(); asymmetric[0,1]=1.
        for bad in (replace(fixture(),mass=-np.eye(74)),replace(fixture(),mass=asymmetric),
                    replace(fixture(),kp=-np.ones(74)), replace(fixture(),armature=-np.ones(74)),
                    replace(fixture(),effort_limits=np.zeros(74))):
            with self.assertRaises((ValueError,np.linalg.LinAlgError)):
                teacher_forced_two_micro(bad,Variant('test'))

    def test_misaligned_native_step_or_clock_rejected(self):
        for bad in (replace(fixture(),native_steps=np.array([10,11])),
                    replace(fixture(),native_times=np.array([.91,1.01])), replace(fixture(),dt_model=0.)):
            with self.assertRaises(ValueError):
                teacher_forced_two_micro(bad,Variant('test'))

    def test_nonconverged_numerical_solve_fails_closed(self):
        with patch.object(h._pd,'implicit_saturated_step',return_value=(np.zeros(74),np.zeros(74),np.zeros(74),{'converged':False})):
            with self.assertRaisesRegex(ValueError,'solve failed'):
                teacher_forced_two_micro(fixture(),Variant('test'))

    def test_wrong_input_sha_rejected_without_writing_source(self):
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):
            Inputs().add(__file__,'negative SHA fixture','0'*64)

    def test_unknowns_are_preserved_even_for_exact_synthetic_model(self):
        result=teacher_forced_two_micro(fixture(),Variant('test'))
        for key in ('CONTACT','ACTUAL_DRIVE_TORQUE','TERMINAL_SET','GLOBAL_MODEL_ERROR_BOUND'):
            self.assertEqual(result[key],'UNKNOWN')
        self.assertIs(result['physical_safety_certified'],False)


class FuturePairingTests(unittest.TestCase):
    def test_all36_exact_targets_enable_comparison(self):
        target=np.zeros((36,74),np.float32)
        self.assertTrue(future_target_pairing(target,target.copy())['full36_direct_error_enabled'])

    def test_future_hand_mismatch_invalidates_all_later_states(self):
        target=np.zeros((36,74),np.float32); actual=target.copy(); actual[14,62]=.1
        r=future_target_pairing(target,actual)
        self.assertFalse(r['full36_direct_error_enabled'])
        self.assertEqual(r['first_target_mismatch_micro'],14)
        self.assertEqual(r['direct_error_valid_micro_mask'],[True]*14+[False]*22)

    def test_incomplete16_native_future_never_becomes_full36(self):
        target=np.zeros((36,74),np.float32)
        r=future_target_pairing(target,target[:16])
        self.assertFalse(r['full36_direct_error_enabled']); self.assertEqual(r['paired_prefix_microsteps'],16)
        self.assertEqual(r['unavailable_microsteps'],list(range(16,36)))

    def test_float64_or26_future_rejected(self):
        target=np.zeros((36,74),np.float32)
        for bad in (target.astype(np.float64),target[:,:26]):
            with self.assertRaises(ValueError): future_target_pairing(target,bad)


class RealWitnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out=D/'two_witness_v1'
        cls.report=json.loads((cls.out/'calibration_report.json').read_text())

    def test_real_native_known_failures_remain_observed(self):
        a,b=self.report['known_witnesses']
        self.assertEqual((a['lane'],a['first_failure_micro']),(18,12))
        hard=a['variants'][0]['known_hard_joint']
        self.assertAlmostEqual(hard['actual_q'],-.001607122365385294)
        self.assertGreater(hard['predicted_q'],0.)
        self.assertGreater(hard['q_error'],.009)
        self.assertEqual((b['lane'],b['first_failure_micro']),(19,13))
        self.assertAlmostEqual(b['observed_scalar_peak_N'],9.281611442565918)

    def test_real_future_repeat_mismatches_disable36(self):
        for case in self.report['known_witnesses']:
            p=case['future36_pairing']
            self.assertFalse(p['full36_direct_error_enabled'])
            self.assertEqual(p['first_target_mismatch_micro'],14)
            self.assertEqual(p['paired_prefix_microsteps'],14)

    def test_all74_fields_all_macros_retained_for_every_variant(self):
        groups={}
        with (self.out/'per74_error_summary.csv').open() as f:
            for r in csv.DictReader(f):
                groups.setdefault((r['lane'],r['variant'],r['substep']),set()).add(int(r['column']))
        self.assertEqual(len(groups),2*8*3)
        self.assertTrue(all(v==set(range(74)) for v in groups.values()))
        with np.load(self.out/'two_witness_arrays.npz',allow_pickle=False) as z:
            for lane in (18,19):
                for variant in self.report['variants']:
                    name=f'lane{lane}_{variant["name"]}'
                    for key,actual in (('q','q'),('qd','qd')):
                        self.assertEqual(z[name+'_'+key+'_error'].shape,(16,74))
                        np.testing.assert_array_equal(z[name+'_'+key+'_error'], z[name+'_predicted_'+key]-z[f'lane{lane}_actual_{actual}'])

    def test_solver_residual_not_promoted_to_native_bound(self):
        self.assertIsNone(self.report['global_bound'])
        self.assertEqual(self.report['independent_holdout_count'],0)
        self.assertEqual(self.report['real_unique_microsteps'],32)
        self.assertFalse(self.report['parameters_fitted_to_witnesses'])
        for r in self.report['summaries']:
            self.assertTrue(r['all_solves_converged'])
            if r['variant']['drive_mode']=='implicit_saturated':
                self.assertLess(r['max_equation_residual'],1e-9)
                self.assertGreater(r['q']['max_abs'],.001)

    def test_all_frozen493_and_all_recorded_inputs_unchanged(self):
        f=json.loads((self.out/'frozen493_before_after.json').read_text())
        self.assertEqual(f['union_count'],493); self.assertTrue(f['all493_unchanged'])
        for p,e in f['sources'].items(): self.assertEqual(sha(p),e['expected_sha256'])
        manifest=json.loads((self.out/'input_source_sha256.json').read_text())
        for p,e in manifest['entries'].items(): self.assertEqual(sha(p),e['sha256'])

    def test_actual_wait_is_finished_zero_within_memory_limit(self):
        r=json.loads((D/'two_witness_attempt01_actual_wait.json').read_text())
        self.assertEqual(r['actual_exit'],0); self.assertTrue(r['wait_returned'])
        self.assertEqual(r['owned_pid'],r['waited_pid'])
        self.assertLess((r['child_max_rss_KiB']+r['supervisor_max_rss_KiB'])*1024,1024**3)
        self.assertEqual(sha(r['log']),r['log_sha256'])

    def test_no_gpu_or_simulator_module_loaded(self):
        self.assertFalse(any(k=='torch' or k.startswith(('isaac','omni.')) for k in sys.modules))


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    output=D/'positive_negative_tests_v1.json'
    with output.open('x') as f:
        json.dump(dict(tests_run=result.testsRun, failures=len(result.failures),errors=len(result.errors),
                       skipped=len(result.skipped),passed=result.wasSuccessful(),
                       test_source_sha256=sha(__file__), helper_source_sha256=sha(h.__file__),
                       real_witness_output_sha256=sha(D/'two_witness_v1/output_sha256.json'),
                       red_capable_negative_inputs=True,physical_safety_certified=False),f,indent=2)
        f.write('\n')
    sys.exit(not result.wasSuccessful())
