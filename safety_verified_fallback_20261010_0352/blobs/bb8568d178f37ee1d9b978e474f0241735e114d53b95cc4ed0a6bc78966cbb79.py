"""Strata/target provenance negatives and complete480x32 numerical artifacts."""
import csv
import json
from pathlib import Path
import sys
import unittest

import numpy as np

import test_native_teacherforced_cpu_v1 as original_tests
from native_calibration_strata_v2 import contact_strata,original_delay6
from run_native_calibration_v1 import sha

D=Path(__file__).resolve().parent
F=D/'first_candidate480x32_v2'


class StrataTests(unittest.TestCase):
    def test_zero_force_and_contact_linked_partition(self):
        scalar=np.zeros((8,2));scalar[3,0]=9.
        m=contact_strata(scalar)
        np.testing.assert_array_equal(m['contact_linked_macro_or_preboundary'][:,0],[False,False,True,True,True,True,False,False])
        np.testing.assert_array_equal(m['recorded_contact_free_proxy'],~m['contact_linked_macro_or_preboundary'])
        self.assertTrue(m['post_contact_quiet'][6,0]);self.assertFalse(m['no_positive_scalar_observed_so_far'][3,0])
        self.assertTrue(m['event_scalar_zero'][2,0]);self.assertTrue(m['contact_linked_macro_or_preboundary'][2,0])

    def test_nonzero_subthreshold_is_not_contact_free(self):
        scalar=np.zeros((2,1));scalar[1]=.01
        m=contact_strata(scalar)
        self.assertTrue(m['event_scalar_positive'][1,0]);self.assertFalse(m['event_scalar_alarm_gt_0p1N'].any())
        self.assertFalse(m['recorded_contact_free_proxy'].any())

    def test_bad_contact_observations_rejected(self):
        for bad in (np.zeros((3,2)),np.full((2,1),np.nan),np.full((2,1),-.1),np.zeros(8)):
            with self.assertRaises(ValueError):contact_strata(bad)

    def test_original_requests_delayed6_not_actual_target(self):
        initial=np.full((2,26),-2.,np.float32)
        requests=np.arange(10*2*26,dtype=np.float32).reshape(10,2,26)
        ref=original_delay6(initial,requests)
        np.testing.assert_array_equal(ref[:6],np.repeat(initial[None],6,axis=0))
        np.testing.assert_array_equal(ref[6:],requests[:4])
        self.assertFalse(np.array_equal(ref,requests))

    def test_wrong_request_width_and_dtype_rejected(self):
        with self.assertRaises(ValueError):original_delay6(np.zeros((2,74),np.float32),np.zeros((10,2,74),np.float32))
        with self.assertRaises(ValueError):original_delay6(np.zeros((2,26)),np.zeros((10,2,26)))


class Full480Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=json.loads((F/'FULL480_NATIVE_CALIBRATION_REPORT_V2.json').read_text())

    def test_complete_sample_denominator_and_unknowns(self):
        r=self.report
        self.assertEqual((r['controls_per_lane'],r['lanes'],r['microsteps_per_lane']),(480,32,960))
        self.assertEqual((r['unique_macro_lane_samples'],r['unique_micro_lane_samples']),(15360,30720))
        self.assertEqual(len(r['summaries']),8);self.assertEqual(r['independent_holdout_count'],0)
        self.assertIsNone(r['certified_bound']);self.assertFalse(r['parameters_fitted'])
        for k in ('CONTACT','ACTUAL_DRIVE_TORQUE','TERMINAL_SET','GLOBAL_MODEL_ERROR_BOUND'):self.assertEqual(r[k],'UNKNOWN')
        self.assertFalse(r['target_alignment']['future36_direct_error_enabled'])

    def test_native_targets_and_original_reference_stay_distinct(self):
        with np.load(F/'actual_native_shared_arrays.npz',allow_pickle=False) as z:
            t=z['actual_targets'];q=z['q'];v=z['qd'];pq=z['pre_q'];pv=z['pre_qd']
            self.assertEqual(t.shape,(960,32,74));np.testing.assert_array_equal(t[0::2],t[1::2])
            np.testing.assert_array_equal(pq[1:],q[1:-1:2]);np.testing.assert_array_equal(pv[1:],v[1:-1:2])
            req=z['original_request26'];ref=z['delayed_original_request26'];idx=z['controlled_columns']
            np.testing.assert_array_equal(ref,original_delay6(pq[0,:,idx].T,req))
            pause=np.array_equal(req[240:272],np.repeat(req[239:240],32,axis=0))
            self.assertEqual(pause,self.report['task_protocol']['original_request_pause240_272_present'])
            self.assertFalse(self.report['task_protocol']['task_acceptance_evaluated'])

    def test_strata_counts_and_empirical_errors_recomputed(self):
        with np.load(F/'actual_native_shared_arrays.npz',allow_pickle=False) as z:masks=contact_strata(z['scalar_peak_N'])
        for name,mask in masks.items():self.assertEqual(int(mask.sum()),self.report['strata_counts'][name])
        self.assertEqual(int(masks['recorded_contact_free_proxy'].sum()+masks['contact_linked_macro_or_preboundary'].sum()),30720)
        with np.load(F/'implicit_add_armature_errors74.npz',allow_pickle=False) as z:
            for metric in ('q','qd'):
                e=z[metric+'_error']
                self.assertEqual(e.shape,(960,32,74))
                for row in self.report['summaries'][0]['strata']:
                    if row['name'] not in masks:continue
                    v=e[masks[row['name']]]
                    if v.size:
                        self.assertAlmostEqual(float(np.max(abs(v))),row[metric]['max_abs'])
                        self.assertAlmostEqual(float(np.sqrt(np.mean(v*v))),row[metric]['rmse'])

    def test_each_variant_every_stratum_has74_fields_and_all_substeps(self):
        groups={}
        with (F/'per74_stratified_error_summary.csv').open() as f:
            for row in csv.DictReader(f):groups.setdefault((row['variant'],row['stratum'],row['substep']),set()).add(int(row['column']))
        self.assertEqual(len(groups),8*len(self.report['strata_counts'])*3)
        self.assertTrue(all(v==set(range(74)) for v in groups.values()))
        for row in self.report['summaries']:
            self.assertTrue(row['all_solves_converged'])
            if row['variant'].startswith('implicit'):self.assertLessEqual(row['max_equation_residual'],1e-9)

    def test_known_witness_strata_and_all_implicit_false_positives_preserved(self):
        r=json.loads((F/'known_witness_stratification_v2.json').read_text())
        a,b=r['findings']
        self.assertTrue(a['recorded_contact_free_proxy']);self.assertTrue(a['all7_implicit_variants_still_miss_native_lower_hard_violation'])
        self.assertEqual(a['event_scalar_N'],0);self.assertGreater(a['model_q_range_rad'][0],0)
        self.assertGreater(b['event_scalar_N'],.1);self.assertTrue(b['variants_do_not_predict_or_bound_contact'])
        rows=[x for x in r['rows'] if x['variant']=='implicit_add_armature' and x['stratum']=='event_scalar_positive']
        self.assertEqual([x['unique_native_micros'] for x in rows],[0,3])

    def test_all_inputs_outputs_and_frozen_protocol_sha_unchanged(self):
        r=json.loads((F/'input_source_sha256.json').read_text())
        for p,entry in r['entries'].items():self.assertEqual(sha(p),entry['sha256'])
        for p,expected in json.loads((F/'output_sha256.json').read_text()).items():self.assertEqual(sha(p),expected)
        r=json.loads((F/'frozen493_before_after.json').read_text());self.assertTrue(r['all493_unchanged']);self.assertEqual(r['union_count'],493)

    def test_real_wait0_and_measured_memory_budget(self):
        r=json.loads((D/'first_candidate480_attempt01_actual_wait.json').read_text())
        self.assertEqual(r['actual_exit'],0);self.assertTrue(r['wait_returned']);self.assertEqual(r['owned_pid'],r['waited_pid'])
        self.assertLess((r['child_max_rss_KiB']+r['supervisor_max_rss_KiB'])*1024,1024**3)
        self.assertEqual(sha(r['log']),r['log_sha256'])


if __name__=='__main__':
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(original_tests),
                             unittest.defaultTestLoader.loadTestsFromTestCase(StrataTests),
                             unittest.defaultTestLoader.loadTestsFromTestCase(Full480Tests)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    with (D/'full480_positive_negative_tests_v2.json').open('x') as f:
        json.dump(dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
            passed=result.wasSuccessful(),source_sha256=sha(__file__),full_report_sha256=sha(F/'FULL480_NATIVE_CALIBRATION_REPORT_V2.json')),f,indent=2)
        f.write('\n')
    sys.exit(not result.wasSuccessful())
