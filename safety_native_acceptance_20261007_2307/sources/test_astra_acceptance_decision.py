"""CPU-only synthetic tests. No fixture is an actual experiment acceptance.

All fixtures live in memory; production cardinalities are unchanged. Three-frame
raw reader tests patch T locally only to exercise real full9021/contact math.
"""
import copy
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import astra_acceptance_decision as a

H = Path(__file__).resolve().parent
CRITERIA = json.loads((H/'ACCEPTANCE_CRITERIA.json').read_text())
INITIAL_SPEC = json.loads((H/'INITIAL_STATE_ACCEPTANCE_REGISTRATION.json').read_text())
LABELS = np.array([x for x in range(6) for _ in range(8)] + [-1]*16)


def measured_fixture():
    q0 = np.zeros((a.N, 26), np.float32)
    q = np.broadcast_to(np.arange(1, a.T+1, dtype=np.float32)[:, None, None]*.001,
                        (a.T, a.N, 26)).copy()
    margins = np.full((a.T, a.N, 4), .01, np.float32)
    cell = dict(q=q, q_initial=q0, official_margins=margins.copy(),
                official_deep=margins < np.float32(-.005))
    return cell, margins


def records_fixture():
    scores = a.score_and_motion(*measured_fixture())
    scores.update(initial_strict_ids=[], initial_deep_ids=[],
                  initial_minimum_by_class_m=np.full(4,.01,np.float32))
    records = [dict(block=b, mode=m, windows=64, frames=960, physics_events=1920,
                    hand_camera_bound=True,
                    labels=LABELS.copy(), **copy.deepcopy(scores), peak=np.zeros(64))
               for b in range(2) for m in a.MODES]
    pairing = [dict(block=r['block'], mode=r['mode'], differences=[]) for r in records]
    return records, pairing


def native_fixture():
    result = {}
    for arm, w in zip(a.ARMS, a.WIDTHS):
        for field in a.FIELDS:
            if field.startswith('pending_'):
                shape = (6, 64, w)
            elif field in ('native_q', 'native_qd'):
                shape = (64, w+8)  # explicitly includes uncontrolled hand DOFs
            elif field == 'native_root_xyzw':
                shape = (64, 7)
            elif field == 'native_root_velocity':
                shape = (64, 6)
            else:
                shape = (64, w)
            result[arm+'_'+field] = np.zeros(shape, np.float32)
    return result


def execution_fixture():
    reg = dict(tag='synthetic_only', registered_utc='2026-10-07T12:00:00+00:00', jobs=[])
    execution = dict(tag=reg['tag'], plan_sha256='abc', status='complete',
        started_utc='2026-10-07T13:00:00+00:00', closed_utc='2026-10-07T14:00:00+00:00', jobs=[])
    for b in range(2):
        for m in a.MODES:
            j = dict(id=f'b{b}_{m}', kind='physics', steps=960, scheduled_groups=21,
                out=f'/synthetic_only/b{b}_{m}', argv=['python', '-B', 'never_executed.py'], env={})
            reg['jobs'].append(j)
            execution['jobs'].append(dict(**j, status='complete', actual_exit=0, pid=123,
                started_utc='2026-10-07T13:01:00+00:00', closed_utc='2026-10-07T13:59:00+00:00'))
    return reg, execution


def audit_fixture():
    cell, computed = measured_fixture()
    s = a.score_and_motion(cell, computed)
    s.update(initial_strict_ids=[], initial_deep_ids=[],
             initial_minimum_by_class_m=np.full(4,.01,np.float32))
    root = Path('/synthetic_only')
    metrics = np.zeros((960, 2, 64, 4), np.float64)
    g = dict(status='PASS_ALL_FULL_RAW_FLOAT32_GEOMETRY_SCORING', root=str(root), windows=64,
        initial_negative_envs=0,initial_negative_ids=[],initial_raw_minimum_by_class_m=s['initial_minimum_by_class_m'].tolist(),
        cell_sha256='cell', frames=960, geometry_rows=9021, raw_env_frames=960*64,
        strict_windows=0, strict_env_frames=0, deep_windows=0, deep_env_frames=0,
        strict_ids=[], deep_ids=[], first_failure_steps=s['first'], minimum_by_class_m=s['minima'].tolist(),
        q_l2_path_by_env=s['path'].tolist(), mean_q_l2_path=float(s['path'].mean()),
        joint_range_by_env=s['ranges'].tolist(), four_arms_moving_fraction=s['four'])
    n = dict(status='PASS_SAME_RUN_NATIVE_AND_NINE_VIEW_BINDING', root=str(root), windows=64,
        cell_sha256='cell', frames=960, boundary_packets=2880, groups=21, PNG=189,
        native_controlled_q_qd_exact=True, actual_FIFO6_exact=True,
        all64_render_native_unchanged=True, original_full_geometry_score_exact=True)
    c = dict(status='PASS_NATIVE_HAND_CONTACT_OBSERVATION_NOT_SAFETY', root=str(root), windows=64,
        control_steps=960, physics_events=1920, raw_normal_diagnostic_threshold_N=.1,
        same_hand_contacts_included_in_raw=True, exemption_adjusted_contact_gate=False,
        capacity=262144, peak_contact_count=0, maximum_net_minus_filtered_normal_abs_N=0.,
        window_peak_normal_N=[0.]*64, windows_exceeding_normal_diagnostic=0)
    peer = dict(status='PASS_RAW_BINDING_AND_ORIGINAL_SCORING_ONLY', root=str(root), windows=64,
        auditor_sha256='peer_source', exitcode=0,
        frames=960, raw_sha256={'cell_001.npz': 'cell'}, full_geometry_rows=9021,
        raw_geometry_env_frames=960*64, native_boundary_packets=2880, camera_groups=21, PNG=189,
        queued_future_status='UNKNOWN', score_threshold_epsilon=0, physical_safety_certified=False,
        strict_windows=0, strict_env_frames=0, deep_windows=0, deep_env_frames=0,
        strict_env_ids=[], deep_env_ids=[], minimum_by_class_m=s['minima'].tolist(),
        contacts=dict(status='PASS_SUBSTEP_BINDING_ONLY', physics_events=1920,
            actual_PhysX_position_targets_verified=True, raw_normal_diagnostic_threshold_N=.1,
            maximum_contact_count=0, maximum_net_minus_filtered_normal_abs_N=0.,
            raw_normal_peak_by_env_arm_N=np.zeros((64, 4)).tolist(), raw_windows_over_threshold=0,
            raw_env_frames_over_threshold=0, non_same_hand_windows_over_0p1N=0))
    return [root, 'cell', s, metrics, g, n, c, peer, 'peer_source']


class MemoryLedger:
    def __init__(self, docs, arrays):
        self.docs, self.arrays = docs, arrays

    def json(self, p, expected=None):
        return self.docs[Path(p).name]

    def npz(self, p, keys=None, expected=None):
        z = self.arrays[Path(p).name]
        return z if keys is None else {k: z[k] for k in keys}


class HandLedger:
    """Real hash binding on in-memory manifest/file bytes; no disk fixtures."""
    def __init__(self):
        self.files, self.hashes = {}, {}

    def put(self, path, value):
        self.files[str(path)] = json.dumps(value, sort_keys=True).encode() if isinstance(value, dict) else value
        return hashlib.sha256(self.files[str(path)]).hexdigest()

    def bind(self, path, expected=None):
        key=str(path)
        a.complete(key in self.files, 'missing synthetic raw '+key)
        digest=hashlib.sha256(self.files[key]).hexdigest()
        a.require(expected is None or digest==expected, 'SHA mismatch '+key)
        a.require(key not in self.hashes or self.hashes[key]==digest, 'raw changed '+key)
        self.hashes[key]=digest
        return digest

    def json(self, path, expected=None):
        self.bind(path,expected)
        return json.loads(self.files[str(path)])

    def verify(self):
        for path,digest in list(self.hashes.items()):self.bind(path,digest)


def hand_fixture():
    root=Path('/synthetic_only'); ledger=HandLedger()
    sources={str(H/'audit_hand_views.py'):'auditor',str(H/'hand_views.py'):'producer'}
    cell_sha=ledger.put(root/'cell_001.npz',b'synthetic closed cell')
    contact_sha=ledger.put(root/'native_contact_identity.json',dict(synthetic=True))
    def ref(relative, value):
        return dict(path=relative,sha256=ledger.put(root/relative,value))
    overview=[];hands=[]
    for t in (75,480,959):
        prefix=f'hand_views/scheduled/step_{t:04d}'
        before=ref(prefix+'/native_before_all64.npz',b'synthetic native before')
        final=ref(prefix+'/native_final_all64.npz',b'synthetic native final')
        mapping=ref(prefix+'/identity_mapping.json',dict(source=dict(sha256=contact_sha)))
        group=[]
        for e in (0,8,16,24,32,40,56):
            identity=dict(step=t,env_id=e,capture_kind='scheduled')
            parent=ref(f'multiview/scheduled/env_{e:03d}/step_{t:04d}_state.json',identity)
            overview.append(dict(**identity,state=parent['path'],sha256=parent['sha256']))
            images=[]
            for arm in a.ARMS:
                for view in ('oblique_above','opposite_below','cross_above'):
                    stem=f'{prefix}/env_{e:03d}/{arm}_{view}'
                    image=ref(stem+'.png',stem.encode())
                    after=ref(stem+'_native_after_all64.npz',b'synthetic native after')
                    images.append(dict(**image,arm=arm,view=view,native_before_all64=before,native_after_all64=after))
            state=dict(**identity,image_count=12,images=images,source_sha256='producer',
                parent_macro_binding=parent,native_before_all64=before,sensor_identity_mapping=mapping)
            state_ref=ref(f'{prefix}/env_{e:03d}/state.json',state)
            capture=dict(**identity,state=state_ref['path'],sha256=state_ref['sha256'],parent_macro_binding=parent,
                images=[{k:im[k] for k in ('arm','view','path','sha256')} for im in images])
            group.append(capture);hands.append(capture)
        ledger.put(root/(prefix+'/receipts.json'),dict(visibility_restored=True,groups=7,PNG=84,
            receipts=group,native_before_all64=before,native_final_all64=final))
    ledger.put(root/'hand_camera_receipts.json',dict(groups=21,PNG=252,receipts=hands))
    ledger.put(root/'camera_receipts.json',dict(groups=21,PNG=189,receipts=overview))
    principal=dict(groups=21,PNG=189,cell_sha256=cell_sha)
    receipt=dict(status='PASS_CLOSED_CELL_NATIVE_HAND_CAMERA_BINDING',root=str(root),frames=960,
        windows=64,groups=21,hand_PNG=252,total_PNG=441,source_sha256='auditor',cell_sha256=cell_sha,
        all64_native_bitwise_render_invariant=True,exact_contact_sensor_bijection=3328,
        full_mesh_certified=False,occlusion_certified=False,hardware_certified=False)
    return root,ledger,receipt,principal,sources,H


class FiniteDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = records_fixture()

    def setUp(self):
        self.records, self.pairing = copy.deepcopy(self.fixture)

    def decide(self):
        return a.decide(self.records, CRITERIA, self.pairing, INITIAL_SPEC)

    def candidate(self):
        return [r for r in self.records if r['mode'] == 'adaptive_joint']

    def test_synthetic_complete_denominators(self):
        r = self.decide()
        self.assertEqual(r['candidate_decision'], 'PASS_BOUNDED_SIMULATION_ONLY')
        self.assertEqual((r['method_windows'], r['unique_paired_cases'], r['independent_initial_banks']), (512,128,2))
        self.assertEqual(r['total_physics_env_substeps'], 983040)
        self.assertEqual([s['cases'] for s in r['strata']], [32,16,16,16,16,16,16])

    def test_missing_window_batch_never_passes(self):
        self.records.pop()
        with self.assertRaises(a.IncompleteEvidence): self.decide()

    def test_duplicate_bank_mode(self):
        self.records[-1] = copy.deepcopy(self.records[0])
        with self.assertRaises(a.InvalidEvidence): self.decide()

    def test_partial_window_or_substep(self):
        for field, value in [('windows',63), ('frames',959), ('physics_events',1919)]:
            with self.subTest(field=field):
                old = self.records[0][field]; self.records[0][field] = value
                with self.assertRaises(a.InvalidEvidence): self.decide()
                self.records[0][field] = old

    def test_missing_risk_stratum(self):
        self.records[-1]['labels'][0] = -1
        with self.assertRaises(a.InvalidEvidence): self.decide()

    def test_strict_new_failure_and_deep(self):
        self.candidate()[0]['strict'][4] = True
        self.candidate()[0]['deep'][4] = True
        r = self.decide()
        self.assertEqual(r['new_paired_strict_failures'], 1)
        self.assertTrue({'candidate_strict_windows','candidate_deep_windows','new_paired_strict_failure'}
                        <= set(r['rejection_reasons']))

    def test_baseline_failure_rescued_not_rejected(self):
        next(r for r in self.records if r['mode']=='zero_inclusive')['strict'][2] = True
        self.assertEqual(self.decide()['candidate_decision'], 'PASS_BOUNDED_SIMULATION_ONLY')

    def test_overall_motion_suppression(self):
        for r in self.candidate(): r['path'] *= .899
        self.assertIn('overall_path_below_floor_or_undefined', self.decide()['rejection_reasons'])

    def test_one_stratum_suppression_despite_high_overall_motion(self):
        for r in self.candidate():
            r['path'] *= 2
            r['path'][r['labels']==5] *= .4
        d = self.decide()
        self.assertGreater(d['actual_motion_path_ratio'], .9)
        self.assertIn('stratum_path_below_floor_or_undefined:5', d['rejection_reasons'])

    def test_four_arm_suppression(self):
        for r in self.candidate(): r['four'] = .899
        self.assertIn('four_arm_motion_below_floor_or_undefined', self.decide()['rejection_reasons'])

    def test_26th_joint_floor_not_global_average(self):
        for r in self.candidate(): r['ranges'][:,25] = .000999
        self.assertIn('one_or_more_26joint_ranges_below_floor', self.decide()['rejection_reasons'])

    def test_exact_motion_and_range_boundaries(self):
        for r in self.records:
            r['path'][:] = 10. if r['mode']=='zero_inclusive' else 9.
            r['ranges'] = np.full((64,26), .001, dtype=np.float32)
        for r in self.candidate(): r['four'] = .9
        self.assertEqual(self.decide()['candidate_decision'], 'PASS_BOUNDED_SIMULATION_ONLY')

    def test_range_mean_accumulates_in_float64_without_floor_roundup(self):
        # Two adjacent float32 raw ranges straddle .001; their exact mean is below
        # .001. A float32 128-row reduction can round it above the acceptance floor.
        self.candidate()[0]['ranges'][:,25] = np.nextafter(np.float32(.001), np.float32(0))
        self.candidate()[1]['ranges'][:,25] = np.float32(.001)
        d = self.decide()
        self.assertLess(d['aggregated']['adaptive_joint']['mean_joint_ranges'][25], .001)
        self.assertIn('one_or_more_26joint_ranges_below_floor', d['rejection_reasons'])

    def test_undefined_zero_motion_baseline_rejected(self):
        for r in self.records:
            if r['mode']=='zero_inclusive': r['path'][:] = 0; r['four'] = 0
        d = self.decide()
        self.assertIsNone(d['actual_motion_path_ratio'])
        self.assertEqual(d['candidate_decision'], 'REJECTED')

    def test_nan_metrics_never_pass(self):
        self.candidate()[0]['path'][0] = np.nan
        with self.assertRaises(a.InvalidEvidence): self.decide()

    def test_contact_limit_inclusive_and_no_epsilon(self):
        self.candidate()[0]['peak'][1] = .1
        self.assertEqual(self.decide()['candidate_decision'], 'PASS_BOUNDED_SIMULATION_ONLY')
        self.candidate()[0]['peak'][1] = np.nextafter(.1, np.inf)
        d = self.decide()
        self.assertEqual(d['candidate_decision'], 'REJECTED')
        self.assertEqual(d['represented_sphere_decision'], 'PASS_BOUNDED_REPRESENTED_SPHERES')

    def test_actual_full_native_mismatch_blocks_causal_acceptance(self):
        self.pairing[-1]['differences'] = [{'field':'F_L_native_qd','env_ids':[63]}]
        self.assertIn('actual_full_native_initial_mismatch', self.decide()['rejection_reasons'])

    def test_initial_bad_rejects_adoption_even_when_all960_post_clean(self):
        before=self.decide()
        self.candidate()[0]['initial_strict_ids']=[11]
        d=self.decide()
        self.assertEqual(d['candidate_decision'],'REJECTED')
        self.assertEqual(d['initial_state_decision'],'REJECTED')
        self.assertEqual(d['initial_negative_cases'],[dict(block=0,env=11)])
        self.assertEqual(d['represented_sphere_decision'],before['represented_sphere_decision'])
        self.assertEqual(d['rejection_reasons'],['candidate_actual_initial_geometry_negative'])
        self.assertEqual(d['method_windows'],512)
        for key in ('strict_windows','deep_windows','mean_q_path','four_arm_fraction','mean_joint_ranges'):
            self.assertEqual(d['aggregated']['adaptive_joint'][key],before['aggregated']['adaptive_joint'][key])

    def test_initial_cases_are_not_filtered_or_collapsed_across_banks(self):
        for r in self.candidate():r['initial_strict_ids']=[63]
        d=self.decide()
        self.assertEqual(d['aggregated']['adaptive_joint']['initial_negative_envs'],2)
        self.assertEqual(d['initial_negative_cases'],[dict(block=0,env=63),dict(block=1,env=63)])
        self.assertEqual(d['unique_paired_cases'],128)
        self.assertEqual(d['aggregated']['adaptive_joint']['windows'],128)

    def test_missing_initial_raw_evidence_never_defaults_to_safe(self):
        del self.candidate()[0]['initial_strict_ids']
        with self.assertRaises(a.IncompleteEvidence):self.decide()

    def test_hand_binding_required_for_every_method_window(self):
        for mode in a.MODES:
            with self.subTest(mode=mode):
                r=next(x for x in self.records if x['mode']==mode)
                del r['hand_camera_bound']
                with self.assertRaises(a.IncompleteEvidence):self.decide()
                r['hand_camera_bound']=True

    def test_invalid_initial_case_ids(self):
        for ids in ([True],[64],[-1],[0,0],[3,2]):
            with self.subTest(ids=ids):
                self.candidate()[0]['initial_strict_ids']=ids
                with self.assertRaises(a.InvalidEvidence):self.decide()

    def test_initial_baseline_failure_retained_without_candidate_exclusion(self):
        next(r for r in self.records if r['mode']=='zero_inclusive')['initial_strict_ids']=[0]
        d=self.decide()
        self.assertEqual(d['aggregated']['zero_inclusive']['initial_negative_envs'],1)
        self.assertEqual(d['candidate_decision'],'PASS_BOUNDED_SIMULATION_ONLY')

    def test_preregistration_change_witness_against_preserved_previous_checker(self):
        snapshot=None
        for line in (H/'astra_acceptance_decision.log').read_text().splitlines():
            if line.startswith('{"record": "PRE_INITIAL_STATE_GATE_SOURCE_SNAPSHOT"'):
                snapshot=json.loads(line);break
        self.assertIsNotNone(snapshot)
        preserved=snapshot['files']['astra_acceptance_decision.py']
        import hashlib
        self.assertEqual(hashlib.sha256(preserved['utf8'].encode()).hexdigest(),
                         '69485111f849db6f588dd6feba24f1d326e35011850cb4b269962890acdf20b8')
        previous={'__name__':'preserved_pre_initial_gate_checker','__file__':str(H/'astra_acceptance_decision.py')}
        exec(compile(preserved['utf8'],'<preserved_pre_initial_gate_checker>','exec'),previous)
        self.candidate()[0]['initial_strict_ids']=[11]
        # Previous criteria used initial geometry only diagnostically. This is a
        # witness for the newly registered behavior, not a fault in the old spec.
        self.assertEqual(previous['decide'](self.records,CRITERIA,self.pairing)['candidate_decision'],
                         'PASS_BOUNDED_SIMULATION_ONLY')
        self.assertEqual(self.decide()['candidate_decision'],'REJECTED')


class PairingAndClosureTests(unittest.TestCase):
    def test_every_full_native_field_one_ulp_mismatch(self):
        base = native_fixture()
        for arm in a.ARMS:
            for field in a.FIELDS:
                with self.subTest(arm=arm, field=field):
                    other = copy.deepcopy(base)
                    key = arm+'_'+field
                    index = (5,63,-1) if field.startswith('pending_') else (63,-1)
                    other[key][index] = np.nextafter(np.float32(0), np.float32(1))
                    diff = a.pairing_difference(base, other)
                    self.assertEqual(len(diff), 1)
                    self.assertEqual(diff[0]['field'], key)
                    self.assertEqual(diff[0]['env_ids'], [63])
                    self.assertGreater(diff[0]['max_abs_difference'], 0)

    def test_same_requested_q_does_not_cover_uncontrolled_hand(self):
        base = native_fixture(); other = copy.deepcopy(base)
        other['F_L_native_q'][0,-1] = .1
        self.assertEqual(a.pairing_difference(base, other)[0]['env_ids'], [0])

    def test_native_nan_invalid(self):
        base = native_fixture(); base['U_R_native_qd'][2,-1] = np.nan
        with self.assertRaises(a.InvalidEvidence): a.pairing_difference(base, base)

    def test_actual_wait_required(self):
        _, execution = execution_fixture(); r=execution['jobs'][0]; del r['actual_exit']
        with self.assertRaises(a.IncompleteEvidence): a.check_wait(r)

    def test_boolean_zero_and_nonzero_signal_exit_rejected(self):
        for exitcode in (False, '0', -15, 241, 1):
            with self.subTest(exitcode=exitcode):
                _, execution = execution_fixture(); r=execution['jobs'][0]; r['actual_exit']=exitcode
                with self.assertRaises(a.InvalidEvidence): a.check_wait(r)

    def test_complete_execution(self):
        reg, ex = execution_fixture()
        planned, ran = a.check_execution(reg, ex, 'abc')
        self.assertEqual(set(planned), set(ran))

    def test_resource_wait_or_kit0_partial_never_accepted(self):
        for status in ('running', 'failed', 'resource_wait', 'child_reaped'):
            reg, ex = execution_fixture(); ex['jobs'][0]['status']=status
            with self.assertRaises(a.IncompleteEvidence): a.check_execution(reg, ex, 'abc')

    def test_execution_mutations(self):
        mutations = [('bad plan SHA', lambda r,e:e.update(plan_sha256='other')),
            ('reused root', lambda r,e:r['jobs'][1].update(out=r['jobs'][0]['out'])),
            ('late registration', lambda r,e:r.update(registered_utc='2026-10-07T13:02:00+00:00')),
            ('actual argv differs', lambda r,e:e['jobs'][0].update(argv=['other'])),
            ('90step development', lambda r,e:r['jobs'][0].update(steps=90))]
        for name, mutate in mutations:
            with self.subTest(name=name):
                reg, ex=execution_fixture(); mutate(reg,ex)
                with self.assertRaises(a.InvalidEvidence): a.check_execution(reg,ex,'abc')

    def test_raw_hash_mismatch(self):
        p=H/'ACCEPTANCE_CRITERIA.json'
        with self.assertRaises(a.InvalidEvidence): a.Ledger().bind(p,'0'*64)

    def test_raw_ledger_detects_change_at_final_closure(self):
        ledger=a.Ledger(); p=H/'ACCEPTANCE_CRITERIA.json'
        with patch.object(a,'sha',side_effect=['before','after']):
            ledger.bind(p)
            with self.assertRaises(a.InvalidEvidence): ledger.verify()

    def test_raw_reference_cannot_escape_root(self):
        for path in ('../other/cell.npz','/absolute.npz'):
            with self.assertRaises(a.InvalidEvidence): a.confined('/synthetic_only',path)

    def test_missing_registration_incomplete_without_any_actual_run(self):
        with self.assertRaises(a.IncompleteEvidence): a.check(H/'NONEXISTENT_SYNTHETIC_REGISTRATION_DIR')

    def test_exact_frozen_criteria_and_pairing(self):
        self.assertEqual(a.sha(H/'ACCEPTANCE_CRITERIA.json'),a.CRITERIA_SHA)
        self.assertEqual(a.sha(H/'PAIRING_REGISTRATION.json'),a.PAIRING_SHA)
        self.assertEqual(a.sha(H/'COMMON_HAND_CONTEXT_REGISTRATION.json'),a.COMMON_CONTEXT_SHA)
        self.assertEqual(a.sha(H/'safe_hand_opening.py'),a.OPENING_SOURCE_SHA)
        self.assertEqual(a.sha(H/'INITIAL_STATE_ACCEPTANCE_REGISTRATION.json'),a.INITIAL_STATE_REG_SHA)

    def test_initial_gate_registration_source_time_and_no_filter_binding(self):
        policy=dict(registered_utc='2026-10-07T17:00:00+00:00')
        sources={str(H/'audit_geometry_v2.py'):INITIAL_SPEC['source_sha256']}
        a.check_initial_registration(INITIAL_SPEC,policy,sources,H)
        cases=[('new_policy_outcomes_observed',True),('all_new_cases_retained',False),
            ('filter_or_resample_policy_outcomes',True),('original_full960_post_scoring_unchanged',False),
            ('requires_candidate_actual_initial_full9021_nonexempt_negative_cases',1),
            ('requires_candidate_actual_initial_full9021_nonexempt_negative_cases',False),
            ('source_sha256','wrong'),('registered_utc','2026-10-07T18:00:00+00:00')]
        for key,value in cases:
            with self.subTest(key=key,value=value):
                changed=copy.deepcopy(INITIAL_SPEC);changed[key]=value
                with self.assertRaises(a.InvalidEvidence):a.check_initial_registration(changed,policy,sources,H)

    def test_camera_registration_is_frozen_by_policy_not_hardcoded_sha(self):
        spec=json.loads((H/'CAMERA_COVERAGE_REGISTRATION.json').read_text())
        policy=dict(registered_utc='2026-10-08T12:00:00+00:00')
        sources={str(H/'audit_hand_views.py'):spec['closed_cell_auditor_sha256'],
                 str(H/'hand_views.py'):spec['producer_sha256']}
        a.check_camera_registration(spec,policy,sources,H)
        changed=copy.deepcopy(spec);changed['producer_sha256']='another-preregistered-source'
        sources[str(H/'hand_views.py')]=changed['producer_sha256']
        a.check_camera_registration(changed,policy,sources,H)
        for key,value in [('principal_views',8),('hand_views_per_arm',2),('total_views_per_group',20),
            ('full_mesh_certified',True),('occlusion_certified',True),('closed_cell_auditor_sha256','wrong'),
            ('registered_utc','2026-10-09T12:00:00+00:00')]:
            with self.subTest(key=key):
                bad=copy.deepcopy(changed);bad[key]=value
                with self.assertRaises(a.InvalidEvidence):a.check_camera_registration(bad,policy,sources,H)

    def test_hand_audit_wait_requires_absolute_auditor_root_exact_tag_and_actual_zero(self):
        root=Path('/synthetic_only');output=H/'b0_adaptive_joint_hands.json';path=H/'synthetic_hand_execution.json'
        receipt=dict(argv=['python','-B',str(H/'audit_hand_views.py'),str(root),output.stem],actual_exit=0,pid=321,
            started_utc='2026-10-07T18:00:00+00:00',closed_utc='2026-10-07T18:01:00+00:00',
            tag='synthetic_hand',log_sha256='log')
        class L:
            def json(self,p):return receipt
            def bind(self,p,digest):
                if p!=H/'synthetic_hand.log' or digest!='log':raise AssertionError('actual log binding')
        def check():a.check_audit_wait(H,L(),'audit_hand_views.py',root,output,'2026-10-07T17:59:00+00:00',absolute_paths=True)
        with patch.object(Path,'glob',return_value=[path]),patch.object(Path,'read_text',side_effect=lambda:json.dumps(receipt)):
            check()
            for index,value in [(2,'audit_hand_views.py'),(3,'relative_root'),(4,'wrong_tag')]:
                original=receipt['argv'][index];receipt['argv'][index]=value
                with self.assertRaises(a.IncompleteEvidence):check()
                receipt['argv'][index]=original
            del receipt['actual_exit']
            with self.assertRaises(a.IncompleteEvidence):check()
            receipt['actual_exit']=False
            with self.assertRaises(a.InvalidEvidence):check()
            receipt['actual_exit']=1
            with self.assertRaises(a.InvalidEvidence):check()

    def test_common_opening_identity_and_actual_drift_retained(self):
        native=native_fixture()
        opening=dict(open_thumb_rad=.35,all_modes_common=True,controlled_arm_targets_changed=False,
            original_actor_FIF06_scores_exemptions_gains_self_collision_unchanged=True,
            constructor_contacts_certified=False,requested_joint_names={'U_L':'left_thumb_1_joint',
            'U_R':'right_thumb_1_joint'},joint_indices={'U_L':6,'U_R':6})
        for arm in ('U_L','U_R'):
            names=['arm'+str(i) for i in range(6)]+[opening['requested_joint_names'][arm]]+['finger'+str(i) for i in range(7)]
            native[arm+'_native_joint_names']=np.asarray(names)
            native[arm+'_controlled_joint_indices']=np.arange(6)
            native[arm+'_native_q'][:,6]=np.float32(.349)
            native[arm+'_native_qd'][:,6]=np.float32(.01)
        idx,observed=a.opening_binding(opening,native)
        self.assertEqual(idx,{'U_L':6,'U_R':6})
        self.assertEqual(observed['U_R']['q'][63],float(np.float32(.349)))
        self.assertEqual(observed['U_R']['qd'][63],float(np.float32(.01)))
        for field,value in [('all_modes_common',False),('open_thumb_rad',.4),('constructor_contacts_certified',True)]:
            with self.subTest(field=field):
                changed=copy.deepcopy(opening);changed[field]=value
                with self.assertRaises(a.InvalidEvidence):a.opening_binding(changed,native)
        native['U_L_controlled_joint_indices'][-1]=6
        with self.assertRaises(a.InvalidEvidence):a.opening_binding(opening,native)

    def test_audit_wait_binds_exact_script_root_output_log(self):
        root=Path('/synthetic_only');output=H/'b0_adaptive_joint_geometry.json'
        path=H/'synthetic_only_execution.json'
        receipt=dict(argv=['python','-B',str(H/'audit_geometry_v2.py'),str(root),output.stem],
            actual_exit=0,pid=321,started_utc='2026-10-07T16:00:00+00:00',
            closed_utc='2026-10-07T16:01:00+00:00',tag='synthetic_only',log_sha256='log')
        class L:
            def json(self,p):return receipt
            def bind(self,p,digest):
                if p != H/'synthetic_only.log' or digest != 'log':raise AssertionError('log binding')
        with patch.object(Path,'glob',return_value=[path]),patch.object(Path,'read_text',side_effect=lambda:json.dumps(receipt)):
            a.check_audit_wait(H,L(),'audit_geometry_v2.py',root,output,'2026-10-07T15:59:00+00:00')
            del receipt['actual_exit']
            with self.assertRaises(a.IncompleteEvidence):
                a.check_audit_wait(H,L(),'audit_geometry_v2.py',root,output,'2026-10-07T15:59:00+00:00')
            receipt['actual_exit']=0;receipt['argv'][-1]='wrong_output'
            with self.assertRaises(a.IncompleteEvidence):
                a.check_audit_wait(H,L(),'audit_geometry_v2.py',root,output,'2026-10-07T15:59:00+00:00')

    def test_bank_qualification_requires_wait_and_pre_outcome_completion(self):
        root=Path('/synthetic_only/banks')
        plan=dict(registered_utc='2026-10-07T12:00:00+00:00',policy_outcomes_used=False,
            cases_per_bank=64,seeds=[1101342169,25496091],sources={},argv=['never_executed'],env={},out=str(root))
        execution=dict(plan_sha256='plan',actual_exit=0,pid=123,status='complete',
            started_utc='2026-10-07T13:00:00+00:00',child_closed_utc='2026-10-07T14:00:00+00:00',
            closed_utc='2026-10-07T14:01:00+00:00',argv=plan['argv'],env={},out=str(root),log_sha256='log')
        metadata=dict(status='complete',selected_count=64,policy_outcomes_used=False,bank_sha256='bank')
        class L:
            def json(self,p,expected=None):
                return {'bank_qualification_plan.json':plan,'bank_qualification_execution.json':execution,
                        'metadata.json':metadata}[p.name]
            def bind(self,p,expected=None):return 'plan' if p.name=='bank_qualification_plan.json' else expected
        sources={str(root/str(seed)/name):name for seed in plan['seeds'] for name in ('bank.npz','metadata.json')}
        reg=dict(registered_utc='2026-10-07T15:00:00+00:00')
        self.assertEqual(len(a.check_banks(H,L(),CRITERIA,sources,reg)),2)
        execution['actual_exit']=-15
        with self.assertRaises(a.InvalidEvidence):a.check_banks(H,L(),CRITERIA,sources,reg)
        execution['actual_exit']=0;reg['registered_utc']='2026-10-07T13:59:00+00:00'
        with self.assertRaises(a.InvalidEvidence):a.check_banks(H,L(),CRITERIA,sources,reg)

    def test_cli_incomplete_and_rejected_have_distinct_exit_semantics(self):
        class Output(io.StringIO):
            def close(self):pass
        scenarios=[(a.IncompleteEvidence('no actual wait'),2,'INCOMPLETE_NOT_ACCEPTED'),
            (a.InvalidEvidence('SHA mismatch'),1,'INVALID_EVIDENCE_NOT_ACCEPTED'),
            (dict(status='COMPLETE_INDEPENDENT_FINITE_DECISION',candidate_decision='REJECTED'),0,
             'COMPLETE_INDEPENDENT_FINITE_DECISION')]
        for result,expected,status in scenarios:
            with self.subTest(status=status):
                output=Output()
                opts={'side_effect':result} if isinstance(result,Exception) else {'return_value':result.copy()}
                with patch.object(a,'check',**opts),patch.object(Path,'open',return_value=output),patch.object(a,'sha',return_value='source'):
                    code=a.main(['/synthetic_only','/synthetic_only/new_receipt.json'])
                receipt=json.loads(output.getvalue())
                self.assertEqual(code,expected);self.assertEqual(receipt['status'],status)
                self.assertEqual(receipt['queued_future_status'],'UNKNOWN')
                self.assertFalse(receipt['physical_safety_certified'])

    def test_cli_refuses_existing_output_before_scoring(self):
        with patch.object(Path,'open',side_effect=FileExistsError('exclusive output')),patch.object(a,'check') as check:
            with self.assertRaises(FileExistsError):a.main(['/synthetic_only','/synthetic_only/existing.json'])
            check.assert_not_called()


class RawAndAuditorTests(unittest.TestCase):
    def test_hand_manifest_binds_all_images_states_native_mapping_and_group_final(self):
        f=hand_fixture();result=a.hand_camera_binding(*f)
        self.assertEqual(result['hand_PNG'],252);self.assertEqual(result['total_PNG'],441)
        self.assertTrue(result['all_raw_references_bound'])
        self.assertEqual(set(f[1].hashes),set(f[1].files))
        self.assertEqual(sum(p.endswith('.png') for p in f[1].hashes),252)

    def test_hand_receipt_wrong_cell_source_counts_scope_or_partial(self):
        mutations=[('frames',90),('windows',63),('groups',20),('hand_PNG',251),('total_PNG',440),
            ('source_sha256','wrong'),('root','/wrong'),('cell_sha256','wrong'),
            ('all64_native_bitwise_render_invariant',False),('exact_contact_sensor_bijection',3327),
            ('full_mesh_certified',True),('occlusion_certified',True),('hardware_certified',True)]
        for key,value in mutations:
            with self.subTest(key=key):
                f=hand_fixture();f[2][key]=value
                with self.assertRaises(a.InvalidEvidence):a.hand_camera_binding(*f)
        f=hand_fixture();f[2]['status']='running'
        with self.assertRaises(a.IncompleteEvidence):a.hand_camera_binding(*f)

    def test_hand_missing_or_tampered_raw_reference_never_passes(self):
        suffixes=('/state.json','.png','native_before_all64.npz','native_after_all64.npz',
                  'identity_mapping.json','native_final_all64.npz')
        for suffix in suffixes:
            for missing in (True,False):
                with self.subTest(suffix=suffix,missing=missing):
                    f=hand_fixture();ledger=f[1];path=next(p for p in ledger.files if p.endswith(suffix))
                    if missing:del ledger.files[path]
                    else:ledger.files[path]+=b'tampered'
                    with self.assertRaises((a.InvalidEvidence,a.IncompleteEvidence)):a.hand_camera_binding(*f)

    def test_hand_group_final_and_top_manifest_have_closure_hashes(self):
        for suffix in ('hand_camera_receipts.json','/receipts.json'):
            f=hand_fixture();ledger=f[1];a.hand_camera_binding(*f)
            path=next(p for p in ledger.files if p.endswith(suffix))
            ledger.files[path]+=b' '
            with self.assertRaises(a.InvalidEvidence):ledger.verify()

    def test_hand_group_identity_must_match_same_principal_group(self):
        f=hand_fixture();ledger=f[1];path=f[0]/'hand_camera_receipts.json'
        raw=json.loads(ledger.files[str(path)]);raw['receipts'][0]['env_id']=1;ledger.put(path,raw)
        with self.assertRaises(a.InvalidEvidence):a.hand_camera_binding(*f)

    def test_hand_raw_top_manifest_missing_or_wrong_png_count(self):
        f=hand_fixture();path=f[0]/'hand_camera_receipts.json';del f[1].files[str(path)]
        with self.assertRaises(a.IncompleteEvidence):a.hand_camera_binding(*f)
        f=hand_fixture();path=f[0]/'hand_camera_receipts.json';raw=json.loads(f[1].files[str(path)])
        raw['PNG']=251;f[1].put(path,raw)
        with self.assertRaises(a.InvalidEvidence):a.hand_camera_binding(*f)

    def test_every_camera_raw_hash_must_be_in_peer_manifest(self):
        root=Path('/synthetic_only')
        state=dict(images=[dict(path='view.png',sha256='png')],fresh_native_snapshot='before.npz',
            fresh_native_before_sha256='before',fresh_native_after_snapshot='after.npz',fresh_native_after_sha256='after')
        capture=dict(state='state.json',sha256='state',images=state['images'])
        docs={name:dict(chunks=[dict(path=name+'.npz',sha256=name)]) for name in
              ('forecast_receipts.json','native_receipts.json','native_contact_receipts.json')}
        raw={name+'.npz':name for name in docs}
        docs.update({'state.json':state,'camera_receipts.json':dict(receipts=[capture])})
        raw.update({'state.json':'state','view.png':'png','before.npz':'before','after.npz':'after'})
        ledger=MemoryLedger(docs,{})
        a.check_peer_manifest_coverage(root,ledger,raw)
        for key in ('view.png','state.json','before.npz','after.npz','native_receipts.json.npz'):
            with self.subTest(key=key):
                changed=dict(raw);del changed[key]
                with self.assertRaises(a.InvalidEvidence):a.check_peer_manifest_coverage(root,ledger,changed)

    def test_native_float32_thresholds_zero_minus5mm(self):
        cell, m=measured_fixture()
        m[0,0,0]=np.nextafter(np.float32(0),np.float32(-1))
        m[0,1,0]=np.float32(-.005)
        m[0,2,0]=np.nextafter(np.float32(-.005),np.float32(-1))
        cell['official_margins']=m.copy();cell['official_deep']=m<np.float32(-.005)
        s=a.score_and_motion(cell,m)
        self.assertEqual(np.flatnonzero(s['strict']).tolist(),[0,1,2])
        self.assertEqual(np.flatnonzero(s['deep']).tolist(),[2])

    def test_epsilon_relaxed_official_score_rejected(self):
        cell,m=measured_fixture();m[0,0,0]=-1e-12
        cell['official_margins'][0,0,0]=0
        with self.assertRaises(a.InvalidEvidence):a.score_and_motion(cell,m)

    def test_auditors_agree_with_independent_raw_recomputation(self):
        a.audit_agreement(*audit_fixture())

    def test_auditor_summary_mutations(self):
        mutations=[('parent geometry',4,'strict_windows',1),('peer geometry',7,'deep_env_frames',1),
            ('initial count',4,'initial_negative_envs',1),('initial IDs',4,'initial_negative_ids',[0]),
            ('initial minima',4,'initial_raw_minimum_by_class_m',[-1,0,0,0]),
            ('wrong cell',5,'cell_sha256','other'),('wrong root',4,'root','/wrong'),
            ('missing window',4,'windows',63),('missing frame',5,'frames',959),
            ('missing microstep',6,'physics_events',1919),('epsilon',7,'score_threshold_epsilon',1e-6),
            ('future called safe',7,'queued_future_status','SAFE'),('wrong FIFO',5,'actual_FIFO6_exact',False),
            ('native boundary omitted',5,'boundary_packets',1920),('native hand exemption',6,'exemption_adjusted_contact_gate',True),
            ('camera omitted',7,'PNG',180),('capacity mismatch',6,'peak_contact_count',1),
            ('wrong peer source',7,'auditor_sha256','wrong'),('fake peer zero',7,'exitcode',False)]
        for name,index,key,value in mutations:
            with self.subTest(name=name):
                f=audit_fixture();f[index][key]=value
                with self.assertRaises(a.InvalidEvidence):a.audit_agreement(*f)

    def test_running_auditor_never_passes(self):
        f=audit_fixture();f[7]['status']='running'
        with self.assertRaises(a.IncompleteEvidence):a.audit_agreement(*f)

    def test_old_geometry_receipt_without_initial_fields_cannot_pass(self):
        f=audit_fixture();del f[4]['initial_negative_envs']
        with self.assertRaises(KeyError):a.audit_agreement(*f)

    def test_two_auditors_same_wrong_count_does_not_override_raw(self):
        f=audit_fixture();f[4]['strict_windows']=f[7]['strict_windows']=1
        with self.assertRaises(a.InvalidEvidence):a.audit_agreement(*f)

    def test_samehand_filtered_channel_cannot_mask_raw_overlimit(self):
        f=audit_fixture();f[3][12,0,9,0]=.2
        f[7]['contacts']['non_same_hand_windows_over_0p1N']=0
        with self.assertRaises(a.InvalidEvidence):a.audit_agreement(*f)

    def test_zero_net_cancellation_does_not_hide_partner_force(self):
        matrix=np.zeros((64,2,3),np.float32)
        matrix[7,0,0]=.2;matrix[7,1,0]=-.2
        self.assertEqual(float(matrix[7].sum(0).max()),0)
        peaks=a.normal_peaks(matrix,np.arange(64),2)
        self.assertGreater(peaks[7],.1)

    def test_measured_q_not_command_motion(self):
        cell,m=measured_fixture();cell['q'][:]=0
        cell['controller_target']=np.ones_like(cell['q'])
        s=a.score_and_motion(cell,m)
        self.assertEqual(float(s['path'].sum()),0)
        self.assertEqual(s['four'],0)
        self.assertEqual(float(s['ranges'].sum()),0)

    def test_raw_full9021_pre_to_post_and_terminal(self):
        with patch.object(a,'T',3):
            cell,m=measured_fixture()
            classes=np.zeros(9021,dtype=int);classes[1:3]=1;classes[3]=2
            pairs=np.zeros((9021,2),int);pairs[2,0]=2
            identity=dict(rows=9021,class_id=classes,pair_id=np.arange(9021),
                          sphere_arm_id=np.arange(4),pair_sphere_idx=pairs)
            d=np.full((3,64,9021),.01,np.float32);d[1,0,0]=-1e-12
            d[0,11,0]=-.007  # initial adoption gate is separate; no post-score rewrite
            d[0,12,0]=np.nextafter(np.float32(0),np.float32(-1))
            d[0,13,0]=0
            d[0,14,0]=-.02  # original packed exemption must still apply at t0
            final=np.full((64,9021),.01,np.float32);final[63,3]=-.006
            ex=np.zeros((64,9021),bool)
            cell['official_margins'][0,0,0]=-1e-12;cell['official_margins'][-1,63,3]=-.006
            cell['official_deep']=cell['official_margins']<np.float32(-.005)
            docs={'full_row_identity.json':identity,'forecast_receipts.json':dict(steps=3,envs=64,rows=9021,
                  chunks=[dict(path='chunk.npz',start=0,stop=3,sha256='synthetic')])}
            packed=np.zeros((3,64,1128),np.uint8);packed[0,14,0]=128
            arrays={'chunk.npz':dict(measured_d=d,exempt=packed),
                    'post_geometry_final.npz':dict(d=final,exempt=ex)}
            cell['initial_violation']=np.zeros(64,bool)  # cannot override actual rawt0
            s=a.raw_geometry(Path('/synthetic_only'),MemoryLedger(docs,arrays),cell)
            self.assertEqual(np.flatnonzero(s['strict']).tolist(),[0,63])
            self.assertEqual(np.flatnonzero(s['deep']).tolist(),[63])
            self.assertEqual(s['first'][63],2)
            self.assertEqual(s['initial_strict_ids'],[11,12])
            self.assertEqual(s['initial_deep_ids'],[11])
            self.assertEqual(s['initial_minimum_by_class_m'][0],np.float32(-.007))
            docs['forecast_receipts.json']['chunks'][0]['stop']=2
            with self.assertRaises(a.InvalidEvidence):a.raw_geometry(Path('/synthetic_only'),MemoryLedger(docs,arrays),cell)

    def test_raw_contact_both_substeps_and_last_event(self):
        with patch.object(a,'T',3):
            views=[dict(arm=arm,env_ids=list(range(64)),filters=[['samehand','ground']]*64) for arm in a.ARMS]
            docs={'native_contact_receipts.json':dict(status='complete',control_steps=3,physics_events=6,
                  substeps_per_control=2,capacity=262144,identity_sha256='synthetic',
                  chunks=[dict(path='contacts.npz',start=0,stop=6,sha256='synthetic')]),
                  'native_contact_identity.json':dict(views=views,environment_count=64,physics_dt_s=.008333)}
            z=dict(frame=np.repeat(np.arange(3),2),substep=np.tile([0,1],3),physics_dt=np.full(6,.008333))
            for arm in a.ARMS:z[arm+'_partner_normal']=np.zeros((6,64,2,3),np.float32)
            z['F_L_partner_normal'][0,0,0,0]=.2
            z['U_R_partner_normal'][5,63,1,2]=.3
            ledger=MemoryLedger(docs,{'contacts.npz':z})
            metrics=a.raw_contacts(Path('/synthetic_only'),ledger)
            self.assertGreater(metrics[0,0,0,0],.1)
            self.assertGreater(metrics[2,1,63,3],.1)
            self.assertEqual(int((metrics>.1).sum()),2)
            for arm in ('U_L','U_R'):z[arm+'_native_position_targets']=np.full((6,64,14),.35,np.float32)
            a.raw_contacts(Path('/synthetic_only'),ledger,{'U_L':6,'U_R':7})
            z['U_R_native_position_targets'][5,63,7]=np.nextafter(np.float32(.35),np.float32(1))
            with self.assertRaises(a.InvalidEvidence):a.raw_contacts(Path('/synthetic_only'),ledger,{'U_L':6,'U_R':7})
            z['substep'][-1]=0
            with self.assertRaises(a.InvalidEvidence):a.raw_contacts(Path('/synthetic_only'),ledger)

    def test_frozen_controller_hashes_unchanged(self):
        self.assertEqual(a.sha(H/'adaptive_reference.py'),'562da73abffb124e8e39908ec973a2de6f2712ad5ee1d1f191ccec7f4754868e')
        self.assertEqual(a.sha(H/'test_adaptive_reference.py'),'00309cc904c70b77ae262f135e175cc3d038b2849b569a6111820319e374f6f2')
        self.assertEqual(a.sha(H/'astra_native_audit.py'),'ea94c3e0d46ab1ce862291d20bae6206216de03b5f4fbc07040942705211ac10')


if __name__=='__main__':
    unittest.main(verbosity=2)
