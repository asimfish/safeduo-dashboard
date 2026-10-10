"""Closed CPU counterfactuals for failure-only diagnostics; never native PASS."""
from evidence_io import cpu_limits
cpu_limits()
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import types
import unittest
from unittest.mock import patch
import numpy as np
from evidence_io import HERE,OUTPUT,ARMS,Evidence,sha,write_new,contained,parse_json,fingerprint
from failed_closure import terminal_numbers,terminal_gate,outer_failure,PendingClosure
from cpu_supervisor import wait_cpu
from initial_reader import initial_only_inventory,actual_entry_readbacks
from failed_camera import selected_metrics,wide_metrics,inspect_attempt,audit_all,pixels_without_hold
from camera_oracle import hand_group,token
from root_restore_reader import evaluate,verify_report
import test_science_baseline as baseline
ROOT=None


def native_wait_fixture(exit_code=0,raw=0):
    return dict(schema='astra.full74.parent_native_wait.v1',backend='isaac_physx_native',child_pid=123,
        waited_pid=123,raw_wait_status=raw,actual_wait_exit=exit_code,waitpid_observed=True,
        wait_mechanism='os.wait4(owned_pid, WNOHANG)',
        owned_identities=[dict(pid=123,ppid=122,pgid=1,sid=123,startticks=500,live=False,
            enrollment=dict(kind='direct_fork_handshake',parent_pid=122))])


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.root=ROOT/self._testMethodName;self.root.mkdir()

    def test_terminal_zero_negative_positive_actual_exit_truth(self):
        for exit_code,raw in [(0,0),(-9,9),(-15,15),(1,256),(7,1792)]:
            with self.subTest(code=exit_code):self.assertEqual(terminal_numbers(native_wait_fixture(exit_code,raw))['startticks'],500)

    def test_raw_status_mismatch_snapshot_stopped_and_bool_denied(self):
        for key,value in [('actual_wait_exit',None),('raw_wait_status',0x137f),('waitpid_observed',False),
                          ('waited_pid',124),('actual_wait_exit',False),('raw_wait_status',9)]:
            w=native_wait_fixture();w[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):terminal_numbers(w)

    def test_live_descendant_and_bad_enrollment_denied(self):
        for which in ('root_live','descendant','startticks','enrollment'):
            w=native_wait_fixture();own=w['owned_identities'][0]
            if which=='root_live':own['live']=True
            if which=='descendant':w['owned_identities'].append(dict(pid=124,live=True))
            if which=='startticks':own['startticks']=0
            if which=='enrollment':own['enrollment']['parent_pid']=121
            with self.subTest(which=which),self.assertRaises(ValueError):terminal_numbers(w)

    def test_previous_attempt_and_cpu_fixtures_denied_before_reads(self):
        anchors=Evidence().js(HERE/'anchors.json');e=Evidence()
        with self.assertRaisesRegex(ValueError,'exact failed V9 root'):
            terminal_gate(e,anchors,self.root,'0'*64,'0'*64,'0'*64)
        self.assertEqual(e.ledger,{})

    def test_missing_wait_denied_without_product_read(self):
        anchors=Evidence().js(HERE/'anchors.json');root=Path(anchors['actual_attempt_root']);e=Evidence()
        with patch('failed_closure.Path.is_file',return_value=False),self.assertRaises(PendingClosure):
            terminal_gate(e,anchors,root,'0'*64,'0'*64,'0'*64)
        self.assertEqual(e.ledger,{})

    def outer_fixture(self):
        anchors=Evidence().js(HERE/'anchors.json');p=self.root/'outer.json';log=self.root/'outer.log';log.write_bytes(b'CPU fixture\n')
        anchors['outer_execution_path']=str(p);anchors['outer_log_path']=str(log)
        value=dict(tag='full74_screen_parent_failure_v9',actual_exit=1,pid=anchors['actual_native_parent_pid'],
            argv=['CPU','-B',str(HERE.parents[1]/'launch_full74_screen_after_failure_v9.py')],
            started_utc='2026-10-10T01:00:00+00:00',closed_utc='2026-10-10T01:10:00+00:00',log_sha256=sha(log))
        w=dict(started_utc='2026-10-10T01:01:00+00:00',closed_utc='2026-10-10T01:09:00+00:00')
        return anchors,p,value,w

    def test_outer1_real_exit_required(self):
        anchors,p,value,w=self.outer_fixture();write_new(p,value)
        self.assertEqual(outer_failure(Evidence(),w,anchors,sha(p))['actual_exit'],1)
        for code in (0,None,False):
            value['actual_exit']=code;q=self.root/('outer_'+str(code)+'.json');write_new(q,value);anchors['outer_execution_path']=str(q)
            with self.assertRaisesRegex(ValueError,'actual failed outer1'):outer_failure(Evidence(),w,anchors,sha(q))

    def test_outer_temporal_or_identity_replay_denied(self):
        anchors,p,value,w=self.outer_fixture()
        for key,val in [('pid',1),('started_utc','2026-10-10T02:00:00+00:00')]:
            x=dict(value);x[key]=val;q=self.root/(key+'.json');write_new(q,x);anchors['outer_execution_path']=str(q)
            with self.assertRaises(ValueError):outer_failure(Evidence(),w,anchors,sha(q))

    def test_real_owned_zero_seven_and_premature_zero_never_native(self):
        for mode,code,raw in [('zero',0,0),('seven',7,1792),('premature-zero',0,0)]:
            w=wait_cpu(HERE/'fixture_child.py',[mode],self.root/mode,20)
            self.assertEqual((w['actual_wait_exit'],w['raw_wait_status']),(code,raw));self.assertEqual(w['waited_pid'],w['child_pid'])
            self.assertIsNone(w['resource_abort']);self.assertFalse(w['native_pass']);self.assertGreater(w['child_identity']['startticks'],0)
            with self.assertRaises(ChildProcessError):os.waitpid(w['child_pid'],os.WNOHANG)
            with self.assertRaisesRegex(ValueError,'schema/backend'):terminal_numbers(w)

    def test_supervisor_rejects_native_executable(self):
        with self.assertRaises(ValueError):wait_cpu(HERE/'binding_kernels.py',[],self.root/'denied',10)
        self.assertFalse((self.root/'denied').exists())

    def test_public_cli_cannot_turn_cpu_fixture_into_native(self):
        args=['--attempt-root',str(self.root),'--parent-wait-sha256','0'*64,'--request-sha256','0'*64,
            '--outer-execution-sha256','0'*64,'--output',str(self.root/'child/REPORT.json')]
        w=wait_cpu(HERE/'failed_reader.py',args,self.root/'child',30)
        self.assertEqual(w['actual_wait_exit'],2);r=Evidence().js(self.root/'child/REPORT.json')
        self.assertFalse(r['native_pass']);self.assertFalse(r['safety_acceptance']);self.assertFalse(r['prefix12_completed'])

    def test_row0_only_ignores_uninitialized_nan_tail(self):
        a=np.full((13,32,74),np.nan,np.float32);a[0]=3;p=self.root/'q.npy';baseline.save_npy(p,a)
        e=Evidence();row=e.npy_first_row(p,a.shape,a.dtype);self.assertTrue(np.all(row==3));self.assertEqual(row.shape,(32,74));e.recheck()
        a[12,31,73]=9
        with p.open('wb') as f:np.save(f,a,allow_pickle=False)
        with self.assertRaisesRegex(ValueError,'changed'):e.recheck()

    def test_row0_wrong_shape_dtype_fortran_or_truncated_rejected(self):
        for mode in ('shape','dtype','fortran','truncated'):
            a=np.zeros((13,32,74),np.float32)
            if mode=='shape':a=a[:12]
            if mode=='dtype':a=a.astype(np.float64)
            if mode=='fortran':a=np.asfortranarray(a)
            p=self.root/(mode+'.npy');baseline.save_npy(p,a)
            if mode=='truncated':p.write_bytes(p.read_bytes()[:-1])
            with self.subTest(mode=mode),self.assertRaises(ValueError):Evidence().npy_first_row(p,(13,32,74),'float32')

    def test_row0_scalar_native_clock(self):
        p=self.root/'clock.npy';baseline.save_npy(p,np.arange(13,dtype=np.int64))
        a=Evidence().npy_first_row(p,(13,),'int64');self.assertEqual(a.shape,());self.assertEqual(int(a),0)

    def initial_inventory_fixture(self):
        batch=self.root/'native/batch_00000';(batch/'native_fields').mkdir(parents=True)
        write_new(batch/'native_fields/row_000_complete.json',{'row':0})
        return batch

    def test_one_initial_row_is_not_prefix_completion(self):
        r=initial_only_inventory(Evidence(),self.root,self.initial_inventory_fixture())
        self.assertEqual(r['observed_prefix_steps'],0);self.assertFalse(r['prefix12_completed']);self.assertFalse(r['final_camera_present'])

    def test_any_prefix_or_final_product_denies_initial_only_scope(self):
        batch=self.initial_inventory_fixture()
        for name in ('points_001','applied_target74_micro_001.npy','final_camera_receipts.json','BATCH_CLOSED.json','native_fields/row_001_complete.json'):
            p=batch/name;p.write_bytes(b'CPU fixture')
            with self.subTest(name=name),self.assertRaises(ValueError):initial_only_inventory(Evidence(),self.root,batch)
            p.unlink()

    def test_missing_actual_parameters_and_geometry_are_unknown(self):
        r=actual_entry_readbacks(Evidence(),self.root,{'parameter_readbacks':{}},{'CPU_REFERENCE_ONLY':np.ones(1)})
        self.assertEqual([x['status'] for x in r['actual_parameter_readbacks'].values()],['UNKNOWN_MISSING_ACTUAL_ARTIFACT']*2)
        self.assertFalse(r['full_native_parameters_certified']);self.assertFalse(r['frozen_reference_parameters_are_actual_readback'])
        self.assertEqual(r['native_geometry_identity']['status'],'UNKNOWN_MISSING_ACTUAL_ARTIFACT')

    def test_present_parameter_requires_actual_entry_sha(self):
        p=self.root/'PARAMETER_READBACK_BEFORE_V1.npz'
        with p.open('xb') as f:np.savez(f,p=np.ones(1))
        with self.assertRaisesRegex(ValueError,'unbound'):actual_entry_readbacks(Evidence(),self.root,{},dict(p=np.ones(1)))
        entry={'parameter_readbacks':{'before':dict(path=str(p),sha256=sha(p))}}
        r=actual_entry_readbacks(Evidence(),self.root,entry,dict(p=np.ones(1)))
        self.assertEqual(r['actual_parameter_readbacks']['BEFORE']['status'],'RECORDED_BITWISE_MATCH_ONLY')
        self.assertEqual(r['actual_parameter_readbacks']['AFTER']['status'],'UNKNOWN_MISSING_ACTUAL_ARTIFACT')
        with self.assertRaises(ValueError):actual_entry_readbacks(Evidence(),self.root,entry,dict(p=np.zeros(1)))

    def test_failed_group_without_selected_trio_is_valid_insufficiency(self):
        r=selected_metrics(dict(status='failed',qualified_images=[],selected_group_images=[],group_coverage=None),[],lambda r:None)
        self.assertEqual(r['integrity_errors'],[]);self.assertFalse(r['selected_raw_predicate_sufficient']);self.assertFalse(r['acceptance'])

    def test_failed_group_not_promoted_by_sufficient_pixels(self):
        records=[dict(rgb=dict(path=str(i),sha256='CPU'),_independent_azimuth=a) for i,a in enumerate((0,60,120))]
        d=dict(status='failed',qualified_images=[],selected_group_images=[x['rgb'] for x in records],group_coverage={'status':'qualified','n':9})
        r=selected_metrics(d,records,lambda r:dict(qualified=True,n=9))
        self.assertTrue(r['selected_raw_predicate_sufficient']);self.assertEqual(r['producer_status'],'failed');self.assertFalse(r['acceptance'])

    def test_failed_group_preserves_bad_palm_predicate(self):
        records=[dict(rgb=dict(path=str(i),sha256='CPU'),_independent_azimuth=a) for i,a in enumerate((0,60,120))]
        d=dict(status='failed',qualified_images=[],selected_group_images=[x['rgb'] for x in records],group_coverage={'status':'failed','palm':4})
        r=selected_metrics(d,records,lambda r:dict(qualified=False,palm=4))
        self.assertEqual(r['integrity_errors'],[]);self.assertFalse(r['selected_raw_predicate_sufficient'])

    def test_forged_coverage_or_qualified_status_detected(self):
        records=[dict(rgb=dict(path=str(i),sha256='CPU'),_independent_azimuth=a) for i,a in enumerate((0,60,120))]
        refs=[x['rgb'] for x in records];d=dict(status='qualified',qualified_images=refs,selected_group_images=refs,group_coverage={'status':'qualified','palm':2048})
        r=selected_metrics(d,records,lambda r:dict(qualified=False,palm=1))
        self.assertGreaterEqual(len(r['integrity_errors']),3)

    def test_coverage_view_reselection_never_occurs(self):
        records=[dict(rgb=dict(path=str(i),sha256='CPU'),_independent_azimuth=a) for i,a in enumerate((0,30,29,90))]
        d=dict(status='failed',qualified_images=[],selected_group_images=[x['rgb'] for x in records[:3]])
        r=selected_metrics(d,records,lambda r:dict(qualified=True))
        self.assertTrue(r['integrity_errors']);self.assertIsNone(r['selected_coverage'])

    def test_independent_root_quaternion_antipodes_and_large_rotation(self):
        anchors=Evidence().js(HERE/'anchors.json');protocol=Evidence().js(anchors['root_protocol']['path'],anchors['root_protocol']['sha256'])
        a=np.zeros((32,7),np.float32);a[:,6]=1;b=a.copy();b[:,3:]*=-1;v=np.zeros((32,6),np.float32)
        r=evaluate(a,b,v,v,protocol);self.assertEqual(len(r['lanes']),32);self.assertEqual(r['summary']['orientation_max_error_rad'],0)
        b[31,3]=1e-4
        with self.assertRaisesRegex(ValueError,'orientation'):evaluate(a,b,v,v,protocol)

    def test_root_report_false_zero_denied(self):
        anchors=Evidence().js(HERE/'anchors.json');protocol=Evidence().js(anchors['root_protocol']['path'],anchors['root_protocol']['sha256'])
        a=np.zeros((32,7),np.float32);a[:,6]=1;b=a.copy();b[31,0]=5e-8;v=np.zeros((32,6),np.float32)
        r=evaluate(a,b,v,v,protocol);reported=dict(r['summary']);reported['position_max_abs_error_m']=0
        with self.assertRaisesRegex(ValueError,'mismatch'):verify_report(r,reported)

    def test_held_failure_still_decodes_raw_pixels_and_records_failure(self):
        helper=baseline.ReaderTests();helper.root=self.root
        rec,state,audit,nz,rgb=helper.camera_bytes_fixture();wrong={k:v.copy() for k,v in state.items()};wrong['U_R_native_position_targets'][31,17]=1
        rec['held_renders'][2]['native_after_all64']=nz('bad.npz',wrong)
        rec.update(attempt=1,arm='F_L',hand_link_positions_world_m=[[0,0,0]],azimuth_deg=0.)
        rec['camera']={};p=self.root/'attempt.json';write_new(p,rec);embedded=dict(rec,attempt_record=dict(path=p.name,sha256=sha(p)))
        _,r=inspect_attempt(audit,embedded,state,np.array([[0,0,0]]),{}, {},{},0,'hand')
        self.assertTrue(r['raw_pixels_decoded']);self.assertFalse(r['held27_verified']);self.assertTrue(r['integrity_errors'])

    def test_all32_groups_visited_when_one_has_integrity_error(self):
        batch=self.root/'batch';cap=batch/'qualified_views/initial/capture';cap.mkdir(parents=True)
        receipts=[]
        for i in range(32):
            p=cap/f'env_{i:03d}/state.json';write_new(p,{'CPU_fixture':i})
            receipts.append(dict(env_id=i,step=-1,substep=None,capture_kind='initial',status='failed',reasons=[],
                arms={},wide_context=None,image_count=0,qualified_image_count=0,scope='CPU_fixture',state=str(p.relative_to(batch)),sha256=sha(p)))
        write_new(batch/'initial_camera_receipts.json',dict(qualified=False,receipts=receipts))
        write_new(cap/'receipts.json',dict(receipts=receipts,native_invariant={},native_before_all64=None,native_final_all64=None,scope='CPU_fixture'))
        initial=dict(batch=batch,body={},identity={},layout={},fields={});visited=[]
        def group(audit,r,*unused):
            visited.append(r['env_id'])
            if r['env_id']==7:raise ValueError('CPU corruption')
            return dict(env_id=r['env_id'],producer_status='failed',raw_selected_predicate_sufficient=False,
                original_png_references=[],original_png_count=0,raw_pixel_attempts_decoded=0,integrity_errors=[]),np.zeros((32,3),np.float32)
        with patch('failed_camera.inspect_group',side_effect=group):result=audit_all(Evidence(),initial,self.root/'out')
        self.assertEqual(visited,list(range(32)));self.assertEqual(result['groups_visited'],32);self.assertFalse(result['complete_raw_diagnostic'])
        self.assertFalse(result['native_pass']);self.assertFalse(result['safety_acceptance'])

    def test_immutable_kernels_and_closed_v9_package_unchanged(self):
        anchors=Evidence().js(HERE/'anchors.json')
        for name,digest in anchors['immutable_reused_sources'].items():self.assertEqual(sha(HERE/name),digest)
        b=anchors['binding_kernels_source'];self.assertEqual(sha(HERE/'binding_kernels.py'),b['sha256']);self.assertEqual(sha(b['path']),b['sha256'])
        old=Evidence().js(OUTPUT/'V9_PRESERVATION_BEFORE.json')
        for path,record in old['files'].items():
            now=fingerprint(path);now.pop('path');self.assertEqual(now,record,path)
        for path,target in old['links'].items():self.assertEqual(os.readlink(path),target)
        write_new(self.root/'PRESERVED_V9.json',dict(files=len(old['files']),symlinks=len(old['links']),native_pass=False))

    def test_whole_group_original_raw_pixels_masks_and_held27(self):
        from camera_fixture import fixture
        from failed_camera import inspect_group
        audit,receipt,fields,asset=fixture(self.root)
        r,origins=inspect_group(audit,receipt,fields,asset,None)
        self.assertEqual(r['integrity_errors'],[]);self.assertEqual(r['original_png_count'],15)
        self.assertEqual(r['raw_pixel_attempts_decoded'],15);self.assertTrue(r['raw_selected_predicate_sufficient'])
        self.assertTrue(r['strong_camera_supplement_v3']['observed_coverage_sufficient'])
        self.assertEqual(audit.snapshots,122);self.assertFalse(r['native_pass']);self.assertFalse(r['safety_acceptance'])
        write_new(self.root/'SYNTHETIC_DIAGNOSTIC.json',r)

    def test_whole_failed_group_palm_masks_remain_insufficient(self):
        from camera_fixture import fixture
        from failed_camera import inspect_group
        audit,receipt,fields,asset=fixture(self.root,weak_palm=True)
        r,origins=inspect_group(audit,receipt,fields,asset,None)
        self.assertEqual(r['integrity_errors'],[]);self.assertEqual(r['raw_pixel_attempts_decoded'],15)
        self.assertEqual(r['arms']['F_L']['selected_coverage']['palm_visual_peak_pixels'],0)
        self.assertFalse(r['raw_selected_predicate_sufficient']);self.assertEqual(r['producer_status'],'failed')
        self.assertFalse(r['native_pass']);write_new(self.root/'SYNTHETIC_DIAGNOSTIC.json',r)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--proof-dir',required=True);a=p.parse_args()
    ROOT=contained(a.proof_dir,OUTPUT);ROOT.mkdir(parents=True,exist_ok=False);baseline.ROOT=ROOT
    import test_camera_index
    test_camera_index.ROOT=ROOT
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(baseline.ReaderTests),
        unittest.defaultTestLoader.loadTestsFromTestCase(DiagnosticTests),unittest.defaultTestLoader.loadTestsFromTestCase(test_camera_index.IndexTests)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    write_new(ROOT/'CPU_TEST_REPORT.json',dict(schema='astra.full74.failed_initial_CPU_tests.v1',
        status='CLOSED_CPU_TESTS_ONLY' if result.wasSuccessful() else 'FAILED_CPU_TESTS',tests_run=result.testsRun,
        failures=[(str(t),v) for t,v in result.failures],errors=[(str(t),v) for t,v in result.errors],
        native_pass=False,native_executions=0,safety_acceptance=False,fixtures='synthetic CPU only; all32 traversal test mocks only group dispatch',
        max_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,address_space_bytes=1024**3))
    raise SystemExit(0 if result.wasSuccessful() else 2)
