"""CPU synthetic oracles for native scoring, pairing, time binding and failure gates.

No campaign outcomes or parent scoring modules are loaded. Test fixtures use an
in-memory ledger except the exact-byte ledger test's owned temporary directory.
"""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch, Mock

import numpy as np

import astra_tracking_score_core as core
import astra_tracking_raw_score as runner
import astra_tracking_raw_supervise as supervisor


class NativeScoreTests(unittest.TestCase):
    def test_native_float32_strict_deep_boundaries(self):
        zero = np.float32(0)
        tiny = np.nextafter(zero, np.float32(-np.inf))
        deep = np.float32(-.005)
        a = np.array([zero, -zero, tiny, deep,
                      np.nextafter(deep, np.float32(-np.inf)),
                      np.nextafter(deep, np.float32(np.inf))], dtype=np.float32)
        margins = np.full((len(a), 1, 4), np.float32(.1))
        margins[:, 0, 0] = a
        strict, damaging = core.native_flags(margins)
        np.testing.assert_array_equal(strict[:, 0, 0], [False, False, True, True, True, True])
        np.testing.assert_array_equal(damaging[:, 0, 0], [False, False, False, False, True, False])
        # Adjacent representable values, not an epsilon band, determine membership.
        # All decisions above precede any JSON/double conversion.
        self.assertEqual(core.score_windows(margins)[0]['deep']['steps'], 1)

    def test_first_and_last_post_frames_and_nonexclusive_classes(self):
        margins = np.ones((960, 2, 4), np.float32)
        margins[0, 0, [0, 1]] = -.01
        margins[959, 0, [0, 3]] = -.001
        metrics = core.score_windows(margins)
        a = metrics[0]
        self.assertEqual(a['strict']['steps'], 2)
        self.assertEqual(a['strict']['first_step'], 0)
        self.assertEqual(a['deep']['steps'], 1)
        self.assertEqual(a['strict']['class_steps'], dict(cross=2, self_F=1, self_U=0, table=1))
        self.assertFalse(metrics[1]['strict']['failed'])
        self.assertIsNone(metrics[1]['strict']['first_step'])

    def test_missing_class_positive_infinity_and_invalid_input(self):
        m = np.ones((2, 1, 4), np.float32)
        m[..., 2] = np.inf
        out = core.score_windows(m)[0]
        self.assertIsNone(out['minimum_nonexempt_margin_m']['self_U'])
        self.assertEqual(out['positive_infinite_margin_class_frames'][2], 2)
        for invalid in (np.nan, -np.inf):
            bad = m.copy(); bad[0, 0, 0] = invalid
            with self.assertRaisesRegex(ValueError, 'invalid native margins'):
                core.native_flags(bad)
        for bad in (m.astype(np.float64), m[0], m[:0]):
            with self.assertRaises(ValueError):
                core.native_flags(bad)

    def test_aggregation_does_not_count_invalid_window_safe(self):
        m = np.ones((2, 1, 4), np.float32); m[1, 0, 3] = -.006
        windows = [dict(status='VERIFIED', metrics=core.score_windows(m)[0]),
                   dict(status='UNAVAILABLE_OR_INVALID', metrics=None)]
        out = core.aggregate(windows)
        self.assertEqual((out['expected_windows'], out['verified_windows'], out['unavailable_or_invalid_windows']), (2, 1, 1))
        self.assertEqual(out['strict']['failed_windows'], 1)
        self.assertEqual(out['deep']['class_env_steps']['table'], 1)

    @staticmethod
    def window(fail):
        m = np.ones((2, 1, 4), np.float32)
        if fail:
            m[1, 0, 0] = -.02
        return dict(status='VERIFIED', metrics=core.score_windows(m)[0],
            q0_sha256='q', qd0_sha256='v', initial_target_sha256='u',
            tape_sha256='t', full_tape_sha256='tf', bank_sha256='b',
            first6_post_q_sha256='p6',first6_pre_qd_sha256='v6',
            first6_post_qd_sha256='vnext6',first6_applied_sha256='u6')

    def test_all_pair_partitions_and_unavailable_identity(self):
        cases = []
        for i, (r, c) in enumerate(((True, False), (False, True), (True, True), (False, False))):
            cases.append(dict(case_id=str(i), windows={core.MODES[0]:self.window(r), core.MODES[2]:self.window(c)}))
        cases.append(dict(case_id='missing', windows={core.MODES[0]:self.window(False),
                     core.MODES[2]:dict(status='UNAVAILABLE_OR_INVALID')}))
        out = core.paired_comparison(cases, core.MODES[2])
        self.assertEqual(out['verified_pairs'], 4)
        for kind in ('strict', 'deep'):
            self.assertEqual(out[kind]['counts'], dict(rescue=1, new_failure=1, both_fail=1, both_safe=1))
            self.assertEqual(out[kind]['rescue'], ['0'])
            self.assertEqual(out[kind]['new_failure'], ['1'])
            self.assertEqual(out[kind]['class_counts']['self_F']['both_safe'], 4)
        self.assertEqual(out['unavailable_or_invalid'][0]['case_id'], 'missing')

    def test_each_pairing_binding_is_required(self):
        for key in ('q0_sha256', 'qd0_sha256', 'initial_target_sha256', 'tape_sha256', 'full_tape_sha256', 'bank_sha256'):
            a, b = self.window(True), self.window(False)
            b[key] += 'different'
            result = core.paired_comparison([dict(case_id='x', windows={core.MODES[0]:a, core.MODES[2]:b})], core.MODES[2])
            self.assertEqual(result['verified_pairs'], 0, key)
            self.assertEqual(result['strict']['rescue'], [])
            self.assertEqual(b['status'], 'VERIFIED')  # individual score stays usable

    def test_fingerprint_preserves_bits_shape_and_dtype(self):
        a = np.zeros(4, np.float32)
        b = a.copy(); b[0] = np.nextafter(np.float32(0), np.float32(1))
        self.assertNotEqual(core.fingerprint(a), core.fingerprint(b))
        self.assertNotEqual(core.fingerprint(a), core.fingerprint(a.reshape(2, 2)))
        self.assertNotEqual(core.fingerprint(a), core.fingerprint(a.astype(np.float64)))
        self.assertEqual(core.fingerprint(a), core.fingerprint(a.copy()))


class GeometryTests(unittest.TestCase):
    def test_native_exemption_aware_class_minima(self):
        identity = dict(class_id=[0, 1, 1, 2, 2, 0], pair_id=list(range(6)),
            pair_sphere_idx=[[0,2],[0,1],[2,3],[1,-1],[3,-1],[1,3]], sphere_arm_id=[0,1,2,3])
        classes = core.evaluation_classes(identity)
        np.testing.assert_array_equal(classes, [0,1,2,3,3,0])
        d = np.array([[-.1,.3,.4,-.5,.2,.1]], np.float32)
        ex = np.array([[True,False,False,True,False,False]])
        np.testing.assert_array_equal(core.reduce_geometry(d, ex, classes), np.array([[.1,.3,.4,.2]], np.float32))
        ex[0, [3,4]] = True
        self.assertTrue(np.isposinf(core.reduce_geometry(d, ex, classes)[0,3]))

    def test_packed_mask_and_padding(self):
        mask = np.array([[True,False,True,False,False,True]])
        packed = np.packbits(mask, axis=-1)
        np.testing.assert_array_equal(core.unpack_exempt(packed, 6), mask)
        packed[0,0] |= 1
        with self.assertRaisesRegex(ValueError, 'padding'):
            core.unpack_exempt(packed, 6)

    def test_all_row_capacity_ids_and_duplicate_rejection(self):
        with patch.object(runner, 'ROWS', 9021):
            ids = np.arange(9021, dtype=np.int32).reshape(1,1,-1)
            self.assertTrue(runner.ids_mask(ids, ids.shape).all())
            ids[0,0,-1] = 0
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                runner.ids_mask(ids, ids.shape)


class BackendArithmeticTests(unittest.TestCase):
    def test_cuda_reciprocal_path_has_distinguishable_bits_without_epsilon(self):
        margins=np.linspace(.010,.050,10001,dtype=np.float32)
        gpu=runner.cuda_scalar_gap(margins)
        cpu=np.float32(.010)+np.float32(.040)*np.clip((margins-np.float32(.010))/np.float32(.040),np.float32(0),np.float32(1))
        self.assertGreater(np.count_nonzero(gpu.view('u4')!=cpu.view('u4')),0)
        expected=[]
        for m in margins:
            delta=np.float32(float(m)-float(np.float32(.010)))
            ratio=np.float32(float(delta)*25.0)
            clamped=max(0.0,min(1.0,float(ratio)))
            scaled=np.float32(clamped*float(np.float32(.040)))
            expected.append(np.float32(float(scaled)+float(np.float32(.010))))
        np.testing.assert_array_equal(gpu.view('u4'),np.array(expected,np.float32).view('u4'))


def fifo_fixture():
    steps, envs, joints = 12, 3, 2
    q0 = np.zeros((envs, joints), np.float32)
    u0 = np.full_like(q0, .125)
    limits = np.broadcast_to(np.array([-.5,.5], np.float32), (envs,joints,2)).copy()
    increments = np.full((steps, envs, joints), np.float32(.125))
    issued, pending, applied = [], [], []
    queue = [u0.copy() for _ in range(6)]
    target = u0.copy()
    for step in range(steps):
        pending.append(np.stack(queue))
        target = np.clip(target+increments[step], limits[...,0], limits[...,1])
        issued.append(target.copy())
        applied.append(queue.pop(0)); queue.append(target.copy())
    issued = np.stack(issued)
    q = np.full_like(issued, .0625)  # not the integrated/applied target
    prior_target = np.concatenate((u0[None], issued[:-1]))
    prior_q = np.concatenate((q0[None], q[:-1]))
    return dict(q_initial=q0, q=q, controller_target=issued,
        actuator_target=np.stack(applied), exec=increments, joint_soft_limits=limits,
        pre_pending_actuator_targets=np.stack(pending), pre_pending_project_history=np.stack(pending),
        pre_target_debt=prior_target-prior_q, effective_target_delta=issued-prior_target,
        pre_qd_compact=np.zeros_like(q))


class FifoAndFirstFailureTests(unittest.TestCase):
    def setUp(self):
        self.patches = [patch.object(runner,k,v) for k,v in dict(STEPS=12, ENVS=3, JOINTS=2).items()]
        for p in self.patches: p.start()
        self.addCleanup(lambda: [p.stop() for p in reversed(self.patches)])

    def test_fifo_first_issued_arrival_seven_and_real_integration(self):
        d = fifo_fixture()
        np.testing.assert_array_equal(runner.verify_fifo(d), np.full((3,2), .125, np.float32))
        np.testing.assert_array_equal(d['actuator_target'][:6], np.full((6,3,2), .125, np.float32))
        np.testing.assert_array_equal(d['actuator_target'][6], d['controller_target'][0])
        self.assertFalse(np.array_equal(d['exec'], d['effective_target_delta']))  # limit saturation
        self.assertFalse(np.array_equal(d['q'], d['actuator_target']))

    def test_wrong_arrival_and_integrated_vs_projected_delta_detected(self):
        d = fifo_fixture()
        d['actuator_target'][5] = d['controller_target'][0]
        with self.assertRaisesRegex(ValueError, 'applied FIFO'):
            runner.verify_fifo(d)
        d = fifo_fixture(); d['effective_target_delta'][:] = d['exec']
        with self.assertRaisesRegex(ValueError, 'effective increment'):
            runner.verify_fifo(d)

    @staticmethod
    def snapshot(d, step, envs):
        previous = d['q_initial'] if step == 0 else d['q'][step-1]
        target = d['pre_pending_actuator_targets'][0,-1] if step == 0 else d['controller_target'][step-1]
        return dict(step=np.array(step), env_ids=np.array(envs), pre_q=previous[envs],
            pre_issued_target=target[envs], pre_pending_actuator_targets=d['pre_pending_actuator_targets'][step,:,envs],
            actual_project_return=d['exec'][step,envs], returned_cmd=d['effective_target_delta'][step,envs],
            snapshot_qd=d['pre_qd_compact'][step,envs], post_q=d['q'][step,envs],
            post_qd=np.zeros_like(d['q'][step,envs]))

    def test_first_failure_selection_and_snapshot_binding(self):
        d = fifo_fixture()
        margins = np.ones((12,3,4), np.float32)
        margins[8,0,0] = -.1; margins[2,1,2] = -.1; margins[2,2,3] = -.1
        receipt = dict(envs_with_failure=3, receipts=[
            dict(step=8,env_ids=[0],path='step8.npz',sha256='unused'),
            dict(step=2,env_ids=[2,1],path='step2.npz',sha256='unused')])
        data = {'step8.npz':self.snapshot(d,8,[0]), 'step2.npz':self.snapshot(d,2,[2,1])}
        class Memory:
            def json(self, path): return receipt
            def npz(self, path, **kwargs): return data[path.name]
        result = runner.selected_failures(Memory(), runner.H, core.score_windows(margins), d)
        self.assertEqual([(r['step'],r['env']) for r in result], [(2,1),(2,2)])
        data['step2.npz']['post_q'][0,0] += np.float32(.125)
        with self.assertRaisesRegex(ValueError, 'post q binding'):
            runner.selected_failures(Memory(), runner.H, core.score_windows(margins), d)

    def test_missing_failure_receipt_is_invalid(self):
        class Empty:
            def json(self, path): return dict(envs_with_failure=0,receipts=[])
        m = np.ones((12,3,4), np.float32); m[-1,2,0] = -.1
        with self.assertRaisesRegex(ValueError, 'independent native score'):
            runner.selected_failures(Empty(), runner.H, core.score_windows(m), fifo_fixture())


def geometry_fixture(mode):
    """Native synthetic full rows: exempt negative table, one newly admitted target row.

    A separate reserve row has a negative margin but is NOT an admission rule.
    Adaptive gap covers low/high plateaus and interior points independently.
    """
    n, rows = 2, 6
    d = np.ones((960,n,rows), np.float32)
    d[:,:,0] = np.arange(960, dtype=np.float32)[:,None]/np.float32(2048)
    d[:,:,3] = -.2
    ex = np.zeros_like(d, bool); ex[:,:,3] = True
    target = d.copy();target[:,:,5] = -.125
    reserve = d.copy()
    reserve[:,:,4] = np.array([-.125,.015625,.03125,.0625],np.float32)[np.arange(960)%4,None]
    margin = np.where(ex,np.float32(np.inf),reserve).min(-1)
    # Scalar per-element oracle, kept separate from runner's vectorized arithmetic.
    gap = np.empty_like(margin)
    for t in range(960):
        for e in range(n):
            if mode==core.MODES[0]:gap[t,e]=np.float32(.05)
            elif mode==core.MODES[1]:gap[t,e]=np.float32(.01)
            else:
                x=np.float32(np.float32(margin[t,e]-np.float32(.01))*np.float32(25))
                gap[t,e]=np.float32(np.float32(.01)+np.float32(.04)*max(np.float32(0),min(np.float32(1),x)))
    base = np.zeros_like(d,bool); base[:,:,[0,3]] = True
    chosen = base | (target <= np.float32(.010))
    def ids(mask):
        out = np.full(mask.shape,-1,np.int32)
        for t in range(960):
            for e in range(n):
                indices=np.flatnonzero(mask[t,e]);out[t,e,:len(indices)]=indices
        return out
    all_arrays = dict(measured_d=d, exempt=np.packbits(ex,axis=-1), dmin=np.zeros_like(d),
        target_forecast=target,forecast=target.copy(),reserve_forecast=reserve,
        reserve_margin=margin,reserve_gap=gap,selected_ids=ids(chosen),baseline_ids=ids(base))
    identity=dict(rows=6,class_id=[0,1,1,2,2,0],pair_id=list(range(6)),
        pair_sphere_idx=[[0,2],[0,1],[2,3],[1,-1],[3,-1],[1,3]],sphere_arm_id=[0,1,2,3])
    classes=np.array([0,1,2,3,3,0])
    pre = core.reduce_geometry(d,ex,classes)
    margins=np.concatenate((pre[1:],np.full((1,n,4),-.03125,np.float32)))
    dense=dict(official_margins=margins,critical_selected_count=chosen.sum(-1))
    entries=[dict(start=s,stop=s+32,path=f'{s}.npz',sha256='unused') for s in range(0,960,32)]
    class Memory:
        def json(self,path):
            return identity if path.name=='full_row_identity.json' else dict(steps=960,rows=6,envs=n,chunks=entries)
        def npz(self,path,keys=None,expected=None):
            s=int(path.stem);return {k:all_arrays[k][s:s+32] for k in keys}
    return Memory(),dense,all_arrays


class RawChunkContractTests(unittest.TestCase):
    def setUp(self):
        self.patches=[patch.object(runner,'ENVS',2),patch.object(runner,'ROWS',6)]
        for p in self.patches:p.start()
        self.addCleanup(lambda:[p.stop() for p in reversed(self.patches)])

    def test_pre_post_index_and_all_three_target_only_modes(self):
        for mode in core.MODES:
            ledger,dense,_=geometry_fixture(mode)
            result=runner.chunk_audit(ledger,runner.H,dense,[],mode)
            self.assertEqual(result['post_frames_crosschecked'],959)
            self.assertFalse(result['final_post_full_row_archive_available'])
            self.assertFalse(result['full_J_archive_recomputed'])
            self.assertEqual(result['non_target_added_row_instances_at_own_state'],0)
            self.assertTrue(all(w['deep']['failed'] for w in core.score_windows(dense['official_margins'])))

    def test_shifted_geometry_binding_fails(self):
        ledger,dense,_=geometry_fixture(core.MODES[0])
        dense['official_margins']=np.roll(dense['official_margins'],1,axis=0)
        with self.assertRaisesRegex(ValueError,'previous native post'):
            runner.chunk_audit(ledger,runner.H,dense,[],core.MODES[0])

    def test_dropped_target_row_and_reserve_as_admission_fail(self):
        for mutation in ['drop','add']:
            ledger,dense,raw=geometry_fixture(core.MODES[2])
            if mutation=='drop':raw['selected_ids'][0,0,2]=-1
            else:raw['selected_ids'][0,0,3]=4
            with self.assertRaisesRegex(ValueError,'dropped/swapped'):
                runner.chunk_audit(ledger,runner.H,dense,[],core.MODES[2])

    def test_reserve_nonfinite_minimum_exemption_and_gap_mutations_fail(self):
        for mutation in ['nan','exempt','minimum','gap']:
            ledger,dense,raw=geometry_fixture(core.MODES[0])
            if mutation=='nan':raw['reserve_forecast'][-1,-1,-1]=np.nan
            elif mutation=='exempt':raw['reserve_margin'][0,0]=np.float32(-.2)
            elif mutation=='minimum':raw['reserve_margin'][0,0]=np.float32(.01)
            else:raw['reserve_gap'][0,0]=np.nextafter(np.float32(.05),np.float32(1))
            with self.assertRaises(ValueError):
                runner.chunk_audit(ledger,runner.H,dense,[],core.MODES[0])


class WholeCellContractTests(unittest.TestCase):
    def test_full_score_entry_binds_962_recipe_to_960_executed_frames(self):
        mode = core.MODES[0]
        chunks, dense, _ = geometry_fixture(mode)
        dense['official_margins'][-1] = dense['official_margins'][-2]
        shape = (960,2,26)
        limits = np.broadcast_to(np.array([-.5,.5],np.float32),(2,26,2)).copy()
        q0 = np.zeros((2,26),np.float32)
        full_tape = np.zeros((962,2,26),np.float32)
        full_tape[60:960] = .03125
        full_tape[960:] = .25  # deliberately unexecuted, still part of full identity
        cmd = full_tape[:960].copy()
        target = q0.copy(); queue = [q0.copy() for _ in range(6)]
        pending, issued, applied = [],[],[]
        for c in cmd:
            pending.append(np.stack(queue))
            target=np.clip(target+c,limits[...,0],limits[...,1])
            issued.append(target.copy());applied.append(queue.pop(0));queue.append(target.copy())
        issued=np.stack(issued)
        prior=np.concatenate((q0[None],issued[:-1]))
        expected=dict(seed=321,flow='risk_burst',method='system0',amp=.05,actuator_delay_steps=6)
        proto=dict(status='complete',completed_cells=1,design=[expected],steps=960,dt=.016666,
                   source_sha256={},checkpoint_sha256='actor',args=dict(actuator_delay_steps=[6],env_yaml='frozen.yaml'))
        dense.update(q=np.zeros(shape,np.float32),q_initial=q0,cmd=cmd,external_unscaled_cmd=cmd,
            exec=cmd,controller_target=issued,actuator_target=np.stack(applied),pre_qd_compact=np.zeros(shape,np.float32),
            pre_target_debt=prior,pre_pending_actuator_targets=np.stack(pending),pre_pending_project_history=np.stack(pending),
            effective_target_delta=issued-prior,joint_soft_limits=limits,initial_violation=np.zeros(2,bool),
            official_deep=dense['official_margins']<np.float32(-.005),
            meta_json=np.array(json.dumps(dict(**expected,dt=.016666,cell_id=1,env_yaml='frozen.yaml',checkpoint_sha256='actor'))))
        recipe=dict(q_initial=q0,sampled_initial=q0,tape=full_tape,joint_soft_limits=limits,initial_violation=np.zeros(2,bool))
        bank=dict(accepted_q=q0,risk_pair_index=np.array([0,-1]),joint_soft_limits=limits)
        bank_path=runner.H/'astra_synthetic_bank.npz'
        root=runner.H/'astra_synthetic_cell'
        job=dict(id=root.name,argv=['synthetic_python','synthetic_guard'],expected=expected,expected_args={},env=dict(SAFEDUO_JOINT_MODE=mode,SAFEDUO_INITIAL_BANK_NPZ=str(bank_path)))
        plan=dict(output_root=str(runner.H),source_sha256={},checkpoint_sha256='actor',research_source_sha256={str(bank_path):'bankhash'})
        arrays={'cell_001.npz':dense,'input_recipe.npz':recipe,'astra_synthetic_bank.npz':bank,'bank_assignment.npz':bank}
        metadata=dict(mode=mode,capacity=6,strict_fifo_steps=6,reference_gap_rad=.05,original_actor_rows=32,
                      joint_repair=False,queue_preemption=False,sources={})
        class Memory:
            hashes={}
            def json(self,path,expected=None):
                docs={'protocol.json':proto,'guard_metadata.json':metadata,'first_failure_receipts.json':dict(envs_with_failure=0,receipts=[])}
                return docs[path.name] if path.name in docs else chunks.json(path)
            def bind(self,path,expected=None):return expected
            def npz(self,path,keys=None,expected=None):
                self.hashes[str(path)]='synthetic_payload'
                if path.name in arrays:return {k:arrays[path.name][k] for k in keys}
                return chunks.npz(path,keys,expected)
        with patch.object(runner,'ENVS',2),patch.object(runner,'ROWS',6):
            result=runner.score_cell(Memory(),plan,job,dict(id=root.name,argv=job['argv']+['--out',str(root)],status='complete',exit_code=0,protocol_sha256='protocol'))
            self.assertEqual(result['counts']['verified_windows'],2)
            self.assertEqual(result['counts']['strict']['failed_windows'],0)
            self.assertEqual(result['windows'][0]['tape_sha256'],core.fingerprint(full_tape[:960,0]))
            self.assertEqual(result['windows'][0]['full_tape_sha256'],core.fingerprint(full_tape[:,0]))
            dense['cmd']=np.roll(cmd,1,axis=0)
            with self.assertRaisesRegex(ValueError,'raw command differs from tape'):
                runner.score_cell(Memory(),plan,job,dict(id=root.name,argv=job['argv']+['--out',str(root)],status='complete',exit_code=0,protocol_sha256='protocol'))

    def test_complete_inventory_and_failed_window_never_replaced_by_safe(self):
        design=dict(rows=[dict(initial_seed=b+10,command_seed=b+20) for b in range(3)])
        cells=[]
        for b in range(3):
            for i,m in enumerate(core.MODES):
                windows=[NativeScoreTests.window(e%2==0 if i==0 else False) for e in range(64)]
                cells.append(dict(registered_index=b*3+i,block=b,mode=m,id=f'{m}_{b}',windows=windows))
        ledger=runner.Ledger()
        report=runner.assemble_report(design,cells,ledger,[])
        self.assertEqual(report['status'],'PASS_COMPLETE_INDEPENDENT_TRACKING_NUMERIC')
        self.assertEqual(report['paired']['joint_reference__vs__delay_reserve']['strict']['counts']['rescue'],96)
        self.assertEqual(report['paired']['tight_reference__vs__delay_reserve']['strict']['counts']['both_safe'],192)
        cells[5]['windows']=[dict(status='UNAVAILABLE_OR_INVALID',metrics=None) for _ in range(64)]
        report=runner.assemble_report(design,cells,ledger,[])
        self.assertEqual(report['status'],'INCOMPLETE_OR_INVALID_TRACKING_NUMERIC')
        self.assertEqual(len(report['cases']),192)
        self.assertEqual(sum(len(c['windows']) for c in report['cases']),576)
        self.assertEqual(report['counts'][core.MODES[2]]['unavailable_or_invalid_windows'],64)
        self.assertEqual(report['paired']['joint_reference__vs__delay_reserve']['verified_pairs'],128)
        self.assertEqual(report['paired']['joint_reference__vs__tight_reference']['verified_pairs'],192)
        self.assertTrue(report['raw_hash_before_after_pass'])
        report=runner.assemble_report(design,cells,ledger,['changed'])
        self.assertFalse(report['raw_hash_before_after_pass'])
        self.assertFalse(report['status'].startswith('PASS'))


class InputClosureTests(unittest.TestCase):
    def test_hash_before_after_and_exact_npz_payload(self):
        with tempfile.TemporaryDirectory(prefix='astra_test_',dir=runner.H) as tmp:
            path=Path(tmp)/'fixture.npz'
            np.savez_compressed(path,x=np.arange(3,dtype=np.float32))
            ledger=runner.Ledger()
            expected=runner.sha(path)
            np.testing.assert_array_equal(ledger.npz(path,['x'],expected)['x'],[0,1,2])
            self.assertEqual(ledger.recheck(),[])
            path.write_bytes(b'changed')
            self.assertEqual(ledger.recheck(),[str(path)])
            with self.assertRaisesRegex(ValueError,'changed on reread'):
                ledger.bytes(path)

    def test_running_campaign_blocks_all_outcome_reads(self):
        class ForbiddenLedger:
            def json(self,*args,**kwargs): raise AssertionError('immutable/outcome read before closure')
        for status in ['running','starting','pending']:
            with patch.object(Path,'exists',return_value=True),patch.object(Path,'read_text',return_value=json.dumps(dict(status=status))):
                self.assertIsNone(runner.canonical_closed(ForbiddenLedger(),dict(output_root='/synthetic')))

    def test_closed_canonical_plan_and_child_identity_required(self):
        plan=dict(output_root='/synthetic',jobs=[dict(id='one')])
        doc=dict(status='complete',plan=plan,jobs=[dict(id='one',status='complete',exit_code=0)])
        class Memory:
            def json(self,*args):return doc
        with patch.object(Path,'exists',return_value=True),patch.object(Path,'read_text',side_effect=lambda:json.dumps(doc)):
            self.assertEqual(set(runner.canonical_closed(Memory(),plan)),{'one'})
            doc['jobs'].append(dict(id='one'))
            with self.assertRaisesRegex(ValueError,'duplicate'):
                runner.canonical_closed(Memory(),plan)
            doc['jobs']=[dict(id='foreign')]
            with self.assertRaisesRegex(ValueError,'unregistered'):
                runner.canonical_closed(Memory(),plan)

    def test_cell_exception_is_64_invalid_not_safe_and_checkpoint_persists(self):
        out={}
        with patch.object(runner,'score_cell',side_effect=ValueError('injected corrupted raw')),patch.object(runner,'sha',return_value='frozen'),\
             patch.object(runner,'write_owned',side_effect=lambda n,v:out.update({n:v})):
            checkpoint=runner.cell_worker(dict(output_root='/synthetic'),dict(id='case',env={'SAFEDUO_JOINT_MODE':core.MODES[0]}),{},0,0)
        self.assertFalse(checkpoint['cell_raw_before_after_pass'])
        self.assertEqual(len(checkpoint['result']['windows']),64)
        self.assertTrue(all(w['status']=='UNAVAILABLE_OR_INVALID' and w['metrics'] is None for w in checkpoint['result']['windows']))
        self.assertIn('astra_tracking_raw_cell_00.json',out)

    def test_conflicting_independent_worker_input_hashes_abort(self):
        ledger=runner.Ledger()
        runner.merge_checkpoint(ledger,dict(input_sha256={'/synthetic/raw':'a'},result={}))
        with self.assertRaisesRegex(ValueError,'changed on reread'):
            runner.merge_checkpoint(ledger,dict(input_sha256={'/synthetic/raw':'b'},result={}))

    def test_write_and_receipt_paths_cannot_escape(self):
        for bad in ('parent.json','../ASTRA_FINAL_BAD.json','sub/astra_bad'):
            with self.assertRaises(ValueError):runner.write_owned(bad,{})
        with self.assertRaises(ValueError):runner.relative_file(runner.H,'../wrong.npz')


class SchedulingTests(unittest.TestCase):
    def test_two_workers_independent_completion_registered_order_and_auto_close(self):
        design=dict(rows=[dict(initial_seed=b+10,command_seed=b+20) for b in range(3)])
        plans=[dict(jobs=[dict(id=f'{m}_{b}',env={'SAFEDUO_JOINT_MODE':m}) for m in core.MODES],
            cwd='/synthetic',source_sha256={},research_source_sha256={},checkpoint_path='/synthetic/actor',checkpoint_sha256='actor') for b in range(3)]
        output={};lock=threading.Lock();state=dict(active=0,maximum=0,finished=[])
        class Memory:
            hashes={}
            def bind(self,*args):return 'frozen'
            def _remember(self,*args):pass
            def recheck(self):return []
        def worker(plan,job,record,block,index):
            self.assertEqual(record['id'],job['id'])
            with lock:
                state['active']+=1;state['maximum']=max(state['maximum'],state['active'])
            threading.Event().wait(.01 if index%2 else .03)
            with lock:
                state['active']-=1;state['finished'].append(index)
            return dict(source_sha256={},input_sha256={},result=dict(registered_index=index,block=block,mode=job['env']['SAFEDUO_JOINT_MODE'],id=job['id'],status='VERIFIED',windows=[NativeScoreTests.window(False) for _ in range(64)]))
        with tempfile.TemporaryDirectory(prefix='astra_test_scheduler_',dir=runner.H) as tmp,\
             patch.object(runner,'H',Path(tmp)),patch.object(runner,'Ledger',Memory),\
             patch.object(runner,'bind_software',return_value=dict(minimum_mem_available_bytes=25*1024**3)),\
             patch.object(runner,'registered_inputs',return_value=(design,plans)),\
             patch.object(runner,'canonical_closed',side_effect=lambda ledger,p:{j['id']:dict(id=j['id']) for j in p['jobs']}),\
             patch.object(runner,'cell_worker',side_effect=worker),\
             patch.object(runner,'available_memory',return_value=30*1024**3),\
             patch.object(runner,'sha',return_value='frozen'),\
             patch.object(runner,'write_owned',side_effect=lambda n,v:output.update({n:v})),\
             contextlib.redirect_stdout(io.StringIO()):
            report=runner.run(watch=True)
        self.assertEqual(state['maximum'],2)
        self.assertEqual(state['active'],0)
        self.assertEqual(sorted(state['finished']),list(range(9)))
        self.assertNotEqual(state['finished'],list(range(9)))
        self.assertEqual([c['registered_index'] for c in report['conditions']],list(range(9)))
        self.assertEqual(report['status'],'PASS_COMPLETE_INDEPENDENT_TRACKING_NUMERIC')
        self.assertEqual(output['astra_tracking_raw_execution.json']['windows'],576)
        self.assertIn('ASTRA_TRACKING_FINAL_SCORE.json',output)


class SupervisorClosureTests(unittest.TestCase):
    def test_actual_failure_signal_and_missing_score_not_forged_as_complete(self):
        for rc,present in [(0,True),(0,False),(1,False),(-15,False)]:
            with self.subTest(rc=rc,present=present),tempfile.TemporaryDirectory(prefix='astra_test_exit_',dir=runner.H) as tmp:
                root=Path(tmp)
                plan=dict(workers=2,source_sha256={Path(supervisor.__file__).name:'frozen'},
                    scorer_argv=[supervisor.sys.executable,'-u',str(root/'astra_tracking_raw_score.py'),'--watch'])
                (root/'astra_tracking_raw_plan.json').write_text(json.dumps(plan))
                if present:(root/'ASTRA_TRACKING_FINAL_SCORE.json').write_text(json.dumps(dict(status='PASS_SYNTHETIC_ONLY')))
                child=Mock(pid=42);child.wait.return_value=rc
                with patch.object(supervisor,'H',root),patch.object(supervisor,'sha',return_value='frozen'),\
                     patch.object(supervisor.subprocess,'Popen',return_value=child) as spawn:
                    status=supervisor.main()
                receipt=json.loads((root/'astra_tracking_raw_process_exit.json').read_text())
                self.assertEqual(receipt['actual_child_exit_code'],rc)
                self.assertEqual(receipt['final_score_exists'],present)
                self.assertEqual(status,rc if rc>=0 else 128-rc)
                self.assertEqual(spawn.call_count,1)
                self.assertEqual(receipt['automatic_retries'],0)
                if not present:self.assertIsNone(receipt['final_score_status'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
