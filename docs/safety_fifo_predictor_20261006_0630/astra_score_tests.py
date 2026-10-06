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
import unittest
from unittest.mock import patch

import numpy as np

import astra_score_core as core
import astra_final_score as runner


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
            tape_sha256='t', full_tape_sha256='tf', bank_sha256='b')

    def test_all_pair_partitions_and_unavailable_identity(self):
        cases = []
        for i, (r, c) in enumerate(((True, False), (False, True), (True, True), (False, False))):
            cases.append(dict(case_id=str(i), windows={core.MODES[0]:self.window(r), core.MODES[3]:self.window(c)}))
        cases.append(dict(case_id='missing', windows={core.MODES[0]:self.window(False),
                     core.MODES[3]:dict(status='UNAVAILABLE_OR_INVALID')}))
        out = core.paired_comparison(cases, core.MODES[3])
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
            result = core.paired_comparison([dict(case_id='x', windows={core.MODES[0]:a, core.MODES[3]:b})], core.MODES[3])
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
    """Distance row5 is motion-only; table row3 is exempt, class minima change per frame."""
    n, rows = 2, 6
    d = np.ones((960,n,rows), np.float32)
    d[:,:,0] = np.arange(960, dtype=np.float32)[:,None]/np.float32(2048)
    d[:,:,3] = -.2
    ex = np.zeros_like(d, bool); ex[:,:,3] = True
    # cross row0 measured critical only at t<=20; manually baseline includes it
    # throughout, which is legal for the original selected rows.
    target = d.copy(); cv = d.copy(); pd = d.copy()
    cv[:,:,5] = -.125; pd[:,:,4] = -.25
    forecast = target.copy()
    if mode in (core.MODES[1],core.MODES[3]): forecast = np.minimum(forecast,cv)
    if mode in (core.MODES[2],core.MODES[3]): forecast = np.minimum(forecast,pd)
    base = np.zeros_like(d,bool); base[:,:,[0,3]] = True
    chosen = base | (forecast <= np.float32(.010))
    def ids(mask):
        out = np.full(mask.shape,-1,np.int32)
        for t in range(960):
            for e in range(n):
                indices=np.flatnonzero(mask[t,e]);out[t,e,:len(indices)]=indices
        return out
    all_arrays = dict(measured_d=d, exempt=np.packbits(ex,axis=-1), dmin=np.zeros_like(d),
        target_forecast=target,cv_forecast=cv,pd_forecast=pd,forecast=forecast,
        selected_ids=ids(chosen),baseline_ids=ids(base))
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

    def test_pre_post_index_and_all_four_row_union_modes(self):
        for mode,added in zip(core.MODES,[0,1920,1920,3840]):
            ledger,dense,_=geometry_fixture(mode)
            result=runner.chunk_audit(ledger,runner.H,dense,[],mode)
            self.assertEqual(result['post_frames_crosschecked'],959)
            self.assertFalse(result['final_post_full_row_archive_available'])
            self.assertEqual(result['motion_only_added_row_instances_at_own_state'],added)
            # The final post frame is scored even though no matching next pre was saved.
            self.assertTrue(all(w['deep']['failed'] for w in core.score_windows(dense['official_margins'])))

    def test_shifted_geometry_binding_fails(self):
        ledger,dense,_=geometry_fixture(core.MODES[0])
        dense['official_margins']=np.roll(dense['official_margins'],1,axis=0)
        with self.assertRaisesRegex(ValueError,'previous native post'):
            runner.chunk_audit(ledger,runner.H,dense,[],core.MODES[0])

    def test_dropped_last_motion_row_and_hidden_nonfinite_fail(self):
        ledger,dense,raw=geometry_fixture(core.MODES[1])
        raw['selected_ids'][0,0,2]=-1
        with self.assertRaisesRegex(ValueError,'dropped/swapped'):
            runner.chunk_audit(ledger,runner.H,dense,[],core.MODES[1])
        ledger,dense,raw=geometry_fixture(core.MODES[0])
        raw['pd_forecast'][-1,-1,-1]=np.nan  # inactive mode must not hide invalid factor
        with self.assertRaisesRegex(ValueError,'nonfinite'):
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
        job=dict(id=root.name,expected=expected,expected_args={},env=dict(SAFEDUO_JOINT_MODE=mode,SAFEDUO_INITIAL_BANK_NPZ=str(bank_path)))
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
            result=runner.score_cell(Memory(),plan,job,dict(status='complete',exit_code=0,protocol_sha256='protocol'))
            self.assertEqual(result['counts']['verified_windows'],2)
            self.assertEqual(result['counts']['strict']['failed_windows'],0)
            self.assertEqual(result['windows'][0]['tape_sha256'],core.fingerprint(full_tape[:960,0]))
            self.assertEqual(result['windows'][0]['full_tape_sha256'],core.fingerprint(full_tape[:,0]))
            dense['cmd']=np.roll(cmd,1,axis=0)
            with self.assertRaisesRegex(ValueError,'raw command differs from tape'):
                runner.score_cell(Memory(),plan,job,dict(status='complete',exit_code=0,protocol_sha256='protocol'))

    def test_complete_inventory_and_failed_window_never_replaced_by_safe(self):
        design=dict(rows=[dict(initial_seed=b+10,command_seed=b+20) for b in range(3)])
        plans=[]
        for block in range(3):
            jobs=[dict(id=f'{m}_{block}',env={'SAFEDUO_JOINT_MODE':m}) for m in core.MODES]
            plans.append(dict(output_root=f'/synthetic/{block}',cwd='/synthetic',source_sha256={},research_source_sha256={},
                              checkpoint_path='/synthetic/actor',checkpoint_sha256='actor',jobs=jobs))
        class Memory:
            hashes={}
            def bind(self,*args):pass
            def json(self,path):
                if path.name!='campaign.json':return {}
                plan=plans[int(path.parent.name)]
                return dict(status='complete',plan=plan,jobs=[dict(id=j['id']) for j in plan['jobs']])
            def recheck(self):return []
        def cell(ledger,plan,job,record):
            if job['id']==f'{core.MODES[2]}_1':raise ValueError('synthetic unavailable raw')
            windows=[NativeScoreTests.window(e%2==0 if job['env']['SAFEDUO_JOINT_MODE']==core.MODES[0] else False) for e in range(64)]
            return dict(id=job['id'],mode=job['env']['SAFEDUO_JOINT_MODE'],status='VERIFIED',windows=windows)
        output={}
        with patch.object(runner,'readiness',return_value=dict(all_terminal=True)),\
             patch.object(runner,'score_cell',side_effect=cell),\
             patch.object(runner,'write_owned',side_effect=lambda name,value:output.update({name:value})),\
             contextlib.redirect_stdout(io.StringIO()):
            runner.run_score(design,plans,Memory())
        report=output['ASTRA_FINAL_SCORE.json']
        self.assertEqual(report['status'],'INCOMPLETE_OR_INVALID_NUMERIC_EVIDENCE')
        self.assertEqual(len(report['cases']),192)
        self.assertEqual(sum(len(c['windows']) for c in report['cases']),768)
        self.assertEqual(report['counts'][core.MODES[2]]['unavailable_or_invalid_windows'],64)
        self.assertEqual(report['comparisons'][core.MODES[2]]['verified_pairs'],128)
        self.assertEqual(report['comparisons'][core.MODES[3]]['strict']['counts']['rescue'],96)
        self.assertEqual(report['verified_outcomes'],[])


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
            def bind(self,*args,**kwargs): raise AssertionError('outcome/source read before closure')
            def json(self,*args,**kwargs): raise AssertionError('outcome read before closure')
        with patch.object(runner,'readiness',return_value=dict(all_terminal=False,status='WAITING')):
            self.assertEqual(runner.run_score({},[],ForbiddenLedger())['status'],'WAITING')

    def test_write_and_receipt_paths_cannot_escape(self):
        for bad in ('parent.json','../ASTRA_FINAL_BAD.json','sub/astra_bad'):
            with self.assertRaises(ValueError):runner.write_owned(bad,{})
        with self.assertRaises(ValueError):runner.relative_file(runner.H,'../wrong.npz')


if __name__ == '__main__':
    unittest.main(verbosity=2)
