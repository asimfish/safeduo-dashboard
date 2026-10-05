"""Rescore all registered cells, dense forecast receipts, paired regressions and exposure."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'safety_risk_strata_20261004'))
from analyze_risk import inspect, load
sys.path.insert(0, str(HERE.parent / 'safety_random_space_20261004'))
from coverage_metrics import measured_coverage, SLICES, PAIR_NAMES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forecast_audit(path, data):
    receipt = json.loads((path / 'forecast_receipts.json').read_text())
    mechanism = load(path / 'mechanism.npz')
    identity = json.loads((path / 'full_row_identity.json').read_text())
    assert receipt['steps'] == 960 and receipt['rows'] == 9021 and len(receipt['chunks']) == 30
    assert (mechanism['full_rows_checked'] == 9021).all() and (mechanism['guard_calls_before_step'] >= 1).all()
    cls = np.array(identity['class_id'])
    mode=json.loads((path/'guard_metadata.json').read_text())['mode']
    capacity=1024 if mode in ['admission_guard','joint_guard'] else 512
    assert len(cls)==9021 and len(identity['pair_id'])==9021
    pair=np.array(identity['pair_sphere_idx']);arm=np.array(identity['sphere_arm_id'])
    assert pair.shape==(9021,2) and pair.dtype.kind in 'iu'
    assert ((pair[:,0]>=0)&(pair[:,0]<len(arm))).all()
    evaluation_cls=cls.copy();evaluation_cls[(cls==1)&(arm[pair[:,0]]>=2)]=2;evaluation_cls[cls==2]=3
    for key,value in mechanism.items():assert value.shape[:2]==(960,64) and np.isfinite(value).all(),key
    end = 0
    shadow_nonexempt_env_steps = 0
    full_negative_pre_env_steps = 0
    below_braking_band_pre_env_steps = 0
    selected = 0
    for chunk in receipt['chunks']:
        file = path / chunk['path']; assert sha(file) == chunk['sha256']
        start, stop = chunk['start'], chunk['stop']; assert start == end
        z = load(file)
        assert z['forecast'].shape == (stop-start, 64, 9021)
        for key in ['forecast', 'measured_d', 'dmin']:
            assert z[key].shape==(stop-start,64,9021) and z[key].dtype==np.float32
            assert np.isfinite(z[key]).all(), (file, key)
        assert np.array_equal(z['forecast'].min(-1), mechanism['full_forecast_min'][start:stop])
        assert (z['forecast'] <= z['measured_d']).all()
        exempt = np.unpackbits(z['exempt'], axis=-1)[..., :9021].astype(bool)
        missed = np.unpackbits(z['shadow_missed'], axis=-1)[..., :9021].astype(bool)
        for key in ['exempt','shadow_missed']:
            assert z[key].shape==(stop-start,64,1128) and z[key].dtype==np.uint8
            assert not np.unpackbits(z[key],axis=-1)[...,9021:].any()
        assert np.array_equal(missed.sum(-1), mechanism['shadow_reference_missed_count'][start:stop])
        shadow_nonexempt_env_steps += int((missed & ~exempt).any(-1).sum())
        full_negative_pre_env_steps += int(((z['measured_d'] < 0) & ~exempt).any(-1).sum())
        below_braking_band_pre_env_steps += int(((z['measured_d']-z['dmin'] < 0) & ~exempt).any(-1).sum())
        masks={}
        for key in ['selected_ids','baseline_ids']:
            ids=z[key];assert ids.shape==(stop-start,64,capacity) and ids.dtype==np.int32
            assert ((ids>=-1)&(ids<9021)).all()
            valid=ids>=0;flat=ids.reshape(-1,capacity);fv=valid.reshape(-1,capacity)
            dense=np.zeros((flat.shape[0],9021),bool)
            rr=np.broadcast_to(np.arange(flat.shape[0])[:,None],flat.shape)
            dense[rr[fv],flat[fv]]=True
            masks[key]=dense.reshape(stop-start,64,9021)
            assert np.array_equal(masks[key].sum(-1),valid.sum(-1)), 'duplicate selected IDs'
        base=masks['baseline_ids'];actual=masks['selected_ids']
        critical=z['measured_d']<=z['dmin']+.010
        assert (base|~critical).all(), 'baseline missing raw-critical row'
        if capacity==1024:
            expected=base|(z['forecast']<=z['dmin']+.010)
            assert np.array_equal(expected,actual), 'forecast union membership differs; no dropped/swapped rows'
            added=actual&~base
            assert np.array_equal(added.sum(-1),mechanism['forecast_added_count'][start:stop])
        else:
            assert np.array_equal(base,actual) and not mechanism['forecast_added_count'][start:stop].any()
        valid=z['selected_ids']>=0
        assert np.array_equal(valid.sum(-1), data['critical_selected_count'][start:stop])
        assert np.array_equal(valid.sum(-1),mechanism['required_rows'][start:stop])
        # Full raw nonexempt pre geometry matches the previous dense post frame.
        margin=np.stack([np.where(exempt| (evaluation_cls!=c),np.inf,z['measured_d']).min(-1) for c in range(4)],-1)
        offset=1 if start==0 else 0
        assert np.array_equal(margin[offset:],data['official_margins'][start+offset-1:stop-1])
        selected += int(valid.sum()); end=stop
    assert end == 960
    diag = load(path / 'project_diagnostics.npz')
    for k, v in diag.items():
        assert v.shape[0] == 960 and np.isfinite(v).all(), k
    # The returned command includes R19. The target delta includes the later
    # original soft-limit target clamp; both are distinct from measured PD motion.
    assert np.array_equal(diag['outer_returned_cmd'], data['exec'])
    assert np.array_equal(diag['external_raw_cmd'], data['cmd'])
    assert np.array_equal(diag['governor_changed'], data['governor_changed'])
    before = np.concatenate([data['q_initial'][None], data['controller_target'][:-1]])
    assert np.array_equal(data['effective_target_delta'], data['controller_target']-before)
    debt = before-np.concatenate([data['q_initial'][None],data['q'][:-1]])
    assert np.array_equal(debt, data['pre_target_debt'])
    fifo_pre = np.stack([np.concatenate([
        np.repeat(data['q_initial'][None], max(0, 6-i), 0),
        data['controller_target'][:max(0,960-(6-i))]])[:960] for i in range(6)], 1)
    assert np.array_equal(fifo_pre, data['pre_pending_actuator_targets'])
    assert data['pre_pending_project_history'].shape==(960,6,64,26)
    assert np.isfinite(data['pre_pending_project_history']).all()
    assert np.array_equal(fifo_pre,data['pre_pending_project_history']), 'independent project target history differs from original FIFO order'
    summaries = {}
    for robot in ['F','U']:
        for kind in ['original_safety', 'returned_safety', 'returned_alpha', 'returned_bound',
                     'target_safety','target_alpha','target_bound']:
            key=kind+'_residual_'+robot
            summaries[key] = dict(maximum=float(diag[key].max()), env_steps_over_solver_tol=int((diag[key] > 1e-6).sum()))
        key='individual_infeasibility_lower_bound_'+robot
        summaries[key]=dict(maximum=float(diag[key].max()), positive_env_steps=int((diag[key]>0).sum()))
    return dict(all_960_full9021_forecast_finite=True, rows_per_step=9021, frames=960,
        values_checked=960*64*9021, selected_ids_count=selected,
        full_J_scope='producer finite checks all9021 before each forecast; selected J at9 snapshots, no full J archive',
        actual_fifo_pending_exact=True, separate_project_history_saved=True,project_history_order_exact=True,
        complete_selected_union_membership_verified=True,full_pre_margin_bound_to_previous_dense_post=True,
        raw_proposal_not_conservative_shadow_nonexempt_env_steps=shadow_nonexempt_env_steps,
        negative_full_pre_geometric_margin_env_steps=full_negative_pre_env_steps,
        below_braking_dmin_pre_env_steps=below_braking_band_pre_env_steps,
        maximum_target_debt_rad=float(np.abs(debt).max()),
        unreachable_reference_env_steps=int(mechanism['reference_envelope_unreachable'].any(-1).sum()),
        governor_modified_env_steps=int(data['governor_changed'].any(-1).sum()),
        max_target_project_difference_rad=float(np.abs(data['effective_target_delta']-data['exec']).max()),
        residuals=summaries)


def forecast_audit_with_receipt(path, data):
    """Reuse only a closed, source-bound audit with every consumed byte rehashed."""
    cache = HERE / 'closed_cell_audits' / (path.name + '.json')
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved['analysis_sha256'] == sha(__file__):
            for rel, digest in saved['input_sha256'].items():
                assert sha(path / rel) == digest, ('closed audit input changed', path, rel)
            return saved['forecast_audit']
    audit = forecast_audit(path, data)
    receipt = json.loads((path / 'forecast_receipts.json').read_text())
    consumed = ['cell_001.npz', 'mechanism.npz', 'full_row_identity.json',
                'guard_metadata.json', 'forecast_receipts.json', 'project_diagnostics.npz']
    consumed += [c['path'] for c in receipt['chunks']]
    inputs = {rel: sha(path / rel) for rel in consumed}
    for c in receipt['chunks']:
        assert inputs[c['path']] == c['sha256']
    cache.parent.mkdir(exist_ok=True)
    assert not cache.exists(), 'preserve an existing source-version audit rather than overwrite'
    cache.write_text(json.dumps(dict(schema='safeduo.closed_forecast_audit.v1',
        path=str(path), analysis_sha256=sha(__file__), input_sha256=inputs,
        forecast_audit=audit), indent=2, allow_nan=False)+'\n')
    return audit


def main():
    plan=json.loads((HERE/'holdout_plan_v2.json').read_text()); root=Path(plan['output_root'])
    campaign=json.loads((root/'campaign.json').read_text())
    assert campaign['status'] in ['complete','complete_with_failures'] and len(campaign['jobs'])==12
    assert len({j['pid'] for j in campaign['jobs']})==12
    registered={job['id']:job for job in plan['jobs']}
    assert {job['id'] for job in campaign['jobs']}==set(registered)
    for path,digest in plan['source_sha256'].items(): assert sha(Path(plan['cwd'])/path)==digest
    for path,digest in plan['research_source_sha256'].items(): assert sha(path)==digest
    assert sha(plan['checkpoint_path'])==plan['checkpoint_sha256']
    rows=[]; datasets={}; inputs={}; labels={}; protocols=[]
    for job in campaign['jobs']:
        mode=job['id'].rsplit('_',1)[0]
        if job['status']!='complete':
            rows.append(dict(id=job['id'], mode=mode,status='invalid',completed_windows=0,invalid_windows=64,error=job.get('error')))
            continue
        row,b,d,p,a=inspect(root,job); row['mode']=mode
        expected=registered[job['id']]
        assert job['argv']==expected['argv']+['--out',str(root/job['id'])]
        assert p['source_sha256']==plan['source_sha256'] and p['checkpoint_sha256']==plan['checkpoint_sha256']
        for key,value in expected['expected'].items():assert p['design'][0][key]==value
        for key,value in expected['expected_args'].items():assert p['args'][key]==value
        bank_path=Path(expected['env']['SAFEDUO_INITIAL_BANK_NPZ']);bank=load(bank_path)
        assert np.array_equal(b['q_initial'],bank['accepted_q'])
        g=json.loads((root/job['id']/'guard_metadata.json').read_text())
        A=mode in ['admission_guard','joint_guard'];E=mode in ['envelope_guard','joint_guard']
        assert g['mode']==expected['env']['SAFEDUO_JOINT_MODE']==mode and g['admission']==A and g['reference']==E
        assert g['capacity']==(1024 if A else 512) and g['strict_fifo_steps']==6 and not g['queue_preemption']
        for path,digest in g['sources'].items():assert digest==plan['research_source_sha256'][path]==sha(path)
        row['forecast_audit']=forecast_audit_with_receipt(root/job['id'],d)
        rows.append(row); datasets[job['id']]=d; inputs[job['id']]=b;labels[job['id']]=a['risk_pair_index']; protocols.append(p)
    for p in protocols:
        for key in ['source_sha256','checkpoint_sha256','resolved_config','effective_backstop','effective_coordinator']:
            assert p[key]==protocols[0][key]
    modes=json.loads((HERE/'DESIGN.json').read_text())['modes']
    totals=[]; coverage=[]; exposures=[]
    for mode in modes:
        selected=[r for r in rows if r['mode']==mode and r['status']=='complete']
        if not selected: continue
        ids=[r['id'] for r in selected]
        d={k:np.concatenate([datasets[i][k] for i in ids],axis=1) for k in ['q','ee','cmd','exec','official_margins','pair_margin']}
        q0=np.concatenate([inputs[i]['q_initial'] for i in ids]);ee0=np.concatenate([inputs[i]['ee_initial'] for i in ids])
        lim=np.concatenate([inputs[i]['joint_soft_limits'] for i in ids]); lab=np.concatenate([labels[i] for i in ids])
        moving=np.diff(np.concatenate([q0[None],d['q']]),axis=0)
        arms=np.stack([np.linalg.norm(moving[...,s],axis=-1) for s in SLICES],-1)
        totals.append(dict(mode=mode, completed_windows=len(ids)*64, violations=sum(r['violations'] for r in selected),
            deep=sum(r['deep'] for r in selected), class_violations=np.array([r['class_violations'] for r in selected]).sum(0).tolist(),
            four_arms_moving_fraction=float((arms[60:]>.001).all(-1).mean()),
            exec_command_l2_ratio=float(np.linalg.norm(d['exec'][60:],axis=-1).sum()/np.linalg.norm(d['cmd'][60:],axis=-1).sum()),
            zero_prefix_violations=sum(r['zero_prefix_violations'] for r in selected),
            min_nonexempt_mm=min(r['min_nonexempt_mm'] for r in selected),
            max_admitted_rows=max(r['max_admitted_rows'] for r in selected)))
        coverage.append(dict(mode=mode, measured=measured_coverage(q0,d['q'],ee0,d['ee'],lim,np.ones(len(lab),bool))))
        for pair,name in enumerate(PAIR_NAMES):
            count=(d['pair_margin'][66:,lab==pair,pair]<.080).sum(0)
            exposures.append(dict(mode=mode,pair=pair,name=name,assigned=int((lab==pair).sum()),
                exposed_ge3_frames=int((count>=3).sum()),underexposed=int((count<3).sum()),actual_frame_counts=count.tolist()))
    comparisons=[]; patterns=[]; cases=[]
    for registration in json.loads((HERE/'DESIGN.json').read_text())['rows']:
        seed=registration['command_seed']; ids={mode:f'{mode}_{seed}' for mode in modes}
        complete=all(i in datasets for i in ids.values())
        if not complete:
            comparisons.append(dict(seed=seed,status='incomplete; no success imputation'));continue
        base=inputs[ids[modes[0]]]
        for i in ids.values():
            assert np.array_equal(inputs[i]['q_initial'],base['q_initial']) and np.array_equal(inputs[i]['tape'],base['tape'])
            assert np.array_equal(labels[i],labels[ids[modes[0]]])
        bad={m:(datasets[i]['official_margins']<0).any((0,2)) for m,i in ids.items()}
        for a,b in [('baseline_guard','admission_guard'),('baseline_guard','envelope_guard'),
                    ('baseline_guard','joint_guard'),('admission_guard','joint_guard'),('envelope_guard','joint_guard')]:
            comparisons.append(dict(seed=seed,a=a,b=b,initial_exact=True,tape_exact=True,
                a_failures=int(bad[a].sum()),b_failures=int(bad[b].sum()),rescued=int((bad[a]&~bad[b]).sum()),
                new_failures=int((~bad[a]&bad[b]).sum()),both=int((bad[a]&bad[b]).sum()),neither=int((~bad[a]&~bad[b]).sum())))
        interaction=bad['joint_guard'].astype(int)-bad['admission_guard'].astype(int)-bad['envelope_guard'].astype(int)+bad['baseline_guard'].astype(int)
        patterns.append(dict(seed=seed,failures_by_mode={m:int(v.sum()) for m,v in bad.items()},
            descriptive_interaction_sum=int(interaction.sum()),interaction_counts={str(i):int((interaction==i).sum()) for i in range(-2,3)}))
        for e in range(64):
            record=dict(seed=seed,env=e,stratum=int(labels[ids[modes[0]]][e]),methods={})
            for m,i in ids.items():
                margins=datasets[i]['official_margins'][:,e]; fail=(margins<0).any(-1)
                record['methods'][m]=dict(failed=bool(fail.any()),deep=bool((margins<-.005).any()),
                    first_failure_step=int(fail.argmax()) if fail.any() else None,
                    min_mm=(margins.min(0)*1000).tolist())
            cases.append(record)
    result=dict(schema='safeduo.joint_guard_results.v1', candidate_status='experimental_not_promoted',
        campaign_status=campaign['status'], registered_method_windows=768,
        completed_method_windows=sum(r['completed_windows'] for r in rows),invalid_method_windows=sum(r['invalid_windows'] for r in rows),
        unique_new_initials=192,unique_new_command_windows=192,rows=rows,totals=totals,
        paired_comparisons=comparisons,paired_four_patterns=patterns,case_endpoints=cases,
        actual_risk_exposure=exposures,measured_coverage=coverage,actor_sha256=plan['checkpoint_sha256'],
        bank_audit=json.loads((HERE/'bank_audit.json').read_text()),development=json.loads((HERE/'development_equivalence_v2.json').read_text()),
        original_negative_endpoint=True,deep_threshold_m=-.005,hardware_approved=False,
        statistics='conditioned risk samples; paired repeated conditions; descriptive counts, no IID reliability or full26D volume claim',
        forecast_limit='raw proposal can miss reference-limited proposal; passive misses retained; linear forecast not nonlinear PD guarantee',
        analysis_sha256=sha(__file__))
    (HERE/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['campaign_status','completed_method_windows','invalid_method_windows','totals']},ensure_ascii=False))


if __name__=='__main__':main()
