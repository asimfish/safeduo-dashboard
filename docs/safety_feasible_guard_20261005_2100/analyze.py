"""Closed raw-file readback, strict endpoints, complete paired cases; no cache."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(HERE.parent/'safety_random_space_20261004'))
from coverage_metrics import measured_coverage,SLICES,PAIR_NAMES
from dense_audit import forecast_audit,load,sha

def freeze_inputs(path):
    names=['protocol.json','cell_001.npz','input_recipe.npz','bank_assignment.npz','guard_metadata.json','mechanism.npz',
           'project_diagnostics.npz','forecast_receipts.json','full_row_identity.json']
    receipt=json.loads((path/'forecast_receipts.json').read_text())
    names += [r['path'] for r in receipt['chunks']]
    return {n:sha(path/n) for n in names}

def inspect(path,job,plan,dense=True):
    p=json.loads((path/'protocol.json').read_text())
    assert p['status']=='complete' and p['completed_cells']==1
    assert p['source_sha256']==plan['source_sha256'] and p['checkpoint_sha256']==plan['checkpoint_sha256']
    for k,v in job['expected'].items():assert p['design'][0][k]==v
    for k,v in job['expected_args'].items():assert p['args'][k]==v
    before=freeze_inputs(path)
    z=load(path/'cell_001.npz');inp=load(path/'input_recipe.npz');assign=load(path/'bank_assignment.npz')
    bank=load(Path(job['env']['SAFEDUO_INITIAL_BANK_NPZ']))
    mode=job['env']['SAFEDUO_JOINT_MODE'];scale=.36 if mode=='admission_scaled_036' else 1.
    assert z['q'].shape==(960,64,26) and z['official_margins'].shape==(960,64,4)
    for k,v in z.items():
        if v.dtype.kind in 'fiu':assert np.isfinite(v).all(),k
    assert np.array_equal(inp['q_initial'],bank['accepted_q']) and np.array_equal(z['q_initial'],inp['q_initial'])
    assert np.array_equal(assign['risk_pair_index'],bank['risk_pair_index'])
    assert np.array_equal(z['external_unscaled_cmd'],inp['tape'][:960])
    assert np.array_equal(z['cmd'],inp['tape'][:960]*np.float32(scale)), 'actual sampled command scale mismatch'
    diag=load(path/'project_diagnostics.npz')
    if dense:audit=forecast_audit(path,z)
    else:audit={'status':'NOT_RUN_IN_DEVELOPMENT_QUICK_LOOK'}
    # Six frames are already pending at each pre-step: this is not zero-delay.
    applied=np.concatenate([np.repeat(z['q_initial'][None],6,axis=0),z['controller_target'][:-6]])
    assert np.array_equal(z['actuator_target'],applied), 'actual FIFO applied target differs'
    audit['actual_applied_fifo_exact']=True
    repair={}
    if mode=='joint_repair':
        for r in ('F','U'):
            repair[r]={k:dict(positive_env_steps=int((diag[f'repair_{k}_{r}']>0).sum()),maximum=float(diag[f'repair_{k}_{r}'].max()))
                for k in ('applied','alpha_min_slack_rad','safety_min_slack_m','qp_fallback','pre_safety_residual_m')}
    assert freeze_inputs(path)==before,'raw files changed during analysis'
    margins=z['official_margins'];bad=(margins<0).any((0,2));deep=(margins<-.005).any((0,2))
    row=dict(id=job['id'],path=str(path),mode=mode,seed=job['expected']['seed'],status='complete',device=job['expected_args']['device'],
        completed_windows=64,invalid_windows=0,violations=int(bad.sum()),deep=int(deep.sum()),
        class_violations=(margins<0).any(0).sum(0).tolist(),min_nonexempt_mm=float(margins.min()*1000),
        max_admitted_rows=int(z['critical_selected_count'].max()),zero_prefix_violations=int((margins[:60]<0).any((0,2)).sum()),
        audit=audit,repair=repair,input_sha256=before)
    return row,z,inp,assign['risk_pair_index'],p

def main(stage='holdout'):
    if stage=='development':plans=[json.loads((HERE/'development_plan.json').read_text())]
    else:plans=[json.loads(p.read_text()) for p in sorted(HERE.glob('holdout_*_plan.json'))]
    assert plans
    records=[];datasets={};inputs={};labels={};protocols=[]
    source_hash=sha(Path(__file__));audit_hash=sha(HERE/'dense_audit.py')
    for plan in plans:
        for rel,digest in plan['source_sha256'].items():assert sha(Path(plan['cwd'])/rel)==digest
        for path,digest in plan['research_source_sha256'].items():assert sha(Path(path))==digest
        campaign=json.loads((Path(plan['output_root'])/'campaign.json').read_text())
        assert campaign['status'] in ('complete','complete_with_failures')
        jobs={j['id']:j for j in campaign['jobs']}
        assert set(jobs)=={j['id'] for j in plan['jobs']}
        for j in plan['jobs']:
            mode=j['env']['SAFEDUO_JOINT_MODE']
            if jobs[j['id']]['status']!='complete':
                records.append(dict(id=j['id'],mode=mode,status='invalid',completed_windows=0,invalid_windows=64,error=jobs[j['id']].get('error')));continue
            path=Path(plan['output_root'])/j['id']
            row,z,inp,lab,p=inspect(path,j,plan,dense=stage=='holdout')
            records.append(row);datasets[j['id']]=z;inputs[j['id']]=inp;labels[j['id']]=lab;protocols.append(p)
            print('RESCORED',j['id'],row['violations'],row['deep'],flush=True)
    modes=list(dict.fromkeys(r['mode'] for r in records))
    totals=[];coverage=[];exposure=[]
    for mode in modes:
        rows=[r for r in records if r['mode']==mode and r['status']=='complete'];ids=[r['id'] for r in rows]
        if not ids:continue
        z={k:np.concatenate([datasets[i][k] for i in ids],axis=1) for k in ('q','ee','cmd','exec','official_margins','pair_margin','external_unscaled_cmd')}
        q0=np.concatenate([inputs[i]['q_initial'] for i in ids]);ee0=np.concatenate([inputs[i]['ee_initial'] for i in ids]);lim=np.concatenate([inputs[i]['joint_soft_limits'] for i in ids]);lab=np.concatenate([labels[i] for i in ids])
        disp=np.diff(np.concatenate([q0[None],z['q']]),axis=0)
        arms=np.stack([np.linalg.norm(disp[...,s],axis=-1) for s in SLICES],-1)
        totals.append(dict(mode=mode,completed_windows=len(ids)*64,violations=sum(r['violations'] for r in rows),deep=sum(r['deep'] for r in rows),
            min_nonexempt_mm=min(r['min_nonexempt_mm'] for r in rows),class_violations=np.array([r['class_violations'] for r in rows]).sum(0).tolist(),
            four_arms_moving_fraction=float((arms[60:]>.001).all(-1).mean()),
            exec_external_l2_ratio=float(np.linalg.norm(z['exec'][60:],axis=-1).sum()/np.linalg.norm(z['external_unscaled_cmd'][60:],axis=-1).sum()),
            max_admitted_rows=max(r['max_admitted_rows'] for r in rows)))
        coverage.append(dict(mode=mode,measured=measured_coverage(q0,z['q'],ee0,z['ee'],lim)))
        for pair,name in enumerate(PAIR_NAMES):
            counts=(z['pair_margin'][66:,lab==pair,pair]<.080).sum(0)
            exposure.append(dict(mode=mode,name=name,assigned=len(counts),exposed=int((counts>=3).sum()),underexposed=int((counts<3).sum())))
    cases=[];paired=[];unique_tapes=[]
    design=json.loads((HERE/'DESIGN.json').read_text())
    seeds=[1701627244] if stage=='development' else [r['command_seed'] for r in design['rows']]
    for seed in seeds:
        ids={m:f'{m}_{seed}' for m in modes}
        if not all(i in datasets for i in ids.values()):
            paired.append(dict(seed=seed,status='incomplete; failures not imputed as success'));continue
        ref=next(iter(ids.values()))
        for i in ids.values():
            assert np.array_equal(inputs[i]['q_initial'],inputs[ref]['q_initial']) and np.array_equal(inputs[i]['tape'],inputs[ref]['tape'])
            assert np.array_equal(labels[i],labels[ref])
        unique_tapes.extend(hashlib.sha256(np.ascontiguousarray(inputs[ref]['tape'][:960,e]).tobytes()).hexdigest() for e in range(64))
        bad={m:(datasets[i]['official_margins']<0).any((0,2)) for m,i in ids.items()}
        comparisons=[('joint_reference','joint_repair')]
        if stage=='holdout':comparisons += [('admission_full','joint_reference'),('admission_full','joint_repair'),('admission_scaled_036','joint_repair')]
        for a,b in comparisons:
            paired.append(dict(seed=seed,a=a,b=b,rescued=int((bad[a]&~bad[b]).sum()),new_failures=int((~bad[a]&bad[b]).sum()),both=int((bad[a]&bad[b]).sum()),neither=int((~bad[a]&~bad[b]).sum())))
        for e in range(64):
            methods={}
            for m,i in ids.items():
                z=datasets[i];margin=z['official_margins'][:,e];fail=(margin<0).any(-1)
                act=np.stack([np.abs(z[k][:,e]).max(-1) for k in ('pre_target_debt','pre_qd_compact','effective_target_delta')],-1)
                indices=np.unique(np.concatenate([np.arange(0,960,4),[959],margin.argmin(0),act.argmax(0),*[np.flatnonzero(margin[:,k]<0)[:1] for k in range(4)]]))
                methods[m]=dict(failed=bool(fail.any()),deep=bool((margin<-.005).any()),first_failure_step=int(fail.argmax()) if fail.any() else None,
                    min_mm=(margin.min(0)*1000).tolist(),indices=indices.tolist(),curves=(margin[indices]*1000).tolist(),actuation=act[indices].tolist())
            cases.append(dict(seed=seed,env=e,stratum=int(labels[ref][e]),methods=methods))
    receipt=dict(schema='safeduo.feasible_guard_results.v1',stage=stage,registered_method_windows=sum(len(p['jobs'])*64 for p in plans),
        completed_method_windows=sum(r['completed_windows'] for r in records),invalid_method_windows=sum(r['invalid_windows'] for r in records),
        rows=records,totals=totals,coverage=coverage,exposure=exposure,paired=paired,cases=cases,unique_actual_tapes=len(set(unique_tapes)),
        physical_verdict='UNVALIDATED; observed failures retained',hardware_approved=False,
        analysis_source_sha256=source_hash,dense_audit_source_sha256=audit_hash,
        statistics='paired conditioned simulation cases; no IID reliability confidence or full26D volume claim',
        solver_limit='alpha slack rad/step and safety slack metres kept separate; unavoidable positive slack is not safe; residual-free linear projection is not physical safety')
    assert sha(Path(__file__))==source_hash and sha(HERE/'dense_audit.py')==audit_hash
    (HERE/f'{stage}_results.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:receipt[k] for k in ('stage','completed_method_windows','invalid_method_windows','totals')},ensure_ascii=False))
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'holdout')
