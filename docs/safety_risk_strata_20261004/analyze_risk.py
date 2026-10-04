"""Full-window matched risk-stratum evidence, including every prefix failure."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'safety_random_space_20261004'))
from coverage_metrics import PAIR_NAMES,SLICES,measured_coverage,pair_exposure
from risk_recipe import validate_bank

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def audit_banks(plan):
    root=Path('/mnt/nas/data/lyf/double_hand/safety_risk_strata_20261004/risk_banks')
    geo=load(root/'all_geometry.npz');settle=load(root/'all_settling.npz')
    assert len(geo['q'])==len(geo['batch_identity'])*64
    identities={tuple(x):i for i,x in enumerate(settle['identity'])};assert len(identities)==len(settle['identity'])
    rows=[]
    for seed in [60317411,80692357,109441003]:
        path=root/str(seed);b=load(path/'bank.npz');m=json.loads((path/'metadata.json').read_text())
        validate_bank(b['accepted_q'],b['risk_pair_index'],m)
        starts=[];drifts=[];velocities=[]
        risk_restarts={i:set() for i in range(6)}
        for q,label,(ref,e) in zip(b['accepted_q'],b['risk_pair_index'],b['selected_refs']):
            index=int(ref)*64+int(e);ident=tuple(geo['batch_identity'][ref]);s=identities[ident]
            assert ident[0]==seed and ident[1]==label
            assert np.array_equal(q,geo['q'][index])
            assert geo['eligible'][index] and geo['margins'][index].min()>=.001-1e-7 and geo['table'][index]>=.001-1e-7
            if label>=0:assert .020-1e-7<=geo['pairs'][index,label]<=.060+1e-7
            assert settle['eligible'][s,e] and settle['valid'][s,e] and not settle['bad'][s,e]
            if label>=0:
                assert int(e)==int(np.flatnonzero(settle['valid'][s])[0]),'not first qualified sampling order'
                assert ident[2] not in risk_restarts[int(label)],'multiple poses from one risk restart'
                risk_restarts[int(label)].add(ident[2])
            assert settle['drift'][s,e]<=.05 and settle['velocity'][s,e]<=.1 and settle['min_table'][s,e]>=0
            starts.append(float(geo['table'][index]));drifts.append(float(settle['drift'][s,e]));velocities.append(float(settle['velocity'][s,e]))
        rows.append({**m,'initial_table_min_mm':min(starts)*1000,'qualification_max_drift_rad':max(drifts),
                     'qualification_max_final_velocity_rad_s':max(velocities),'all_selection_refs_valid':True,
                     'independent_selected_restarts_per_pair':[len(risk_restarts[i]) for i in range(6)]})
    return dict(rows=rows,sampling=json.loads((root/'sampling_summary.json').read_text()),
                geometry_sha256=hashlib.sha256((root/'all_geometry.npz').read_bytes()).hexdigest(),
                settling_sha256=hashlib.sha256((root/'all_settling.npz').read_bytes()).hexdigest(),
                geometry_pass_candidates=int(geo['eligible'].sum()),settling_pass_candidates=int(settle['valid'].sum()),
                selected_before_policy=True)

def inspect(root,job):
    path=root/job['id'];p=json.loads((path/'protocol.json').read_text());b=load(path/'input_recipe.npz')
    m=json.loads((path/'random_manifest.json').read_text());assignment=load(path/'bank_assignment.npz')
    assert p['args']['num_envs']==64 and p['args']['duration_s']==16 and p['steps']==960
    assert np.array_equal(b['q_initial'],b['sampled_initial']) and not b['initial_violation'].any()
    assert (assignment['initial_table_raw_min']>=.001-1e-7).all()
    assert m['schema']=='safeduo.risk_strata_source.v1' and not m['state_feedback'] and not m['posthoc_stop']
    assert not b['tape'][:60].any() and b['tape'].shape==(962,64,26)
    assert m['temporal']['tape_sha256']==hashlib.sha256(b['tape'].tobytes()).hexdigest()
    row=dict(id=job['id'],path=str(path),seed=p['design'][0]['seed'],method=p['design'][0]['method'],
             status='complete' if p['status']=='complete' and job['status']=='complete' else 'invalid',windows=64,
             initial_violations=int(b['initial_violation'].sum()),sampling=m,command_tape_sha256=m['temporal']['tape_sha256'])
    if row['status']=='invalid':
        row.update(completed_windows=0,invalid_windows=64,error=p.get('error',job.get('error')))
        return row,b,None,p,assignment
    d=load(path/'cell_001.npz');episodes=json.loads((path/'cell_001.json').read_text())['episodes']
    assert d['q'].shape==(960,64,26) and len(episodes)==64
    for key in ['q','ee','cmd','exec','official_margins','pair_margin','table_raw_min','pre_qd_compact','controller_target','actuator_target']:
        assert np.isfinite(d[key]).all(),(job['id'],key)
    assert np.array_equal(d['q_initial'],b['q_initial']) and np.array_equal(d['initial_violation'],b['initial_violation'])
    assert np.array_equal(d['cmd'],b['tape'][:960])
    fifo=np.concatenate([np.repeat(b['q_initial'][None],6,0),d['controller_target'][:-6]])
    assert np.array_equal(d['actuator_target'],fifo),'actual FIFO mismatch'
    limits=d['joint_soft_limits'];target=d['controller_target']
    assert (target>=limits[...,0]-5e-7).all() and (target<=limits[...,1]+5e-7).all()
    if row['method']=='raw':assert np.array_equal(d['cmd'],d['exec'])
    bystep=(d['official_margins']<0).any(-1);bad=bystep.any(0);deep=(d['official_margins']<-.005).any((0,2))
    assert bad.tolist()==[e['violation'] for e in episodes]
    first=np.where(bad,bystep.argmax(0),960)
    issue=target-np.concatenate([b['q_initial'][None],target[:-1]])
    movement=np.diff(np.concatenate([b['q_initial'][None],d['q']]),axis=0)
    arms=np.stack([np.linalg.norm(movement[...,s],axis=-1) for s in SLICES],-1)
    intent=np.stack([np.linalg.norm(d['cmd'][60:,...,s],axis=-1) for s in SLICES],-1)
    row.update(completed_windows=64,invalid_windows=0,violations=int(bad.sum()),deep=int(deep.sum()),
               zero_prefix_violations=int((first<60).sum()),before_first_random_delivery=int((first<66).sum()),
               zero_input_output_nonzero_windows=int((np.abs(d['exec'][:60]).max((0,2))>1e-7).sum()),
               zero_prefix_max_joint_drift_rad=float(np.abs(d['q'][:60]-b['q_initial']).max()),
               class_violations=(d['official_margins']<0).any(0).sum(0).tolist(),
               min_nonexempt_mm=float(d['official_margins'].min()*1000),
               raw_table_overlap_windows=int((d['table_raw_min']<0).any(0).sum()),
               four_arm_intent_fraction=float((intent>1e-7).all(-1).mean()),
               four_arms_moving_fraction=float((arms[60:]>.001).all(-1).mean()),
               exec_command_l2_ratio=float(np.linalg.norm(d['exec'][60:],axis=-1).sum()/np.maximum(np.linalg.norm(d['cmd'][60:],axis=-1).sum(),1e-12)),
               zero_exec_pressure_fraction=float((np.linalg.norm(d['exec'][60:],axis=-1)<=1e-7).mean()),
               intervention_pressure_fraction=float((np.abs(d['exec'][60:]-d['cmd'][60:]).max(-1)>1e-7).mean()),
               max_actual_target_increment_rad=float(np.abs(issue).max()),nominal_solver_box_rad=p['effective_backstop']['vmax']*p['dt'],
               over_nominal_box_env_steps=int((np.abs(issue).max(-1)>p['effective_backstop']['vmax']*p['dt']+5e-7).sum()),
               max_admitted_rows=int(d['critical_selected_count'].max()),fifo_exact=True,soft_target_limits_pass=True,
               external_command_min_rad_per_joint=d['cmd'][60:].min((0,1)).tolist(),
               external_command_max_rad_per_joint=d['cmd'][60:].max((0,1)).tolist(),
               external_joint_window_both_signs_fraction=float(((d['cmd'][60:].min(0)<0)&(d['cmd'][60:].max(0)>0)).mean()),
               observed_hold_counts={str(h):int((b['holds'][60:960]==h).sum()) for h in [1,4,15,30,90,180]},
               observed_amplitude_counts={str(a):int(((np.abs(b['segment_amplitudes'][60:960]-a)<1e-8)&b['updates'][60:960]).sum()) for a in [.005,.015,.025,.05]},
               failed_envs=np.flatnonzero(bad).tolist(),first_failure_step=first.tolist(),risk_pair_index=assignment['risk_pair_index'].tolist())
    return row,b,d,p,assignment

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    plan=json.loads((HERE/'campaign_plan.json').read_text());root=Path(plan['output_root'])
    campaign=json.loads((root/'campaign.json').read_text())
    if not args.partial:assert campaign['status'] in ['complete','complete_with_failures'] and len(campaign['jobs'])==6
    for key,sha in plan['source_sha256'].items():assert hashlib.sha256((Path(plan['cwd'])/key).read_bytes()).hexdigest()==sha,key
    for key,sha in plan['research_source_sha256'].items():assert hashlib.sha256(Path(key).read_bytes()).hexdigest()==sha,key
    assert hashlib.sha256(Path(plan['checkpoint_path']).read_bytes()).hexdigest()==plan['checkpoint_sha256']
    rows=[];banks={};datasets={};protocols=[];assignments={}
    for job in campaign['jobs']:
        if job['status']=='running':continue
        r,b,d,p,a=inspect(root,job);rows.append(r);banks[r['id']]=b;datasets[r['id']]=d;protocols.append(p);assignments[r['id']]=a
    assert len({j['pid'] for j in campaign['jobs']})==len(campaign['jobs'])
    first=protocols[0]
    for p in protocols:
        for key in ['source_sha256','checkpoint_sha256','resolved_config','effective_backstop','effective_coordinator']:
            assert p[key]==first[key],key
    strata=[];coverage=[];paired=[]
    for method in ['raw','system0']:
        selected=[r for r in rows if r['method']==method and r['status']=='complete']
        if not selected:continue
        d={k:np.concatenate([datasets[r['id']][k] for r in selected],axis=1) for k in ['q','ee','official_margins','pair_margin','table_raw_min']}
        q0=np.concatenate([banks[r['id']]['q_initial'] for r in selected]);ee0=np.concatenate([banks[r['id']]['ee_initial'] for r in selected])
        limits=np.concatenate([banks[r['id']]['joint_soft_limits'] for r in selected]);labels=np.concatenate([assignments[r['id']]['risk_pair_index'] for r in selected])
        for label,name in [(-2,'全部风险分层'),(-1,'广域稳定初态'),*enumerate(PAIR_NAMES)]:
            mask=labels>=0 if label==-2 else labels==label
            margins=d['official_margins'][:,mask];bad=(margins<0).any((0,2));deep=(margins<-.005).any((0,2))
            bystep=(margins<0).any(-1);failure=np.where(bad,bystep.argmax(0),960)
            n=int(mask.sum());pairs=pair_exposure(d['pair_margin'],d['ee'],mask)
            pressure_pairs=pair_exposure(d['pair_margin'][66:],d['ee'][66:],mask)
            strata.append(dict(method=method,stratum=label,label=name,windows=n,planned_windows=144 if label==-2 else 48 if label==-1 else 24,
                               violations=int(bad.sum()),deep=int(deep.sum()),zero_prefix_violations=int((failure<60).sum()),
                               before_first_random_delivery=int((failure<66).sum()),class_violations=(margins<0).any(0).sum(0).tolist(),
                               target_pair_exposure=None if label<0 else pairs[label],
                               target_pair_pressure_exposure=None if label<0 else pressure_pairs[label],
                               raw_table_overlap_windows=int((d['table_raw_min'][:,mask]<0).any(0).sum())))
            if label<0:
                c=measured_coverage(q0,d['q'],ee0,d['ee'],limits,mask)
                coverage.append(dict(method=method,stratum=label,label=name,seeds=[r['seed'] for r in selected],
                                     coverage=c,pairs=pairs,pressure_pairs=pressure_pairs,all_failures_retained=True))
    for sysrow in [r for r in rows if r['method']=='system0']:
        raw=next(r for r in rows if r['method']=='raw' and r['seed']==sysrow['seed']);a,b=banks[raw['id']],banks[sysrow['id']]
        assert np.array_equal(a['tape'],b['tape']) and np.array_equal(a['q_initial'],b['q_initial']) and np.array_equal(a['initial_violation'],b['initial_violation'])
        pair=dict(raw=raw['id'],protected=sysrow['id'],command_exact=True,initial_exact=True,initial_flags_exact=True)
        if sysrow['status']=='complete' and raw['status']=='complete':
            da,db=datasets[raw['id']],datasets[sysrow['id']];ra=(da['official_margins']<0).any((0,2));sy=(db['official_margins']<0).any((0,2))
            pair.update(pre_first_random_delivery_q_exact=bool(np.array_equal(da['q'][:66],db['q'][:66])),
                        pre_first_random_delivery_q_max_diff_rad=float(np.abs(da['q'][:66]-db['q'][:66]).max()),
                        raw_only_failure=int((ra&~sy).sum()),protected_only_failure=int((sy&~ra).sum()),both_failure=int((ra&sy).sum()),neither_failure=int((~ra&~sy).sum()))
        paired.append(pair)
    result=dict(schema='safeduo.risk_strata_evidence.v1',campaign_status=campaign['status'],actor_sha256=plan['checkpoint_sha256'],
                source_sha256=plan['source_sha256'],candidate_status='experimental_not_promoted',
                completed_windows=sum(r['completed_windows'] for r in rows),invalid_windows=sum(r['invalid_windows'] for r in rows),
                pending_windows=(6-len(rows))*64,new_unique_command_windows=192,registered_windows=384,
                zero_diagnostic=json.loads((HERE/'zero_results.json').read_text()),bank_audit=audit_banks(plan),
                dt=first['dt'],rows=rows,strata=strata,group_coverage=coverage,paired_inputs=paired,
                statistics='conditioned risk distributions and matched correlated windows; no IID confidence interval',
                analysis_source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'risk_recipe.py',HERE.parent/'safety_random_space_20261004/coverage_metrics.py']},
                secondary_analysis_note='Post-random-delivery exposure added as descriptive supplement after first paired cell; no bank reselection, endpoint changes or exclusion',
                test_contracts=2,production_source_unchanged=True)
    (HERE/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    with (HERE/'windows.csv').open('w') as f:
        fields=['id','seed','method','status','windows','completed_windows','invalid_windows','violations','deep','zero_prefix_violations','before_first_random_delivery']
        writer=csv.DictWriter(f,fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:result[k] for k in ['campaign_status','completed_windows','invalid_windows','pending_windows']}))

if __name__=='__main__':main()
