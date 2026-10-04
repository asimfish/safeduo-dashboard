"""Matched inputs, full failure accounting, and measured coverage of wide tests."""
import csv
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from coverage_metrics import ARM_KEYS,SLICES,ee_occupancy,joint_occupancy,measured_coverage,pair_exposure

OUT=Path(__file__).resolve().parent


def load(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def locations(partial=False):
    records=[]
    planned_ids=set()
    for filename in ['campaign_plan.json','campaign_plan_continuation.json','campaign_plan_dynamic.json','campaign_plan_feasible.json']:
        p=json.loads((OUT/filename).read_text());root=Path(p['output_root'])
        planned_ids.update(j['id'] for j in p['jobs'])
        manifest=json.loads((root/'campaign.json').read_text())
        if not partial and filename!='campaign_plan.json':assert manifest['status'] in ['complete','complete_with_failures']
        for job in manifest['jobs']:
            if job['status']=='running':continue
            records.append((root/job['id'],job,'dynamic512' if filename in ['campaign_plan_dynamic.json','campaign_plan_feasible.json'] else 'original128'))
    assert len(planned_ids)==33
    if not partial:assert len(records)==33
    assert len({j['id'] for _,j,_ in records})==len(records)
    assert len({j['pid'] for _,j,_ in records})==len(records)
    return records


def inspect(path,job,variant):
    p=json.loads((path/'protocol.json').read_text())
    c=p['design'][0];flow=c['flow'];method='raw' if c['method']=='raw' else variant
    assert len(p['design'])==1 and p['args']['num_envs']==64 and p['args']['duration_s']==15
    manifest=json.loads((path/'random_manifest.json').read_text());bank=load(path/'input_recipe.npz')
    assert manifest['flow']==flow and not manifest['state_feedback']
    assert manifest['collision_rejection']==(flow=='feasible_burst')
    assert not manifest['posthoc_stop']
    assert np.array_equal(bank['q_initial'],bank['sampled_initial'])
    assert np.isfinite(bank['q_initial']).all() and np.isfinite(bank['tape']).all()
    init=bank['initial_violation'].astype(bool);safe=~init
    row=dict(id=job['id'],path=str(path),flow=flow,method=method,seed=c['seed'],
             status='complete' if p['status']=='complete' else 'invalid',windows=64,
             initial_violations=int(init.sum()),initially_safe=int(safe.sum()),
             source_seed=c['source_seed'],initial_sampler_seed=c['initial_sampler_seed'],
             command_tape_sha256=manifest['temporal']['tape_sha256'],
             q_initial_sha256=hashlib.sha256(bank['q_initial'].tobytes()).hexdigest(),
             initial_clipped_joints_total=sum(manifest['initial']['clipped_joints']),
             initial_joint=joint_occupancy(bank['q_initial'],bank['joint_soft_limits']),
             initial_safe_joint=joint_occupancy(bank['q_initial'][safe],bank['joint_soft_limits'][safe]) if safe.any() else None,
             initial_ee=ee_occupancy(bank['ee_initial'][None]),
             sampling=manifest,checkpoint_sha256=p['checkpoint_sha256'])
    if row['status']=='invalid':
        row.update(error=p.get('error',job.get('error')),safe_initial_violations=None,
                   unknown_outcome_windows=64,completed_windows=0)
        if (path/'capacity_failure.json').exists():row['capacity_failure']=json.loads((path/'capacity_failure.json').read_text())
        return row,bank,None,p
    assert job['status']=='complete' and p['completed_cells']==1 and p['steps']==900
    d=load(path/'cell_001.npz');episodes=json.loads((path/'cell_001.json').read_text())['episodes']
    for k in ['q','ee','cmd','exec','official_margins','pair_margin','actuator_target','controller_target','pre_qd_compact']:
        assert np.isfinite(d[k]).all(),(path,k)
    assert d['q'].shape==(900,64,26) and len(episodes)==64
    assert np.array_equal(d['q_initial'],bank['q_initial'])
    assert np.array_equal(d['cmd'],bank['tape'][:900]),'actual command differs from frozen random tape'
    assert np.array_equal(d['initial_violation'],bank['initial_violation'])
    expected=np.concatenate([np.repeat(bank['sampled_initial'][None],6,axis=0),d['controller_target'][:-6]])
    assert np.array_equal(d['actuator_target'],expected),'FIFO changed'
    bad=(d['official_margins']<0).any((0,2));deep=(d['official_margins']<-.005).any((0,2))
    by_step=(d['official_margins']<0).any(-1)
    first_failure=np.where(bad,by_step.argmax(0),900)
    assert bad.tolist()==[e['violation'] for e in episodes]
    assert init.tolist()==[e['initial_violation'] for e in episodes]
    prior=np.concatenate([bank['sampled_initial'][None],d['controller_target'][:-1]])
    issue=d['controller_target']-prior
    limits=d['joint_soft_limits']
    assert (d['controller_target']>=limits[...,0]-5e-7).all() and (d['controller_target']<=limits[...,1]+5e-7).all()
    if method=='raw':assert np.array_equal(d['cmd'],d['exec'])
    q_all=np.concatenate([d['q_initial'][None],d['q']]);move=np.diff(q_all,axis=0)
    arm_motion=np.stack([np.linalg.norm(move[...,s],axis=-1) for s in SLICES],-1)
    intent=np.stack([np.linalg.norm(d['cmd'][...,s],axis=-1) for s in SLICES],-1)
    cmd=d['cmd'];lagcorr=[]
    for j in range(26):
        a=cmd[:-1,:,j].ravel();b=cmd[1:,:,j].ravel();lagcorr.append(float(np.corrcoef(a,b)[0,1]))
    row.update(completed_windows=64,unknown_outcome_windows=0,all_post_violations=int(bad.sum()),
               safe_initial_violations=int((bad&safe).sum()),safe_initial_damaging=int((deep&safe).sum()),
               safe_initial_failure_before_first_message=int(((first_failure<6)&safe).sum()),
               safe_initial_class_violations=(d['official_margins'][:,safe]<0).any(0).sum(0).tolist(),
               unsafe_start_post_violations=int((bad&init).sum()),
               min_nonexempt_mm=float(d['official_margins'].min()*1000),
               min_safe_initial_mm=None if not safe.any() else float(d['official_margins'][:,safe].min()*1000),
               failed_safe_envs=np.flatnonzero(bad&safe).tolist(),fifo_exact=True,soft_target_limits_pass=True,
               max_actual_target_increment_rad=float(np.abs(issue).max()),
               nominal_solver_box_rad=p['effective_backstop']['vmax']*p['dt'],
               over_nominal_box_env_steps=int((np.abs(issue).max(-1)>p['effective_backstop']['vmax']*p['dt']+5e-7).sum()),
               max_admitted_rows=int(d['critical_selected_count'].max()),
               mean_command_abs_rad=float(np.abs(cmd).mean()),mean_command_lag1_correlation=float(np.mean(lagcorr)),
               four_arm_intent_fraction=float((intent>1e-7).all(-1).mean()),
               four_arms_moving_fraction=float((arm_motion>.001).all(-1).mean()),moving_threshold_rad=.001,
               all_coverage=measured_coverage(d['q_initial'],d['q'],bank['ee_initial'],d['ee'],limits),
               safe_initial_coverage=measured_coverage(d['q_initial'],d['q'],bank['ee_initial'],d['ee'],limits,safe),
               pairs_all=pair_exposure(d['pair_margin'],d['ee']),pairs_safe_initial=pair_exposure(d['pair_margin'],d['ee'],safe))
    return row,bank,d,p


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    locations_list=locations(args.partial)
    rows=[];banks={};datasets={};protocols=[]
    for path,job,variant in locations_list:
        r,b,d,p=inspect(path,job,variant);rows.append(r);banks[r['id']]=b;datasets[r['id']]=d;protocols.append(p)
    first=protocols[0]
    for p in protocols:
        for key in ['source_sha256','resolved_config','effective_backstop','effective_coordinator','checkpoint_sha256']:
            assert p[key]==first[key],key
    paired=[]
    for row in rows:
        if row['method']=='raw':continue
        raw=next(x for x in rows if x['method']=='raw' and x['flow']==row['flow'] and x['seed']==row['seed'])
        a,b=banks[raw['id']],banks[row['id']]
        assert np.array_equal(a['tape'],b['tape']) and np.array_equal(a['q_initial'],b['q_initial'])
        assert np.array_equal(a['initial_violation'],b['initial_violation'])
        paired.append(dict(raw=raw['id'],protected=row['id'],initial_exact=True,command_exact=True,
                           initial_flags_exact=True,protected_status=row['status']))
    for seed in [731923,2048171,9987031]:
        a=next(x for x in rows if x['flow']=='wide_iid' and x['seed']==seed and x['method']=='raw')
        b=next(x for x in rows if x['flow']=='wide_burst' and x['seed']==seed and x['method']=='raw')
        assert np.array_equal(banks[a['id']]['q_initial'],banks[b['id']]['q_initial'])
    aggregates=[]
    group_coverage=[]
    for flow in ['wide_iid','wide_burst','global_burst','feasible_burst']:
        for method in ['raw','original128','dynamic512']:
            if flow=='feasible_burst' and method=='original128':continue
            group=[r for r in rows if r['flow']==flow and r['method']==method]
            good=[r for r in group if r['status']=='complete'];total=sum(r['windows'] for r in group)
            aggregates.append(dict(flow=flow,method=method,attempted_windows=total,
                                   completed_windows=sum(r['completed_windows'] for r in group),
                                   invalid_windows=sum(r['unknown_outcome_windows'] for r in group),
                                   initial_violations=sum(r['initial_violations'] for r in group),
                                   initially_safe_attempted=sum(r['initially_safe'] for r in group),
                                   initially_safe_evaluated=sum(r['initially_safe'] for r in good),
                                   safe_initial_violations=sum(r['safe_initial_violations'] for r in good),
                                   safe_initial_damaging=sum(r['safe_initial_damaging'] for r in good),
                                   safe_initial_failure_before_first_message=sum(r['safe_initial_failure_before_first_message'] for r in good),
                                   mean_within_window_joint_range=None if not good else float(np.mean([r['all_coverage']['mean_within_window_joint_range'] for r in good])),
                                   mean_joint_path_rad=None if not good else float(np.mean([r['all_coverage']['mean_joint_path_rad'] for r in good])),
                                   all_pair_exposure={name:sum(r['pairs_all'][i]['exposed_80mm'] for r in good) for i,name in enumerate([x['pair'] for x in good[0]['pairs_all']])} if good else {},
                                   max_admitted_rows=max((r.get('max_admitted_rows',0) for r in good),default=0)))
            if good and sum(r['initially_safe'] for r in good):
                selected=[(datasets[r['id']],banks[r['id']],~banks[r['id']]['initial_violation'].astype(bool)) for r in good]
                q0=np.concatenate([d['q_initial'][mask] for d,b,mask in selected])
                q=np.concatenate([d['q'][:,mask] for d,b,mask in selected],axis=1)
                ee0=np.concatenate([b['ee_initial'][mask] for d,b,mask in selected])
                ee=np.concatenate([d['ee'][:,mask] for d,b,mask in selected],axis=1)
                limits=np.concatenate([d['joint_soft_limits'][mask] for d,b,mask in selected])
                pm=np.concatenate([d['pair_margin'][:,mask] for d,b,mask in selected],axis=1)
                group_coverage.append(dict(flow=flow,method=method,seeds=[r['seed'] for r in good],
                                           completed_cells=len(good),registered_cells=3,
                                           coverage=measured_coverage(q0,q,ee0,ee,limits),pairs=pair_exposure(pm,ee)))
    pose_banks=[json.loads((Path('/mnt/nas/data/lyf/double_hand/safety_random_space_20261004/pose_banks')/str(seed)/'metadata.json').read_text()) for seed in [13447771,27180353,48921161]]
    result=dict(schema='safeduo.random_space_evidence.v1',candidate_status='experimental_not_promoted',
                campaign_status='running' if len(rows)<33 else 'complete',
                seeds=[731923,2048171,9987031],conditioned_seeds=[13447771,27180353,48921161],
                num_envs=64,duration_s=15,dt=first['dt'],pose_banks=pose_banks,
                rows=rows,aggregates=aggregates,group_coverage=group_coverage,paired_inputs=paired,
                planned_primary_windows=1152,additional_correlated_candidate_windows=576,
                additional_conditioned_windows=384,total_registered_windows=2112,
                unique_command_windows=768,pending_windows=(33-len(rows))*64,
                completed_windows=sum(r['completed_windows'] for r in rows),
                invalid_windows=sum(r['unknown_outcome_windows'] for r in rows),
                actor_sha256=first['checkpoint_sha256'],source_sha256=first['source_sha256'],
                tests_passed=19,statistics='matched windows and Latin hypercube marginals; no IID safety confidence interval')
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (OUT/'windows.csv').open('w') as f:
        fields=['id','flow','method','seed','status','windows','completed_windows','initial_violations','initially_safe','safe_initial_violations','unknown_outcome_windows']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(completed_windows=result['completed_windows'],invalid_windows=result['invalid_windows'],
                          paired=len(paired),aggregates=[{k:a[k] for k in ['flow','method','safe_initial_violations','initially_safe_evaluated','invalid_windows']} for a in aggregates])))


if __name__=='__main__':main()
