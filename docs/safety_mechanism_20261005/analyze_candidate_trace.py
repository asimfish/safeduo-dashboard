"""Distinguish row admission, constraint semantics and delayed physical failure."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/nas/data/lyf/double_hand/safety_mechanism_20261005/candidate_trace_cells_v2/causal_envelope_60317411_v2')
OLD=Path('/mnt/nas/data/lyf/double_hand/safety_mechanism_20261005/development_cells/mechanism_60317411_envelope_050')
NAMES=['cross','self_F','self_U','table']

def actual_constraints(d,p,t,e,robot):
    cfg=p['effective_backstop'];sl=slice(0,14) if robot==0 else slice(14,26);r='F' if robot==0 else 'U'
    G=-d['pre_row_J_'+r][t,e].astype(float);cap=d['solver_cap'][t,e].astype(float)
    dm=d['pre_row_dmin'][t,e];deff=cap/(cfg['gamma']*p['dt'])+dm
    rel=d['pre_row_valid'][t,e]&d['pre_row_arm_mask'][t,e,:,robot*2:robot*2+2].any(-1)
    if cfg['engage_dist'] is not None:rel&=(deff<cfg['engage_dist'])|(cap<0)
    if cfg['exempt_structural_rows']:
        structural=d['pre_struct_exempt'][t,e].copy()
        if cfg['retain_conditional_rows']:structural&=~d['pre_row_exempt'][t,e]
        drop=structural&(cap>=0)
        if cfg['struct_engage_dist'] is not None:drop&=d['pre_row_d'][t,e]>=cfg['struct_engage_dist']
        rel&=~drop
    box=cfg['vmax']*p['dt'];target=d['pre_target'][t,e,sl];q=d['pre_q'][t,e,sl]
    limits=d['joint_soft_limits'][e,sl];lo=np.maximum(-box,limits[:,0]-target);hi=np.minimum(box,limits[:,1]-target)
    lo=np.clip(q-.050-target,lo,hi);hi=np.clip(q+.050-target,np.maximum(-box,limits[:,0]-target),hi)
    bud=(1+d['solver_priority_p'][t,e])*.5 if robot==0 else (1-d['solver_priority_p'][t,e])*.5
    h=np.where((d['pre_row_cls'][t,e]==0)&(cap>=0),bud*cap,cap)
    if cfg['backlog_aware']:
        debit=G@np.clip(target-q,-box,box)
        if cfg['predict_backlog']:
            for past in d['pre_pending_targets'][t,e,:,sl]:debit=np.maximum(debit,G@np.clip(past-q,-box,box))
        h-=debit
    if cfg['row_authority_clamp'] or cfg['backlog_aware']:
        authority=np.where(G>=0,G*lo,G*hi).sum(-1);h=np.maximum(h,.9*authority)
    cmd=np.clip(d['cmd'][t,e,sl],lo,hi);aG=[];ah=[];off=0
    for arm,width in zip(range(robot*2,robot*2+2),[7,7] if robot==0 else [6,6]):
        norm=np.linalg.norm(cmd[off:off+width]);alpha=d['alpha'][t,e,arm]
        if norm>1e-9 and alpha<1-1e-6:
            row=np.zeros(len(cmd));row[off:off+width]=cmd[off:off+width]/norm;aG.append(row);ah.append(alpha*norm)
        off+=width
    return G,h,rel,lo,hi,np.array(aG).reshape(-1,len(cmd)),np.array(ah)

def main():
    p=json.loads((ROOT/'protocol.json').read_text());assert p['status']=='complete'
    with np.load(ROOT/'cell_001.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    km=json.loads(str(d['kinematic_meta_json']));meta=json.loads((ROOT/'mechanism_metadata.json').read_text())
    with np.load(OLD/'cell_001.npz',allow_pickle=False) as a:
        exact={k:bool(np.array_equal(a[k],d[k])) for k in ['q_initial','cmd','q','exec','official_margins','controller_target','actuator_target']}
        diffs={k:float(np.abs(a[k]-d[k]).max()) for k in ['q','exec','official_margins']}
    queue_exact=bool(np.array_equal(d['pre_pending_targets'],d['pre_actual_queue']))
    bad=(d['official_margins']<0).any(-1);events=[];audits=[]
    cls=np.array(meta['class_index']);conditional=np.array(meta['conditional']);nominal=np.array(meta['nominal_dmin'])
    for e in range(64):
        for c in range(4):
            failures=np.flatnonzero(d['official_margins'][:,e,c]<0)
            if not len(failures):continue
            t=int(failures[0]);pid=int(d['post_class_min_row_id'][t,e,c]);assert cls[pid]==c
            i,j=km['pair_sphere_idx'][pid];sphere1=km['sphere_names'][i]
            sphere2=km['sphere_names'][j] if pid<km['robot_pair_count'] else f'table[{j}]'
            context=[]
            for k in range(max(0,t-12),t+1):
                rows=np.flatnonzero(d['pre_row_valid'][k,e]&(d['pre_row_id'][k,e]==pid));row=int(rows[0]) if len(rows) else None
                rec=dict(step=k,admitted=row is not None)
                if row is not None:
                    distance=float(d['pre_row_d'][k,e,row]);cap=float(d['solver_cap'][k,e,row])
                    structure=bool(d['pre_struct_exempt'][k,e,row]);contact=bool(d['pre_row_exempt'][k,e,row])
                    rec.update(distance_mm=distance*1000,effective_dmin_mm=float(d['pre_row_dmin'][k,e,row]*1000),
                               structural=structure,contact=contact,cap_mm=cap*1000,
                               structural_positive_cap_drop=bool(structure and not contact and cap>=0),
                               arm_rate_mm_s=float(d['pre_row_rate'][k,e,row]*1000),
                               body_rate_mm_s=float(d['pre_row_body_rate'][k,e,row]*1000),
                               full_rate_mm_s=float(d['pre_row_full_rate'][k,e,row]*1000),
                               delivered_linear_backlog_mm=float(d['delivered_row_backlog'][k,e,row]*1000),
                               issued_linear_backlog_mm=float(d['issued_row_backlog'][k,e,row]*1000))
                if c==3:
                    ti=pid-meta['table_start'];rec.update(post_raw_table_mm=float(d['full_table_distance'][k,e,ti]*1000),post_exempt=bool(d['full_table_exempt'][k,e,ti]))
                context.append(rec)
            event=dict(env=e,channel=NAMES[c],first_step=t,row_id=pid,spheres=[sphere1,sphere2],
                       conditional=bool(conditional[pid]),nominal_dmin_mm=float(nominal[pid]*1000),context=context)
            events.append(event)
        if not bad[:,e].any():continue
        first=int(np.flatnonzero(bad[:,e])[0])
        for t in sorted({max(0,first-6),max(0,first-1),first}):
            for robot in range(2):
                G,h,rel,lo,hi,aG,ah=actual_constraints(d,p,t,e,robot);g=G[rel];b=h[rel];n=len(lo)
                A=np.vstack([np.column_stack([g,-np.ones(len(g))]),np.column_stack([aG,np.zeros(len(aG))])])
                res=linprog(np.r_[np.zeros(n),1.],A_ub=A,b_ub=np.r_[b,ah],bounds=list(zip(lo,hi))+[(0,None)],method='highs')
                # Hard alpha rows can conflict with a mandatory nonzero recovery interval;
                # infeasible status is evidence rather than a dropped failure.
                alpha_bounds_infeasible=not res.success
                if alpha_bounds_infeasible:
                    alpha_only=linprog(np.zeros(n),A_ub=aG if len(aG) else None,b_ub=ah if len(ah) else None,bounds=list(zip(lo,hi)),method='highs')
                    assert not alpha_only.success, 'LP failed for a reason other than hard alpha and box infeasibility'
                u=d['exec'][t,e,:14] if robot==0 else d['exec'][t,e,14:]
                actual=float(np.maximum(g@u-b,0).max(initial=0));recorded=float(d['solver_residual'][t,e,robot])
                bypass=bool(d['pre_bypass_arm'][t,e,robot*2:robot*2+2].any())
                if not bypass:assert abs(actual-recorded)<3e-6,(t,e,robot,actual,recorded)
                audits.append(dict(env=e,step=t,robot='F' if robot==0 else 'U',slack_mm=float(res.x[-1]*1000) if res.success else None,alpha_bounds_infeasible=alpha_bounds_infeasible,
                                   actual_residual_mm=actual*1000,recorded_residual_mm=recorded*1000,bypass=bypass))
    table=[x for x in events if x['channel']=='table']
    result=dict(schema='safeduo.mechanism_causal_trace.v1',new_independent_windows=0,repeated_windows=64,
                baseline_observables_exact=exact,max_differences=diffs,history_equals_actual_fifo=queue_exact,
                violation_windows=int(bad.any(0).sum()),class_events={n:sum(e['channel']==n for e in events) for n in NAMES},
                table_conditional_events=sum(e['conditional'] for e in table),
                table_nominal_dmin_mm_counts={str(v):sum(abs(e['nominal_dmin_mm']-v)<1e-4 for e in table) for v in [1.,5.,15.,20.]},
                table_events_admitted_with_structural_drop=sum(any(k.get('structural_positive_cap_drop',False) for k in e['context']) for e in table),
                table_events_missing_rows=sum(any(not k['admitted'] for k in e['context']) for e in table),
                near_failure_lp_snapshots=len(audits),near_failure_lp_infeasible=sum(x['alpha_bounds_infeasible'] or x['slack_mm']>1e-3 for x in audits),
                events=events,solver_audits=audits,analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='frozen envelope candidate exact old-development replay; min safety slack with hard alpha/bounds, explicit hard alpha/bounds infeasibility; linear snapshot is not dynamic safety proof')
    (HERE/'candidate_causal_results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['baseline_observables_exact','history_equals_actual_fifo','violation_windows','class_events','table_conditional_events','table_events_admitted_with_structural_drop','table_events_missing_rows','near_failure_lp_infeasible']}))

if __name__=='__main__':main()
