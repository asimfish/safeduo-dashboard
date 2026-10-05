"""Exact first-physical-failure state binding and independent hard-set LP check."""
from pathlib import Path
import json,hashlib,numpy as np
from scipy.optimize import linprog
HERE=Path(__file__).resolve().parent
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    result=[];source=sha(Path(__file__))
    for plan_path in sorted((HERE/'holdout_v2').glob('*_plan.json')):
        plan=json.loads(plan_path.read_text());root=Path(plan['output_root']);campaign=json.loads((root/'campaign.json').read_text());assert campaign['status'] in ('complete','complete_with_failures')
        for job in campaign['jobs']:
            if job['status']!='complete':continue
            path=root/job['id'];z=load(path/'cell_001.npz');identity=json.loads((path/'full_row_identity.json').read_text())
            receipt_path=path/'first_failure_receipts.json';receipt_sha=sha(receipt_path);receipt=json.loads(receipt_path.read_text());seen=set()
            first=((z['official_margins']<0).any(-1));bad=first.any(0);assert receipt['envs_with_failure']==int(bad.sum())
            for r in receipt['receipts']:
                file=path/r['path'];assert sha(file)==r['sha256'];s=load(file);t=r['step'];assert int(s['step'])==t and np.array_equal(s['env_ids'],r['env_ids'])
                for j,e in enumerate(s['env_ids']):
                    e=int(e);assert e not in seen and first[:,e].any() and int(first[:,e].argmax())==t;seen.add(e)
                    assert np.array_equal(s['pre_q'][j],z['q_initial'][e] if t==0 else z['q'][t-1,e])
                    assert np.array_equal(s['post_q'][j],z['q'][t,e]) and np.array_equal(s['snapshot_qd'][j],z['pre_qd_compact'][t,e])
                    assert np.array_equal(s['pre_pending_actuator_targets'][j],z['pre_pending_actuator_targets'][t,:,e])
                    assert np.array_equal(s['actual_project_return'][j],z['exec'][t,e]) and np.array_equal(s['returned_cmd'][j],z['effective_target_delta'][t,e])
                    assert np.array_equal(s['pre_issued_target'][j],z['q_initial'][e] if t==0 else z['controller_target'][t-1,e])
                    cls=np.array(identity['class_id']);pairs=np.array(identity['pair_sphere_idx']);arms=np.array(identity['sphere_arm_id']);ec=cls.copy();ec[(cls==1)&(arms[pairs[:,0]]>=2)]=2;ec[cls==2]=3
                    margin=np.array([np.where(s['post_full_exempt'][j]|(ec!=k),np.inf,s['post_full_d'][j]).min() for k in range(4)],np.float32)
                    assert np.array_equal(margin,z['official_margins'][t,e])
                    neg=np.flatnonzero((s['post_full_d'][j]<0)&~s['post_full_exempt'][j]);assert len(neg)
                    nearest=int(neg[np.argmin(s['post_full_d'][j,neg])]);idx=s['selected_ids'][j];pos=np.flatnonzero(idx==nearest)
                    checks={};min_residual=0.
                    for robot,sl in [('F',slice(0,14)),('U',slice(14,26))]:
                        rel=s['snapshot_rel_'+robot][j];arel=s['snapshot_alpha_rel_'+robot][j]
                        G=s['snapshot_G_'+robot][j][rel].astype(np.float64);h=s['snapshot_h_after_authority_'+robot][j][rel].astype(np.float64)
                        ag=s['snapshot_alpha_G_'+robot][j][arel].astype(np.float64);ah=s['snapshot_alpha_h_'+robot][j][arel].astype(np.float64)
                        lo=s['bounds_lower'][j,sl].astype(np.float64);hi=s['bounds_upper'][j,sl].astype(np.float64)
                        A=np.vstack([G,ag]);b=np.r_[h,ah];bounds=list(zip(lo,hi))
                        norm=np.linalg.norm(A,axis=1);norm=np.where(norm>0,norm,1.)
                        An=A/norm[:,None];bn=b/norm
                        lp=linprog(np.zeros(len(lo)),A_ub=An if len(b) else None,b_ub=bn if len(b) else None,bounds=bounds,method='highs',options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
                        if lp.status not in (0,2):raise ValueError('undetermined LP at exact failure')
                        u=s['returned_cmd'][j,sl].astype(np.float64)
                        sr=float(np.maximum(G@u-h,0).max(initial=0.));ar=float(np.maximum(ag@u-ah,0).max(initial=0.));min_residual=max(min_residual,sr)
                        checks[robot]=dict(hard_set_feasible=lp.success,highs_status=lp.status,returned_target_safety_residual_m=sr,returned_target_alpha_residual_rad=ar,
                            witness_max_residual=None if not lp.success else float(np.maximum(A@lp.x-b,0).max(initial=0.)),
                            normalized_witness_max_residual=None if not lp.success else float(np.maximum(An@lp.x-bn,0).max(initial=0.)),
                            infeasibility_scope='HiGHS bounded numeric decision; no separate dual certificate in this parent audit')
                    relevant={r:bool(len(pos) and pos[0]<s['snapshot_rel_'+r].shape[1] and s['snapshot_rel_'+r][j,pos[0]]) for r in ('F','U')}
                    result.append(dict(condition=job['id'],mode=job['id'].rsplit('_',1)[0],seed=int(job['id'].rsplit('_',1)[1]),env=e,first_failure_step=t,
                        snapshot_path=str(file),snapshot_sha256=r['sha256'],nearest_negative_row=nearest,nearest_class=int(ec[nearest]),post_negative_rows=neg.tolist(),
                        nearest_selected_previous=bool(len(pos)),nearest_relevant_previous=relevant,nearest_post_margin_mm=float(s['post_full_d'][j,nearest]*1000),
                        maximum_pre_qd_rad_s=float(np.abs(s['snapshot_qd'][j]).max()),maximum_pre_target_debt_rad=float(np.abs(s['pre_issued_target'][j]-s['pre_q'][j]).max()),
                        joint_set_checks=checks,scope='exact preceding selected linear set and next measured first violation; neither unique physical cause nor all9021 future constraints certified'))
            assert seen==set(np.flatnonzero(bad).tolist()) and sha(receipt_path)==receipt_sha
    summary=dict(status='PASS_EXACT_FIRST_FAILURE_BINDING_AND_LP',cases=len(result),records=result,source_sha256=source,
        solver_status_not_physical_safety=True,scope='original strict endpoints, exact first failure snapshots; LP binary64 of saved float32 selected constraints, no threshold epsilon')
    assert sha(Path(__file__))==source
    (HERE/'first_failure_audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary['status'],len(result))
if __name__=='__main__':main()
