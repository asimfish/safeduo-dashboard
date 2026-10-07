"""Frozen-state constraint decomposition; all changes are mathematical probes."""
from pathlib import Path
import hashlib,json
import numpy as np
from queue_guard import diagnose_set,queue_linear_probe
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    reg=json.loads((HERE/'DIAGNOSTIC_REGISTRATION.json').read_text())
    for p,v in reg['sources'].items():assert sha(p)==v
    records=json.loads(Path(reg['prior_failure_audit']).read_text())['records']
    output=[];bindings={};cache={};lp_count=0
    for r in records:
        path=Path(r['snapshot_path']);digest=sha(path);assert digest==r['snapshot_sha256']
        if path not in cache:
            with np.load(path,allow_pickle=False) as z:cache[path]={k:z[k].copy() for k in z.files}
        s=cache[path];j=int(np.flatnonzero(s['env_ids']==r['env'])[0]);checks={}
        cell=path.parent.parent/'cell_001.npz'
        if cell not in cache:
            with np.load(cell,allow_pickle=False) as z:cache[cell]={'joint_soft_limits':z['joint_soft_limits'].copy()}
            bindings[str(cell)]=sha(cell)
        lim=cache[cell]['joint_soft_limits'][r['env']].astype(float)
        target=s['pre_issued_target'][j].astype(float)
        reachable_lo=np.maximum(lim[:,0]-target,-reg['vmax_rad_s']*reg['dt_s'])
        reachable_hi=np.minimum(lim[:,1]-target,reg['vmax_rad_s']*reg['dt_s'])
        for robot,sl in [('F',slice(0,14)),('U',slice(14,26))]:
            rel=s['snapshot_rel_'+robot][j];arel=s['snapshot_alpha_rel_'+robot][j]
            G=s['snapshot_G_'+robot][j,rel];h=s['snapshot_h_after_authority_'+robot][j,rel]
            aG=s['snapshot_alpha_G_'+robot][j,arel];ah=s['snapshot_alpha_h_'+robot][j,arel]
            lo=s['bounds_lower'][j,sl];hi=s['bounds_upper'][j,sl]
            variants={
                'captured':(h,lo,hi,aG,ah),
                'widen_bounds_same_frozen_rows_rhs':(h,reachable_lo[sl],reachable_hi[sl],aG,ah),
                'remove_alpha_same_frozen_safety_box':(h,lo,hi,None,None),
                'before_authority_same_frozen_box':(s['snapshot_h_before_authority_'+robot][j,rel],lo,hi,aG,ah),
                'raw_budget_same_frozen_box':(s['snapshot_raw_h_'+robot][j,rel],lo,hi,aG,ah)}
            checks[robot]={name:diagnose_set(G,*args) for name,args in variants.items()};lp_count+=len(variants)
            assert (checks[robot]['captured']['highs_status']==0)==r['joint_set_checks'][robot]['hard_set_feasible']
        valid=s['snapshot_valid'][j] & ~s['snapshot_struct_exempt'][j]
        # Retain conditional rows; this mask is illustrative and never a scoring exemption.
        valid |= s['snapshot_valid'][j] & s['snapshot_contact_exempt'][j]
        J=np.concatenate([s['snapshot_J_F'][j],s['snapshot_J_U'][j]],axis=-1)
        probe=queue_linear_probe(s['snapshot_d'][j],s['snapshot_dmin'][j],J,s['pre_q'][j],s['snapshot_qd'][j],
                                 s['pre_pending_actuator_targets'][j],reg['dt_s'],valid)
        output.append(dict(condition=r['condition'],mode=r['mode'],seed=r['seed'],env=r['env'],
                           first_failure_step=r['first_failure_step'],nearest_negative_row=r['nearest_negative_row'],
                           nearest_class=r['nearest_class'],nearest_post_margin_mm=r['nearest_post_margin_mm'],
                           checks=checks,queue_probe=probe,snapshot_path=str(path),snapshot_sha256=digest,
                           numerical_native_render_available=False))
        assert sha(path)==digest;bindings[str(path)]=digest
    for p,v in bindings.items():assert sha(p)==v
    for p,v in reg['sources'].items():assert sha(p)==v
    counts=[]
    for mode in reg['modes']:
        for robot in ['F','U']:
            group=[r for r in output if r['mode']==mode]
            counts.append(dict(mode=mode,robot=robot,cases=len(group),
                infeasible={v:sum(r['checks'][robot][v]['highs_status']==2 for r in group) for v in group[0]['checks'][robot]},
                captured_infeasible_without_individual_contradiction=sum(r['checks'][robot]['captured']['highs_status']==2 and
                         r['checks'][robot]['captured']['individual_impossible_rows']==0 for r in group)))
    with (HERE/'CONSTRAINT_DECOMPOSITION.json').open('x') as f:
        json.dump(dict(status='PASS_CAPTURED_SELECTED_SETS_RECONCILED',cases=len(output),lp_sets=lp_count,
                       counts=counts,records=output,bindings=bindings,
                       scope='Post-outcome development; fixed selected G/h, altered mathematical bounds/alpha/budgets only. Not recomputed policy, queued trajectory, unique cause or safety improvement.',physical_safety_certified=False),f,indent=2);f.write('\n')
    print('CLOSED_DECOMPOSITION',len(output),lp_count,flush=True)


if __name__=='__main__':main()
