"""Measured sampled joint support and native wrist positions; no workspace proof."""
import hashlib
import json
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parent
H=P.parent
N=Path('/mnt/nas/data/lyf/double_hand/safety_physical_qualified_random_20261009_1613/strong_random_recipe_v1')
ARMS=('F_L','F_R','U_L','U_R')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}

def summarize(q,box):
    u=(q-box[:,0])/(box[:,1]-box[:,0]);assert np.isfinite(u).all()
    assert ((u>=-1e-6)&(u<=1+1e-6)).all()
    bins=np.minimum(np.maximum((u*10).astype(int),0),9)
    pairs=[]
    for i in range(26):
        for j in range(i+1,26):
            pairs.append(dict(joint_i=i,joint_j=j,occupied_10x10_cells=int(np.unique(bins[:,i]*10+bins[:,j]).size)))
    occupied=np.array([p['occupied_10x10_cells']/100 for p in pairs])
    return dict(inputs=len(q),normalized_marginal_min=u.min(0).tolist(),normalized_marginal_max=u.max(0).tolist(),
        marginal_span_fraction=(u.max(0)-u.min(0)).tolist(),lower5percent_observations=(u<=.05).sum(0).tolist(),
        upper5percent_observations=(u>=.95).sum(0).tolist(),joint_pair_grid=dict(bins_per_axis=10,pairs=325,
        occupied_fraction_min=float(occupied.min()),occupied_fraction_median=float(np.median(occupied)),
        occupied_fraction_max=float(occupied.max()),rows=pairs),
        occupied_pair_grids_are_not_26D_or_reachable_volume=True)

def main():
    proposals=load(N/'initial_proposals.npz');geo=load(N/'geometric_bank_v1/bank.npz')
    prefix=json.loads((H/'strong_prefix512_v1/PREFIX512_RESULT_V1.json').read_text());bank=load(N/'long128_v1/selected_inputs_and_targets.npz')
    q=proposals['controlled_q'];box=proposals['soft_box_rad']
    assert np.array_equal(q,geo['all_proposal_q'])
    qualified=np.array([r['bank_position'] for r in prefix['states'] if r['full74_prefix_pass']],int)
    selected=bank['source_bank_position'];assert len(qualified)==263 and len(selected)==128
    assert np.isin(selected,qualified).all()
    cohorts={'proposed32000':q,'geometry_eligible963':q[geo['all_eligible_indices']],
        'first_geometric512':geo['accepted_q'],'full74_prefix263':geo['accepted_q'][qualified],
        'selected128':geo['accepted_q'][selected]}
    coverage={k:summarize(v,box) for k,v in cohorts.items()}
    wrist={};points={};bycohort={'first_geometric512':np.arange(512),'full74_prefix263':qualified,'selected128':selected}
    for arm in ARMS:
        body='left_hand_base' if arm=='F_L' else 'right_hand_base' if arm=='F_R' else 'wrist_3_link'
        i=geo[arm+'_body_names'].tolist().index(body)
        pos=geo[arm+'_body_pos_local'][:,i];assert pos.shape==(512,3) and np.isfinite(pos).all()
        points[arm]=pos
        wrist[arm]=dict(body=body,coordinate='Native body positions minus environment origin; fixed registered roots; meters',
            cohorts={name:dict(samples=len(ids),minimum_xyz_m=pos[ids].min(0).tolist(),maximum_xyz_m=pos[ids].max(0).tolist(),
                span_xyz_m=np.ptp(pos[ids],axis=0).tolist()) for name,ids in bycohort.items()})
    distances=[]
    for i,a in enumerate(ARMS):
        for b in ARMS[i+1:]:
            d=np.linalg.norm(points[a]-points[b],axis=1)
            distances.append(dict(arm_a=a,arm_b=b,cohorts={name:dict(minimum_m=float(d[ids].min()),median_m=float(np.median(d[ids])),maximum_m=float(d[ids].max())) for name,ids in bycohort.items()},wrist_distance_is_not_robot_clearance=True))
    command=bank['reference_target'];modes={}
    for mode in range(4):
        ids=np.flatnonzero(bank['command_mode']==mode);t=command[:,ids];cadence=int(bank['refresh_controls'][ids[0]])
        assert len(ids)==32 and np.array_equal(bank['refresh_controls'][ids],np.full(32,cadence))
        nonrefresh=[step for step in range(1,480) if step%cadence]
        assert all(np.array_equal(t[step],t[step-1]) for step in nonrefresh)
        du=np.abs(np.diff(t,axis=0));normalized=(t-box[:,0])/(box[:,1]-box[:,0])
        modes[str(mode)]=dict(inputs=32,refresh_controls=cadence,planned_distinct_target_samples=32*((480-1)//cadence+1),
            normalized_marginal_span_fraction=np.ptp(normalized,axis=(0,1)).tolist(),
            maximum_registered_target_jump_rad=float(du.max()),registered_target_jump_quantiles_rad=np.quantile(du,[.5,.9,.99,1]).tolist(),
            target_tape_only_not_measured_motion=True)
    result=dict(status='PASS_REGISTERED_INPUT_SUPPORT_AND_NATIVE_WRIST_POSITION_AUDIT',source_sha256={str(x):sha(x) for x in [N/'initial_proposals.npz',N/'geometric_bank_v1/bank.npz',H/'strong_prefix512_v1/PREFIX512_RESULT_V1.json',N/'long128_v1/selected_inputs_and_targets.npz',Path(__file__)]},
        joint_names=proposals['joint_names'].tolist(),soft_box_rad=box.tolist(),cohorts=coverage,
        native_wrist_ranges=wrist,native_inter_wrist_distances=distances,planned_command_support=modes,
        bounds_scope='Observed finite samples at fixed roots, fixed hand and no objects. Not certified feasible/reachable workspace or complete random coverage.',
        prospective_IID_proposals=True,conditioned_selected_states_are_not_unconditional_IID=True,
        formal_holdout=False,safety_acceptance=False)
    (P/'INPUT_COVERAGE_V1.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(P/'native_wrist_positions_v1.npz',**points,qualified_bank_positions=qualified,selected_bank_positions=selected)
    print(result['status'],{k:v['joint_pair_grid']['occupied_fraction_median'] for k,v in coverage.items()},flush=True)

if __name__=='__main__':main()
