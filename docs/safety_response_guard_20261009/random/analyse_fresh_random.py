"""Paired all-arm force/geometry scoring with shared six-step qualification."""
from pathlib import Path
import hashlib,json,numpy as np
H=Path(__file__).resolve().parent
R=Path('/mnt/nas/data/lyf/double_hand/safety_feasible_response_20261009_0316')
ARMS=('F_L','F_R','U_L','U_R')
reg=json.loads((H/'REGISTRATION_DYNAMIC_V1.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
outputs={};prefixes={}; coverage={}
with np.load(R/'geometric_bank_v1/bank.npz') as b:
    limits=b['soft_limits'];initial_q=b['accepted_q'];selected=b['selected_indices']
    bank_span=np.ptp(initial_q,axis=0)/(limits[:,1]-limits[:,0])
    proposal_span=np.ptp(b['all_proposal_q'],axis=0)/(limits[:,1]-limits[:,0])
    bank_eef={a:b[a+'_body_pos_local'][:,list(b[a+'_body_names']).index('left_hand_base' if a=='F_L' else 'right_hand_base' if a=='F_R' else 'wrist_3_link')] for a in ARMS}
for method in ['raw','multirow']:
    root=R/(method+'_v1');protocol=json.loads((root/'response_protocol.json').read_text())
    assert protocol['status']=='complete' and protocol['completed_steps']==480
    oracle=json.loads((H/('FORCE_ORACLE_'+method+'.json')).read_text())
    assert oracle['every_scalar_summary_exact'] and oracle['all_enabled_arm_collider_owners_covered']
    with np.load(root/'resolved_native_parameters.npz') as p:
        idx=[p[a+'_controlled_joint_indices'] for a in ARMS]
        hard=np.concatenate([p[a+'_hard_limits'][0,i] for a,i in zip(ARMS,idx)],0)
        vmax=np.concatenate([p[a+'_max_velocity'][0,i] for a,i in zip(ARMS,idx)])
    with np.load(root/'initial_geometry.npz') as z:
        assert (z['d'].min(1)>=.0001).all()
        assert np.array_equal(np.concatenate([z[a+'_q'][:,i] for a,i in zip(ARMS,idx)],-1),initial_q)
        init_min=z['d'].min(1)
        actual_velocity=np.concatenate([z[a+'_qd'][:,i] for a,i in zip(ARMS,idx)],-1)
    with np.load(root/'response_stream.npz') as z:
        post=z['post_d'];min_d=post.min(-1);del post
        force=z['scalar_normal_max_N']
        geom_bad=min_d.min(0)<0;force_bad=force.max(0)>.1
        prefixes[method]={k:z[k][:6] for k in ['applied_target']+[f'post_{a}_{f}' for a in ARMS for f in ['q','qd']]}
        prefix_eligible=(min_d[:6].min(0)>=0)&(force[:6].max(0)<=.1)
        q=np.concatenate([z['post_'+a+'_q'][:,:,i] for a,i in zip(ARMS,idx)],-1)
        q=np.concatenate([initial_q[None],q],0)
        paths=np.abs(np.diff(q,axis=0)).sum((0,2));arm_paths=[np.abs(np.diff(q[:,:,sl],axis=0)).sum((0,2)) for sl in [slice(0,7),slice(7,14),slice(14,20),slice(20,26)]]
        per_window_span=np.ptp(q,axis=0)/(limits[:,1]-limits[:,0])
        actual_union_span=np.ptp(q.reshape(-1,26),axis=0)/(limits[:,1]-limits[:,0])
        hard_breach=((q<hard[None,None,:,0]-1e-5)|(q>hard[None,None,:,1]+1e-5)).any((0,2))
        soft_breach=((q<limits[None,None,:,0]-1e-5)|(q>limits[None,None,:,1]+1e-5)).any((0,2))
        qd=np.concatenate([z['post_'+a+'_qd'][:,:,i] for a,i in zip(ARMS,idx)],-1)
        velocity_ratio=(np.abs(qd)/vmax[None,None]).max((0,2))
        coverage[method]=dict(global_joint_span_fraction=actual_union_span.tolist(),mean_per_window_joint_span_fraction=per_window_span.mean(0).tolist())
        outputs[method]=dict(prefix_eligible=prefix_eligible.tolist(),any_raw_geometry_negative=geom_bad.tolist(),
            any_all_arm_force_over_0p1N=force_bad.tolist(),min_raw_gap_m=min_d.min(0).tolist(),peak_all_arm_partner_scalar_N=force.max(0).tolist(),
            controlled_path_rad=paths.tolist(),four_arms_moved=(np.stack(arm_paths)>1e-4).all(0).tolist(),
            secondary_hard_limit_breach=hard_breach.tolist(),secondary_soft_limit_breach=soft_breach.tolist(),
            secondary_max_native_velocity_limit_ratio=velocity_ratio.tolist(),
            initial_velocity_rad_s=actual_velocity.tolist(),native_sensors_env0=oracle['covered_owners_env0'],enabled_collider_owners_env0=oracle['enabled_owners_env0'],
            stream_sha256=sha(root/'response_stream.npz'),point_receipt_sha256=sha(root/'point_contact_receipts.json'))
for k in prefixes['raw']:assert np.array_equal(prefixes['raw'][k],prefixes['multirow'][k]),k
eligible=np.asarray(outputs['raw']['prefix_eligible']);assert np.array_equal(eligible,outputs['multirow']['prefix_eligible'])
summary={}
for m,o in outputs.items():
    g=np.asarray(o['any_raw_geometry_negative']);f=np.asarray(o['any_all_arm_force_over_0p1N']);bad=g|f
    summary[m]=dict(total_states=64,qualified_states=int(eligible.sum()),all_draws_geometry_failed=int(g.sum()),all_draws_force_failed=int(f.sum()),
        qualified_geometry_failed=int((g&eligible).sum()),qualified_force_failed=int((f&eligible).sum()),qualified_either_failed=int((bad&eligible).sum()),
        qualified_peak_scalar_N=float(np.asarray(o['peak_all_arm_partner_scalar_N'])[eligible].max()) if eligible.any() else None,
        mean_qualified_path_rad=float(np.asarray(o['controlled_path_rad'])[eligible].mean()) if eligible.any() else None)
    summary[m]['secondary_qualified_hard_limit_breaches']=int((np.asarray(o['secondary_hard_limit_breach'])&eligible).sum())
    summary[m]['secondary_qualified_soft_limit_breaches']=int((np.asarray(o['secondary_soft_limit_breach'])&eligible).sum())
    summary[m]['secondary_qualified_velocity_limit_exceedances']=int(((np.asarray(o['secondary_max_native_velocity_limit_ratio'])>1.0001)&eligible).sum())
bg=np.asarray(outputs['raw']['any_raw_geometry_negative'])|np.asarray(outputs['raw']['any_all_arm_force_over_0p1N'])
cg=np.asarray(outputs['multirow']['any_raw_geometry_negative'])|np.asarray(outputs['multirow']['any_all_arm_force_over_0p1N'])
result=dict(status='CLOSED_FRESH_GEOMETRY_CONDITIONED_RANDOM_PAIRED_DEVELOPMENT',safety_acceptance=False,formal_holdout=False,
    source_initial_states=64,source_proposals=1920,method_windows=128,seconds_per_window=480*.016666,physics_events_per_method=960,
    common_first_six_applied_q_qd_exact=True,qualified_states=int(eligible.sum()),excluded_initial_prefix_states=int((~eligible).sum()),
    qualification_uses_policy_outcomes=False,qualification_note='Full first six immutable targets and post q/qd match; exclusion based on shared prefix before any new issue actuates.',
    qualified_rescues=np.flatnonzero(eligible&bg&~cg).tolist(),qualified_new_failures=np.flatnonzero(eligible&~bg&cg).tolist(),
    summary=summary,states=outputs,coverage=coverage,bank_joint_span_fraction=bank_span.tolist(),proposal_joint_span_fraction=proposal_span.tolist(),
    selected_proposal_indices=selected.tolist(),bank_eef_positions_local={a:p.tolist() for a,p in bank_eef.items()},
    nonclaims=['complete 26D volume coverage','task objects or grasp','trained full System0 improvement','continuous-time clearance','friction contact bound','hardware safety'])
(H/'FRESH_RANDOM_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
np.savez_compressed(H/'COVERAGE_PLOT_DATA.npz',proposal_joint_span_fraction=proposal_span,bank_joint_span_fraction=bank_span,
    **{a+'_eef':p for a,p in bank_eef.items()},**{m+'_global_joint_span_fraction':np.asarray(c['global_joint_span_fraction']) for m,c in coverage.items()})
print(result['status'],summary,'rescues',result['qualified_rescues'],'new',result['qualified_new_failures'])
