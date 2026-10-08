"""One-physics-step actuator diagnostic on raw native contact-free observations."""
import json,hashlib
from pathlib import Path
import numpy as np
from nominal_response import advance
from constrained_response import advance_constrained
H=Path(__file__).resolve().parent
reg=json.loads((H/'REGISTRATION_V2.json').read_text());R=Path(reg['source'])
HF=Path('/home/liyufeng/safeduo/artifacts/safety_feasible_response_20261009_0316')
ARMS=('F_L','F_R','U_L','U_R')
qualified=np.asarray(json.loads((HF/'FRESH_RANDOM_RESULT.json').read_text())['states']['raw']['prefix_eligible'])
with np.load(R/'response_stream.npz') as z:
    force=z['scalar_normal_max_N'];free=(force==0)&qualified[None]
    free[1:] &= force[:-1]==0
with np.load(R/'resolved_native_parameters.npz') as z:p={k:z[k] for k in z.files}
with np.load(R/'dynamics_stream.npz') as z:
    fields=[f'{a}_{f}' for a in ARMS for f in ['q','qd','actual_position_target','mass_matrix','gravity','coriolis']]
    native={k:z[k] for k in fields}
first={a:{f:[] for f in ['q','qd','target']} for a in ARMS}
receipt=json.loads((R/'point_contact_receipts.json').read_text())
for chunk in receipt['chunks']:
    with np.load(R/chunk['path']) as z:
        keep=z['substep']==0
        for a in ARMS:
            for f,key in [('q','native_q'),('qd','native_qd'),('target','native_position_targets')]:
                first[a][f].append(z[a+'_'+key][keep])
for a in ARMS:
    for f in first[a]:first[a][f]=np.concatenate(first[a][f])
    assert np.array_equal(first[a]['target'],native[a+'_actual_position_target'])
    hard=p[a+'_hard_limits'];vmax=p[a+'_max_velocity']
    for q in [native[a+'_q'],first[a]['q']]:
        cols=p[a+'_controlled_joint_indices']
        free &= ((q>=hard[None,...,0]-.001)&(q<=hard[None,...,1]+.001)).all(-1)
        free &= ((q[:,:,cols]>hard[None,:,cols,0]+.001)&(q[:,:,cols]<hard[None,:,cols,1]-.001)).all(-1)
    for v in [native[a+'_qd'],first[a]['qd']]:free &= (np.abs(v)<.95*vmax[None]).all(-1)
origins=np.argwhere(free);rng=np.random.default_rng(reg['seed'])
origins=origins[rng.choice(len(origins),min(len(origins),reg['sample_cap']),replace=False)]
assert len(origins)>0
t,l=origins[:,0],origins[:,1];results={};arrays={'sampled_macro_lane':origins}
for a in ARMS:
    q=native[a+'_q'][t,l].astype(float);v=native[a+'_qd'][t,l].astype(float);target=native[a+'_actual_position_target'][t,l].astype(float)
    mass=native[a+'_mass_matrix'][t,l].astype(float);diag=np.arange(q.shape[-1]);mass[:,diag,diag]+=p[a+'_armature'][l]
    kp=p[a+'_stiffness'][l].astype(float);kd=p[a+'_damping'][l].astype(float);eff=p[a+'_max_force'][l].astype(float)
    bias=native[a+'_coriolis'][t,l]+np.where(p[a+'_disable_gravity_cfg'][l,None],0,native[a+'_gravity'][t,l])
    dt=reg['physics_dt_s'];system=mass.copy();system[:,diag,diag]+=dt*kd+dt*dt*kp
    rhs=np.einsum('nij,nj->ni',mass,v)+dt*(kp*(target-q)-bias)
    u=np.linalg.solve(system,rhs[...,None])[...,0]
    qc,vc,info=advance(q,v,target,mass,kp,kd,eff,p[a+'_max_velocity'][l],bias,dt)
    qs,vs,stop_info=advance_constrained(q,v,target,mass,kp,kd,eff,p[a+'_max_velocity'][l],bias,p[a+'_hard_limits'][l],dt)
    aq=first[a]['q'][t,l];av=first[a]['qd'][t,l]
    results[a]={}
    for name,pq,pv in [('unclipped',q+dt*u,u),('clipped',qc,vc),('joint_stops',qs,vs)]:
        qe=np.abs(pq-aq);ve=np.abs(pv-av);cols=p[a+'_controlled_joint_indices'];hand=np.setdiff1d(diag,cols)
        results[a][name]=dict(observations=len(t),full_dof_q_max_rad=float(qe.max()),full_dof_qd_max_rad_s=float(ve.max()),
            controlled_q_max_rad=float(qe[:,cols].max()),controlled_qd_max_rad_s=float(ve[:,cols].max()),
            controlled_qd_mean_rad_s=float(ve[:,cols].mean()),hand_qd_max_rad_s=float(ve[:,hand].max()),
            local_descriptive_gate=bool(qe.max()<=.0001 and ve.max()<=.02),safety_certificate=False)
        arrays[a+'_'+name+'_q_error']=qe;arrays[a+'_'+name+'_qd_error']=ve
    results[a]['clipped'].update(converged=int(info['converged'].sum()),predicted_velocity_limit_clips=int(info['velocity_clipped'].any(-1).sum()))
results[a]['joint_stops'].update(converged=int(stop_info['converged'].sum()),joint_stop_active_samples=int((stop_info['lower_active']|stop_info['upper_active']).any(-1).sum()))
np.savez_compressed(H/'CALIBRATION_ERRORS_V2.npz',**arrays)
result=dict(status='CLOSED_ALL_BODY_CONTACT_FREE_ACTUATOR_CALIBRATION',eligible_origin_observations=int(free.sum()),sampled_origins=len(t),
    source_qualified_windows=int(qualified.sum()),physics_horizon=1,no_future_contact_leak_into_model_inputs=True,
    observational_selection_note='Observed current/past force and firstmicro limit status select calibration observations only, not an online admission rule.',
    all_native_targets_matched_exactly=True,results=results,new_native_physics_trials=0,safety_certificate=False,
    cannot_validate=['FIFO stopping prediction','nonlinear future clearance','contact or manipulation dynamics','continuous-time safety'],
    source_hashes={name:hashlib.sha256((R/name).read_bytes()).hexdigest() for name in ['response_stream.npz','dynamics_stream.npz','resolved_native_parameters.npz','point_contact_receipts.json']})
(H/'CALIBRATION_RESULT_V2.json').write_text(json.dumps(result,indent=2)+'\n');print(result['status'],len(t),results)
