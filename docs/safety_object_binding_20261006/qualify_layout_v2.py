"""Development search over declared tabletop centers, no physical outcomes."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from object_binding_v3 import bind_reference,pack_task_input
R=Path(__file__).resolve().parent
def main():
    old=json.loads((R/'REGISTRATION_V4.json').read_text());limits=json.loads((R/'NATIVE_LIMITS.json').read_text());qlimits={a:v['soft_limits_rad'] for a,v in limits['arms'].items()}
    candidates=[.50,.47,.44,.41,.38]
    registration=dict(candidates_F_center_x_m=candidates,U_center_unchanged_m=[-.4,0,.84],XY_variation_m=.02,yaw_variation_deg=8,
        error_limit_m=.001,orientation_error_limit_rad=.01,scope='development IK admission only; solver rejection is not proof of globally unreachable poses; no physical outcomes or old heldout tasks used',
        module_sha256=hashlib.sha256((R/'object_binding_v3.py').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (R/'POSE_SEARCH_REGISTRATION_V2.json').write_text(json.dumps(registration,indent=2)+'\n')
    with np.load(old['trajectory'],allow_pickle=False) as z:payload={k:z[k] for k in z.files}
    q={k[2:]:v for k,v in payload.items() if k.startswith('q_')};dt=float(payload['dt']);sizes=torch.tensor(old['sizes_m']);goals=torch.tensor(old['goals_local_m']);origins=torch.zeros(8,3);cases=[]
    selected=None
    for x in candidates:
        centers=[[x,0,.84],[-.4,0,.84]]
        initial=torch.zeros(1,2,13);initial[0,:,:3]=torch.tensor(centers);initial[0,0,3]=1;initial[0,1,3]=initial[0,1,6]=2**-.5
        packet=pack_task_input(initial,torch.zeros(1,3),sizes,goals,0,0,torch.ones(1,2,dtype=torch.bool),.016666)
        try:
            base,base_audit=bind_reference(q,dt,packet,torch.tensor([True]),old['nominal_centers'],old['nominal_yaws_rad'],joint_limits=qlimits)
            base={a:v[0] for a,v in base.items()};s=torch.zeros(8,2,13)
            for e,c in enumerate(old['cases']):
                for oi,o in enumerate(['beam700','beam300']):
                    par=c['objects'][o];s[e,oi,:3]=torch.tensor(centers[oi])+torch.tensor([par['dx'],par['dy'],0]);angle=old['nominal_yaws_rad'][oi]+np.deg2rad(par['yaw_delta_deg']);s[e,oi,3]=np.cos(angle/2);s[e,oi,6]=np.sin(angle/2)
            packet=pack_task_input(s,origins,sizes,goals,0,0,torch.ones(8,2,dtype=torch.bool),.016666)
            bound,audit=bind_reference(base,dt,packet,torch.tensor([c['binding'] for c in old['cases']]),centers,old['nominal_yaws_rad'],joint_limits=qlimits)
            for a,v in bound.items():
                lim=np.asarray(qlimits[a]);assert (v>=lim[:,0]-1e-5).all() and (v<=lim[:,1]+1e-5).all()
                assert np.abs(np.diff(v,axis=1)).max()<=.06
                assert np.array_equal(v[0],v[4])
            selected=dict(centers=centers,base=base,base_audit=base_audit,audit=audit)
            cases.append(dict(x_m=x,status='PASS_LIMITED_IK_ADMISSION',maximum_position_error_m=max(v['max_position_error_m'] for v in audit),maximum_joint_step_rad=max(float(np.abs(np.diff(v,axis=1)).max()) for v in bound.values())))
            # First passing center is selected by a declared ordering, not task success.
            break
        except (ValueError,AssertionError) as exc:
            cases.append(dict(x_m=x,status='REJECTED_BY_THIS_IK_SOLVER',reason=str(exc)));print('LAYOUT_REJECT',x,str(exc),flush=True)
    result=dict(status='PASS_DEV_IK_LAYOUT_SELECTION' if selected else 'FAIL_NO_LAYOUT_SELECTED',cases=cases,new_physics_trials=0,registration_sha256=hashlib.sha256((R/'POSE_SEARCH_REGISTRATION_V2.json').read_bytes()).hexdigest())
    if selected:
        meta=json.loads(str(payload['meta']));task=meta['s9_task'];task['object']['init_pos']=selected['centers'][0]
        for oi,obj in enumerate(task['objects']):obj['init_pos']=selected['centers'][oi]
        meta['object_binding_development_layout']={'source_trajectory_sha256':hashlib.sha256(Path(old['trajectory']).read_bytes()).hexdigest(),'centers':selected['centers'],'selection':'first IK-admitted declared center, no physical outcome selection'}
        payload['meta']=np.array(json.dumps(meta));payload.update({'q_'+a:v for a,v in selected['base'].items()})
        dest=R/'task_qualified_two_pair.npz';np.savez_compressed(dest,**payload)
        result.update(selected_centers=selected['centers'],qualified_trajectory_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),base_audit=selected['base_audit'],candidate_audit=selected['audit'])
    (R/'LAYOUT_QUALIFICATION_V2.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('base_audit','candidate_audit')},indent=2));assert selected is not None
if __name__=='__main__':main()
