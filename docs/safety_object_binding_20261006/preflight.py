"""Synthetic planner checks against separately read native limits; no physics."""
import argparse,json,importlib
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--module',default='object_binding_v3');p.add_argument('--output',default='CONSTRAINED_PREFLIGHT.json');a=p.parse_args()
    module=importlib.import_module(a.module)
    reg=json.loads((R/'REGISTRATION_V4.json').read_text());lim=json.loads((R/'NATIVE_LIMITS.json').read_text());z=np.load(reg['trajectory'])
    q={k[2:]:z[k] for k in z.files if k.startswith('q_')};origins=torch.tensor([[x,y,0] for x in [3.5,0,-3.5] for y in [-3.5,0,3.5]])[:8]
    nominal=(torch.tensor(reg['nominal_centers'])[None]+origins[:,None,:])-origins[:,None,:];s=torch.zeros(8,2,13)
    for e,c in enumerate(reg['cases']):
        for oi,o in enumerate(['beam700','beam300']):
            s[e,oi,:3]=torch.tensor(reg['nominal_centers'][oi])+origins[e]+torch.tensor([c['objects'][o]['dx'],c['objects'][o]['dy'],0])
            ang=reg['nominal_yaws_rad'][oi]+np.deg2rad(c['objects'][o]['yaw_delta_deg']);s[e,oi,3]=np.cos(ang/2);s[e,oi,6]=np.sin(ang/2)
    x=module.pack_task_input(s,origins,torch.tensor(reg['sizes_m']),torch.tensor(reg['goals_local_m']),0.,0.,torch.ones(8,2,dtype=torch.bool),.016666)
    kwargs={} if a.module=='object_binding_v2' else {'joint_limits':{arm:v['soft_limits_rad'] for arm,v in lim['arms'].items()}}
    refs,audit=module.bind_reference(q,float(z['dt']),x,torch.tensor([c['binding'] for c in reg['cases']]),reg['nominal_centers'],reg['nominal_yaws_rad'],nominal.numpy(),**kwargs)
    assert all(np.array_equal(refs[arm][0],refs[arm][4]) for arm in q)
    errors={}
    for arm,v in refs.items():
        limits=np.asarray(lim['arms'][arm]['soft_limits_rad']);outside=np.maximum(limits[:,0]-v,v-limits[:,1]);errors[arm]=float(outside.max())
    result=dict(status='PASS' if max(errors.values())<=1e-5 else 'FAIL_NATIVE_LIMIT_CONSTRAINT',not_physics=True,module=a.module,nominal_bitexact=True,max_outside_native_soft_limit_rad=errors,
        maximum_reference_step_rad={arm:float(np.abs(np.diff(v,axis=1)).max()) for arm,v in refs.items()},ik=audit)
    (R/a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='ik'},indent=2));assert result['status']=='PASS','reference outside native limits'
if __name__=='__main__':main()
