"""Exploratory comparison, explicitly separate from the predeclared scorer."""
import json,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;RAW=Path('/mnt/nas/data/lyf/double_hand/safety_release_feedback_20261006')
result=json.loads((R/'RESULTS.json').read_text());reg=json.loads((R/'REGISTRATION.json').read_text());pairs=[];rest=[]
for b in range(2):
    folder=RAW/f'block_{b}';receipt=json.loads((folder/'recording_receipt.json').read_text());init=json.loads((folder/'native_initial.json').read_text());d={}
    for chunk in receipt['chunks']:
        with np.load(folder/chunk['file']) as z:
            for k in z.files:d.setdefault(k,[]).append(z[k])
    d={k:np.concatenate(v) for k,v in d.items()};orig=np.asarray(init['origins']);s=d['objects'].astype(np.float64);s[...,:3]-=orig[None,:,None,:]
    before=d['time'][:,1]<15.2;preindex=int(np.flatnonzero(before)[-1])
    with np.load(folder/'bound_reference.npz') as z:
        for layout in range(4):
            baseline=next(c['env'] for c in init['cases'] if c['layout']==layout and c['method']=='scheduled_debt')
            for method in ('stationary_release','clearance_feedback'):
                e=next(c['env'] for c in init['cases'] if c['layout']==layout and c['method']==method)
                pairs.append(dict(block=b,layout=layout,baseline_env=baseline,other_env=e,other_method=method,
                    reference_bitexact=all(np.array_equal(z['q_'+a][e],z['q_'+a][baseline]) for a in ('F_L','F_R','U_L','U_R')),
                    max_object_position_difference_before15_2_m=float(np.linalg.norm(s[before,e,:,:3]-s[before,baseline,:,:3],axis=-1).max()),
                    object_position_difference_at15_2_m=np.linalg.norm(s[preindex,e,:,:3]-s[preindex,baseline,:,:3],axis=-1).tolist(),
                    max_arm_q_difference_before15_2_rad=max(float(np.abs(d[a+':q'][before,e]-d[a+':q'][before,baseline]).max()) for a in ('F_L','F_R','U_L','U_R'))))
    ti=[next(j for j,p in enumerate(init['partners']) if p.get('table')==t) for t in ('TableF','TableU')]
    for e,c in enumerate(init['cases']):
        for i,o in enumerate(('beam700','beam300')):
            hids=[j for j,p in enumerate(init['partners']) if p['arm'] in ('F_L','F_R','U_L','U_R')[2*i:2*i+2] and p['hand']]
            contact=np.linalg.norm(d['object_partner_normal'][:,e,i,hids],axis=-1).sum(-1)
            speed=np.linalg.norm(s[:,e,i,7:10],axis=-1)
            mask=(d['time'][:,1]>22)&(contact<.1)&(speed<.01)
            rest.append(dict(block=b,env=e,method=c['method'],object=o,contact_free_slow_terminal_states=int(mask.sum()),
                net_z_median_n=float(np.median(d['object_net_contact'][mask,e,i,2])) if mask.any() else None,
                filtered_z_median_n=float(np.median(d['object_partner_normal'][mask,e,i,:,2].sum(-1))) if mask.any() else None,
                table_z_median_n=float(np.median(d['object_partner_normal'][mask,e,i,ti[i],2])) if mask.any() else None))
out=dict(scope='Post-run exploratory diagnostics only; does not change original scorer or gates',paired_layout_contrasts=pairs,resting_force_attribution=rest,
    reference_identity_is_not_physics_identity=True,causal_effect_accepted=False,
    explanation='Nominal/reference identities alone do not establish matched physical pickup or placement states. Report pre-intervention differences; seeds and clone slots co-vary. Net-minus-filtered alone is not full-scene/friction calibration.',new_final_trials=0)
(R/'POSTRUN_DIAGNOSIS.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(pairs=len(pairs),max_preintervention_position_difference_m=max(p['max_object_position_difference_before15_2_m'] for p in pairs)),indent=2))
