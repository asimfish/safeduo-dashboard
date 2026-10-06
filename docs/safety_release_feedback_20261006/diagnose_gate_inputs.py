"""Post-run audit of the previous-step native inputs actually seen by the gate."""
from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parent;RAW=Path('/mnt/nas/data/lyf/double_hand/safety_release_feedback_20261006')
reg=json.loads((R/'REGISTRATION.json').read_text());p=reg['gate'];rows=[]
for b in range(2):
    folder=RAW/f'block_{b}';receipt=json.loads((folder/'recording_receipt.json').read_text());init=json.loads((folder/'native_initial.json').read_text());d={}
    for chunk in receipt['chunks']:
        with np.load(folder/chunk['file']) as z:
            for k in z.files:d.setdefault(k,[]).append(z[k])
    d={k:np.concatenate(v) for k,v in d.items()}
    for c in init['cases']:
        if c['method']!='clearance_feedback':continue
        e=c['env'];waiting=np.flatnonzero(d['gate_stage'][:,e]==1);assert len(waiting)>0 and waiting.min()>0
        x={k.split(':',1)[1]:v[waiting-1,e] for k,v in d.items() if k.startswith('gate_input:')}
        conditions=dict(table_support=(x['table_n']>.1).all(-1),bottom_geometry=(abs(x['bottom_delta_m'])<=.006).all(-1),
            linear_stability=(x['speed_m_s']<=.03).all(-1),angular_stability=(x['angular_rad_s']<=.3).all(-1),
            upright=(x['tilt_deg']<=10).all(-1),contact_free=(x['hand_n']<=.1).reshape(len(waiting),-1).all(-1),
            physically_open=(x['hand_open_max_error_rad']<=.2).all(-1))
        allgood=np.stack(list(conditions.values()),-1).all(-1);run=0;longest=0
        for value in allgood:run=run+1 if value else 0;longest=max(longest,run)
        before=d['time'][:,1]<18.7
        q=d['objects'][:,e,:,3:7].astype(float);q/=np.linalg.norm(q,axis=-1,keepdims=True)
        tilt=np.degrees(np.arccos(np.clip(1-2*(q[:,:,1]**2+q[:,:,2]**2),-1,1)))
        rows.append(dict(block=b,env=e,layout=c['layout'],wait_states=len(waiting),failed_state_counts={k:int((~v).sum()) for k,v in conditions.items()},longest_all_good_run=longest,
            max_wait_input_bottom_abs_m=float(abs(x['bottom_delta_m']).max()),max_wait_input_open_error_rad=float(x['hand_open_max_error_rad'].max()),
            any_tilt80_before_gate=bool((tilt[before]>80).any()),max_tilt_before_gate_deg=float(tilt[before].max()),abort_reason=int(d['gate_reason'][-1,e])))
out=dict(scope='Exploratory, independent replay of actual previous-step gate inputs; no threshold/scorer change',cases=rows,admitted_clearances=0,
    candidate_decision='REJECTED_NO_COMPLETED_TASK_AND_INCOMPLETE_SUPPORT_ORACLE',
    note='All8 time out. A reaction interlock never reaches admitted state, so its physical post-contact recovery path remains untested. Geometry/support and pre-intervention discrepancies prevent a causal reliability claim. Later halt does not repair earlier placement/release failure.',new_final_trials=0)
(R/'GUARD_INPUT_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(rows,indent=2))
