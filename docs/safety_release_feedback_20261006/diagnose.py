"""Read-only phase/physics probe of the sealed v5 execution."""
import json, hashlib
from pathlib import Path
import numpy as np
import torch
from safeduo.delta.task_record_s9 import make_s9_provider
R=Path(__file__).resolve().parent
OLD=Path('/home/liyufeng/safeduo/artifacts/safety_object_binding_20261006')
RAW=Path('/mnt/nas/data/lyf/double_hand/safety_object_binding_20261006/native_v5')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rec=json.loads((RAW/'recording_receipt.json').read_text())
initial=json.loads((RAW/'native_initial.json').read_text())
reg=json.loads((OLD/'REGISTRATION_V5.json').read_text())
arrays={}
for c in rec['chunks']:
    p=RAW/c['file'];assert sha(p)==c['sha256']
    with np.load(p,allow_pickle=False) as z:
        for k in z.files:arrays.setdefault(k,[]).append(z[k])
d={k:np.concatenate(v) for k,v in arrays.items()}
provider=make_s9_provider(1,device='cpu',spheres='r16')
phase=[14.2,15.2,15.5,16.1,17.7,18.7,19.5,20.2]
with np.load(RAW/'bound_reference.npz',allow_pickle=False) as z:
    refs={a:z['q_'+a] for a in ('F_L','F_R','U_L','U_R')}
fk={}
for a,q in refs.items():
    ids=np.round(np.asarray(phase)/reg['trajectory_grid_dt_s']).astype(int)
    pos,yaw=provider.layout.base_pose(a)
    points=provider.kin[a[0]].fk(torch.tensor(q[4,ids]),pos,yaw)['t_flange'].numpy()
    fk[a]=dict(time_s=phase,reference_flange_m=points.tolist(),release_15_2_to_17_7_delta_m=(points[4]-points[1]).tolist())
partners=initial['partners'];fn=np.linalg.norm(d['object_partner_normal'],axis=-1)
rows=[]
for e in (0,3,4,5,6,7):
    for t in phase:
        s=int(np.argmin(abs(d['time'][:,1]-t)))
        arms=('U_L','U_R');oi=1
        hand=[float(fn[s,e,oi,[i for i,p in enumerate(partners) if p['arm']==a and p['hand']]].sum()) for a in arms]
        ti=next(i for i,p in enumerate(partners) if p.get('table')=='TableU')
        rows.append(dict(env=e,time_s=float(d['time'][s,1]),U_speed_m_s=float(np.linalg.norm(d['objects'][s,e,oi,7:10])),U_angular_speed_rad_s=float(np.linalg.norm(d['objects'][s,e,oi,10:13])),U_hand_normal_n=hand,U_table_normal_n=float(fn[s,e,oi,ti]),command_fraction=d['hand_fraction'][s,2:].tolist(),hand_distance_from_first_sample_rad=[float(np.linalg.norm(d[a+':hand_q'][s,e]-d[a+':hand_q'][0,e])) for a in arms]))
out=dict(status='PASS_READ_ONLY_TRACE_LOCALIZATION',source_receipt_sha256=sha(RAW/'recording_receipt.json'),reference_fk=fk,rows=rows,
    observed='Commanded open fingers can remain in contact; contact-free rests can later be recontacted during clearance. Existing table-only normal attribution is not a complete support force.',
    hypotheses=[
      dict(id='H1',claim='Axial release movement starts before the hands have opened',prediction='Hold reference at 15.2 and open before moving; compare fixed dwell and feedback under identical harness',evidence='FK reference displacement and closed fractions at 15.5'),
      dict(id='H2',claim='Opening target does not ensure physical detachment',prediction='Native hand joint error and normal contacts remain nonzero at fraction zero',evidence='Raw e0/e5/e7 after 16.5'),
      dict(id='H3',claim='Later clearance recontacts an already released object',prediction='Detect a new hand normal after a >=0.1s contact-free interval and correlate with object velocity',evidence='e3/e6 at 19.5 after contact-free 16.5..18.7'),
      dict(id='H4',claim='Unfiltered fixture support makes table-only weight balance incomplete',prediction='Compare all-partner matrix sum to native net normal; inventory retained static collision shapes',evidence='e4 final table force ~0.76N vs 1.962N weight'),
      dict(id='H5',claim='Freezing replay time alone still commands motion',prediction='Repeated sample at held phase returns same nonzero delta; require zero command including pending/debt',evidence='SkillReplayDelta.sample computes qref(t+dt)-qref(t) without advancing')],
    failure_contract=dict(expected='Do not begin release/clearance while physically engaged or unstable; do not call a stopped/aborted task successful',scope='isolated task-layer candidate; unchanged safety checkpoint, physics, old task gates',clock_domains='native control .016666, physics .008333, trajectory 1/60; decisions use previous complete native state, never future state'),
    root_cause_proven=False,new_final_trials=0)
(R/'DIAGNOSIS.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(status=out['status'],reference_fk=fk,rows=len(rows)),indent=2))
