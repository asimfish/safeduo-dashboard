"""Standalone scientific plots; every native frame retained, no smoothing."""
import json,hashlib,time,argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent;raw=Path('/mnt/nas/data/lyf/double_hand/safety_scene_geometry_gate_20261008/matched_scene');sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest();pa=argparse.ArgumentParser();pa.add_argument('--wait',action='store_true');a=pa.parse_args();deadline=time.monotonic()+900
while a.wait and not (raw/'recording_receipt.json').exists():
 assert not (raw/'failure.txt').exists() and time.monotonic()<deadline;time.sleep(20)
receipt=json.loads((raw/'recording_receipt.json').read_text());reg=json.loads((p/'REGISTRATION_MATCHED_SCENE.json').read_text());traces={};steps=[]
for ch in receipt['chunks'][:6]:
 assert sha(raw/ch['file'])==ch['sha256']
 with np.load(raw/ch['file']) as z:
  steps.extend(z['step'].tolist())
  for env in range(8):
   for arm in ['U_L','U_R']:traces.setdefault((env,arm),[]).extend(z[f'e{env}:{arm}:hand_partner_scalar_normal'].max(axis=(1,2)).tolist())
assert steps==list(range(720));x=(np.array(steps)+1)*reg['physics_dt_s'];fig,axs=plt.subplots(4,2,figsize=(11,10),sharex=True,layout='constrained')
for i in range(4):
 for j,arm in enumerate(['U_L','U_R']):
  ax=axs[i,j]
  for env,method,color in [(2*i,'V3','#b53636'),(2*i+1,'Scene admission','#17689a')]:ax.plot(x,traces[env,arm],label=method,color=color,linewidth=1.2)
  ax.axhline(.1,linestyle='--',color='#555555',linewidth=.8,label='0.1 N criterion');ax.set_ylim(bottom=-.02);ax.set_xlim(0,6);ax.set_title(f"Reference {reg['cases'][2*i]['reference_time_s']:.4f} s / {arm}");ax.set_ylabel('Normal (N)');ax.grid(alpha=.2)
  if i==3:ax.set_xlabel('Completed native physics time (s)')
  if i==0 and j==0:ax.legend(fontsize=8)
fig.suptitle('Matched initial states: first frozen hand goal, full six-second path\nMaximum absolute point-sum per sensor/partner; two U-hand closures\nStatic four-arm configurations; no manipulation acceptance',fontsize=12)
out=p/'figures';out.mkdir(exist_ok=False);files=[]
for ext in ['png','svg','pdf']:
 f=out/('matched_contact_first_goal.'+ext);fig.savefig(f,dpi=200);files.append(dict(file=f.name,sha256=sha(f),bytes=f.stat().st_size))
plt.close(fig);(p/'FIGURE_RECEIPT.json').write_text(json.dumps(dict(status='PASS_ALL_720_NATIVE_FRAMES_PER_EIGHT_MATCHED_PAIRS',source_receipt_sha256=sha(raw/'recording_receipt.json'),source_registration_sha256=sha(p/'REGISTRATION_MATCHED_SCENE.json'),source_chunks=receipt['chunks'][:6],figure_pairs=8,frames_per_trace=720,smoothing=False,new_physics_trials=0,files=files),indent=2)+'\n');print(files)
