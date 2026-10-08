"""Actual native .35 pose at reset and full PhysX hand targets every microstep."""
from pathlib import Path
import json,sys,hashlib,numpy as np,datetime
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(root,tag):
 root=Path(root);record=json.loads((root/'open_hand_initialization.json').read_text());assert record['open_thumb_rad']==.35 and record['all_modes_common'];receipts=json.loads((root/'native_contact_receipts.json').read_text());init=np.load(root/'native_initial.npz',allow_pickle=False);peaks={a:0. for a in ['U_L','U_R']};events=0
 for a,name in record['requested_joint_names'].items():
  idx=record['joint_indices'][a];assert str(init[a+'_native_joint_names'][idx])==name
  assert np.array_equal(init[a+'_native_q'][:,idx],np.full(64,np.float32(.35))) and not init[a+'_native_qd'][:,idx].any(),'actual initialized thumb state differs'
 for chunk in receipts['chunks']:
  p=root/chunk['path'];assert sha(p)==chunk['sha256']
  with np.load(p,allow_pickle=False) as z:
   for a in peaks:
    idx=record['joint_indices'][a];goal=z[a+'_native_position_targets'][...,idx];assert np.array_equal(goal,np.full_like(goal,np.float32(.35))),'actual thumb target does not hold calibrated opening'
    error=np.abs(z[a+'_native_q'][...,idx].astype(np.float64)-float(np.float32(.35)));peaks[a]=max(peaks[a],float(error.max()))
   events+=len(z['frame'])
  assert sha(p)==chunk['sha256']
 assert events==2*receipts['control_steps'];out=dict(status='PASS_ACTUAL_CALIBRATED_OPEN_POSE_AND_TARGET_BINDING',root=str(root),physics_events=events,thumb_tracking_error_peak_rad=peaks,tracking_qualified_1mrad=all(v<=.001 for v in peaks.values()),no_runtime_thumb_target_change=True,constructor_contacts_certified=False,hardware_certified=False,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 with (H/(tag+'.json')).open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
