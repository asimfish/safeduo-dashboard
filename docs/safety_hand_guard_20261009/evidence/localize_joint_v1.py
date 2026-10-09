"""Post hoc joint localization from original native microsteps; never changes scores."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

P=Path(__file__).resolve().parent
H=P.parent
FIX=H/'fixedhand_triplet64_v1'
REG=json.loads((H/'FIXEDHAND_TRIPLET_DEV_REG_V1.json').read_text())
ARMS=('F_L','F_R','U_L','U_R')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(method):
    assert method in REG['methods']
    out=P/f'JOINT_LOCALIZATION_{method}_V1.json'
    assert not out.exists(),'Preserve first post hoc attempt'
    leaf=Path(REG['native_namespace'])/f'paired_batch0_{method}_v8'
    oraclepath=FIX/f'PAIRED_ORACLE_batch0_{method}_V1.json'
    oracle=json.loads(oraclepath.read_text());assert oracle['status'].startswith('PASS_')
    with np.load(leaf/'resolved_native_parameters.npz') as z:params={k:z[k] for k in z.files}
    receipt=json.loads((leaf/'point_contact_receipts.json').read_text())
    maxima={a:{k:np.zeros((64,len(params[a+'_joint_names']))) for k in ['hard','speed']} for a in ARMS}
    peaks={a:{} for a in ARMS}
    events=[]
    for chunk in receipt['chunks']:
        source=leaf/chunk['path'];assert sha(source)==chunk['sha256']
        with np.load(source) as z:
            frames,subs=z['frame'],z['substep'];events.extend(zip(frames.tolist(),subs.tolist()))
            for arm in ARMS:
                q,qd=z[arm+'_native_q'],z[arm+'_native_qd'];bounds=params[arm+'_hard_limits'];vmax=params[arm+'_max_velocity']
                hard=np.maximum(np.maximum(bounds[None,...,0]-q,q-bounds[None,...,1]),0)
                speed=np.maximum(abs(qd)-vmax[None],0)
                for kind,value,native in [('hard',hard,q),('speed',speed,qd)]:
                    assert np.isfinite(value).all()
                    maxima[arm][kind]=np.maximum(maxima[arm][kind],value.max(0))
                    event,lane,joint=np.unravel_index(value.argmax(),value.shape);peak=float(value[event,lane,joint])
                    if kind not in peaks[arm] or peak>peaks[arm][kind]['exceedance']:
                        peaks[arm][kind]=dict(exceedance=peak,macro=int(frames[event]),substep=int(subs[event]),input_id=int(params['global_input_id'][lane]),joint=str(params[arm+'_joint_names'][joint]),controlled_arm_joint=joint in params[arm+'_controlled_joint_indices'],native_value=float(native[event,lane,joint]),native_hard_interval_rad=bounds[lane,joint].tolist(),native_max_velocity_rad_s=float(vmax[lane,joint]))
    assert events==[(frame,sub) for frame in range(480) for sub in range(2)]
    for kind,key in [('hard','supplementary_max_all74_hard_violation_rad'),('speed','supplementary_max_all74_velocity_exceedance_rad_s')]:
        full=np.max(np.stack([maxima[a][kind].max(-1) for a in ARMS]),0)
        assert np.array_equal(full,[r[key] for r in oracle['states']]),'Independent peak must match unchanged original oracle'
    rows=[]
    for arm in ARMS:
        idx=params[arm+'_controlled_joint_indices'];hand=np.array([j for j in range(len(params[arm+'_joint_names'])) if j not in idx])
        rows.append(dict(arm=arm,peaks=peaks[arm],controlled_hard_bad_lanes=int((maxima[arm]['hard'][:,idx].max(-1)>1e-5).sum()),hand_hard_bad_lanes=int((maxima[arm]['hard'][:,hand].max(-1)>1e-5).sum()),per_joint=[dict(name=str(name),controlled=j in idx,hard_bad_lanes=int((maxima[arm]['hard'][:,j]>1e-5).sum()),speed_bad_lanes=int((maxima[arm]['speed'][:,j]>1e-5).sum()),max_hard_breach_rad=float(maxima[arm]['hard'][:,j].max()),max_speed_exceedance_rad_s=float(maxima[arm]['speed'][:,j].max())) for j,name in enumerate(params[arm+'_joint_names'])]))
    result=dict(status='PASS_NEW64_ALL960_POSTHOC_JOINT_LOCALIZATION',method=method,arms=rows,source_oracle_sha256=sha(oraclepath),source_point_receipt_sha256=sha(leaf/'point_contact_receipts.json'),source_sha256=sha(Path(__file__)),original_scores_unchanged=True,safety_acceptance=False)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['status'],method,[(r['arm'],r['hand_hard_bad_lanes'],r['controlled_hard_bad_lanes']) for r in rows],flush=True)


if __name__=='__main__':main(sys.argv[1])
