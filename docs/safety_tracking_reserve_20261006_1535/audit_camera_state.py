"""Supplementary exact native velocity/target/FIFO binding for every image group."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def main():
    visual=json.loads((HERE/'visual_verification.json').read_text())
    root=Path(json.loads((HERE/'visual_plan.json').read_text())['actual_visual_root'])
    rows=[];groups=0;minimum=np.inf
    for mode in ('joint_reference','delay_reserve'):
        path=root/mode;file=path/'cell_001.npz';digest=sha(file);data=load(file);native_bindings=0
        for g in (v for v in visual['galleries'] if v['mode']==mode):
            state_path=path/g['state'];assert sha(state_path)==g['state_sha256']
            s=json.loads(state_path.read_text());native=load(path/g['before']);t,e=s['step'],s['env_id']
            for arm in ('F_L','F_R','U_L','U_R'):
                indices=s['fresh_native_controlled_joint_indices'][arm]
                q=native[arm+'_q'];qd=native[arm+'_qd']
                assert np.array_equal(q[e,indices],np.array(s['arms'][arm]['q'],dtype=q.dtype))
                assert np.array_equal(qd[e,indices],np.array(s['arms'][arm]['qd'],dtype=qd.dtype))
                for field in ('q','qd','root','root_vel'):
                    observed=native[arm+'_'+field]
                    assert np.array_equal(observed[e],np.array(s['fresh_native_selected'][arm][field],dtype=observed.dtype))
                    native_bindings+=1
            cat=lambda key:np.array([x for a in ('F_L','F_R','U_L','U_R') for x in s[key][a]],dtype=np.float32)
            assert np.array_equal(cat('controller_target'),data['controller_target'][t,e])
            assert np.array_equal(cat('actuator_target'),data['actuator_target'][t,e])
            pending=np.array([[x for a in ('F_L','F_R','U_L','U_R') for x in target[a]]
                             for target in s['pending_actuator_targets']],dtype=np.float32)
            expected=np.stack([data['q_initial'][e] if t-5+i<0 else data['controller_target'][t-5+i,e] for i in range(6)])
            assert np.array_equal(pending,expected)
            if t<959:
                qd=np.array([x for a in ('F_L','F_R','U_L','U_R') for x in s['arms'][a]['qd']],np.float32)
                assert np.array_equal(qd,data['pre_qd_compact'][t+1,e])
                assert np.array_equal(pending,data['pre_pending_actuator_targets'][t+1,:,e])
            minimum=min(minimum,min(v['observed_sphere_frustum']['minimum_plane_margin_m'] for v in s['views'].values()))
            groups+=1
        assert sha(file)==digest
        rows.append(dict(mode=mode,cell_sha256=digest,native_selected_exact_bindings=native_bindings,
             controller_applied_and_pending_targets_exact=True,native_controlled_q_and_qd_exact=True,
             captured_post_velocity_matches_next_pre_velocity_when_available=True))
    assert groups==visual['groups'] and minimum>0
    with (HERE/'CAMERA_STATE_AUDIT.json').open('x') as f:
        json.dump(dict(status='PASS_EXACT_NATIVE_Q_QD_TARGET_AND_FIFO_BINDING',groups=groups,rows=rows,
            minimum_actual_sphere_plane_margin_m=float(minimum),utc=datetime.now(timezone.utc).isoformat(),
            scope='each camera run own state; supplements sixplane/nativebeforeafter audit; no numericforward or occlusion certificate'),f,indent=2);f.write('\n')
    print('CAMERA_NATIVE_QD_AND_QUEUE_EXACT',groups,minimum,flush=True)
if __name__=='__main__':main()
