"""Supplement actual camera observer verification with full control/input/FIFO readback."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,types
import numpy as np
HERE=Path(__file__).resolve().parent
RAW=Path('/mnt/nas/data/lyf/double_hand')/HERE.name
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def main():
    plan=json.loads((HERE/'visual_plan_v3.json').read_text());execution=json.loads((HERE/'visual_execution_v2.json').read_text())
    assert execution['status']=='complete' and execution['plan_sha256']==sha(HERE/'visual_plan_v3.json')
    for p,d in plan['sources'].items():assert sha(Path(p))==d,p
    for p,d in plan['original_source_sha256'].items():assert sha(Path(plan['cwd'])/p)==d,p
    actual_actor=Path(plan['jobs'][0]['argv'][plan['jobs'][0]['argv'].index('--ckpt')+1]);assert sha(actual_actor)==plan['actor_sha256']
    source=HERE/'dense_audit.py';expected=json.loads((HERE/'holdout_v2/holdout_0_plan.json').read_text())['research_source_sha256'][str(source)]
    code=source.read_bytes();assert hashlib.sha256(code).hexdigest()==expected
    m=types.ModuleType('dense_audit_camera_gate');m.__file__=str(source);exec(compile(code,str(source),'exec'),m.__dict__)
    runs=[]
    for job in plan['jobs']:
        mode=job['mode'];root=Path(plan['actual_visual_root'])/mode;protocol=json.loads((root/'visual_protocol.json').read_text())
        assert protocol['status']=='complete' and protocol['actor_sha256']==plan['actor_sha256']
        assert protocol['observer_sha256']==plan['sources'][job['argv'][1]]
        seed=job['argv'][job['argv'].index('--seeds')+1]
        original=RAW/'holdout_0'/f'{mode}_{seed}'
        z=load(root/'cell_001.npz');inp=load(root/'input_recipe.npz');numinp=load(original/'input_recipe.npz')
        assert np.array_equal(inp['tape'],numinp['tape']) and np.array_equal(inp['q_initial'],numinp['q_initial'])
        assert np.array_equal(z['external_unscaled_cmd'],inp['tape'][:960]) and np.array_equal(z['cmd'],inp['tape'][:960])
        before={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
        audit=m.forecast_audit(root,z)
        applied=np.concatenate([np.repeat(z['q_initial'][None],6,axis=0),z['controller_target'][:-6]])
        assert np.array_equal(z['actuator_target'],applied)
        rec=json.loads((root/'camera_receipts.json').read_text())
        for cap in rec['receipts']:
            s=json.loads((root/cap['state']).read_text());e=s['env_id'];t=s['step']
            slices={'F_L':slice(0,7),'F_R':slice(7,14),'U_L':slice(14,20),'U_R':slice(20,26)}
            for a,sl in slices.items():
                assert np.array_equal(np.array(s['controller_target'][a],np.float32),z['controller_target'][t,e,sl])
                assert np.array_equal(np.array(s['actuator_target'][a],np.float32),z['actuator_target'][t,e,sl])
                postqueue=np.concatenate([z['pre_pending_actuator_targets'][t,1:,e],z['controller_target'][t,e][None]],axis=0)
                assert np.array_equal(np.array([p[a] for p in s['pending_actuator_targets']],np.float32),postqueue[:,sl])
        assert before=={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
        runs.append(dict(mode=mode,full_control_audit=audit,actual_applied_fifo_exact=True,own_captured_post_fifo_exact=True,own_captured_target_exact=True,original_tape_and_initial_exact=True,input_files=len(before),input_bytes=sum((root/p).stat().st_size for p in before),input_sha256=before))
    receipt=dict(status='PASS_CAMERA_CONTROL_INPUT_FIFO_READBACK',runs=runs,plan_sha256=sha(HERE/'visual_plan_v3.json'),dense_audit_sha256=expected,source_sha256=sha(Path(__file__)),actor_sha256=plan['actor_sha256'],utc=datetime.now(timezone.utc).isoformat(),scope='observer control, full forecast and ownstate queue audit after capture; separate forward cross-run comparison remains in visual_verification.json')
    (HERE/'visual_control_audit.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['status'],flush=True)
if __name__=='__main__':main()
