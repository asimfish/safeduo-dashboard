"""Preserved GPU0 camera retry after original reference child closes."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,time
from camera_launch_contract import require_supported_camera_device
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def verify(plan,job):
    for p,d in plan['sources'].items():assert sha(p)==d,p
    for p,d in plan['original_source_sha256'].items():assert sha(Path(plan['cwd'])/p)==d,p
    assert sha(job['argv'][job['argv'].index('--ckpt')+1])==plan['actor_sha256']
def main():
    plan_path=HERE/'visual_retry_plan.json';plan=json.loads(plan_path.read_text());digest=sha(plan_path)
    registration=json.loads((HERE/'CAMERA_MITIGATION_REGISTRATION.json').read_text())
    assert all(sha(p)==d for p,d in registration['sources'].items())
    while True:
        old=json.loads((HERE/'visual_execution.json').read_text())
        reference=next(j for j in old['jobs'] if j['mode']=='joint_reference')
        if reference['status']=='complete' and not (Path('/proc')/str(reference['pid'])).exists():break
        time.sleep(20)
    assert sha(plan_path)==digest
    job=plan['jobs'][0];require_supported_camera_device(job['argv']);verify(plan,job)
    out=Path(job['argv'][job['argv'].index('--out')+1]);assert not out.exists()
    target=HERE/'visual_retry_execution.json';assert not target.exists()
    receipt=dict(status='running',plan_sha256=digest,started_utc=datetime.now(timezone.utc).isoformat(),jobs=[])
    write(target,receipt)
    row=dict(mode=job['mode'],status='running',argv=job['argv'],out=str(out));receipt['jobs'].append(row)
    with (HERE/'visual_retry_motion_admission.log').open('x') as log:
        child=subprocess.Popen(job['argv'],cwd=plan['cwd'],env={**os.environ,**job['env']},stdout=log,stderr=subprocess.STDOUT)
        row['pid']=child.pid;write(target,receipt);rc=child.wait()
    verify(plan,job)
    protocol_path=out/'visual_protocol.json';protocol=json.loads(protocol_path.read_text()) if protocol_path.exists() else None
    complete=rc==0 and protocol is not None and protocol['status']=='complete' and protocol.get('completed_windows')==64
    row.update(exit_code=rc,status='complete' if complete else 'failed')
    if protocol is not None:row['protocol_sha256']=sha(protocol_path)
    else:row['error']='missing protocol, no completed camera window'
    receipt.update(status='complete' if complete else 'failed',finished_utc=datetime.now(timezone.utc).isoformat());write(target,receipt)
    assert complete,'retry failed, retained, not counted complete'
    old=json.loads((HERE/'visual_execution.json').read_text());reference=next(j for j in old['jobs'] if j['mode']=='joint_reference')
    effective=dict(status='complete',jobs=[reference,row],source_execution_receipts=['visual_execution.json','visual_retry_execution.json'],
        original_cuda1_attempt='aborted before scene/window/PNG, preserved in CAMERA_DEVICE_TRACE and originalexecution',
        completed_windows=128,additional_unique_initial_states=0,additional_unique_input_tapes=0,utc=datetime.now(timezone.utc).isoformat())
    write(HERE/'VISUAL_EFFECTIVE_EXECUTION.json',effective)
    print('CAMERA_EFFECTIVE_TWO_COMPLETE_WINDOWSETS',flush=True)
if __name__=='__main__':main()
