"""Independent complete FIFO6 and actual native position-target binding."""
from pathlib import Path
import hashlib,json,numpy as np,sys,datetime
H=Path(__file__).resolve().parent
ARMS=('F_L','F_R','U_L','U_R')
def bits(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def check(method):
    active=json.loads((H/'ACTIVE_ATTEMPT_V6.json').read_text());root=Path(active['primary_roots'][method]);execution=json.loads(Path(active['primary_executions'][method]).read_text());job=next(j for j in execution['jobs'] if j['out']==str(root))
    assert job['actual_exit']==0 and job['closed_utc']
    if method=='multirow':assert job['status']=='complete' and job.get('resource_abort') is None
    else:assert job['resource_abort']=='owned host RSS exceeds registered bound'
    assert json.loads((root/'response_protocol.json').read_text())['status']=='complete'
    assert not (Path('/proc')/str(job['pid'])).exists() or (Path('/proc')/str(job['pid'])/'stat').read_text().rsplit(')',1)[1].split()[19]!=job['start_ticks']
    with np.load(root/'resolved_native_parameters.npz') as z:idx={a:z[a+'_controlled_joint_indices'] for a in ARMS}
    with np.load(root/'native_initial_all64.npz') as z:initial=np.concatenate([z[a+'_targets'][:,idx[a]] for a in ARMS],-1)
    with np.load(root/'response_stream.npz') as z:
        issued=z['issued_target'];applied=z['applied_target'];pending=z['pre_pending'];unmodified=z['unmodified_pre_pending']
    assert applied.shape==issued.shape==(480,64,26)
    expected=np.concatenate([np.repeat(initial[None],6,axis=0),issued[:-6]],0);assert bits(expected,applied),'entireFIFO6 applied values differ'
    assert bits(pending,unmodified),'pending queue was rewritten before push'
    for step in range(480):
        for slot in range(6):assert bits(pending[step,slot],initial if step+slot<6 else issued[step+slot-6]),('pending mismatch',step,slot)
    receipt=json.loads((root/'point_contact_receipts.json').read_text());events=0
    for c in receipt['chunks']:
        p=root/c['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==c['sha256']
        with np.load(p) as z:
            frames=z['frame'];sub=z['substep'];assert np.array_equal(frames,np.arange(events,events+len(frames))//2);assert np.array_equal(sub,np.arange(events,events+len(frames))%2)
            actual=np.concatenate([z[a+'_native_position_targets'][:,:,idx[a]] for a in ARMS],-1)
            assert bits(actual,applied[frames]),('native target binding',c['path'])
            events+=len(frames)
    assert events==960
    report=dict(status='PASS_COMPLETE_FIFO6_AND_960_NATIVE_TARGET_BINDINGS',method=method,source_root=str(root),controls=480,lanes=64,controlled_joints=26,native_microsteps=events,all_480_applied_targets_exactly_six_steps_delayed=True,all_480_six_pending_slots_exact=True,unmodified_pending_queue_preserved=True,actual_native_controlled_targets_all960_microsteps_bitwise_exact=True,guard_source_sha256=hashlib.sha256((H/'multirow_response_v3.py').read_bytes()).hexdigest(),oracle_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),safety_acceptance=False,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with (H/('FIFO_NATIVE_'+method+'_V6.json')).open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(report['status'],method,flush=True)
if __name__=='__main__':check(sys.argv[1])
