"""Preserve the failed pre-window attempt and evaluate registered device mitigation."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(n):return json.loads((HERE/n).read_text())
def main():
    a=argparse.ArgumentParser();a.add_argument('--original-outer-exit',required=True,type=int);args=a.parse_args();assert args.original_outer_exit==1
    trace=load('CAMERA_DEVICE_TRACE.json');original=load('visual_execution.json');bad=next(j for j in original['jobs'] if j['mode']=='motion_admission');ref=next(j for j in original['jobs'] if j['mode']=='joint_reference')
    assert bad['pid']==trace['pid'] and not (Path('/proc')/str(bad['pid'])).exists()
    assert ref['status']=='complete' and not (Path('/proc')/str(ref['pid'])).exists()
    out=Path(bad['out']);assert not (out/'visual_protocol.json').exists() and not (out/'cell_001.npz').exists() and not list(out.rglob('*.png'))
    closure=dict(status='PASS_RETAINED_FAILED_PRE_WINDOW_CAMERA_ATTEMPT',failed_child_pid=bad['pid'],failed_child_stop='SIGTERM did not exit; later exact-owned SIGKILL confirmedprocessgone',
       original_supervisor_exit=args.original_outer_exit,original_manifest_status=original['status'],original_manifest_not_rewritten=True,
       failure='motion logicalcuda1 scene initialization stalled; original supervisor then raised FileNotFoundError for absent protocol',
       failed_scored_frames=0,failed_camera_windows=0,failed_images=0,reference_kept_complete=True,
       input_sha256={n:sha(HERE/n) for n in ['visual_execution.json','CAMERA_DEVICE_TRACE.json','CAMERA_FAILED_PROCESS_EXIT.json','visual_motion_admission.log','camera_parallel_v3_driver.log']},
       retained_output_files=[str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()],utc=datetime.now(timezone.utc).isoformat())
    with (HERE/'CAMERA_ORIGINAL_ATTEMPT_CLOSURE.json').open('x') as f:json.dump(closure,f,indent=2);f.write('\n')
    effective=load('VISUAL_EFFECTIVE_EXECUTION.json');retry=load('visual_retry_execution.json');assert effective['status']==retry['status']=='complete'
    good=retry['jobs'][0];assert good['mode']=='motion_admission' and good['status']=='complete'
    oldargs=bad['argv'];newargs=good['argv'];assert len(oldargs)==len(newargs)
    allowed={oldargs.index('--device')+1,oldargs.index('--out')+1};differences=[i for i,(x,y) in enumerate(zip(oldargs,newargs)) if x!=y];assert set(differences)==allowed
    assert oldargs[oldargs.index('--device')+1]=='cuda:1' and newargs[newargs.index('--device')+1]=='cuda:0'
    for p,d in trace['source_sha256'].items():assert sha(p)==d
    protocol=json.loads((Path(good['out'])/'visual_protocol.json').read_text());assert protocol['status']=='complete' and protocol['completed_windows']==64
    r=dict(status='PASS_REGISTERED_CUDA0_MITIGATION_COMPLETE',original_startup_failure_retained=True,registered_retry_complete_windows=64,
       valid_total_illustrative_camera_windows=128,additional_unique_initial_states=0,additional_unique_input_tapes=0,numeric_primary_windows_unchanged=768,
       source_code_unchanged=True,actual_argv_changes=[dict(index=i,old=oldargs[i],new=newargs[i]) for i in differences],
       causal_scope='same-source device/output counterfactual succeeded; capturedlocal GPU1renderpath startup failure mitigated, not driver/rootcause repair nor allmultiGPU certification',
       evidence_sha256={n:sha(HERE/n) for n in ['CAMERA_ORIGINAL_ATTEMPT_CLOSURE.json','CAMERA_MITIGATION_REGISTRATION.json','VISUAL_EFFECTIVE_EXECUTION.json','VISUAL_EFFECTIVE_PLAN.json','visual_retry_execution.json']},utc=datetime.now(timezone.utc).isoformat())
    with (HERE/'CAMERA_MITIGATION_RESULT.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(r['status'],flush=True)
if __name__=='__main__':main()
