"""Finite GPU1 initial-state qualification, independent of policy outcomes."""
from pathlib import Path
import json,os,hashlib,subprocess,datetime,sys
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,r):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(r,indent=2)+'\n');q.replace(p)
if __name__=='__main__':
 p=H/'bank_qualification_plan.json';plan=json.loads(p.read_text());receipt=H/'bank_qualification_execution.json';assert not receipt.exists();r=dict(status='running',started_utc=now(),plan_sha256=sha(p));write(receipt,r)
 try:
  for source,digest in plan['sources'].items():assert sha(source)==digest,source
  gate=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True);r['resource_gate']=dict(actual_exit=gate.returncode,stdout=gate.stdout,stderr=gate.stderr,required_free_MiB=25600,selected_gpu=1,utc=now());write(receipt,r)
  assert gate.returncode==0 and int(gate.stdout.splitlines()[1].split(',')[1])>=25600,'GPU1 below registered gate; no child'
  out=Path(plan['out']);assert not out.exists();r.update(argv=plan['argv'],env=plan['env'],out=str(out))
  with (H/'bank_qualification.log').open('xb') as log:
   c=subprocess.Popen(plan['argv'],cwd=plan['cwd'],env={**os.environ,**plan['env']},stdout=log,stderr=subprocess.STDOUT);r['pid']=c.pid;r['start_ticks']=Path('/proc/'+str(c.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19];write(receipt,r);r['actual_exit']=c.wait();r['child_closed_utc']=now();r['status']='child_reaped';write(receipt,r)
  r['log_sha256']=sha(H/'bank_qualification.log');write(receipt,r);assert r['actual_exit']==0,'bank child nonzero'
  for source,digest in plan['sources'].items():assert sha(source)==digest,source
  summary=json.loads((out/'sampling_summary.json').read_text())
  for seed in plan['seeds']:
   b=out/str(seed)/'bank.npz';m=json.loads(b.with_name('metadata.json').read_text());assert b.is_file() and m['status']=='complete' and m['selected_count']==64 and not m['policy_outcomes_used'];assert m['bank_sha256']==sha(b)
  r['sampling_summary']=summary;r['status']='complete';write(receipt,r);print('BANK_QUALIFICATION_COMPLETE',plan['seeds'],flush=True)
 except BaseException as e:r.update(status='failed',error=type(e).__name__+': '+str(e));raise
 finally:r['closed_utc']=now();write(receipt,r)
