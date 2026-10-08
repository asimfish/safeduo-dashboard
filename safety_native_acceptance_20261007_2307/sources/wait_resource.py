"""Wait for registered GPU0 room; never terminate or modify foreign jobs."""
from pathlib import Path
import json,sys,time,datetime,subprocess,hashlib
H=Path(__file__).resolve().parent
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,r):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(r,indent=2)+'\n');q.replace(p)
if __name__=='__main__':
 p=Path(sys.argv[1]);plan=json.loads(p.read_text());target=H/(plan['tag']+'_resource_wait.json');assert not target.exists()
 r=dict(status='waiting',plan_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),started_utc=now(),required_free_MiB=25600,timeout_s=7200,probes=[]);start=time.monotonic();save(target,r)
 try:
  while True:
   probe=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True);row=dict(utc=now(),actual_exit=probe.returncode,stdout=probe.stdout,stderr=probe.stderr);r['probes'].append(row);save(target,r)
   if probe.returncode==0 and int(probe.stdout.splitlines()[0].split(',')[1])>=25600:break
   if time.monotonic()-start>7200:raise TimeoutError('resource remained below registered minimum; no GPU child launched')
   time.sleep(30)
  r['status']='qualified';save(target,r)
  c=subprocess.Popen([sys.executable,'-B',str(H/'execute_plan.py'),str(p)]);r['executor_pid']=c.pid;save(target,r);rc=c.wait();r['executor_actual_exit']=rc;r['status']='complete' if rc==0 else 'failed';save(target,r);sys.exit(rc)
 finally:r['closed_utc']=now();save(target,r)
