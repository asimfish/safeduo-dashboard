"""Execute immutable finite jobs; record actual wait before product validation."""
from pathlib import Path
import json,sys,os,subprocess,datetime,hashlib,time,signal
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
if __name__=='__main__':
 planpath=Path(sys.argv[1]);plan=json.loads(planpath.read_text());tag=plan['tag'];receipt=H/(tag+'_execution.json')
 if receipt.exists():raise ValueError('receipt exists; never silently rerun')
 r=dict(tag=tag,status='running',plan_sha256=sha(planpath),started_utc=now(),jobs=[]);write(receipt,r)
 try:
  for job in plan['jobs']:
   for p,d in plan['sources'].items():
    if sha(p)!=d:raise ValueError('frozen source changed '+p)
   gate=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
   gr=dict(actual_exit=gate.returncode,stdout=gate.stdout,stderr=gate.stderr,utc=now(),required_free_MiB=plan['resource']['min_initial_free_MiB'],selected_gpu=0)
   write(H/(tag+'_'+job['id']+'_resource_gate.json'),gr)
   if gate.returncode or int(gate.stdout.splitlines()[0].split(',')[1])<plan['resource']['min_initial_free_MiB']:raise RuntimeError('GPU0 resource gate failed; no child launched')
   out=Path(job['out']);
   if out.exists():raise ValueError('output exists; immutable attempt required')
   row=dict(id=job['id'],argv=job['argv'],env=job['env'],out=str(out),started_utc=now(),status='running');r['jobs'].append(row)
   with (H/(tag+'_'+job['id']+'.log')).open('xb') as log:
    c=subprocess.Popen(job['argv'],cwd=plan['cwd'],env={**os.environ,**job['env']},stdout=log,stderr=subprocess.STDOUT)
    row['pid']=c.pid;row['start_ticks']=Path('/proc/'+str(c.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19];write(receipt,r)
    monitor_start=time.monotonic();row['resource_samples']=[];row['resource_abort']=None
    while c.poll() is None:
     probe=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
     free=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True)
     abort=None;used=0;room=0
     if probe.returncode or free.returncode:abort='GPU resource monitor unavailable'
     else:
      gpu_uuid,room_text=free.stdout.splitlines()[0].split(',');room=int(room_text)
      for line in probe.stdout.splitlines():
       fields=[x.strip() for x in line.split(',')]
       if fields[0]==gpu_uuid.strip() and int(fields[1])==c.pid:used+=int(fields[2])
      if used>plan['resource']['max_owned_process_MiB']:abort='owned process exceeds registered GPU footprint'
      if room<plan['resource']['min_runtime_free_MiB']:abort='whole GPU remaining room below registered floor'
     if time.monotonic()-monitor_start>job['max_wall_seconds']:abort='registered finite job wall-clock exceeded'
     row['resource_samples'].append(dict(utc=now(),owned_GPU0_MiB=used,GPU0_free_MiB=room,probe_actual_exit=probe.returncode,free_probe_actual_exit=free.returncode))
     write(receipt,r)
     if abort:
      pidpath=Path('/proc')/str(c.pid)
      assert pidpath.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]==row['start_ticks'],'PID identity changed'
      assert [v.decode() for v in pidpath.joinpath('cmdline').read_bytes().split(b'\0') if v]==job['argv'],'owned argv mismatch'
      row['resource_abort']=abort;row['stop_signal']='SIGTERM';write(receipt,r);c.terminate()
      try:c.wait(timeout=15)
      except subprocess.TimeoutExpired:
       assert pidpath.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]==row['start_ticks'];c.kill();row['stop_signal']='SIGKILL';c.wait()
      break
     time.sleep(1)
    row['actual_exit']=c.wait();row['closed_utc']=now();row['status']='child_reaped';write(receipt,r)
   row['log_sha256']=sha(H/(tag+'_'+job['id']+'.log'));write(receipt,r)
   if row['actual_exit'] or row['resource_abort']:raise RuntimeError('actual child nonzero or resource-aborted; never accepted')
   for p,d in plan['sources'].items():
    if sha(p)!=d:raise ValueError('frozen source changed after child '+p)
   if job['kind']=='physics':
    product=json.loads((out/'visual_protocol.json').read_text())
    if product['status']!='complete' or product.get('completed_windows')!=64:raise RuntimeError('Kit zero exit but protocol incomplete '+str(product.get('error')))
    for n in ['cell_001.npz','input_recipe.npz','project_diagnostics.npz','native_receipts.json','camera_receipts.json','episodes.json']:
     if not (out/n).is_file():raise RuntimeError('missing closed product '+n)
    nr=json.loads((out/'native_receipts.json').read_text());cr=json.loads((out/'camera_receipts.json').read_text())
    if nr['steps']!=job['steps'] or cr['scheduled_groups']!=job['scheduled_groups']:raise RuntimeError('incomplete native/camera denominator')
   elif job['kind']=='bank':
    product=json.loads((out/'sampling_summary.json').read_text())
    for seed in job['bank_seeds']:
     if not (out/str(seed)/'bank.npz').is_file():raise RuntimeError('missing qualified bank')
   row['status']='complete';write(receipt,r);print('COMPLETE',tag,job['id'],flush=True)
  r['status']='complete'
 except BaseException as e:r.update(status='failed',error=type(e).__name__+': '+str(e));raise
 finally:r['closed_utc']=now();write(receipt,r)
