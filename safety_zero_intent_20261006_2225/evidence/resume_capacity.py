"""Resume only exact paused own processes after equivalent preallocation capacity."""
from pathlib import Path
import csv,io,json,os,signal,subprocess,time
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent

def query():
    r=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    g={int(x[0]):dict(uuid=x[1].strip(),free_MiB=int(x[2])) for x in csv.reader(io.StringIO(r.stdout))}
    r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    apps=[dict(pid=int(x[0]),uuid=x[1].strip(),MiB=int(x[2])) for x in csv.reader(io.StringIO(r.stdout))]
    return g,apps

def identity(row):
    p=Path('/proc')/str(row['pid']);args=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\0') if v]
    ticks=(p/'stat').read_text().rsplit(')',1)[1].split()[19]
    assert args==row['argv'] and ticks==row['start_ticks'],'paused PID identity changed; do not signal'

if __name__=='__main__':
    original=json.loads((HERE/'INITIAL_RESOURCE_GATE_CONTAINMENT.json').read_text());rows=original['paused_exact_own_processes'];owned={r['pid'] for r in rows};history=[];began=time.monotonic()
    while True:
        g,apps=query();ownmem={i:sum(a['MiB'] for a in apps if a['pid'] in owned and a['uuid']==v['uuid']) for i,v in g.items()}
        state=dict(utc=datetime.now(timezone.utc).isoformat(),gpus={str(i):dict(actual_free_MiB=v['free_MiB'],paused_own_allocations_MiB=ownmem[i],equivalent_preallocation_headroom_MiB=v['free_MiB']+ownmem[i]) for i,v in g.items()})
        history.append(state)
        if all(v['free_MiB']+ownmem[i]>=25600 for i,v in g.items()):
            for row in rows:identity(row)
            before=dict(status='REGISTERED_RESUME_AT_EQUIVALENT_PREALLOCATION_CAPACITY',threshold_MiB=25600,original_initial_gate='FAIL',original_initial_failure_not_overwritten=True,observations=history,scope='actual free plus already allocated exact paused own compute/graphics process memory; conservative excludes unreported own allocations',source_or_state_or_FIFO_changed=False)
            with (HERE/'RESOURCE_PAUSE_RESUME_GATE.json').open('x') as f:json.dump(before,f,indent=2)
            for row in rows:os.kill(row['pid'],signal.SIGCONT)
            with (HERE/'RESOURCE_PAUSE_RESUME_EXECUTION.json').open('x') as f:json.dump(dict(status='PASS_EXACT_PAUSED_PROCESSES_RESUMED_AFTER_EQUIVALENT_HEADROOM_GATE',rows=rows,utc=datetime.now(timezone.utc).isoformat(),initial_protocol_deviation_retained=True),f,indent=2)
            print('EXACT_OWN_PAUSED_PROCESSES_RESUMED',len(rows),flush=True);break
        if time.monotonic()-began>3600:raise TimeoutError('registered resource pause exceeded1hour; no resumePASS')
        time.sleep(5)
