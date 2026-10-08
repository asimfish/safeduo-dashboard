"""Audit each fully reaped raw cell while the frozen finite producer continues."""
from pathlib import Path
import json, subprocess, concurrent.futures, time, sys, datetime
H=Path(__file__).resolve().parent
PY='/home/liyufeng/miniforge3/envs/safeduo/bin/python'
def audit(job, kind, script):
    name=job['id'];out=str(H/('ASTRA_NATIVE_REAL_'+name+'.json')) if kind=='peer' else name+'_'+kind
    argv=[PY,'-B',str(H/script),job['out'],out]+(['--require-contacts'] if kind=='peer' else [])
    tag='audit_'+name+'_'+kind
    p=subprocess.Popen([PY,'-B',str(H/'run_step.py'),tag,*argv],cwd=H,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    data,_=p.communicate();rc=p.wait()
    result=dict(job_id=name,kind=kind,tag=tag,pid=p.pid,actual_exit=rc,output=data.decode(errors='replace'))
    print(json.dumps(result),flush=True)
    return result
def main():
    reg=json.loads((H/'ACCEPTANCE_REGISTRATION.json').read_text());producer=H/(reg['tag']+'_execution.json');target=H/'ACCEPTANCE_AUDIT_MATRIX_RESULT.json'
    assert not target.exists();submitted=set();results=[];futures=[];failed=False
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        while True:
            if not producer.exists():time.sleep(5);continue
            x=json.loads(producer.read_text())
            for ran in x['jobs']:
                if ran['status']!='complete' or ran['id'] in submitted:continue
                assert ran['actual_exit']==0
                job=next(j for j in reg['jobs'] if j['id']==ran['id']);submitted.add(ran['id'])
                for kind,script in [('native','audit_native.py'),('geometry','audit_geometry_v2.py'),('contacts','audit_contacts.py'),('opening','audit_opening.py'),('hands','audit_hand_views.py'),('peer','astra_native_audit.py')]:
                    futures.append(pool.submit(audit,job,kind,script))
            if x['status'] in ['complete','failed']:
                failed=x['status']=='failed';break
            time.sleep(10)
        for f in futures:results.append(f.result())
    success=not failed and len(submitted)==8 and len(results)==48 and all(v['actual_exit']==0 for v in results)
    with target.open('x') as f:json.dump(dict(status='complete' if success else 'failed',closed_cells=len(submitted),audits=results,producer_failed=failed,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),f,indent=2);f.write('\n')
    return 0 if success else 1
if __name__=='__main__':sys.exit(main())
