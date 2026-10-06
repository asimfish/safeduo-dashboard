"""Start fresh registered analysis only after canonical numerical closure."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib,json,os,subprocess,time

HERE=Path(__file__).resolve().parent


def run(name,python):
    argv=[python,str(HERE/name)]
    digest=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    with (HERE/(name[:-3]+'_execution.log')).open('x') as f:
        child=subprocess.Popen(argv,cwd=HERE,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdout=f,stderr=subprocess.STDOUT)
        rc=child.wait()
    assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
    row=dict(name=name,actual_argv=argv,child_pid=child.pid,exit_code=rc,source_sha256=digest)
    print('READBACK_CLOSED',name,rc,flush=True);return row


def wait_closed(name,status):
    while True:
        file=HERE/name
        if file.exists():
            data=json.loads(file.read_text())
            if data.get('status') in ['FAIL','failed']:raise RuntimeError(name+' failed; no completion fabricated')
            if data.get('status')==status:return
        time.sleep(5)


def main():
    wait_closed('NUMERIC_EXECUTION.json','PASS_ALL_NUMERIC_CLOSED')
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(lambda task:run(*task),[('execute_analysis.py','python3'),('failure_audit.py','/home/liyufeng/miniforge3/envs/safeduo/bin/python'),('audit_fixed_feasibility.py','/home/liyufeng/miniforge3/envs/safeduo/bin/python'),('audit_commands.py','python3')]))
    assert all(r['exit_code']==0 for r in rows),rows
    wait_closed('visual_execution.json','complete')
    rows.append(run('verify_visual.py','python3'))
    assert rows[-1]['exit_code']==0
    rows.append(run('audit_camera_state.py','python3'))
    assert rows[-1]['exit_code']==0
    with (HERE/'ALL_ANALYSIS_EXECUTION.json').open('x') as f:
        json.dump(dict(status='PASS_ALL6_FRESH_ANALYSIS_JOBS',jobs=rows,utc=datetime.now(timezone.utc).isoformat()),f,indent=2);f.write('\n')
    print('ALL_ANALYSIS_CLOSED',flush=True)


if __name__=='__main__':main()
