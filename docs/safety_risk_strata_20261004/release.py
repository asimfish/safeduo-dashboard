"""One evidence release; all checks precede normal, explicitly scoped Git pushes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import requests

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DASH=Path('/home/liyufeng/safeduo-dashboard')
DATA=Path('/home/liyufeng/safeduo-dashboard-data')
SIM_PY='/home/liyufeng/miniforge3/envs/safeduo/bin/python'
BROWSER_ENV={**os.environ,'PLAYWRIGHT_BROWSERS_PATH':'/tmp/safeduo_dashboard_browser',
             'PYTHONPATH':'/tmp/safeduo_dashboard_browser_tools'}
SOURCE_ENV={**os.environ,'PYTHONPATH':'src:'+str(HERE),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}


def command(argv,cwd,env=None,log=None):
    r=subprocess.run(argv,cwd=cwd,env=env,capture_output=True,text=True)
    if log:(HERE/log).write_text(r.stdout+r.stderr)
    if r.returncode:raise RuntimeError(f'command failed {argv[0:3]}: {r.stderr[-1500:]} {r.stdout[-1500:]}')
    return r.stdout.rstrip()


def git(repo,*args):return command(['git',*args],repo)


def sync_clean(repo,branch):
    assert not git(repo,'status','--porcelain'),'unrelated or unfinished work prevents automatic publication'
    assert git(repo,'branch','--show-current')==branch
    assert git(repo,'remote','get-url','origin')=='https://github.com/asimfish/safeduo-dashboard.git'
    response=requests.get(f'https://api.github.com/repos/asimfish/safeduo-dashboard/branches/{branch}',timeout=30)
    response.raise_for_status();assert response.json()['protected'] is False,'protected branch; publication withheld'
    git(repo,'fetch','origin',branch);git(repo,'merge','--ff-only',f'origin/{branch}')


def commit_and_push(repo,branch,paths,title):
    git(repo,'add','--',*paths);git(repo,'diff','--cached','--check')
    changed=git(repo,'diff','--cached','--name-only').splitlines()
    assert changed and all(any(n==p or n.startswith(p.rstrip('/')+'/') for p in paths) for n in changed)
    (HERE/f'release_{branch}_staged.txt').write_text(git(repo,'diff','--cached','--stat')+'\n')
    git(repo,'commit','-m',title)
    # Only the owned unpublished data commit may move after a verified bot-only race.
    try:git(repo,'push','origin',branch)
    except RuntimeError:
        if branch!='data':raise
        assert not git(repo,'status','--porcelain')
        git(repo,'fetch','origin',branch)
        remote_changes=git(repo,'diff','--name-only','HEAD^..origin/data').splitlines()
        assert set(remote_changes)<=set(['gates.json','matrix.json','runs.json','runs_5090.json']),remote_changes
        git(repo,'rebase','origin/data');git(repo,'push','origin','data')
    return git(repo,'rev-parse','HEAD')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    plan=json.loads((HERE/'campaign_plan.json').read_text())
    for name,sha in plan['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    assert hashlib.sha256(Path(plan['checkpoint_path']).read_bytes()).hexdigest()==plan['checkpoint_sha256']
    changes=git(DASH,'status','--porcelain').splitlines()
    assert all(line[3:]=='index.html' or line[3:].startswith('docs/'+HERE.name) for line in changes),changes
    assert git(DASH,'branch','--show-current')=='main'
    assert git(DASH,'remote','get-url','origin')=='https://github.com/asimfish/safeduo-dashboard.git'
    response=requests.get('https://api.github.com/repos/asimfish/safeduo-dashboard/branches/main',timeout=30)
    response.raise_for_status();assert response.json()['protected'] is False
    git(DASH,'fetch','origin','main');assert git(DASH,'rev-parse','HEAD')==git(DASH,'rev-parse','origin/main'),'parallel UI changes need review'
    sync_clean(DATA,'data')
    previous=json.loads((DATA/'scientific_eval.json').read_text());assert previous.get('safety_risk_strata') is None,'new field already exists; do not overwrite'
    (HERE/'previous_scientific.json').write_text(json.dumps(previous,ensure_ascii=False)+'\n')
    command([SIM_PY,str(HERE/'analyze_risk.py')]+(['--partial'] if args.partial else []),ROOT,SOURCE_ENV,'release_analysis.log')
    command([SIM_PY,'-m','unittest','discover','-s',str(HERE),'-p','test_risk_recipe.py','-v'],ROOT,SOURCE_ENV,'contract_tests.log')
    assert 'Ran 2 tests' in (HERE/'contract_tests.log').read_text() and 'OK' in (HERE/'contract_tests.log').read_text()
    command([SIM_PY,str(HERE/'prepare_panel.py'),'--preview-json',str(HERE/'preview_scientific.json')],ROOT,SOURCE_ENV,'release_prepare.log')
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    server=subprocess.Popen(['/usr/bin/python3','-m','http.server',str(port),'--bind','127.0.0.1'],cwd=DASH,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:requests.get(f'http://127.0.0.1:{port}/',timeout=1).raise_for_status();break
            except requests.RequestException:
                assert server.poll() is None;time.sleep(.1)
        command(['/usr/bin/python3',str(HERE/'check_browser.py'),'--local-url',f'http://127.0.0.1:{port}/#scientific'],HERE,BROWSER_ENV,'browser_check.log')
    finally:server.terminate();server.wait(timeout=10)
    public=DASH/'docs'/HERE.name
    for name in ['browser_check.log','panel.png','contract_tests.log']:shutil.copy2(HERE/name,public/name)
    git(DASH,'diff','--check')
    main_sha=commit_and_push(DASH,'main',['index.html','docs/'+HERE.name],'[dashboard/feat]: publish four-arm pair risk experiments')
    sync_clean(DATA,'data')
    now=json.loads((DATA/'scientific_eval.json').read_text());assert now==previous,'parallel scientific data changed during verification'
    command([SIM_PY,str(HERE/'prepare_panel.py')],ROOT,SOURCE_ENV,'release_data_prepare.log')
    assert not git(DASH,'status','--porcelain'),'report artifacts diverged after verified commit'
    shutil.copy2(DATA/'scientific_eval.json',HERE/'preview_scientific.json')
    data_sha=commit_and_push(DATA,'data',['scientific_eval.json'],'[data/feat]: report stable-initial risk safety and failures')
    expected=json.loads((HERE/'preview_scientific.json').read_text());url='https://asimfish.github.io/safeduo-dashboard/'
    errors=[]
    for attempt in range(30):
        try:
            page=requests.get(url+'?risk_release='+data_sha,timeout=30);page.raise_for_status();assert 'function renderRiskStrata' in page.text
            raw=requests.get('https://raw.githubusercontent.com/asimfish/safeduo-dashboard/data/scientific_eval.json?risk_release='+data_sha,timeout=30)
            raw.raise_for_status();assert raw.json()==expected
            report=requests.get(url+'docs/'+HERE.name+'/REPORT.md?risk_release='+main_sha,timeout=30);report.raise_for_status()
            assert report.text==(HERE/'REPORT.md').read_text()
            command(['/usr/bin/python3',str(HERE/'check_browser.py'),'--live-url',url+'?risk_release='+data_sha+'#scientific'],HERE,BROWSER_ENV,'live_browser_check.log')
            break
        except (AssertionError,RuntimeError,requests.RequestException) as e:
            errors.append(str(e))
            if attempt==29:raise
            time.sleep(10)
    r=expected['safety_risk_strata']
    delivery=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),main_commit=main_sha,data_commit=data_sha,
                  campaign_status=r['campaign_status'],completed_windows=r['completed_windows'],invalid_windows=r['invalid_windows'],pending_windows=r['pending_windows'],
                  local_browser=json.loads((HERE/'browser_check.log').read_text()),live_browser=json.loads((HERE/'live_browser_check.log').read_text()),
                  actor_sha256=plan['checkpoint_sha256'],production_source_unchanged=True,legacy_all_fields_preserved=True,page_report_raw_http_200=True,
                  url=url+'#scientific',deployment_wait_attempts=len(errors))
    (HERE/'DELIVERY.json').write_text(json.dumps(delivery,indent=2)+'\n');print(json.dumps(delivery))

if __name__=='__main__':main()
