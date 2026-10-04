"""Publish audited experimental evidence; safety promotion is a separate gate."""
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import requests

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'safety_risk_strata_20261004'))
# Import shared scoped Git operations without changing the prior evidence files.
import importlib.util
spec=importlib.util.spec_from_file_location('risk_release_helpers',HERE.parent/'safety_risk_strata_20261004/release.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior);prior.HERE=HERE
DASH,DATA,SIM_PY=prior.DASH,prior.DATA,prior.SIM_PY
command,git,sync_clean,commit_and_push=prior.command,prior.git,prior.sync_clean,prior.commit_and_push
BROWSER_ENV=prior.BROWSER_ENV
SOURCE_ENV={**os.environ,'PYTHONPATH':'src:'+str(HERE),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}


def main():
    plan=json.loads((HERE/'holdout_plan.json').read_text())
    changes=git(DASH,'status','--porcelain').splitlines()
    assert all(line[3:]=='index.html' or line[3:].startswith('docs/'+HERE.name) for line in changes),changes
    assert git(DASH,'branch','--show-current')=='main'
    assert git(DASH,'remote','get-url','origin')=='https://github.com/asimfish/safeduo-dashboard.git'
    api=requests.get('https://api.github.com/repos/asimfish/safeduo-dashboard/branches/main',timeout=30)
    api.raise_for_status();assert api.json()['protected'] is False
    git(DASH,'fetch','origin','main');assert git(DASH,'rev-parse','HEAD')==git(DASH,'rev-parse','origin/main')
    sync_clean(DATA,'data')
    previous=json.loads((DATA/'scientific_eval.json').read_text())
    assert 'safety_mechanism' not in previous
    (HERE/'previous_scientific.json').write_text(json.dumps(previous,ensure_ascii=False)+'\n')
    command([SIM_PY,str(HERE/'analyze_mechanism.py')],ROOT,SOURCE_ENV,'release_analysis.log')
    command([SIM_PY,str(HERE/'test_reference_envelope.py')],ROOT,SOURCE_ENV,'contract_tests.log')
    assert 'Ran 4 tests' in (HERE/'contract_tests.log').read_text() and 'OK' in (HERE/'contract_tests.log').read_text()
    assert 'EXPECTED RED' in (HERE/'red_contract.log').read_text()
    command([SIM_PY,'-m','compileall','-q',str(HERE)],ROOT,SOURCE_ENV,'compile.log')
    command([SIM_PY,str(HERE/'prepare_panel.py'),'--preview-json',str(HERE/'preview_scientific.json')],ROOT,SOURCE_ENV,'prepare.log')
    expected=json.loads((HERE/'preview_scientific.json').read_text());r=expected['safety_mechanism']
    assert r['completed_windows']==768 and r['invalid_windows']==0 and r['pending_windows']==0
    assert r['development_completed_windows']==128
    assert len(r['aggregate_rows'])==4 and len(r['pair_rows'])==24 and len(r['paired_inputs'])==12
    for k,v in previous.items():
        if k not in ('summary','updated','links'):assert expected[k]==v,k
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
    for name in ('panel.png','browser_check.log','contract_tests.log','red_contract.log','compile.log'):
        shutil.copy2(HERE/name,public/name)
    git(DASH,'diff','--check')
    # Small security/data check over the exact owned publication artifacts.
    forbidden=re.compile(r'(?i)(authorization\s*:\s*bearer\s+[A-Za-z0-9_-]{8,}|ct-[a-f0-9]{32}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|BEGIN [A-Z ]*PRIVATE KEY)')
    for p in [DASH/'index.html',*public.rglob('*')]:
        if p.is_file() and p.suffix in ('.json','.py','.md','.html','.log'):
            content=p.read_text()
            assert forbidden.search(content) is None,p
        assert p.suffix not in ('.npz','.pt'), 'bulk/private training artifacts must remain NAS-local'
    main_sha=commit_and_push(DASH,'main',['index.html','docs/'+HERE.name],'[dashboard/feat]: publish fixed-factor safety mechanism holdout')
    sync_clean(DATA,'data')
    assert json.loads((DATA/'scientific_eval.json').read_text())==previous
    command([SIM_PY,str(HERE/'prepare_panel.py')],ROOT,SOURCE_ENV,'data_prepare.log')
    assert not git(DASH,'status','--porcelain'),'public report must be deterministic after reviewed commit'
    shutil.copy2(DATA/'scientific_eval.json',HERE/'preview_scientific.json')
    data_sha=commit_and_push(DATA,'data',['scientific_eval.json'],'[data/feat]: retain full safety mechanism holdout results')
    expected=json.loads((HERE/'preview_scientific.json').read_text())
    url='https://asimfish.github.io/safeduo-dashboard/'
    errors=[]
    for attempt in range(30):
        try:
            page=requests.get(url+'?mechanism_release='+data_sha,timeout=30);page.raise_for_status()
            assert 'const renderMechanism' in page.text
            raw=requests.get('https://raw.githubusercontent.com/asimfish/safeduo-dashboard/data/scientific_eval.json?mechanism_release='+data_sha,timeout=30)
            raw.raise_for_status();assert raw.json()==expected
            report=requests.get(url+'docs/'+HERE.name+'/REPORT.md?mechanism_release='+main_sha,timeout=30)
            report.raise_for_status();assert report.text==(HERE/'REPORT.md').read_text()
            command(['/usr/bin/python3',str(HERE/'check_browser.py'),'--live-url',url+'?mechanism_release='+data_sha+'#scientific'],HERE,BROWSER_ENV,'live_browser_check.log')
            break
        except (AssertionError,RuntimeError,requests.RequestException) as e:
            errors.append(str(e))
            if attempt==29:raise
            time.sleep(10)
    delivery=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),url=url+'#scientific',
                  main_commit=main_sha,data_commit=data_sha,completed_windows=768,invalid_windows=0,pending_windows=0,
                  development_windows=128,actor_sha256=plan['checkpoint_sha256'],
                  production_source_unchanged=True,legacy_all_fields_preserved=True,page_report_raw_http_200=True,
                  local_browser=json.loads((HERE/'browser_check.log').read_text()),
                  live_browser=json.loads((HERE/'live_browser_check.log').read_text()),
                  deployment_wait_attempts=len(errors),
                  safety_promotion='not promoted; evaluate reported failures and motion limits separately')
    (HERE/'DELIVERY.json').write_text(json.dumps(delivery,indent=2)+'\n')
    print(json.dumps(delivery))


if __name__=='__main__':main()
