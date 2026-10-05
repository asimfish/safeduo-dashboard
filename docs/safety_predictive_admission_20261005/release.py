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
    for branch in ('main','data'):
        api=requests.get('https://api.github.com/repos/asimfish/safeduo-dashboard/branches/'+branch,timeout=30)
        api.raise_for_status();assert api.json()['protected'] is False,branch
    git(DASH,'fetch','origin','main');assert git(DASH,'rev-parse','HEAD')==git(DASH,'rev-parse','origin/main')
    sync_clean(DATA,'data')
    previous=json.loads((DATA/'scientific_eval.json').read_text())
    assert 'safety_predictive_admission' not in previous
    (HERE/'previous_scientific.json').write_text(json.dumps(previous,ensure_ascii=False)+'\n')
    metrics=json.loads((HERE/'METRICS_REGISTRATION.json').read_text())
    for name,sha in metrics['source_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
    command([SIM_PY,str(HERE/'analyze_predictive.py')],ROOT,SOURCE_ENV,'release_analysis.log')
    command([SIM_PY,str(HERE.parent/'safety_zero_command_20261005/analyze_zero.py')],ROOT,SOURCE_ENV,'zero_analysis.log')
    zero=json.loads((HERE.parent/'safety_zero_command_20261005/results.json').read_text())
    assert zero['completed_windows']==576 and zero['invalid_windows']==0 and zero['pending_windows']==0
    command([SIM_PY,str(HERE.parent/'safety_zero_predictive_20261005/analyze_zero_predictive.py')],ROOT,SOURCE_ENV,'zero_predictive_analysis.log')
    zero_all=json.loads((HERE.parent/'safety_zero_predictive_20261005/results.json').read_text())
    assert zero_all['completed_windows']==960 and zero_all['invalid_windows']==0 and zero_all['pending_windows']==0
    command([SIM_PY,str(HERE.parent/'safety_speed_box_control_20261005/analyze_speed.py')],ROOT,SOURCE_ENV,'speed_analysis.log')
    speed=json.loads((HERE.parent/'safety_speed_box_control_20261005/results.json').read_text())
    assert speed['completed_windows']==192 and speed['invalid_windows']==0 and speed['pending_windows']==0
    assert len(speed['paired_inputs'])==6
    command([SIM_PY,str(HERE.parent/'safety_speed_box_control_20261005/analyze_same_device.py')],ROOT,SOURCE_ENV,'speed_same_device_analysis.log')
    speed_same=json.loads((HERE.parent/'safety_speed_box_control_20261005/same_device_results.json').read_text())
    assert speed_same['same_device'] and speed_same['comparison_completed_windows']==384
    assert speed_same['supplementary_total_windows']==576 and len(speed_same['paired_inputs'])==6

    command([SIM_PY,str(HERE/'test_predictive_rows.py')],ROOT,SOURCE_ENV,'contract_tests.log')
    command([SIM_PY,str(HERE/'test_operation_metrics.py')],ROOT,SOURCE_ENV,'metric_tests.log')
    assert 'Ran 2 tests' in (HERE/'metric_tests.log').read_text() and 'OK' in (HERE/'metric_tests.log').read_text()
    assert 'Ran 6 tests' in (HERE/'contract_tests.log').read_text() and 'OK' in (HERE/'contract_tests.log').read_text()
    assert 'FAIL: test_closing_far_row_must_be_admitted_before_instant_band' in (HERE/'red_contract.log').read_text()
    command([SIM_PY,'-m','compileall','-q',str(HERE),str(HERE.parent/'safety_zero_command_20261005'),
             str(HERE.parent/'safety_zero_predictive_20261005'),str(HERE.parent/'safety_speed_box_control_20261005')],ROOT,SOURCE_ENV,'compile.log')
    command([SIM_PY,str(HERE/'prepare_panel.py'),'--preview-json',str(HERE/'preview_scientific.json')],ROOT,SOURCE_ENV,'prepare.log')
    expected=json.loads((HERE/'preview_scientific.json').read_text());r=expected['safety_predictive_admission']
    assert r['completed_windows']==960 and r['invalid_windows']==0 and r['pending_windows']==0
    assert r['development_completed_windows']==128
    assert len(r['aggregate_rows'])==5 and len(r['pair_rows'])==30 and len(r['paired_inputs'])==15
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
    for name in ('panel.png','panel_table.png','panel_safe.png','panel_trace.png','browser_check.log','contract_tests.log','red_contract.log','compile.log'):
        shutil.copy2(HERE/name,public/name)
    git(DASH,'diff','--check')
    # Small security/data check over the exact owned publication artifacts.
    forbidden=re.compile(r'(?i)(authorization\s*:\s*bearer\s+[A-Za-z0-9_-]{8,}|ct-[a-f0-9]{32}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|BEGIN [A-Z ]*PRIVATE KEY)')
    for p in [DASH/'index.html',*public.rglob('*')]:
        if p.is_file() and p.suffix in ('.json','.py','.md','.html','.log'):
            content=p.read_text()
            assert forbidden.search(content) is None,p
        assert p.suffix not in ('.npz','.pt'), 'bulk/private training artifacts must remain NAS-local'
    main_sha=commit_and_push(DASH,'main',['index.html','docs/'+HERE.name],'[dashboard/feat]: publish predictive admission and fresh-bank five-method holdout')
    sync_clean(DATA,'data')
    assert json.loads((DATA/'scientific_eval.json').read_text())==previous
    command([SIM_PY,str(HERE/'prepare_panel.py')],ROOT,SOURCE_ENV,'data_prepare.log')
    assert not git(DASH,'status','--porcelain'),'public report must be deterministic after reviewed commit'
    shutil.copy2(DATA/'scientific_eval.json',HERE/'preview_scientific.json')
    data_sha=commit_and_push(DATA,'data',['scientific_eval.json'],'[data/feat]: retain full predictive admission five-method results')
    expected=json.loads((HERE/'preview_scientific.json').read_text())
    url='https://asimfish.github.io/safeduo-dashboard/'
    errors=[]
    for attempt in range(30):
        try:
            page=requests.get(url+'?admission_release='+data_sha,timeout=30);page.raise_for_status()
            assert 'const renderAdmission' in page.text
            raw=requests.get('https://raw.githubusercontent.com/asimfish/safeduo-dashboard/data/scientific_eval.json?admission_release='+data_sha,timeout=30)
            raw.raise_for_status();assert raw.json()==expected
            report=requests.get(url+'docs/'+HERE.name+'/REPORT.md?admission_release='+main_sha,timeout=30)
            report.raise_for_status();assert report.text==(HERE/'REPORT.md').read_text()
            command(['/usr/bin/python3',str(HERE/'check_browser.py'),'--live-url',url+'?admission_release='+data_sha+'#scientific'],HERE,BROWSER_ENV,'live_browser_check.log')
            break
        except (AssertionError,RuntimeError,requests.RequestException) as e:
            errors.append(str(e))
            if attempt==29:raise
            time.sleep(10)
    delivery=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),url=url+'#scientific',
                  main_commit=main_sha,data_commit=data_sha,completed_windows=960,invalid_windows=0,pending_windows=0,
                  development_windows=128,neutral_complete_windows=960,neutral_new_random_windows=0,
                  speed_box_complete_windows=192,speed_box_same_gpu_repeated_windows=384,
                  speed_box_new_random_windows=0,actor_sha256=plan['checkpoint_sha256'],
                  production_source_unchanged=True,legacy_all_fields_preserved=True,page_report_raw_http_200=True,
                  local_browser=json.loads((HERE/'browser_check.log').read_text()),
                  live_browser=json.loads((HERE/'live_browser_check.log').read_text()),
                  deployment_wait_attempts=len(errors),
                  safety_promotion='not promoted; evaluate reported failures and motion limits separately')
    (HERE/'DELIVERY.json').write_text(json.dumps(delivery,indent=2)+'\n')
    print(json.dumps(delivery))


if __name__=='__main__':main()
