"""Separate terminal gate for evidence integrity, scientific scope and publication."""
from pathlib import Path
import argparse,hashlib,json,os,re,subprocess,threading,functools,http.server,csv
import numpy as np
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(argv,cwd,env=None):return subprocess.run(argv,cwd=cwd,env=env,text=True,capture_output=True,check=True).stdout.strip()
def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args();doc=a.repo/'docs'/HERE.name
    audit=json.loads((doc/'coverage_audit.json').read_text());data=json.loads((doc/'data.json').read_text());initial=json.loads((doc/'initial_coverage.json').read_text())
    reg=json.loads((doc/'AUDIT_REGISTRATION.json').read_text())
    for path,h in reg['source_sha256'].items():assert sha(doc/Path(path).name)==h,path
    for name,h in json.loads((doc/'INITIAL_SUPPLEMENT_REGISTRATION.json').read_text())['source_sha256'].items():assert sha(doc/name)==h,name
    assert audit['audited_historical_windows']==768 and audit['new_physical_windows']==0 and len(audit['rows'])==12
    assert audit['contact_oracle'] is None and audit['operational_reliability_ci'] is None
    assert initial['actual_initial_states_bitexact_between_methods'] is True
    assert len(data['matrix']['rows'])==20 and data['matrix']['final_inputs_registered'] is False
    assert data['matrix']['final_protocol_experiments_executed']==0
    assert [g['violations'] for g in data['groups']]==[50,16,25,30]
    assert all(g['windows']==192 for g in data['groups'])
    assert all(r['status']=='NOT_READY_TO_REGISTER' for r in data['readiness'].values())
    for g in data['groups']:
        assert sha(doc/'histograms'/(g['mode']+'.npz'))==g['histogram_sha256']
        with np.load(doc/'histograms'/(g['mode']+'.npz'),allow_pickle=False) as z:
            assert z['full_pairs'].shape==(325,100) and z['full_marginal'].shape==(26,10)
            assert np.array_equal(z['full_pairs'],g['full_counts']) and np.array_equal(z['prefix_pairs'],g['prefix_counts'])
            assert np.all(z['prefix_pairs']<=z['full_pairs'])
        assert g['full_pairs']['mean_percent']==np.mean((np.array(g['full_counts'])>0).sum(-1))
        assert g['initial_mean_pair_percent']==np.mean((np.array(g['initial_counts'])>0).sum(-1))
    old=data['inventory']['historical_two_pair_physical'];assert (old['pass_count'],old['fail_count'],old['measurement_status'])==(19,173,'FAIL_UNCALIBRATED')
    old_source=json.loads((doc/'prior_evidence/analysis.json').read_text());review=data['prior_task_review']
    assert sha(doc/'prior_evidence/analysis.json')==review['source_sha256']
    assert review['tasks']==192 and review['task_pass']==19 and review['task_fail']==173
    assert review['failed_gates']==old_source['totals']['failed_gates']
    for name in ('all_joint_pairs.csv','all_window_motion.csv'):
        with (doc/name).open() as f:rows=list(csv.DictReader(f))
        assert len(rows)==(1300 if 'pairs' in name else 768)
    patterns=[r'Bearer\s+[a-zA-Z0-9_.-]{15,}',r'ct-[0-9a-f]{24,}',r'gh[pousr]_[a-zA-Z0-9]{25,}',r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY']
    for file in doc.rglob('*'):
        if file.is_file() and file.suffix in ('.json','.md','.log','.py','.html'):
            content=file.read_text()
            for pattern in patterns:assert not re.search(pattern,content),file
            if file.suffix=='.py':compile(content,str(file),'exec')
    run(['git','diff','origin/main','--check','--','index.html','docs/'+HERE.name],a.repo)
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(http.server.SimpleHTTPRequestHandler,directory=str(a.repo)))
    t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
    env={**os.environ,'PLAYWRIGHT_BROWSERS_PATH':'/tmp/safeduo_dashboard_browser','PYTHONPATH':'/tmp/safeduo_dashboard_browser_tools'}
    try:result=run(['/usr/bin/python3',str(HERE/'check_browser.py'),'--url',f'http://127.0.0.1:{server.server_port}/docs/{HERE.name}/','--out',str(HERE/'browser_local')],a.repo,env)
    finally:server.shutdown();server.server_close();t.join()
    receipt=dict(status='PASS_DELIVERY_GATE_NOT_PHYSICAL_ACCEPTANCE',audit_windows=768,new_physics=0,old_endpoints_unchanged=True,
        all_source_digests_pass=True,all_pair_histograms_exact=True,all325_pairs=True,design_families=20,
        metadata_inventory_not_raw_reanalysis=True,scope_limits_preserved=True,browser=json.loads(result))
    (HERE/'TERMINAL_GATE.json').write_text(json.dumps(receipt,indent=2)+'\n')
    import shutil;shutil.copy2(HERE/'TERMINAL_GATE.json',doc/'TERMINAL_GATE.json')
    manifest={str(f.relative_to(doc)):sha(f) for f in doc.rglob('*') if f.is_file() and f.name!='PUBLIC_MANIFEST.json'}
    (doc/'PUBLIC_MANIFEST.json').write_text(json.dumps(dict(status='retrospective_audit_and_future_design',files=manifest),indent=2)+'\n')
    print(json.dumps(receipt))
if __name__=='__main__':main()
