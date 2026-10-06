"""Bounded terminal gate; does not weaken scientific task acceptance."""
import argparse,json,hashlib,subprocess,os,ast,threading,http.server,functools
from pathlib import Path
from range_preview import RangePreview
R=Path(__file__).resolve().parent
SIMPY='/home/liyufeng/miniforge3/envs/safeduo/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args();out=a.repo/'docs'/R.name
    results=json.loads((R/'native_results.json').read_text());assert results['status']=='PASS_NATIVE_TRACE_AUDIT'
    assert results['complete_task_windows']==8 and results['new_final_trials']==0 and not results['full_g0_pass']
    assert results['nominal_reference_bitexact'];assert all(v[0]==0 and all(x>1e-4 for x in v[1:]) for v in results['reference_max_difference_rad'].values())
    initial=json.loads((out/'native_initial.json').read_text());assert initial['no_fixed_grasp_constraint']
    for c in results['cases']:assert c['pass_task']==all(v['pass_task'] for v in c['objects'].values())
    for name in ('REGISTRATION.json','REGISTRATION_V2.json','REGISTRATION_V3.json','REGISTRATION_V4.json','REGISTRATION_V5.json'):
        reg=json.loads((R/name).read_text())
        for path,digest in reg['source_sha256'].items():assert sha(path)==digest,(name,path)
    analysis=json.loads((R/'ANALYSIS_REGISTRATION.json').read_text());assert sha(R/'analyze_native.py')==analysis['source_sha256']
    repair=json.loads((R/'ANALYSIS_OUTPUT_REPAIR.json').read_text())
    assert sha(R/'analyze_native_v2.py')==repair['corrected_source_sha256']==results['source_sha256']
    assert (R/'analyze_native_v2.py').read_text().replace(repair['new_expression'],repair['old_expression'])==(R/'analyze_native.py').read_text()
    assets=json.loads((R/'ASSET_BEFORE.json').read_text())
    for path,digest in assets['files'].items():assert sha(path)==digest,path
    before=json.loads((R/'SHARED_SOURCE_BEFORE.json').read_text())
    for rel,digest in before['files'].items():assert sha(Path('/home/liyufeng/safeduo')/rel)==digest,rel
    for file in out.rglob('*.py'):ast.parse(file.read_text())
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':'/home/liyufeng/safeduo/src','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    for name in ('test_binding.py','test_clock.py','test_legacy_input.py','test_reference.py','test_binding_current.py','test_reference_current.py'):
        subprocess.run([SIMPY,'-m','unittest','discover','-s',str(R),'-p',name,'-v'],env=env,check=True)
    with (R/'terminal_baseline_red.log').open('w') as f:
        red=subprocess.run([SIMPY,'-m','unittest','discover','-s',str(R),'-p','test_reference_v1.py','-v'],env=env,stdout=f,stderr=subprocess.STDOUT)
    assert red.returncode!=0 and 'binding changed registered q0' in (R/'terminal_baseline_red.log').read_text()
    for item in json.loads((out/'LOG_MANIFEST.json').read_text()):
        import gzip
        assert hashlib.sha256(gzip.decompress((out/item['file']).read_bytes())).hexdigest()==item['raw_sha256']
    for rel,digest in json.loads((out/'PUBLIC_MANIFEST.json').read_text())['files'].items():assert sha(out/rel)==digest,rel
    source=ast.parse((R/'native_runner_v5.py').read_text());main=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    execution=next(n for n in main.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step')
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('write_root_pose_to_sim','write_root_velocity_to_sim') for n in ast.walk(execution))
    # Retained legacy stdout contains incidental whitespace; compressed raw logs
    # preserve those bytes. Generated text must still pass the scoped Git gate.
    for staged in (False,True):
        diff=subprocess.run(['git','diff',*(['--cached'] if staged else []),'--check','--','index.html','docs/'+R.name],cwd=a.repo,capture_output=True,text=True)
        assert diff.returncode==0,diff.stdout[:2000]
    media=json.loads((R/'MEDIA_ARCHIVE.json').read_text());assert media['status']=='PASS_ALL_REMOTE_MEDIA_SHA'
    assert media['checked_files']==len(media['files'])==151

    patterns=['Authorization:'+' Bearer','gh'+'p_','sk-'+'proj-']
    for file in out.rglob('*'):
        if file.is_file() and file.suffix in ('.py','.html','.json','.md','.csv'):
            text=file.read_text()
            assert not any(token in text for token in patterns),file
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(RangePreview,directory=str(a.repo)))
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    browserenv={**os.environ,'PLAYWRIGHT_BROWSERS_PATH':'/tmp/safeduo_dashboard_browser','PYTHONPATH':'/tmp/safeduo_dashboard_browser_tools'}
    try:
        url=f'http://127.0.0.1:{server.server_port}/docs/{R.name}/'
        subprocess.run(['/usr/bin/python3',str(R/'check_browser.py'),'--url',url,'--out',str(R/'browser_local')],env=browserenv,check=True)
    finally:server.shutdown();server.server_close();worker.join(timeout=3)
    verdict=dict(status='PASS_LOCAL_DIAGNOSTIC_DELIVERY_GATE',native_trials=8,new_final_trials=0,positive_software_tests=11,legacy_red_reproduced=True,all_shared_files_unchanged=before['count'],actual_browser_desktop_mobile=True,scientific_full_g0=False,scope='bounded complete development evidence; software and delivery PASS do not replace failed task/safety gates')
    (R/'TERMINAL_GATE.json').write_text(json.dumps(verdict,indent=2)+'\n');print(json.dumps(verdict))
if __name__=='__main__':main()
