"""Publish own bounded evidence and verify exact remote commit and public files."""
from pathlib import Path
import argparse,json,hashlib,subprocess,threading,http.server,functools,os,time,urllib.request,re
HERE=Path(__file__).resolve().parent
SIM='/home/liyufeng/miniforge3/envs/safeduo/bin/python'
DOC=HERE.name
PUBLIC='https://asimfish.github.io/safeduo-dashboard/'
def run(argv,cwd,env=None):return subprocess.run(argv,cwd=cwd,env=env,check=True,text=True,capture_output=True).stdout.strip()
def get(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'SafeDuo-evidence-check'}),timeout=60) as r:return r.read()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def test(repo,phase,out):
    handler=functools.partial(http.server.SimpleHTTPRequestHandler,directory=str(repo))
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{server.server_port}/docs/{DOC}/'
    env={**os.environ,'PLAYWRIGHT_BROWSERS_PATH':'/tmp/safeduo_dashboard_browser','PYTHONPATH':'/tmp/safeduo_dashboard_browser_tools'}
    try:run(['/usr/bin/python3',str(HERE/'check_browser.py'),'--url',url,'--phase',phase,'--out',str(out)],repo,env)
    finally:server.shutdown();server.server_close();thread.join()
def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--phase',choices=['running','complete'],required=True);p.add_argument('--no-push',action='store_true');a=p.parse_args()
    repo=a.repo;doc=repo/'docs'/DOC
    run([SIM,str(HERE/'present.py'),'--repo',str(repo),'--phase',a.phase],repo)
    for name in ('VERIFY.md','check_browser.py','publish.py'):
        import shutil;shutil.copy2(HERE/name,doc/name)
    test(repo,a.phase,HERE/('browser_'+a.phase))
    for file in doc.glob('*.py'):compile(file.read_text(),str(file),'exec')
    patterns=[r'Bearer\s+[a-zA-Z0-9_.-]{15,}',r'ct-[0-9a-f]{24,}',r'gh[pousr]_[a-zA-Z0-9]{25,}',r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY']
    for file in doc.rglob('*'):
        if file.is_file() and file.suffix in ('.json','.py','.md','.html','.log'):
            content=file.read_text()
            for pattern in patterns:assert not re.search(pattern,content),file
    run(['git','diff','--check'],repo)
    # Public manifest covers final source copies and generated presentation.
    manifest={str(f.relative_to(doc)):sha(f) for f in doc.rglob('*') if f.is_file() and f.name!='PUBLIC_MANIFEST.json'}
    (doc/'PUBLIC_MANIFEST.json').write_text(json.dumps(dict(files=manifest,status=a.phase),indent=2)+'\n')
    if a.no_push:print('PASS_LOCAL',a.phase);return
    remote=json.loads(get('https://api.github.com/repos/asimfish/safeduo-dashboard/branches/main'))
    assert not remote['protected'],'protected branch: external integration requires its policy'
    run(['git','add','--','index.html','docs/'+DOC],repo)
    run(['git','diff','--cached','--check'],repo)
    staged=run(['git','diff','--cached','--name-only'],repo).splitlines()
    assert staged and all(f=='index.html' or f.startswith('docs/'+DOC+'/') for f in staged)
    run(['git','commit','-m','[dashboard/docs]: publish '+('passive dynamics random experiment' if a.phase=='complete' else 'frozen passive dynamics experiment progress')],repo)
    commit=run(['git','rev-parse','HEAD'],repo)
    run(['git','push','origin','HEAD:main'],repo)
    receipt=dict(status='pushed_pending_ci',commit=commit,phase=a.phase,public_files=len(manifest),utc=time.time())
    dest=HERE/('delivery_'+a.phase+'.json');dest.write_text(json.dumps(receipt,indent=2)+'\n')
    deadline=time.monotonic()+900;ci=None
    while time.monotonic()<deadline:
        runs=json.loads(get('https://api.github.com/repos/asimfish/safeduo-dashboard/actions/runs?head_sha='+commit+'&per_page=20'))['workflow_runs']
        ci=next((r for r in runs if r['name']=='pages build and deployment' and r['head_sha']==commit),None)
        if ci and ci['status']=='completed':break
        time.sleep(15)
    assert ci and ci['status']=='completed' and ci['conclusion']=='success',ci
    for rel,h in manifest.items():
        data=get(PUBLIC+'docs/'+DOC+'/'+rel);assert hashlib.sha256(data).hexdigest()==h,rel
    env={**os.environ,'PLAYWRIGHT_BROWSERS_PATH':'/tmp/safeduo_dashboard_browser','PYTHONPATH':'/tmp/safeduo_dashboard_browser_tools'}
    run(['/usr/bin/python3',str(HERE/'check_browser.py'),'--url',PUBLIC+'docs/'+DOC+'/','--phase',a.phase,'--out',str(HERE/('live_browser_'+a.phase))],repo,env)
    receipt.update(status='PASS_EXACT_COMMIT_CI_PUBLIC_READBACK_BROWSER',ci_run_id=ci['id'],public_url=PUBLIC+'docs/'+DOC+'/',finished_utc=time.time())
    dest.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
