"""Read the actual deployed commit and every owned Pages file, original bytes."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
import argparse,hashlib,json,socket,subprocess,threading,time,urllib.error,urllib.request
HERE=Path(__file__).resolve().parent
WORK=Path('/home/liyufeng/safeduo-dashboard-feasible-guard-20261005')
API='https://api.github.com/repos/asimfish/safeduo-dashboard/'
BASE='https://asimfish.github.io/safeduo-dashboard/'
NETWORK_FAILURES=[];LOCK=threading.Lock()
def sha(b):return hashlib.sha256(b).hexdigest()
def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'SafeDuo-frozen-evidence-readback','Cache-Control':'no-cache'})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req,timeout=45) as response:return response.read()
        except (urllib.error.URLError,TimeoutError,socket.timeout,ConnectionError) as e:
            if isinstance(e,urllib.error.HTTPError):raise
            with LOCK:NETWORK_FAILURES.append(dict(url=url,attempt=attempt+1,type=type(e).__name__,utc=datetime.now(timezone.utc).isoformat()))
            if attempt==3:raise
            time.sleep([2,5,10][attempt])
def probe():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip()
    branch=json.loads(get(API+'branches/main'));runs=json.loads(get(API+'actions/runs?head_sha='+commit+'&per_page=20'))['workflow_runs']
    pages=next((r for r in runs if r['name']=='pages build and deployment'),{})
    return dict(commit=commit,remote_main=branch['commit']['sha'],protected=branch['protected'],pages_commit=pages.get('head_sha'),
        pages_status=pages.get('conclusion') if pages.get('status')=='completed' else pages.get('status'),pages_run_url=pages.get('html_url'),utc=datetime.now(timezone.utc).isoformat())
def main():
    a=argparse.ArgumentParser();a.add_argument('--probe',action='store_true');a.add_argument('--receipt',default='remote_verification.json');args=a.parse_args()
    assert Path(args.receipt).name==args.receipt
    p=probe();(HERE/'pages_probe_latest.json').write_text(json.dumps(p,indent=2)+'\n')
    if args.probe:print(json.dumps(p));return
    assert p['remote_main']==p['commit']==p['pages_commit'] and p['pages_status']=='success',p
    paths=[WORK/'index.html',*sorted((WORK/'docs'/HERE.name).rglob('*'))]
    paths=[x for x in paths if x.is_file()]
    frozen={x.relative_to(WORK).as_posix():dict(sha256=sha(x.read_bytes()),bytes=x.stat().st_size) for x in paths}
    def read(rel):
        data=get(BASE+rel+'?fifo_commit='+p['commit']);assert len(data)==frozen[rel]['bytes'] and sha(data)==frozen[rel]['sha256'],rel
        return dict(path=rel,**frozen[rel])
    with ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(read,frozen))
    assert all(sha((WORK/x).read_bytes())==v['sha256'] for x,v in frozen.items())
    cert='docs/'+HERE.name+'/VERIFICATION.json'
    receipt=dict(status='PASS_EXACT_COMMIT_AND_ALL_OWNED_PAGES_BYTES',**p,url=BASE+'docs/'+HERE.name+'/',files=len(rows),bytes=sum(x['bytes'] for x in rows),
       readback=rows,source_sha256=sha(Path(__file__).read_bytes()),asset_publication_sha256=sha((HERE/'ASSET_PUBLICATION.json').read_bytes()),
       network_failures_retried=NETWORK_FAILURES,certificate_path=cert,scope='all new owned Pages files and current inherited root; unaltered image/state/result assets independently verified on their pinned commit')
    with (HERE/args.receipt).open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    if args.receipt=='remote_verification_initial.json':(HERE/'PUBLIC_INITIAL_ROOT.html').write_bytes((WORK/'index.html').read_bytes())
    print('REMOTE_PASS',p['commit'],len(rows),receipt['bytes'],flush=True)
if __name__=='__main__':main()
