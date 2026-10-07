"""Read actual pinned assets and deployed Pages files twice, with bounded retries."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import argparse,hashlib,json,time,urllib.request

HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nas/data/lyf/double_hand/safeduo_dashboard_zero_worktree_20261006')
ASSET_WORK=Path('/mnt/nas/data/lyf/double_hand/safeduo_dashboard_zero_assets_20261006')
API='https://api.github.com/repos/asimfish/safeduo-dashboard/'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def get(url):
    last=None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'SafeDuo-evidence-verifier'}),timeout=50) as response:return response.read()
        except Exception as error:
            last=error;time.sleep(min(2**attempt,8))
    raise last


def verify(rows,base):
    def inspect(row):
        value=get(base+row['path'])
        assert len(value)==row['bytes'] and hashlib.sha256(value).hexdigest()==row['sha256'],row['path']
        return row
    with ThreadPoolExecutor(max_workers=6) as pool:return list(pool.map(inspect,rows))


def assets(commit):
    root=ASSET_WORK/HERE.name;manifest=root/'asset_manifest.json';m=json.loads(manifest.read_text())
    rows=m['files']+[dict(path='asset_manifest.json',bytes=manifest.stat().st_size,sha256=sha(manifest))]
    base='https://raw.githubusercontent.com/asimfish/safeduo-dashboard/'+commit+'/'+HERE.name+'/'
    for read in range(2):verify(rows,base);print('ASSET_PUBLIC_BYTE_PASS',read,len(rows),flush=True)
    receipt=dict(status='PASS_TWO_PUBLIC_BYTE_READS',commit=commit,files=len(rows),bytes=sum(r['bytes'] for r in rows),asset_bytes=m['bytes'],base=base,manifest_sha256=sha(manifest),utc=datetime.now(timezone.utc).isoformat())
    with (HERE/'ASSET_PUBLICATION.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')


def pages(commit,tag):
    branch=json.loads(get(API+'branches/main'));assert branch['commit']['sha']==commit and not branch['protected']
    runs=json.loads(get(API+'actions/runs?head_sha='+commit+'&per_page=30'))['workflow_runs']
    pages=[r for r in runs if 'pages' in r['name'].lower()];assert pages and any(r['conclusion']=='success' and r['head_sha']==commit for r in pages),[(r['name'],r['status'],r['conclusion']) for r in runs]
    public=WORK/'docs'/HERE.name
    rows=[dict(path=str(p.relative_to(public)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(public.rglob('*')) if p.is_file()]
    base='https://asimfish.github.io/safeduo-dashboard/docs/'+HERE.name+'/'
    for read in range(2):verify(rows,base);print('PAGES_PUBLIC_BYTE_PASS',read,len(rows),flush=True)
    root=get('https://asimfish.github.io/safeduo-dashboard/?tracking='+commit);assert root==(WORK/'index.html').read_bytes()
    receipt=dict(status='PASS_DEPLOYED_EXACT_COMMIT_TWO_BYTE_READS',commit=commit,pages_success=True,files=rows,count=len(rows),bytes=sum(r['bytes'] for r in rows),root_exact=True,utc=datetime.now(timezone.utc).isoformat())
    with (HERE/('PUBLIC_'+tag+'.json')).open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['assets','pages']);parser.add_argument('--commit',required=True);parser.add_argument('--tag',default='final');args=parser.parse_args()
    assets(args.commit) if args.stage=='assets' else pages(args.commit,args.tag)
