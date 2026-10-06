import argparse,json,hashlib,urllib.request,concurrent.futures,datetime
from pathlib import Path
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
base='https://asimfish.github.io/safeduo-dashboard/';d=a.repo/'docs/safety_release_feedback_20261006'
def get(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'SafeDuo-release-delivery'}),timeout=45) as r:return r.read()
manifest=json.loads((d/'PUBLIC_MANIFEST.json').read_text())
assert get(base+'docs/safety_release_feedback_20261006/PUBLIC_MANIFEST.json')==(d/'PUBLIC_MANIFEST.json').read_bytes()
def check(kv):
    rel,digest=kv;assert hashlib.sha256(get(base+'docs/safety_release_feedback_20261006/'+rel)).hexdigest()==digest,rel;return rel
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:checked=list(pool.map(check,manifest['files'].items()))
assert get(base+'index.html')==(a.repo/'index.html').read_bytes()
api='https://api.github.com/repos/asimfish/safeduo-dashboard/actions/runs?per_page=30'
runs=json.loads(get(api))['workflow_runs'];match=[r for r in runs if r['head_sha']==a.commit and r['name']=='pages build and deployment'];assert match and any(r['conclusion']=='success' for r in match),[(r['status'],r['conclusion']) for r in match]
receipt=dict(status='PASS_EXACT_PUBLIC_BYTES_AND_PAGES_COMMIT',commit=a.commit,checked_files=len(checked),main_byteexact=True,pages_runs=[dict(id=r['id'],status=r['status'],conclusion=r['conclusion']) for r in match],url=base+'docs/safety_release_feedback_20261006/',verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(R/'PUBLIC_DELIVERY.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
