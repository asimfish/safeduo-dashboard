from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import requests,json,hashlib,subprocess
p=Path(__file__).resolve().parent;repo=Path('/home/liyufeng/safeduo-dashboard-benchmark-protocol-20261006');commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip();base='https://asimfish.github.io/safeduo-dashboard/';paths=['index.html']+[str(f.relative_to(repo)) for f in sorted((repo/'docs/safety_hand_closure_qualification_20261007').iterdir()) if f.is_file()]
resp=requests.get('https://api.github.com/repos/asimfish/safeduo-dashboard/actions/runs',params={'head_sha':commit,'per_page':10},timeout=25);resp.raise_for_status();runs=[dict(id=v['id'],name=v['name'],status=v['status'],conclusion=v['conclusion'],head_sha=v['head_sha']) for v in resp.json()['workflow_runs']];(p/'PAGES_STATUS.json').write_text(json.dumps(dict(commit=commit,runs=runs),indent=2)+'\n');assert any(v['head_sha']==commit and v['name']=='pages build and deployment' and v['status']=='completed' and v['conclusion']=='success' for v in runs),runs
def check(path):
 response=requests.get(base+path,params={'verify':commit},headers={'Cache-Control':'no-cache'},timeout=35);response.raise_for_status();actual=response.content;expected=(repo/path).read_bytes();assert actual==expected,path;return dict(file=path,bytes=len(actual),sha256=hashlib.sha256(actual).hexdigest())
with ThreadPoolExecutor(max_workers=5) as pool:files=list(pool.map(check,paths))
out=dict(status='PASS_EXACT_PUBLIC_FILES_AND_PAGES_COMMIT',commit=commit,files=files,pages_runs=runs,checked_utc=datetime.now(timezone.utc).isoformat(),url=base+'docs/safety_hand_closure_qualification_20261007/');(p/'PUBLIC_DELIVERY.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='files'},indent=2))
