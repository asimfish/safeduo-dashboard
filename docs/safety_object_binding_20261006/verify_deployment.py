"""Exact Pages commit, concurrent public SHA readbacks, actual live browser."""
import argparse,json,time,hashlib,urllib.request,concurrent.futures,subprocess,os
from pathlib import Path
R=Path(__file__).resolve().parent
BASE='https://asimfish.github.io/safeduo-dashboard/'
def get(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'SafeDuo-object-evidence'}),timeout=40) as response:return response.read()
def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
    url=BASE+'docs/'+R.name+'/';files=json.loads((a.repo/'docs'/R.name/'PUBLIC_MANIFEST.json').read_text())['files']
    receipt=dict(status='WAITING_EXACT_PAGES_BUILD',commit=a.commit,url=url,owned_files=len(files));dest=R/'delivery.json'
    dest.write_text(json.dumps(receipt,indent=2)+'\n')
    deadline=time.monotonic()+900;ci=None
    while time.monotonic()<deadline:
        runs=json.loads(get('https://api.github.com/repos/asimfish/safeduo-dashboard/actions/runs?head_sha='+a.commit+'&per_page=20'))['workflow_runs']
        ci=next((r for r in runs if r['name']=='pages build and deployment' and r['head_sha']==a.commit),None)
        if ci and ci['status']=='completed':break
        time.sleep(15)
    assert ci and ci['status']=='completed' and ci['conclusion']=='success',ci
    receipt.update(status='READING_PUBLIC_BYTES',ci_id=ci['id']);dest.write_text(json.dumps(receipt,indent=2)+'\n')
    public_manifest=get(url+'PUBLIC_MANIFEST.json?commit='+a.commit);assert public_manifest==(a.repo/'docs'/R.name/'PUBLIC_MANIFEST.json').read_bytes()
    def check(item):
        rel,digest=item;actual=hashlib.sha256(get(url+rel+'?commit='+a.commit)).hexdigest();assert actual==digest,(rel,actual,digest);return rel
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:checked=list(pool.map(check,files.items()))
    local_root=(a.repo/'index.html').read_bytes();assert hashlib.sha256(get(BASE+'index.html?commit='+a.commit)).digest()==hashlib.sha256(local_root).digest()
    env={**os.environ,'PLAYWRIGHT_BROWSERS_PATH':'/tmp/safeduo_dashboard_browser','PYTHONPATH':'/tmp/safeduo_dashboard_browser_tools'}
    subprocess.run(['/usr/bin/python3',str(R/'check_browser.py'),'--url',url,'--out',str(R/'browser_live')],env=env,check=True)
    receipt.update(status='PASS_EXACT_COMMIT_CI_PUBLIC_SHA_LIVE_BROWSER',native_media_files_preverified=151,native_media_commit=json.loads((R/'MEDIA_ARCHIVE.json').read_text())['commit'],ci_url=ci['html_url'],checked_files=len(checked),finished_utc=time.time(),scientific_safety_accepted=False)
    dest.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
