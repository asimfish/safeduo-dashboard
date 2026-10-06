"""Verify the exact Pages build, all owned public bytes and live browser journeys."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,time,urllib.request
HERE=Path(__file__).resolve().parent
BASE='https://asimfish.github.io/safeduo-dashboard/'
def get(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'SafeDuo-evidence-check'}),timeout=30) as r:return r.read()
def main():
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--repo',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((a.repo/'docs'/HERE.name/'PUBLIC_MANIFEST.json').read_text())['files']
    receipt=dict(status='WAITING_FOR_EXACT_PAGES_BUILD',commit=a.commit,owned_files=len(manifest),public_url=BASE+'docs/'+HERE.name+'/',new_protocol_physics=0)
    dest=HERE/'delivery.json';dest.write_text(json.dumps(receipt,indent=2)+'\n')
    deadline=time.monotonic()+900;ci=None
    while time.monotonic()<deadline:
        runs=json.loads(get('https://api.github.com/repos/asimfish/safeduo-dashboard/actions/runs?head_sha='+a.commit+'&per_page=20'))['workflow_runs']
        ci=next((r for r in runs if r['name']=='pages build and deployment' and r['head_sha']==a.commit),None)
        if ci and ci['status']=='completed':break
        time.sleep(15)
    assert ci and ci['status']=='completed' and ci['conclusion']=='success',ci
    url=receipt['public_url']
    for rel,h in manifest.items():assert hashlib.sha256(get(url+rel)).hexdigest()==h,rel
    env={**os.environ,'PLAYWRIGHT_BROWSERS_PATH':'/tmp/safeduo_dashboard_browser','PYTHONPATH':'/tmp/safeduo_dashboard_browser_tools'}
    subprocess.run(['/usr/bin/python3',str(HERE/'check_browser.py'),'--url',url,'--out',str(HERE/'browser_live')],env=env,check=True)
    receipt.update(status='PASS_EXACT_COMMIT_CI_PUBLIC_BYTES_LIVE_BROWSER',ci_run_id=ci['id'],ci_url=ci['html_url'],finished_utc=time.time(),scope='design and retrospective evidence delivery, not comprehensive physical acceptance')
    dest.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
