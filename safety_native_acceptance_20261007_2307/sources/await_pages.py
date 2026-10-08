"""Observe an exact Pages commit; never rerun or turn queued jobs into PASS."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,time,urllib.request
HERE=Path(__file__).resolve().parent
API='https://api.github.com/repos/asimfish/safeduo-dashboard/'
def get(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'SafeDuo-CI-observer'}),timeout=30) as r:return json.load(r)
def main():
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--tag',required=True);a=p.parse_args();history=[]
    for n in range(80):
        runs=get(API+'actions/runs?head_sha='+a.commit+'&per_page=30')['workflow_runs']
        pages=[r for r in runs if 'pages' in r['name'].lower() and r['head_sha']==a.commit]
        row=dict(utc=datetime.now(timezone.utc).isoformat(),runs=[{k:r[k] for k in ['id','name','head_sha','head_branch','status','conclusion','html_url']} for r in pages]);history.append(row)
        print('EXACT_PAGES_STATE',[(r['status'],r['conclusion']) for r in pages],flush=True)
        if any(r['conclusion']=='success' for r in pages):
            head=get(API+'branches/main');assert head['commit']['sha']==a.commit and not head['protected'],'main advanced or protection changed'
            with (HERE/('CI_'+a.tag+'.json')).open('x') as f:json.dump(dict(status='PASS_EXACT_PAGES_DEPLOYMENT_SUCCESS',commit=a.commit,observations=history,actual_main_sha=head['commit']['sha'],utc=datetime.now(timezone.utc).isoformat()),f,indent=2)
            return
        if pages and all(r['status']=='completed' for r in pages):raise RuntimeError('exact Pages deployment completed without success: '+str(row))
        time.sleep(15)
    raise TimeoutError('Exact Pages deployment still unavailable; no PASS recorded')
if __name__=='__main__':main()
