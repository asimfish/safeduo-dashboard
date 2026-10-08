"""Publish only the owned asset branch, using bounded additive fast-forward pushes."""
from pathlib import Path
import subprocess,json,datetime,urllib.request,urllib.error
H=Path(__file__).resolve().parent;W='/mnt/nas/data/lyf/double_hand/safeduo_dashboard_native_assets_20261007'
BRANCH='exp/native-acceptance-assets-20261007'
def main():
    assert json.loads((H/'local_browser.json').read_text())['status']=='PASS_ACTUAL_CHROMIUM_NATIVE_EVIDENCE_JOURNEYS'
    plan=json.loads((H/'ASSET_GIT_COMMITS.json').read_text());previous=None;rows=[]
    for item in plan['commits']:
        try:
            with urllib.request.urlopen(urllib.request.Request('https://api.github.com/repos/asimfish/safeduo-dashboard/branches/'+BRANCH,headers={'User-Agent':'SafeDuo-publisher'}),timeout=30) as r:state=json.load(r)
        except urllib.error.HTTPError as e:
            assert e.code==404 and previous is None;state=None
        if state is not None:assert not state['protected'] and state['commit']['sha']==previous
        argv=['git','-c','core.filemode=false','-C',W,'push','--porcelain','origin',item['commit']+':refs/heads/'+BRANCH]
        run=subprocess.run(argv,capture_output=True,text=True);assert run.returncode==0,run.stderr
        actual=subprocess.check_output(['git','-c','core.filemode=false','-C',W,'ls-remote','origin','refs/heads/'+BRANCH],text=True).split()[0];assert actual==item['commit']
        rows.append(dict(batch=item['batch'],commit=actual,actual_exit=run.returncode,source_bytes=item['source_bytes']))
        previous=actual;print('BOUNDED_ASSET_PUSH',rows[-1],flush=True)
    with (H/'PANEL_ASSET_PUSH.json').open('x') as f:json.dump(dict(status='PASS_ALL_OWNED_ADDITIVE_ASSET_PUSHES',branch=BRANCH,commits=rows,final_commit=previous,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),f,indent=2);f.write('\n')
if __name__=='__main__':main()
