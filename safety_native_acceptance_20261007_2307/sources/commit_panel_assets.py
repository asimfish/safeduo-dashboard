"""Commit the owned orphan asset branch in bounded additive batches."""
from pathlib import Path
import subprocess,json,datetime
H=Path(__file__).resolve().parent
W=Path('/mnt/nas/data/lyf/double_hand/safeduo_dashboard_native_assets_20261007');A=W/H.name
def git(*args):return subprocess.check_output(['git','-c','core.filemode=false','-C',str(W),*args],text=True).strip()
def main():
    assert json.loads((H/'PANEL_ASSET_SUPPLEMENT.json').read_text())['status']=='PASS_CLOSED_PRECOMMIT_ASSET_SUPPLEMENT'
    assert git('branch','--show-current')=='exp/native-acceptance-assets-20261007'
    assert git('status','--porcelain')=='?? '+H.name+'/'
    files=sorted(p for p in A.rglob('*') if p.is_file());batches=[];current=[];size=0
    for p in files:
        n=p.stat().st_size
        if current and size+n>700*1024**2:batches.append((current,size));current=[];size=0
        current.append(p);size+=n
    if current:batches.append((current,size))
    rows=[]
    for i,(paths,n) in enumerate(batches):
        spec=H/f'asset_stage_batch_{i}.nul'
        with spec.open('xb') as f:
            for p in paths:f.write(str(p.relative_to(W)).encode()+b'\0')
        git('add','--pathspec-from-file='+str(spec),'--pathspec-file-nul')
        git('commit','-m',f'Archive native acceptance evidence {i+1}/{len(batches)}: original bytes and complete finite verdict')
        row=dict(batch=i,commit=git('rev-parse','HEAD'),files=len(paths),source_bytes=n);rows.append(row);print('ASSET_COMMIT',row,flush=True)
    assert git('status','--porcelain')==''
    out=dict(status='PASS_LOCAL_BOUNDED_ASSET_COMMITS',branch='exp/native-acceptance-assets-20261007',commits=rows,final_commit=rows[-1]['commit'],files=len(files),bytes=sum(p.stat().st_size for p in files),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with (H/'ASSET_GIT_COMMITS.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
if __name__=='__main__':main()
