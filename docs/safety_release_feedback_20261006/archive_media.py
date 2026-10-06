import argparse,json,hashlib,shutil
from pathlib import Path
R=Path(__file__).resolve().parent;RAW=Path('/mnt/nas/data/lyf/double_hand/safety_release_feedback_20261006')
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args();dest=a.repo/'safety_release_feedback_20261006';dest.mkdir(exist_ok=False)
for b in range(2):
    folder=RAW/f'block_{b}';rec=json.loads((folder/'recording_receipt.json').read_text());out=dest/f'block_{b}';out.mkdir()
    items=[i['file'] for i in rec['images']]+[c['file'] for c in rec['chunks']]+['native_initial.json','recording_receipt.json','bound_reference.npz','native_limits_before_planning.json']
    for rel in items:
        target=out/rel;target.parent.mkdir(exist_ok=True);shutil.copyfile(folder/rel,target)
shutil.copyfile(R/'timeseries.csv',dest/'timeseries.csv')
files={str(p.relative_to(a.repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(dest.rglob('*')) if p.is_file()}
manifest=dict(scope='Every completed native chunk and original image of both blocks; prior experiments on this archive branch remain intact.',files=files,bytes=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()),final_trials=0)
(dest/'MEDIA_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2));print('files',len(files))
