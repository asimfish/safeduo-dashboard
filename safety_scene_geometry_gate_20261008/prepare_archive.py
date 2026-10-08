import json,hashlib,shutil,re,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent;raw=Path('/mnt/nas/data/lyf/double_hand/safety_scene_geometry_gate_20261008');repo=Path('/home/liyufeng/safeduo-dashboard-object-media-20261006');dest=repo/p.name
assert not dest.exists();dest.mkdir()
for f in p.iterdir():
    if f.is_file() and f.name not in ['AUTO_DELIVERY.log','DELIVERY_STATE.json']:shutil.copy2(f,dest/f.name)
    elif f.name in ['runtime','runtime_camera','runtime_camera_ur','runtime_batch','runtime_batch_probe','source_dependencies','reference_native512','figures','figures_first_layout','videos','panel','browser_local']:
        shutil.copytree(f,dest/f.name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for run in ['matched_scene','fresh_scene','fresh_scene_retry','fresh_scene_batch','batch_probe','batch_pilot','batch_pilot_retry','batch_pilot_ready','camera_replay','camera_replay_ur','replay_source_first_cycle','replay_fresh_first_cycle']:
    shutil.copytree(raw/run,dest/run)
shutil.copytree(repo/'safety_hand_closure_qualification_20261007'/'source_snapshot',dest/'source_snapshot')
entries=[];secret=re.compile(rb'(?:Bearer\s+ct-[0-9a-f]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})')
for f in sorted(dest.rglob('*')):
    if not f.is_file():continue
    b=f.read_bytes();assert len(b)<100*1024*1024,(f,'GitHub blob limit');assert not secret.search(b),('credential pattern',f)
    entries.append(dict(file=str(f.relative_to(dest)),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
manifest=dict(status='PASS_ALL_LOCAL_FILES_HASHED_AND_CREDENTIAL_PATTERN_SCAN',entries=entries,count=len(entries),bytes=sum(x['bytes'] for x in entries),largest=max(x['bytes'] for x in entries),scope='Original native point data, all screenshots including occluded/rejected diagnostic, supplementary replay and all outcome/protocol/runtime files. Manifest self excluded.')
(dest/'MEDIA_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');shutil.copy2(dest/'MEDIA_MANIFEST.json',p/'MEDIA_MANIFEST.json');print(json.dumps({k:v for k,v in manifest.items() if k!='entries'},indent=2))
