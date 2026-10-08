import json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
p=Path(__file__).resolve().parent;raw=Path('/mnt/nas/data/lyf/double_hand/safety_scene_geometry_gate_20261008');src=raw/'fresh_scene_batch';dest=raw/'replay_fresh_first_cycle';sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest();progress=json.loads((src/'progress.json').read_text());chunks=[x for x in progress['chunks'] if x['last_step']<720];assert len(chunks)==6 and chunks[-1]['last_step']==719;dest.mkdir(exist_ok=False)
for fn in ['native_metadata.json','contact_identities.json']+[x['file'] for x in chunks]:shutil.copy2(src/fn,dest/fn)
for ch in chunks:assert sha(src/ch['file'])==sha(dest/ch['file'])==ch['sha256']
subset=dict(status='PHYSICS_COMPLETE_FIRST_CYCLE_PREFIX_FULL_RUN_ONGOING',steps=720,chunks=chunks,images=[],scope='Source states for render-only UR close-up; not full 12-cycle qualification. Must bind to final native receipt before delivery.')
(dest/'recording_receipt.json').write_text(json.dumps(subset,indent=2)+'\n');r=json.loads((p/'REGISTRATION_CAMERA.json').read_text());r.update(source_raw=str(dest),source_envs=[0],hand_focus='U_R',created_utc=datetime.now(timezone.utc).isoformat(),scope='Render-only U_R focus of frozen fresh native env0 first goal, including close/return. No new physics or safety trials.')
for f in [p/'runtime_camera_ur/native_camera_replay.py',dest/'recording_receipt.json',dest/'native_metadata.json',dest/'contact_identities.json']+[dest/x['file'] for x in chunks]:r['source_sha256'][str(f)]=sha(f)
fn=p/'REGISTRATION_CAMERA_UR.json';assert not fn.exists();fn.write_text(json.dumps(r,indent=2)+'\n');print(dict(status='REGISTERED_BEFORE_RENDER',registration_sha256=sha(fn),frames=60,source_chunk_count=6))
