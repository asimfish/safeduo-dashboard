import json,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent;raw=Path('/mnt/nas/data/lyf/double_hand/safety_scene_geometry_gate_20261008');sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest();complete=json.loads((raw/'fresh_scene_batch'/'recording_receipt.json').read_text());subset=json.loads((raw/'replay_fresh_first_cycle'/'recording_receipt.json').read_text());render=json.loads((raw/'camera_replay_ur'/'recording_receipt.json').read_text());checks=[]
for a in subset['chunks']:
 b=next(x for x in complete['chunks'] if x['file']==a['file']);assert a==b;assert sha(raw/'fresh_scene_batch'/a['file'])==sha(raw/'replay_fresh_first_cycle'/a['file'])==a['sha256'];checks.append(a)
assert len(checks)==6 and checks[-1]['last_step']==719
assert render['new_physics_steps']==0 and render['new_safety_trials']==0 and len(render['images'])==60
assert render['source_receipt_sha256']==sha(raw/'replay_fresh_first_cycle'/'recording_receipt.json')
for im in render['images']:assert sha(raw/'camera_replay_ur'/im['file'])==im['sha256']
(p/'CAMERA_UR_SOURCE_CLOSURE.json').write_text(json.dumps(dict(status='PASS_UR_REPLAY_BOUND_TO_FINAL_NATIVE_RECEIPT',final_native_receipt_sha256=sha(raw/'fresh_scene_batch'/'recording_receipt.json'),subset_receipt_sha256=sha(raw/'replay_fresh_first_cycle'/'recording_receipt.json'),render_receipt_sha256=sha(raw/'camera_replay_ur'/'recording_receipt.json'),chunks=checks,images=60,new_physics_steps=0,new_safety_trials=0),indent=2)+'\n')
