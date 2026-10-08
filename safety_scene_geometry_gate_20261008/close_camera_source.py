import json,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent;raw=Path('/mnt/nas/data/lyf/double_hand/safety_scene_geometry_gate_20261008');sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
complete=json.loads((raw/'matched_scene'/'recording_receipt.json').read_text());subset=json.loads((raw/'replay_source_first_cycle'/'recording_receipt.json').read_text());records=[]
for a in subset['chunks']:
 b=next(x for x in complete['chunks'] if x['file']==a['file']);assert a==b;assert sha(raw/'matched_scene'/a['file'])==sha(raw/'replay_source_first_cycle'/a['file'])==a['sha256'];records.append(a)
assert len(records)==6 and records[-1]['last_step']==719
(p/'CAMERA_SOURCE_CLOSURE.json').write_text(json.dumps(dict(status='PASS_REPLAY_SOURCE_BOUND_TO_FINAL_MATCHED_RECEIPT',final_native_receipt_sha256=sha(raw/'matched_scene'/'recording_receipt.json'),subset_receipt_sha256=sha(raw/'replay_source_first_cycle'/'recording_receipt.json'),chunks=records,new_physics_steps=0,new_safety_trials=0),indent=2)+'\n')
