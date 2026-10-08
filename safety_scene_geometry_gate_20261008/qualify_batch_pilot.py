import json,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent;sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest();res=json.loads((p/'BATCH_PILOT_READY_FULL_RESULTS.json').read_text());mapping=json.loads((p/'BATCH_PILOT_READY_MAPPING_GATE.json').read_text());m=res['methods']['old_v3']
assert res['env_states']==1440 and res['native_cycles']==2 and res['original_images']==0
assert res['max_target_error_rad']<=1e-6 and res['arm_target_max_error_rad']==0
assert mapping['steps']==720 and mapping['original_point_values_compared']>0
assert m['requests']==4 and m['admitted']==4 and m['executed_qualified']==2 and m['aborts']==2
for recovery in res['recoveries']:
 assert recovery['arm']=='U_L' and recovery['last_second_max_n']<=.1 and recovery['last_second_neutral_error_rad']<=.02 and recovery['recovery_s'] is not None
assert all(row['executed_and_qualified'] for row in res['rows'] if row['arm']=='U_R')
r=dict(status='PASS_RECORDER_WITH_EXPECTED_UNSAFE_NEGATIVE_CONTROLS',scientific_safety_status=res['status'],summary=m,scorer_sha256=sha(p/'BATCH_PILOT_READY_FULL_RESULTS.json'),mapping_gate_sha256=sha(p/'BATCH_PILOT_READY_MAPPING_GATE.json'),mapping_points=mapping['original_point_values_compared'],activation='fresh_scene_batch authorized on same frozen 24 references',new_independent_contexts=0)
(p/'BATCH_PILOT_READY_RECORDER_GATE.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
