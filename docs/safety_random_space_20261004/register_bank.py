import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
base = json.loads((HERE / 'campaign_plan_dynamic.json').read_text())
path = HERE / 'bank_registration.json'
assert not path.exists()
registration = dict(
    registered_utc=datetime.now(timezone.utc).isoformat(),
    source_sha256=base['source_sha256'], checkpoint_path=base['checkpoint_path'],
    checkpoint_sha256=base['checkpoint_sha256'],
    seeds=[13447771, 27180353, 48921161], batch_size=64,
    max_batches_per_seed=200, sampled_joint_fraction=[.025, .975],
    selection='first 64 in sampling order, all nonexempt margins >=0.1mm and no initial violation',
    policy_outcomes_used=False, gpu=1,
    output_root='/mnt/nas/data/lyf/double_hand/safety_random_space_20261004/pose_banks',
    research_source_sha256={str(HERE / name): hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                           for name in ['FEASIBLE_PLAN.md', 'pose_bank.py', 'wide_random.py', 'register_bank.py']})
for name, sha in registration['source_sha256'].items():
    assert hashlib.sha256((Path(base['cwd']) / name).read_bytes()).hexdigest() == sha, name
assert hashlib.sha256(Path(base['checkpoint_path']).read_bytes()).hexdigest() == base['checkpoint_sha256']
path.write_text(json.dumps(registration, indent=2) + '\n')
print(json.dumps({'registered': str(path), 'maximum_candidates': 38400, 'required_selected': 192}))
