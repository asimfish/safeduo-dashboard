import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
old = json.loads((HERE.parent / 'safety_predictive_admission_20261005/holdout_plan.json').read_text())
helpers = dict(json.loads((HERE / 'campaign_plan.json').read_text())['research_source_sha256'])
for name in ('COMPARISON_PLAN.md', 'register_comparison.py', 'analyze_same_device.py'):
    helpers[str(HERE / name)] = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
for name, sha in helpers.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha, name
jobs = []
for seed in old['seeds']:
    for mode in ('raw', 'baseline'):
        job = copy.deepcopy(next(j for j in old['jobs'] if j['expected']['seed'] == seed and j['mode'] == mode))
        job['id'] = f'same_device_{seed}_{mode}'
        job['argv'][1] = str(HERE.parent / 'safety_mechanism_20261005/mechanism_runner.py')
        job['argv'][job['argv'].index('--device') + 1] = 'cuda:1'
        job['expected_args']['device'] = 'cuda:1'
        job['env'].pop('SAFEDUO_ADMISSION_MODE')
        job['env']['SAFEDUO_MECHANISM'] = 'baseline'
        jobs.append(job)
path = HERE / 'comparison_plan.json'; assert not path.exists()
plan = {**copy.deepcopy(old), 'jobs': jobs, 'research_source_sha256': helpers,
        'registered_utc': datetime.now(timezone.utc).isoformat(),
        'output_root': '/mnt/nas/data/lyf/double_hand/safety_speed_box_control_20261005/comparison_cells',
        'planned_cells': 6, 'planned_windows': 384, 'new_unique_command_windows': 0,
        'candidate': 'unchanged raw and original System0 repeated on speed-box control GPU1',
        'modes': ['raw', 'baseline'], 'supplementary_not_primary_holdout': True,
        'prior_pressure_partial_outcomes_seen': True, 'speed_only_campaign_already_completed': True,
        'speed_only_analyzed_outcomes_seen': False, 'own_comparison_outcomes_seen': False}
path.write_text(json.dumps(plan, indent=2) + '\n')
print('Six same-GPU comparator conditions frozen; 384 repeated windows, zero new inputs.')
