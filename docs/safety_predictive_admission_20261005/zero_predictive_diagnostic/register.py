import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
old = json.loads((HERE.parent / 'safety_predictive_admission_20261005/holdout_plan.json').read_text())
helpers = dict(old['research_source_sha256'])
sources = [HERE / n for n in ('PLAN.md', 'zero_predictive_runner.py', 'register.py', 'analyze_zero_predictive.py')]
sources += [HERE.parent / 'safety_zero_command_20261005/zero_runner.py',
            HERE.parent / 'safety_zero_command_20261005/analyze_zero.py',
            HERE.parent / 'safety_predictive_admission_20261005/analyze_predictive.py',
            HERE.parent / 'safety_predictive_admission_20261005/operation_metrics.py']
for path in sources:
    helpers[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
for name, sha in helpers.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha, name
jobs = []
for seed in old['seeds']:
    for mode in ('predictive', 'predictive_envelope'):
        job = copy.deepcopy(next(j for j in old['jobs'] if j['expected']['seed'] == seed and j['mode'] == mode))
        job['id'] = f'zero_{seed}_{mode}'
        job['argv'][1] = str(HERE / 'zero_predictive_runner.py')
        job['argv'][job['argv'].index('--device') + 1] = 'cuda:1'
        job['expected_args']['device'] = 'cuda:1'
        jobs.append(job)
path = HERE / 'campaign_plan.json'; assert not path.exists()
plan = {**copy.deepcopy(old), 'jobs': jobs, 'research_source_sha256': helpers,
        'registered_utc': datetime.now(timezone.utc).isoformat(),
        'output_root': '/mnt/nas/data/lyf/double_hand/safety_zero_predictive_20261005/cells',
        'planned_cells': 6, 'planned_windows': 384, 'new_unique_command_windows': 0,
        'candidate': 'unchanged two predictive factors; all960 external commands exactly zero',
        'modes': ['predictive', 'predictive_envelope'], 'supplementary_not_primary_holdout': True,
        'prior_pressure_partial_outcomes_seen': True, 'prior_zero_and_speed_results_seen': True,
        'own_zero_predictive_outcomes_seen': False}
path.write_text(json.dumps(plan, indent=2) + '\n')
print('Six zero-input new-factor conditions frozen; zero new independent inputs.')
