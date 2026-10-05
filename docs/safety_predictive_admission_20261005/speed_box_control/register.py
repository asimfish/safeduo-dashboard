"""Register this supplemental comparator before any of its outcomes."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
old = json.loads((HERE.parent / 'safety_predictive_admission_20261005/holdout_plan.json').read_text())
helpers = dict(old['research_source_sha256'])
for name in ('speed_box_runner.py', 'PLAN.md', 'register.py', 'analyze_speed.py'):
    helpers[str(HERE / name)] = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
for source in (HERE.parent / 'safety_predictive_admission_20261005/operation_metrics.py',
               HERE.parent / 'safety_mechanism_20261005/analyze_mechanism.py',
               HERE.parent / 'safety_risk_strata_20261004/analyze_risk.py'):
    helpers[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
for name, sha in helpers.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha, name
jobs = []
for seed in old['seeds']:
    job = copy.deepcopy(next(j for j in old['jobs'] if j['expected']['seed'] == seed and j['mode'] == 'raw'))
    job['id'] = f'speed_box_{seed}'
    job['mode'] = 'speed_box_only'
    job['argv'][1] = str(HERE / 'speed_box_runner.py')
    job['argv'][job['argv'].index('--device') + 1] = 'cuda:1'
    job['argv'][job['argv'].index('--methods') + 1] = 'backstop_only'
    job['expected_args']['device'] = 'cuda:1'
    job['expected']['method'] = 'backstop_only'
    job['env'].pop('SAFEDUO_ADMISSION_MODE')
    jobs.append(job)
path = HERE / 'campaign_plan.json'
assert not path.exists()
plan = {**copy.deepcopy(old), 'jobs': jobs, 'research_source_sha256': helpers,
        'registered_utc': datetime.now(timezone.utc).isoformat(),
        'output_root': '/mnt/nas/data/lyf/double_hand/safety_speed_box_control_20261005/cells',
        'planned_cells': 3, 'planned_windows': 192, 'new_unique_command_windows': 0,
        'candidate': 'original speed box only; all actor gating and geometry projection disabled',
        'design': 'existing banks and commands; no post-outcome selection', 'modes': ['speed_box_only'],
        'supplementary_not_primary_holdout': True, 'prior_pressure_partial_outcomes_seen': True,
        'own_supplement_outcomes_seen': False}
path.write_text(json.dumps(plan, indent=2) + '\n')
print('Three speed-box-only conditions registered; no new independent random trajectories.')
