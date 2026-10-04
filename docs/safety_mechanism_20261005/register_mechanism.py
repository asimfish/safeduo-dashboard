import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'safety_risk_strata_20261004'))
from risk_recipe import validate_bank
old = json.loads((HERE.parent / 'safety_risk_strata_20261004/campaign_plan.json').read_text())
holdout = json.loads((HERE / 'holdout_registration.json').read_text())
nas = Path('/mnt/nas/data/lyf/double_hand/safety_mechanism_20261005')
assert (nas / 'holdout_banks/sampling_summary.json').exists()
helpers = {**old['research_source_sha256'], **holdout['research_source_sha256']}
for name in ['reference_envelope.py', 'mechanism_runner.py', 'MECHANISM_PLAN.md',
             'test_reference_envelope.py', 'register_mechanism.py']:
    helpers[str(HERE / name)] = hashlib.sha256((HERE / name).read_bytes()).hexdigest()


def job(seed, mode, bank, device):
    meta = json.loads((bank / 'metadata.json').read_text())
    with np.load(bank / 'bank.npz') as z:
        validate_bank(z['accepted_q'], z['risk_pair_index'], meta)
    assert meta['risk_quotas'] == [8]*6 and meta['general_count'] == 16
    for name in ('bank.npz', 'metadata.json'):
        helpers[str(bank / name)] = hashlib.sha256((bank / name).read_bytes()).hexdigest()
    method = 'raw' if mode == 'raw' else 'system0'
    argv = [old['jobs'][0]['argv'][0], str(HERE / 'mechanism_runner.py'),
            '--ckpt', old['checkpoint_path'], '--env-yaml', 'duo_env_a31_pending_guard.yaml',
            '--num-envs', '64', '--duration-s', '16', '--seeds', str(seed),
            '--amps', '.05', '--flows', 'risk_burst', '--methods', method,
            '--init-jitter-rad', '0', '--actuator-delay-steps', '6',
            '--device', device, '--headless', '--log-every', '300']
    return dict(id=f'mechanism_{seed}_{mode}', argv=argv, mode=mode,
                expected=dict(flow='risk_burst', seed=seed, method=method,
                              amp=.05, actuator_delay_steps=6),
                expected_args=dict(num_envs=64, duration_s=16., device=device),
                env=dict(SAFEDUO_INITIAL_BANK_NPZ=str(bank / 'bank.npz'),
                         SAFEDUO_MECHANISM='baseline' if mode == 'raw' else mode))


development = [job(60317411, mode,
                   Path('/mnt/nas/data/lyf/double_hand/safety_risk_strata_20261004/risk_banks/60317411'),
                   'cuda:1') for mode in ('box_only', 'envelope_050')]
independent = [job(seed, mode, nas / 'holdout_banks' / str(seed), 'cuda:0')
               for seed in (152684921, 198470327, 237901613)
               for mode in ('raw', 'baseline', 'box_only', 'envelope_050')]
for name, sha in old['source_sha256'].items():
    assert hashlib.sha256((Path(old['cwd']) / name).read_bytes()).hexdigest() == sha, name
for name, sha in helpers.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha, name
for name, jobs, unique in [('development', development, 0), ('holdout', independent, 192)]:
    path = HERE / f'{name}_plan.json'
    assert not path.exists(), 'immutable candidate registration'
    plan = {**copy.deepcopy(old), 'jobs':jobs, 'research_source_sha256':helpers,
            'registered_utc':datetime.now(timezone.utc).isoformat(),
            'output_root':str(nas / f'{name}_cells'), 'planned_cells':len(jobs),
            'planned_windows':len(jobs)*64, 'new_unique_command_windows':unique,
            'design':'frozen .050 reference envelope and box factor; no tuning after development or holdout',
            'candidate':'evaluation-only reachable reference envelope; strict FIFO and original actor'}
    path.write_text(json.dumps(plan, indent=2) + '\n')
print(json.dumps(dict(development_windows=128, holdout_windows=768,
                      new_external_command_windows=192, candidate_frozen=True)))
