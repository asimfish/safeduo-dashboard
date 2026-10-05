"""Freeze executable versions and fresh bank identities before any candidate run."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'safety_mechanism_20261005_causal_obs'
RAW = Path('/mnt/nas/data/lyf/double_hand') / HERE.name


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    if path.exists():
        raise ValueError('registration is immutable: ' + str(path))
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def build(jobs, root, scope):
    old = json.loads((OLD / 'development_plan_v2.json').read_text())
    helpers = dict(old['research_source_sha256'])
    for name in ['DESIGN.json', 'register_campaigns_v2.py', 'guard_runner_v4.py',
                 'full_finite_guard.py', 'reference_envelope.py', 'projection_diagnostics.py']:
        helpers[str(HERE / name)] = sha(HERE / name)
    for job in jobs:
        bank = Path(job['env']['SAFEDUO_INITIAL_BANK_NPZ'])
        for path in [bank, bank.with_name('metadata.json')]:
            helpers[str(path)] = sha(path)
    return dict(cwd=old['cwd'], output_root=str(root), continue_after_child_failure=True,
        env=old['env'], source_sha256=old['source_sha256'], research_source_sha256=helpers,
        checkpoint_path=old['checkpoint_path'], checkpoint_sha256=old['checkpoint_sha256'],
        jobs=jobs, registered_utc=datetime.now(timezone.utc).isoformat(), scope=scope,
        sources_frozen=True, production_promoted=False, actual_cuda_device='cuda:0' if 'holdout' in scope else 'cuda:1')


def job(mode, seed, bank, device):
    old = json.loads((OLD / 'development_plan_v2.json').read_text())['jobs'][0]
    argv = list(old['argv'])
    argv[1] = str(HERE / 'guard_runner_v4.py')
    argv[argv.index('--seeds') + 1] = str(seed)
    argv[argv.index('--device') + 1] = device
    return dict(id=f'{mode}_{seed}', argv=argv,
        expected={**old['expected'], 'seed': seed},
        expected_args={**old['expected_args'], 'device': device},
        env=dict(SAFEDUO_INITIAL_BANK_NPZ=str(bank), SAFEDUO_JOINT_MODE=mode))


if __name__ == '__main__':
    design = json.loads((HERE / 'DESIGN.json').read_text())
    bank_root = RAW / 'banks'
    summary = json.loads((bank_root / 'sampling_summary.json').read_text())
    for row in design['rows']:
        meta = json.loads((bank_root / str(row['initial_seed']) / 'metadata.json').read_text())
        assert meta['status'] == 'complete', 'bank quota failure retained; do not invent replacement seeds'
    dev_bank = Path('/mnt/nas/data/lyf/double_hand/safety_risk_strata_20261004/risk_banks/60317411/bank.npz')
    dev = build([job(mode, 60317411, dev_bank, 'cuda:1') for mode in ['baseline_guard', 'admission_guard']],
                RAW / 'development_v2', 'full16s development equivalence only; no new initial/tape coverage')
    hold = build([job(mode, row['command_seed'], bank_root / str(row['initial_seed']) / 'bank.npz', 'cuda:0')
                  for row in design['rows'] for mode in design['modes']],
                 RAW / 'holdout', 'fresh initial+tape holdout;2x2 fixed factors;192 cases768 method windows')
    write(HERE / 'development_plan_v2.json', dev)
    write(HERE / 'holdout_plan_v2.json', hold)
    write(HERE / 'source_freeze_v2.json', dict(registered_utc=hold['registered_utc'],
        dev_plan_sha256=sha(HERE / 'development_plan_v2.json'), holdout_plan_sha256=sha(HERE / 'holdout_plan_v2.json'),
        bank_summary_sha256=sha(bank_root / 'sampling_summary.json'), all_parameters_design_sha256=sha(HERE / 'DESIGN.json'),
        prior_failures_preserved=['fresh_bank.py', 'bank_driver.log', 'bank_registration.json'],
        raw_proposal_factor_unchanged=True, shadow_reference_passive=True, J_not_fully_persisted=True))
    print('FROZEN 2 development cells +12 holdout cells; no outcomes used', flush=True)
