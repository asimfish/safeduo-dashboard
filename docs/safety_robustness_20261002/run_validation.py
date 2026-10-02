"""Frozen follow-up cases, executed serially without sharing simulation state."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/liyufeng/safeduo')
OUT = ROOT / 'artifacts/safety_robustness_20261002'
# New random seeds are not selected from failed episodes. All random increments
# remain <= .015 rad, and each arm's 6/7 joints are independently sampled.
JOBS = [
    ('legacy_held', 'duo_env_a31_lag_guard.yaml', ['held_random'], [9, 10], ['system0'], 5, 30),
    ('candidate_random', 'duo_env_a31_priority_guard.yaml', ['uniform_random', 'held_random'],
     [9, 10], ['raw', 'backstop_only', 'system0'], 5, 30),
    ('candidate_regression', 'duo_env_a31_priority_guard.yaml', ['pair_stratified'],
     [0, 1, 2, 3], ['system0'], 5, 30),
    ('candidate_pair_long', 'duo_env_a31_priority_guard.yaml', ['pair_stratified'],
     [0], ['system0'], 10, 30),
    ('candidate_directed', 'duo_env_a31_priority_guard.yaml', ['directed_all'],
     [8], ['system0'], 5, 30),
    ('candidate_held_long', 'duo_env_a31_priority_guard.yaml', ['held_random'],
     [11], ['raw', 'system0'], 10, 90),
]

run_env = {**os.environ, 'OMNI_KIT_ACCEPT_EULA': 'YES', 'PRIVACY_CONSENT': 'Y',
           'CUDA_VISIBLE_DEVICES': '0', 'PYTHONPATH': 'src', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
for name, config, flows, seeds, methods, duration, hold in JOBS:
    if (OUT / name / 'protocol.json').exists():
        status = json.loads((OUT / name / 'protocol.json').read_text())['status']
        if status == 'complete':
            print('ALREADY COMPLETE', name, flush=True)
            continue
        raise RuntimeError(f'{name}: preserve {status} evidence; select a new directory before retry')
    cmd = [sys.executable, '-m', 'safeduo.eval.research_battery', '--ckpt',
           'artifacts/runs/a31b_refine_cross_20260912/model_last.pt', '--env-yaml', config,
           '--num-envs', '32', '--duration-s', str(duration), '--seeds', *map(str, seeds),
           '--amps', *(['.005', '.015'] if name == 'candidate_regression' else ['.015']),
           '--flows', *flows, '--methods', *methods, '--hold-steps', str(hold),
           '--out', str(OUT / name), '--log-every', '150', '--headless']
    print('START', name, flush=True)
    with (OUT / f'{name}.log').open('w') as stream:
        subprocess.run(cmd, cwd=ROOT, env=run_env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    summary = json.loads((OUT / name / 'summary.json').read_text())
    print('DONE', name, [(x['flow'], x['method'], x['seed'], x['violation_episodes'])
                         for x in summary['windows']], flush=True)
    if any(x['method'] == 'system0' and x['violation_episodes'] for x in summary['windows']):
        # Legacy is a comparison arm, so only candidate failures stop promotion.
        if name != 'legacy_held':
            raise RuntimeError(f'{name}: candidate safety regression; preserve traces and diagnose before continuing')
