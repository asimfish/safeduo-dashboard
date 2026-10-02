"""Frozen, serial GPU validation jobs; stop immediately on a failed campaign."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/liyufeng/safeduo')
OUT = ROOT / 'artifacts/safety_refine_20261002'
JOBS = [
    ('candidate_matrix64', 'duo_env_a31_self_la15.yaml', 64, 5, [4, 5, 6, 7], [.005, .015], ['pair_stratified']),
    ('baseline_long', 'duo_env_a31.yaml', 32, 10, [0], [.015], ['pair_stratified']),
    ('candidate_long', 'duo_env_a31_self_la15.yaml', 32, 10, [0], [.015], ['pair_stratified']),
    ('baseline_other', 'duo_env_a31.yaml', 32, 5, [8], [.015], ['l1_full', 'directed_all']),
    ('candidate_other', 'duo_env_a31_self_la15.yaml', 32, 5, [8], [.015], ['l1_full', 'directed_all']),
    ('baseline_random', 'duo_env_a31.yaml', 32, 10, [8], [.005, .015], ['uniform_random']),
    ('candidate_random', 'duo_env_a31_self_la15.yaml', 32, 10, [8], [.005, .015], ['uniform_random']),
]
env = {**os.environ, 'OMNI_KIT_ACCEPT_EULA': 'YES', 'PRIVACY_CONSENT': 'Y',
       'CUDA_VISIBLE_DEVICES': '0', 'PYTHONPATH': 'src'}
for name, cfg, n, duration, seeds, amps, flows in JOBS:
    cmd = [sys.executable, '-m', 'safeduo.eval.research_battery', '--ckpt',
           'artifacts/runs/a31b_refine_cross_20260912/model_last.pt', '--env-yaml', cfg,
           '--num-envs', str(n), '--duration-s', str(duration), '--seeds', *map(str, seeds),
           '--amps', *map(str, amps), '--flows', *flows, '--methods', 'system0',
           '--out', str(OUT / name), '--log-every', '300', '--headless']
    print('START', name, flush=True)
    with (OUT / (name + '.log')).open('w') as stream:
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print('DONE', name, flush=True)
