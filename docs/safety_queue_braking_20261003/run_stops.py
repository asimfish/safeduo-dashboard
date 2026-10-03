"""Two frozen diagnostic interventions, serialized on the available GPU."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/liyufeng/safeduo')
OUT = Path(__file__).resolve().parent
assert (OUT / 'stop_schedule.json').exists()
for mode in ['stored', 'measured']:
    cmd = [sys.executable, str(OUT / 'stop_runner.py'), '--ckpt',
           'artifacts/runs/a31b_refine_cross_20260912/model_last.pt', '--env-yaml',
           'duo_env_a31_pending_guard.yaml', '--num-envs', '32', '--duration-s', '10',
           '--seeds', '12', '--amps', '.015', '--flows', 'held_random', '--hold-steps', '90',
           '--methods', 'system0', '--init-jitter-rad', '.3', '--actuator-delay-steps', '6',
           '--causal-trace', '--sphere-trace', '--out', str(OUT / f'stop_{mode}'),
           '--headless', '--log-every', '150']
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '1', 'OMNI_KIT_ACCEPT_EULA': 'YES',
           'PRIVACY_CONSENT': 'Y', 'PYTHONPATH': 'src', 'OMP_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'SAFEDUO_FORENSIC_CRITICAL_ROWS': '1',
           'SAFEDUO_STOP_SCHEDULE': str(OUT / 'stop_schedule.json'), 'SAFEDUO_STOP_MODE': mode}
    print('START', mode, flush=True)
    with (OUT / f'stop_{mode}.log').open('w') as log:
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('DONE', mode, flush=True)
