"""Single first cell in a fresh process; isolate previous-cell history."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/liyufeng/safeduo')
OUT = Path(__file__).resolve().parent
OLD = OUT.parent / 'safety_latency_forensics_20261003'
cmd = [sys.executable, str(OLD / 'counterfactual_runner.py'), '--ckpt',
       'artifacts/runs/a31b_refine_cross_20260912/model_last.pt', '--env-yaml',
       'duo_env_a31_pending_guard.yaml', '--num-envs', '32', '--duration-s', '10',
       '--seeds', '12', '--amps', '.015', '--flows', 'held_random', '--hold-steps', '90',
       '--methods', 'system0', '--init-jitter-rad', '.3',
       '--actuator-delay-steps', '3', '--causal-trace', '--sphere-trace',
       '--out', str(OUT / 'cold_50'), '--headless', '--log-every', '150']
env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '1', 'OMNI_KIT_ACCEPT_EULA': 'YES',
       'PRIVACY_CONSENT': 'Y', 'PYTHONPATH': 'src', 'OMP_NUM_THREADS': '1',
       'MKL_NUM_THREADS': '1', 'SAFEDUO_FORENSIC_CRITICAL_ROWS': '1'}
with (OUT / 'cold_50.log').open('w') as log:
    subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
