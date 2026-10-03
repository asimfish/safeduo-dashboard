"""Serial observational replays; no modifications to the frozen safety policy."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path('/home/liyufeng/safeduo')
OUT = Path(__file__).resolve().parent
OLD = ROOT / 'artifacts/safety_perturbation_20261003'
CORE = ['src/safeduo/safety/backstop.py', 'src/safeduo/envs/duo_env.py',
        'src/safeduo/safety/geometry.py', 'src/safeduo/safety/sphere_distance.py']
reference = json.loads((OLD / 'pending_final_pilot/protocol.json').read_text())
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for name in CORE:
    assert digest(ROOT / name) == reference['source_sha256'][name], name
actor = ROOT / 'artifacts/runs/a31b_refine_cross_20260912/model_last.pt'
assert digest(actor) == reference['checkpoint_sha256']
env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '1', 'OMNI_KIT_ACCEPT_EULA': 'YES',
       'PRIVACY_CONSENT': 'Y', 'PYTHONPATH': 'src', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
jobs = [('delayed', 'held_random', 12, 10, '.3', '6'),
        ('nominal_table', 'l1_full', 8, 5, '0', '0')]
manifest = {'observer': 'trace_runner.py', 'observer_sha256': digest(OUT / 'trace_runner.py'),
            'core_sources': {n: digest(ROOT / n) for n in CORE},
            'checkpoint_sha256': digest(actor), 'count_as_new_independent_windows': False,
            'intervention': 'none; all original project outputs returned unchanged', 'jobs': jobs}
(OUT / 'observer_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
for name, flow, seed, duration, jitter, delay in jobs:
    cmd = [sys.executable, str(OUT / 'trace_runner.py'), '--ckpt', str(actor),
           '--env-yaml', 'duo_env_a31_pending_guard.yaml', '--num-envs', '32',
           '--duration-s', str(duration), '--seeds', str(seed), '--amps', '.015',
           '--flows', flow, '--hold-steps', '90', '--methods', 'system0',
           '--init-jitter-rad', jitter, '--actuator-delay-steps', delay,
           '--causal-trace', '--kinematic-trace', '--out', str(OUT / name),
           '--headless', '--log-every', '150']
    print('START', name, flush=True)
    with (OUT / f'{name}.log').open('w') as log:
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    data = json.loads((OUT / name / 'episodes.json').read_text())
    print('DONE', name, 'violations', sum(e['violation'] for e in data), flush=True)
