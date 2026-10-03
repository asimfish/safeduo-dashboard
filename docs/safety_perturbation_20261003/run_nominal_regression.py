"""Frozen stored-target candidate: preserve old critical zero-jitter cases."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path('/home/liyufeng/safeduo')
OUT=ROOT/'artifacts/safety_perturbation_20261003'
env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','OMNI_KIT_ACCEPT_EULA':'YES','PRIVACY_CONSENT':'Y',
     'PYTHONPATH':'src','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
for name,flow,seed,duration in [('pending_nominal_long','held_random',11,10),('pending_nominal_l1','l1_full',8,5)]:
    cmd=[sys.executable,'-m','safeduo.eval.research_battery','--ckpt',
        'artifacts/runs/a31b_refine_cross_20260912/model_last.pt','--env-yaml','duo_env_a31_pending_guard.yaml',
        '--num-envs','32','--duration-s',str(duration),'--seeds',str(seed),'--amps','.015',
        '--flows',flow,'--hold-steps','90','--methods','system0','--out',str(OUT/name),
        '--headless','--log-every','150']
    print('START',name,flush=True)
    with (OUT/f'{name}.log').open('w') as f:
        subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    result=json.loads((OUT/name/'episodes.json').read_text())
    print('DONE',name,'violations',sum(e['violation'] for e in result),flush=True)
