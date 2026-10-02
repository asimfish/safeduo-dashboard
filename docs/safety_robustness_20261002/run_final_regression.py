"""Final frozen implementation: target bounds plus feasible Dykstra stopping."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/home/liyufeng/safeduo')
OUT=ROOT/'artifacts/safety_robustness_20261002'
jobs=[('bounded_final_regression',['pair_stratified'],[0,1,2,3],[.005,.015],5)]
env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','OMNI_KIT_ACCEPT_EULA':'YES','PRIVACY_CONSENT':'Y',
     'PYTHONPATH':'src','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
for name,flows,seeds,amps,duration in jobs:
    cmd=[sys.executable,'-m','safeduo.eval.research_battery','--ckpt',
         'artifacts/runs/a31b_refine_cross_20260912/model_last.pt','--env-yaml','duo_env_a31_bounded_guard.yaml',
         '--num-envs','32','--duration-s',str(duration),'--flows',*flows,'--seeds',*map(str,seeds),
         '--amps',*map(str,amps),'--methods','system0','--out',str(OUT/name),'--headless','--log-every','150']
    print('START',name,flush=True)
    with (OUT/f'{name}.log').open('w') as f:
        subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    s=json.loads((OUT/name/'summary.json').read_text())
    print('DONE',name,[(w['seed'],w['amp'],w['violation_episodes']) for w in s['windows']],flush=True)
    if any(w['violation_episodes'] for w in s['windows']):
        raise RuntimeError('candidate safety regression; preserve full evidence')
