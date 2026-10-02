"""Regression gate for the combined self/table lag protection."""
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/home/liyufeng/safeduo')
OUT=ROOT/'artifacts/safety_refine_20261002'
jobs=[
    ('combined_matrix32',5,[0,1,2,3],[.005,.015],['pair_stratified']),
    ('combined_long',10,[0],[.015],['pair_stratified']),
    ('combined_directed',5,[8],[.015],['directed_all']),
    ('combined_random',10,[8],[.005,.015],['uniform_random']),
]
env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES','PRIVACY_CONSENT':'Y',
     'CUDA_VISIBLE_DEVICES':'0','PYTHONPATH':'src'}
for name,duration,seeds,amps,flows in jobs:
    cmd=[sys.executable,'-m','safeduo.eval.research_battery','--ckpt',
         'artifacts/runs/a31b_refine_cross_20260912/model_last.pt','--env-yaml','duo_env_a31_lag_guard.yaml',
         '--num-envs','32','--duration-s',str(duration),'--seeds',*map(str,seeds),
         '--amps',*map(str,amps),'--flows',*flows,'--methods','system0','--out',str(OUT/name),
         '--log-every','300','--headless']
    print('START',name,flush=True)
    with (OUT/(name+'.log')).open('w') as stream:
        subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    print('DONE',name,flush=True)
