"""Fixed critical-only rule, new seed and four registered perturbation cells."""
import os
from pathlib import Path
import subprocess
import sys


ROOT=Path('/home/liyufeng/safeduo')
OUT=Path(__file__).resolve().parent
cmd=[sys.executable,str(OUT/'counterfactual_runner.py'),'--ckpt',
    'artifacts/runs/a31b_refine_cross_20260912/model_last.pt',
    '--env-yaml','duo_env_a31_pending_guard.yaml','--num-envs','32',
    '--duration-s','5','--seeds','14','--amps','.015','--flows','held_random',
    '--hold-steps','30','--methods','system0','--init-jitter-rad','.1','.3',
    '--actuator-delay-steps','0','6','--causal-trace','--sphere-trace',
    '--out',str(OUT/'critical_holdout'),'--headless','--log-every','150']
env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','OMNI_KIT_ACCEPT_EULA':'YES',
    'PRIVACY_CONSENT':'Y','PYTHONPATH':'src','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
    'SAFEDUO_FORENSIC_CRITICAL_ROWS':'1'}
with (OUT/'critical_holdout.log').open('w') as log:
    subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
