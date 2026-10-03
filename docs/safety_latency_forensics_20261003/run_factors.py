"""Six preregistered single-factor and combined replays; no default promotion."""
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT=Path('/home/liyufeng/safeduo')
OUT=Path(__file__).resolve().parent
env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','OMNI_KIT_ACCEPT_EULA':'YES',
     'PRIVACY_CONSENT':'Y','PYTHONPATH':'src','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
factors=[('debit',True,False),('critical',False,True),('both',True,True)]
scenarios=[('nominal','l1_full',8,5,'0','0'),('delayed','held_random',12,10,'.3','6')]
for scenario,flow,seed,duration,jitter,delay in scenarios:
    for name,debit,critical in factors:
        run=f'{scenario}_{name}'
        cmd=[sys.executable,str(OUT/'counterfactual_runner.py'),'--ckpt',
            'artifacts/runs/a31b_refine_cross_20260912/model_last.pt','--env-yaml',
            'duo_env_a31_pending_debit_guard.yaml' if debit else 'duo_env_a31_pending_guard.yaml',
            '--num-envs','32','--duration-s',str(duration),'--seeds',str(seed),'--amps','.015',
            '--flows',flow,'--hold-steps','90','--methods','system0','--init-jitter-rad',jitter,
            '--actuator-delay-steps',delay,'--causal-trace','--sphere-trace','--out',str(OUT/run),
            '--headless','--log-every','150']
        print('START',run,flush=True)
        with (OUT/f'{run}.log').open('w') as log:
            subprocess.run(cmd,cwd=ROOT,env={**env,'SAFEDUO_FORENSIC_CRITICAL_ROWS':str(int(critical))},
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        data=json.loads((OUT/run/'episodes.json').read_text())
        print('DONE',run,'violations',sum(e['violation'] for e in data),flush=True)
