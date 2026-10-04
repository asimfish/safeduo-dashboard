"""Freeze the full design before simulation; no selection from outcome metrics."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path('/home/liyufeng/safeduo')
OUT=Path(__file__).resolve().parent
CELLS=Path('/mnt/nas/data/lyf/double_hand/safety_random_space_20261004/cells')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (OUT/'campaign_plan.json').exists(),'existing preregistration must be preserved'
ckpt=ROOT/'artifacts/runs/a31b_refine_cross_20260912/model_last.pt'
jobs=[]
for kind in ['wide_iid','wide_burst','global_burst']:
    for seed in [731923,2048171,9987031]:
        for method in ['raw','system0']:
            args=[sys.executable,str(OUT/'wide_runner.py'),'--ckpt',str(ckpt),
                  '--env-yaml','duo_env_a31_pending_guard.yaml','--num-envs','64',
                  '--duration-s','15','--seeds',str(seed),'--amps','.05','--flows',kind,
                  '--methods',method,'--init-jitter-rad','0','--actuator-delay-steps','6',
                  '--headless','--log-every','300']
            jobs.append(dict(id=f'{kind}_{seed}_{method}',argv=args,
                             expected=dict(flow=kind,seed=seed,method=method,amp=.05,actuator_delay_steps=6),
                             expected_args=dict(num_envs=64,duration_s=15.)))
source=sorted((ROOT/'src/safeduo').rglob('*.py'))+sorted((ROOT/'src/safeduo/configs').glob('*.yaml'))
plan=dict(cwd=str(ROOT),output_root=str(CELLS),continue_after_child_failure=True,
          env={'CUDA_VISIBLE_DEVICES':'0','OMNI_KIT_ACCEPT_EULA':'YES','PRIVACY_CONSENT':'Y',
               'PYTHONPATH':'src','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},jobs=jobs,
          checkpoint_path=str(ckpt),checkpoint_sha256=sha(ckpt),
          source_sha256={str(p.relative_to(ROOT)):sha(p) for p in source},
          research_source_sha256={str(p):sha(p) for p in [*OUT.glob('*.py'),OUT/'PLAN.md',OUT/'dependencies/critical_rows.py']},
          design='3 sources x 3 preregistered seeds x 2 paired methods; 64 environments x 15 seconds; no IID confidence interval',
          planned_cells=18,planned_windows=1152)
(OUT/'campaign_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps(dict(cells=len(jobs),windows=1152,raw_output=str(CELLS),gpu=0)))
