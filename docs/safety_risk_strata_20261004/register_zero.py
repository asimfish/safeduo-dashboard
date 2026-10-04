import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'safety_random_space_20261004'
previous=json.loads((BASE/'campaign_plan_feasible.json').read_text())
jobs=[]
for seed in [13447771,27180353,48921161]:
    original=next(j for j in previous['jobs'] if j['expected']['seed']==seed and j['expected']['method']=='raw')
    argv=original['argv'].copy();argv[1]=str(HERE/'zero_runner.py')
    for flag,value in [('--flows','feasible_zero'),('--amps','0'),('--duration-s','1'),('--log-every','30')]:argv[argv.index(flag)+1]=value
    jobs.append({**original,'id':f'zero_{seed}','argv':argv,
                 'expected':{**original['expected'],'flow':'feasible_zero','amp':0.},
                 'expected_args':{**original['expected_args'],'duration_s':1.}})
helpers=previous['research_source_sha256'].copy()
for name in ['PLAN.md','zero_runner.py','register_zero.py']:
    helpers[str(HERE/name)]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
plan={**previous,'jobs':jobs,'research_source_sha256':helpers,
      'registered_utc':datetime.now(timezone.utc).isoformat(),
      'output_root':'/mnt/nas/data/lyf/double_hand/safety_risk_strata_20261004/zero_cells',
      'planned_cells':3,'planned_windows':192,'design':'zero-command diagnostic; original banks; 64x1s; not random safety evidence'}
path=HERE/'zero_plan.json';assert not path.exists();path.write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps({'registered':3,'diagnostic_windows':192,'new_random_windows':0}))
