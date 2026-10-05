"""Freeze neutral diagnostics before any of their outcomes."""
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
old=json.loads((HERE.parent/'safety_predictive_admission_20261005/holdout_plan.json').read_text())
helpers=dict(old['research_source_sha256'])
for n in ('zero_runner.py','PLAN.md','register.py'):helpers[str(HERE/n)]=hashlib.sha256((HERE/n).read_bytes()).hexdigest()
for n,s in helpers.items():assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==s,n
jobs=[]
for seed in old['seeds']:
    for mode in ('raw','baseline','envelope_050'):
        prior=next(j for j in old['jobs'] if j['expected']['seed']==seed and j['mode']==mode)
        job=copy.deepcopy(prior);job['id']=f'zero_{seed}_{mode}'
        job['argv'][1]=str(HERE/'zero_runner.py')
        job['argv'][job['argv'].index('--device')+1]='cuda:1';job['expected_args']['device']='cuda:1'
        job['env'].pop('SAFEDUO_ADMISSION_MODE');job['env']['SAFEDUO_MECHANISM']='envelope_050' if mode=='envelope_050' else 'baseline'
        jobs.append(job)
path=HERE/'campaign_plan.json';assert not path.exists()
plan={**copy.deepcopy(old),'jobs':jobs,'research_source_sha256':helpers,'registered_utc':datetime.now(timezone.utc).isoformat(),
      'output_root':'/mnt/nas/data/lyf/double_hand/safety_zero_command_20261005/cells','planned_cells':9,'planned_windows':576,
      'new_unique_command_windows':0,'candidate':'no new controller;16s exact-zero diagnostic on existing new banks',
      'design':'raw/baseline/original envelope; all inputs zero; no post-outcome initial selection',
      'modes':['raw','baseline','envelope_050'],'supplementary_not_primary_holdout':True,
      'prior_pressure_partial_outcomes_seen':True,'zero_diagnostic_outcomes_seen':False}
path.write_text(json.dumps(plan,indent=2)+'\n');print('Nine constant-zero diagnostics registered; no additional random trajectories.')
