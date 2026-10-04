import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from risk_recipe import validate_bank

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'safety_random_space_20261004'
path=HERE/'campaign_plan.json'
assert not path.exists(),'immutable registration'
base=json.loads((OLD/'campaign_plan_feasible.json').read_text())
registration=json.loads((HERE/'risk_bank_registration.json').read_text())
helpers={**base['research_source_sha256'],**registration['research_source_sha256']}
for name in ['risk_runner.py','risk_recipe.py','register_risk_campaign.py','RISK_PLAN.md']:
    helpers[str(HERE/name)]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
root=Path('/mnt/nas/data/lyf/double_hand/safety_risk_strata_20261004')
assert (root/'risk_banks/sampling_summary.json').exists(),'all candidates must be retained first'
jobs=[]
for seed in [60317411,80692357,109441003]:
    bank=root/'risk_banks'/str(seed);meta=json.loads((bank/'metadata.json').read_text())
    with np.load(bank/'bank.npz') as z:validate_bank(z['accepted_q'],z['risk_pair_index'],meta)
    assert meta['risk_quotas']==[8]*6 and meta['general_count']==16
    for name in ['bank.npz','metadata.json']:helpers[str(bank/name)]=hashlib.sha256((bank/name).read_bytes()).hexdigest()
    for method in ['raw','system0']:
        argv=[base['jobs'][0]['argv'][0],str(HERE/'risk_runner.py'),'--ckpt',base['checkpoint_path'],
              '--env-yaml','duo_env_a31_pending_guard.yaml','--num-envs','64','--duration-s','16',
              '--seeds',str(seed),'--amps','.05','--flows','risk_burst','--methods',method,
              '--init-jitter-rad','0','--actuator-delay-steps','6','--device','cuda:1','--headless','--log-every','300']
        jobs.append(dict(id=f'risk_{seed}_{method}',argv=argv,
                         expected=dict(flow='risk_burst',seed=seed,method=method,amp=.05,actuator_delay_steps=6),
                         expected_args=dict(num_envs=64,duration_s=16.,device='cuda:1'),
                         env=dict(SAFEDUO_INITIAL_BANK_NPZ=str(bank/'bank.npz'))))
plan={**base,'jobs':jobs,'research_source_sha256':helpers,'source_sha256':registration['source_sha256'],
      'registered_utc':datetime.now(timezone.utc).isoformat(),'output_root':str(root/'cells'),
      'planned_cells':6,'planned_windows':384,'new_unique_command_windows':192,
      'design':'3 new seeds x paired raw and dynamic512; each6 risk quotas8 + general16; full16s endpoint'}
for name,sha in plan['source_sha256'].items():assert hashlib.sha256((Path(plan['cwd'])/name).read_bytes()).hexdigest()==sha,name
for name,sha in helpers.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
path.write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps(dict(registered_cells=6,paired_windows=384,new_command_windows=192)))
