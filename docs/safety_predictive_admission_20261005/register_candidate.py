"""Freeze candidate code and both factor conditions before any new outcome."""
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
old=json.loads((HERE.parent/'safety_mechanism_20261005/holdout_plan.json').read_text())
bank=json.loads((HERE/'bank_registration.json').read_text())
helpers={**old['research_source_sha256'],**bank['research_source_sha256']}
for n in ('predictive_rows.py','predictive_runner.py','test_predictive_rows.py','admission_audit.py','register_candidate.py','register_holdout.py','bank_registration.json'):
    helpers[str(HERE/n)]=hashlib.sha256((HERE/n).read_bytes()).hexdigest()
for n,s in old['source_sha256'].items():assert hashlib.sha256((Path(old['cwd'])/n).read_bytes()).hexdigest()==s,n
for n,s in helpers.items():assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==s,n
assert hashlib.sha256(Path(old['checkpoint_path']).read_bytes()).hexdigest()==old['checkpoint_sha256']

def job(seed,mode,bank,device):
    method='raw' if mode=='raw' else 'system0'
    argv=[old['jobs'][0]['argv'][0],str(HERE/'predictive_runner.py'),'--ckpt',old['checkpoint_path'],
          '--env-yaml','duo_env_a31_pending_guard.yaml','--num-envs','64','--duration-s','16','--seeds',str(seed),
          '--amps','.05','--flows','risk_burst','--methods',method,'--init-jitter-rad','0','--actuator-delay-steps','6',
          '--device',device,'--headless','--log-every','300']
    return dict(id=f'predictive_{seed}_{mode}',argv=argv,mode=mode,
                expected=dict(flow='risk_burst',seed=seed,method=method,amp=.05,actuator_delay_steps=6),
                expected_args=dict(num_envs=64,duration_s=16.,device=device),
                env=dict(SAFEDUO_INITIAL_BANK_NPZ=str(bank/'bank.npz'),SAFEDUO_ADMISSION_MODE=mode))

def main():
    path=HERE/'candidate_registration.json';assert not path.exists()
    plan={**copy.deepcopy(old),'registered_utc':datetime.now(timezone.utc).isoformat(),
          'research_source_sha256':helpers,'output_root':'/mnt/nas/data/lyf/double_hand/safety_predictive_admission_20261005/development_cells',
          'jobs':[job(60317411,m,Path('/mnt/nas/data/lyf/double_hand/safety_risk_strata_20261004/risk_banks/60317411'),'cuda:1') for m in ('predictive','predictive_envelope')],
          'planned_cells':2,'planned_windows':128,'new_unique_command_windows':0,
          'candidate':'evaluation-only online full-body predictive row admission, factorial original envelope',
          'design':'new seeds frozen before outcomes; no gain/FIFO/actor changes; parameters fixed before development',
          'seeds':bank['seeds'],'modes':['raw','baseline','envelope_050','predictive','predictive_envelope'],
          'holdout_planned_cells':15,'holdout_planned_windows':960,'holdout_new_unique_command_windows':192,
          'horizons_s':[.16,.4,.3],'band_m':.010,'predictive_capacity':2048}
    path.write_text(json.dumps(plan,indent=2)+'\n')
    assert not (HERE/'development_plan.json').exists()
    (HERE/'development_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print('Candidate frozen before development/holdout: two isolated factors, three new seeds, five methods.')

if __name__=='__main__':main()
