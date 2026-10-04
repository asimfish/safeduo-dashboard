import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
base=json.loads((HERE/'campaign_plan_dynamic.json').read_text())
path=HERE/'campaign_plan_feasible.json'
assert not path.exists()
jobs=[]
helper_names=['wide_runner_feasible.py','pose_bank.py','FEASIBLE_PLAN.md','register_bank.py',
              'bank_registration.json','register_feasible.py']
helpers={**base['research_source_sha256'],**{str(HERE/name):hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in helper_names}}
for seed in [13447771,27180353,48921161]:
    bank=Path('/mnt/nas/data/lyf/double_hand/safety_random_space_20261004/pose_banks')/str(seed)
    m=json.loads((bank/'metadata.json').read_text())
    assert m['status']=='complete' and m['selected_count']==64
    for name in ['bank.npz','metadata.json']:
        helpers[str(bank/name)]=hashlib.sha256((bank/name).read_bytes()).hexdigest()
    for method in ['raw','system0']:
        argv=[base['jobs'][0]['argv'][0],str(HERE/'wide_runner_feasible.py'),
              '--ckpt',base['checkpoint_path'],'--env-yaml','duo_env_a31_pending_guard.yaml',
              '--num-envs','64','--duration-s','15','--seeds',str(seed),'--amps','.05',
              '--flows','feasible_burst','--methods',method,'--init-jitter-rad','0',
              '--actuator-delay-steps','6','--device','cuda:1','--headless','--log-every','300']
        jobs.append(dict(id=f'feasible_burst_{seed}_{method}',argv=argv,
                         expected=dict(flow='feasible_burst',seed=seed,method=method,amp=.05,actuator_delay_steps=6),
                         expected_args=dict(num_envs=64,duration_s=15.,device='cuda:1'),
                         env=dict(SAFEDUO_INITIAL_BANK_NPZ=str(bank/'bank.npz'))))
plan={**base,'jobs':jobs,'env':{**base['env'],'CUDA_VISIBLE_DEVICES':'0,1'},
      'registered_utc':datetime.now(timezone.utc).isoformat(),
      'output_root':'/mnt/nas/data/lyf/double_hand/safety_random_space_20261004/cells_feasible',
      'research_source_sha256':helpers,'planned_cells':6,'planned_windows':384,
      'design':'3 new seeds x 2 paired methods; conditioned initial geometry bank; no future-outcome selection',
      'device_note':'explicit cuda:1 with both devices visible avoids CUDA/Vulkan remapping; original bank geometry and all inputs retained'}
for name,sha in plan['source_sha256'].items():
    assert hashlib.sha256((Path(plan['cwd'])/name).read_bytes()).hexdigest()==sha,name
path.write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps({'registered_cells':6,'new_unique_command_windows':192,'gpu':1}))
