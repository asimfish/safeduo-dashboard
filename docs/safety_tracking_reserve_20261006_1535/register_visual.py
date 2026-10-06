"""Two own-state nine-view camera replays; do not add unique numerical cases."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
HERE=Path(__file__).resolve().parent
RAW=Path('/mnt/nas/data/lyf/double_hand')/HERE.name
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if __name__=='__main__':
    numerical=json.loads((HERE/'plans/holdout_0_plan.json').read_text())
    d=json.loads((HERE/'RANDOM_EXPERIMENT_DESIGN.json').read_text())
    b=d['rows'][0];bank=RAW/'banks'/str(b['initial_seed'])/'bank.npz'
    sources={str(HERE/p):sha(HERE/p) for p in ['guard_runner.py','tracking_reserve.py','target_forecast.py',
        'reference_envelope.py','projection_diagnostics.py','visual_runner.py','register_visual.py',
        'execute_visual.py','verify_visual.py','RANDOM_EXPERIMENT_DESIGN.json']}
    for p in (bank,bank.with_name('metadata.json')):sources[str(p)]=sha(p)
    jobs=[]
    for mode in ('joint_reference','delay_reserve'):
        argv=[numerical['jobs'][0]['argv'][0],str(HERE/'visual_runner.py'),'--out',str(RAW/'visual'/mode),
             '--ckpt',numerical['checkpoint_path'],'--env-yaml','duo_env_a31_pending_guard.yaml',
             '--seeds',str(b['command_seed']),'--methods','system0','--duration-s','16',
             '--device','cuda:0','--headless','--enable_cameras']
        jobs.append(dict(mode=mode,argv=argv,env={**numerical['env'],'SAFEDUO_INITIAL_BANK_NPZ':str(bank),
                          'SAFEDUO_JOINT_MODE':mode}))
    plan=dict(status='REGISTERED_BEFORE_CAMERA_OUTCOMES',registered_utc=datetime.now(timezone.utc).isoformat(),
          cwd=numerical['cwd'],original_source_sha256=numerical['source_sha256'],sources=sources,
          actor_sha256=numerical['checkpoint_sha256'],actual_visual_root=str(RAW/'visual'),jobs=jobs,
          scheduled_groups=42,views_per_group=9,minimum_scheduled_images=378,
          scope='primary comparison only, first fresh bank, separate own-state replays; visual denominator separate; fixed tight ablation numerical only',
          native_proof='all64 native q/qd/root/rootvelocity before and after rendering exact; actual USD optics and sixplane sphere containment',
          nonclaims='no occlusion/silhouette/hardware certificate; forward exact replay must be checked, never assumed')
    with (HERE/'visual_plan.json').open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
    print('FROZEN two camera replays,42 scheduled groups,378 required images',flush=True)
