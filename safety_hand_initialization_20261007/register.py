import json, hashlib
from pathlib import Path
from datetime import datetime, timezone

p = Path(__file__).resolve().parent
old = p.parent/'safety_hand_support_calibration_20261007'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
template = json.loads((old/'REGISTRATION_OPEN_POSE.json').read_text())
assets = json.loads((old/'ASSET_READBACK.json').read_text())['files']
legacy = p/'legacy_native_metadata.json'
meta = json.loads(legacy.read_text())
arms = ['F_L','F_R','U_L','U_R']
task = {'F_L':(.5264,None),'F_R':(.5182,None),'U_L':(.4769,.7569),'U_R':(.4746,.7546)}
events = []
for t, mode in [(0,'open'),(3,'calibration'),(6,'open'),(9,'task'),(12,'open')]:
    event = dict(time_s=t,ramp_s=.6,arms={})
    for arm in arms:
        f, ft = (0,None) if mode=='open' else ((.45,None) if mode=='calibration' else task[arm])
        event['arms'][arm] = dict(fraction=f, fraction_thumb=ft)
    events.append(event)
cases = []
for label, t in [('initial',0),('pickup',4.5),('placement',15.2)]:
    for post in [0.,.35]:
        cases.append(dict(env=len(cases),kind='isolated_hand',label=label,reference_time_s=t,post_write_thumb_rad=post,
                          centers_m=[[.5,1.3,1.5],[-.4,-1.3,1.5]]))
sources = {**template['source_sha256'], **assets, str(legacy):sha(legacy)}
for name in ['native_probe.py','absolute_hand_targets.py','legacy_hand_driver.py','analyze.py','FAILURE_CONTRACT.md','register.py','check_driver.py']:
    sources[str(p/name)] = sha(p/name)
for path, digest in sources.items():
    assert sha(path)==digest, path
common = dict(created_utc=datetime.now(timezone.utc).isoformat(),status='FROZEN_BEFORE_NATIVE_EXECUTION', seed=90610753,
  scope='Two six-environment static initialization diagnostics; three unique four-arm references x two post-reset hand states. No payload grasp, System0, task or held-out trials.',
  env_yaml=template['env_yaml'],trajectory=template['trajectory'],source_sha256=sources,
  disable_self_collision=False,cases=cases,legacy_metadata=str(legacy),steps=2160,physics_dt_s=template['physics_dt_s'],
  hand_events=events,object_yaws_rad=template['object_yaws_rad'],contact_capacity=4096,target_open_thumb_rad=.35,
  capture_steps=[0,659,1019,1379,1979],capture_envs=list(range(6)),views=template['views'],
  evaluation_windows_s={'startup':[0,3],'closed_calibration':[5,6],'open_after_calibration':[8,9], 'closed_task_endpoint':[11,12],'open_after_task_endpoint':[15,17.999]},
  criteria=dict(pair_normal_max_n=.1,body_net_max_n=.1,net_filter_accounting_max_abs_n=.001,open_error_max_rad=.02,target_error_max_rad=1e-6,closed_tracking_error_max_rad=.02),
  limitations=['No contact tensors during constructor internal steps; startup gate covers only explicit loop steps after diagnostic initialization.',
               'Absolute closed endpoints preserved. Intermediate path uses full-duration joint interpolation and differs from original scalar-fraction ramp.',
               'Static arm references do not imply fixed actual arms. No payloads near hands. No grasps or safety-policy performance established.'])
for name, birth in [('old_default',0.),('neutral_default',.35)]:
    reg = dict(**common,run=name,constructor_thumb_rad=birth)
    path=p/f'REGISTRATION_{name}.json'
    assert not path.exists(), path
    path.write_text(json.dumps(reg,indent=2,allow_nan=False)+'\n')
seal = dict(created_utc=datetime.now(timezone.utc).isoformat(),registrations={f'REGISTRATION_{n}.json':sha(p/f'REGISTRATION_{n}.json') for n in ['old_default','neutral_default']},analyzer_sha256=sha(p/'analyze.py'),harness_sha256=sha(p/'native_probe.py'),driver_sha256=sha(p/'absolute_hand_targets.py'))
(p/'REGISTRATION_SEAL.json').write_text(json.dumps(seal,indent=2)+'\n')
print(json.dumps(seal,indent=2))
