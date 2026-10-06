import json,hashlib,datetime
from pathlib import Path
R=Path(__file__).resolve().parent;OLD=Path('/home/liyufeng/safeduo/artifacts/safety_object_binding_20261006')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=json.loads((OLD/'REGISTRATION_V5.json').read_text())
reg={k:old[k] for k in ('trajectory','env_yaml','checkpoint','nominal_centers','nominal_yaws_rad','sizes_m','goals_local_m','views','physical_task_gates','supplementary_grasp_description','native_control_dt_s','trajectory_grid_dt_s','native_limits_readback')}
reg.update(schema='safeduo.release_clearance.development.v1',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),steps=1800,
    methods=['scheduled_debt','stationary_release','clearance_feedback'],
    gate=dict(release_start_s=15.2,clearance_start_s=18.7,support_normal_min_n=.1,bottom_abs_max_m=.006,settled_speed_max_m_s=.03,settled_angular_max_rad_s=.3,tilt_max_deg=10.,hand_normal_max_n=.1,open_error_max_rad=.2,stable_steps=6,max_wait_s=2.,reaction_speed_max_m_s=.1,reaction_angular_max_rad_s=1.,reaction_steps=3),
    capture_steps=[899,971,1120,1223,1799],capture_envs=list(range(12)),video_steps=list(range(887,1248,6)),
    scope='24 intentional development windows; 4 previously qualified small layouts, 3 methods, 2 seeds with clone-slot crossover; not expanded workspace, not IID, no final holdout, no hardware acceptance',
    intervention='stationary_release zeros all 4 arm nominal and R37 debt deltas during 15.2<=phase<18.7; preserves hand schedule and integrated targets. clearance_feedback adds a global 4-arm clearance gate at 18.7, max wait2s, then explicit ABORT; reactive post-contact stop after3 states. No predictive separation or grasp repair claimed.',
    common_harness_changes='12 clones per block; 30s common horizon; clear _pending_cmd before each step for every method and recompute from authoritative per-clone phase; this is a new common baseline, not bitexact reproduction of v5. HandDriver unchanged, common wall-clock hand events complete before clearance wait.',
    zero_command_not_zero_physical_velocity=True,
    measurement='84 named robot/table partners at last physics substep plus native net reported force. Static collider inventory retained; net-minus-filtered not full friction/wrench calibration. Native direct pose/velocity and all articulation DOF position/velocity reread around rendering.',
    video='Block0 layout0 in all3 methods selected prospectively: native front-camera frames every6 controlsteps from14.8..20.8s (~10Hz). Gallery top/front/side all24cases at5 fixed times; no interpolation.',
    analysis='Original physical_task_gates retained exactly; additionally report aborts/reference completion, >=80deg tilt, early dual-hand lift, recontact, command and physical path. Accepted task requires original gates AND full reference completion AND no abort. No statistical population reliability inference.')
reg['blocks']=[]
for b,seed in enumerate((90610741,90610742)):
    cases=[]
    for slot in range(12):
        combo=(slot+b*5)%12;mi=combo//4;layout=combo%4
        cases.append(dict(env=slot,layout=layout,method=reg['methods'][mi],binding=True,objects=old['cases'][layout+4]['objects']))
    reg['blocks'].append(dict(physics_seed=seed,cases=cases))
files=[R/'native_runner.py',R/'clearance_gate.py',R/'analyze.py',R/'test_gate.py',R/'DIAGNOSIS.json']+[OLD/p for p in ('object_binding_v3.py','hand_driver.py','clock_contract.py','NATIVE_LIMITS.json')]+[Path(reg[k]) for k in ('trajectory','env_yaml','checkpoint')]
files+=list(Path('/home/liyufeng/safeduo/src').rglob('*.py'))+list(Path('/home/liyufeng/safeduo/configs').rglob('*.yaml'))
reg['source_sha256']={str(p):sha(p) for p in sorted(set(files))}
out=R/'REGISTRATION.json';assert not out.exists();out.write_text(json.dumps(reg,indent=2,allow_nan=False)+'\n')
update=dict(registered_utc=reg['registered_utc'],H1='FALSIFIED: flange reference unchanged15.2..18.7 in all4 arms. This is not an early axial reference withdrawal.',H2='SUPPORTED OBSERVATION: fraction0 can coexist with contact and joint distance from first sample. First sample is not exact default open.',H3='SUPPORTED OBSERVATION: later clearance can recontact; reactive gate cannot prevent first contact.',H4='UNRESOLVED: inventory/net-minus-filtered diagnostic added; table-only support is not a complete weight balance.',H5='MECHANISM VERIFIED: held source emits repeated nonzero delta unless explicitly masked; common pending recomputation and nominal/debt mask are tested.',replacement='R37 compensation and existing target backlog/physical contacts remain alternative causes during reference hold. stationary_release is a controlled mitigation probe, not a proven grasp fix.',new_final_trials=0)
(R/'HYPOTHESIS_UPDATE.json').write_text(json.dumps(update,indent=2)+'\n')
print(json.dumps(dict(registration_sha256=sha(out),windows=24,seeds=2,source_files=len(files)),indent=2))
