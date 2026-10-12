"""Execution replica111: repair hardcoded clock checker, preserve failed startup."""
from pathlib import Path
import ast,copy,datetime,hashlib,json,shutil

D=Path(__file__).resolve().parent
H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
B=H/'fourhand_native_probe_v111';P=H/'fourhand_native_probe_v112'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert not P.exists()
    r=read(B/'RESOURCE_EXECUTION_V1.json');raw=Path(r['out']);protocol=read(raw/'fourhand_protocol.json')
    assert r['status']=='failed' and r['native_owned_process_reaped_and_absent']
    assert protocol['physics_events']==protocol['completed_controls']==0 and protocol['status']=='failed'
    assert 'native_clock_binding_v1.py' in protocol['traceback'] and 'n==16' in protocol['traceback']
    assert not list(raw.glob('physics_*.npz'))
    reg=read(B/'REGISTRATION_V1.json')
    for f,d in reg['source_sha256'].items():assert sha(f)==d,f
    scope={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'execution_version':112,'scientific_candidate':111,'unchanged_scientific_plan':str(D/'NATIVE111_PROSPECTIVE_PLAN_V1.json'),
        'failed_native111_startup_retained':True,'native111_rollout_steps':0,
        'only_repair':'Clock checker accepts registered16 or32 microsteps while retaining exact SDK getter/USD scene/registration/60Hz/FIFO6 agreement; no controller, physics, target, kernel or acceptance threshold change from111.',
        'native111_outer_exit':1,'native111_worker_exit':0,'worker_exit_not_treated_as_protocol_success':True,
        'new_execution_maximum_native_launches':1,'physical_episode_trials_before_repair':0,
        'fullSystem0_accepted':False}
    scopefile=D/'NATIVE112_STARTUP_REPAIR_SCOPE_V1.json';scopefile.write_text(json.dumps(scope,indent=2)+'\n')
    P.mkdir()
    for f in B.iterdir():
        if f.suffix in ('.py','.usda') or f.name=='PRIVATE_KIT_USER_CONFIG_FROZEN_V1.json':shutil.copyfile(f,P/f.name)
    shutil.copytree(B/'private_kit_user_data',P/'private_kit_user_data')
    shutil.copyfile(D/'native_clock_binding_v2.py',P/'native_clock_binding_v1.py')
    assert sha(P/'native_common_beam_v1.py')==sha(B/'native_common_beam_v1.py')
    launcher=H/'launch_native_clock_admission_v112.py'
    s=(H/'launch_native_clock_admission_v111.py').read_text().replace('v111','v112').replace('V111','V112').replace('NATIVE111','NATIVE112')
    ast.parse(s);launcher.write_text(s)
    auditor=H/'audit_motion_capture112_v1.py'
    s=(H/'audit_motion_capture111_v1.py').read_text().replace('assert version == 111','assert version == 112')
    ast.parse(s);auditor.write_text(s)
    new=copy.deepcopy(reg)
    new.update(attempt_version=112,scientific_candidate_version=111,registered_utc=scope['utc'],
        execution_only_retry=True,execution_replica_of_version=111,
        execution_implementation_repair=scope['only_repair'],
        independent_capture_auditor=str(auditor),
        private_asset_settings=dict(reg['private_asset_settings'],private_directory=str(P/'private_kit_user_data')))
    new['actual_state_guard']['source']=str(P/'actual_state_arm_guard_v1.py')
    sources={}
    for file in reg['source_sha256']:
        old=Path(file)
        if old.is_relative_to(B):dest=P/old.relative_to(B)
        elif old==H/'launch_native_clock_admission_v111.py':dest=launcher
        elif old==H/'audit_motion_capture111_v1.py':dest=auditor
        else:dest=old
        sources[str(dest)]=sha(dest)
    for f in (Path(__file__),scopefile,D/'native_clock_binding_v2.py',B/'RESOURCE_EXECUTION_V1.json',raw/'fourhand_protocol.json'):
        sources[str(f)]=sha(f)
    new['source_sha256']=sources
    assert new['physics_dt_s']==reg['physics_dt_s'] and new['physics_substeps_per_control']==32
    assert new['native_sweep_spike']==reg['native_sweep_spike']
    assert new['full_motor_effort_controller']==reg['full_motor_effort_controller']
    (P/'REGISTRATION_V1.json').write_text(json.dumps(new,indent=2)+'\n')
    print('NATIVE112_STARTUP_REPAIR_REGISTERED;scientific111 unchanged;native111zero-rollout retained;one execution',flush=True)


if __name__=='__main__':main()
