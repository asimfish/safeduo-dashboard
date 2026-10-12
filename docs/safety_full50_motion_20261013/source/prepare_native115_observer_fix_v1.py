"""Preserve failed114 and fix only its arm endpoint observer dimensions."""
from pathlib import Path
import ast,copy,datetime,hashlib,json,shutil

E=Path(__file__).resolve().parent
H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
B=H/'fourhand_native_probe_v114';P=H/'fourhand_native_probe_v115'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert not P.exists();reg=read(B/'REGISTRATION_V1.json');res=read(B/'RESOURCE_EXECUTION_V1.json');raw=Path(res['out']);protocol=read(raw/'fourhand_protocol.json')
    assert res['native_owned_process_reaped_and_absent'] and protocol['status']=='failed'
    assert "tensor a (13)" in protocol['error'] and protocol['completed_controls']==0
    assert not list(raw.glob('physics_*.npz'))
    for f,h in reg['source_sha256'].items():assert sha(f)==h,f
    plan={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'candidate':115,'input_bank_and_controller_candidate':114,
        'previous_failure':{'version':114,'outer_launcher_exit':1,'native_child_exit':0,'completed_controls':0,'archived_physics_events':0,
            'native_one_step_reached_before_observer_exception_by_source_trace':True,'one_unarchived_step_is_not_a_safety_trial':True},
        'change':'Select explicit original arm indices from50 targets in two unchanged task endpoint error observers; controller, geometry, FIFO, source requests and physical gates are byte-identical114',
        'maximum_native_launches':1,'maximum_controls':128,'maximum_microsteps':4096,'physics_hz':1920,'maximum_owned_wall_s':1200,
        'same_frozen_full50_IID_bank':True,'new_random_draws':0,'formal_holdouts':0,'new_training_jobs':0,'fullSystem0_accepted':False}
    planfile=E/'NATIVE115_OBSERVER_REPAIR_PROSPECTIVE_V1.json';planfile.write_text(json.dumps(plan,indent=2)+'\n')
    P.mkdir()
    for f in B.iterdir():
        if f.suffix in ('.py','.usda') or f.name=='PRIVATE_KIT_USER_CONFIG_FROZEN_V1.json':shutil.copyfile(f,P/f.name)
    shutil.copytree(B/'private_kit_user_data',P/'private_kit_user_data')
    worker=P/'native_common_beam_v1.py';before=worker.read_text();after=before
    for index in (0,1):
        old=f'applied[a]-endpoints[a][{index}][None]';new=f'applied[a][:,arm_in_full50[a]]-endpoints[a][{index}][None]'
        assert after.count(old)==1;after=after.replace(old,new)
    ast.parse(after);worker.write_text(after)
    launcher=H/'launch_native_clock_admission_v115.py';s=(H/'launch_native_clock_admission_v114.py').read_text().replace('v114','v115').replace('V114','V115').replace('NATIVE114','NATIVE115');ast.parse(s);launcher.write_text(s)
    auditor=H/'audit_motion_capture115_v1.py';s=(H/'audit_motion_capture114_v1.py').read_text().replace('assert version == 114','assert version == 115');ast.parse(s);auditor.write_text(s)
    new=copy.deepcopy(reg);new.update(attempt_version=115,scientific_candidate_version=115,registered_utc=plan['created_utc'],
        execution_only_retry=False,execution_replica_of_version=None,change_scope=plan['change'],independent_capture_auditor=str(auditor),
        previous_failed_full50_integration=plan['previous_failure'],private_asset_settings=dict(reg['private_asset_settings'],private_directory=str(P/'private_kit_user_data')))
    new['actual_state_guard']['source']=str(P/'full50_actual_guard_v1.py');new['full50_guard']['source']=str(P/'full50_actual_guard_v1.py')
    sources={}
    for f in reg['source_sha256']:
        old=Path(f)
        if old.is_relative_to(B):dest=P/old.relative_to(B)
        elif old==H/'launch_native_clock_admission_v114.py':dest=launcher
        elif old==H/'audit_motion_capture114_v1.py':dest=auditor
        else:dest=old
        sources[str(dest)]=sha(dest)
    for f in [Path(__file__),planfile,E/'test_full50_endpoint_binding_v1.py',raw/'fourhand_protocol.json',B/'native_common_beam_v1.py']:sources[str(f)]=sha(f)
    new['source_sha256']=sources
    assert new['native_sweep_spike']==reg['native_sweep_spike'] and new['full_motor_effort_controller']==reg['full_motor_effort_controller']
    (P/'REGISTRATION_V1.json').write_text(json.dumps(new,indent=2)+'\n');print('NATIVE115_REGISTERED;only two endpoint observer projections changed;failed114 preserved;one native step before failure not scored as safe',flush=True)


if __name__=='__main__':main()
