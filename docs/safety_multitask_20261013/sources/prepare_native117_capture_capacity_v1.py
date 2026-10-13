"""One known physical replay, changing only the host ledger capacity."""
from pathlib import Path
import ast,copy,datetime,hashlib,json,shutil

K=Path(__file__).resolve().parent
H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
B=H/'fourhand_native_probe_v115';P=H/'fourhand_native_probe_v117';N=K/'capture_capacity65536_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def main():
    assert not P.exists()
    assert read(N/'CPU_BOUNDARY_VERIFICATION_V1.json')['actual_exit']==0
    reg=read(B/'REGISTRATION_V1.json')
    for f,h in reg['source_sha256'].items():assert sha(f)==h,f
    plan={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'candidate':117,'baseline':115,
        'change':'Host count array and associated boundary checks8192→65536 only; no physics numerical change',
        'maximum_native_launches':1,'maximum_controls':128,'maximum_microsteps':4096,'maximum_owned_wall_s':1200,
        'required_byte_identical_physical_events':832,'expected_baseline_failure_event':824,
        'expected_model_gate_pass':False,'new_random_draws':0,'formal_holdouts':0,'fullSystem0_accepted':False}
    planfile=K/'NATIVE117_PROSPECTIVE_CAPACITY_ONLY_V1.json';planfile.write_text(json.dumps(plan,indent=2)+'\n')
    P.mkdir()
    for f in B.iterdir():
        if f.suffix in ['.py','.usda'] or f.name=='PRIVATE_KIT_USER_CONFIG_FROZEN_V1.json':shutil.copyfile(f,P/f.name)
    shutil.copytree(B/'private_kit_user_data',P/'private_kit_user_data')
    launcher=H/'launch_native_clock_admission_v117.py'
    source=(H/'launch_native_clock_admission_v115.py').read_text().replace('v115','v117').replace('V115','V117').replace('NATIVE115','NATIVE117')
    ast.parse(source);launcher.write_text(source)
    auditor=H/'audit_motion_capture117_v1.py'
    source=(H/'audit_motion_capture115_v1.py').read_text().replace('assert version == 115','assert version == 117')
    ast.parse(source);auditor.write_text(source)
    new=copy.deepcopy(reg);new.update(attempt_version=117,scientific_candidate_version=117,registered_utc=plan['created_utc'],
        change_scope=plan['change'],independent_capture_auditor=str(auditor),execution_only_retry=False,
        private_asset_settings=dict(reg['private_asset_settings'],private_directory=str(P/'private_kit_user_data')))
    new['native_sweep_spike']=dict(reg['native_sweep_spike'],library=str(N/'bridge.so'),event_ledger_capacity=65536)
    new['actual_state_guard']['source']=str(P/'full50_actual_guard_v1.py');new['full50_guard']['source']=str(P/'full50_actual_guard_v1.py')
    sources={}
    for f in reg['source_sha256']:
        old=Path(f)
        if old.is_relative_to(B):dest=P/old.relative_to(B)
        elif old==H/'launch_native_clock_admission_v115.py':dest=launcher
        elif old==H/'audit_motion_capture115_v1.py':dest=auditor
        elif old.is_relative_to(H/'spike/native_mimic_long_duration_v1'):dest=N/old.name
        else:dest=old
        sources[str(dest)]=sha(dest)
    for f in [Path(__file__),planfile,N/'bridge.so',N/'bridge.cpp',N/'CPU_BOUNDARY_VERIFICATION_V1.json',K/'ADR_CONTINUOUS_TASK_CAPTURE_V1.md']:
        sources[str(f)]=sha(f)
    new['source_sha256']=sources
    (P/'REGISTRATION_V1.json').write_text(json.dumps(new,indent=2)+'\n')
    print('NATIVE117_CAPTURE_CAPACITY_ONLY_REGISTERED',flush=True)


if __name__=='__main__':main()
