"""Preregister one clock-refinement diagnosis; no kernel modification."""
from pathlib import Path
import ast,copy,datetime,hashlib,json,shutil,sys

D=Path(__file__).resolve().parent
H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
B=H/'fourhand_native_probe_v110';P=H/'fourhand_native_probe_v111'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert not P.exists()
    capture=read(B/'INDEPENDENT_CAPTURE_AUDIT_V1.json');model=read(B/'INDEPENDENT_NATIVE_MIMIC_WINDOW_V6.json')
    assert capture['record_integrity_verified'] and capture['physical_gates_pass'] and not model['model_window_pass']
    reg=read(B/'REGISTRATION_V1.json')
    for f,d in reg['source_sha256'].items():assert sha(f)==d,f
    plan=copy.deepcopy(read(D/'NATIVE110_PROSPECTIVE_PLAN_V1.json'))
    plan.update(candidate=111,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        previous_failed_native_version=110,maximum_controls=128,maximum_microsteps=4096,
        maximum_simulation_duration_s=128/60.,physics_hz=1920,physics_substeps_per_control=32,
        command_clock_hz=60,FIFO6_real_duration_s=.1,hand_target_tape_original_hz=960,
        numerical_change='Halve physics integration dt; double physical samples per control; repeat original hand seed-target tape per physical time. Keep16-pass native boundary and original acceptance limits unchanged.',
        hypothesis='Timestep sensitivity may explain persistent FR thumb mimic qd failure despite guarded proxy/limits/speeds/forces. No claim before independent native audit.',
        initial64_physics_prefix_byte_identity_not_expected=True,
        allowed_native_parameter_array_difference=['actual_physics_dt_s'],
        all_other_native_properties_and13_seed_fields_exactly_equal110_required=True,
        maximum_native_launches=1,maximum_native_wall_seconds=1200,
        learnedSystem0_active=False,fullSystem0_accepted=False)
    planfile=D/'NATIVE111_PROSPECTIVE_PLAN_V1.json';planfile.write_text(json.dumps(plan,indent=2)+'\n')
    P.mkdir()
    for f in B.iterdir():
        if f.suffix in ('.py','.usda') or f.name=='PRIVATE_KIT_USER_CONFIG_FROZEN_V1.json':shutil.copyfile(f,P/f.name)
    shutil.copytree(B/'private_kit_user_data',P/'private_kit_user_data')
    worker=P/'native_common_beam_v1.py';s=worker.read_text()
    before='                tape_index=min(step*substeps+sub,7)';assert s.count(before)==1
    s=s.replace(before,'                tape_index=min((step*substeps+sub)//reg["execution_time_refinement_factor"],7)')
    ast.parse(s);worker.write_text(s)
    launcher=H/'launch_native_clock_admission_v111.py'
    s=(H/'launch_native_clock_admission_v110.py').read_text().replace('v110','v111').replace('V110','V111').replace('NATIVE110','NATIVE111')
    ast.parse(s);launcher.write_text(s)
    auditor=H/'audit_motion_capture111_v1.py'
    s=(H/'audit_motion_capture110_v1.py').read_text().replace('assert version == 110','assert version == 111')
    s=s.replace("reg['maximum_controls']==256","reg['maximum_controls']==128")
    ast.parse(s);auditor.write_text(s)
    new=copy.deepcopy(reg)
    new.update(attempt_version=111,scientific_candidate_version=111,registered_utc=plan['utc'],
        previous_failed_attempt=110,independent_capture_auditor=str(auditor),
        change_scope=plan['numerical_change'],maximum_controls=128,physics_dt_s=1./1920.,
        physics_substeps_per_control=32,execution_time_refinement_factor=2,
        time_resolution_refinement_factor=2*reg['time_resolution_refinement_factor'],
        private_asset_settings=dict(reg['private_asset_settings'],private_directory=str(P/'private_kit_user_data')))
    new['grasp_settle_gate']['physics_dt_s']=new['physics_dt_s']
    new['actual_state_guard']['source']=str(P/'actual_state_arm_guard_v1.py')
    new['motion_test']['paired_versions']=[107,108,109,110,111]
    new['motion_test']['first64_physical_fields_byte_identical105_required']=False
    new['numerical_clock_only_diagnostic']=plan
    sources={}
    for file in reg['source_sha256']:
        old=Path(file)
        if old.is_relative_to(B):dest=P/old.relative_to(B)
        elif old==H/'launch_native_clock_admission_v110.py':dest=launcher
        elif old==H/'audit_motion_capture110_v1.py':dest=auditor
        else:dest=old
        sources[str(dest)]=sha(dest)
    for f in (Path(__file__),planfile,B/'INDEPENDENT_CAPTURE_AUDIT_V1.json',
              B/'INDEPENDENT_NATIVE_MIMIC_WINDOW_V6.json',
              Path('/home/liyufeng/safeduo/docs/adr/20261012-random-model-clock-refinement.md')):
        sources[str(f)]=sha(f)
    new['source_sha256']=sources
    sys.path.insert(0,str(P))
    from microstep_contract_v1 import microsteps
    from refined_settle_clock_binding_v1 import validate_settle_clock
    assert microsteps(new)==32 and validate_settle_clock(new)
    assert new['native_sweep_spike']==reg['native_sweep_spike']
    assert new['full_motor_effort_controller']==reg['full_motor_effort_controller']
    assert new['expected_maximum_physics_events']==4096==128*32
    (P/'REGISTRATION_V1.json').write_text(json.dumps(new,indent=2)+'\n')
    print('NATIVE111_CLOCK_REGISTERED;1920Hz physics;60Hz control;FIFO6;original16pass bridge;4096microsteps;one native attempt',flush=True)


if __name__=='__main__':main()
