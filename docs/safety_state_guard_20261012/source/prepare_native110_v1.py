"""Independent slew-limited counterfactual; failed109 is never restarted."""
from pathlib import Path
import ast,copy,datetime,hashlib,json,shutil

D=Path(__file__).resolve().parent
H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
B=H/'fourhand_native_probe_v109';P=H/'fourhand_native_probe_v110'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def once(s,before,after):
    assert s.count(before)==1,before[:80]
    return s.replace(before,after)


def main():
    assert not P.exists()
    capture=read(B/'INDEPENDENT_CAPTURE_AUDIT_V1.json')
    model=read(B/'INDEPENDENT_NATIVE_MIMIC_WINDOW_V6.json')
    assert capture['record_integrity_verified'] and capture['physical_gates_pass'] and not model['model_window_pass']
    reg=read(B/'REGISTRATION_V1.json')
    for f,d in reg['source_sha256'].items():assert sha(f)==d,f
    prospective=copy.deepcopy(read(D/'NATIVE109_PROSPECTIVE_PLAN_V1.json'))
    prospective.update(candidate=110,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        command_rate_rad_s=.10,backup_retreat_rate_rad_s=.05,command_acceleration_rad_s2=.5,
        hypothesis='Limit command speed and acceleration relative to previous actual submission.109 anchored its target to actual q and showed up to0.00826174rad consecutive FR target jumps; FR thumb mimic qd failed later at event1207. Causation not proved; this is a single independent counterfactual.',
        previous_failed_native_version=109,maximum_native_launches=1,
        old_native_attempt_not_retried=True,fullSystem0_accepted=False)
    planfile=D/'NATIVE110_PROSPECTIVE_PLAN_V1.json';planfile.write_text(json.dumps(prospective,indent=2)+'\n')
    P.mkdir()
    for f in B.iterdir():
        if f.suffix in ('.py','.usda') or f.name=='PRIVATE_KIT_USER_CONFIG_FROZEN_V1.json':shutil.copyfile(f,P/f.name)
    shutil.copytree(B/'private_kit_user_data',P/'private_kit_user_data')
    shutil.copyfile(D/'actual_state_arm_guard_v2.py',P/'actual_state_arm_guard_v1.py')
    worker=P/'native_common_beam_v1.py';s=worker.read_text()
    before="        capture(0)\n        for step in range(reg['maximum_controls']):"
    s=once(s,before,"""        guard_previous_target={a:array(env._arms[a].root_physx_view.get_dof_position_targets())[0,parameters[a+'_controlled_indices']].copy() for a in ARMS}
        guard_previous_velocity={a:np.zeros_like(guard_previous_target[a],dtype=float) for a in ARMS}
"""+before)
    before="            if step>=4:\n                guard_decision=actual_state_guard.decide(decision_q,decision_v,"
    s=once(s,before,"""            guard_prior_target={a:z.copy() for a,z in guard_previous_target.items()}
            guard_prior_velocity={a:z.copy() for a,z in guard_previous_velocity.items()}
"""+before)
    before="                    {a:array(fifo_applied[a])[0] for a in ARMS},guard_pending,decision_native_gaps)"
    s=once(s,before,"                    {a:array(fifo_applied[a])[0] for a in ARMS},guard_pending,decision_native_gaps,guard_previous_target,guard_previous_velocity)")
    before="            nominal_hand_targets={};filtered_targets={}"
    s=once(s,before,"""            for a in ARMS:
                submitted=array(applied[a])[0].copy()
                guard_previous_velocity[a]=(submitted.astype(float)-guard_previous_target[a].astype(float))/dt
                guard_previous_target[a]=submitted
"""+before)
    before="                    data[a+'_guard_decision_q']=decision_q[a];data[a+'_guard_decision_qd']=decision_v[a]"
    s=once(s,before,before+"\n                    data[a+'_guard_previous_command_target']=guard_prior_target[a];data[a+'_guard_previous_command_velocity']=guard_prior_velocity[a]")
    s=s.replace('RANDOM26_ACTUAL_STATE_POST_FIFO_GUARD','RANDOM26_ACTUAL_STATE_SLEW_POST_FIFO_GUARD')
    ast.parse(s);worker.write_text(s)
    launcher=H/'launch_native_clock_admission_v110.py'
    s=(H/'launch_native_clock_admission_v109.py').read_text().replace('v109','v110').replace('V109','V110').replace('NATIVE109','NATIVE110')
    ast.parse(s);launcher.write_text(s)
    auditor=H/'audit_motion_capture110_v1.py'
    s=(H/'audit_motion_capture109_v1.py').read_text().replace('assert version == 109','assert version == 110')
    ast.parse(s);auditor.write_text(s)
    new=copy.deepcopy(reg)
    new.update(attempt_version=110,scientific_candidate_version=110,registered_utc=prospective['utc'],
        previous_failed_attempt=109,independent_capture_auditor=str(auditor),
        change_scope=prospective['hypothesis'],
        private_asset_settings=dict(reg['private_asset_settings'],private_directory=str(P/'private_kit_user_data')))
    new['motion_test']['paired_versions']=[107,108,109,110]
    new['motion_test']['guard_mode']='online_actual_state_post_FIFO_previous_command_slew'
    new['actual_state_guard']['source']=str(P/'actual_state_arm_guard_v1.py')
    new['actual_state_guard']['settings'].update(command_rate_rad_s=.10,backup_rate_rad_s=.05,command_acceleration_rad_s2=.5)
    sources={}
    for file in reg['source_sha256']:
        old=Path(file)
        if old.is_relative_to(B):dest=P/old.relative_to(B)
        elif old==H/'launch_native_clock_admission_v109.py':dest=launcher
        elif old==H/'audit_motion_capture109_v1.py':dest=auditor
        else:dest=old
        sources[str(dest)]=sha(dest)
    for f in (Path(__file__),planfile,D/'actual_state_arm_guard_v2.py',
              D/'SLEW_GUARD_CHANGE_SCOPE_V2.json',B/'INDEPENDENT_CAPTURE_AUDIT_V1.json',
              B/'INDEPENDENT_NATIVE_MIMIC_WINDOW_V6.json',
              Path('/home/liyufeng/safeduo/docs/adr/20261012-actual-state-random-guard.md')):
        sources[str(f)]=sha(f)
    new['source_sha256']=sources
    (P/'REGISTRATION_V1.json').write_text(json.dumps(new,indent=2)+'\n')
    print('NATIVE110_REGISTERED',len(sources),'source hashes;same raw input;4.2667s;one native attempt;original thresholds',flush=True)


if __name__=='__main__':main()
