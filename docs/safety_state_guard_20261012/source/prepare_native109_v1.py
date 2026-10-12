"""Isolated single native counterfactual; original physics/boundary unchanged."""
from pathlib import Path
import ast,copy,datetime,hashlib,json,shutil

D=Path(__file__).resolve().parent
H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
B=H/'fourhand_native_probe_v108'
P=H/'fourhand_native_probe_v109'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def replace_once(s,before,after):
    assert s.count(before)==1,before[:100]
    return s.replace(before,after)


def main():
    assert not P.exists()
    replay=read(D/'KNOWN108_ACTUAL_STATE_REPLAY_V1.json')
    assert replay['candidate_intervenes_before_recorded_failure']
    reg=read(B/'REGISTRATION_V1.json')
    for f,d in reg['source_sha256'].items():assert sha(f)==d,'frozen source changed:'+f
    plan={
        'status':'REGISTERED_BEFORE_NATIVE_ACTUAL_STATE_COUNTERFACTUAL',
        'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'candidate':109,'maximum_controls':256,'maximum_microsteps':4096,
        'maximum_simulation_duration_s':4096/960.,'maximum_native_wall_seconds':1200,
        'maximum_native_launches':1,'failed_native_launches_never_retried':True,
        'paired_request_source':str(H/'motion_safety_benchmark_v1/RANDOM_ARM_COMMAND_BANK_107_V1.npz'),
        'unchanged_native16_pass_boundary':True,'original960Hz_60Hz_and_FIFO6':True,
        'guard_placement':'Online after actual FIFO pop, before actuation. Every control reads actual full74q/qd, current native9021 proxy gaps and pendingFIFO6; no queue flush or state writes.',
        'command_rate_rad_s':.25,'backup_retreat_rate_rad_s':.15,
        'reserve_m':.0005,'forecast_horizon_s':.12,'sampled_forecast_points':9,
        'forecast_assumption':'Persistent current arm velocity plus bounded target changes, with per-pair native current gap anchoring. NOT a validated nonlinear tracking/braking bound or whole-hand future geometry certificate.',
        'hands':'Original actual-target feedback and coupled hand effort unchanged; random hand coordinates0 in this causal test.',
        'scope':'One known initial state and one previously recorded random26 intent stream. Deterministic candidate, no new learned training, no full50 random or long-horizon acceptance.',
        'native_threshold_changes':0,'formal_holdouts':0,'fullSystem0_accepted':False}
    planfile=D/'NATIVE109_PROSPECTIVE_PLAN_V1.json'
    planfile.write_text(json.dumps(plan,indent=2)+'\n')
    P.mkdir()
    for f in B.iterdir():
        if f.suffix in ('.py','.usda') or f.name=='PRIVATE_KIT_USER_CONFIG_FROZEN_V1.json':shutil.copyfile(f,P/f.name)
    shutil.copytree(B/'private_kit_user_data',P/'private_kit_user_data')
    shutil.copyfile(D/'actual_state_arm_guard_v1.py',P/'actual_state_arm_guard_v1.py')
    worker=P/'native_common_beam_v1.py'
    text=worker.read_text()
    setup="""        from actual_state_arm_guard_v1 import ActualStateArmGuard
        sys.path.insert(0,str(H))
        from prepare_random_arm_pair107_108_v1 import proxy_gaps
        from safeduo.delta.task_record_s9 import make_s9_provider
        guard_provider=make_s9_provider(1,device='cpu',spheres='v7')
        with np.load(reg['actual_state_guard']['retreat_bank'],allow_pickle=False) as z:
            guard_retreat={a:z[a+'_issued_target'][-1].copy() for a in ARMS}
        actual_state_guard=ActualStateArmGuard(lambda q:proxy_gaps(guard_provider,q),
            {a:parameters[a+'_hard_limits'][0] for a in ARMS},
            {a:parameters[a+'_max_velocity'][0] for a in ARMS},
            {a:parameters[a+'_controlled_indices'] for a in ARMS},guard_retreat,
            **reg['actual_state_guard']['settings'])
"""
    before="        capture(0)\n        for step in range(reg['maximum_controls']):"
    text=replace_once(text,before,setup+before)
    before="            pending.append({a:issued[a].clone() for a in ARMS});applied=pending.popleft();u=1."
    after="""            pending.append({a:issued[a].clone() for a in ARMS});fifo_applied=pending.popleft();u=1.
            decision_q={a:array(env._arms[a].root_physx_view.get_dof_positions())[0] for a in ARMS}
            decision_v={a:array(env._arms[a].root_physx_view.get_dof_velocities())[0] for a in ARMS}
            decision_native_gaps=array(env.compute_dist().dists)[0]
            guard_pending=[{a:array(packet[a])[0] for a in ARMS} for packet in pending]
            if step>=4:
                guard_decision=actual_state_guard.decide(decision_q,decision_v,
                    {a:array(fifo_applied[a])[0] for a in ARMS},guard_pending,decision_native_gaps)
                applied={a:torch.as_tensor(guard_decision.target[a],device=env.device,dtype=initial[a].dtype)[None] for a in ARMS}
            else:
                guard_decision=None;applied={a:fifo_applied[a].clone() for a in ARMS}
"""
    text=replace_once(text,before,after)
    # Post-FIFO target must actually reach every microstep's root effort controller.
    before="                        ids=hands.ids[a][0]\n                        previous=array(env._arms[a].root_physx_view.get_dof_position_targets())[0,ids]"
    after="                        env._arms[a].set_joint_position_target(applied[a],joint_ids=parameters[a+'_controlled_indices'].tolist())\n"+before
    text=replace_once(text,before,after)
    before="                for a in ARMS:\n                    view=env._arms[a].root_physx_view;q=array(view.get_dof_positions());qd=array(view.get_dof_velocities())"
    after="""                data.update(guard_mode=np.asarray(guard_decision.mode if guard_decision else 'COMMON_PREFIX'),
                    guard_alpha=np.asarray(guard_decision.alpha if guard_decision else 1.),
                    guard_forecast_minimum_m=np.asarray(guard_decision.forecast_minimum_m if guard_decision else float(decision_native_gaps.min())),
                    guard_decision_actual_geometry=decision_native_gaps,
                    guard_prediction_qualified=np.asarray(False))
"""+before
    text=replace_once(text,before,after)
    before="                    data[a+'_actual_full_position_target']=array(view.get_dof_position_targets());data[a+'_extra_effort']=array(effort[a])"
    after="""                    data[a+'_fifo_applied_target']=array(fifo_applied[a])
                    data[a+'_pending_fifo_targets']=np.stack([packet[a] for packet in guard_pending])
                    data[a+'_guard_decision_q']=decision_q[a];data[a+'_guard_decision_qd']=decision_v[a]
"""+before
    text=replace_once(text,before,after)
    text=text.replace("SYNTHETIC_RANDOM26_ARM_WITH_FIFO6","RANDOM26_ACTUAL_STATE_POST_FIFO_GUARD")
    text=text.replace("Frozen unselected random26 arm targets or deterministic source-proxy guarded commands through real FIFO6; original full50 effort controller; held hand tape; not learnedSystem0",
        "Raw frozen random26 intents through actualFIFO6 and online actual-state/velocity/queue guard; original full50 effort and held hand feedback; deterministic candidate, not learnedSystem0")
    text=replace_once(text,"np.asarray('RANDOM_ARM_RAW' if reg['attempt_version']==107 else 'RANDOM_ARM_COMMAND_GUARD')","np.asarray('RANDOM_ARM_ACTUAL_STATE_GUARD')")
    ast.parse(text);worker.write_text(text)
    launcher=H/'launch_native_clock_admission_v109.py'
    text=(H/'launch_native_clock_admission_v108.py').read_text().replace('v108','v109').replace('V108','V109').replace('NATIVE108','NATIVE109')
    ast.parse(text);launcher.write_text(text)
    capture=H/'audit_motion_capture109_v1.py'
    text=(H/'audit_motion_capture108_v1.py').read_text().replace('assert version == 108','assert version == 109')
    before="        control_fifo_targets(applied,issued,initial_command[a][:,ix],substeps,reg['position_target_FIFO_controls'])"
    text=replace_once(text,before,"        control_fifo_targets(data[a+'_fifo_applied_target'],issued,initial_command[a][:,ix],substeps,reg['position_target_FIFO_controls'])")
    # Raw intent discontinuity is not presented as an executable command-slew pass.
    before="        if version==108:\n            assert np.max(abs(np.diff(np.concatenate([initial_command[a][:,ix][None],issued[0::substeps]],axis=0),axis=0)))<=.015+1e-6"
    text=replace_once(text,before,"        assert data[a+'_pending_fifo_targets'].shape==(count,6,len(ix))")
    ast.parse(text);capture.write_text(text)
    new=copy.deepcopy(reg)
    new.update(attempt_version=109,scientific_candidate_version=109,registered_utc=plan['utc'],
        change_scope=plan['scope']+' '+plan['guard_placement'],
        independent_capture_auditor=str(capture),
        private_asset_settings=dict(reg['private_asset_settings'],private_directory=str(P/'private_kit_user_data')),
        motion_test=dict(reg['motion_test'],command_bank=plan['paired_request_source'],
                         paired_versions=[107,108,109],guard_mode='online_actual_state_post_FIFO'),
        actual_state_guard={'source':str(P/'actual_state_arm_guard_v1.py'),
            'retreat_bank':str(H/'motion_safety_benchmark_v1/RETREAT_COMMAND_BANK_V1.npz'),
            'settings':{'command_rate_rad_s':.25,'backup_rate_rad_s':.15,'reserve_m':.0005,
                        'horizon_s':.12,'control_dt_s':1./60.,'fifo_controls':6},
            'nonclaims':[plan['forecast_assumption'],plan['scope']]})
    sources={}
    for file in reg['source_sha256']:
        old=Path(file)
        if old.is_relative_to(B):dest=P/old.relative_to(B)
        elif old==H/'launch_native_clock_admission_v108.py':dest=launcher
        elif old==H/'audit_motion_capture108_v1.py':dest=capture
        else:dest=old
        sources[str(dest)]=sha(dest)
    for f in (Path(__file__),P/'actual_state_arm_guard_v1.py',planfile,
              D/'KNOWN108_ACTUAL_STATE_REPLAY_V1.json',Path(plan['paired_request_source'])):
        sources[str(f)]=sha(f)
    new['source_sha256']=sources
    (P/'REGISTRATION_V1.json').write_text(json.dumps(new,indent=2)+'\n')
    print('NATIVE109_REGISTERED',len(sources),'source hashes;4096microsteps;one GPU1 launch;original gates;not learned',flush=True)


if __name__=='__main__':main()
