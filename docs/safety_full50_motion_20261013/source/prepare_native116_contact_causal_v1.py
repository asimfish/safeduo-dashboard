"""A declared single contact-pair ablation, never a qualified safety candidate."""
from pathlib import Path
import ast,copy,datetime,hashlib,json,shutil

E=Path(__file__).resolve().parent
H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
B=H/'fourhand_native_probe_v115';P=H/'fourhand_native_probe_v116'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert not P.exists();reg=read(B/'REGISTRATION_V1.json');res=read(B/'RESOURCE_EXECUTION_V1.json')
    assert res['native_owned_process_reaped_and_absent'] and res['actual_exit']==0
    failure=read(E/'NATIVE115_FAILURE_LOCALIZATION_V1.json');assert failure['first_original_model_failure_event']==failure['first_new_FR_object_contact_event']==824
    for f,h in reg['source_sha256'].items():assert sha(f)==h,f
    docs=Path('/home/liyufeng/miniforge3/envs/safeduo/lib/python3.11/site-packages/isaacsim/extscache/omni.physx.tests-107.3.26+107.3.3.cp311.u353/omni/physxtests/tests/PhysicsFilteredPairsAPI.py')
    assert 'filteredPairsAPI.CreateFilteredPairsRel().AddTarget(self._box0.GetPrimPath())' in docs.read_text()
    plan={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'candidate':116,'paired_contact_baseline':115,
        'type':'CAUSAL_ABLATION_NOT_A_SAFETY_CANDIDATE','change':'Filter exactly beam_common vs F_R/right_thumb_4 before physics initialization using official USD FilteredPairsAPI; keep all other contact pairs, masses, input, FIFO, gains, native kernel and q/qdot gates.',
        'filter_owner':'/World/envs/env_0/Obj_beam_common','filter_target':'/World/envs/env_0/F_R/f2_hand/right_thumb_4',
        'maximum_native_launches':1,'maximum_controls':128,'maximum_microsteps':4096,'maximum_owned_wall_s':1200,
        'original_pair_event824_coincidence':'FR contact0->0.0343168N and right_thumb_3 relationqd residual0.000140568 at same native event',
        'precontact_equivalence_gate':'Compare actual physical fields bitwise until event823; if diverged before contact, do not attribute event824 disappearance solely to this contact.',
        'causal_observer':'ActualUSD relation exactly one target; raw contact0 to the filtered link each physicalstep; original q/qd at event824 or retained earlier failure; post824 contact/dynamics and all failures retained.',
        'no_safety_or_task_acceptance_from_this_changed_collision_world':True,'formal_holdouts':0,'new_training_jobs':0,'fullSystem0_accepted':False}
    planfile=E/'NATIVE116_SINGLE_CONTACT_CAUSAL_PLAN_V1.json';planfile.write_text(json.dumps(plan,indent=2)+'\n')
    P.mkdir()
    for f in B.iterdir():
        if f.suffix in ('.py','.usda') or f.name=='PRIVATE_KIT_USER_CONFIG_FROZEN_V1.json':shutil.copyfile(f,P/f.name)
    shutil.copytree(B/'private_kit_user_data',P/'private_kit_user_data')
    worker=P/'native_common_beam_v1.py';s=worker.read_text()
    old="                super()._setup_scene()"
    new=old+"""
                # Scientific contact ablation, explicitly excluded from acceptance.
                stage=omni.usd.get_context().get_stage()
                owner=stage.GetPrimAtPath(reg['contact_causal_ablation']['filter_owner'])
                target=stage.GetPrimAtPath(reg['contact_causal_ablation']['filter_target'])
                assert owner.IsValid() and target.IsValid()
                api=UsdPhysics.FilteredPairsAPI.Apply(owner)
                assert not api.GetFilteredPairsRel().GetTargets()
                api.CreateFilteredPairsRel().AddTarget(target.GetPath())
                assert api.GetFilteredPairsRel().GetTargets()==[target.GetPath()]
                save(out/'causal_contact_filter_binding_v1.json',dict(owner=str(owner.GetPath()),
                    actual_targets=[str(x) for x in api.GetFilteredPairsRel().GetTargets()],
                    before_physics_initialization=True,excluded_from_all_safety_and_task_acceptance=True))
"""
    assert s.count(old)==1;s=s.replace(old,new)
    old="        protocol.update(status='complete',task_status="
    assert s.count(old)==1;s=s.replace(old,"        protocol.update(causal_contact_ablation=True,excluded_from_safety_acceptance=True)\n"+old)
    ast.parse(s);worker.write_text(s)
    launcher=H/'launch_native_clock_admission_v116.py';s=(H/'launch_native_clock_admission_v115.py').read_text().replace('v115','v116').replace('V115','V116').replace('NATIVE115','NATIVE116');ast.parse(s);launcher.write_text(s)
    auditor=H/'audit_motion_capture116_v1.py';s=(H/'audit_motion_capture115_v1.py').read_text().replace('assert version == 115','assert version == 116');ast.parse(s);auditor.write_text(s)
    new=copy.deepcopy(reg);new.update(attempt_version=116,scientific_candidate_version=116,registered_utc=plan['created_utc'],
        change_scope=plan['change'],contact_causal_ablation=plan,excluded_from_all_safety_and_task_acceptance=True,
        independent_capture_auditor=str(auditor),private_asset_settings=dict(reg['private_asset_settings'],private_directory=str(P/'private_kit_user_data')))
    new['actual_state_guard']['source']=str(P/'full50_actual_guard_v1.py');new['full50_guard']['source']=str(P/'full50_actual_guard_v1.py')
    sources={}
    for f in reg['source_sha256']:
        old=Path(f)
        if old.is_relative_to(B):dest=P/old.relative_to(B)
        elif old==H/'launch_native_clock_admission_v115.py':dest=launcher
        elif old==H/'audit_motion_capture115_v1.py':dest=auditor
        else:dest=old
        sources[str(dest)]=sha(dest)
    for f in [Path(__file__),planfile,docs,E/'NATIVE115_FAILURE_LOCALIZATION_V1.json']:sources[str(f)]=sha(f)
    new['source_sha256']=sources
    assert new['native_sweep_spike']==reg['native_sweep_spike'] and new['full_motor_effort_controller']==reg['full_motor_effort_controller']
    (P/'REGISTRATION_V1.json').write_text(json.dumps(new,indent=2)+'\n')
    print('NATIVE116_CAUSAL_REGISTERED;one contact pair ablation;excluded from every safety/task acceptance;original115 retained',flush=True)


if __name__=='__main__':main()
