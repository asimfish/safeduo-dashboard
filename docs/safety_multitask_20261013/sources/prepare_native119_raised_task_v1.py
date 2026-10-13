"""Preserve118 and test a source-admitted raised initial arm state."""
from pathlib import Path
import ast,copy,datetime,hashlib,json,shutil

K=Path(__file__).resolve().parent
H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
B=H/'fourhand_native_probe_v118';P=H/'fourhand_native_probe_v119'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert not P.exists();admission=read(K/'PAIR_TASK_RAISED_ADMISSION_V3.json');assert admission['admitted']
    reg=read(B/'REGISTRATION_V1.json')
    for f,h in reg['source_sha256'].items():assert sha(f)==h,f
    plan={'candidate':119,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'baseline':118,
        'hypothesis':'Removing initialized hand-payload overlap by raising both F arms60mm delays the contact reconstruction spike until actual physical approach',
        'change':'Initial arm q and corresponding arm targets raised60mm by source IK; all finger states, all velocities, object state, controller, solver and gates unchanged',
        'maximum_native_launches':1,'maximum_controls':1024,'maximum_microsteps':32768,'maximum_owned_wall_s':1200,
        'causal_scope':'Startup-state counterfactual; same adaptive controller need not issue identical closed-loop requests',
        'original_model_failure_never_retried_or_relaxed':True,'learned_System0_active':False,'fullSystem0_accepted':False}
    planfile=K/'NATIVE119_RAISED_TASK_PROSPECTIVE_V1.json';planfile.write_text(json.dumps(plan,indent=2)+'\n')
    P.mkdir()
    for f in B.iterdir():
        if f.suffix in ['.py','.usda'] or f.name=='PRIVATE_KIT_USER_CONFIG_FROZEN_V1.json':shutil.copyfile(f,P/f.name)
    shutil.copytree(B/'private_kit_user_data',P/'private_kit_user_data')
    worker=P/'native_common_beam_v1.py';source=worker.read_text()
    old="approach_tested=False)";assert source.count(old)==1
    source=source.replace(old,"approach_tested=False,initial_seed_is_exact_previous_native_capture=False,initial_only_source_IK_arm_raise_m=.06)")
    ast.parse(source);worker.write_text(source)
    launcher=H/'launch_native_clock_admission_v119.py';s=(H/'launch_native_clock_admission_v118.py').read_text().replace('v118','v119').replace('V118','V119').replace('NATIVE118','NATIVE119');ast.parse(s);launcher.write_text(s)
    auditor=H/'audit_motion_capture119_v1.py';s=(H/'audit_motion_capture118_v1.py').read_text().replace('assert version == 118','assert version == 119');ast.parse(s);auditor.write_text(s)
    taskauditor=K/'audit_native119_pair_task_v1.py';s=(K/'audit_native118_pair_task_v1.py').read_text().replace('118','119');ast.parse(s);taskauditor.write_text(s)
    new=copy.deepcopy(reg);new.update(attempt_version=119,scientific_candidate_version=119,registered_utc=plan['created_utc'],change_scope=plan['change'],
        independent_capture_auditor=str(auditor),private_asset_settings=dict(reg['private_asset_settings'],private_directory=str(P/'private_kit_user_data')))
    seed=K/'PAIR_TASK_RAISED_NATIVE_SEED_V3.npz'
    new['contact_reproducer']=dict(reg['contact_reproducer'],fixture=str(seed),fixture_sha256=sha(seed),seed_derived_with_initial_only_IK_raise_m=.06)
    new['actual_state_guard']['source']=str(P/'full50_actual_guard_v1.py');new['full50_guard']['source']=str(P/'full50_actual_guard_v1.py')
    sources={}
    for f in reg['source_sha256']:
        old=Path(f)
        if old.is_relative_to(B):dest=P/old.relative_to(B)
        elif old==H/'launch_native_clock_admission_v118.py':dest=launcher
        elif old==H/'audit_motion_capture118_v1.py':dest=auditor
        else:dest=old
        sources[str(dest)]=sha(dest)
    for f in [Path(__file__),planfile,seed,K/'PAIR_TASK_RAISED_ADMISSION_V3.json',K/'NATIVE118_INDEPENDENT_PAIR_TASK_V1.json',taskauditor]:sources[str(f)]=sha(f)
    new['source_sha256']=sources;(P/'REGISTRATION_V1.json').write_text(json.dumps(new,indent=2)+'\n')
    print('NATIVE119_RAISED_INITIAL_POSE_TASK_REGISTERED;same task controller and original gates;one launch',flush=True)


if __name__=='__main__':main()
