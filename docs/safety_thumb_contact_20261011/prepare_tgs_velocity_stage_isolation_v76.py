"""Isolate the final TGS velocity stage, with the original physical/model oracle."""
from pathlib import Path
import ast
import copy
import datetime
import hashlib
import json
import shutil

H = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    base = H / 'fourhand_native_probe_v71'
    p = H / 'fourhand_native_probe_v76'
    assert not p.exists()
    assert read(H / 'fourhand_native_probe_v75/ACTUAL_COMPLETION_EXIT_V75.json')['actual_outer_exit'] == 0
    assert not read(H / 'fourhand_native_probe_v75/INDEPENDENT_NATIVE_MIMIC_WINDOW_V6.json')['model_window_pass']
    raw = Path(read(base / 'RESOURCE_EXECUTION_V1.json')['out'])
    caps = read(raw / 'native_scene_iteration_caps_v1.json')['scenes']
    assert len(caps) == 1 and caps[0]['min_velocity'] == 0
    p.mkdir()
    for f in base.iterdir():
        if f.suffix in ['.py', '.usda']:
            shutil.copyfile(f, p / f.name)
    source = (base / 'native_common_beam_v1.py').read_text()
    replacements = {'solver_velocity_iteration_count=128': 'solver_velocity_iteration_count=0',
                    "values['min_velocity']<=128<=values['max_velocity']": "values['min_velocity']<=0<=values['max_velocity']",
                    '(pi,vi)==(32,128)': '(pi,vi)==(32,0)',
                    'position_iterations=32,velocity_iterations=128': 'position_iterations=32,velocity_iterations=0'}
    for before, after in replacements.items():
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    ast.parse(source)
    (p / 'native_common_beam_v1.py').write_text(source)
    launcher = H / 'launch_fourhand_native_probe_v76.py'
    wrapper = H / 'run_isolated_microstep_convergence_v76.py'
    for old, new in [(H / 'launch_fourhand_native_probe_v71.py', launcher), (H / 'run_isolated_microstep_convergence_v71.py', wrapper)]:
        assert not new.exists()
        text = old.read_text().replace('probe_v71', 'probe_v76').replace('PLAN_V71', 'PLAN_V76').replace('convergence-v71', 'convergence-v76')
        ast.parse(text)
        new.write_text(text)
    reg = copy.deepcopy(read(base / 'REGISTRATION_V1.json'))
    reg.update(attempt_version=76, scientific_candidate_version=76, numerical_baseline_version=71,
               registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               native_solver_profile=dict(position_iterations=32, velocity_iterations=0),
               change_scope='Single numerical stage isolation: native TGS velocity iterations128->0, position32 unchanged. Original full50 backward-Euler motor effort, original hardware/inertias/armature/friction, native24 ideal relations, seed and all submitted targets fixed. All original q AND qd<=1e-4 and physical gates unchanged. Short64 diagnostic; no fulltask or new random/formal safety acceptance.')
    for f in [*p.iterdir(), Path(__file__), launcher, wrapper,
              H / 'audit_short_contact_capture_v1.py', H / 'audit_native_mimic_window_v6.py',
              H / 'audit_short_contact_target_tape_v1.py', H / 'audit_full_motor_effort_native_v4.py']:
        reg['source_sha256'][str(f)] = sha(f)
    (p / 'REGISTRATION_V1.json').write_text(json.dumps(reg, indent=2) + '\n')
    source_file = H / 'gpu_mimic_solver_source_v1/PxgTGSCudaSolverCore.cpp'
    receipt = read(H / 'gpu_mimic_solver_source_v1/SOURCE_RECEIPT_V1.json')
    assert receipt['installed_gpu_binary_exact_source_identity_not_proven'] is True
    plan = H / 'TGS_VELOCITY_STAGE76_PROSPECTIVE_PLAN_V1.json'
    assert not plan.exists()
    plan.write_text(json.dumps(dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), baseline=71, candidate=76,
                                   change='TGS velocity iterations128->0; position32 and all original physical/controller/model parameters fixed',
                                   maximum_physics_events=64, maximum_native_wall_seconds=600,
                                   all77_native_parameter_arrays_and13_initial_state_fields_must_equal71=True,
                                   every_original_submitted_target_must_match_fixture=True, both_original_model_tolerances=1e-4,
                                   diagnostic_weak_contact_persistence=dict(F_L_normal_gt_N=.005, continuous_physics_events_ge=20),
                                   require_complete64_original_model_original_physical_and_contact_persistence=True,
                                   native_scene_metadata_min_velocity_zero_previously_observed=True,
                                   actual_binary_iteration_execution_count_not_directly_observed=True,
                                   primary_documentation='https://nvidia-omniverse.github.io/PhysX/physx/5.5.1/docs/RigidBodyDynamics.html',
                                   primary_source='https://github.com/NVIDIA-Omniverse/PhysX/blob/' + receipt['commit'] + '/physx/source/gpusolver/src/PxgTGSCudaSolverCore.cpp#L1524',
                                   rationale='Official documentation distinguishes biased position and unbiased velocity iterations and permits TGS without velocity iterations in suitable scenes. Existing primary GPU source comments describe a changed solver objective in velocity iterations. This motivates an isolated native test, without asserting that the installed binary shares that exact source or that removing the stage is a safe fix.',
                                   installed_binary_exact_source_identity_not_proven=True, no_SDK_patch=True,
                                   fulltask_acceptance=False, new_independent_random_cases=0, formal_holdout_count=0, fullSystem0_accepted=False,
                                   source_sha256={str(f): sha(f) for f in [Path(__file__), p / 'REGISTRATION_V1.json', source_file]}), indent=2) + '\n')
    print('FROZEN76_TGS_VELOCITY_STAGE_ZERO_ORIGINAL_CONTROLLER_GATES', flush=True)


if __name__ == '__main__':
    main()
