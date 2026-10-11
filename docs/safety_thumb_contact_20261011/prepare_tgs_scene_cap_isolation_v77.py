"""Prospective scene-wide velocity cap, after root-only0 equals root-only1."""
from pathlib import Path
import ast
import copy
import datetime
import hashlib
import json
import shutil
import numpy as np

H = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    base = H / 'fourhand_native_probe_v76'
    p = H / 'fourhand_native_probe_v77'
    assert not p.exists()
    assert read(base / 'ACTUAL_COMPLETION_EXIT_V76.json')['actual_outer_exit'] == 0
    assert not read(base / 'INDEPENDENT_NATIVE_MIMIC_WINDOW_V6.json')['model_window_pass']
    source_files = []
    for version in [65, 76]:
        raw = Path(read(H / f'fourhand_native_probe_v{version}/RESOURCE_EXECUTION_V1.json')['out'])
        chunks = read(raw / 'physics_chunks.json')
        assert len(chunks) == 1
        file = raw / chunks[0]['path']
        assert sha(file) == chunks[0]['sha256']
        source_files.append(file)
    with np.load(source_files[0], allow_pickle=False) as a, np.load(source_files[1], allow_pickle=False) as b:
        assert set(a.files) == set(b.files) and len(a.files) == 164
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    parity = base / 'ROOT_VELOCITY0_VS1_ALL164_STATE_PARITY_V1.json'
    assert not parity.exists()
    parity.write_text(json.dumps(dict(status='CLOSED_BITEXACT_ROOT_ONLY_ZERO_VS_ONE_VELOCITY_ITERATION',
                                      versions=[65, 76], physics_events=16, bitexact_fields=164,
                                      inference='The root setting alone did not change this short trajectory. Effective island iteration count and rigid-body requirements were not recorded; that is an unresolved mechanism hypothesis.',
                                      source_sha256={str(f): sha(f) for f in source_files}, fullSystem0_accepted=False), indent=2) + '\n')
    p.mkdir()
    for f in base.iterdir():
        if f.suffix in ['.py', '.usda']:
            shutil.copyfile(f, p / f.name)
    text = (base / 'native_common_beam_v1.py').read_text()
    needle = "        cfg.sim.physx.solver_type=reg['native_solver_strategy']"
    assert text.count(needle) == 1
    text = text.replace(needle, needle + "\n        cfg.sim.physx.max_velocity_iteration_count=0")
    needle = "        save(out/'native_scene_iteration_caps_v1.json',dict(scenes=caps,"
    assert text.count(needle) == 1
    text = text.replace(needle, """        assert caps[0]['min_velocity']==caps[0]['max_velocity']==0
        actor_iterations=[]
        for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                value=prim.GetAttribute('physxRigidBody:solverVelocityIterationCount').Get()
                position=prim.GetAttribute('physxRigidBody:solverPositionIterationCount').Get()
                actor_iterations.append(dict(path=str(prim.GetPath()),
                    USD_velocity_iteration_requirement=int(value) if value is not None else None,
                    USD_position_iteration_requirement=int(position) if position is not None else None))
        save(out/'native_actor_iteration_requirements_v1.json',dict(records=actor_iterations,
            actual_binary_island_iteration_execution_not_directly_observed=True,
            metadata_scene_velocity_cap=0))
""" + needle)
    ast.parse(text)
    (p / 'native_common_beam_v1.py').write_text(text)
    launcher = H / 'launch_fourhand_native_probe_v77.py'
    wrapper = H / 'run_isolated_microstep_convergence_v77.py'
    for old, new in [(H / 'launch_fourhand_native_probe_v76.py', launcher), (H / 'run_isolated_microstep_convergence_v76.py', wrapper)]:
        assert not new.exists()
        source = old.read_text().replace('probe_v76', 'probe_v77').replace('PLAN_V76', 'PLAN_V77').replace('convergence-v76', 'convergence-v77')
        ast.parse(source)
        new.write_text(source)
    reg = copy.deepcopy(read(base / 'REGISTRATION_V1.json'))
    reg.update(attempt_version=77, scientific_candidate_version=77, numerical_baseline_version=76,
               registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               native_scene_velocity_iteration_cap=0,
               change_scope='Single numerical factor versus76: scene maximum velocity iterations255->0, articulation roots already0; additionally read metadata requirements of all rigid bodies. Full50 original motor predictor, actual targets, seed, masses/inertias/armature/materials and all original q/qd/physical gates fixed. Short64 known-state diagnostic, not a new random or fulltask trial.')
    for f in [*p.iterdir(), Path(__file__), launcher, wrapper, H / 'audit_short_contact_capture_v1.py',
              H / 'audit_native_mimic_window_v6.py', H / 'audit_short_contact_target_tape_v1.py', H / 'audit_full_motor_effort_native_v4.py']:
        reg['source_sha256'][str(f)] = sha(f)
    (p / 'REGISTRATION_V1.json').write_text(json.dumps(reg, indent=2) + '\n')
    plan = H / 'TGS_SCENE_CAP77_PROSPECTIVE_PLAN_V1.json'
    assert not plan.exists()
    plan.write_text(json.dumps(dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), baseline=76, candidate=77,
                                   maximum_physics_events=64, maximum_native_wall_seconds=600,
                                   native_root_velocity_iterations=0, native_scene_min_velocity=0, native_scene_max_velocity=0,
                                   all77_native_parameter_arrays_and13_seed_fields_must_equal71=True,
                                   original_full50_controller_and_target_tape_unchanged=True, both_original_model_tolerances=1e-4,
                                   diagnostic_weak_contact_persistence=dict(F_L_normal_gt_N=.005, continuous_physics_events_ge=20),
                                   require_complete64_original_model_original_physical_and_contact_persistence=True,
                                   native_actor_iteration_requirements_observer_metadata_only=True,
                                   installed_binary_exact_source_identity_not_proven=True,
                                   actual_binary_iteration_execution_count_not_directly_observed=True,
                                   hypothesis='Rigid-body requirements within a solver island may keep one velocity iteration after all root settings become0; observe their metadata and clamp the scene maximum to0 before physics starts.',
                                   no_native_physical_property_or_gate_change=True, no_SDK_patch=True,
                                   fulltask_acceptance=False, new_independent_random_cases=0, formal_holdout_count=0, fullSystem0_accepted=False,
                                   source_sha256={str(f): sha(f) for f in [Path(__file__), p / 'REGISTRATION_V1.json', parity]}), indent=2) + '\n')
    print('FROZEN77_SCENE_MAX_VELOCITY_ZERO_ROOTS0_AND_ACTOR_REQUIREMENT_METADATA', flush=True)


if __name__ == '__main__':
    main()
