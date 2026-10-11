"""Prospective two-coordinate causal isolation, never a grasp safety acceptance."""
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
    assert read(H / 'fourhand_native_probe_v73/ACTUAL_COMPLETION_EXIT_V73.json')['actual_outer_exit'] == 0
    assert not read(H / 'fourhand_native_probe_v73/INDEPENDENT_NATIVE_MIMIC_WINDOW_V8.json')['model_window_pass']
    oracle = H / 'audit_single_thumb_pd_effort_v1.py'
    assert not oracle.exists()
    text = (H / 'audit_full_motor_effort_native_v4.py').read_text()
    needle = "                recorded = data[arm + '_software_full_motor_PD_preclip']"
    assert text.count(needle) == 1
    text = text.replace(needle, """                selected = reg['isolated_zero_PD_motor']
                assert selected in [dict(arm='F_L', joint='left_thumb_2_joint'), dict(arm='F_L', joint='left_thumb_1_joint')]
                original_roots, _, _ = matrix(asset['hand_joint_names'], asset['original_URDF'])
                if arm == selected['arm']:
                    assert selected['joint'] in original_roots
                    paused_id = names.index(selected['joint'])
                    assert paused_id in roots and paused_id not in followers
                    pd[..., paused_id] = 0.
""" + needle)
    text = text.replace('PASS_INDEPENDENT_FULL50_ORIGINAL_FREE_PREDICTOR_WITHOUT_NATIVE_CORIOLIS_QUERY', 'PASS_FULL50_INDEPENDENT_IMPULSE_ORACLE_WITH_DECLARED_ONE_THUMB_PD_ZERO')
    text = text.replace("p / 'INDEPENDENT_FULL_MOTOR_EFFORT_V4.json'", "p / 'INDEPENDENT_SINGLE_THUMB_PD_EFFORT_V1.json'")
    ast.parse(text)
    oracle.write_text(text)
    specs = {74: 'left_thumb_2_joint', 75: 'left_thumb_1_joint'}
    evidence = {}
    for version, joint in specs.items():
        p = H / f'fourhand_native_probe_v{version}'
        assert not p.exists()
        p.mkdir()
        for f in base.iterdir():
            if f.suffix in ['.py', '.usda']:
                shutil.copyfile(f, p / f.name)
        worker = (base / 'native_common_beam_v1.py').read_text()
        needle = '                    tau=solution.torque'
        assert worker.count(needle) == 1
        worker = worker.replace(needle, """                    tau=solution.torque.copy()
                    selected=reg['isolated_zero_PD_motor']
                    if a==selected['arm']:
                        native_names=parameters[a+'_joint_names'].tolist()
                        selected_id=native_names.index(selected['joint'])
                        assert selected_id in rootids
                        tau[rootids.index(selected_id)]=0.
""".rstrip())
        ast.parse(worker)
        (p / 'native_common_beam_v1.py').write_text(worker)
        launcher = H / f'launch_fourhand_native_probe_v{version}.py'
        wrapper = H / f'run_isolated_microstep_convergence_v{version}.py'
        for old, new in [(H / 'launch_fourhand_native_probe_v71.py', launcher), (H / 'run_isolated_microstep_convergence_v71.py', wrapper)]:
            assert not new.exists()
            source = old.read_text().replace('probe_v71', f'probe_v{version}').replace('PLAN_V71', f'PLAN_V{version}').replace('convergence-v71', f'convergence-v{version}')
            ast.parse(source)
            new.write_text(source)
        reg = copy.deepcopy(read(base / 'REGISTRATION_V1.json'))
        reg.update(attempt_version=version, scientific_candidate_version=version,
                   numerical_baseline_version=71, registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   isolated_zero_PD_motor=dict(arm='F_L', joint=joint),
                   change_scope='One original independent F_L thumb motor software PD term set zero after the full50 free predictor; gravity and other49 motor formulas unchanged. All original24 ideal relations, native assets/masses/materials/clock, initial seed and submitted position target tape unchanged. Short64 microstep causal diagnostic. No model or physical threshold change, no new random/formal case, no grasp or learned System0 acceptance.')
        for f in [*p.iterdir(), Path(__file__), launcher, wrapper, oracle,
                  H / 'audit_short_contact_capture_v1.py', H / 'audit_short_contact_target_tape_v1.py', H / 'audit_native_mimic_window_v6.py']:
            reg['source_sha256'][str(f)] = sha(f)
        (p / 'REGISTRATION_V1.json').write_text(json.dumps(reg, indent=2) + '\n')
        evidence[str(version)] = dict(isolated_zero_PD_motor=reg['isolated_zero_PD_motor'], registration_sha256=sha(p / 'REGISTRATION_V1.json'),
                                      worker_sha256=sha(p / 'native_common_beam_v1.py'))
    plan = H / 'SINGLE_THUMB_PD74_75_PROSPECTIVE_PLAN_V1.json'
    assert not plan.exists()
    plan.write_text(json.dumps(dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), baseline=71, candidates=evidence,
                                   launch_serially=True, selected_GPU=1, all77_native_parameter_arrays_and_initial_states_must_equal71=True,
                                   submitted_full_target_tape_must_match_original_reproducer=True,
                                   maximum_physics_events=64, maximum_native_wall_seconds=600,
                                   both_original_model_tolerances=1e-4, all_original_physical_gates_unchanged=True,
                                   diagnostic_weak_contact_persistence=dict(F_L_normal_gt_N=.005, continuous_physics_events_ge=20),
                                   require_complete64_and_original_model_and_physical_and_persistence_for_short_diagnostic_pass=True,
                                   no_fulltask_grasp_or_safe_stop_qualification_from_this_short_diagnostic=True,
                                   inference_scope='Known precontact seed, hidden solver warm starts not restored. Tests isolate necessity of each original thumb PD term, not sufficient full-load safety.',
                                   causal_hypothesis='Determine whether original thumb2 flexion PD or thumb1 opposition PD is necessary for short weak-contact mimic velocity failure. Preserve the other motor rather than pausing all6 hand motors.',
                                   frozen_decision_checker=str(H / 'audit_single_thumb_pd_decision_v1.py'),
                                   decision_checker_sha256=sha(H / 'audit_single_thumb_pd_decision_v1.py'),
                                   new_independent_random_cases=0, formal_holdout_count=0, fullSystem0_accepted=False,
                                   source_sha256={str(p): sha(p) for p in [Path(__file__), oracle]}), indent=2) + '\n')
    print('FROZEN_SINGLE_THUMB_PD74_75_ORIGINAL_GATES_64_EVENT_CAUSAL_DIAGNOSTICS', flush=True)


if __name__ == '__main__':
    main()
