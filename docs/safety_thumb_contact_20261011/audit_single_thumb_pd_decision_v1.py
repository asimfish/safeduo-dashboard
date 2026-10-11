"""Prospectively fixed weak-contact diagnostic decision, no safety acceptance."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from audit_short_hand_pd_filter_decision_v1 import longest

H = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(version):
    assert version in [74, 75]
    p = H / f'fourhand_native_probe_v{version}'
    output = p / 'SINGLE_THUMB_PD_DIAGNOSTIC_DECISION_V1.json'
    assert not output.exists()
    plan = read(H / 'SINGLE_THUMB_PD74_75_PROSPECTIVE_PLAN_V1.json')
    assert plan['decision_checker_sha256'] == sha(Path(__file__))
    actual = read(p / f'ACTUAL_COMPLETION_EXIT_V{version}.json')
    for k in ['actual_outer_exit', 'actual_native_exit', 'actual_finite_exit', 'actual_launcher_subprocess_exit']:
        assert type(actual[k]) is int and actual[k] == 0
    for name in ['INDEPENDENT_CAPTURE_AUDIT_V1', 'INDEPENDENT_NATIVE_MIMIC_WINDOW_V6', 'INDEPENDENT_SINGLE_THUMB_PD_EFFORT_V1', 'INDEPENDENT_SHORT_CONTACT_TARGET_TAPE_V1']:
        assert read(p / (name + '_ACTUAL_EXIT.json'))['actual_outer_exit'] == 0
    capture = read(p / 'INDEPENDENT_CAPTURE_AUDIT_V1.json')
    model = read(p / 'INDEPENDENT_NATIVE_MIMIC_WINDOW_V6.json')
    assert capture['record_integrity_verified'] and capture['all_owned_native_children_reaped']
    raw = Path(read(p / 'RESOURCE_EXECUTION_V1.json')['out'])
    baseline = Path(read(H / 'fourhand_native_probe_v71/RESOURCE_EXECUTION_V1.json')['out'])
    for name in ['native_parameters.npz', 'actual_initial_seed_binding.npz']:
        with np.load(raw / name, allow_pickle=False) as candidate, np.load(baseline / name, allow_pickle=False) as control:
            assert set(candidate.files) == set(control.files)
            for key in candidate.files:
                np.testing.assert_array_equal(candidate[key], control[key], err_msg=key)
    sources = [Path(__file__), H / 'SINGLE_THUMB_PD74_75_PROSPECTIVE_PLAN_V1.json', p / 'REGISTRATION_V1.json']
    forces = []
    for chunk in read(raw / 'physics_chunks.json'):
        file = raw / chunk['path']
        assert sha(file) == chunk['sha256']
        sources.append(file)
        with np.load(file, allow_pickle=False) as z:
            forces.append(z['four_hand_normal_N'].copy())
    force = np.concatenate(forces)
    assert len(force) == capture['physics_events']
    criterion = plan['diagnostic_weak_contact_persistence']
    assert criterion == dict(F_L_normal_gt_N=.005, continuous_physics_events_ge=20)
    count = longest(force[:, 0] > criterion['F_L_normal_gt_N'])
    checks = dict(complete64_events=len(force) == 64, original_model=model['model_window_pass'],
                  original_physical=capture['physical_gates_pass'], diagnostic_weak_contact_persistence=count >= 20)
    result = dict(status='CLOSED_SINGLE_THUMB_PD_CAUSAL_DIAGNOSTIC', version=version,
                  isolated_zero_PD_motor=read(p / 'REGISTRATION_V1.json')['isolated_zero_PD_motor'],
                  checks=checks, short_diagnostic_pass=all(checks.values()), blockers=[k for k, v in checks.items() if not v],
                  physics_events=len(force), maximum_F_L_contact_N=float(force[:, 0].max()),
                  F_L_positive_contact_events=int((force[:, 0] > 0).sum()),
                  F_L_events_above_original_point1_N=int((force[:, 0] > .1).sum()),
                  F_L_longest_diagnostic_contact_events=count,
                  original_q_residual_rad=model['maximum_q_residual_rad'], original_qd_residual_rad_s=model['maximum_qd_residual_rad_s'],
                  paired_all77_native_arrays_and_initial_states_exact=True, target_tape_independently_verified=True,
                  candidate_adopted=False, original_loaded_grasp_qualified=False, fulltask_qualified=False,
                  learned_system0_tested=False, physical_safe_stop_certified=False, new_independent_random_cases=0,
                  formal_holdout_count=0, all9_jointly_accepted=False, fullSystem0_accepted=False,
                  source_sha256={str(file): sha(file) for file in sources})
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'], version, checks, 'maxqd', result['original_qd_residual_rad_s'], 'contact_events', count, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('version', type=int)
    main(parser.parse_args().version)
