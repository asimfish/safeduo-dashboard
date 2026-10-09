"""Closed diagnostic comparison; inactivity is reported rather than called success."""
import hashlib
import json
from pathlib import Path

import numpy as np

import analyse_paired128_v1 as independent

H = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    reg = json.loads((H / 'HOLD_FALLBACK_DEV_REG_V1.json').read_text())
    execution = json.loads((H / 'hold_fallback64_v1_execution.json').read_text())
    assert execution['status'] == 'complete' and len(execution['jobs']) == 1
    assert execution['jobs'][0]['actual_exit'] == 0
    independent.ROOT = Path(reg['audit_namespace'])
    method = 'multirow_hold_fallback'
    independent.audit(0, method)
    leaf = Path(reg['native_out'])
    with np.load(leaf / 'response_stream.npz') as z:
        stream = {k: z[k] for k in z.files}
    with np.load(leaf / 'resolved_native_parameters.npz') as z:
        params = {k: z[k] for k in z.files}
    initial = np.concatenate([params[a + '_initial_q'][:, params[a + '_controlled_joint_indices']] for a in independent.ARMS], -1)
    blocked = stream['fallback_from_model_unsatisfied']
    assert np.array_equal(blocked, ~stream['multi_model_constraints_satisfied'])
    expected = np.where(blocked[..., None], initial[None], stream['nominal_issued_target'])
    assert np.array_equal(stream['issued_target'], expected)
    prior_root = Path(reg['paired_raw_root'])
    for method0 in ('raw', 'multirow'):
        with np.load(prior_root / f'paired_batch0_{method0}_v8/response_stream.npz') as z:
            assert np.array_equal(stream['reference_target'], z['reference_target'])
            for key in ['applied_target'] + ['post_' + a + '_' + f for a in independent.ARMS for f in ('q', 'qd')]:
                assert np.array_equal(stream[key][:6], z[key][:6]), 'Unmatched common prefix: ' + key
    oracles = {}
    for method0 in ('raw', 'multirow', 'multirow_hold_fallback'):
        path = H / f'PAIRED_ORACLE_batch0_{method0}_V1.json'
        oracles[method0] = json.loads(path.read_text())
    states = oracles['multirow_hold_fallback']['states']
    raw = oracles['raw']['states']
    assert [r['input_id'] for r in states] == [r['input_id'] for r in raw]
    counts = {}
    for method0, oracle in oracles.items():
        rows = oracle['states']
        counts[method0] = dict(states=len(rows), primary_failures=sum(r['primary_failure'] for r in rows),
            geometry_failures=sum(r['geometry_failure'] for r in rows), force_failures=sum(r['force_failure'] for r in rows),
            all74_hard_bad=sum(r['supplementary_max_all74_hard_violation_rad'] > 1e-5 for r in rows),
            all74_speed_bad=sum(r['supplementary_max_all74_velocity_exceedance_rad_s'] > 1e-5 for r in rows),
            all4_arms_moved=sum(r['four_arms_moved'] for r in rows),
            total_controlled_path_rad=sum(r['controlled_path_rad'] for r in rows),
            joint_and_primary_pass=sum(not r['primary_failure'] and r['supplementary_max_all74_hard_violation_rad'] <= 1e-5 and r['supplementary_max_all74_velocity_exceedance_rad_s'] <= 1e-5 for r in rows))
    ratios = [r['controlled_path_rad'] / baseline['controlled_path_rad'] for r, baseline in zip(states, raw)]
    result = dict(status='CLOSED_FIRST64_MODEL_INVALID_INITIAL_HOLD_DEVELOPMENT_DIAGNOSTIC',
        registration_sha256=sha(H / 'HOLD_FALLBACK_DEV_REG_V1.json'), wrapper_source_sha256=sha(Path(__file__)),
        closed_before_new_policy_adoption=True, all960_point_geometry_FIFO_oracle=True,
        all480_disposition_exact=True, full74_common_prefix_exact=True, frozen_reference_tape_exact=True,
        counts=counts, fallback_lane_controls=int(blocked.sum()), total_lane_controls=int(blocked.size),
        trajectory_path_ratio_to_raw=dict(min=float(min(ratios)), median=float(np.median(ratios)), max=float(max(ratios))),
        all_inputs_retained=True, raw_task_success_evaluated=False,
        interpretation='Holding an initial target is an unproven backup attempt. Any risk reduction accompanied by suppression is not cooperative task success.',
        full_system0_accepted=False, no_hardware_or_trained_actor=True)
    (H / 'HOLD_FALLBACK64_RESULT_V1.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(result['status'], json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
