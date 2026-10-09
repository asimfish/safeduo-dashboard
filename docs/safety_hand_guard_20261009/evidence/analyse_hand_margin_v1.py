"""Independent fixed-position64 hand-margin admission diagnostic; no adoption."""
import json
from pathlib import Path

import numpy as np

from analyse_prefix512_v2 import oracle, sha

H = Path(__file__).resolve().parent
ARMS = ('F_L', 'F_R', 'U_L', 'U_R')


def load(path):
    with np.load(path) as z:
        return {k:z[k] for k in z.files}


def main():
    reg = json.loads((H / 'HAND_MARGIN_DIAGNOSTIC_REG_V1.json').read_text())
    execution = json.loads((H / 'hand_margin64_v1_execution.json').read_text())
    assert execution['status'] == 'complete' and len(execution['jobs']) == 4
    assert all(j['status'] == 'complete' and j['actual_exit'] == 0 for j in execution['jobs'])
    reference = load(Path(reg['reference_inputs']))
    base_output = load(Path(reg['reference_outputs']))
    rows = []
    for job, margin in zip(execution['jobs'], [0., .01, .03, .05]):
        root = Path(job['out'])
        p = load(root / 'prefix_inputs.npz')
        r = load(root / 'prefix_result.npz')
        protocol = json.loads((root / 'prefix_protocol.json').read_text())
        assert protocol['status'] == 'complete' and protocol['hand_margin_rad'] == margin
        assert protocol['prefix_inputs_sha256'] == sha(root / 'prefix_inputs.npz')
        assert protocol['prefix_result_sha256'] == sha(root / 'prefix_result.npz')
        roots = json.loads((root / 'actual_solver_roots.json').read_text())
        assert len(roots) == 4 and all(x['position_iterations'] == 64 and x['velocity_iterations'] == 0 for x in roots)
        for arm in ARMS:
            idx = p[arm + '_controlled_joint_indices']
            hand = np.ones(len(p[arm + '_joint_names']), bool)
            hand[idx] = False
            bounds = p[arm + '_hard_limits']
            for field in ['initial_qd', 'controlled_joint_indices', 'hard_limits', 'vmax', 'kp', 'kd', 'effort', 'armature']:
                assert np.array_equal(p[arm+'_'+field], reference[arm+'_'+field]), (margin, arm, field)
            assert np.array_equal(p[arm+'_initial_q'][:, idx], reference[arm+'_initial_q'][:, idx])
            assert np.array_equal(p[arm+'_full_hold_target'][:, idx], reference[arm+'_full_hold_target'][:, idx])
            for field in ['initial_q', 'full_hold_target']:
                expected = np.clip(reference[arm+'_'+field][:, hand], bounds[:, hand, 0] + np.float32(margin), bounds[:, hand, 1] - np.float32(margin))
                assert np.array_equal(p[arm+'_'+field][:, hand], expected), (margin, arm, field)
        point = oracle(root, r, p)
        geometry = r['all_raw_geometry']
        force = r['scalar_normal_max_N']
        hard = r['supplementary_hard_limit_violation_rad']
        vel = r['supplementary_velocity_limit_violation_rad_s']
        assert geometry.shape == (12, 64, 9021)
        assert all(np.isfinite(value).all() for value in [geometry, force, hard, vel, p['initial_all_raw_geometry']])
        initial_ok = p['initial_all_raw_geometry'].min(-1) >= .0001
        primary = initial_ok & (geometry.min((0, 2)) >= 0) & (force.max(0) <= .1)
        assert np.array_equal(primary, r['admitted'])
        full = primary & (hard.max(0) <= 1e-5) & (vel.max(0) <= 1e-5)
        rows.append(dict(hand_margin_rad=margin, source_states=64, initial_geometry_admitted=int(initial_ok.sum()),
            primary_admitted=int(primary.sum()), all74_hard_bad=int((hard.max(0)>1e-5).sum()),
            all74_speed_bad=int((vel.max(0)>1e-5).sum()), full_prefix_admitted=int(full.sum()),
            maximum_hard_breach_rad=float(hard.max()), all64_peak_normal_N=float(force.max()),
            point_oracle=point, full_prefix_input_ids=p['global_input_id'][full].tolist(),
            per_input=[dict(input_id=int(i), initial_geometry=bool(g), primary=bool(a), full_prefix=bool(f),
                hard_max_rad=float(h), speed_exceedance_max_rad_s=float(v)) for i,g,a,f,h,v in zip(p['global_input_id'], initial_ok, primary, full, hard.max(0),vel.max(0))],
            inputs_sha256=sha(root/'prefix_inputs.npz'), outputs_sha256=sha(root/'prefix_result.npz')))
        if margin == 0:
            for key in ['all_raw_geometry','scalar_normal_max_N','supplementary_hard_limit_violation_rad','supplementary_velocity_limit_violation_rad_s','admitted']:
                assert np.array_equal(r[key], base_output[key]), 'Baseline reproduction ' + key
    result = dict(status='CLOSED_NATIVE_HAND_MARGIN64_DEVELOPMENT_DIAGNOSTIC', rows=rows,
        factor='Hand initial positions and held targets inset from native hard limits by registered margin; positioniterations fixed64',
        all4_native_actual_exit0=True, all4_point_oracles_pass=True, baseline_margin0_reproduced_exact=True,
        original64_denominator_retained=True, no_threshold_or_row_exemption_change=True,
        registration_sha256=sha(H/'HAND_MARGIN_DIAGNOSTIC_REG_V1.json'), analysis_sha256=sha(Path(__file__)),
        complete_controller_or_object_task_evaluated=False, production_configuration_adopted=False,
        fullSystem0_accepted=False, interpretation='Only6control/12physics waiting steps on one known development bank; no480-step safety, holdout, grasp or hardware claim')
    (H/'HAND_MARGIN64_RESULT_V1.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(result['status'], [{k:r[k] for k in ['hand_margin_rad','initial_geometry_admitted','primary_admitted','all74_hard_bad','full_prefix_admitted']} for r in rows], flush=True)


if __name__ == '__main__':
    main()
