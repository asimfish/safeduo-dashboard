"""Post-run trajectory difference from same-state holding; original scores stay fixed."""
import hashlib
import json
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent
H = P.parent
G = H / 'gravity_hold128_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def main():
    output = P / 'COMMAND_EFFECT_DESCRIPTION_V1.json'
    assert not output.exists()
    finished = json.loads((P / 'FINISH_EXECUTION_V6.json').read_text())
    assert finished['status'] == 'complete' and all(s['actual_exit'] == 0 for s in finished['steps'])
    reg = json.loads((P / 'REGISTRATION_V1.json').read_text())
    greg = json.loads((G / 'REGISTRATION_V1.json').read_text())
    rows = []
    sources = {str(Path(__file__).resolve()): sha(Path(__file__).resolve())}
    slices = [slice(0, 7), slice(7, 14), slice(14, 20), slice(20, 26)]
    arms = ['F_L', 'F_R', 'U_L', 'U_R']
    for batch in range(2):
        held_path = G / f'PAIRED_METRICS_batch{batch}_on_V1.npz'
        held = load(held_path)
        sources[str(held_path)] = sha(held_path)
        held_params_path = Path(greg['raw']) / f'paired_batch{batch}_on_v8/resolved_native_parameters.npz'
        held_params = load(held_params_path)
        sources[str(held_params_path)] = sha(held_params_path)
        for mode in reg['action_modes']:
            metrics_path = P / f'PAIRED_METRICS_batch{batch}_{mode}_V1.npz'
            metrics = load(metrics_path)
            leaf = Path(reg['raw']) / f'paired_batch{batch}_{mode}_v8'
            params_path = leaf / 'resolved_native_parameters.npz'
            params = load(params_path)
            stream_path = leaf / 'response_stream.npz'
            stream = load(stream_path)
            sources.update({str(p): sha(p) for p in [metrics_path, params_path, stream_path]})
            assert set(params) == set(held_params)
            assert all(np.array_equal(params[k], held_params[k]) for k in params)
            assert np.array_equal(metrics['global_input_id'], held['global_input_id'])
            assert np.array_equal(metrics['controlled_q'][:7], held['controlled_q'][:7]), 'same first6 actual controls required'
            difference = metrics['controlled_q'] - held['controlled_q']
            issued_delta = stream['issued_target'] - stream['queued_centre']
            for lane, input_id in enumerate(metrics['global_input_id']):
                per_arm = {}
                for arm, sl in zip(arms, slices):
                    per_arm[arm] = dict(max_trajectory_difference_from_hold_rad=float(abs(difference[:, lane, sl]).max()),
                        final_L1_difference_from_hold_rad=float(abs(difference[-1, lane, sl]).sum()),
                        issued_target_L1_step_rad=float(abs(issued_delta[:, lane, sl]).sum()))
                rows.append(dict(mode=mode, input_id=int(input_id), cell=int(metrics['cell_id'][lane]), arms=per_arm,
                    all4_commanded=all(v['issued_target_L1_step_rad'] > 1e-6 for v in per_arm.values()),
                    all4_observed_difference_gt1mrad=all(v['max_trajectory_difference_from_hold_rad'] > .001 for v in per_arm.values())))
    assert len(rows) == 256
    result = dict(status='COMPLETE_POSTRUN_COMMAND_EFFECT_DESCRIPTION', source_sha256=sources, states=rows,
        counts={m:dict(inputs=128, all4_commanded=sum(r['all4_commanded'] for r in rows if r['mode'] == m),
                      all4_trajectory_differences_gt1mrad=sum(r['all4_observed_difference_gt1mrad'] for r in rows if r['mode'] == m))
                for m in reg['action_modes']},
        original_native_and_independent_verdicts_unchanged=True, new_native_trials=0,
        first6_physical_controls_bitexact_with_same_gravity_holding=True,
        scope='Descriptive post-run comparison cancels shared initial inertia using exact same-state holding '
              'reference.1mrad is a display threshold, not a formal task/intent acceptance gate. '
              'Trajectory difference can include unsafe motion; consult original safety scores.',
        full_system0_accepted=False)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(result['status'], result['counts'], flush=True)


if __name__ == '__main__':
    main()
