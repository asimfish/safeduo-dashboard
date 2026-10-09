"""Post hoc localization of the closed second64 pair; never modifies scoring."""
import hashlib
import json
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
ROOT = Path(json.loads((H / 'PAIRED_REGISTRATION_V1.json').read_text())['raw'])
ARMS = ('F_L', 'F_R', 'U_L', 'U_R')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = H / 'BATCH1_JOINT_FAILURE_LOCALIZATION_V1.json'
    assert not output.exists(), 'Preserve this second64 post hoc observation'
    pair = {}
    for method in ('raw', 'multirow'):
        leaf = ROOT / f'paired_batch1_{method}_v8'
        oracle_path = H / f'PAIRED_ORACLE_batch1_{method}_V1.json'
        oracle = json.loads(oracle_path.read_text())
        assert oracle['status'].startswith('PASS_')
        with np.load(leaf / 'resolved_native_parameters.npz') as z:
            params = {k: z[k] for k in z.files}
        with np.load(leaf / 'response_stream.npz') as z:
            common = {k: z[k][:6] for k in ['applied_target'] + ['post_' + a + '_' + f for a in ARMS for f in ('q', 'qd')]}
            common['reference_target'] = z['reference_target']
        receipt = json.loads((leaf / 'point_contact_receipts.json').read_text())
        full_hard = np.zeros(64)
        full_speed = np.zeros(64)
        records = []
        for arm in ARMS:
            names = params[arm + '_joint_names'].tolist()
            n = len(names)
            controlled = set(params[arm + '_controlled_joint_indices'].tolist())
            maxima = {kind: np.zeros((64, n)) for kind in ('hard', 'speed')}
            peaks = {}
            prefix_hard, prefix_speed = np.zeros(64), np.zeros(64)
            arm_hard, arm_speed = np.zeros(64), np.zeros(64)
            hand_hard, hand_speed = np.zeros(64), np.zeros(64)
            for chunk in receipt['chunks']:
                path = leaf / chunk['path']
                assert sha(path) == chunk['sha256']
                with np.load(path) as z:
                    q, v = z[arm + '_native_q'], z[arm + '_native_qd']
                    frames, subs = z['frame'], z['substep']
                bounds = params[arm + '_hard_limits']
                vmax = params[arm + '_max_velocity']
                hard = np.maximum(np.maximum(bounds[None, ..., 0] - q, q - bounds[None, ..., 1]), 0.)
                speed = np.maximum(abs(v) - vmax[None], 0.)
                assert np.isfinite(hard).all() and np.isfinite(speed).all()
                for kind, values, state in [('hard', hard, q), ('speed', speed, v)]:
                    maxima[kind] = np.maximum(maxima[kind], values.max(0))
                    event, lane, joint = np.unravel_index(values.argmax(), values.shape)
                    value = float(values[event, lane, joint])
                    if kind not in peaks or value > peaks[kind]['exceedance']:
                        peaks[kind] = dict(exceedance=value, macro=int(frames[event]), substep=int(subs[event]),
                            lane=int(lane), input_id=int(params['global_input_id'][lane]), joint=names[joint],
                            joint_index=int(joint), controlled_arm_joint=joint in controlled,
                            native_value=float(state[event, lane, joint]),
                            native_hard_interval_rad=bounds[lane, joint].tolist(),
                            native_max_velocity_rad_s=float(vmax[lane, joint]))
                prefix_mask = frames < 6
                if prefix_mask.any():
                    prefix_hard = np.maximum(prefix_hard, hard[prefix_mask].max((0, 2)))
                    prefix_speed = np.maximum(prefix_speed, speed[prefix_mask].max((0, 2)))
                arm_idx = sorted(controlled)
                hand_idx = sorted(set(range(n)) - controlled)
                arm_hard = np.maximum(arm_hard, hard[:, :, arm_idx].max((0, 2)))
                arm_speed = np.maximum(arm_speed, speed[:, :, arm_idx].max((0, 2)))
                hand_hard = np.maximum(hand_hard, hard[:, :, hand_idx].max((0, 2)))
                hand_speed = np.maximum(hand_speed, speed[:, :, hand_idx].max((0, 2)))
            full_hard = np.maximum(full_hard, maxima['hard'].max(-1))
            full_speed = np.maximum(full_speed, maxima['speed'].max(-1))
            records.append(dict(arm=arm, total_dof=n, controlled_dof=len(controlled), peaks=peaks,
                prefix_hard_bad_lanes=int((prefix_hard > 1e-5).sum()), prefix_speed_bad_lanes=int((prefix_speed > 1e-5).sum()),
                controlled_hard_bad_lanes=int((arm_hard > 1e-5).sum()), controlled_speed_bad_lanes=int((arm_speed > 1e-5).sum()),
                hand_hard_bad_lanes=int((hand_hard > 1e-5).sum()), hand_speed_bad_lanes=int((hand_speed > 1e-5).sum()),
                per_joint=[dict(name=name, controlled=j in controlled,
                    hard_bad_lanes=int((maxima['hard'][:, j] > 1e-5).sum()), speed_bad_lanes=int((maxima['speed'][:, j] > 1e-5).sum()),
                    max_hard_breach_rad=float(maxima['hard'][:, j].max()), max_speed_exceedance_rad_s=float(maxima['speed'][:, j].max()))
                    for j, name in enumerate(names)]))
        assert np.array_equal(full_hard, [s['supplementary_max_all74_hard_violation_rad'] for s in oracle['states']])
        assert np.array_equal(full_speed, [s['supplementary_max_all74_velocity_exceedance_rad_s'] for s in oracle['states']])
        pair[method] = dict(arms=records, states=oracle['states'], source_oracle_sha256=sha(oracle_path),
            source_point_receipt_sha256=sha(leaf / 'point_contact_receipts.json'), common=common)
    for key in pair['raw']['common']:
        assert np.array_equal(pair['raw']['common'][key], pair['multirow']['common'][key]), key
    for value in pair.values():
        del value['common']
    raw, multi = pair['raw']['states'], pair['multirow']['states']
    assert [s['input_id'] for s in raw] == [s['input_id'] for s in multi]
    result = dict(status='PASS_CLOSED_SECOND64_JOINT_LOCALIZATION', scope='Post hoc second64 only; no 128 outcome inferred',
        source_sha256=sha(Path(__file__)), both_common_prefix_full74_q_qd_and_target_exact=True,
        all480_reference_targets_exact=True, original_primary_and_supplementary_scores_unchanged=True,
        rescued_input_ids=[r['input_id'] for r, m in zip(raw, multi) if r['primary_failure'] and not m['primary_failure']],
        new_failure_input_ids=[r['input_id'] for r, m in zip(raw, multi) if not r['primary_failure'] and m['primary_failure']],
        methods=pair, safety_acceptance=False)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(result['status'], 'rescues', result['rescued_input_ids'], 'new', result['new_failure_input_ids'], flush=True)
    for method, value in pair.items():
        for arm in value['arms']:
            print(method, arm['arm'], 'hard', arm['peaks']['hard'], 'speed', arm['peaks']['speed'], flush=True)


if __name__ == '__main__':
    main()
