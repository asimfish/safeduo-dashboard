"""Bind every first geometric failure to the original native target clock."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def main(directory):
    reg = json.loads((directory / 'REGISTRATION.json').read_text())
    root = Path(reg['root'])
    for name, expected in reg['source_receipts'].items():
        assert sha(root / name) == expected, name
    cell = read(root / 'cell_001.npz')
    identity = json.loads((root / 'full_row_identity.json').read_text())
    native = json.loads((root / 'native_receipts.json').read_text())
    forecast = json.loads((root / 'forecast_receipts.json').read_text())
    snapshots = json.loads((root / 'first_failure_receipts.json').read_text())
    cache, checked, events = {}, {}, []
    arm_names = ['F_L', 'F_R', 'U_L', 'U_R']

    def original_frame(step):
        entry = next(e for e in native['chunks'] if e['start'] <= step < e['stop'])
        if entry['path'] not in cache:
            path = root / entry['path']
            assert sha(path) == entry['sha256']
            cache[entry['path']] = read(path)
            checked[entry['path']] = entry['sha256']
        return cache[entry['path']], step - entry['start']

    def controlled(data, phase, kind, offset, env):
        values = []
        for arm in arm_names:
            prefix = phase + '_' + arm + '_'
            if kind in ['q', 'qd']:
                indices = data[prefix + 'controlled_joint_indices'][offset]
                value = data[prefix + 'native_' + kind][offset, env, indices]
            elif kind == 'pending':
                value = data[prefix + 'pending_controlled_targets'][offset, :, env]
            else:
                value = data[prefix + kind + '_controlled_target'][offset, env]
            values.append(value)
        return np.concatenate(values, axis=-1)

    for entry in snapshots['receipts']:
        path = root / entry['path']
        assert sha(path) == entry['sha256']
        checked[entry['path']] = entry['sha256']
        snap = read(path)
        step = entry['step']
        data, offset = original_frame(step)
        old, old_offset = original_frame(step - 6) if step >= 6 else (None, None)
        fentry = next(e for e in forecast['chunks'] if e['start'] <= step < e['stop'])
        assert sha(root / fentry['path']) == fentry['sha256']
        checked[fentry['path']] = fentry['sha256']
        with np.load(root / fentry['path'], allow_pickle=False) as stored:
            full_d = stored['measured_d'][step - fentry['start']]
            full_v = stored['velocity'][step - fentry['start']]
        for index, env in enumerate(snap['env_ids']):
            env = int(env)
            pre_q = controlled(data, 'pre', 'q', offset, env)
            pre_qd = controlled(data, 'pre', 'qd', offset, env)
            post_q = controlled(data, 'post', 'q', offset, env)
            pending = controlled(data, 'pre', 'pending', offset, env)
            issued_before = controlled(data, 'pre', 'issued', offset, env)
            issued_now = controlled(data, 'physics', 'issued', offset, env)
            applied = controlled(data, 'physics', 'applied', offset, env)
            expected_pre_q = cell['q_initial'][env] if step == 0 else cell['q'][step - 1, env]
            comparisons = {
                'pre_q': (pre_q, snap['pre_q'][index]),
                'pre_q_vs_previous_post': (pre_q, expected_pre_q),
                'pre_qd': (pre_qd, snap['snapshot_qd'][index]),
                'pre_qd_vs_cell': (pre_qd, cell['pre_qd_compact'][step, env]),
                'post_q': (post_q, snap['post_q'][index]),
                'post_q_vs_cell': (post_q, cell['q'][step, env]),
                'post_qd': (controlled(data, 'post', 'qd', offset, env), snap['post_qd'][index]),
                'pending': (pending, snap['pre_pending_actuator_targets'][index]),
                'pending_vs_cell': (pending, cell['pre_pending_actuator_targets'][step, :, env]),
                'issued_before': (issued_before, snap['pre_issued_target'][index]),
                'issued_now': (issued_now, cell['controller_target'][step, env]),
                'applied': (applied, cell['actuator_target'][step, env]),
                'fifo_head_applied': (applied, pending[0]),
                'physics_pre_q_unchanged': (controlled(data, 'physics', 'q', offset, env), pre_q),
            }
            if old is not None:
                comparisons['issued_six_steps_earlier_applied'] = (
                    applied, controlled(old, 'physics', 'issued', old_offset, env))
            for label, (actual, expected) in comparisons.items():
                assert np.array_equal(actual, expected), (step, env, label)
            increment_error = float(np.max(np.abs(
                (issued_before + snap['returned_cmd'][index]) - issued_now)))
            assert increment_error <= 5e-7, (step, env, 'issued_increment', increment_error)
            for phase in ['pre', 'physics', 'post']:
                assert int(data[phase + '_frame'][offset]) == step
            bad_rows = np.flatnonzero((snap['post_full_d'][index] < 0) &
                                     ~snap['post_full_exempt'][index])
            rows = []
            for row in bad_rows:
                row = int(row)
                positions = np.flatnonzero(snap['selected_ids'][index] == row)
                pair = identity['pair_sphere_idx'][row]
                cls = int(identity['class_id'][row])
                names = [identity['sphere_names'][pair[0]]]
                names.append('table_face_' + str(pair[1]) if cls == 2 else identity['sphere_names'][pair[1]])
                item = dict(row=row, class_id=cls, partners=names,
                            selected=bool(len(positions)), post_gap_m=float(snap['post_full_d'][index, row]),
                            pre_gap_m=float(full_d[env, row]),
                            raw_signed_rate_mps=float(full_v[env, row]))
                if len(positions):
                    pos = int(positions[0])
                    assert snap['snapshot_valid'][index, pos]
                    assert snap['snapshot_d'][index, pos] == full_d[env, row]
                    jac = np.concatenate([snap['snapshot_J_F'][index, pos],
                                          snap['snapshot_J_U'][index, pos]]).astype(np.float64)
                    projected_rate = float(jac @ pre_qd)
                    assert abs(projected_rate - full_v[env, row]) <= 2e-6
                    item.update(
                        selected_position=pos,
                        native_Jqd_mps=projected_rate,
                        raw_rate_binding_abs_error_mps=abs(projected_rate - float(full_v[env, row])),
                        J_actual_q_increment_m=float(jac @ (post_q - pre_q)),
                        actual_gap_increment_m=float(snap['post_full_d'][index, row] - full_d[env, row]),
                        J_returned_target_increment_m=float(jac @ snap['returned_cmd'][index]),
                        J_new_issued_offset_m=float(jac @ (issued_now - pre_q)),
                        J_actually_applied_offset_m=float(jac @ (applied - pre_q)),
                        J_pending_offsets_m=((pending - pre_q) @ jac).tolist(),
                        effective_dmin_m=float(snap['snapshot_dmin'][index, pos]),
                        row_gate=bool(snap['snapshot_row_gate'][index, pos]),
                    )
                rows.append(item)
            root_changes = {}
            for arm in arm_names:
                before = data['pre_' + arm + '_native_root_xyzw'][offset, env]
                after = data['post_' + arm + '_native_root_xyzw'][offset, env]
                root_changes[arm] = float(np.max(np.abs(after - before)))
            events.append(dict(env=env, first_failure_step=step,
                               initial_geometry_negative=bool(cell['initial_violation'][env]),
                               snapshot_sha256=entry['sha256'], native_clock_all_exact=True,
                               native_comparisons=list(comparisons), target_increment_max_error=increment_error,
                               first_completed_macro_receiving_current_issued_target=step + 6,
                               matching_pre_state_boundary=step + 7,
                               returned_residuals={key: float(snap[key][index]) for key in snap
                                                  if key.startswith('returned_') and key.endswith(('residual_F', 'residual_U'))},
                               native_root_pose_max_absolute_changes=root_changes, rows=rows))
    result = dict(status='PASS_ORIGINAL_EVENT_NATIVE_CLOCK_BINDING', events=events,
                  sources=checked, registration_sha256=sha(directory / 'REGISTRATION.json'),
                  analysis_source_sha256=sha(Path(__file__)),
                  limitations=['Signed target offsets use the frozen current row Jacobian, not nonlinear future clearance.',
                               'Away-pointing position targets do not certify braking or future motion.',
                               'Exact six-slot delivery is not a causal demonstration that seven-step forecasting fixes failures.',
                               'Known-bank diagnostic only; no new policy trial or fresh acceptance.'])
    (directory / 'FIRST_FAILURE_NATIVE_BINDING.json').write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'], len(events), 'events', sum(len(e['rows']) for e in events), 'rows')


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
