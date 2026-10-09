"""Read preserved capture; write exclusively new diagnostic artifacts here."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import sys

import numpy as np

from native_teacherforced_cpu_v1 import (ARMS, WIDTHS, Macro74, UNKNOWN, VARIANTS,
    array_sha, future_target_pairing, require, same_bytes, teacher_forced_two_micro)

D = Path(__file__).resolve().parent
H = D.parent.parent
A = H / 'astra/next_mechanism'
ROOT = Path('/mnt/nas/data/lyf/double_hand/safety_verified_fallback_20261010_0352/shard_integration_v1/batch0_shard0_verified')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def native(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o).__name__)


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, default=native, allow_nan=False)
        f.write('\n')


class Inputs:
    def __init__(self):
        self.entries = {}

    def add(self, path, role, expected=None):
        path = Path(path).resolve()
        s = path.stat()
        actual = sha(path)
        require(expected is None or actual == expected, 'source SHA mismatch: ' + str(path))
        if str(path) in self.entries:
            require(self.entries[str(path)]['sha256'] == actual, 'input changed during read')
        self.entries[str(path)] = dict(sha256=actual, bytes=s.st_size, mtime_ns=s.st_mtime_ns,
                                      role=role, expected_sha256=expected)
        return path

    def recheck(self):
        for p, entry in self.entries.items():
            require(sha(p) == entry['sha256'], 'input changed after calibration: ' + p)
        return dict(count=len(self.entries), all_unchanged=True)


def frozen_hashes(inputs):
    union = {}
    counts = {}
    for name in ('QUEUE74_PREFLIGHT_PLAN_V2.json', 'QUEUE74_FULL_PLAN_V2.json'):
        p = inputs.add(H / name, 'read-only frozen source manifest')
        source = json.loads(p.read_text())['sources']
        counts[name] = len(source)
        for path, expected in source.items():
            require(path not in union or union[path] == expected, 'conflicting frozen manifests')
            union[path] = expected
    require(len(union) == 493, 'expected493 frozen sources')
    rows = {}
    for p, expected in union.items():
        actual = sha(p)
        rows[p] = dict(expected_sha256=expected, before_sha256=actual, matched=actual == expected)
    require(all(v['matched'] for v in rows.values()), 'preexisting frozen source mismatch')
    return dict(manifest_counts=counts, union_count=len(rows), sources=rows)


def finish_frozen(audit):
    for path, row in audit['sources'].items():
        row['after_sha256'] = sha(path)
        row['unchanged'] = row['after_sha256'] == row['before_sha256'] == row['expected_sha256']
    audit['all493_unchanged'] = all(v['unchanged'] for v in audit['sources'].values())
    require(audit['all493_unchanged'], 'frozen sources changed')


def block_diag(parts):
    m = np.zeros((74, 74))
    offset = 0
    for part in parts:
        n = len(part)
        m[offset:offset+n, offset:offset+n] = part
        offset += n
    return m


def load_witness(lane, inputs, analysis, frontier):
    spec = next(w for w in analysis['witnesses'] if w['lane'] == lane)
    path = inputs.add(spec['model_inputs_snapshot'], 'immutable original lane witness', spec['model_inputs_snapshot_sha256'])
    with np.load(path, allow_pickle=False) as z:
        # Do not load A/J/mesh/camera arrays. These are irrelevant to this calibration.
        keys = [k for k in z.files if k.startswith(('parameters__', 'micro_native__', 'ds__'))
                and not any(s in k for s in ('jacobians', 'body_com', 'sph_'))]
        keys += ['rs__pre_sim_time_s', 'rs__pre_sim_step_index', 'rs__pre_pending',
                 'rs__applied_target', 'rs__issued_target', 'actual_scalar_peak_N']
        data = {k: z[k] for k in keys}
    # Verify extracted lane snapshots against retained ORIGINAL stream payloads.
    ledgers = {Path(x['export']).name: x for x in analysis['input_stream_ledgers']}
    for filename, prefix in [('dynamics_stream.npz', 'ds__'), ('response_stream.npz', 'rs__')]:
        p = inputs.add(ROOT / filename, 'original native stream cross-check', ledgers[filename]['export_sha256'])
        with np.load(p, allow_pickle=False) as z:
            for key, value in data.items():
                if not key.startswith(prefix):
                    continue
                name = key[len(prefix):]
                raw = z[name]
                if name == 'pre_pending':
                    selected = raw[:, :, lane]
                else:
                    selected = raw[:, lane] if raw.ndim >= 2 and raw.shape[1] == 32 else raw
                require(same_bytes(value, selected), 'original source payload mismatch: ' + key)
    pp = inputs.add(ROOT / 'resolved_native_parameters.npz', 'original full native parameters',
                    analysis['file_sha256'][str(ROOT / 'resolved_native_parameters.npz')])
    with np.load(pp, allow_pickle=False) as z:
        for key, value in data.items():
            if key.startswith('parameters__'):
                raw = z[key[len('parameters__'):]]
                selected = raw[lane] if raw.ndim >= 1 and raw.shape[0] == 32 else raw
                require(same_bytes(value, selected), 'original parameter mismatch: ' + key)
    receipt_path = inputs.add(ROOT / 'point_contact_receipts.json', 'native micro chunk SHA authority',
                             analysis['file_sha256'][str(ROOT / 'point_contact_receipts.json')])
    receipt = json.loads(receipt_path.read_text())
    require(len(receipt['chunks']) == 1, 'integration expects one closed16micro chunk')
    chunk = receipt['chunks'][0]
    p = inputs.add(ROOT / chunk['path'], 'original native q/qd/target micro payload', chunk['sha256'])
    identity_path = inputs.add(ROOT / 'native_contact_identity.json', 'owner identity for scalar witness only', receipt['identity_sha256'])
    identity = json.loads(identity_path.read_text())
    with np.load(p, allow_pickle=False) as z:
        for key, value in data.items():
            if key.startswith('micro_native__'):
                raw = z[key[len('micro_native__'):]]
                selected = raw[:, lane] if raw.ndim >= 2 and raw.shape[1] == 32 else raw
                require(same_bytes(value, selected), 'original micro mismatch: ' + key)
        force = []
        for view in identity['views']:
            ids = np.flatnonzero(np.asarray(view['env_ids']) == lane)
            force.append(z[view['arm'] + '_partner_scalar_abs_N'][:, ids].max(-1))
        owner_force = np.concatenate(force, axis=1)
        require(owner_force.shape == (16, 82), 'all82 native scalar owners expected')
        require(np.array_equal(data['actual_scalar_peak_N'], owner_force.max(-1)), 'witness scalar mismatch')
    case = next(c for c in frontier['cases'] if c['lane'] == lane)
    known_path = inputs.add(case['arrays_path'], 'existing36micro plan for target pairing only', case['arrays_sha256'])
    with np.load(known_path, allow_pickle=False) as z:
        idx = []; offset = 0
        for arm, width in zip(ARMS, WIDTHS):
            idx.extend(offset + data['parameters__' + arm + '_controlled_joint_indices'])
            offset += width
        idx = np.asarray(idx, dtype=int)
        future = np.repeat(z['immutable_fifo74'][-1][None], 12, axis=0)
        future[:, idx] = z['predicted_quantized_targets']
        planned = np.repeat(np.concatenate([z['immutable_fifo74'], future]), 2, axis=0)
        actual = np.concatenate([data['micro_native__' + a + '_native_position_targets'] for a in ARMS], axis=-1)
        require(same_bytes(actual, z['actual_targets']), 'known error target source drift')
        pairing = future_target_pairing(planned, actual)
    return data, pairing, owner_force


def make_samples(data):
    def cat(prefix, key, step=None):
        items = [data[prefix + a + '_' + key] for a in ARMS]
        if step is not None:
            items = [x[step] for x in items]
        return np.concatenate(items, axis=-1)
    target = cat('micro_native__', 'native_position_targets')
    actual_q = cat('micro_native__', 'native_q')
    actual_qd = cat('micro_native__', 'native_qd')
    n = len(data['ds__step'])
    require(n == 8 and actual_q.shape == (16, 74), 'closed integration8/16 schema')
    require(np.array_equal(data['ds__step'], np.arange(n)), 'macro clock')
    require(np.array_equal(data['micro_native__frame'], np.repeat(np.arange(n), 2)), 'micro macro clock')
    require(np.array_equal(data['micro_native__substep'], np.tile(np.arange(2), n)), 'micro substep clock')
    ids = [a + '/' + str(name) for a in ARMS for name in data['parameters__' + a + '_native_joint_names']]
    require(len(ids) == len(set(ids)) == 74, 'canonical74 joint identity')
    controlled = []; offset = 0
    for a, w in zip(ARMS, WIDTHS):
        controlled.extend((data['parameters__' + a + '_controlled_joint_indices'] + offset).tolist()); offset += w
    require(len(set(controlled)) == 26, 'controlled26 inventory')
    flags = np.concatenate([np.full(w, data['parameters__' + a + '_disable_gravity_cfg'], bool) for a, w in zip(ARMS, WIDTHS)])
    samples = []
    for c in range(n):
        q, v = cat('ds__', 'q', c), cat('ds__', 'qd', c)
        if c:
            require(same_bytes(q, actual_q[2*c-1]) and same_bytes(v, actual_qd[2*c-1]), 'teacher reset does not match prior native macro end')
        require(same_bytes(cat('ds__', 'actual_position_target', c), target[2*c]), 'dynamics target differs from applied native target')
        require(same_bytes(data['rs__applied_target'][c], target[2*c, controlled]), 'controlled26 differs from full74 actual target')
        s = Macro74(q=q, qd=v, actual_targets=target[2*c:2*c+2],
            native_q=actual_q[2*c:2*c+2], native_qd=actual_qd[2*c:2*c+2],
            mass=block_diag([data['ds__' + a + '_mass_matrix'][c] for a in ARMS]),
            armature=cat('parameters__', 'armature'), kp=cat('parameters__', 'stiffness'), kd=cat('parameters__', 'damping'),
            gravity=cat('ds__', 'gravity', c), coriolis=cat('ds__', 'coriolis', c), gravity_disabled=flags,
            velocity_target=cat('ds__', 'actual_velocity_target', c), explicit_actuation=cat('ds__', 'actuation_force', c),
            effort_limits=cat('parameters__', 'max_force'), dt_model=.008333,
            pre_native_step=int(data['rs__pre_sim_step_index'][c]), native_steps=data['micro_native__simulation_time_step_index'][2*c:2*c+2],
            pre_native_time=float(data['rs__pre_sim_time_s'][c]), native_times=data['micro_native__simulation_time_s'][2*c:2*c+2],
            native_tick=float(np.float32(1/120)))
        s.validate(); samples.append(s)
    return samples, ids, controlled


def stats(error):
    return dict(rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(np.abs(error))),
                max_abs=float(np.max(np.abs(error))), signed_mean=float(np.mean(error)),
                empirical_p95_abs=float(np.quantile(np.abs(error), .95)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    require(args.tag.replace('_', '').isalnum(), 'safe unique output tag')
    out = D / args.tag
    out.mkdir(exist_ok=False)
    inputs = Inputs()
    frozen = frozen_hashes(inputs)
    print('frozen493 before: unchanged', flush=True)
    for name in ('MODEL_FRONTIER_KNOWN_ERRORS_V1.json', 'original_witness_adapter_v2.py',
                 'native_snapshot_adapter_v3.py', 'queue74_predictor_admission_v3.py',
                 'QUEUE74_TASK_AND_NATIVE_ERROR_FRONTIER_V2.json'):
        inputs.add(A / name, 'required source interpretation')
    if (H / 'SHARD_INTEGRATION_PLAN_V1.json').exists():
        inputs.add(H / 'SHARD_INTEGRATION_PLAN_V1.json', 'original capture registration')
    for name in ('native_verified_v4.py', 'native_state_capture.py'):
        inputs.add(H / name, 'capture getter/clock semantics, read as text only')
    for path in sorted(D.glob('*.py')):
        inputs.add(path, 'calibration helper/runner/test/supervisor source')
    analysis_path = inputs.add(H / 'astra/adaptation32/strategy_short_analysis_v4.json', 'original stream/witness SHA ledger')
    analysis = json.loads(analysis_path.read_text())
    require(analysis['closure']['actual_exit'] == 0 and analysis['closure']['actual_wait_receipt_present'], 'native capture not closed')
    inputs.add(analysis['closure']['execution'], 'original native actual-wait receipt', analysis['closure']['execution_sha256'])
    frontier = json.loads((A / 'MODEL_FRONTIER_KNOWN_ERRORS_V1.json').read_text())
    results = []; aggregates = []; detailed = []; witnesses = []; arrays = {}; metadata = []
    for lane in (18, 19):
        data, pairing, force = load_witness(lane, inputs, analysis, frontier)
        samples, ids, controlled = make_samples(data)
        first = samples[0]
        initial_bias = first.coriolis.astype(float) + first.gravity.astype(float) * ~first.gravity_disabled
        hard = np.concatenate([data['parameters__' + a + '_hard_limits'] for a in ARMS])
        raw_q = np.concatenate([s.native_q for s in samples]); raw_v = np.concatenate([s.native_qd for s in samples])
        arrays[f'lane{lane}_actual_q'] = raw_q; arrays[f'lane{lane}_actual_qd'] = raw_v
        arrays[f'lane{lane}_actual_target'] = np.concatenate([s.actual_targets for s in samples])
        arrays[f'lane{lane}_pre_q'] = np.stack([s.q for s in samples]); arrays[f'lane{lane}_pre_qd'] = np.stack([s.qd for s in samples])
        witness_micro = 12 if lane == 18 else 13
        if lane == 18:
            require(ids[62] == 'U_R/right_index_1_joint' and raw_q[12, 62] < hard[62, 0] - 1e-5, 'known lane18 hard witness absent')
        else:
            require(force[13].max() > .1 and np.all(force[:13] <= .1), 'known lane19 micro13 scalar witness absent')
        w = dict(lane=lane, first_failure_micro=witness_micro, control=6, sub=witness_micro % 2,
                 observed_failure='HAND_HARD_POSITION' if lane == 18 else 'OMITTED_SELF_CONTACT_FROM_EXISTING_FRONTIER',
                 observed_scalar_peak_N=float(force[witness_micro].max()),
                 original_witness_reproduced=True, future36_pairing=pairing, variants=[], **UNKNOWN)
        for variant in VARIANTS:
            runs = [teacher_forced_two_micro(s, variant, initial_mass=first.mass, initial_bias=initial_bias) for s in samples]
            qe = np.concatenate([r['q_error'] for r in runs]); ve = np.concatenate([r['qd_error'] for r in runs])
            pq = np.concatenate([r['predicted_q'] for r in runs]); pv = np.concatenate([r['predicted_qd'] for r in runs])
            name = variant.name
            for key, value in [('q_error', qe), ('qd_error', ve), ('predicted_q', pq), ('predicted_qd', pv),
                               ('model_torque', np.concatenate([r['model_torque'] for r in runs]))]:
                arrays[f'lane{lane}_{name}_{key}'] = value
            diagnostics = [d for r in runs for d in r['numerical']]
            summary = dict(lane=lane, variant=asdict(variant), macro_count=8, micro_count=16,
                           q=stats(qe), qd=stats(ve), all_solves_converged=all(d['converged'] for d in diagnostics),
                           max_equation_residual=max(d['residual_max'] for d in diagnostics),
                           saturated_model_coordinate_micro_count=int(sum(d['saturation_mask'].sum() for d in diagnostics)))
            results.append(summary)
            for sub in ('both', 0, 1):
                e, ev = (qe, ve) if sub == 'both' else (qe[sub::2], ve[sub::2])
                for j, joint in enumerate(ids):
                    row = dict(lane=lane, variant=name, substep=sub, column=j, joint=joint,
                               group='controlled26' if j in controlled else 'hand48', sample_count=len(e))
                    row.update({'q_' + k: v for k, v in stats(e[:, j]).items()})
                    row.update({'qd_' + k: v for k, v in stats(ev[:, j]).items()})
                    aggregates.append(row)
            for micro in range(16):
                for j, joint in enumerate(ids):
                    detailed.append(dict(lane=lane, variant=name, control=micro//2, substep=micro%2, micro=micro,
                        native_step=int(samples[micro//2].native_steps[micro%2]), column=j, joint=joint,
                        actual_target=float(samples[micro//2].actual_targets[micro%2, j]),
                        native_q=float(raw_q[micro,j]), predicted_q=float(pq[micro,j]), q_error=float(qe[micro,j]),
                        native_qd=float(raw_v[micro,j]), predicted_qd=float(pv[micro,j]), qd_error=float(ve[micro,j])))
            entry = dict(name=name, at_failure_q=stats(qe[witness_micro]), at_failure_qd=stats(ve[witness_micro]))
            if lane == 18:
                j = 62
                entry['known_hard_joint'] = dict(column=j, joint=ids[j], actual_q=float(raw_q[witness_micro,j]),
                    predicted_q=float(pq[witness_micro,j]), q_error=float(qe[witness_micro,j]),
                    actual_qd=float(raw_v[witness_micro,j]), predicted_qd=float(pv[witness_micro,j]),
                    bounds=hard[j], actual_violation=float(hard[j,0]-raw_q[witness_micro,j]),
                    model_hard_violation=float(max(0.,hard[j,0]-pq[witness_micro,j],pq[witness_micro,j]-hard[j,1])))
            w['variants'].append(entry)
            print(f'lane{lane} {name}: qmax={summary["q"]["max_abs"]:.8g} qdmax={summary["qd"]["max_abs"]:.8g}', flush=True)
        witnesses.append(w)
        metadata.append(dict(lane=lane, native_ids=ids, controlled_columns=controlled,
            gravity_disabled=first.gravity_disabled, armature=first.armature, stiffness=first.kp, damping=first.kd,
            friction=np.concatenate([data['parameters__'+a+'_friction'] for a in ARMS]),
            drive_properties=np.concatenate([data['parameters__'+a+'_drive_model'] for a in ARMS]),
            actual_targets_sha256=array_sha(arrays[f'lane{lane}_actual_target']),
            actual_q_sha256=array_sha(raw_q), actual_qd_sha256=array_sha(raw_v),
            max_abs_explicit_actuation=float(max(np.abs(s.explicit_actuation).max() for s in samples)),
            max_abs_recorded_projected_force=float(max(np.abs(data['ds__'+a+'_projected_joint_force']).max() for a in ARMS))))
    for filename, rows in [('per74_error_summary.csv', aggregates), ('per74_micro_errors.csv', detailed)]:
        with (out / filename).open('x', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (out / 'two_witness_arrays.npz').open('xb') as f:
        np.savez_compressed(f, **arrays)
    report = dict(schema='safeduo.astra.native_teacherforced_calibration.v1',
        scope='DEVELOPMENT_TWO_KNOWN_FAILURE_WITNESSES_8CONTROLS_EACH',
        real_unique_macros=16, real_unique_microsteps=32, native_coordinate_count=74,
        teacher_forcing='RESET each control to actual native pre-q/qd; same ACTUAL APPLIED full74 target for two model steps',
        errors_sign='model minus native; q rad; qd rad/s', summaries=results, variants=[asdict(v) for v in VARIANTS],
        parameters=metadata, known_witnesses=witnesses,
        assumptions=dict(M='Recorded generalized M frozen within2micro; block diagonal four arms; armature as-recorded vs add-diagonal both unvalidated.',
            C='Recorded Coriolis/centrifugal compensation treated as subtracted bias; frozen within macro; zero-C sensitivity is not a native intervention.',
            gravity='Recorded gravity compensation subtracted only where disable_gravity_cfg is false; zero-gravity sensitivity never modifies native physics.',
            drive='Implicit force-form saturated PD is an approximation; explicit saturated PD is a diagnostic comparison; no native implicit solver identification.',
            frozen_time='Current-macro M/C/g refreshed from native every control; initial-M and initial-bias variants isolate staleness sensitivity while q/qd still teacher forced.',
            clock='Baseline model dt .008333; separate native tick float32(1/120). No replacement of actual native event clocks.',
            omitted='Contact impulses, joint-limit constraints, velocity clamps, hidden solver state and native friction/drive semantics not modeled.',
            torque='Explicit actuation command and projected incoming joint forces do not identify actual implicit drive torque.'),
        global_bound=None, independent_holdout_count=0, parameters_fitted_to_witnesses=False,
        empirical_maximum_is_not_a_global_bound=True, variant_ranking_is_not_identification=True,
        full480_960_evaluated=False, full480_960_reason='Deliver minimal real2-witness calibration first; longer corpus not required to establish numerical mismatch or a global bound.',
        gpu_started=False, isaac_or_applauncher_imported=False,
        extra_astra_review='UNAVAILABLE: collaboration tools reported missing binding; no independent reviewer result claimed', **UNKNOWN)
    write_json(out / 'calibration_report.json', report)
    write_json(out / 'known_witness_report.json', dict(cases=witnesses, **UNKNOWN))
    finish_frozen(frozen)
    write_json(out / 'frozen493_before_after.json', frozen)
    verification = inputs.recheck()
    write_json(out / 'input_source_sha256.json', dict(schema='all_material_inputs_sources.v1', entries=inputs.entries,
        verification=verification, numpy_version=np.__version__, python_version=sys.version,
        interpreter=dict(path=sys.executable, sha256=sha(sys.executable)), max_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    outputs = {str(p): sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    write_json(out / 'output_sha256.json', outputs)
    require(not any(k == 'torch' or k.startswith(('isaac', 'omni.')) for k in sys.modules), 'forbidden simulator/GPU module import')
    print(json.dumps(dict(status='COMPLETE', unique_macros=16, unique_micros=32, variants=len(VARIANTS),
        all493_unchanged=True, output=str(out), max_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)), flush=True)


if __name__ == '__main__':
    main()
