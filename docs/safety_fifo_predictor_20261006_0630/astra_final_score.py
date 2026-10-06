"""Independent CPU raw-score pipeline; reads outcome arrays only after closure.

All writes are confined to this directory's ASTRA_FINAL_* or astra_* names.
No parent analysis, simulator, torch, model fit, or GPU imports. --watch polls
campaign completion and then performs exactly the same --score operation.
"""
import argparse
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np

from astra_score_core import (CLASSES, MODES, aggregate, evaluation_classes,
    fingerprint, native_flags, paired_comparison, reduce_geometry, require,
    score_windows, unpack_exempt)

H = Path(__file__).resolve().parent
RAW = Path('/mnt/nas/data/lyf/double_hand') / H.name
STEPS, ENVS, JOINTS, ROWS = 960, 64, 26, 9021
TERMINAL = ('complete', 'complete_with_failures', 'failed')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_owned(name, value):
    require(Path(name).name == name and name.startswith(('ASTRA_FINAL_', 'astra_')), 'write outside owned namespace')
    target = H/name
    temp = H/f'astra_tmp_{os.getpid()}_{name}'
    text = value if isinstance(value, str) else json.dumps(value, indent=2, allow_nan=False)+'\n'
    temp.write_text(text)
    temp.replace(target)


class Ledger:
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = Path(path)
        digest = sha(path)
        self._remember(path, digest, expected)
        return digest

    def _remember(self, path, digest, expected):
        key = str(path)
        if key in self.hashes:
            require(self.hashes[key] == digest, f'input changed on reread: {path}')
        self.hashes[key] = digest
        if expected is not None:
            require(digest == expected, f'input hash mismatch: {path}')

    def bytes(self, path, expected=None):
        path = Path(path)
        payload = path.read_bytes()
        self._remember(path, hashlib.sha256(payload).hexdigest(), expected)
        return payload

    def json(self, path, expected=None):
        return json.loads(self.bytes(path, expected))

    def npz(self, path, keys=None, expected=None):
        payload = self.bytes(path, expected)
        with np.load(io.BytesIO(payload), allow_pickle=False) as z:
            return {key: z[key] for key in (z.files if keys is None else keys)}

    def recheck(self):
        changed = []
        for name, expected in self.hashes.items():
            try:
                if sha(name) != expected:
                    changed.append(name)
            except OSError:
                changed.append(name)
        return changed


def registration(ledger):
    reg = ledger.json(H/'NUMERIC_REGISTRATION.json')
    require(reg['status'] == 'FROZEN_BEFORE_ALL_NEW_POLICY_OUTCOMES', 'unfrozen registration')
    require(reg['method_windows'] == 768 and reg['paired_cases'] == 192, 'registered denominator mismatch')
    design = ledger.json(H/'RANDOM_EXPERIMENT_DESIGN.json')
    require(tuple(design['modes']) == MODES and design['frames_per_window'] == STEPS, 'registered mode/frame mismatch')
    plans = []
    for block in range(3):
        path = H/'plans'/f'holdout_{block}_plan.json'
        plan = ledger.json(path, reg['plans'][str(path)])
        require(plan['sources_frozen'] and not plan['production_promoted'], 'plan not isolated/frozen')
        require(Path(plan['output_root']) == RAW/f'holdout_{block}', 'wrong raw namespace')
        require(len(plan['jobs']) == 4 and {j['env']['SAFEDUO_JOINT_MODE'] for j in plan['jobs']} == set(MODES), 'mode set mismatch')
        require(len({j['id'] for j in plan['jobs']}) == 4, 'duplicate job')
        row = design['rows'][block]
        require(row['block'] == block, 'design block order changed')
        for job in plan['jobs']:
            require(job['expected']['seed'] == row['command_seed'], 'design/plan command seed mismatch')
            require(Path(job['env']['SAFEDUO_INITIAL_BANK_NPZ']) ==
                    RAW/'banks'/str(row['initial_seed'])/'bank.npz', 'design/plan bank mismatch')
        plans.append(plan)
    require(len(reg['plans']) == 3, 'extra/missing registered block')
    return design, plans


def readiness(plans):
    blocks = []
    for block, plan in enumerate(plans):
        path = Path(plan['output_root'])/'campaign.json'
        if not path.exists():
            blocks.append(dict(block=block, status='not_started', jobs=[]))
            continue
        c = json.loads(path.read_text())
        blocks.append(dict(block=block, status=c['status'],
            jobs=[{k: j.get(k) for k in ('id', 'status', 'exit_code')} for j in c.get('jobs', [])]))
    closed = all(b['status'] in TERMINAL for b in blocks)
    result = dict(status='READY_ALL_CAMPAIGNS_TERMINAL' if closed else 'WAITING_FOR_ALL_CAMPAIGNS',
        utc=now(), all_terminal=closed, blocks=blocks, outcome_arrays_opened=False)
    write_owned('ASTRA_FINAL_READINESS.json', result)
    return result


def bind_scoring_software(ledger):
    specification = ledger.json(H/'ASTRA_FINAL_SCORING_PLAN.json')
    require(specification['status'] == 'FROZEN_INDEPENDENT_SCORER_BEFORE_OUTCOME_READS',
            'independent scoring specification not frozen')
    require(specification['expected_cases'] == 192 and specification['expected_windows'] == 768,
            'scorer denominator changed')
    expected_names = {'astra_final_score.py', 'astra_score_core.py', 'astra_score_tests.py'}
    require(set(specification['scorer_source_sha256']) == expected_names, 'scorer source inventory differs')
    for name, expected in specification['scorer_source_sha256'].items():
        ledger.bind(H/name, expected)


def float_array(a, shape, name, allow_positive_inf=False):
    require(a.shape == shape and a.dtype == np.float32, f'{name}: shape/dtype mismatch')
    valid = ~np.isnan(a) & ~np.isneginf(a) if allow_positive_inf else np.isfinite(a)
    require(valid.all(), f'{name}: nonfinite data')


def exact(a, b, label):
    require(np.array_equal(a, b), label)


def relative_file(root, name):
    path = root/name
    require(path.resolve().is_relative_to(root.resolve()), 'receipt escaped cell directory')
    return path


def ids_mask(ids, shape):
    require(ids.dtype == np.int32 and ids.shape == shape, 'selected IDs shape/dtype')
    require(((ids >= -1) & (ids < ROWS)).all(), 'selected ID range')
    flat = ids.reshape(-1, ROWS)
    valid = flat >= 0
    r, c = np.nonzero(valid)
    dense = np.zeros_like(flat, dtype=bool)
    dense[r, flat[r, c]] = True
    exact(dense.sum(axis=1), valid.sum(axis=1), 'duplicate selected IDs')
    return dense.reshape(shape)


def verify_fifo(data):
    pending = data['pre_pending_actuator_targets']
    initial_target = pending[0, -1]
    initial_six = np.broadcast_to(initial_target, (6, ENVS, JOINTS))
    exact(pending[0], initial_six, 'initial FIFO must contain six initial targets')
    stream = np.concatenate((initial_six, data['controller_target']))
    exact(data['actuator_target'], stream[:STEPS], 'actual applied FIFO sequence')
    for lag in range(6):
        exact(pending[:, lag], stream[lag:lag+STEPS], f'pending FIFO slot {lag}')
    exact(data['pre_pending_project_history'], pending, 'project history differs from actuator FIFO')
    before_target = np.concatenate((initial_target[None], data['controller_target'][:-1]))
    before_q = np.concatenate((data['q_initial'][None], data['q'][:-1]))
    exact(data['pre_target_debt'], before_target-before_q, 'target debt is not target minus q')
    exact(data['effective_target_delta'], data['controller_target']-before_target, 'effective increment binding')
    limits = data['joint_soft_limits']
    integrated = np.clip(before_target+data['exec'], limits[..., 0], limits[..., 1])
    exact(integrated, data['controller_target'], 'native float32 actual target integration')
    return initial_target


def selected_failures(ledger, root, windows, data):
    receipt = ledger.json(root/'first_failure_receipts.json')
    expected = {env: w['strict']['first_step'] for env, w in enumerate(windows) if w['strict']['failed']}
    observed, candidates = {}, []
    for entry in receipt['receipts']:
        step = entry['step']
        require(isinstance(step, int) and 0 <= step < STEPS, 'bad first-failure step')
        for index, env in enumerate(entry['env_ids']):
            require(env not in observed and 0 <= env < ENVS, 'duplicate/invalid first-failure env')
            observed[env] = step
            candidates.append((step, env, index, entry))
    require(observed == expected and receipt['envs_with_failure'] == len(expected), 'first-failure receipt differs from independent native score')
    selected = []
    for step, env, index, entry in sorted(candidates, key=lambda x: (x[0], x[1]))[:2]:
        file = relative_file(root, entry['path'])
        raw = ledger.npz(file, expected=entry['sha256'])
        require(int(raw['step']) == step and int(raw['env_ids'][index]) == env, 'first-failure snapshot identity')
        s = {k: v[index] for k, v in raw.items() if k not in ('step', 'env_ids')}
        for k, v in s.items():
            if v.dtype.kind in 'fc':
                require(np.isfinite(v).all(), f'nonfinite first-failure {k}')
        pre_q = data['q_initial'][env] if step == 0 else data['q'][step-1, env]
        pre_target = data['pre_pending_actuator_targets'][0, -1, env] if step == 0 else data['controller_target'][step-1, env]
        for label, a, b in (
            ('pre q', s['pre_q'], pre_q), ('post q', s['post_q'], data['q'][step, env]),
            ('pre qd', s['snapshot_qd'], data['pre_qd_compact'][step, env]),
            ('pre target', s['pre_issued_target'], pre_target),
            ('pending', s['pre_pending_actuator_targets'], data['pre_pending_actuator_targets'][step, :, env]),
            ('project output', s['actual_project_return'], data['exec'][step, env]),
            ('integrated delta', s['returned_cmd'], data['effective_target_delta'][step, env])):
            exact(a, b, f'first failure {label} binding')
        if step < STEPS-1:
            exact(s['post_qd'], data['pre_qd_compact'][step+1, env], 'first failure next pre qd binding')
        selected.append(dict(step=step, env=env, path=str(file), raw=s))
    return selected


def chunk_audit(ledger, root, data, selected, mode):
    identity = ledger.json(root/'full_row_identity.json')
    classes = evaluation_classes(identity)
    require(len(classes) == ROWS and identity['rows'] == ROWS, 'full geometry row count')
    receipt = ledger.json(root/'forecast_receipts.json')
    require(receipt['steps'] == STEPS and receipt['rows'] == ROWS and receipt['envs'] == ENVS, 'incomplete forecast receipt')
    require(len(receipt['chunks']) == 30, 'expected 30 complete forecast chunks')
    total_pre = np.empty((STEPS, ENVS, 4), dtype=np.float32)
    end, motion_rows, motion_env_steps, row_instances = 0, 0, 0, 0
    keys = ('measured_d', 'exempt', 'dmin', 'forecast', 'target_forecast',
            'cv_forecast', 'pd_forecast', 'selected_ids', 'baseline_ids')
    for entry in receipt['chunks']:
        start, stop = entry['start'], entry['stop']
        require(start == end and stop-start == 32 and stop <= STEPS, 'chunk gap/overlap/length')
        path = relative_file(root, entry['path'])
        z = ledger.npz(path, keys, entry['sha256'])
        shape = (stop-start, ENVS, ROWS)
        for k in keys[:1] + keys[2:7]:
            float_array(z[k], shape, k)
        exempt = unpack_exempt(z['exempt'], ROWS)
        require(exempt.shape == shape, 'exemption shape')
        minimum = reduce_geometry(z['measured_d'], exempt, classes)
        total_pre[start:stop] = minimum
        skip = 1 if start == 0 else 0
        exact(minimum[skip:], data['official_margins'][start+skip-1:stop-1], 'full pre geometry vs previous native post margins')
        expected_forecast = z['target_forecast']
        if mode in ('velocity_admission', 'motion_admission'):
            expected_forecast = np.minimum(expected_forecast, z['cv_forecast'])
        if mode in ('pd_admission', 'motion_admission'):
            expected_forecast = np.minimum(expected_forecast, z['pd_forecast'])
        exact(z['forecast'], expected_forecast, 'mode forecast membership')
        for k in ('forecast', 'target_forecast', 'cv_forecast', 'pd_forecast'):
            require((z[k] <= z['measured_d']).all(), 'forecast minimum omitted current distance')
        chosen = ids_mask(z['selected_ids'], shape)
        measured_base = ids_mask(z['baseline_ids'], shape)
        boundary = z['dmin'] + np.float32(.010)
        require((measured_base | ~(z['measured_d'] <= boundary)).all(), 'measured baseline omitted critical row')
        exact(chosen, measured_base | (z['forecast'] <= boundary), 'actual row union dropped/swapped rows')
        target_mask = measured_base | (z['target_forecast'] <= boundary)
        require((chosen | ~target_mask).all(), 'old target-based row missing')
        additional = chosen & ~target_mask
        if mode == 'joint_reference':
            require(not additional.any(), 'reference has motion-only rows')
        motion_rows += int(additional.sum()); motion_env_steps += int(additional.any(axis=-1).sum())
        row_instances += int(chosen.sum())
        exact(chosen.sum(axis=-1), data['critical_selected_count'][start:stop], 'selected count dense binding')
        for case in selected:
            t, e, s = case['step'], case['env'], case['raw']
            if not start <= t < stop:
                continue
            offset = t-start
            valid = s['snapshot_valid'].astype(bool)
            ids = s['selected_ids'][:len(valid)]
            exact(s['selected_ids'], z['selected_ids'][offset, e], 'first-failure selected ID binding')
            exact(s['snapshot_d'][valid], z['measured_d'][offset, e, ids[valid]], 'first-failure pre geometry binding')
            post_min = reduce_geometry(s['post_full_d'], s['post_full_exempt'], classes)
            exact(post_min, data['official_margins'][t, e], 'first-failure full post geometry binding')
            bad = np.flatnonzero((s['post_full_d'] < np.float32(0)) & ~s['post_full_exempt'])
            require(len(bad) > 0, 'selected first failure has no native negative row')
            J = np.concatenate((s['snapshot_J_F'], s['snapshot_J_U']), axis=-1).astype(np.float64)
            dq = s['post_q'].astype(np.float64)-s['pre_q'].astype(np.float64)
            effects = []
            for pid in bad:
                positions = np.flatnonzero(valid & (ids == pid))
                item = dict(pair_id=int(pid), class_name=CLASSES[classes[pid]],
                    pre_distance_m=float(z['measured_d'][offset, e, pid]),
                    post_distance_m=float(s['post_full_d'][pid]), selected=bool(len(positions)))
                if len(positions):
                    j = J[int(positions[0])]
                    item.update(J_actual_q_motion_m=float(j@dq),
                        J_integrated_target_increment_m=float(j@s['returned_cmd'].astype(np.float64)),
                        J_applied_target_error_m=float(j@(s['pre_pending_actuator_targets'][0].astype(np.float64)-s['pre_q'].astype(np.float64))))
                effects.append(item)
            case['binding'] = dict(status='PASS_SELECTED_FIRST_FAILURE_BINDINGS', step=t, env=e,
                path=case['path'], rows=effects, post_qd_dense_check=t < STEPS-1,
                no_unique_physical_cause_claim=True)
        end = stop
        del z, chosen, measured_base, target_mask, additional
    require(end == STEPS and all('binding' in c for c in selected), 'incomplete chunk/snapshot coverage')
    return dict(full_pre_frames=STEPS, full_rows=ROWS, post_frames_crosschecked=STEPS-1,
        final_post_full_row_archive_available=False, final_post_source='native cell official_margins[959]',
        full_geometry_pre_strict_env_steps=int((total_pre < np.float32(0)).any(axis=-1).sum()),
        full_geometry_pre_deep_env_steps=int((total_pre < np.float32(-.005)).any(axis=-1).sum()),
        stored_forecast_mode_minima_and_unions_verified=True,
        motion_only_added_row_instances_at_own_state=motion_rows,
        motion_only_added_env_steps_at_own_state=motion_env_steps, selected_row_instances=row_instances,
        first_failure_selection='first two distinct failing envs by (step,env) per condition, if present',
        selected_first_failures=[c['binding'] for c in selected])


def score_cell(ledger, plan, job, campaign_record):
    root = Path(plan['output_root'])/job['id']
    require(campaign_record['status'] == 'complete' and campaign_record['exit_code'] == 0, 'condition did not complete successfully')
    proto = ledger.json(root/'protocol.json', campaign_record['protocol_sha256'])
    require(proto['status'] == 'complete' and proto['completed_cells'] == 1 and len(proto['design']) == 1, 'protocol incomplete')
    require(proto['steps'] == STEPS and proto['dt'] == .016666, 'timebase mismatch')
    require(proto['source_sha256'] == plan['source_sha256'] and proto['checkpoint_sha256'] == plan['checkpoint_sha256'], 'source/actor protocol mismatch')
    for k, v in job['expected'].items():
        require(proto['design'][0][k] == v, f'condition differs: {k}')
    for k, v in job['expected_args'].items():
        require(proto['args'][k] == v, f'protocol argument differs: {k}')
    require(proto['args']['actuator_delay_steps'] == [6], 'actuator delay changed')
    mode = job['env']['SAFEDUO_JOINT_MODE']
    metadata = ledger.json(root/'guard_metadata.json')
    require(metadata['mode'] == mode and metadata['capacity'] == ROWS and metadata['strict_fifo_steps'] == 6, 'guard registration differs')
    require(metadata['reference_gap_rad'] == .050 and metadata['original_actor_rows'] == 32 and not metadata['joint_repair'] and not metadata['queue_preemption'], 'guard control contract differs')
    for path, digest in metadata['sources'].items():
        require(plan['research_source_sha256'].get(path) == digest, f'guard source not registered: {path}')
    keys = ('official_margins', 'official_deep', 'q', 'q_initial', 'cmd', 'external_unscaled_cmd',
            'exec', 'controller_target', 'actuator_target', 'pre_qd_compact', 'pre_target_debt',
            'pre_pending_actuator_targets', 'pre_pending_project_history', 'effective_target_delta',
            'joint_soft_limits', 'initial_violation', 'critical_selected_count', 'meta_json')
    data = ledger.npz(root/'cell_001.npz', keys)
    cell_meta = json.loads(str(data['meta_json']))
    for k, v in proto['design'][0].items():
        require(cell_meta[k] == v, f'raw cell condition differs: {k}')
    require(cell_meta['dt'] == proto['dt'] and cell_meta['cell_id'] == 1
            and cell_meta['checkpoint_sha256'] == proto['checkpoint_sha256']
            and cell_meta['env_yaml'] == proto['args']['env_yaml'], 'raw cell metadata differs')
    for key in ('q', 'cmd', 'external_unscaled_cmd', 'exec', 'controller_target', 'actuator_target',
                'pre_qd_compact', 'pre_target_debt', 'effective_target_delta'):
        float_array(data[key], (STEPS, ENVS, JOINTS), key)
    float_array(data['q_initial'], (ENVS, JOINTS), 'q_initial')
    float_array(data['joint_soft_limits'], (ENVS, JOINTS, 2), 'limits')
    float_array(data['official_margins'], (STEPS, ENVS, 4), 'official_margins', True)
    for key in ('pre_pending_actuator_targets', 'pre_pending_project_history'):
        float_array(data[key], (STEPS, 6, ENVS, JOINTS), key)
    strict, deep = native_flags(data['official_margins'])
    require(data['official_deep'].dtype == np.bool_ and data['official_deep'].shape == deep.shape,
            'producer deep flag shape/dtype')
    require(data['initial_violation'].shape == (ENVS,) and data['initial_violation'].dtype == np.bool_,
            'initial violation shape/dtype')
    require(data['critical_selected_count'].shape == (STEPS, ENVS)
            and data['critical_selected_count'].dtype.kind in 'iu', 'selected count shape/dtype')
    require((data['joint_soft_limits'][..., 0] <= data['joint_soft_limits'][..., 1]).all(), 'inverted soft limits')
    exact(data['official_deep'], deep, 'producer deep flags vs independently recomputed native threshold')
    initial_target = verify_fifo(data)
    bank_path = Path(job['env']['SAFEDUO_INITIAL_BANK_NPZ'])
    bank_hash = ledger.bind(bank_path, plan['research_source_sha256'][str(bank_path)])
    bank = ledger.npz(bank_path, ['accepted_q', 'risk_pair_index', 'joint_soft_limits'], bank_hash)
    assignment = ledger.npz(root/'bank_assignment.npz', ['risk_pair_index'])
    require(bank['risk_pair_index'].shape == (ENVS,) and bank['risk_pair_index'].dtype.kind in 'iu', 'bank label shape/dtype')
    exact(assignment['risk_pair_index'], bank['risk_pair_index'], 'actual bank label assignment')
    recipe = ledger.npz(root/'input_recipe.npz', ['q_initial', 'sampled_initial', 'tape', 'joint_soft_limits', 'initial_violation'])
    float_array(recipe['tape'], (STEPS+2, ENVS, JOINTS), 'registered run tape plus two unexecuted frames')
    float_array(recipe['q_initial'], (ENVS, JOINTS), 'recipe actual initial state')
    float_array(recipe['sampled_initial'], (ENVS, JOINTS), 'recipe sampled initial state')
    float_array(bank['accepted_q'], (ENVS, JOINTS), 'bank sampled initial state')
    exact(recipe['tape'][:60], np.zeros((60, ENVS, JOINTS), dtype=np.float32), 'registered zero prefix')
    exact(recipe['q_initial'], data['q_initial'], 'input recipe actual q0 binding')
    exact(recipe['sampled_initial'], bank['accepted_q'], 'sampled q0 vs registered bank')
    exact(recipe['joint_soft_limits'], data['joint_soft_limits'], 'recipe soft limits binding')
    exact(bank['joint_soft_limits'], data['joint_soft_limits'], 'bank soft limits binding')
    exact(recipe['initial_violation'], data['initial_violation'], 'initial violation binding')
    exact(data['cmd'], recipe['tape'][:STEPS], 'executed external raw command differs from tape')
    exact(data['external_unscaled_cmd'], recipe['tape'][:STEPS], 'unscaled command/tape binding')
    metrics = score_windows(data['official_margins'])
    selected = selected_failures(ledger, root, metrics, data)
    geometry = chunk_audit(ledger, root, data, selected, mode)
    qseq = np.concatenate((data['q_initial'][None], data['q']), axis=0).astype(np.float64)
    path = np.abs(np.diff(qseq, axis=0)).sum(axis=(0, 2))
    ranges = np.ptp(qseq, axis=0).sum(axis=1)
    windows = []
    for env in range(ENVS):
        windows.append(dict(status='VERIFIED', env=env, metrics=metrics[env],
            q0_sha256=fingerprint(data['q_initial'][env]), qd0_sha256=fingerprint(data['pre_qd_compact'][0, env]),
            initial_target_sha256=fingerprint(initial_target[env]), tape_sha256=fingerprint(recipe['tape'][:STEPS, env]),
            full_tape_sha256=fingerprint(recipe['tape'][:, env]),
            bank_sha256=bank_hash, risk_pair_index=int(bank['risk_pair_index'][env]),
            initial_violation_recorded=bool(data['initial_violation'][env]),
            actual_joint_L1_path_rad=float(path[env]), actual_joint_range_sum_rad=float(ranges[env])))
    return dict(id=job['id'], mode=mode, root=str(root), status='VERIFIED',
                windows=windows, counts=aggregate(windows), geometry=geometry,
                raw_cell_sha256=ledger.hashes[str(root/'cell_001.npz')])


def run_score(design, plans, ledger):
    ready = readiness(plans)
    if not ready['all_terminal']:
        return ready
    # Read/hashes only; never import frozen parent analysis scripts or use counts.
    source_bindings = {}
    for plan in plans:
        source_bindings.update({str(Path(plan['cwd'])/p): s for p, s in plan['source_sha256'].items()})
        source_bindings.update(plan['research_source_sha256'])
        source_bindings[plan['checkpoint_path']] = plan['checkpoint_sha256']
    for path, digest in source_bindings.items():
        ledger.bind(path, digest)
    scheduling = {}
    for name in ('SCHEDULING_SUPERSESSION.json', 'CAMERA_FOLLOW_SUPERSESSION.json'):
        path = H/name
        if path.exists():
            scheduling[name] = ledger.json(path)
    cells = []
    for block, plan in enumerate(plans):
        campaign = ledger.json(Path(plan['output_root'])/'campaign.json')
        require(campaign['plan'] == plan, 'campaign embedded plan differs from frozen plan')
        require(campaign['status'] in TERMINAL, 'campaign reopened')
        records = {r['id']: r for r in campaign['jobs']}
        require(len(records) == len(campaign['jobs']), 'duplicate campaign job record')
        for job in plan['jobs']:
            mode = job['env']['SAFEDUO_JOINT_MODE']
            try:
                require(job['id'] in records, 'job unavailable after terminal campaign')
                cell = score_cell(ledger, plan, job, records[job['id']])
            except Exception as error:
                cell = dict(id=job['id'], mode=mode, status='UNAVAILABLE_OR_INVALID',
                    error=f'{type(error).__name__}: {error}',
                    windows=[dict(status='UNAVAILABLE_OR_INVALID', env=e, metrics=None) for e in range(ENVS)])
            cell['block'] = block
            cells.append(cell)
            write_owned('ASTRA_FINAL_SCORE_PROGRESS.json', dict(utc=now(), conditions_processed=len(cells),
                total_conditions=12, conditions=[{k:c.get(k) for k in ('block', 'id', 'status', 'error')} for c in cells]))
            print(json.dumps(dict(event='condition_scored', block=block, mode=mode, status=cell['status'], error=cell.get('error'))), flush=True)
    changed = ledger.recheck()
    cases = []
    for block, row in enumerate(design['rows']):
        by_mode = {c['mode']: c for c in cells if c['block'] == block}
        for env in range(ENVS):
            windows = {m: by_mode[m]['windows'][env] for m in MODES}
            cases.append(dict(case_id=f'b{block}:i{row["initial_seed"]}:c{row["command_seed"]}:e{env:02d}',
                block=block, initial_seed=row['initial_seed'], command_seed=row['command_seed'], env=env, windows=windows))
    require(len(cases) == 192 and sum(len(c['windows']) for c in cases) == 768, 'identity denominator lost')
    # A pairing failure invalidates the comparison, not the individual raw count.
    comparisons = {m: paired_comparison(cases, m) for m in MODES[1:]}
    all_valid = all(w['status'] == 'VERIFIED' for c in cases for w in c['windows'].values())
    all_pairs = all(v['verified_pairs'] == 192 for v in comparisons.values())
    report = dict(schema='astra.independent.final_numeric.v1', utc=now(),
        status='PASS_COMPLETE_INDEPENDENT_NUMERIC' if all_valid and all_pairs and not changed else 'INCOMPLETE_OR_INVALID_NUMERIC_EVIDENCE',
        expected_cases=192, expected_windows=768, actual_case_identities=len(cases), actual_window_records=768,
        modes=MODES, class_order=CLASSES, steps_per_window=STEPS, zero_prefix_included=True,
        completion_contract='canonical campaign and per-condition final protocol; outer orchestration exit excluded',
        scheduling_notes=scheduling,
        comparison_dtype='native float32 before any JSON/float64 conversion',
        strict_boundary='margin < np.float32(0)', deep_boundary='margin < np.float32(-.005)',
        counts={m: aggregate([c['windows'][m] for c in cases]) for m in MODES},
        primary_comparison='joint_reference vs motion_admission', comparisons=comparisons,
        blocks={str(b): {m: aggregate([c['windows'][m] for c in cases if c['block']==b]) for m in MODES} for b in range(3)},
        conditions=[{k:v for k,v in c.items() if k!='windows'} for c in cells], cases=cases,
        input_sha256=ledger.hashes, raw_hash_before_after_pass=not changed, changed_inputs=changed,
        hash_scope='all consumed inputs and frozen registered bindings; unselected snapshot payloads and unrelated raw files not consumed',
        scorer_source_sha256={str(p):sha(p) for p in (Path(__file__), H/'astra_score_core.py')},
        verified_outcomes=(['independent native numeric strict/deep/class window and env-step counts',
            'all192 identity/all768 window inventory including unavailable/invalid placeholders',
            'own-case q0/tape/bank paired rescues/new failures for three comparisons',
            'full9021 pre geometry binds previous959 post margin frames; native final post margins retained',
            'actual FIFO order/integration and selected first-failure own-state bindings',
            'registered mode row union and old target-row retention at each own state'] if all_valid and all_pairs and not changed else []),
        not_verified=['Isaac/GPU counterfactual replay','H6 raw prediction error audit','camera replay own-state evidence',
                      'camera exact numeric forward equivalence','parent aggregate results comparison','parent panel/report truth'],
        parent_counts_used=False, parent_analysis_imported=False, policy_outcomes_used_for_tuning=False,
        empirical_fit_certifies_safety=False, hardware_approved=False)
    write_owned('ASTRA_FINAL_SCORE.json', report)
    lines = [f'Independent numeric scoring: {report["status"]}.', '',
        'All 192 case identities and 768 window records are retained. Counts cover all960 post frames, including the zero prefix.', '',
        '| Mode | Verified windows | Strict failures | Deep failures |', '|---|---:|---:|---:|']
    for m,c in report['counts'].items():
        lines.append(f'| {m} | {c["verified_windows"]}/192 | {c["strict"]["failed_windows"]} | {c["deep"]["failed_windows"]} |')
    lines += ['', 'Counts for incomplete/invalid evidence are observed-subset counts, never a full-denominator safe claim.',
        'The first959 post frames are cross-bound to full raw pre geometry; the last post frame uses native saved class minima.',
        'H6 raw errors, cameras, parent results, and the parent panel remain unverified by this stage. No safety certification or hardware approval.']
    write_owned('ASTRA_FINAL_NUMERIC.md', '\n'.join(lines)+'\n')
    return {k:report[k] for k in ('status','expected_cases','expected_windows','counts','raw_hash_before_after_pass')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--readiness', action='store_true')
    group.add_argument('--score', action='store_true')
    group.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    ledger = Ledger()
    design, plans = registration(ledger)
    if args.readiness:
        print(json.dumps(readiness(plans)), flush=True)
        return
    bind_scoring_software(ledger)
    if args.watch:
        while True:
            state = readiness(plans)
            print(json.dumps(state), flush=True)
            if state['all_terminal']:
                break
            time.sleep(30)
    try:
        result = run_score(design, plans, ledger)
    except Exception as error:
        result = dict(status='INVALID_SCORING_PRECONDITION_OR_INPUT', utc=now(),
                      error=f'{type(error).__name__}: {error}', expected_cases=192, expected_windows=768,
                      no_missing_window_counted_safe=True, input_sha256=ledger.hashes)
        write_owned('ASTRA_FINAL_SCORE_ERROR.json', result)
        print(json.dumps(result), flush=True)
        raise
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
