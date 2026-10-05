#!/usr/bin/env python3
"""Independent, CPU-only scoring of this study's 12 CLOSED cells.

No imports of parent analyzers, caches, geometry or simulation packages. Run
--check-ready for metadata only; --score refuses ALL dense reads if a single
registered cell/campaign is not terminal. Final output is H/ASTRA_FINAL_SCORE.json.
Measured endpoints are sphere-layer observations, not a safety approval.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone

import numpy as np

H_DEFAULT = Path('/home/liyufeng/safeduo/artifacts/safety_feasible_guard_20261005_2100')
RAW_DEFAULT = Path('/mnt/nas/data/lyf/double_hand/safety_feasible_guard_20261005_2100')
MODES = ('admission_full', 'admission_scaled_036', 'joint_reference', 'joint_repair')
CLASSES = ('cross', 'self_F', 'self_U', 'table')
ARM_SLICES = ((0, 7), (7, 14), (14, 20), (20, 26))
STEPS, ENVS, DOFS, DELAY = 960, 64, 26, 6
MOVING_START, MOVING_L2_RAD = 60, .001
SCHEMA = 'astra.independent_feasible_guard_score.v3'


class AuditError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class Inputs:
    """Parse the exact bytes hashed once; rehash EVERY consumed file on finish."""
    def __init__(self):
        self.receipts = {}

    def read(self, path):
        path = Path(path).resolve(strict=True)
        data = path.read_bytes()
        self.remember(path, sha(data), len(data))
        return data

    def remember(self, path, digest, size):
        key = str(Path(path).resolve(strict=True))
        if key in self.receipts:
            require(self.receipts[key]['sha256_before'] == digest,
                    f'input changed between consumers: {key}')
        else:
            self.receipts[key] = dict(path=key, sha256_before=digest, bytes=size)

    def bind_file(self, path, expected):
        path = Path(path).resolve(strict=True)
        actual = hash_file(path)
        self.remember(path, actual, path.stat().st_size)
        require(actual == expected, f'frozen source/checkpoint SHA mismatch: {path}')

    def json(self, path):
        return json.loads(self.read(path))

    def npz(self, path, keys=None):
        data = self.read(path)
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            wanted = z.files if keys is None else keys
            require(set(wanted) <= set(z.files), f'missing NPZ keys {set(wanted)-set(z.files)}: {path}')
            return {k: z[k].copy() for k in wanted}

    def finish(self):
        for entry in self.receipts.values():
            entry['sha256_after'] = hash_file(entry['path'])
            entry['unchanged'] = entry['sha256_after'] == entry['sha256_before']
        changed = [x['path'] for x in self.receipts.values() if not x['unchanged']]
        require(not changed, f'consumed files changed during audit: {changed}')
        return list(self.receipts.values())


def under(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    require(path.is_relative_to(root), f'path outside declared input root: {path}')
    return path


def preflight(h, raw, inputs):
    """Only JSON metadata here. Never load any NPZ before all 12 are closed."""
    design = inputs.json(h / 'DESIGN.json')
    require(design['modes'] == list(MODES), 'unregistered methods/order')
    require(design['steps'] == STEPS and design['delay_steps'] == DELAY,
            'registered duration/delay changed')
    require(design['command_scale_control'] == .36, 'scale control changed')
    require(len(design['rows']) == 3, 'expected exactly three registered banks')
    require(len({r['command_seed'] for r in design['rows']}) == 3 and
            len({r['initial_seed'] for r in design['rows']}) == 3, 'repeated registered seed/bank')
    cells, issues, plans = [], [], []
    for block, seeds in enumerate(design['rows']):
        plan_path = h / 'holdout_v2' / f'holdout_{block}_plan.json'
        plan = inputs.json(plan_path)
        plans.append(plan)
        root = under(plan['output_root'], raw)
        require(root == raw.resolve() / f'holdout_{block}', 'wrong campaign output root')
        require(plan['sources_frozen'] is True and not plan['production_promoted'], 'unfrozen/promoted plan')
        jobs = plan['jobs']
        expected_ids = [f'{mode}_{seeds["command_seed"]}' for mode in MODES]
        require([j['id'] for j in jobs] == expected_ids, 'missing/duplicate/unregistered plan job')
        campaign_path = root / 'campaign.json'
        if campaign_path.exists():
            campaign = inputs.json(campaign_path)
        else:
            campaign = dict(status='missing', jobs=[])
        cj = campaign.get('jobs', [])
        require(len({j['id'] for j in cj}) == len(cj), 'duplicate runtime job IDs')
        require({j['id'] for j in cj} <= set(expected_ids), 'unregistered runtime job')
        runtime = {j['id']: j for j in cj}
        campaign_closed = campaign.get('status') in ('complete', 'complete_with_failures', 'failed')
        if not campaign_closed:
            issues.append(dict(block=block, finding='campaign_not_terminal', status=campaign.get('status')))
        for mode, job in zip(MODES, jobs):
            env = job['env']
            require(env['SAFEDUO_JOINT_MODE'] == mode, 'plan mode/env mismatch')
            bank = under(env['SAFEDUO_INITIAL_BANK_NPZ'], raw)
            require(bank == raw.resolve() / 'banks' / str(seeds['initial_seed']) / 'bank.npz',
                    'bank path/registered initial seed mismatch')
            expected = job['expected']
            require(expected == dict(flow='risk_burst', seed=seeds['command_seed'],
                    method='system0', amp=.05, actuator_delay_steps=DELAY), 'registered condition changed')
            require(job['expected_args']['num_envs'] == ENVS and
                    job['expected_args']['duration_s'] == 16., 'registered dimensions changed')
            cell_root = under(root / job['id'], root)
            info = dict(block=block, mode=mode, seed=seeds['command_seed'], initial_seed=seeds['initial_seed'],
                        id=job['id'], root=cell_root, bank=bank, plan=plan, job=job)
            running = runtime.get(job['id'], {})
            protocol_path = cell_root / 'protocol.json'
            protocol = inputs.json(protocol_path) if protocol_path.exists() else {}
            good = (campaign_closed and running.get('status') == 'complete' and
                    running.get('exit_code') == 0 and running.get('completed_cells') == 1 and
                    protocol.get('status') == 'complete' and protocol.get('completed_cells') == 1 and
                    protocol.get('steps') == STEPS and protocol.get('args', {}).get('num_envs') == ENVS and
                    bool(protocol.get('finished_utc')))
            info.update(protocol=protocol, ready=good)
            if not good:
                issues.append(dict(block=block, mode=mode, finding='cell_not_complete',
                                   runtime_status=running.get('status', 'missing'),
                                   protocol_status=protocol.get('status', 'missing'), invalid_windows=ENVS))
            else:
                digest = inputs.receipts[str(protocol_path.resolve())]['sha256_before']
                require(running.get('protocol_sha256') == digest, 'campaign/protocol final SHA mismatch')
                require(len(protocol['design']) == 1, 'more than one condition per cell')
                require(running.get('condition') == protocol['design'][0], 'campaign condition differs from protocol')
                for k, v in expected.items():
                    require(protocol['design'][0].get(k) == v, f'protocol condition mismatch: {k}')
                for k, v in job['expected_args'].items():
                    require(protocol['args'].get(k) == v, f'protocol argument mismatch: {k}')
                require(protocol['source_sha256'] == plan['source_sha256'], 'production source map differs from plan')
                require(protocol['checkpoint_sha256'] == plan['checkpoint_sha256'], 'checkpoint freeze mismatch')
                require(protocol['resolved_config']['sim']['decimation'] == 2 and
                        protocol['dt'] == protocol['resolved_config']['sim']['dt'] * 2,
                        'actual control timestep/config mismatch')
                require(protocol['effective_backstop']['pending_target_steps'] == DELAY and
                        protocol['effective_backstop']['backlog_aware'] is False,
                        'effective backlog/FIFO configuration changed')
                for name in ('cell_001.npz', 'input_recipe.npz', 'bank_assignment.npz',
                             'guard_metadata.json', 'random_manifest.json'):
                    if not (cell_root / name).is_file():
                        info['ready'] = False
                        issues.append(dict(block=block, mode=mode, finding=f'closed_artifact_missing:{name}',
                                           invalid_windows=ENVS))
            cells.append(info)
    common_research = []
    for p in plans:
        bank = str(Path(p['jobs'][0]['env']['SAFEDUO_INITIAL_BANK_NPZ']).resolve())
        metadata = str(Path(bank).with_name('metadata.json'))
        registered = p['research_source_sha256']
        require(bank in registered and metadata in registered, 'block bank/metadata not frozen in its own plan')
        # Exactly these TWO block-specific inputs differ by design. No shared
        # source difference, extra removal, or weak digest match is permitted.
        common_research.append({k:v for k,v in registered.items() if k not in (bank,metadata)})
    for i, p in enumerate(plans[1:], 1):
        require(p['source_sha256'] == plans[0]['source_sha256'] and
                common_research[i] == common_research[0], 'different shared frozen sources across blocks')
    return design, cells, issues


def exact(a, b, description):
    require(a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b),
            f'exact array binding failed: {description}')


def shape(a, wanted, name):
    require(a.shape == wanted, f'{name} shape {a.shape}, expected {wanted}')


def endpoints(margins, recorded_deep=None):
    require(margins.ndim == 3 and margins.shape[-1] == 4 and margins.dtype == np.float32,
            'official margin schema/dtype changed')
    # +inf can mean an empty/exempt class. NaN/-inf must never become safe.
    require(not np.isnan(margins).any() and not np.isneginf(margins).any(), 'invalid official margins')
    negative = margins < np.float32(0.)
    deep = margins < np.float32(-.005)  # same float32 strict scalar comparison as original torch producer
    if recorded_deep is not None:
        require(recorded_deep.dtype == np.bool_, 'official_deep must be boolean')
        exact(recorded_deep, deep, 'recorded deep versus original strict threshold')
    any_step = negative.any(-1)
    endpoint = any_step.any(0)
    damaging = deep.any((0, 2))
    first = np.where(endpoint, any_step.argmax(0), -1)
    first_deep = np.where(damaging, deep.any(-1).argmax(0), -1)
    return dict(failure=endpoint, deep_failure=damaging, class_failure=negative.any(0),
                class_deep=deep.any(0), first_failure_step=first, first_deep_step=first_deep,
                first_failure_class_mask=np.stack([negative[max(int(first[e]), 0), e] if endpoint[e]
                                                  else np.zeros(4, bool) for e in range(len(endpoint))]),
                violation_steps=any_step.sum(0), deep_steps=deep.any(-1).sum(0),
                min_margin=margins.min((0, 2)), min_by_class=margins.min(0),
                negative_class_env_steps=negative.sum((0, 1)))


def movement(q0, q, limits, cmd, executed, external=None):
    full = np.concatenate((q0[None], q), axis=0).astype(np.float64)
    delta = np.diff(full, axis=0)
    spans = limits[..., 1].astype(np.float64) - limits[..., 0].astype(np.float64)
    require(np.isfinite(full).all() and np.isfinite(spans).all() and (spans > 0).all(), 'invalid q/soft limits')
    per_arm_norm = np.stack([np.linalg.norm(delta[..., lo:hi], axis=-1) for lo, hi in ARM_SLICES], -1)
    moving = per_arm_norm > MOVING_L2_RAD
    active = slice(MOVING_START, None)
    require(q.shape[0] > MOVING_START, 'too short for registered moving window')
    command_l2 = np.linalg.norm(cmd.astype(np.float64)[active], axis=-1).sum(0)
    exec_l2 = np.linalg.norm(executed.astype(np.float64)[active], axis=-1).sum(0)
    external_l2 = np.linalg.norm((cmd if external is None else external).astype(np.float64)[active], axis=-1).sum(0)
    return dict(path_rad=np.abs(delta).sum((0, 2)),
                range_fraction=((full.max(0) - full.min(0)) / spans).mean(-1),
                all_arms_moving_fraction=moving[active].all(-1).mean(0),
                arm_moving_fraction=moving[active].mean(0),
                command_l2_sum=command_l2, executed_l2_sum=exec_l2, external_l2_sum=external_l2,
                path_after_zero_prefix_rad=np.abs(delta[active]).sum((0, 2)))


def native_descriptive_buffers(dense):
    """A separate float32 arithmetic oracle, not a physical tolerance.

    Native ratios reduce [900,192] in time-major/environment order. Native
    range means reduce [192,26]. Preserve these and independent float64 values.
    Original strict-negative endpoint comparisons are untouched.
    """
    full = np.concatenate((dense['q_initial'][None],dense['q']),axis=0)
    spans = dense['joint_soft_limits'][...,1]-dense['joint_soft_limits'][...,0]
    return dict(exec_norm=np.linalg.norm(dense['exec'][60:],axis=-1),
                external_norm=np.linalg.norm(dense['external_unscaled_cmd'][60:],axis=-1),
                ranges=(full.max(0)-full.min(0))/spans)


def native_descriptive_aggregate(buffers):
    exe=np.concatenate([b['exec_norm'] for b in buffers],axis=1)
    raw=np.concatenate([b['external_norm'] for b in buffers],axis=1)
    ranges=np.concatenate([b['ranges'] for b in buffers],axis=0)
    require(exe.dtype==raw.dtype==ranges.dtype==np.float32,'native description dtype changed')
    return dict(exec_external_l2_ratio=float(exe.sum()/raw.sum()),
                mean_joint_range_fraction=float(ranges.mean()),
                dtype='float32', ratio_reduction_axis='time-major [900,192]', range_reduction_axis='[192,26]')


def verify_fifo(dense, delay=DELAY):
    issued, applied = dense['controller_target'], dense['actuator_target']
    pending = dense['pre_pending_actuator_targets']
    history = dense['pre_pending_project_history']
    t, n, dofs = issued.shape
    shape(pending, (t, delay, n, dofs), 'pending [time,queue,env,joint]')
    shape(history, pending.shape, 'project pending history')
    q0 = dense['q_initial']
    exact(pending[0], np.broadcast_to(q0, (delay, n, dofs)), 'FIFO initial queue')
    initial_and_issued = np.concatenate((np.broadcast_to(q0, (delay, n, dofs)), issued), axis=0)
    exact(applied, initial_and_issued[:t], 'delivered target[t-6]')
    for k in range(delay):
        exact(pending[:, k], initial_and_issued[k:k+t], f'pending FIFO slot {k}')
    exact(history, pending, 'original pending-project history versus actuator queue')
    pre_target = np.concatenate((q0[None], issued[:-1]), axis=0)
    pre_q = np.concatenate((q0[None], dense['q'][:-1]), axis=0)
    exact(dense['pre_target_debt'], pre_target - pre_q, 'pre-target minus actual pre-q debt')
    exact(dense['effective_target_delta'], issued - pre_target, 'actual target increment')
    return dict(status='PASS_EXACT', steps=t, delay_steps=delay, pending_axis='[time,queue,env,joint]',
                applied_target_scalar_checks=int(applied.size), pending_scalar_checks=int(pending.size),
                project_history_scalar_checks=int(history.size), target_debt_scalar_checks=int(pre_q.size),
                scope='recorded controller-to-PD FIFO targets; not measured physical response or future trajectory')


def verify_commands(dense, recipe, scale):
    tape = recipe['tape']
    require(tape.ndim == 3 and tape.shape[0] >= dense['cmd'].shape[0] and tape.shape[1:] == dense['cmd'].shape[1:],
            'recipe tape too short or wrong env/joint axes')
    require(tape.dtype == np.float32 and np.isfinite(tape).all(), 'invalid command tape')
    actual = tape[:dense['cmd'].shape[0]]
    exact(dense['external_unscaled_cmd'], actual, 'actual external tape frame indexing')
    exact(dense['cmd'], actual * np.float32(scale), 'actual input scaling (float32 .36 or 1)')
    require(not np.any(actual[:60] != 0), 'registered zero60 prefix changed')
    return dict(status='PASS_EXACT', scale=scale, recorded_recipe_frames=int(len(tape)),
                executed_frames=int(len(actual)), scalar_checks=int(actual.size),
                scope='matched raw tape; scaled mode has different actual inputs; controlled outputs may diverge')


def json_number(x):
    return None if not np.isfinite(x) else float(x)


def scored_cell(info, dense, recipe, bank, assignment):
    required = ['q_initial', 'q', 'joint_soft_limits', 'initial_violation', 'cmd', 'exec',
                'official_margins', 'official_deep', 'controller_target', 'actuator_target',
                'external_unscaled_cmd', 'pre_target_debt', 'effective_target_delta',
                'pre_pending_actuator_targets', 'pre_pending_project_history', 'meta_json']
    require(set(required) <= set(dense), 'missing dense keys')
    shape(dense['q_initial'], (ENVS, DOFS), 'q_initial')
    shape(dense['joint_soft_limits'], (ENVS, DOFS, 2), 'soft limits')
    shape(dense['initial_violation'], (ENVS,), 'initial violations')
    for k in ('q', 'cmd', 'exec', 'controller_target', 'actuator_target', 'external_unscaled_cmd',
              'pre_target_debt', 'effective_target_delta'):
        shape(dense[k], (STEPS, ENVS, DOFS), k)
        require(dense[k].dtype == np.float32 and np.isfinite(dense[k]).all(), f'invalid dense {k}')
    for k in ('pre_pending_actuator_targets', 'pre_pending_project_history'):
        require(dense[k].dtype == np.float32 and np.isfinite(dense[k]).all(), f'invalid dense {k}')
    shape(dense['official_margins'], (STEPS, ENVS, 4), 'official margins')
    shape(dense['official_deep'], (STEPS, ENVS, 4), 'official deep')
    require(dense['initial_violation'].dtype == np.bool_ and not dense['initial_violation'].any(),
            'initial overlap/violation in preregistered qualified bank')
    for k in ('q_initial', 'initial_violation', 'joint_soft_limits'):
        exact(dense[k], recipe[k], f'dense versus recipe {k}')
    exact(dense['q_initial'], bank['accepted_q'], 'native initial q versus registered bank')
    exact(recipe['sampled_initial'], bank['accepted_q'], 'requested initial q versus bank')
    exact(dense['joint_soft_limits'], bank['joint_soft_limits'], 'native limits versus bank')
    labels = assignment['risk_pair_index']
    exact(labels, bank['risk_pair_index'], 'actual bank-assignment labels')
    shape(labels, (ENVS,), 'risk labels')
    require(dict(zip(*np.unique(labels, return_counts=True))) == {-1: 16, 0: 8, 1: 8, 2: 8, 3: 8, 4: 8, 5: 8},
            'changed actual bank strata')
    meta = json.loads(str(dense['meta_json'].item()))
    for k, v in info['job']['expected'].items():
        require(meta.get(k) == v, f'dense metadata mismatch: {k}')
    require(meta['cell_id'] == 1 and meta['dt'] == info['protocol']['dt'], 'dense cell/time metadata mismatch')
    scale = .36 if info['mode'] == 'admission_scaled_036' else 1.
    commands = verify_commands(dense, recipe, scale)
    fifo = verify_fifo(dense)
    ep = endpoints(dense['official_margins'], dense['official_deep'])
    mv = movement(dense['q_initial'], dense['q'], dense['joint_soft_limits'], dense['cmd'], dense['exec'],
                  dense['external_unscaled_cmd'])
    windows = []
    for e in range(ENVS):
        windows.append(dict(env=e, risk_pair_index=int(labels[e]), violation=bool(ep['failure'][e]),
            deep=bool(ep['deep_failure'][e]), first_failure_step=int(ep['first_failure_step'][e]),
            first_deep_step=int(ep['first_deep_step'][e]), first_failure_class_mask=ep['first_failure_class_mask'][e].tolist(),
            violation_steps=int(ep['violation_steps'][e]), deep_steps=int(ep['deep_steps'][e]),
            class_failure=ep['class_failure'][e].tolist(), class_deep=ep['class_deep'][e].tolist(),
            min_margin_m=json_number(ep['min_margin'][e]),
            min_by_class_m=[json_number(x) for x in ep['min_by_class'][e]],
            measured_joint_path_rad=float(mv['path_rad'][e]), range_fraction=float(mv['range_fraction'][e]),
            all_arms_moving_fraction=float(mv['all_arms_moving_fraction'][e]),
            arm_moving_fraction=mv['arm_moving_fraction'][e].tolist(),
            command_l2_sum=float(mv['command_l2_sum'][e]), executed_l2_sum=float(mv['executed_l2_sum'][e]),
            external_l2_sum=float(mv['external_l2_sum'][e]),
            path_after_zero_prefix_rad=float(mv['path_after_zero_prefix_rad'][e])))
    result = dict(block=info['block'], mode=info['mode'], seed=info['seed'], initial_seed=info['initial_seed'],
                  id=info['id'], root=str(info['root']), valid_windows=ENVS, invalid_windows=0,
                  command_audit=commands, fifo_audit=fifo, windows=windows)
    result.update(summarize_windows(windows))
    return result


def summarize_windows(windows):
    require(bool(windows), 'empty score denominator')
    command = sum(w['command_l2_sum'] for w in windows)
    executed = sum(w['executed_l2_sum'] for w in windows)
    external = sum(w.get('external_l2_sum', w['command_l2_sum']) for w in windows)
    minima = [w['min_margin_m'] for w in windows if w['min_margin_m'] is not None]
    return dict(valid_windows=len(windows), violations=sum(w['violation'] for w in windows),
                deep=sum(w['deep'] for w in windows),
                class_counts=[sum(w['class_failure'][k] for w in windows) for k in range(4)],
                deep_class_counts=[sum(w['class_deep'][k] for w in windows) for k in range(4)],
                negative_env_steps=sum(w['violation_steps'] for w in windows),
                worst_margin_m=min(minima) if minima else None,
                mean_joint_path_rad=float(np.mean([w['measured_joint_path_rad'] for w in windows])),
                mean_joint_range_fraction=float(np.mean([w['range_fraction'] for w in windows])),
                four_arms_moving_fraction=float(np.mean([w['all_arms_moving_fraction'] for w in windows])),
                command_l2_sum=command, executed_l2_sum=executed, external_l2_sum=external,
                exec_input_l2_ratio=executed / command if command else None,
                exec_external_l2_ratio=executed / external if external else None)


def paired_result(before, after):
    """Case keys must match exactly; return ALL discordant case IDs."""
    require(set(before) == set(after), 'pair missing/unmatched case keys')
    rescued = sorted(k for k in before if before[k] and not after[k])
    new = sorted(k for k in before if not before[k] and after[k])
    both = sorted(k for k in before if before[k] and after[k])
    neither = sorted(k for k in before if not before[k] and not after[k])
    return dict(paired_cases=len(before), rescued=len(rescued), new_failures=len(new),
                both_fail=len(both), neither_fail=len(neither), rescued_case_ids=rescued, new_case_ids=new)


def bind_sources(info, metadata, inputs):
    plan, protocol = info['plan'], info['protocol']
    # Full production source map is read-only bound. Unrelated historical raw
    # datasets and parent analyzer implementations are NOT consumed for scoring.
    for relative, expected in plan['source_sha256'].items():
        inputs.bind_file(under(Path(plan['cwd']) / relative, plan['cwd']), expected)
    inputs.bind_file(plan['checkpoint_path'], plan['checkpoint_sha256'])
    for path, expected in metadata['sources'].items():
        require(plan['research_source_sha256'].get(path) == expected, 'helper not registered under frozen SHA')
        inputs.bind_file(path, expected)
    require(metadata['mode'] == info['mode'] and metadata['strict_fifo_steps'] == DELAY and
            metadata['capacity'] == 1024 and metadata['original_actor_rows'] == 32 and
            metadata['queue_preemption'] is False and metadata['production_promoted'] is False and
            metadata['structural_and_conditional_exemptions_unchanged'] is True and
            metadata['unavoidable_slack_is_safety'] is False,
            'guard contract changed')
    require(metadata['command_scale'] == (.36 if info['mode'] == 'admission_scaled_036' else 1.),
            'guard scale metadata mismatch')
    require(metadata['reference'] == (info['mode'] in ('joint_reference', 'joint_repair')) and
            metadata['joint_repair'] == (info['mode'] == 'joint_repair'), 'method/reference/repair mismatch')
    require(protocol['effective_backstop']['tol'] == 1e-6, 'original solver tolerance changed')


def compare_parent(parent, score):
    """Explicit delivery schema from build_report.py/verify_science.py.

    These delivery interfaces were read, not executed/imported. Parent scoring
    implementations are neither read nor imported. Missing/different schema is
    an error, not a guessed alias. Per-case parent schema remains pending until
    actual final data; our complete per-case ledger is independently generated.
    """
    for key in ('totals', 'coverage', 'paired', 'rows', 'cases',
                'completed_method_windows', 'invalid_method_windows'):
        require(key in parent, f'final parent delivery schema missing {key}')
    diffs, checked, precision_differences = [], 0, []

    def check(path, actual, expected, arithmetic=False):
        nonlocal checked
        checked += 1
        if arithmetic:
            good = ((actual is None and expected is None) or
                    (actual is not None and expected is not None and
                     bool(np.isclose(actual, expected, rtol=1e-7, atol=1e-9))))
        else:
            good = actual == expected
        if not good:
            diffs.append(dict(path=path, parent=actual, independent=expected))

    check('completed_method_windows', parent['completed_method_windows'], 768)
    check('invalid_method_windows', parent['invalid_method_windows'], 0)
    check('cases.length', len(parent['cases']), 192)
    check('rows.length', len(parent['rows']), 12)
    own_cells = {(c['seed'], c['mode']): c for c in score['cells']}
    require(len(own_cells) == 12, 'independent cell identity duplicate')
    parent_cells = {(r['seed'], r['mode']): r for r in parent['rows']}
    require(len(parent_cells) == 12 and set(parent_cells) == set(own_cells), 'parent cell seed/mode membership differs')
    for (seed, mode), row in parent_cells.items():
        own = own_cells[(seed, mode)]
        for k, v in (('status','complete'), ('completed_windows',64), ('invalid_windows',0),
                     ('violations',own['violations']), ('deep',own['deep']), ('class_violations',own['class_counts'])):
            check(f'rows.{seed}.{mode}.{k}', row[k], v)
    parent_cases = {(r['seed'], r['env']):r for r in parent['cases']}
    own_cases = {(c['seed'], w['env']):{} for c in score['cells'] for w in c['windows']}
    for c in score['cells']:
        for w in c['windows']:own_cases[(c['seed'],w['env'])][c['mode']] = w
    require(len(parent_cases)==192 and set(parent_cases)==set(own_cases), 'parent paired-case identity differs/duplicates')
    for (seed, env), case in parent_cases.items():
        require(set(case['methods'])==set(MODES), 'parent case method set differs')
        for mode in MODES:
            given=case['methods'][mode];own=own_cases[(seed,env)][mode]
            path=f'cases.{seed}.{env}.{mode}'
            check(path+'.stratum', case['stratum'], own['risk_pair_index'])
            check(path+'.failed', given['failed'], own['violation'])
            check(path+'.deep', given['deep'], own['deep'])
            check(path+'.first_failure_step', given['first_failure_step'],
                  own['first_failure_step'] if own['violation'] else None)
            require(len(given['min_mm'])==4, 'parent class minimum shape changed')
            for k in range(4):
                v=own['min_by_class_m'][k]
                check(path+f'.min_mm.{k}', given['min_mm'][k], v*1000 if v is not None else None, True)
    totals, coverage = {}, {}
    for r in parent['totals']:
        require(r['mode'] not in totals, 'duplicate parent total mode')
        totals[r['mode']] = r
    for r in parent['coverage']:
        require(r['mode'] not in coverage, 'duplicate parent coverage mode')
        coverage[r['mode']] = r['measured']
    require(set(totals) == set(MODES) and set(coverage) == set(MODES), 'parent methods incomplete/extra')
    for mode in MODES:
        expected = score['by_mode'][mode]
        for key in ('violations', 'deep'):
            check(f'totals.{mode}.{key}', totals[mode][key], expected[key])
        check(f'totals.{mode}.completed_windows', totals[mode]['completed_windows'], expected['valid_windows'])
        check(f'totals.{mode}.class_violations', totals[mode]['class_violations'], expected['class_counts'])
        check(f'totals.{mode}.min_nonexempt_mm', totals[mode]['min_nonexempt_mm'],
              expected['worst_margin_m'] * 1000 if expected['worst_margin_m'] is not None else None, True)
        check(f'totals.{mode}.four_arms_moving_fraction', totals[mode]['four_arms_moving_fraction'],
              expected['four_arms_moving_fraction'], True)
        check(f'totals.{mode}.exec_external_l2_ratio', totals[mode]['exec_external_l2_ratio'],
              score['native_float32_aggregates'][mode]['exec_external_l2_ratio'])
        for parent_key, own_key in (('mean_joint_path_rad', 'mean_joint_path_rad'),
                                    ('mean_within_window_joint_range', 'mean_joint_range_fraction')):
            if own_key=='mean_joint_range_fraction':
                check(f'coverage.{mode}.{parent_key}', coverage[mode][parent_key],
                      score['native_float32_aggregates'][mode][own_key])
            else:
                check(f'coverage.{mode}.{parent_key}', coverage[mode][parent_key], expected[own_key], True)
        for path, value, double in (
            (f'totals.{mode}.exec_external_l2_ratio',totals[mode]['exec_external_l2_ratio'],expected['exec_external_l2_ratio']),
            (f'coverage.{mode}.mean_within_window_joint_range',coverage[mode]['mean_within_window_joint_range'],expected['mean_joint_range_fraction'])):
            precision_differences.append(dict(path=path,parent_float32=value,independent_float64=double,
                                             signed_difference=double-value,native_float32_matches_exact=True,
                                             scope='descriptive reduction precision only; no endpoint tolerance change'))
    paired = parent['paired']
    groups = {}
    for record in paired:
        key = f'{record["a"]}__{record["b"]}'
        require(key in score['paired'], f'unrecognized parent pair direction: {key}')
        group = groups.setdefault(key, dict(rescued=0, new_failures=0, both=0, neither=0))
        for k in group:
            require(isinstance(record[k], int) and record[k] >= 0, 'invalid parent paired count')
            group[k] += record[k]
    require(bool(groups), 'parent paired table missing')
    for key, record in groups.items():
        # Exactly the summation used by the report producer; do not guess
        # unpublished parent row-level seed/ID field names.
        require(sum(record.values()) == 192, f'parent paired denominator incomplete: {key}')
        expected = score['paired'][key]['strict_negative']
        for parent_key, own_key in (('rescued', 'rescued'), ('new_failures', 'new_failures'),
                                    ('both', 'both_fail'), ('neither', 'neither_fail')):
            check(f'paired.{key}.{parent_key}', record[parent_key], expected[own_key])
    return dict(status='FAIL' if diffs else 'PASS_EXPLICIT_DELIVERY_FIELDS', checked_fields=checked,
                discrepancies=diffs, descriptive_comparison_rtol=1e-7, descriptive_comparison_atol=1e-9,
                float64_descriptive_differences=precision_differences,
                compared_pairs=list(groups), unrepresented_parent_pairs=sorted(set(score['paired'])-set(groups)),
                scope='all method totals/classes/movement and reported aggregate paired counts; own all-six-pair '
                      'per-case ledger saved; '
                      'parent per-case field mapping, report/plots/full-forecast/camera remain separate')


def score_all(cells, inputs):
    require(len(cells) == 12 and all(c['ready'] for c in cells), 'all 12 cells must be closed BEFORE any dense consumption')
    results, binding, case_inputs = [], [], []
    native_buffers={mode:[] for mode in MODES}
    dense_keys = ('q_initial', 'q', 'joint_soft_limits', 'initial_violation', 'cmd', 'exec',
                  'official_margins', 'official_deep', 'controller_target', 'actuator_target',
                  'external_unscaled_cmd', 'pre_target_debt', 'effective_target_delta',
                  'pre_pending_actuator_targets', 'pre_pending_project_history', 'meta_json')
    for block in range(3):
        block_cells = [c for c in cells if c['block'] == block]
        require([c['mode'] for c in block_cells] == list(MODES), 'invalid block method collection')
        bank_path = block_cells[0]['bank']
        bank = inputs.npz(bank_path, ('accepted_q', 'joint_soft_limits', 'risk_pair_index'))
        bank_meta = inputs.json(bank_path.with_name('metadata.json'))
        require(inputs.receipts[str(bank_path.resolve())]['sha256_before'] == bank_meta['bank_sha256'], 'bank metadata SHA mismatch')
        registered = block_cells[0]['plan']['research_source_sha256'][str(bank_path)]
        require(registered == bank_meta['bank_sha256'], 'registered bank SHA mismatch')
        metadata_path = bank_path.with_name('metadata.json').resolve()
        require(block_cells[0]['plan']['research_source_sha256'][str(metadata_path)] ==
                inputs.receipts[str(metadata_path)]['sha256_before'], 'registered bank metadata SHA mismatch')
        prior_recipe, prior_protocol = None, None
        for c in block_cells:
            root = c['root']
            metadata = inputs.json(root / 'guard_metadata.json')
            bind_sources(c, metadata, inputs)
            manifest = inputs.json(root / 'random_manifest.json')
            require(manifest['initial']['bank_sha256'] == bank_meta['bank_sha256'] and
                    Path(manifest['initial']['bank_path']).resolve() == bank_path.resolve(), 'manifest bank identity mismatch')
            dense = inputs.npz(root / 'cell_001.npz', dense_keys)
            recipe = inputs.npz(root / 'input_recipe.npz')
            assignment = inputs.npz(root / 'bank_assignment.npz', ('risk_pair_index', 'initial_table_raw_min'))
            if prior_recipe is not None:
                require(set(recipe) == set(prior_recipe), 'paired recipe key set differs')
                for k in recipe:
                    exact(recipe[k], prior_recipe[k], f'paired full input recipe {k}')
                for k in ('resolved_config', 'effective_backstop', 'effective_coordinator', 'design', 'dt'):
                    require(c['protocol'][k] == prior_protocol[k], f'paired protocol differs: {k}')
            else:
                prior_recipe, prior_protocol = recipe, c['protocol']
                case_inputs.extend(dict(block=block, seed=c['seed'], env=e,
                                        raw_tape_sha256=sha(recipe['tape'][:, e].tobytes()),
                                        initial_q_sha256=sha(recipe['q_initial'][e].tobytes())) for e in range(ENVS))
            result = scored_cell(c, dense, recipe, bank, assignment)
            native_buffers[c['mode']].append(native_descriptive_buffers(dense))
            result['dense_sha256'] = inputs.receipts[str((root / 'cell_001.npz').resolve())]['sha256_before']
            results.append(result)
            binding.append(dict(block=block, mode=c['mode'], initial_sha256=sha(dense['q_initial'].tobytes()),
                                raw_tape_sha256=sha(recipe['tape'].tobytes()),
                                bank_sha256=bank_meta['bank_sha256'], exact_full_recipe_pairing=True))
            del dense
    by_mode, pairs = {}, {}
    outcomes, deep_outcomes = {}, {}
    for mode in MODES:
        windows = [w for c in results if c['mode'] == mode for w in c['windows']]
        require(len(windows) == 192, 'incomplete method denominator')
        by_mode[mode] = summarize_windows(windows)
        outcomes[mode] = {f'{c["seed"]}/env_{w["env"]:03d}': w['violation']
                          for c in results if c['mode'] == mode for w in c['windows']}
        deep_outcomes[mode] = {f'{c["seed"]}/env_{w["env"]:03d}': w['deep']
                               for c in results if c['mode'] == mode for w in c['windows']}
    for before, after in itertools.combinations(MODES, 2):
        pairs[f'{before}__{after}'] = dict(reference=before, comparison=after,
            strict_negative=paired_result(outcomes[before], outcomes[after]),
            deep_negative=paired_result(deep_outcomes[before], deep_outcomes[after]))
    require(len({x['raw_tape_sha256'] for x in binding}) == 3, 'identical full tapes across independent seed blocks')
    require(len({x['initial_sha256'] for x in binding}) == 3, 'identical initial bank arrays across blocks')
    return dict(cells=results, by_mode=by_mode, paired=pairs, input_binding=binding,
                native_float32_aggregates={mode:native_descriptive_aggregate(native_buffers[mode]) for mode in MODES},
                case_input_identities=case_inputs,
                unique_actual_raw_tapes=len({x['raw_tape_sha256'] for x in case_inputs}),
                unique_actual_initial_q=len({x['initial_q_sha256'] for x in case_inputs}),
                paired_cases=192, correlated_method_windows=768, valid_windows=768, invalid_windows=0)


def run(h, raw, metadata_only=False, parent_path=None):
    inputs = Inputs()
    report = dict(schema=SCHEMA, started_utc=datetime.now(timezone.utc).isoformat(),
                  executed_source_sha256=globals().get('ASTRA_EXECUTED_SOURCE_SHA256'),
                  python=platform.python_version(), numpy=np.__version__,
                  physical_strategy_status='BLOCKED', safety_strategy_approved=False,
                  hardware_authorized=False, production_promoted=False,
                  scope='this study only: 3 banks, 192 paired cases, 768 correlated method windows; no IID interval',
                  report_plot_claims_review='PENDING_SEPARATE_FINAL_REVIEW',
                  parent_full_forecast_first_failure_camera_audits='outside this independent dense scorer',
                  definitions=dict(violation='float32 original nonexempt official margin strictly <0; all960 frames',
                      deep='float32 original official margin strictly <-0.005m; no collision tolerance',
                      moving='frame60..959: all four arm displacement L2 each strictly >0.001rad',
                      path='float64 absolute joint increments summed; includes initial-to-frame0',
                      range='float64 per-joint peak-to-peak including initial divided by native soft-limit span, mean26',
                      exec_input='sum frame60..959 full26-joint L2(exec) divided by same L2(input)',
                      exec_external='same executed numerator divided by saved unscaled raw-tape L2; separate from actual scaled input',
                      class_counts='per-window any negative per class; overlapping, not mutually exclusive'),
                  geometric_reconstruction='official post-state margins and producer exemption convention; '
                      'no independent whole-stream FK/mesh/contact rederivation')
    try:
        _, cells, issues = preflight(h, raw, inputs)
        invalid = sum(not c['ready'] for c in cells) * ENVS
        report.update(status='NOT_READY' if issues else 'READY_METADATA_ONLY',
                      registered_method_windows=768, invalid_windows=invalid,
                      scoring_executed=False, blocking_findings=issues)
        if not issues and not metadata_only:
            report.update(score_all(cells, inputs))
            report['status'] = 'PASS_INDEPENDENT_DENSE_SCORE'
            report['scoring_executed'] = True
            prior_path=h/'astra_final_score_attempt2.json'
            if prior_path.is_file():
                prior=inputs.json(prior_path)
                require(prior['status']=='BLOCKED_PARENT_DISCREPANCY' and
                        prior['by_mode']==report['by_mode'] and prior['paired']==report['paired'] and
                        prior['cells']==report['cells'], 'physical endpoint/case/pair/movement changed across oracle precision revision')
                require(len(prior['parent_comparison']['discrepancies'])==3,'unexpected old binary64 comparison failure scope')
                old_source=h/'astra_final_score_v2.py'
                inputs.bind_file(old_source,prior['executed_source_sha256'])
                report['prior_binary64_comparison']=dict(status='BLOCKED_PARENT_DISCREPANCY',
                    receipt_path=str(prior_path), receipt_sha256=inputs.receipts[str(prior_path.resolve())]['sha256_before'],
                    executed_source_sha256=prior['executed_source_sha256'],
                    discrepancies=prior['parent_comparison']['discrepancies'],
                    old_counts_cases_all_six_pairs_and_float64_results_exactly_unchanged=True,
                    scope='retained original FAIL; not relabeled as all-PASS or hidden by wider tolerances')
            if parent_path is not None and parent_path.is_file():
                try:
                    parent = inputs.json(parent_path)
                    execution_path = h / 'offline_analysis_execution.json'
                    execution = inputs.json(execution_path)
                    require(execution['status'] == 'PASS_EXACT_REGISTERED_SOURCE_EXECUTION' and
                            execution['result_sha256'] == inputs.receipts[str(parent_path.resolve())]['sha256_before'],
                            'closed parent execution/result SHA binding failed')
                    report['parent_offline_execution_receipt'] = execution
                    for source, digest in execution['executed_source_sha256'].items():
                        inputs.bind_file(source,digest)
                    parent_input_bindings=[]
                    for row in parent['rows']:
                        own=next(c for c in report['cells'] if c['seed']==row['seed'] and c['mode']==row['mode'])
                        require(Path(row['path']).resolve()==Path(own['root']).resolve(),'parent row raw path changed')
                        for name in ('protocol.json','cell_001.npz','input_recipe.npz','bank_assignment.npz','guard_metadata.json'):
                            path=(Path(own['root'])/name).resolve()
                            actual=inputs.receipts[str(path)]['sha256_before']
                            require(row['input_sha256'][name]==actual,f'parent consumed input SHA mismatch: {path}')
                            parent_input_bindings.append(dict(path=str(path),sha256=actual,exact=True))
                    report['parent_consumed_core_input_sha_matches']=parent_input_bindings
                    completion_path=h/'analysis_completion_readback.json'
                    if completion_path.is_file():
                        completion=inputs.json(completion_path)
                        require(completion['result_sha256']==execution['result_sha256'] and
                                completion['offline_execution_receipt_sha256']==inputs.receipts[str(execution_path.resolve())]['sha256_before'],
                                'parent completion readback SHA binding failed')
                        report['parent_analysis_completion_readback'] = completion
                    report['parent_execution_boundary'] = dict(outer_exit_status=143,
                        provenance='user disclosed outer exit143; closed analysis_completion_readback.json preserved if present',
                        reason='undetermined', clean_CLI_exit_proven=False,
                        scope='closed result/compiled-source binding does not prove clean CLI exit; independent raw scoring is separate')
                    report['parent_comparison'] = compare_parent(parent, report)
                    if report['parent_comparison']['discrepancies']:
                        report['status'] = 'BLOCKED_PARENT_DISCREPANCY'
                    else:
                        report['status'] = 'PASS_ENDPOINTS_AND_NATIVE_FLOAT32_REDUCTION_BINDING'
                        report['binary64_all_fields_comparison_status'] = 'FAIL_THREE_DESCRIPTIVE_FIELDS_RETAINED'
                except (AuditError, KeyError, ValueError) as error:
                    report['parent_comparison'] = dict(status='BLOCKED_INTERFACE', error=str(error))
                    report['status'] = 'BLOCKED_PARENT_INTERFACE'
            else:
                report['parent_comparison'] = dict(status='PENDING_PARENT_RESULTS')
    except (AuditError, KeyError, ValueError, OSError) as error:
        report.update(status='BLOCKED', scoring_executed=False,
                      invalid_windows=768, blocking_findings=[dict(error=f'{type(error).__name__}: {error}')])
    try:
        report['consumed_inputs'] = inputs.finish()
    except (AuditError, OSError) as error:
        report.update(status='BLOCKED_INPUT_MUTATION', scoring_executed=False, invalid_windows=768)
        report['scored_results_admissible'] = False
        report.setdefault('blocking_findings', []).append(dict(error=str(error)))
        report['consumed_inputs'] = list(inputs.receipts.values())
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check-ready', action='store_true')
    action.add_argument('--score', action='store_true')
    parser.add_argument('--h', type=Path, default=H_DEFAULT)
    parser.add_argument('--raw', type=Path, default=RAW_DEFAULT)
    parser.add_argument('--output', type=Path, help='default H/ASTRA_FINAL_SCORE.json; refuse overwrite')
    parser.add_argument('--parent', type=Path, help='default H/holdout_results.json, if available')
    args = parser.parse_args()
    h, raw = args.h.resolve(), args.raw.resolve()
    output = (args.output or h / 'ASTRA_FINAL_SCORE.json').absolute()
    require(output.parent.resolve() == h and
            (output.name == 'ASTRA_FINAL_SCORE.json' or (output.name.startswith('astra_') and output.suffix == '.json')),
            'output must be H/ASTRA_FINAL_SCORE.json or an owned H/astra_*.json')
    require(not output.exists(), f'refuse overwrite: {output}')
    result = run(h, raw, args.check_ready, args.parent or h / 'holdout_results.json')
    source = Path(__file__).resolve()
    require(hash_file(source) == ASTRA_EXECUTED_SOURCE_SHA256, 'oracle source changed while executing')
    # The exact executed oracle is already this immutable input source; record
    # its pre/post bytes, do not generate a post-hoc implementation archive.
    result['executed_source'] = dict(path=str(source), sha256=ASTRA_EXECUTED_SOURCE_SHA256,
                                    sha256_after=hash_file(source), byte_length=ASTRA_EXECUTED_SOURCE_LENGTH,
                                    execution='read once, compile/exec those exact bytes')
    with output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], output=str(output), scoring_executed=result['scoring_executed'],
                          invalid_windows=result['invalid_windows'], executed_source_sha256=ASTRA_EXECUTED_SOURCE_SHA256)))
    return 0 if result['status'] in ('PASS_INDEPENDENT_DENSE_SCORE', 'READY_METADATA_ONLY',
                                    'PASS_ENDPOINTS_AND_NATIVE_FLOAT32_REDUCTION_BINDING') else 2


if __name__ == '__main__' and not globals().get('ASTRA_SOURCE_COMPILED'):
    # Execute the bytes whose SHA is recorded, not a later reread of the file.
    source_path = Path(__file__).resolve()
    source_bytes = source_path.read_bytes()
    scope = dict(__name__='__main__', __file__=str(source_path), ASTRA_SOURCE_COMPILED=True,
                 ASTRA_EXECUTED_SOURCE_SHA256=sha(source_bytes), ASTRA_EXECUTED_SOURCE_LENGTH=len(source_bytes))
    exec(compile(source_bytes, str(source_path), 'exec'), scope)
elif __name__ == '__main__':
    sys.exit(main())
