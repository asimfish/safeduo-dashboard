"""Bounded CPU-only joint-feasibility oracle; never imports the simulator.

First run --freeze (no LP), then --self-test and --evaluate. Choices are immutable
across evaluation. HiGHS tests the captured, authority-adjusted constraints, not a
new policy. Arithmetic tolerances are not physical safety thresholds.
"""
import argparse
import datetime as dtm
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
OLD = Path('/home/liyufeng/safeduo/artifacts/safety_joint_guard_20261005_1005')
CELL = Path('/mnt/nas/data/lyf/double_hand/safety_joint_guard_20261005_1005/holdout/joint_guard_1701627244')
OUT = HERE / 'ASTRA_JOINT_ORACLE.json'
STEPS = [0, 60, 62, 66, 71, 75, 180, 480, 959]
ARMS = [(0, 7), (7, 14), (14, 20), (20, 26)]
SLICES = {'F': slice(0, 14), 'U': slice(14, 26)}
ARITH = 5e-7
LP_TOL = 1e-9
WITNESS_TOL = 1e-8


def now():
    return dtm.datetime.now(dtm.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest_json(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_npz(path, keys=None, expected=None):
    raw = Path(path).read_bytes()
    if expected is not None:
        assert hashlib.sha256(raw).hexdigest() == expected, f'input changed: {path}'
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        return {k: z[k] for k in (z.files if keys is None else keys)}


def read_json(path):
    return json.loads(Path(path).read_bytes())


def save(x):
    OUT.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def manifest():
    metadata = read_json(CELL / 'guard_metadata.json')
    paths = [CELL / name for name in ('protocol.json', 'guard_metadata.json', 'cell_001.npz', 'project_diagnostics.npz')]
    paths += [CELL / 'projection_snapshots' / f'step_{t:04d}.npz' for t in STEPS]
    paths += [OLD / 'regression_context.json', Path('/home/liyufeng/safeduo/src/safeduo/safety/backstop.py')]
    paths += list(map(Path, metadata['sources']))
    return {str(p): sha(p) for p in paths}


def freeze():
    assert not OUT.exists(), 'refuse overwriting an existing choice/result receipt'
    inputs = manifest()
    p, meta = read_json(CELL / 'protocol.json'), read_json(CELL / 'guard_metadata.json')
    assert p['status'] == 'complete' and p['steps'] == 960 and p['args']['num_envs'] == 64
    assert meta['mode'] == 'joint_guard' and meta['reference_gap_rad'] == .05 and meta['capacity'] == 1024
    summaries = []
    keys = [f'{k}_{r}' for r in ('F', 'U') for k in (
        'returned_safety_residual', 'returned_alpha_residual', 'returned_bound_residual',
        'individual_infeasibility_lower_bound')]
    for t in STEPS:
        path = CELL / 'projection_snapshots' / f'step_{t:04d}.npz'
        s = read_npz(path, keys, inputs[str(path)])
        for e in range(64):
            safety = max(float(s[f'returned_safety_residual_{r}'][e]) for r in ('F', 'U'))
            all_res = max(float(s[f'returned_{k}_residual_{r}'][e]) for r in ('F', 'U') for k in ('safety', 'alpha', 'bound'))
            indiv = max(float(s[f'individual_infeasibility_lower_bound_{r}'][e]) for r in ('F', 'U'))
            summaries.append(dict(step=t, env=e, saved_safety_residual=safety, saved_all_residual=all_res,
                                  saved_individual_lower_bound=indiv))
    choices = []
    seen = set()

    def add(row, reason):
        key = (row['step'], row['env'])
        assert key not in seen
        seen.add(key)
        choices.append(dict(row, reason=reason))

    contexts = [x for x in read_json(OLD / 'regression_context.json')['rows']
                if x['mode'] == 'joint_guard' and x['seed'] == 1701627244]
    assert sorted(x['env'] for x in contexts) == [21, 49]
    for context in sorted(contexts, key=lambda x: x['env']):
        failure = context['first_failure_step']
        adjacent = [max(t for t in STEPS if t < failure), min(t for t in STEPS if t > failure)]
        for t in adjacent:
            add(next(x for x in summaries if x['step'] == t and x['env'] == context['env']),
                f'context env{context["env"]} failure@{failure}: adjacent registered frame, not failure frame')
    for t in (66, 71, 75, 180, 480, 959):
        eligible = [x for x in summaries if x['step'] == t and (t, x['env']) not in seen]
        x = sorted(eligible, key=lambda x: (-x['saved_safety_residual'], x['env']))[0]
        assert x['saved_safety_residual'] > 0
        add(x, 'largest saved returned safety residual at this registered step')
    count = 0
    for t in STEPS:
        eligible = [x for x in summaries if x['step'] == t and x['saved_all_residual'] == 0 and (t, x['env']) not in seen]
        if eligible and count < 6:
            add(min(eligible, key=lambda x: x['env']), 'exact-zero saved safety/alpha/bound residual control; first unselected env')
            count += 1
    assert count == 6
    eligible = [x for x in summaries if (x['step'], x['env']) not in seen and x['saved_individual_lower_bound'] > 0]
    for x in sorted(eligible, key=lambda x: (-x['saved_individual_lower_bound'], x['step'], x['env']))[:2]:
        add(x, 'largest remaining saved individual infeasibility lower bound')
    assert len(choices) == 18
    # Selection has used producer residuals and context ONLY; no linprog call.
    receipt = dict(schema='astra.joint_feasibility.v1', status='CHOICES_FROZEN_NOT_EVALUATED',
                   choices_frozen_utc=now(), choices=choices, choice_sha256=digest_json(choices),
                   oracle_source_sha256=sha(__file__), input_sha256=inputs,
                   selection_lp_calls=0, maximum_env_snapshots=18, raw_cell=str(CELL),
                   context_rows=contexts)
    for path, expected in inputs.items():
        assert sha(path) == expected, f'input changed during freeze: {path}'
    save(receipt)
    print(json.dumps({'status': receipt['status'], 'choice_sha256': receipt['choice_sha256'],
                      'choices': [(x['step'], x['env']) for x in choices]}))


class Checks:
    def __init__(self):
        self.count, self.errors, self.failures = 0, {}, []

    def check(self, label, value):
        self.count += 1
        if not bool(value):
            self.failures.append(label)

    def exact(self, label, a, b):
        self.check(label, np.array_equal(a, b))

    def close(self, label, a, b):
        a, b = np.asarray(a), np.asarray(b)
        self.count += 1
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            self.failures.append(label + ': shape/nonfinite')
            return
        err = float(np.abs(a.astype(float) - b.astype(float)).max(initial=0))
        self.errors[label] = err
        if err > ARITH:
            self.failures.append(label)


def effective_distance(s, cfg):
    d = s['snapshot_d'].astype(float)
    rate = sum(np.sum(s['snapshot_J_' + r].astype(float) * s['snapshot_qd'][SLICES[r]][None].astype(float), -1) for r in ('F', 'U'))
    horizon = np.full(d.shape, cfg['lookahead_s'] or 0)
    horizon[s['snapshot_cls'] == 1] = max(cfg['lookahead_s'] or 0, cfg['self_lookahead_s'] or 0)
    table = s['snapshot_cls'] == 2
    if bool(s['snapshot_struct_exempt_available']):
        permanent = s['snapshot_struct_exempt'].copy()
        if bool(s['snapshot_contact_exempt_available']):
            permanent &= ~s['snapshot_contact_exempt']
        table &= ~permanent
    horizon[table] = max(cfg['lookahead_s'] or 0, cfg['table_lookahead_s'] or 0)
    eff = d + horizon * np.minimum(rate, 0)
    if cfg['predict_backlog']:
        targets = [s['snapshot_backlog']] + list(s['snapshot_past_backlogs'])
        displacement = [sum(np.sum(s['snapshot_J_' + r].astype(float) * target[SLICES[r]][None].astype(float), -1) for r in ('F', 'U')) for target in targets]
        eff = np.minimum(eff, d + np.minimum(np.minimum.reduce(displacement), 0))
    return eff


def positive_max(x):
    return float(np.maximum(x, 0).max(initial=0))


def residuals(A, b, labels, lo, hi, u):
    signed = A @ u - b
    return dict(safety=positive_max(signed[[i for i, row in enumerate(labels) if row['kind'] == 'safety']]),
                alpha=positive_max(signed[[i for i, row in enumerate(labels) if row['kind'] == 'alpha']]),
                bounds=positive_max(np.r_[lo - u, u - hi]))


def reconstruct(s, dense, diagnostics, cfg, t, e):
    c = Checks()
    for k, v in s.items():
        c.check('finite_' + k, np.isfinite(v).all())
    m = len(s['snapshot_d'])
    ids = s['selected_ids'][:m]
    c.exact('valid_ID_mask', ids >= 0, s['snapshot_valid'])
    c.check('IDs_padding_range_unique', (s['selected_ids'][m:] == -1).all() and
            ((ids[ids >= 0] < 9021).all()) and len(np.unique(ids[ids >= 0])) == int((ids >= 0).sum()))
    q = dense['q_initial'][e] if t == 0 else dense['q'][t - 1, e]
    target = dense['q_initial'][e] if t == 0 else dense['controller_target'][t - 1, e]
    box = cfg['vmax'] * .016666
    limits = dense['joint_soft_limits'][e]
    base_lo = np.maximum(limits[:, 0] - target, np.float32(-box))
    base_hi = np.minimum(limits[:, 1] - target, np.float32(box))
    c.check('original_bounds_nonempty', (base_lo <= base_hi).all())
    lower = np.clip((q - np.float32(.05)) - target, base_lo, base_hi)
    upper = np.clip((q + np.float32(.05)) - target, base_lo, base_hi)
    c.exact('reference_lower', lower, s['bounds_lower'])
    c.exact('reference_upper', upper, s['bounds_upper'])
    c.exact('limited_input', np.clip(dense['cmd'][t, e], lower, upper), s['project_input_cmd'])
    c.exact('raw_cmd', dense['cmd'][t, e], s['external_raw_cmd'])
    c.exact('returned_dense', dense['exec'][t, e], s['returned_cmd'])
    c.exact('returned_outer', s['returned_cmd'], s['outer_returned_cmd'])
    delta = dense['controller_target'][t, e] - target
    c.exact('actual_target_delta', delta, dense['effective_target_delta'][t, e])
    c.exact('backlog', target - q, s['snapshot_backlog'])
    c.exact('backlog_dense', dense['pre_target_debt'][t, e], s['snapshot_backlog'])
    c.exact('qd', dense['pre_qd_compact'][t, e], s['snapshot_qd'])
    history = dense['pre_pending_project_history'][t, :, e] - q[None]
    c.exact('pending_history', history, s['snapshot_past_backlogs'])
    c.exact('FIFO_actual_queue', dense['pre_pending_project_history'][t, :, e], dense['pre_pending_actuator_targets'][t, :, e])
    c.check('six_pending_entries', history.shape == (6, 26))
    c.close('d_eff', effective_distance(s, cfg), s['snapshot_d_eff'])
    c.close('cap', cfg['gamma'] * (s['snapshot_d_eff'].astype(float) - s['snapshot_dmin'].astype(float)) * .016666, s['snapshot_cap'])
    gate = s['snapshot_valid'].copy()
    if cfg['engage_dist'] is not None:
        gate &= (s['snapshot_d_eff'] < np.float32(cfg['engage_dist'])) | (s['snapshot_cap'] < 0)
    if cfg['exempt_structural_rows'] and bool(s['snapshot_struct_exempt_available']):
        structural = s['snapshot_struct_exempt'].copy()
        if cfg['retain_conditional_rows'] and bool(s['snapshot_contact_exempt_available']):
            structural &= ~s['snapshot_contact_exempt']
        drop = structural & (s['snapshot_cap'] >= 0)
        if cfg['struct_engage_dist'] is not None:
            drop &= s['snapshot_d'] >= np.float32(cfg['struct_engage_dist'])
        gate &= ~drop
    c.exact('strict_gate', gate, s['snapshot_row_gate'])
    lo, hi = lower.astype(float), upper.astype(float)
    A, b, labels = [], [], []
    for robot in ('F', 'U'):
        sl = SLICES[robot]
        arm_sl = slice(0, 2) if robot == 'F' else slice(2, 4)
        G = -s['J_' + robot].astype(float)
        c.exact('J_duplicate_' + robot, s['J_' + robot], s['snapshot_J_' + robot])
        c.exact('G_' + robot, G, s['snapshot_G_' + robot])
        rel = gate & s['snapshot_arm_mask'][:, arm_sl].any(-1)
        c.exact('rel_' + robot, rel, s['snapshot_rel_' + robot])
        min_gu = np.sum(np.where(G >= 0, G * lo[sl], G * hi[sl]), -1)
        c.close('min_Gu_' + robot, min_gu, s['snapshot_min_Gu_' + robot])
        cap = s['snapshot_cap'].astype(float)
        budget = (1 + float(s['p'])) / 2 if robot == 'F' else (1 - float(s['p'])) / 2
        raw_h = np.where(s['snapshot_cls'] == 0, np.where(cap >= 0, budget * cap, cap), cap)
        c.close('raw_h_' + robot, raw_h, s['snapshot_raw_h_' + robot])
        c.exact('debit_zero_' + robot, np.zeros(m), s['snapshot_backlog_debit_' + robot])
        c.exact('h_before_' + robot, s['snapshot_raw_h_' + robot], s['snapshot_h_before_authority_' + robot])
        c.close('h_after_' + robot, np.maximum(raw_h, .9 * min_gu), s['snapshot_h_after_authority_' + robot])
        # LP uses captured binary32 coefficients, after reconstructing their recipe.
        for pos in np.flatnonzero(rel):
            row = np.zeros(26)
            row[sl] = s['snapshot_G_' + robot][pos]
            A.append(row)
            b.append(float(s['snapshot_h_after_authority_' + robot][pos]))
            labels.append(dict(kind='safety', robot=robot, position=int(pos), row_id=int(ids[pos]),
                               physical_class=int(s['snapshot_cls'][pos]), gate=True,
                               structural_mask=bool(s['snapshot_struct_exempt'][pos]),
                               contact_mask=bool(s['snapshot_contact_exempt'][pos])))
        limited = np.clip(s['project_input_cmd'][sl], -np.float32(box), np.float32(box)).astype(float)
        ag = np.zeros_like(s['snapshot_alpha_G_' + robot], dtype=float)
        ah, ar = np.zeros(2), np.zeros(2, bool)
        robot_start = 0 if robot == 'F' else 2
        start = 0
        for local, (a, z) in enumerate(ARMS[robot_start:robot_start + 2]):
            width = z - a
            norm = np.linalg.norm(limited[start:start + width])
            ag[local, start:start + width] = limited[start:start + width] / max(norm, 1e-9)
            ah[local] = float(s['alpha'][robot_start + local]) * norm
            ar[local] = norm > 1e-9 and s['alpha'][robot_start + local] < np.float32(1 - 1e-6)
            start += width
        c.close('alpha_G_' + robot, ag, s['snapshot_alpha_G_' + robot])
        c.close('alpha_h_' + robot, ah, s['snapshot_alpha_h_' + robot])
        c.exact('alpha_rel_' + robot, ar, s['snapshot_alpha_rel_' + robot])
        for local in np.flatnonzero(ar):
            row = np.zeros(26)
            row[sl] = s['snapshot_alpha_G_' + robot][local]
            A.append(row)
            b.append(float(s['snapshot_alpha_h_' + robot][local]))
            labels.append(dict(kind='alpha', robot=robot, arm=robot_start + int(local)))
        mask = s['bypass_arm'][arm_sl]
        for local, arm in enumerate(ARMS[robot_start:robot_start + 2]):
            if mask[local]:
                c.exact('R19_limited_passthrough_' + str(robot_start + local), s['returned_cmd'][slice(*arm)], s['project_input_cmd'][slice(*arm)])
    A = np.asarray(A, dtype=float).reshape(-1, 26)
    b = np.asarray(b, dtype=float)
    returned = s['returned_cmd'].astype(float)
    rret = residuals(A, b, labels, lo, hi, returned)
    rtarget = residuals(A, b, labels, lo, hi, delta.astype(float))
    for robot in ('F', 'U'):
        inds = [i for i, x in enumerate(labels) if x['robot'] == robot]
        lrobot = [labels[i] for i in inds]
        # Full-width residuals; bounds restricted to this robot for stored comparison.
        for phase, u in [('returned', returned), ('target', delta.astype(float))]:
            r = residuals(A[inds], b[inds], lrobot, lo, hi, u)
            r['bounds'] = positive_max(np.r_[lo[SLICES[robot]] - u[SLICES[robot]], u[SLICES[robot]] - hi[SLICES[robot]]])
            for kind, name in [('safety', 'safety'), ('alpha', 'alpha'), ('bounds', 'bound')]:
                actual = s[f'returned_{name}_residual_{robot}'] if phase == 'returned' else diagnostics[f'target_{name}_residual_{robot}'][t, e]
                c.close(f'{phase}_{name}_{robot}', r[kind], actual)
            if phase == 'returned' and not s['bypass_arm'][slice(0, 2) if robot == 'F' else slice(2, 4)].any():
                c.close('original_no_bypass_' + robot, r['safety'], s['original_safety_residual_' + robot])
    detail = dict(check_count=c.count, failures=c.failures, max_arithmetic_error=max(c.errors.values(), default=0),
                  arithmetic_errors=c.errors, returned_residual=rret, actual_target_residual=rtarget,
                  original_reported_safety={r: float(s['original_safety_residual_' + r]) for r in ('F', 'U')},
                  batch_global_passes={r: int(s['original_passes_' + r]) for r in ('F', 'U')},
                  bypass_arms=s['bypass_arm'].tolist(), singleton_coordinates=int((lower == upper).sum()),
                  unreachable_coordinates=int(((q + np.float32(.05) < target + base_lo) | (q - np.float32(.05) > target + base_hi)).sum()))
    return A, b, labels, lo, hi, returned, detail


def feasibility(A, b, lo, hi, labels):
    """LP plus primal witness or phase-I primal/dual certificate, fixed box."""
    n = len(lo)
    assert A.shape == (len(b), n) and (lo <= hi).all()
    assert all(np.isfinite(x).all() for x in (A, b, lo, hi))
    norms = np.linalg.norm(A, axis=1)
    scales = np.where(norms > 0, norms, 1)
    an, bn = A / scales[:, None], b / scales
    opts = dict(primal_feasibility_tolerance=LP_TOL, dual_feasibility_tolerance=LP_TOL)
    result = linprog(np.zeros(n), A_ub=an if len(b) else None, b_ub=bn if len(b) else None,
                     bounds=list(zip(lo, hi)), method='highs', options=opts)
    min_gu = np.sum(np.where(A >= 0, A * lo, A * hi), -1)
    gaps = min_gu - b
    worst = int(np.argmax(gaps)) if len(gaps) else None
    info = dict(highs_status=int(result.status), highs_message=result.message,
                individually_infeasible_positive_rows=int((gaps > 0).sum()),
                individual_gap_max=float(gaps.max(initial=0)),
                normalized_individual_gap_max=float((gaps / scales).max(initial=0)),
                individual_worst_row=labels[worst] if worst is not None else None)
    if result.status == 0:
        raw_error = positive_max(np.r_[A @ result.x - b, lo - result.x, result.x - hi])
        norm_error = positive_max(np.r_[an @ result.x - bn, lo - result.x, result.x - hi])
        info.update(status='FEASIBLE' if norm_error <= WITNESS_TOL else 'NUMERICAL_UNRESOLVED',
                    witness=result.x.tolist(), witness_raw_residual=raw_error, witness_normalized_residual=norm_error)
        return info
    if result.status != 2:
        info['status'] = 'NUMERICAL_UNRESOLVED'
        return info
    # Slack is in row-normalized displacement units, not a changed physics gate.
    objective = np.r_[np.zeros(n), 1.0]
    phase = linprog(objective, A_ub=np.c_[an, -np.ones(len(b))], b_ub=bn,
                    bounds=list(zip(lo, hi)) + [(0, None)], method='highs', options=opts)
    info['phase1_status'] = int(phase.status)
    if phase.status != 0:
        info['status'] = 'NUMERICAL_UNRESOLVED'
        return info
    y = phase.ineqlin.marginals
    l, u = phase.lower.marginals[:n], phase.upper.marginals[:n]
    dual = float(bn @ y + lo @ l + hi @ u)
    stationarity = positive_max(np.abs(an.T @ y + l + u))
    gap = abs(float(phase.fun) - dual)
    phase_error = positive_max(np.r_[an @ phase.x[:n] - phase.x[n] - bn, lo - phase.x[:n], phase.x[:n] - hi])
    signed_ok = (y <= WITNESS_TOL).all() and (l >= -WITNESS_TOL).all() and (u <= WITNESS_TOL).all()
    robust = dual > WITNESS_TOL and stationarity <= WITNESS_TOL and gap <= WITNESS_TOL and phase_error <= WITNESS_TOL and signed_ok
    active = [dict(label=labels[i], dual=float(y[i]), normalized_min_gu_gap=float(gaps[i] / scales[i]),
                   normalized_phase_slack=float(an[i] @ phase.x[:n] - bn[i])) for i in np.flatnonzero(np.abs(y) > 1e-10)]
    info.update(status='INFEASIBLE_CERTIFIED' if robust else 'NUMERICAL_UNRESOLVED',
                phase1_minimum_normalized_slack=float(phase.fun), phase1_witness=phase.x[:n].tolist(),
                phase1_primal_error=phase_error, dual_lower_bound=dual, duality_gap=gap,
                dual_stationarity_error=stationarity, certificate_rows=active,
                lower_bound_dual=l.tolist(), upper_bound_dual=u.tolist())
    return info


def self_test():
    tests = []

    def solve(name, A, b, bounds, kinds, expected):
        A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
        lo, hi = np.asarray(bounds, dtype=float).T
        labels = [dict(kind=k, fixture=i) for i, k in enumerate(kinds)]
        r = feasibility(A, b, lo, hi, labels)
        assert r['status'] == expected, (name, r)
        tests.append(dict(name=name, status='PASS', lp_status=r['status'], individual_gap=r['individual_gap_max']))
        return r

    solve('known_feasible_interval', [[1]], [.01], [[-.025, .025]], ['safety'], 'FEASIBLE')
    x = solve('individually_feasible_opposing_safety_rows', [[-1], [1]], [-.02, -.02], [[-.025, .025]], ['safety', 'safety'], 'INFEASIBLE_CERTIFIED')
    assert x['individual_gap_max'] == 0 and abs(x['phase1_minimum_normalized_slack'] - .02) < 1e-10
    x = solve('safety_alpha_conflict', [[-1], [1]], [-.02, .01], [[-.025, .025]], ['safety', 'alpha'], 'INFEASIBLE_CERTIFIED')
    assert x['individual_gap_max'] == 0 and abs(x['phase1_minimum_normalized_slack'] - .005) < 1e-10
    solve('forced_singleton_violates_safety', [[1]], [.01], [[.02, .02]], ['safety'], 'INFEASIBLE_CERTIFIED')
    r = solve('positive_returned_residual_does_not_imply_empty_set', [[1]], [.01], [[-.025, .025]], ['safety'], 'FEASIBLE')
    assert .015 - .01 > 0 and r['witness_raw_residual'] == 0
    return dict(status='PASS', count=len(tests), tests=tests, scope='analytic synthetic CPU fixtures; not physical snapshots')


def evaluate():
    receipt = read_json(OUT)
    assert receipt['choice_sha256'] == digest_json(receipt['choices']) and len(receipt['choices']) == 18
    assert receipt['oracle_source_sha256'] == sha(__file__), 'source must match pre-LP frozen oracle'
    for path, expected in receipt['input_sha256'].items():
        assert sha(path) == expected, f'frozen input changed: {path}'
    p, meta = read_json(CELL / 'protocol.json'), read_json(CELL / 'guard_metadata.json')
    cfg = p['effective_backstop']
    assert p['dt'] == .016666 and not cfg['backlog_aware'] and cfg['predict_backlog'] and cfg['row_authority_clamp']
    assert cfg['pending_target_steps'] == meta['strict_fifo_steps'] == 6 and not meta['queue_preemption']
    for path, expected in meta['sources'].items():
        assert receipt['input_sha256'][path] == expected
    backstop = '/home/liyufeng/safeduo/src/safeduo/safety/backstop.py'
    assert receipt['input_sha256'][backstop] == p['source_sha256']['src/safeduo/safety/backstop.py']
    dense_keys = ['q_initial', 'q', 'cmd', 'exec', 'controller_target', 'joint_soft_limits', 'effective_target_delta',
                  'pre_target_debt', 'pre_qd_compact', 'pre_pending_project_history', 'pre_pending_actuator_targets']
    d = read_npz(CELL / 'cell_001.npz', dense_keys, receipt['input_sha256'][str(CELL / 'cell_001.npz')])
    dg = read_npz(CELL / 'project_diagnostics.npz', [f'target_{kind}_residual_{r}' for r in ('F', 'U') for kind in ('safety', 'alpha', 'bound')],
                  receipt['input_sha256'][str(CELL / 'project_diagnostics.npz')])
    rows, failures = [], []
    cached = {}
    for choice in receipt['choices']:
        t, e = choice['step'], choice['env']
        if t not in cached:
            path = CELL / 'projection_snapshots' / f'step_{t:04d}.npz'
            cached[t] = read_npz(path, expected=receipt['input_sha256'][str(path)])
        s = {k: v[e] for k, v in cached[t].items()}
        A, b, labels, lo, hi, returned, detail = reconstruct(s, d, dg, cfg, t, e)
        result = dict(choice=choice, reconstruction=detail, rows=labels, lower=lo.tolist(), upper=hi.tolist(),
                      returned_command=returned.tolist())
        if detail['failures']:
            failures.append(dict(step=t, env=e, checks=detail['failures']))
            result['classification'] = 'RECONSTRUCTION_BLOCKED_NO_LP'
            rows.append(result)
            continue
        full = feasibility(A, b, lo, hi, labels)
        safety_i = [i for i, x in enumerate(labels) if x['kind'] == 'safety']
        safety = feasibility(A[safety_i], b[safety_i], lo, hi, [labels[i] for i in safety_i])
        result.update(full_constraints=full, safety_and_bounds_only=safety)
        retmax = max(detail['returned_residual'].values())
        if full['status'] == 'FEASIBLE':
            result['classification'] = ('FEASIBLE_SET_RETURNED_VIOLATION_GT_ORIGINAL_TOL' if retmax > cfg['tol']
                                        else 'FEASIBLE_SET_POSITIVE_WITHIN_ORIGINAL_TOL' if retmax > 0
                                        else 'FEASIBLE_SET_ZERO_RETURNED_RESIDUAL')
            result['returned_failure_attribution'] = ('original solver exit; no robot R19 bypass' if not any(detail['bypass_arms'])
                                                      else 'returned command includes R19 override; pre-override solver vector not saved')
        elif full['status'] == 'INFEASIBLE_CERTIFIED':
            if full['normalized_individual_gap_max'] > WITNESS_TOL:
                result['classification'] = 'INFEASIBLE_INDIVIDUAL_ROW_VS_BOUNDS'
            else:
                result['classification'] = ('JOINT_INFEASIBLE_ALPHA_COMPOSITION' if safety['status'] == 'FEASIBLE'
                                            else 'JOINT_INFEASIBLE_SAFETY_COMPOSITION')
        else:
            result['classification'] = 'NUMERICAL_UNRESOLVED'
            failures.append(dict(step=t, env=e, checks=['LP status/certificate unresolved']))
        rows.append(result)
    checks = self_test()
    for path, expected in receipt['input_sha256'].items():
        assert sha(path) == expected, f'input changed during evaluation: {path}'
    assert sha(__file__) == receipt['oracle_source_sha256']
    receipt.update(status='PASS_BOUNDED_JOINT_FEASIBILITY_ORACLE' if not failures else 'BLOCKED_ORACLE',
                   evaluated_utc=now(), scipy_version=scipy.__version__, numpy_version=np.__version__,
                   lp_method='scipy.optimize.linprog(method=highs), equivalent L2 row normalization',
                   arithmetic_compare_abs=ARITH, highs_primal_dual_tol=LP_TOL, witness_certificate_tol=WITNESS_TOL,
                   original_projection_tol=cfg['tol'], self_tests=checks, evaluated_env_snapshots=len(rows),
                   input_sha_readback_unchanged=True, outcomes=rows, blocking_findings=failures,
                   classifications=dict(Counter(x['classification'] for x in rows)),
                   safety_strategy_approved=False, candidate_enabled=False,
                   scope='Selected linear sets only, fixed original p and authority-adjusted h; no PD/physics replay, no failed-step J reconstruction.')
    save(receipt)
    print(json.dumps({k: receipt[k] for k in ('status', 'choice_sha256', 'evaluated_env_snapshots', 'classifications', 'blocking_findings')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--freeze', action='store_true')
    action.add_argument('--evaluate', action='store_true')
    action.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.evaluate:
        evaluate()
    else:
        print(json.dumps(self_test()))
