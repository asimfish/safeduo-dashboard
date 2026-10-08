"""Finite CPU decision, independent of analyse_acceptance/ACCEPTANCE_RESULT.

CLI: python -B astra_acceptance_decision.py REGISTRATION_DIR NEW_RECEIPT
Audit subprocess waits are discovered in REGISTRATION_DIR/*_execution.json.
Each must be a run_step.py-style receipt for the exact auditor/root/output.
No child, simulator, renderer, parent analysis, or network is launched here.
Exit 0 = complete decision (including scientific REJECTED); 1 = invalid
evidence; 2 = incomplete. The output is exclusive-create; never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

CRITERIA_SHA = '01a5b45d1ada7635946f15c1b2e0cee141e8128266bdbed423844cf833c054a3'
PAIRING_SHA = '1247213588c332c32af34d736d7cf3b77ae36a873bb26d384937332a91bbbea7'
COMMON_CONTEXT_SHA = '0984ed6d3f131876f27e4182dcd72910fc3bfcff4d14b8c99839512a494e263c'
OPENING_SOURCE_SHA = '140dbb17a7313a0dc09a946cbc715c619e37cd501207f36c0362ac1539739ee6'
INITIAL_STATE_REG_SHA = 'e7649098e7b5a054654b22f63208e7dfadf64273cbb799e31eaa689037c91bc9'
MODES = ('joint_reference', 'zero_inclusive', 'box_admission', 'adaptive_joint')
ARMS = ('F_L', 'F_R', 'U_L', 'U_R')
WIDTHS = (7, 7, 6, 6)
LABELS = (-1, 0, 1, 2, 3, 4, 5)
FIELDS = ('native_q', 'native_qd', 'native_root_xyzw', 'native_root_velocity',
          'issued_controlled_target', 'applied_controlled_target',
          'pending_controlled_targets', 'pending_project_targets')
N, T, ROWS = 64, 960, 9021
ARM_MOTION_THRESHOLD = .0005  # frozen audit_geometry.py measurement definition


class InvalidEvidence(ValueError):
    pass


class IncompleteEvidence(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidEvidence(message)


def complete(condition, message):
    if not condition:
        raise IncompleteEvidence(message)


def exact(a, b, message, dtype=False):
    a, b = np.asarray(a), np.asarray(b)
    require(a.shape == b.shape and (not dtype or a.dtype == b.dtype)
            and np.array_equal(a, b), message)


def finite(a, shape, message):
    a = np.asarray(a)
    require(a.shape == shape and np.issubdtype(a.dtype, np.number)
            and np.isfinite(a).all(), message)
    return a


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def instant(value):
    d = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(d.tzinfo is not None, 'timestamp timezone required')
    return d


class Ledger:
    """Hash consumed inputs before/after reads and again before a decision."""
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        complete(path.is_file(), 'missing evidence: ' + str(path))
        value = sha(path)
        if expected is not None:
            require(value == expected, 'SHA mismatch: ' + str(path))
        if str(path) in self.hashes:
            require(value == self.hashes[str(path)], 'input changed: ' + str(path))
        self.hashes[str(path)] = value
        return value

    def json(self, path, expected=None):
        self.bind(path, expected)
        value = json.loads(Path(path).read_text())
        self.bind(path)
        return value

    def npz(self, path, keys=None, expected=None):
        self.bind(path, expected)
        with np.load(path, allow_pickle=False) as z:
            value = {k: z[k].copy() for k in (z.files if keys is None else keys)}
        self.bind(path)
        return value

    def verify(self):
        for path, digest in list(self.hashes.items()):
            self.bind(path, digest)


def confined(root, relative):
    p = Path(relative)
    require(not p.is_absolute(), 'raw reference must be relative')
    target = (Path(root) / p).resolve()
    require(target.is_relative_to(Path(root).resolve()), 'raw reference escapes root')
    return target


def check_wait(receipt):
    complete('actual_exit' in receipt and 'closed_utc' in receipt,
             'no actual child wait/closure')
    require(type(receipt['actual_exit']) is int and receipt['actual_exit'] == 0,
            'actual child exit must be integer zero')
    require(type(receipt.get('pid')) is int and receipt['pid'] > 0,
            'actual child pid missing')
    require(instant(receipt['started_utc']) <= instant(receipt['closed_utc']),
            'child timestamps reversed')


def index_jobs(jobs):
    wanted = {f'b{b}_{m}' for b in range(2) for m in MODES}
    complete(len(jobs) >= 8, 'missing registered/executed window batch')
    require(len(jobs) == 8 and {j['id'] for j in jobs} == wanted,
            'exactly eight unique bank/mode jobs required')
    return {j['id']: j for j in jobs}


def check_execution(reg, execution, reg_sha):
    complete(execution.get('status') == 'complete', 'policy execution not complete')
    require(execution['plan_sha256'] == reg_sha and execution['tag'] == reg['tag'],
            'execution is not this frozen registration')
    planned, ran = index_jobs(reg['jobs']), index_jobs(execution['jobs'])
    require(len({str(Path(j['out']).resolve()) for j in planned.values()}) == 8,
            'reused raw root across bank/mode jobs')
    registered = instant(reg['registered_utc'])
    require(registered <= instant(execution['started_utc']) <= instant(execution['closed_utc']),
            'registration must precede policy execution')
    for name, job in planned.items():
        r = ran[name]
        complete(r.get('status') == 'complete', 'unfinished child: ' + name)
        check_wait(r)
        require(registered <= instant(r['started_utc']) <= instant(r['closed_utc'])
                <= instant(execution['closed_utc']), 'job temporal binding: ' + name)
        for key in ('argv', 'env', 'out'):
            require(r[key] == job[key], 'actual job differs from registration: ' + key)
        require(job['kind'] == 'physics' and job['steps'] == T
                and job['scheduled_groups'] == 21, 'full physics/camera denominator')
    return planned, ran


def check_audit_wait(home, ledger, script, root, output, closed, absolute_paths=False):
    """Only exact, direct run_step argv matches; shell command strings do not."""
    expected_output = str(output) if script == 'astra_native_audit.py' else output.stem
    candidates = []
    for path in sorted(home.glob('*_execution.json')):
        # Unrelated running receipts are not inputs and may keep changing.
        try:
            r = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        argv = r.get('argv', [])
        if not isinstance(argv, list):
            continue
        for i, arg in enumerate(argv[:-2]):
            if absolute_paths and not (Path(str(arg)).is_absolute() and Path(str(argv[i+1])).is_absolute()):
                continue
            if Path(str(arg)).resolve() == (home / script).resolve():
                if (Path(str(argv[i+1])).resolve() == root.resolve()
                        and str(argv[i+2]) == expected_output):
                    candidates.append(path)
    complete(bool(candidates), 'no actual audit wait: ' + str(output))
    require(len(candidates) == 1, 'ambiguous/repeated audit execution: ' + str(output))
    path = candidates[0]
    r = ledger.json(path)
    check_wait(r)
    require(instant(closed) <= instant(r['started_utc']), 'audit predates closed physics')
    ledger.bind(home / (r['tag'] + '.log'), r['log_sha256'])


def check_peer_manifest_coverage(root, ledger, raw):
    for manifest in ('forecast_receipts.json', 'native_receipts.json', 'native_contact_receipts.json'):
        for ch in ledger.json(root/manifest)['chunks']:
            require(raw.get(ch['path']) == ch['sha256'], 'peer/producer chunk SHA mismatch')
    cameras = ledger.json(root/'camera_receipts.json')
    for capture in cameras['receipts']:
        require(raw.get(capture['state']) == capture['sha256'], 'peer missing camera state SHA')
        state = ledger.json(confined(root, capture['state']), capture['sha256'])
        require(state['images'] == capture['images'], 'camera image manifest agreement')
        for im in state['images']:
            require(raw.get(im['path']) == im['sha256'], 'peer missing camera PNG SHA')
        for field, digest in (('fresh_native_snapshot', 'fresh_native_before_sha256'),
                              ('fresh_native_after_snapshot', 'fresh_native_after_sha256')):
            require(raw.get(state[field]) == state[digest], 'peer missing render native snapshot SHA')


def check_camera_registration(camera, policy_reg, sources, home):
    # This registration is frozen by the future policy manifest, not a hardcoded
    # SHA. Its measurement contract and its own source references are checked.
    require(camera['schema'] == 'safeduo.same_run.camera_coverage.v1'
            and camera['principal_views'] == 9 and camera['hand_views_per_arm'] == 3
            and camera['arms'] == list(ARMS) and camera['total_views_per_group'] == 21
            and camera['scheduled_steps'] == [75, 480, 959]
            and camera['scheduled_slots'] == [0, 8, 16, 24, 32, 40, 56]
            and camera['seven_risk_strata'] is True
            and camera['exact_hand_sensor_bijection'] == 3328
            and camera['physical_world_coordinates'] is True
            and camera['original_score_epsilon_changed'] is False,
            'registered nine principal plus twelve native hand views')
    require(camera['hand_framing'] == dict(link_center_sphere_radius_m=.06,
            minimum_actual_USD_six_plane_sphere_clearance_m=.05), 'registered hand framing contract')
    require(camera['event_groups'] == ['first_original_sphere_violation_per_stratum',
            'first_raw_native_hand_normal_above_0.1N_per_stratum'], 'registered event image coverage')
    require(instant(camera['registered_utc']) <= instant(policy_reg['registered_utc']),
            'camera registration must precede fresh policy registration')
    require(camera['closed_cell_auditor'] == 'audit_hand_views.py'
            and sources.get(str(home/'audit_hand_views.py')) == camera['closed_cell_auditor_sha256']
            and sources.get(str(home/'hand_views.py')) == camera['producer_sha256'],
            'frozen camera auditor and hand producer SHA')
    for key in ('full_mesh_certified', 'occlusion_certified', 'hardware_certified'):
        require(camera[key] is False, 'camera registration scope: '+key)


def hand_camera_binding(root, ledger, receipt, principal, sources, home):
    """Bind delegated hand pixel/frustum/native oracle to this exact closed cell.

    No image decoding, frustum calculation or native-state oracle is implemented
    here. The source-frozen audit_hand_views.py proves those properties; this
    function closes its receipt, counts, identities and every raw reference.
    """
    complete(receipt.get('status') == 'PASS_CLOSED_CELL_NATIVE_HAND_CAMERA_BINDING',
             'closed hand-camera auditor not complete')
    require(Path(receipt['root']).resolve() == root.resolve()
            and receipt['frames'] == T and receipt['windows'] == N, 'same-job full hand-camera denominator')
    require(receipt['source_sha256'] == sources[str(home/'audit_hand_views.py')], 'hand auditor source SHA')
    require(receipt['cell_sha256'] == principal['cell_sha256'] == ledger.bind(root/'cell_001.npz'),
            'hand camera vs principal same closed cell SHA')
    groups = receipt['groups']
    require(type(groups) is int and groups >= 21 and groups == principal['groups']
            and receipt['hand_PNG'] == 12*groups and receipt['total_PNG'] == 21*groups
            and principal['PNG'] == 9*groups, 'hand/principal/total image counts')
    require(receipt['all64_native_bitwise_render_invariant'] is True
            and receipt['exact_contact_sensor_bijection'] == 3328, 'delegated hand native/mapping proof')
    for key in ('full_mesh_certified', 'occlusion_certified', 'hardware_certified'):
        require(receipt[key] is False, 'hand auditor scope: '+key)
    hands = ledger.json(root/'hand_camera_receipts.json')
    overview = ledger.json(root/'camera_receipts.json')
    require(hands['groups'] == overview['groups'] == groups
            and hands['PNG'] == 12*groups and overview['PNG'] == 9*groups, 'raw hand image denominator')

    def key(c):
        require(type(c['step']) is int and 0 <= c['step'] < T and type(c['env_id']) is int
                and 0 <= c['env_id'] < N and c['capture_kind'] in ('scheduled', 'first_failure', 'first_native_contact'),
                'hand/principal capture identity range')
        return c['step'], c['env_id'], c['capture_kind']

    top = {key(c): c for c in hands['receipts']}
    macro = {key(c): c for c in overview['receipts']}
    require(len(top) == len(hands['receipts']) == len(macro) == len(overview['receipts']) == groups
            and set(top) == set(macro), 'all hand groups exactly match principal groups')
    group_entries, group_before, png_paths = {}, {}, set()

    def bind(ref):
        p = confined(root, ref['path'])
        ledger.bind(p, ref['sha256'])
        return p

    for identity, capture in top.items():
        sp = confined(root, capture['state'])
        state = ledger.json(sp, capture['sha256'])
        require(key(state) == identity and state['image_count'] == len(state['images']) == 12,
                'hand state identity and12 images')
        require(state['source_sha256'] == sources[str(home/'hand_views.py')], 'hand state producer source SHA')
        refs = [{k: im[k] for k in ('arm', 'view', 'path', 'sha256')} for im in state['images']]
        require(capture['images'] == refs and len({(im['arm'], im['view']) for im in refs}) == 12
                and {(im['arm'], im['view']) for im in refs}
                == {(arm, view) for arm in ARMS for view in ('oblique_above', 'opposite_below', 'cross_above')},
                'hand image manifest and four-arm three-view identity')
        parent = state['parent_macro_binding']
        require(capture['parent_macro_binding'] == parent and parent['path'] == macro[identity]['state']
                and parent['sha256'] == macro[identity]['sha256'], 'hand state vs same principal macro group')
        bind(parent)
        bind(state['native_before_all64'])
        mapping_ref = state['sensor_identity_mapping']
        mapping = ledger.json(bind(mapping_ref), mapping_ref['sha256'])
        require(mapping['source']['sha256'] == ledger.bind(root/'native_contact_identity.json'),
                'hand mapping vs same raw native contact identity')
        group_path = sp.parent.parent/'receipts.json'
        require(group_path.is_relative_to(root.resolve()), 'hand group manifest stays in root')
        group_entries.setdefault(group_path, []).append(capture)
        if group_path in group_before:
            require(group_before[group_path] == state['native_before_all64'], 'shared hand group before reference')
        group_before[group_path] = state['native_before_all64']
        for im in state['images']:
            require(im['native_before_all64'] == state['native_before_all64'], 'per-image hand before reference')
            png = bind(im)
            require(png not in png_paths, 'duplicate hand PNG across groups/views')
            png_paths.add(png)
            bind(im['native_after_all64'])
    for path, captures in group_entries.items():
        group = ledger.json(path)  # SHA recorded even though parent derives this path.
        require(group['visibility_restored'] is True and group['groups'] == len(captures)
                and group['PNG'] == 12*len(captures), 'complete hand group final manifest')
        require({key(c): c for c in group['receipts']} == {key(c): c for c in captures}
                and len(group['receipts']) == len(captures), 'hand group vs top-level receipts')
        require(group['native_before_all64'] == group_before[path], 'group native before reference')
        bind(group['native_before_all64'])
        bind(group['native_final_all64'])
    require(len(png_paths) == receipt['hand_PNG'], 'every hand PNG raw SHA bound')
    return dict(groups=groups, hand_PNG=12*groups, total_PNG=21*groups,
                source_sha256=receipt['source_sha256'], all_raw_references_bound=True)


def check_banks(home, ledger, criteria, sources, reg):
    planpath = home/'bank_qualification_plan.json'
    plan = ledger.json(planpath)
    execution = ledger.json(home/'bank_qualification_execution.json')
    complete(execution.get('status') == 'complete', 'fresh bank qualification not complete')
    check_wait(execution)
    require(execution['plan_sha256'] == ledger.bind(planpath), 'bank qualification plan SHA')
    require(plan['policy_outcomes_used'] is False and plan['cases_per_bank'] == N
            and plan['seeds'] == [r['initial_seed'] for r in criteria['initial_bank_rows']],
            'two preregistered policy-blind qualification banks')
    require(instant(plan['registered_utc']) <= instant(execution['started_utc'])
            <= instant(execution['child_closed_utc']) <= instant(execution['closed_utc'])
            <= instant(reg['registered_utc']), 'bank qualification before outcome registration')
    for key in ('argv', 'env', 'out'):
        require(execution[key] == plan[key], 'actual bank qualification binding: '+key)
    ledger.bind(home/'bank_qualification.log', execution['log_sha256'])
    for source, digest in plan['sources'].items():
        ledger.bind(source, digest)
    expected = set()
    for row in criteria['initial_bank_rows']:
        path = (Path(plan['out'])/str(row['initial_seed'])/'bank.npz').resolve()
        metadata = path.with_name('metadata.json')
        require(str(path) in sources and str(metadata) in sources, 'freeze bank and qualification metadata')
        m = ledger.json(metadata, sources[str(metadata)])
        require(m['status'] == 'complete' and m['selected_count'] == N and m['policy_outcomes_used'] is False,
                'qualified bank metadata')
        ledger.bind(path, m['bank_sha256'])
        expected.add(str(path))
    return expected


def argument(argv, flag):
    require(argv.count(flag) == 1, 'exactly one registered argument: '+flag)
    return argv[argv.index(flag)+1]


def check_initial_registration(initial_spec, policy_reg, sources, home):
    require(initial_spec['schema'] == 'safeduo.actual_initial_geometry.acceptance.v1'
            and initial_spec['new_policy_outcomes_observed'] is False
            and type(initial_spec['requires_candidate_actual_initial_full9021_nonexempt_negative_cases']) is int
            and initial_spec['requires_candidate_actual_initial_full9021_nonexempt_negative_cases'] == 0
            and initial_spec['all_new_cases_retained'] is True
            and initial_spec['original_full960_post_scoring_unchanged'] is True
            and initial_spec['initial_failure_is_extra_adoption_rejection'] is True
            and initial_spec['filter_or_resample_policy_outcomes'] is False,
            'registered actual initial gate; no post-score relaxation or case filtering')
    require(instant(initial_spec['registered_utc']) <= instant(policy_reg['registered_utc']),
            'initial gate must be registered before fresh policy registration')
    require(sources.get(str(home/'audit_geometry_v2.py')) == initial_spec['source_sha256'],
            'registered geometry v2 source SHA')


def opening_binding(opening, initial):
    """Requested common context and native joint identity, not a pose safety test."""
    require(opening['open_thumb_rad'] == .35 and opening['all_modes_common'] is True
            and opening['controlled_arm_targets_changed'] is False
            and opening['original_actor_FIF06_scores_exemptions_gains_self_collision_unchanged'] is True
            and opening['constructor_contacts_certified'] is False, 'registered common hand context receipt')
    wanted = {'U_L':'left_thumb_1_joint', 'U_R':'right_thumb_1_joint'}
    require(opening['requested_joint_names'] == wanted and set(opening['joint_indices']) == set(wanted),
            'exact registered opening joint names')
    indices, observed = {}, {}
    for arm, name in wanted.items():
        names = initial[arm+'_native_joint_names'].tolist()
        require(names.count(name) == 1, 'unique native thumb joint')
        idx = names.index(name)
        require(opening['joint_indices'][arm] == idx
                and idx not in initial[arm+'_controlled_joint_indices'], 'thumb context vs actual native index')
        indices[arm] = idx
        observed[arm] = dict(q=initial[arm+'_native_q'][:,idx].tolist(),
                             qd=initial[arm+'_native_qd'][:,idx].tolist())
    # Measured pose/velocity may respond to physics: preserve them, never force
    # them to .35/0. All eight full-native pairing fields remain exact gates.
    return indices, observed


def geometry_labels(identity):
    require(identity['rows'] == ROWS, 'full9021 row identity')
    classes = finite(identity['class_id'], (ROWS,), 'geometry classes')
    pairs = np.asarray(identity['pair_sphere_idx'])
    arms = np.asarray(identity['sphere_arm_id'])
    require(pairs.shape == (ROWS, 2) and np.issubdtype(pairs.dtype, np.integer), 'row pairs')
    require(set(np.unique(classes)) <= {0, 1, 2}, 'unknown geometry class')
    require(np.all((pairs >= 0) & (pairs < len(arms))), 'row sphere indices')
    exact(identity['pair_id'], np.arange(ROWS), 'row identity ordering')
    result = classes.astype(np.int64).copy()
    result[(classes == 1) & (arms[pairs[:, 0]] >= 2)] = 2
    result[classes == 2] = 3
    return result


def margins(d, exempt, labels):
    d = finite(d, (N, ROWS), 'all9021 native distances')
    require(d.dtype == np.float32 and exempt.shape == d.shape
            and exempt.dtype == np.bool_, 'native float32 geometry / boolean original exemptions')
    return np.stack([np.where(exempt | (labels[None] != k), np.float32(np.inf), d).min(-1)
                     for k in range(4)], axis=-1)


def score_and_motion(cell, computed):
    q = finite(cell['q'], (T, N, 26), '960x64x26 actual measured q')
    q0 = finite(cell['q_initial'], (N, 26), 'actual initial compact q')
    require(computed.shape == (T, N, 4) and computed.dtype == np.float32
            and not np.isnan(computed).any(), 'all measured float32 class margins')
    exact(computed, cell['official_margins'], 'raw geometry vs official margins', dtype=True)
    exact(computed < np.float32(-.005), cell['official_deep'], 'raw deep class flags')
    strict, deep = (computed < np.float32(0)).any(-1), (computed < np.float32(-.005)).any(-1)
    trajectory = np.concatenate([q0[None], q])
    increments = np.diff(trajectory.astype(np.float64), axis=0)
    path = np.linalg.norm(increments, axis=-1).sum(0)
    ranges = np.ptp(trajectory, axis=0)
    starts = np.cumsum((0,) + WIDTHS)
    moving = np.stack([np.linalg.norm(increments[..., starts[i]:starts[i+1]], axis=-1)
                       > ARM_MOTION_THRESHOLD for i in range(4)], -1).all(-1)
    return dict(strict=strict.any(0), deep=deep.any(0), strict_frames=int(strict.sum()),
                deep_frames=int(deep.sum()), first=[int(np.flatnonzero(strict[:, e])[0])
                if strict[:, e].any() else None for e in range(N)],
                minima=computed.min((0, 1)), path=path, ranges=ranges, four=float(moving.mean()))


def raw_geometry(root, ledger, cell):
    labels = geometry_labels(ledger.json(root / 'full_row_identity.json'))
    receipt = ledger.json(root / 'forecast_receipts.json')
    require((receipt['steps'], receipt['envs'], receipt['rows']) == (T, N, ROWS),
            'full raw geometry dimensions')
    computed = np.empty((T, N, 4), np.float32)
    seen = 0
    initial_margins = None
    for ch in receipt['chunks']:
        require(ch['start'] == seen and seen < ch['stop'] <= T, 'geometry chunk gap/overlap')
        data = ledger.npz(confined(root, ch['path']), ['measured_d', 'exempt'], ch['sha256'])
        require(data['measured_d'].shape == (ch['stop']-seen, N, ROWS), 'geometry chunk shape')
        require(data['exempt'].shape == (ch['stop']-seen, N, 1128)
                and data['exempt'].dtype == np.uint8, 'packed original exemption shape')
        for i, t in enumerate(range(seen, ch['stop'])):
            ex = np.unpackbits(data['exempt'][i], axis=-1, count=ROWS).astype(bool)
            m = margins(data['measured_d'][i], ex, labels)
            if t:
                computed[t-1] = m
            else:
                initial_margins = m.copy()
        seen = ch['stop']
    complete(seen == T, 'missing full geometry frames')
    final = ledger.npz(root / 'post_geometry_final.npz', ['d', 'exempt'])
    computed[-1] = margins(final['d'], final['exempt'], labels)
    result = score_and_motion(cell, computed)
    result.update(initial_strict_ids=np.flatnonzero((initial_margins < np.float32(0)).any(-1)).tolist(),
                  initial_deep_ids=np.flatnonzero((initial_margins < np.float32(-.005)).any(-1)).tolist(),
                  initial_minimum_by_class_m=initial_margins.min(0))
    return result


def normal_peaks(matrix, env_ids, partners):
    matrix = finite(matrix, (len(env_ids), partners, 3), 'raw allpartner normal matrix')
    require(set(env_ids.tolist()) == set(range(N)), 'all64 native hand sensor coverage')
    norms = np.linalg.norm(matrix.astype(np.float64), axis=-1)
    # Deliberately includes SAME-HAND partners. No exemption/filtering here.
    return np.array([norms[env_ids == e].max() for e in range(N)])


def raw_contacts(root, ledger, opening_indices=None):
    r = ledger.json(root / 'native_contact_receipts.json')
    complete(r.get('status') == 'complete', 'contact stream incomplete')
    require((r['control_steps'], r['physics_events'], r['substeps_per_control'], r['capacity'])
            == (T, 2*T, 2, 262144), 'all1920 contact events/capacity')
    identity = ledger.json(root / 'native_contact_identity.json', r['identity_sha256'])
    require(identity['environment_count'] == N and identity['physics_dt_s'] == .008333,
            'native contact environment count/dt')
    views = {v['arm']: v for v in identity['views']}
    require(set(views) == set(ARMS) and len(identity['views']) == 4, 'all four native hand views')
    metrics = np.empty((T, 2, N, 4), np.float64)
    seen = 0
    keys = ['frame', 'substep', 'physics_dt'] + [a + '_partner_normal' for a in ARMS]
    if opening_indices is not None:
        keys += [arm+'_native_position_targets' for arm in opening_indices]
    for ch in r['chunks']:
        require(ch['start'] == seen and seen < ch['stop'] <= 2*T, 'contact chunk gap/overlap')
        z = ledger.npz(confined(root, ch['path']), keys, ch['sha256'])
        require(all(len(v) == ch['stop']-seen for v in z.values()), 'contact chunk length')
        for j, event in enumerate(range(seen, ch['stop'])):
            t, sub = divmod(event, 2)
            require(int(z['frame'][j]) == t and int(z['substep'][j]) == sub
                    and float(z['physics_dt'][j]) == identity['physics_dt_s'], 'contact time/substep')
            for ai, arm in enumerate(ARMS):
                view = views[arm]
                envs = np.asarray(view['env_ids'])
                require(np.issubdtype(envs.dtype, np.integer), 'integer contact sensor environment IDs')
                metrics[t, sub, :, ai] = normal_peaks(z[arm+'_partner_normal'][j], envs,
                                                    len(view['filters'][0]))
            if opening_indices is not None:
                for arm, idx in opening_indices.items():
                    target = z[arm+'_native_position_targets'][j][:,idx]
                    require(target.dtype == np.float32, 'native opening target dtype')
                    exact(target, np.full(N, .35, np.float32), 'actual common thumb goal each physics substep', dtype=True)
        seen = ch['stop']
    complete(seen == 2*T, 'missing physics substeps')
    return metrics


def audit_agreement(root, cell_sha, scores, metrics, g, n, c, peer, peer_source_sha):
    statuses = ('PASS_ALL_FULL_RAW_FLOAT32_GEOMETRY_SCORING',
                'PASS_SAME_RUN_NATIVE_AND_NINE_VIEW_BINDING',
                'PASS_NATIVE_HAND_CONTACT_OBSERVATION_NOT_SAFETY',
                'PASS_RAW_BINDING_AND_ORIGINAL_SCORING_ONLY')
    for receipt, status in zip((g, n, c, peer), statuses):
        complete(receipt.get('status') == status, 'required raw auditor not complete: ' + status)
        require(Path(receipt['root']).resolve() == root.resolve(), 'auditor root mismatch')
        require(receipt['windows'] == N, 'auditor window denominator')
    require(g['cell_sha256'] == n['cell_sha256'] == peer['raw_sha256']['cell_001.npz'] == cell_sha,
            'auditors must bind identical cell bytes')
    require(peer['auditor_sha256'] == peer_source_sha and type(peer['exitcode']) is int
            and peer['exitcode'] == 0, 'peer frozen source and declared completion')
    require(g['frames'] == n['frames'] == peer['frames'] == T
            and c['control_steps'] == T and c['physics_events'] == 2*T,
            'auditor full window/frame/substep denominators')
    require(g['geometry_rows'] == peer['full_geometry_rows'] == ROWS
            and g['raw_env_frames'] == peer['raw_geometry_env_frames'] == T*N,
            'auditor geometry row/envframe denominators')
    require(n['boundary_packets'] == peer['native_boundary_packets'] == 3*T
            and n['native_controlled_q_qd_exact'] is True and n['actual_FIFO6_exact'] is True
            and n['all64_render_native_unchanged'] is True
            and n['original_full_geometry_score_exact'] is True, 'native3boundary binding')
    require(n['groups'] == peer['camera_groups'] and n['PNG'] == peer['PNG'] == n['groups']*9,
            'same nine-view image denominator')
    require(peer['queued_future_status'] == 'UNKNOWN' and peer['score_threshold_epsilon'] == 0
            and peer['physical_safety_certified'] is False, 'raw audit scope/epsilon')
    require(type(g['initial_negative_envs']) is int
            and g['initial_negative_envs'] == len(scores['initial_strict_ids']),
            'raw vs v2 auditor initial negative count')
    exact(g['initial_negative_ids'], scores['initial_strict_ids'], 'raw vs v2 auditor initial IDs')
    exact(g['initial_raw_minimum_by_class_m'], scores['initial_minimum_by_class_m'],
          'raw vs v2 auditor initial minima')
    for name in ('strict', 'deep'):
        ids = np.flatnonzero(scores[name]).tolist()
        require(g[name+'_windows'] == peer[name+'_windows'] == len(ids)
                and g[name+'_env_frames'] == peer[name+'_env_frames'] == scores[name+'_frames'],
                'raw/auditor geometry counts: ' + name)
        exact(g[name+'_ids'], ids, 'parent raw geometry IDs: ' + name)
        exact(peer[name+'_env_ids'], ids, 'independent raw geometry IDs: ' + name)
    require(g['first_failure_steps'] == scores['first'], 'first raw failure time')
    for value, actual, message in ((g['minimum_by_class_m'], scores['minima'], 'parent minima'),
            (peer['minimum_by_class_m'], scores['minima'], 'peer minima'),
            (g['q_l2_path_by_env'], scores['path'], 'path'),
            (g['joint_range_by_env'], scores['ranges'], '26joint ranges')):
        exact(value, actual, 'raw vs auditor ' + message)
    require(g['mean_q_l2_path'] == float(scores['path'].mean())
            and g['four_arms_moving_fraction'] == scores['four'], 'raw vs auditor movement')
    pc = peer['contacts']
    require(pc['status'] == 'PASS_SUBSTEP_BINDING_ONLY' and pc['physics_events'] == 2*T
            and pc['actual_PhysX_position_targets_verified'] is True, 'contact microstep/actuator binding')
    require(c['raw_normal_diagnostic_threshold_N'] == pc['raw_normal_diagnostic_threshold_N'] == .1
            and c['same_hand_contacts_included_in_raw'] is True
            and c['exemption_adjusted_contact_gate'] is False, 'raw same-hand-inclusive diagnostic scope')
    require(c['capacity'] == 262144 and c['peak_contact_count'] == pc['maximum_contact_count']
            and c['maximum_net_minus_filtered_normal_abs_N'] == pc['maximum_net_minus_filtered_normal_abs_N'],
            'contact accounting/capacity auditors disagree')
    peak = metrics.max((0, 1, 3))
    over = (metrics > .1).any((1, 3))
    exact(c['window_peak_normal_N'], peak, 'raw vs parent contact peaks')
    exact(pc['raw_normal_peak_by_env_arm_N'], metrics.max((0, 1)), 'raw vs peer contact peaks')
    require(c['windows_exceeding_normal_diagnostic'] == pc['raw_windows_over_threshold']
            == int(over.any(0).sum()) and pc['raw_env_frames_over_threshold'] == int(over.sum()),
            'raw/auditor contact exceedances disagree')


def pairing_difference(base, other, fields=FIELDS):
    result = []
    for arm in ARMS:
        for field in fields:
            key = arm + '_' + field
            a, b = np.asarray(base[key]), np.asarray(other[key])
            require(a.shape == b.shape and a.dtype == b.dtype and np.isfinite(a).all()
                    and np.isfinite(b).all(), 'full native pairing shape/dtype/finite: ' + key)
            env_axis = 1 if field.startswith('pending_') else 0
            require(a.shape[env_axis] == N, 'native initial environment axis')
            if not np.array_equal(a, b):
                delta = np.abs(a.astype(np.float64)-b.astype(np.float64))
                moved = np.moveaxis(a != b, env_axis, 0).reshape(N, -1)
                envs = np.flatnonzero(moved.any(-1))
                per_env = np.moveaxis(delta, env_axis, 0).reshape(N, -1)
                result.append(dict(field=key, env_ids=envs.tolist(),
                    max_abs_difference=float(delta.max()), different_elements=int((a != b).sum()),
                    max_difference_by_env=[float(per_env[e].max()) for e in envs]))
    return result


def decide(records, criteria, pairing, initial_spec):
    """Pure finite decision over independently verified raw-derived records."""
    complete(len(records) >= 8, 'missing method windows')
    require(len(records) == 8 and {(r['block'], r['mode']) for r in records}
            == {(b, m) for b in range(2) for m in MODES}, 'eight distinct bank/mode records')
    for r in records:
        require((r['windows'], r['frames'], r['physics_events']) == (N, T, 2*T),
                'closed512/128/960/1920 denominators')
        complete(r.get('hand_camera_bound') is True, 'missing complete same-cell hand-camera binding')
        labels = np.asarray(r['labels'])
        require(labels.shape == (N,) and all(int((labels == l).sum()) == (16 if l == -1 else 8)
                for l in LABELS), 'all seven registered risk strata')
        for key, shape in (('path', (N,)), ('ranges', (N, 26)), ('peak', (N,))):
            x = finite(r[key], shape, 'finite decision inputs: ' + key)
            require((x >= 0).all(), 'nonnegative decision metric: ' + key)
        require(np.isfinite(r['four']) and 0 <= r['four'] <= 1, 'finite four-arm fraction')
        for key in ('strict', 'deep'):
            require(np.asarray(r[key]).shape == (N,) and np.asarray(r[key]).dtype == np.bool_,
                    'raw boolean failure flags')
        require(np.all(~r['deep'] | r['strict']), 'deep implies strict')
        complete('initial_strict_ids' in r, 'missing actual pre-step0 geometry evidence')
        ids = r['initial_strict_ids']
        require(isinstance(ids, list) and all(type(e) is int and 0 <= e < N for e in ids)
                and ids == sorted(set(ids)), 'actual initial negative IDs must be unique/in-range/sorted')
    by = {m: sorted([r for r in records if r['mode'] == m], key=lambda x: x['block']) for m in MODES}
    for b in range(2):
        for mode in MODES:
            exact(by[mode][b]['labels'], by['zero_inclusive'][b]['labels'], 'paired risk assignment')
    agg = {}
    for mode in MODES:
        rs = by[mode]
        agg[mode] = dict(windows=128, strict_windows=sum(int(x['strict'].sum()) for x in rs),
            deep_windows=sum(int(x['deep'].sum()) for x in rs),
            initial_negative_envs=sum(len(x['initial_strict_ids']) for x in rs),
            raw_contact_over_windows=sum(int((x['peak'] > .1).sum()) for x in rs),
            raw_contact_max_N=float(max(x['peak'].max() for x in rs)),
            mean_q_path=float(np.concatenate([x['path'] for x in rs]).mean()),
            four_arm_fraction=float(np.mean([x['four'] for x in rs])),
            mean_joint_ranges=np.concatenate([x['ranges'] for x in rs]).astype(np.float64).mean(0).tolist())
    spec = criteria['represented_sphere_acceptance']
    base, candidate = agg['zero_inclusive'], agg['adaptive_joint']
    reasons = []
    if candidate['strict_windows'] != spec['candidate_strict_windows_required']:
        reasons.append('candidate_strict_windows')
    if candidate['deep_windows'] != spec['candidate_deep_windows_required']:
        reasons.append('candidate_deep_windows')
    new_fail = sum(int((c['strict'] & ~a['strict']).sum())
                   for a, c in zip(by['zero_inclusive'], by['adaptive_joint']))
    if new_fail > spec['new_paired_strict_failures_allowed']:
        reasons.append('new_paired_strict_failure')
    def ratio(a, b):
        return b/a if a > 0 else None
    path = ratio(base['mean_q_path'], candidate['mean_q_path'])
    four = ratio(base['four_arm_fraction'], candidate['four_arm_fraction'])
    if path is None or path < spec['mean_q_path_minimum_ratio_vs_zero']:
        reasons.append('overall_path_below_floor_or_undefined')
    if four is None or four < spec['four_arm_movement_minimum_ratio_vs_zero']:
        reasons.append('four_arm_motion_below_floor_or_undefined')
    strata = []
    for label in LABELS:
        vals = {m: np.concatenate([x['path'][x['labels'] == label] for x in by[m]]) for m in MODES}
        value = ratio(float(vals['zero_inclusive'].mean()), float(vals['adaptive_joint'].mean()))
        strata.append(dict(label=label, cases=len(vals['zero_inclusive']), path_ratio=value))
        if value is None or value < spec['mean_q_path_minimum_ratio_each_stratum_vs_zero']:
            reasons.append('stratum_path_below_floor_or_undefined:' + str(label))
    if min(candidate['mean_joint_ranges']) < spec['minimum_all26_joints_mean_measured_range_rad']:
        reasons.append('one_or_more_26joint_ranges_below_floor')
    require(len(pairing) == 8 and {(p['block'], p['mode']) for p in pairing}
            == {(b, m) for b in range(2) for m in MODES}, 'full native pairing coverage')
    if any(p['differences'] for p in pairing):
        reasons.append('actual_full_native_initial_mismatch')
    sphere_reasons = list(reasons)
    initial_bad = candidate['initial_negative_envs'] != initial_spec['requires_candidate_actual_initial_full9021_nonexempt_negative_cases']
    if initial_bad:
        reasons.append('candidate_actual_initial_geometry_negative')
    if candidate['raw_contact_over_windows'] > criteria['native_hand_qualification']['candidate_windows_exceeding_limit_allowed']:
        reasons.append('raw_same_hand_inclusive_contact_over_registered_diagnostic')
    return dict(status='COMPLETE_INDEPENDENT_FINITE_DECISION',
        candidate_decision='REJECTED' if reasons else 'PASS_BOUNDED_SIMULATION_ONLY',
        represented_sphere_decision='REJECTED' if sphere_reasons else 'PASS_BOUNDED_REPRESENTED_SPHERES',
        represented_sphere_decision_scope='original960post_frames_and_existing_motion_pairing_gates',
        initial_state_decision='REJECTED' if initial_bad else 'PASS_OBSERVED_INITIAL_REPRESENTED_GEOMETRY_ONLY',
        initial_negative_cases=[dict(block=r['block'], env=e) for r in by['adaptive_joint']
                                for e in r['initial_strict_ids']],
        rejection_reasons=reasons, aggregated=agg, new_paired_strict_failures=new_fail,
        actual_motion_path_ratio=path, actual_four_arm_movement_ratio=four, strata=strata,
        native_initial_pairing=pairing, method_windows=512, unique_paired_cases=128,
        independent_initial_banks=2, steps_per_window=T, physics_substeps_per_window=2*T,
        total_control_env_frames=512*T, total_physics_env_substeps=512*2*T)


def inspect_job(home, job, row, bank_row, ledger, sources):
    root = Path(job['out']).resolve()
    name = job['id']
    mode = job['env']['SAFEDUO_JOINT_MODE']
    paths = [home/(name+'_geometry.json'), home/(name+'_native.json'),
             home/(name+'_contacts.json'), home/('ASTRA_NATIVE_REAL_'+name+'.json')]
    audits = [ledger.json(p) for p in paths]
    g, n, c, peer = audits
    for script, path in zip(('audit_geometry_v2.py', 'audit_native.py', 'audit_contacts.py',
                             'astra_native_audit.py'), paths):
        check_audit_wait(home, ledger, script, root, path, row['closed_utc'])
    hand_path = home/(name+'_hands.json')
    hand_receipt = ledger.json(hand_path)
    check_audit_wait(home, ledger, 'audit_hand_views.py', root, hand_path, row['closed_utc'], absolute_paths=True)
    complete(peer.get('status') == 'PASS_RAW_BINDING_AND_ORIGINAL_SCORING_ONLY', 'peer not complete')
    raw = peer['raw_sha256']
    needed = {'cell_001.npz', 'visual_protocol.json', 'full_row_identity.json', 'forecast_receipts.json',
              'post_geometry_final.npz', 'native_receipts.json', 'camera_receipts.json',
              'guard_metadata.json', 'episodes.json', 'native_initial.npz', 'bank_assignment.npz',
              'native_contact_receipts.json', 'native_contact_identity.json'}
    require(needed <= set(raw), 'peer missing raw binding')
    for rel, digest in raw.items():
        ledger.bind(confined(root, rel), digest)
    check_peer_manifest_coverage(root, ledger, raw)
    protocol = ledger.json(root/'visual_protocol.json')
    complete(protocol.get('status') == 'complete', 'protocol incomplete despite child exit')
    require(protocol['steps'] == T and protocol['completed_windows'] == N
            and protocol['dt'] == .016666 and protocol['capture_steps'] == [75, 480, 959]
            and protocol['slots'] == [0, 8, 16, 24, 32, 40, 56], 'registered protocol denominator/cameras')
    require(protocol['args']['seeds'] == bank_row['command_seed'], 'registered raw command seed')
    argv = job['argv']
    require('-B' in argv and str(home/'native_full_view_runner.py') in argv, 'registered full-view producer and Python -B')
    require(Path(argument(argv, '--out')).resolve() == root, 'registered argv output vs job root')
    require(protocol['observer_sha256'] == sources[str(home/'native_full_view_runner.py')], 'actual observer source SHA')
    ckpt = str(Path(argument(argv, '--ckpt')).resolve())
    require(ckpt in sources and protocol['actor_sha256'] == sources[ckpt], 'actual frozen actor SHA')
    for flag in ('out', 'ckpt', 'env-yaml', 'methods', 'device'):
        require(str(protocol['args'][flag.replace('-', '_')]) == argument(argv, '--'+flag),
                'protocol/registered argument: '+flag)
    guard = ledger.json(root/'guard_metadata.json')
    require(guard['mode'] == mode and guard['strict_fifo_steps'] == 6
            and guard['queue_preemption'] is False, 'raw mode/irrevocable FIFO6')
    cell = ledger.npz(root/'cell_001.npz', ['q', 'q_initial', 'official_margins', 'official_deep', 'meta_json'])
    meta = json.loads(str(cell['meta_json']))
    require(meta['seed'] == bank_row['command_seed'] and meta['actuator_delay_steps'] == 6
            and meta['dt'] == .016666, 'cell seed/FIFO/dt')
    recipe = ledger.npz(root/'input_recipe.npz', ['tape', 'sampled_initial', 'joint_soft_limits', 'q_initial'])
    finite(recipe['sampled_initial'], (N, 26), 'finite registered sampled q')
    require(np.isfinite(recipe['joint_soft_limits']).all(), 'finite requested joint limits')
    exact(recipe['q_initial'], cell['q_initial'], 'recipe vs actual cell initial', dtype=True)
    require(recipe['tape'].shape[0] == T+2 and np.isfinite(recipe['tape']).all()
            and np.all(recipe['tape'][:60] == 0), 'actual full962 tape with60 zero prefix')
    bank_path = Path(job['env']['SAFEDUO_INITIAL_BANK_NPZ'])
    require(bank_path.parent.name == str(bank_row['initial_seed']), 'registered fresh bank seed path')
    bank = ledger.npz(bank_path, ['accepted_q', 'risk_pair_index'])
    exact(recipe['sampled_initial'], bank['accepted_q'], 'requested q from registered bank', dtype=True)
    labels = ledger.npz(root/'bank_assignment.npz', ['risk_pair_index'])['risk_pair_index']
    exact(labels, bank['risk_pair_index'], 'raw bank assignments')
    initial = ledger.npz(root/'native_initial.npz')
    require(int(initial['frame']) == -1 and str(initial['boundary']) == 'pre_control_pre_physics',
            'native actual initial boundary')
    opening = ledger.json(root/'open_hand_initialization.json')
    opening_indices, observed_thumbs = opening_binding(opening, initial)
    offset = 0
    for arm, width in zip(ARMS, WIDTHS):
        idx = initial[arm+'_controlled_joint_indices']
        require(idx.shape == (width,), 'native controlled indices')
        exact(initial[arm+'_native_q'][:, idx], cell['q_initial'][:, offset:offset+width],
              'full native vs compact measured initial', dtype=True)
        offset += width
    scores = raw_geometry(root, ledger, cell)
    metrics = raw_contacts(root, ledger, opening_indices)
    metric_file = ledger.npz(c['metric_file'], ['partner_normal_max_N'], c['metric_sha256'])
    exact(metric_file['partner_normal_max_N'], metrics, 'raw vs parent all-substep contact metrics', dtype=True)
    audit_agreement(root, ledger.bind(root/'cell_001.npz'), scores, metrics, g, n, c, peer,
                    sources[str(home/'astra_native_audit.py')])
    hands = hand_camera_binding(root, ledger, hand_receipt, n, sources, home)
    return dict(block=bank_row['block'], mode=mode, windows=N, frames=T, physics_events=2*T,
                labels=labels, **scores, peak=metrics.max((0, 1, 3)),
                observed_initial_thumbs=observed_thumbs, hand_camera_bound=True, hand_camera=hands), recipe, initial


def check(home):
    home = Path(home).resolve()
    ledger = Ledger()
    criteria = ledger.json(home/'ACCEPTANCE_CRITERIA.json', CRITERIA_SHA)
    pairreg = ledger.json(home/'PAIRING_REGISTRATION.json', PAIRING_SHA)
    context = ledger.json(home/'COMMON_HAND_CONTEXT_REGISTRATION.json', COMMON_CONTEXT_SHA)
    initial_spec = ledger.json(home/'INITIAL_STATE_ACCEPTANCE_REGISTRATION.json', INITIAL_STATE_REG_SHA)
    camera_spec = ledger.json(home/'CAMERA_COVERAGE_REGISTRATION.json')
    require(pairreg['criteria_sha256'] == CRITERIA_SHA
            and tuple(pairreg['full_native_actual_initial_fields']) == FIELDS, 'frozen pairing spec')
    regpath = home/'ACCEPTANCE_REGISTRATION.json'
    reg = ledger.json(regpath)
    require(instant(criteria['registered_utc']) <= instant(reg['registered_utc'])
            and instant(pairreg['registered_utc']) <= instant(reg['registered_utc']),
            'criteria and pairing registration precede policy registration')
    require(context['new_random_policy_outcomes_observed'] is False
            and context['source_sha256'] == OPENING_SOURCE_SHA
            and instant(context['registered_utc']) <= instant(reg['registered_utc']),
            'common hand context registered before fresh policy outcomes')
    execution = ledger.json(home/(reg['tag']+'_execution.json'))
    jobs, ran = check_execution(reg, execution, ledger.bind(regpath))
    sources = {str(Path(p).resolve()): digest for p, digest in reg['sources'].items()}
    needed = ['ACCEPTANCE_CRITERIA.json', 'PAIRING_REGISTRATION.json', 'adaptive_reference.py',
              'guard_native.py', 'native_full_view_runner.py', 'native_contacts.py', 'box_support.py',
              'risk_source.py', 'audit_geometry_v2.py', 'audit_native.py', 'audit_contacts.py',
              'astra_native_audit.py', 'astra_acceptance_decision.py', 'bank_qualification_plan.json',
              'COMMON_HAND_CONTEXT_REGISTRATION.json', 'safe_hand_opening.py',
              'INITIAL_STATE_ACCEPTANCE_REGISTRATION.json', 'CAMERA_COVERAGE_REGISTRATION.json',
              'audit_hand_views.py', 'hand_views.py']
    require(all(str(home/p) in sources for p in needed), 'registration missing decision/observer/controller source SHA')
    check_initial_registration(initial_spec, reg, sources, home)
    check_camera_registration(camera_spec, reg, sources, home)
    for path, digest in sources.items():
        ledger.bind(path, digest)
    require(sources[str(home/'safe_hand_opening.py')] == OPENING_SOURCE_SHA,
            'frozen common initialization source')
    expected_banks = check_banks(home, ledger, criteria, sources, reg)
    require(len({sources.get(str(Path(argument(j['argv'], '--ckpt')).resolve())) for j in jobs.values()}) == 1
            and len({argument(j['argv'], '--env-yaml') for j in jobs.values()}) == 1,
            'same frozen actor and configuration across all modes and banks')
    records, pairing, seen_banks = [], [], []
    for bank_row in criteria['initial_bank_rows']:
        block = bank_row['block']
        base_recipe = base_native = None
        for mode in ('zero_inclusive',) + tuple(m for m in MODES if m != 'zero_inclusive'):
            name = f'b{block}_{mode}'
            job, row = jobs[name], ran[name]
            require(job['env']['SAFEDUO_JOINT_MODE'] == mode, 'registered mode vs job ID')
            bank_path = str(Path(job['env']['SAFEDUO_INITIAL_BANK_NPZ']).resolve())
            require(bank_path in sources and bank_path in expected_banks, 'initial bank not frozen/qualified')
            argv = job['argv']
            require(argument(argv, '--seeds') == str(bank_row['command_seed']), 'planned command seed')
            ledger.bind(home/(reg['tag']+'_'+name+'.log'), row['log_sha256'])
            record, recipe, native = inspect_job(home, job, row, bank_row, ledger, sources)
            if base_recipe is None:
                base_recipe, base_native = recipe, native
                seen_banks.append(recipe['sampled_initial'])
            for key in ('tape', 'sampled_initial', 'joint_soft_limits'):
                exact(recipe[key], base_recipe[key], 'exact requested/actual pairing: '+key, dtype=True)
            for key in ('environment_origins',) + tuple(arm+'_'+field for arm in ARMS
                    for field in ('controlled_joint_indices', 'native_joint_names')):
                exact(native[key], base_native[key], 'native initial identity pairing: '+key, dtype=True)
            differences = pairing_difference(base_native, native)
            pairing.append(dict(block=block, mode=mode, differences=differences,
                controlled_actual_initial_exact=bool(np.array_equal(recipe['q_initial'], base_recipe['q_initial'])),
                full_native_actual_initial_exact=not differences))
            records.append(record)
    all_initial = np.concatenate(seen_banks)
    require(all_initial.shape == (128, 26) and np.unique(all_initial, axis=0).shape[0] == 128,
            '128 distinct sampled cases across two fresh banks')
    result = decide(records, criteria, pairing, initial_spec)
    ledger.verify()
    result.update(raw_and_evidence_sha256=ledger.hashes, criteria_sha256=CRITERIA_SHA,
                  pairing_registration_sha256=PAIRING_SHA,
                  acceptance_registration_sha256=ledger.bind(regpath),
                  all_raw_auditor_agreements_checked=True, actual_wait_receipts_checked=True,
                  parent_decision_used=False, arm_motion_threshold_rad=ARM_MOTION_THRESHOLD,
                  common_hand_context_registration_sha256=COMMON_CONTEXT_SHA,
                  initial_state_acceptance_registration_sha256=INITIAL_STATE_REG_SHA,
                  camera_coverage_registration_sha256=ledger.bind(home/'CAMERA_COVERAGE_REGISTRATION.json'),
                  all_hand_camera_bindings_checked=True,
                  hand_camera_bindings=[dict(block=r['block'], mode=r['mode'], **r['hand_camera']) for r in records],
                  full_mesh_camera_coverage_certified=False, camera_occlusion_certified=False,
                  prior_bank_native_hand_qualified=False,
                  common_context_amends_prior_no_initial_state_intervention=True,
                  all1920_common_thumb_native_targets_exact=True,
                  initial_observations=[dict(block=r['block'],mode=r['mode'],
                      strict_env_ids=r['initial_strict_ids'],deep_env_ids=r['initial_deep_ids'],
                      minimum_by_class_m=[None if np.isposinf(v) else float(v)
                                          for v in r['initial_minimum_by_class_m']],
                      actual_native_thumbs=r['observed_initial_thumbs']) for r in records],
                  initial_geometry_scope='actual pre-step0 full9021 raw negative candidate cases must be0 for adoption; original all960 post scoring unchanged',
                  initial_minimum_null_means='no nonexempt rows in that class across all64 initial cases',
                  initial_thumb_pose_is_not_forced_to_requested_goal=True,
                  constructor_initial_position_targets_independently_observed=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('registration_dir', type=Path)
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args(argv)
    # Reserve output first, preventing accidental replacement or wasted rescoring.
    with args.receipt.open('x') as output:
        try:
            result, code = check(args.registration_dir), 0
        except (IncompleteEvidence, FileNotFoundError) as e:
            result, code = dict(status='INCOMPLETE_NOT_ACCEPTED', candidate_decision='NOT_EVALUATED',
                                error=str(e)), 2
        except (InvalidEvidence, KeyError, ValueError, TypeError, IndexError, OSError) as e:
            result, code = dict(status='INVALID_EVIDENCE_NOT_ACCEPTED', candidate_decision='NOT_EVALUATED',
                                error=type(e).__name__+': '+str(e)), 1
        result.update(utc=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
            process_declared_exitcode=code, queued_future_status='UNKNOWN',
            whole_machine_safety_certified=False, physical_safety_certified=False,
            hardware_approved=False, production_promoted=False, exact_restore_certified=False,
            constructor_contacts_observed=False, full_friction_observed=False,
            full_mesh_camera_coverage_certified=False, camera_occlusion_certified=False,
            raw_contact_limit_scope='same-hand-inclusive normal-force diagnostic only; not global harm safety')
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(result['status'], result['candidate_decision'], flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
