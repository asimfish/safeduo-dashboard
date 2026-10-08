"""Independent CPU raw evidence auditor; no parent oracle or Isaac imports.

Usage: python -B astra_native_audit.py ROOT NEW_RECEIPT_PATH [--require-contacts]
       python -B astra_native_audit.py --self-test

ROOT is read-only. A NEW receipt is written with exclusive creation. Exit 0
means raw binding/scoring checks passed, NOT physical safety; collisions are
reported, never filtered. Exit 2 means incomplete, exit 1 means invalid evidence.
The self-test uses only in-memory fixtures and prints its result to stdout.

Three boundaries, all full native joints/roots, exact FIFO6/history, every raw
post geometry score, episodes, camera/native/cell identity, PNG hashes and
frustum geometry are checked independently. No LP/current-feasibility result is
used for physical scoring. UNKNOWN future remains UNKNOWN. No exact restore,
independent forward kinematics, render-pixel provenance, silhouette visibility,
absence of occlusion, or unobserved continuous-time safety is certified.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image

ARMS = ('F_L', 'F_R', 'U_L', 'U_R')
WIDTHS = (7, 7, 6, 6)
CLASSES = ('cross', 'self_F', 'self_U', 'table')
BOUNDARIES = {'pre': 'pre_control_pre_physics', 'physics': 'pre_physics',
              'post': 'post_physics_pre_render'}
NATIVE = ('native_q', 'native_qd', 'native_root_xyzw', 'native_root_velocity')
TARGETS = ('issued_controlled_target', 'applied_controlled_target',
           'pending_controlled_targets', 'pending_project_targets')
VIEWS = {'overview', 'front', 'reverse', 'f_pair', 'f_opposite_low',
         'f_opposite_high', 'u_pair', 'u_opposite_low', 'u_opposite_high'}


class InvalidEvidence(ValueError):
    pass


class IncompleteEvidence(InvalidEvidence):
    pass


def require(condition, label):
    if not condition:
        raise InvalidEvidence(label)


def equal(actual, expected, label):
    require(np.array_equal(actual, expected), label)


def finite(value, label):
    require(np.isfinite(value).all(), 'nonfinite ' + label)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            value.update(block)
    return value.hexdigest()


class RawReader:
    """Hash every consumed raw artifact, reject mutation during the audit."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.hashes = {}

    def path(self, relative):
        path = (self.root / relative).resolve()
        require(path.is_relative_to(self.root), 'artifact escapes root: ' + str(relative))
        require(path.is_file(), 'missing raw artifact: ' + str(relative))
        return path

    def read(self, relative, loader, expected=None):
        path = self.path(relative)
        sha = digest(path)
        if expected is not None:
            require(sha == expected, 'declared SHA mismatch: ' + str(relative))
        if str(relative) in self.hashes:
            require(sha == self.hashes[str(relative)], 'artifact changed: ' + str(relative))
        value = loader(path)
        require(digest(path) == sha, 'artifact changed during read: ' + str(relative))
        self.hashes[str(relative)] = sha
        return value

    def json(self, relative, expected=None):
        return self.read(relative, lambda p: json.loads(p.read_text()), expected)

    def npz(self, relative, keys=None, expected=None):
        def load(path):
            with np.load(path, allow_pickle=False) as data:
                return {k: data[k].copy() for k in (data.files if keys is None else keys)}
        return self.read(relative, load, expected)

    def verify_unchanged(self):
        for relative, sha in self.hashes.items():
            require(digest(self.path(relative)) == sha, 'artifact changed at closure: ' + relative)


def issued_at(cell, step):
    return cell['q_initial'] if step < 0 else cell['controller_target'][step]


def queue_at(cell, step):
    """Queue AFTER step: last six issued targets, initial target before step zero."""
    return np.stack([issued_at(cell, k) for k in range(step - 5, step + 1)])


def compact(packet, boundary, key):
    return np.concatenate([packet[f'{boundary}_{a}_{key}'] for a in ARMS], axis=-1)


def check_packet(packet, cell, step, initial, previous=None):
    require(step >= 0 and step < len(cell['q']), 'packet frame range')
    origins = initial['environment_origins']
    require(origins.shape == (64, 3), 'origin shape')
    finite(origins, 'initial origins')
    for boundary, label in BOUNDARIES.items():
        require(int(packet[boundary + '_frame']) == step, 'frame mismatch')
        require(str(packet[boundary + '_boundary']) == label, 'boundary mismatch')
        equal(packet[boundary + '_environment_origins'], origins, 'origin identity')
        qs, qds = [], []
        for arm, width in zip(ARMS, WIDTHS):
            prefix = f'{boundary}_{arm}_'
            q, qd = (packet[prefix + key] for key in ('native_q', 'native_qd'))
            idx = packet[prefix + 'controlled_joint_indices']
            names = packet[prefix + 'native_joint_names']
            require(q.ndim == 2 and q.shape[0] == 64 and qd.shape == q.shape,
                    'full joint shape ' + arm)
            require(idx.shape == (width,) and np.issubdtype(idx.dtype, np.integer),
                    'controlled indices type/shape ' + arm)
            require(len(np.unique(idx)) == width and (idx >= 0).all() and (idx < q.shape[1]).all(),
                    'controlled indices uniqueness/range ' + arm)
            require(names.shape == (q.shape[1],) and len(np.unique(names)) == len(names),
                    'full joint names ' + arm)
            for key in ('controlled_joint_indices', 'native_joint_names'):
                equal(packet[prefix + key], initial[arm + '_' + key], 'joint identity ' + arm)
            require(packet[prefix + 'native_root_xyzw'].shape == (64, 7), 'root shape')
            require(packet[prefix + 'native_root_velocity'].shape == (64, 6), 'root velocity shape')
            for key in NATIVE + TARGETS:
                finite(packet[prefix + key], prefix + key)
            for key in TARGETS:
                expected_shape = (6, 64, width) if key.startswith('pending') else (64, width)
                require(packet[prefix + key].shape == expected_shape, 'target shape ' + prefix + key)
            if boundary == 'pre':
                for key in NATIVE + TARGETS:
                    predecessor = initial[arm + '_' + key] if previous is None else previous['post_' + arm + '_' + key]
                    equal(packet[prefix + key], predecessor, 'native/target continuity ' + prefix + key)
            if boundary == 'physics':
                for key in NATIVE:
                    equal(packet[prefix + key], packet['pre_' + arm + '_' + key],
                          'control advanced native physics ' + arm + '_' + key)
            if boundary == 'post':
                for key in TARGETS:
                    equal(packet[prefix + key], packet['physics_' + arm + '_' + key],
                          'target changed during physics ' + arm + '_' + key)
            qs.append(q[:, idx]); qds.append(qd[:, idx])
        q, qd = np.concatenate(qs, -1), np.concatenate(qds, -1)
        expected_q = cell['q'][step] if boundary == 'post' else cell['q_initial'] if step == 0 else cell['q'][step - 1]
        equal(q, expected_q, 'native controlled q vs cell')
        if boundary != 'post':
            equal(qd, cell['pre_qd_compact'][step], 'native controlled qd vs pre cell')
        elif step + 1 < len(cell['q']):
            equal(qd, cell['pre_qd_compact'][step + 1], 'post qd vs next cell pre')
        target_step = step - 1 if boundary == 'pre' else step
        equal(compact(packet, boundary, 'issued_controlled_target'), issued_at(cell, target_step),
              'issued target at boundary')
        equal(compact(packet, boundary, 'applied_controlled_target'), issued_at(cell, target_step - 6),
              'irrevocable FIFO6 applied truth')
        expected_queue = queue_at(cell, target_step)
        for key in ('pending_controlled_targets', 'pending_project_targets'):
            equal(compact(packet, boundary, key), expected_queue, 'FIFO6/history truth ' + boundary + key)
        if boundary == 'pre':
            equal(expected_queue, cell['pre_pending_actuator_targets'][step], 'cell pre FIFO6')
            equal(expected_queue, cell['pre_pending_project_history'][step], 'cell pre history')
        else:
            equal(compact(packet, boundary, 'applied_controlled_target'), cell['actuator_target'][step],
                  'cell applied target')


def row_buckets(identity):
    classes = np.asarray(identity['class_id'])
    pairs = np.asarray(identity['pair_sphere_idx'])
    arms = np.asarray(identity['sphere_arm_id'])
    radii = np.asarray(identity['sphere_radii_m'])
    require(identity['rows'] == 9021 and classes.shape == (9021,) and pairs.shape == (9021, 2),
            'full9021 row identity')
    equal(np.asarray(identity['pair_id']), np.arange(9021), 'stable row IDs')
    require(np.isin(classes, [0, 1, 2]).all() and np.isin(arms, [0, 1, 2, 3]).all(), 'class/arm IDs')
    require(np.issubdtype(pairs.dtype, np.integer), 'pair indices type')
    require(len(identity['sphere_names']) == len(arms) == len(radii), 'sphere identity lengths')
    finite(radii, 'sphere radii'); require((radii > 0).all(), 'positive radii')
    require((pairs[:, 0] >= 0).all() and (pairs[:, 0] < len(arms)).all(), 'first sphere index')
    robot = classes != 2
    require((pairs[robot, 1] >= 0).all() and (pairs[robot, 1] < len(arms)).all(), 'second sphere index')
    require(np.isin(pairs[~robot, 1], [0, 1]).all(), 'table indices')
    labels = np.zeros(9021, np.int64)
    labels[classes == 1] = np.where(arms[pairs[classes == 1, 0]] < 2, 1, 2)
    labels[classes == 2] = 3
    require(all((labels == k).any() for k in range(4)), 'four geometry classes present')
    return labels


def raw_margins(d, exempt, labels):
    require(d.shape == (64, 9021) and d.dtype == np.float32, 'raw float32 full geometry shape')
    require(exempt.shape == d.shape and exempt.dtype == np.bool_, 'raw exemption shape/type')
    finite(d, 'full raw geometry')
    require(not exempt[:, labels != 3].any(), 'exemptions must retain original table-only semantics')
    return np.stack([np.min(np.where(exempt | (labels != k), np.float32(np.inf), d), axis=1)
                     for k in range(4)], axis=-1)


def check_scores(cell, computed, episodes, protocol):
    equal(computed, cell['official_margins'], 'every raw margin vs cell EXACT, no epsilon')
    strict_by_class, deep_by_class = computed < 0, computed < -.005
    equal(deep_by_class, cell['official_deep'], 'original strict -5mm damaging threshold')
    strict, deep = strict_by_class.any(-1), deep_by_class.any(-1)
    require(len(episodes) == 64 and {e['env_id'] for e in episodes} == set(range(64)), 'episode identities')
    for e in episodes:
        i = e['env_id']
        require(e['violation'] == bool(strict[:, i].any()), 'episode strict flag')
        require(e['violation_steps'] == int(strict[:, i].sum()), 'episode strict count')
        require(e['damaging'] == bool(deep[:, i].any()), 'episode damaging flag')
        require(e['violation_by_class'] == dict(zip(CLASSES, strict_by_class[:, i].any(0).tolist())), 'episode class violation')
        require(e['damaging_by_class'] == dict(zip(CLASSES, deep_by_class[:, i].any(0).tolist())), 'episode class damaging')
        equal(np.array([e['min_nonexempt_margin_m'][k] for k in CLASSES], np.float32),
              computed[:, i].min(0), 'episode class minimum')
    require(protocol['violations'] == int(strict.any(0).sum()), 'protocol violation aggregate')
    return strict, deep


def check_first_images(captures, kind, hits, assignments, declared_count):
    expected = set()
    for label in np.unique(assignments):
        hit = hits[:, assignments == label].any(-1)
        if hit.any():
            t = int(np.flatnonzero(hit)[0])
            e = int(np.flatnonzero(hits[t] & (assignments == label))[0])
            expected.add((e, t))
    groups = [c for c in captures if c['capture_kind'] == kind]
    actual = {(c['env_id'], c['step']) for c in groups}
    require(actual == expected and len(groups) == len(actual) == declared_count,
            'earliest ' + kind + ' image per risk label')


def check_frustum(view, centers, radii):
    W, K = np.asarray(view['actual_camera_to_world_row_matrix']), np.asarray(view['actual_intrinsic_matrix'])
    clip = np.asarray(view['actual_clipping_range_m'])
    require(W.shape == (4, 4) and K.shape == (3, 3) and clip.shape == (2,), 'camera shapes')
    for x in (W, K, clip, centers, radii): finite(x, 'camera geometry')
    require(0 < clip[0] < clip[1] and K[0, 0] > 0 and K[1, 1] > 0, 'camera optics')
    require(np.allclose(W[:3, :3] @ W[:3, :3].T, np.eye(3), rtol=0, atol=1e-6)
            and np.isclose(np.linalg.det(W[:3, :3]), 1, rtol=0, atol=1e-6), 'proper camera rotation')
    equal(W[:3, 3], np.zeros(3), 'camera homogeneous rotation column')
    require(W[3, 3] == 1, 'camera homogeneous element')
    require(np.allclose(W[3, :3], view['actual_position_world_m'], rtol=0, atol=1e-6), 'camera position binding')
    pc = np.c_[centers, np.ones(len(centers))] @ np.linalg.inv(W)
    x, y, z = pc[:, 0], pc[:, 1], -pc[:, 2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    planes = np.stack([(fx*x + cx*z)/np.hypot(fx, cx) - radii,
        ((1280-cx)*z - fx*x)/np.hypot(fx, 1280-cx) - radii,
        (cy*z - fy*y)/np.hypot(fy, cy) - radii,
        ((720-cy)*z + fy*y)/np.hypot(fy, 720-cy) - radii,
        z - clip[0] - radii, clip[1] - z - radii], -1)
    # This tolerance is only camera fitting; physical scores above use equality.
    require(planes.min() >= .049999, 'sphere clipping/50mm camera clearance')
    require(np.allclose(planes.min(0), view['observed_sphere_frustum']['minimum_by_plane_m'],
                        rtol=0, atol=1e-7), 'frustum summary mismatch')
    return float(planes.min())


def contact_context(reader, control_steps, control_dt, required):
    present = any((reader.root / name).is_file() for name in
                  ('native_contact_identity.json', 'native_contact_receipts.json'))
    if not present:
        if required:
            raise IncompleteEvidence('required physics-substep contact evidence absent')
        return None
    receipts = reader.json('native_contact_receipts.json')
    if receipts.get('status') != 'complete':
        raise IncompleteEvidence('physics-substep contact stream incomplete')
    identity = reader.json('native_contact_identity.json', receipts['identity_sha256'])
    require(receipts['control_steps'] == control_steps and receipts['physics_events'] == 2*control_steps
            and receipts['substeps_per_control'] == 2, 'contact event coverage')
    require(receipts['capacity'] == 262144 and identity['environment_count'] == 64, 'registered contact capacity/envs')
    require(identity['physics_dt_s'] * 2 == control_dt, 'contact physics dt')
    views = {v['arm']: v for v in identity['views']}
    require(set(views) == set(ARMS) and len(identity['views']) == 4, 'four contact arm identities')
    partners = identity['partners_env0']
    require(len(partners) == len(set(partners)) and len(partners) > 0, 'contact partner identity')
    for arm, view in views.items():
        sensors, filters = view['sensors'], view['filters']
        envs = np.asarray(view['env_ids'])
        require(len(sensors) == len(set(sensors)) == len(filters) == len(envs), 'contact sensor identities')
        require(np.issubdtype(envs.dtype, np.integer) and set(envs.tolist()) == set(range(64)), 'contact all64 coverage')
        require(view['capacity'] == receipts['capacity'], 'contact view capacity')
        for sensor, row, env in zip(sensors, filters, envs):
            match = re.match(r'^/World/envs/env_(\d+)/', sensor)
            require(match is not None and int(match.group(1)) == env and '/'+arm+'/' in sensor,
                    'contact sensor exact environment/arm')
            expected = [path.replace('/env_0/', f'/env_{env}/') for path in partners]
            require(row == expected, 'contact exact own-env/static/ground partner identity')
    return dict(receipts=receipts, identity=identity, views=views)


def contact_events(reader, context):
    seen = 0
    for chunk in context['receipts']['chunks']:
        require(chunk['start'] == seen and seen < chunk['stop'] <= context['receipts']['physics_events'],
                'contact contiguous event chunks')
        data = reader.npz(chunk['path'], expected=chunk['sha256'])
        require(all(len(value) == chunk['stop'] - seen for value in data.values()), 'contact chunk length')
        for i in range(chunk['stop'] - seen):
            yield {key: value[i] for key, value in data.items()}
        seen = chunk['stop']
    require(seen == context['receipts']['physics_events'], 'all contact events present')


def check_contact_event(event, frame, substep, cell, initial, packet, context):
    require(int(event['frame']) == frame and int(event['substep']) == substep, 'contact frame/substep')
    require(float(event['physics_dt']) == context['identity']['physics_dt_s'], 'contact dt')
    maxima = np.zeros((64, 4), np.float64)
    outside_hand = np.zeros_like(maxima)
    accounting, peak_count, offset = 0., 0, 0
    for ai, (arm, width) in enumerate(zip(ARMS, WIDTHS)):
        view = context['views'][arm]; envs = np.asarray(view['env_ids'])
        matrix, net = event[arm+'_partner_normal'], event[arm+'_net']
        counts, starts = event[arm+'_normal_counts'], event[arm+'_normal_starts']
        require(matrix.shape == (len(envs), len(view['filters'][0]), 3)
                and net.shape == (len(envs), 3) and counts.shape == starts.shape == matrix.shape[:2], 'contact tensor shapes')
        for key in ('partner_normal', 'net', 'normal_counts', 'normal_starts'):
            finite(event[arm+'_'+key], 'contact '+arm+'_'+key)
        require(np.issubdtype(counts.dtype, np.integer) and np.issubdtype(starts.dtype, np.integer), 'contact count integer type')
        cap = context['receipts']['capacity']; total = int(counts.sum(dtype=np.int64))
        require((counts >= 0).all() and (starts >= 0).all() and total < cap
                and (starts.astype(np.int64) + counts.astype(np.int64) <= cap).all(), 'contact capacity/truncation')
        used = counts > 0
        indices = np.argsort(starts[used]); begin = starts[used][indices]; size = counts[used][indices]
        require(len(begin) < 2 or (begin[1:] >= begin[:-1]+size[:-1]).all(), 'contact data segment overlap')
        peak_count = max(peak_count, total)
        norm = np.linalg.norm(matrix.astype(np.float64), axis=-1)
        for env in range(64):
            selected = np.flatnonzero(envs == env)
            known_hand = {view['sensors'][i] for i in selected}
            same_hand = np.asarray([[p in known_hand for p in view['filters'][i]] for i in selected])
            maxima[env, ai] = norm[selected].max()
            outside_hand[env, ai] = np.where(same_hand, 0., norm[selected]).max()
        accounting = max(accounting, float(np.abs(matrix.astype(np.float64).sum(1)-net).max()))
        idx = initial[arm+'_controlled_joint_indices']
        for key in NATIVE + ('native_position_targets',):
            finite(event[arm+'_'+key], 'micro native '+arm+'_'+key)
            expected_shape = initial[arm+'_native_q'].shape if key == 'native_position_targets' else initial[arm+'_'+key].shape
            require(event[arm+'_'+key].shape == expected_shape, 'micro full native shape')
            if substep == 1 and key in NATIVE:
                equal(event[arm+'_'+key], packet['post_'+arm+'_'+key], 'micro2 full native state vs macro post')
        equal(event[arm+'_native_position_targets'][:, idx], cell['actuator_target'][frame, :, offset:offset+width],
              'EACH physics substep actual PhysX position target vs FIFO6')
        offset += width
    return maxima, outside_hand, accounting, peak_count


def check_camera(reader, capture, cell, packet, identity, raw, dt, mode):
    state = reader.json(capture['state'], capture['sha256'])
    step, env = capture['step'], capture['env_id']
    require(state['step'] == step and state['env_id'] == env
            and state['capture_kind'] == capture['capture_kind'], 'camera receipt identity')
    require(state['visual_mode'] == mode, 'camera method identity')
    require(state['state_time_s'] == (step + 1) * dt and state['environment_count'] == 64, 'camera time/env count')
    if capture['capture_kind'] == 'first_native_contact':
        require(state['native_contact_observer'] == 'native_contact_receipts.json'
                and state['native_contact_image_scope'].startswith('macro post state;'),
                'contact image must represent macro post, never relabel micro peak')
    require(set(state['views']) == VIEWS and len(state['images']) == 9, 'nine distinct camera views')
    require(capture['images'] == state['images'], 'camera manifest image bindings')
    before = reader.npz(state['fresh_native_snapshot'], expected=state['fresh_native_before_sha256'])
    after = reader.npz(state['fresh_native_after_snapshot'], expected=state['fresh_native_after_sha256'])
    require(set(before) == set(after) == {a + '_' + k for a in ARMS for k in ('q', 'qd', 'root', 'root_vel')},
            'full native render snapshot fields')
    for arm in ARMS:
        idx = packet['post_' + arm + '_controlled_joint_indices']
        equal(np.asarray(state['fresh_native_controlled_joint_indices'][arm]), idx, 'camera controlled indices')
        equal(np.asarray(state['fresh_native_all_joint_names'][arm]), packet['post_' + arm + '_native_joint_names'], 'camera full joint names')
        for short, key in zip(('q', 'qd', 'root', 'root_vel'), NATIVE):
            equal(before[arm + '_' + short], after[arm + '_' + short], 'all64 render native unchanged')
            equal(before[arm + '_' + short], packet['post_' + arm + '_' + key], 'render native vs scored boundary')
            equal(np.asarray(state['fresh_native_selected'][arm][short], np.float32), before[arm + '_' + short][env], 'selected native JSON binding')
        for field in ('q', 'qd'):
            equal(np.asarray(state['arms'][arm][field], np.float32), before[arm + '_' + field][env, idx], 'camera controlled ' + field)
        equal(np.asarray(state['controller_target'][arm], np.float32), packet['post_' + arm + '_issued_controlled_target'][env], 'camera issued target')
        equal(np.asarray(state['actuator_target'][arm], np.float32), packet['post_' + arm + '_applied_controlled_target'][env], 'camera applied target')
        equal(np.asarray([q[arm] for q in state['pending_actuator_targets']], np.float32),
              packet['post_' + arm + '_pending_controlled_targets'][:, env], 'camera pending FIFO6')
    equal(np.asarray(state['origin_world_m'], np.float32), packet['post_environment_origins'][env], 'camera origin')
    centers = np.asarray(state['sphere_centers_world_m'], np.float32)
    equal(centers, cell['sphere_centers'][step, env], 'camera centers vs actual cell centers')
    equal(np.asarray(state['sphere_radii_m'], np.float32), np.asarray(identity['sphere_radii_m'], np.float32), 'camera sphere radii identity')
    for field, key, dtype in [('full_distances_m', 'd', np.float32), ('full_exempt', 'exempt', bool),
                              ('full_braking_dmin_m', 'dmin', np.float32)]:
        equal(np.asarray(state[field], dtype=dtype), raw[key], 'camera raw geometry ' + key)
    paths = [v['path'] for v in state['images']]
    require(len(set(paths)) == 9, 'distinct PNG paths')
    radii, arms = np.asarray(state['sphere_radii_m']), np.asarray(identity['sphere_arm_id'])
    clearance = []
    for name, view in state['views'].items():
        mask = arms < 2 if name.startswith('f_') else arms >= 2 if name.startswith('u_') else np.ones(len(arms), bool)
        clearance.append(check_frustum(view, centers[mask].astype(np.float64), radii[mask]))
        images = [v for v in state['images'] if Path(v['path']).name == f'step_{step:04d}_{name}.png']
        require(len(images) == 1, 'view PNG identity')
        def validate_png(path):
            with Image.open(path) as im:
                im.load()
                require(im.size == (1280, 720) and im.mode == 'RGB', 'PNG dimensions/mode')
                require(np.asarray(im).std() > 1, 'blank PNG')
        reader.read(images[0]['path'], validate_png, images[0]['sha256'])
    return min(clearance)


def audit(root, require_contacts=False):
    root = Path(root)
    if not (root / 'visual_protocol.json').is_file():
        raise IncompleteEvidence('no closed visual_protocol.json')
    reader = RawReader(root)
    protocol = reader.json('visual_protocol.json')
    if protocol.get('status') != 'complete':
        raise IncompleteEvidence('protocol status is ' + str(protocol.get('status')))
    T = protocol['steps']
    require(isinstance(T, int) and T > 0 and protocol['completed_windows'] == 64, 'complete frame/window count')
    keys = ['q', 'q_initial', 'pre_qd_compact', 'controller_target', 'actuator_target',
            'pre_pending_actuator_targets', 'pre_pending_project_history', 'official_margins',
            'official_deep', 'sphere_centers', 'meta_json']
    cell = reader.npz('cell_001.npz', keys)
    require(cell['q'].shape == (T, 64, 26) and cell['q_initial'].shape == (64, 26), 'cell shape')
    for key in keys:
        if key not in ('meta_json', 'official_margins'):
            finite(cell[key], 'cell ' + key)
    require(cell['official_margins'].shape == (T, 64, 4) and not np.isnan(cell['official_margins']).any(), 'official margin shape/NaN')
    meta = json.loads(str(cell['meta_json']))
    require(meta['dt'] == protocol['dt'] and meta['actuator_delay_steps'] == 6, 'cell dt/FIFO config')
    identity = reader.json('full_row_identity.json')
    labels = row_buckets(identity)
    native = reader.json('native_receipts.json')
    cameras = reader.json('camera_receipts.json')
    guard = reader.json('guard_metadata.json')
    require(native['steps'] == T and native['boundaries'] == list(BOUNDARIES.values()), 'three native boundaries coverage')
    require(guard['strict_fifo_steps'] == 6 and guard['queue_preemption'] is False, 'registered irrevocable FIFO6')
    captures = cameras['receipts']
    require(len(captures) == cameras['groups'] and cameras['PNG'] == 9 * len(captures), 'camera receipt counts')
    identities = [(c['capture_kind'], c['env_id'], c['step']) for c in captures]
    require(len(set(identities)) == len(identities), 'duplicate camera group')
    require(all(0 <= c['step'] < T and 0 <= c['env_id'] < 64 for c in captures), 'camera group range')
    scheduled = {(c['env_id'], c['step']) for c in captures if c['capture_kind'] == 'scheduled'}
    require(scheduled == {(e, t) for e in protocol['slots'] for t in protocol['capture_steps']}, 'complete scheduled cameras')
    require(cameras['scheduled_groups'] == len(scheduled), 'scheduled group count')
    require(all(c['capture_kind'] in ('scheduled', 'first_failure', 'first_native_contact') for c in captures), 'camera group kind')
    wanted = {(c['step'], c['env_id']) for c in captures}
    raw_camera = {}
    computed = np.empty((T, 64, 4), np.float32)
    coverage = np.zeros(T, bool)
    forecast = reader.json('forecast_receipts.json')
    require(forecast['steps'] == T and forecast['rows'] == 9021 and forecast['envs'] == 64, 'forecast dimensions')
    seen = 0
    for chunk in forecast['chunks']:
        require(chunk['start'] == seen and seen < chunk['stop'] <= T, 'forecast contiguous chunks')
        data = reader.npz(chunk['path'], ['measured_d', 'exempt', 'dmin'], chunk['sha256'])
        require(len(data['measured_d']) == chunk['stop'] - seen, 'forecast chunk length')
        for i, t in enumerate(range(seen, chunk['stop'])):
            require(data['exempt'][i].shape == (64, 1128) and data['exempt'].dtype == np.uint8, 'packed exemptions')
            ex = np.unpackbits(data['exempt'][i], axis=-1, count=9021).astype(bool)
            d, dm = data['measured_d'][i], data['dmin'][i]
            m = raw_margins(d, ex, labels)
            if t > 0:
                computed[t - 1], coverage[t - 1] = m, True
                for frame, env in wanted:
                    if frame == t - 1:
                        raw_camera[frame, env] = {'d': d[env].copy(), 'exempt': ex[env].copy(), 'dmin': dm[env].copy()}
        seen = chunk['stop']
    require(seen == T, 'all pre geometry frames')
    final = reader.npz('post_geometry_final.npz')
    computed[-1], coverage[-1] = raw_margins(final['d'], final['exempt'], labels), True
    equal(final['centers'], cell['sphere_centers'][-1], 'terminal geometry centers vs same cell')
    for frame, env in wanted:
        if frame == T - 1:
            raw_camera[frame, env] = {k: final[k][env].copy() for k in ('d', 'exempt', 'dmin')}
    require(coverage.all(), 'every post geometry frame including terminal')
    strict, deep = check_scores(cell, computed, reader.json('episodes.json'), protocol)
    assignments = reader.npz('bank_assignment.npz', ['risk_pair_index'])['risk_pair_index']
    require(assignments.shape == (64,), 'risk labels shape')
    check_first_images(captures, 'first_failure', strict, assignments, cameras['first_failure_groups'])
    initial = reader.npz('native_initial.npz')
    require(int(initial['frame']) == -1 and str(initial['boundary']) == BOUNDARIES['pre'], 'native initial boundary')
    has_contact_images = any(c['capture_kind'] == 'first_native_contact' for c in captures)
    contacts = contact_context(reader, T, meta['dt'], require_contacts or has_contact_images)
    micro = contact_events(reader, contacts) if contacts is not None else None
    contact_peaks = np.zeros((64, 4), np.float64)
    contact_nonself_peaks = np.zeros_like(contact_peaks)
    contact_over = np.zeros((T, 64), bool)
    contact_nonself_over = np.zeros_like(contact_over)
    contact_accounting, contact_capacity_peak = 0., 0
    previous, seen, camera_count, minimum = None, 0, 0, None
    for chunk in native['chunks']:
        require(chunk['start'] == seen and seen < chunk['stop'] <= T, 'native contiguous chunks')
        data = reader.npz(chunk['path'], expected=chunk['sha256'])
        require(all(len(v) == chunk['stop'] - seen for v in data.values()), 'native chunk length')
        for i, t in enumerate(range(seen, chunk['stop'])):
            packet = {k: v[i] for k, v in data.items()}
            check_packet(packet, cell, t, initial, previous)
            if micro is not None:
                for substep in (0, 1):
                    try:
                        event = next(micro)
                    except StopIteration:
                        raise InvalidEvidence('missing contact substep') from None
                    peak, nonself, accounting, count = check_contact_event(event, t, substep, cell, initial, packet, contacts)
                    contact_peaks = np.maximum(contact_peaks, peak)
                    contact_nonself_peaks = np.maximum(contact_nonself_peaks, nonself)
                    contact_over[t] |= (peak > .1).any(-1)
                    contact_nonself_over[t] |= (nonself > .1).any(-1)
                    contact_accounting = max(contact_accounting, accounting)
                    contact_capacity_peak = max(contact_capacity_peak, count)
            for capture in captures:
                if capture['step'] == t:
                    value = check_camera(reader, capture, cell, packet, identity,
                        raw_camera[t, capture['env_id']], meta['dt'], guard['mode'])
                    minimum = value if minimum is None else min(minimum, value)
                    camera_count += 1
            previous = {k: v.copy() for k, v in packet.items() if k.startswith('post_')}
        seen = chunk['stop']
    require(seen == T and camera_count == len(captures), 'all native frames and cameras checked')
    if micro is not None:
        require(next(micro, None) is None, 'extra contact substeps')
        # Producer registration uses RAW >.1N at either microstep, then captures
        # macro post. Same-hand-filtered diagnostics do not change image selection.
        check_first_images(captures, 'first_native_contact', contact_over, assignments,
                           cameras['native_contact_groups'])
    reader.verify_unchanged()
    contact_result = (dict(status='NOT_OBSERVED', actual_PhysX_position_targets_verified=False) if contacts is None else
        dict(status='PASS_SUBSTEP_BINDING_ONLY', physics_events=2*T,
             actual_PhysX_position_targets_verified=True, raw_normal_diagnostic_threshold_N=.1,
             raw_windows_over_threshold=int(contact_over.any(0).sum()), raw_env_frames_over_threshold=int(contact_over.sum()),
             raw_normal_peak_by_env_arm_N=contact_peaks.tolist(), maximum_net_minus_filtered_normal_abs_N=contact_accounting,
             maximum_contact_count=contact_capacity_peak, known_same_hand_exemption_applied=True,
             same_hand_definition='partner exact path is a registered hand sensor of this same arm and environment; diagnostic only',
             non_same_hand_windows_over_0p1N=int(contact_nonself_over.any(0).sum()),
             non_same_hand_env_frames_over_0p1N=int(contact_nonself_over.sum()),
             non_same_hand_peak_by_env_arm_N=contact_nonself_peaks.tolist(),
             first_native_contact_images='earliest raw partner normal >.1N at either microstep per label; macro post camera state',
             constructor_contacts_observed=False, full_friction_observed=False, whole_mesh_safety_certified=False))
    return dict(status='PASS_RAW_BINDING_AND_ORIGINAL_SCORING_ONLY', root=str(reader.root),
        frames=T, windows=64, native_boundary_packets=3*T, full_geometry_rows=9021,
        raw_geometry_env_frames=T*64, camera_groups=camera_count, PNG=camera_count*9,
        minimum_camera_clearance_m=minimum, strict_windows=int(strict.any(0).sum()),
        strict_env_frames=int(strict.sum()), deep_windows=int(deep.any(0).sum()),
        deep_env_frames=int(deep.sum()), strict_env_ids=np.flatnonzero(strict.any(0)).tolist(),
        deep_env_ids=np.flatnonzero(deep.any(0)).tolist(),
        minimum_by_class_m=computed.min((0, 1)).tolist(), raw_sha256=reader.hashes, contacts=contact_result,
        queued_future_status='UNKNOWN', physical_safety_certified=False,
        exact_restore_certified=False, score_threshold_epsilon=0,
        terminal_post_qd_independent_cell_channel=False,
        scope='Raw same-run binding and original discrete-frame scoring; not final experiment acceptance or a safety certificate')


def synthetic_packet_fixture():
    """Distinct per-step commands make FIFO timing mistakes observable."""
    T = 9
    q0 = np.zeros((64, 26), np.float32)
    targets = np.stack([np.full_like(q0, (t + 1) / 100) for t in range(T)])
    cell = dict(q_initial=q0, q=targets.copy(), controller_target=targets,
                pre_qd_compact=np.zeros_like(targets),
                actuator_target=np.stack([q0 if t < 6 else targets[t - 6] for t in range(T)]))
    cell['pre_pending_actuator_targets'] = np.stack([queue_at(cell, t - 1) for t in range(T)])
    cell['pre_pending_project_history'] = cell['pre_pending_actuator_targets'].copy()
    initial = dict(frame=np.array(-1), boundary=np.array(BOUNDARIES['pre']), environment_origins=np.zeros((64, 3), np.float32))
    offset = 0
    for arm, width in zip(ARMS, WIDTHS):
        full_q = np.zeros((64, width + 2), np.float32)
        initial.update({arm + '_native_q': full_q, arm + '_native_qd': full_q.copy(),
            arm + '_native_root_xyzw': np.tile(np.array([0, 0, 0, 0, 0, 0, 1], np.float32), (64, 1)),
            arm + '_native_root_velocity': np.zeros((64, 6), np.float32),
            arm + '_controlled_joint_indices': np.arange(width),
            arm + '_native_joint_names': np.array([arm + str(i) for i in range(width + 2)])})
        for key in TARGETS:
            initial[arm + '_' + key] = np.zeros((6, 64, width) if key.startswith('pending') else (64, width), np.float32)
        offset += width
    packets = []
    for t in range(T):
        packet = {}
        for boundary, label in BOUNDARIES.items():
            packet.update({boundary + '_' + k: v.copy() for k, v in initial.items()})
            packet[boundary + '_frame'] = np.array(t)
            packet[boundary + '_boundary'] = np.array(label)
            offset = 0
            target_t = t - 1 if boundary == 'pre' else t
            for arm, width in zip(ARMS, WIDTHS):
                sl = slice(offset, offset + width); prefix = boundary + '_' + arm + '_'
                packet[prefix + 'native_q'][:, :width] = (cell['q'][t] if boundary == 'post' else q0 if t == 0 else cell['q'][t - 1])[:, sl]
                packet[prefix + 'issued_controlled_target'] = issued_at(cell, target_t)[:, sl].copy()
                packet[prefix + 'applied_controlled_target'] = issued_at(cell, target_t - 6)[:, sl].copy()
                for key in ('pending_controlled_targets', 'pending_project_targets'):
                    packet[prefix + key] = queue_at(cell, target_t)[..., sl].copy()
                offset += width
        packets.append(packet)
    return cell, initial, packets


def synthetic_closed_fixture():
    """An in-memory closed root exercises traversal without inventing real evidence."""
    cell, initial, packets = synthetic_packet_fixture()
    T = len(packets)
    identity = dict(rows=9021, class_id=(np.arange(9021) % 3).tolist(),
        pair_id=list(range(9021)), sphere_names=list(ARMS), sphere_arm_id=list(range(4)),
        sphere_radii_m=[.01] * 4, pair_sphere_idx=[])
    for i, cls in enumerate(identity['class_id']):
        identity['pair_sphere_idx'].append([0, 2] if cls == 0 else
            ([0, 1] if i % 2 else [2, 3]) if cls == 1 else [0, 0])
    centers = np.tile(np.array([0, 0, -1], np.float32), (64, 4, 1))
    cell.update(sphere_centers=np.stack([centers] * T), official_margins=np.ones((T, 64, 4), np.float32),
        official_deep=np.zeros((T, 64, 4), bool), meta_json=np.array(json.dumps(dict(dt=.1, actuator_delay_steps=6))))
    d = np.ones((64, 9021), np.float32); ex = np.zeros(d.shape, bool); dm = np.zeros_like(d)
    protocol = dict(status='complete', steps=T, dt=.1, completed_windows=64, violations=0,
                    slots=[0], capture_steps=[T - 1])
    episodes = [dict(env_id=e, violation=False, violation_steps=0, damaging=False,
                    violation_by_class=dict.fromkeys(CLASSES, False),
                    damaging_by_class=dict.fromkeys(CLASSES, False),
                    min_nonexempt_margin_m=dict.fromkeys(CLASSES, 1.)) for e in range(64)]
    final_packet = packets[-1]
    render = {a + '_' + short: final_packet['post_' + a + '_' + key].copy()
              for a in ARMS for short, key in zip(('q', 'qd', 'root', 'root_vel'), NATIVE)}
    planes = [640/np.hypot(500, 640)-.01] * 2 + [360/np.hypot(500, 360)-.01] * 2 + [.98, 8.99]
    view = dict(actual_camera_to_world_row_matrix=np.eye(4).tolist(),
                actual_intrinsic_matrix=[[500, 0, 640], [0, 500, 360], [0, 0, 1]],
                actual_clipping_range_m=[.01, 10], actual_position_world_m=[0, 0, 0],
                observed_sphere_frustum=dict(minimum_by_plane_m=planes))
    images = [dict(path=f'images/step_{T-1:04d}_{name}.png', sha256='synthetic') for name in sorted(VIEWS)]
    state = dict(step=T-1, env_id=0, capture_kind='scheduled', visual_mode='zero_inclusive',
        state_time_s=T*.1, environment_count=64, views={v: copy.deepcopy(view) for v in VIEWS}, images=images,
        fresh_native_snapshot='before.npz', fresh_native_after_snapshot='after.npz',
        fresh_native_before_sha256='synthetic', fresh_native_after_sha256='synthetic',
        fresh_native_controlled_joint_indices={a: initial[a+'_controlled_joint_indices'].tolist() for a in ARMS},
        fresh_native_all_joint_names={a: initial[a+'_native_joint_names'].tolist() for a in ARMS},
        fresh_native_selected={a: {s: render[a+'_'+s][0].tolist() for s in ('q', 'qd', 'root', 'root_vel')} for a in ARMS},
        arms={a: {s: render[a+'_'+s][0, initial[a+'_controlled_joint_indices']].tolist() for s in ('q', 'qd')} for a in ARMS},
        controller_target={a: final_packet['post_'+a+'_issued_controlled_target'][0].tolist() for a in ARMS},
        actuator_target={a: final_packet['post_'+a+'_applied_controlled_target'][0].tolist() for a in ARMS},
        pending_actuator_targets=[{a: final_packet['post_'+a+'_pending_controlled_targets'][j, 0].tolist() for a in ARMS} for j in range(6)],
        origin_world_m=[0, 0, 0], sphere_centers_world_m=centers[0].tolist(),
        sphere_radii_m=identity['sphere_radii_m'], full_distances_m=d[0].tolist(),
        full_exempt=ex[0].tolist(), full_braking_dmin_m=dm[0].tolist())
    capture = dict(state='state.json', sha256='synthetic', env_id=0, step=T-1, capture_kind='scheduled', images=images)
    objects = {'visual_protocol.json': protocol, 'cell_001.npz': cell,
        'full_row_identity.json': identity, 'guard_metadata.json': dict(mode='zero_inclusive', strict_fifo_steps=6, queue_preemption=False),
        'native_receipts.json': dict(steps=T, boundaries=list(BOUNDARIES.values()),
            chunks=[dict(path='native.npz', start=0, stop=T, sha256='synthetic')]),
        'camera_receipts.json': dict(groups=1, PNG=9, scheduled_groups=1, first_failure_groups=0, receipts=[capture]),
        'forecast_receipts.json': dict(steps=T, rows=9021, envs=64,
            chunks=[dict(path='forecast.npz', start=0, stop=T, sha256='synthetic')]),
        'forecast.npz': dict(measured_d=np.stack([d]*T), exempt=np.stack([np.packbits(ex, axis=-1)]*T), dmin=np.stack([dm]*T)),
        'post_geometry_final.npz': dict(d=d, exempt=ex, dmin=dm, centers=centers),
        'episodes.json': episodes, 'bank_assignment.npz': dict(risk_pair_index=np.full(64, -1)),
        'native_initial.npz': initial, 'native.npz': {k: np.stack([p[k] for p in packets]) for k in packets[0]},
        'state.json': state, 'before.npz': render, 'after.npz': copy.deepcopy(render)}
    png = io.BytesIO()
    pixels = np.zeros((720, 1280, 3), np.uint8); pixels[:, 640:] = 200
    Image.fromarray(pixels).save(png, format='PNG')
    return objects, png.getvalue()


def test_closed_traversal():
    from unittest.mock import patch
    objects, png = synthetic_closed_fixture()

    class MemoryReader:
        def __init__(self, root): self.root, self.hashes = Path(root), {}
        def json(self, name, expected=None): return copy.deepcopy(objects[name])
        def npz(self, name, keys=None, expected=None):
            data = objects[name]
            return {k: v.copy() for k, v in data.items() if keys is None or k in keys}
        def read(self, name, loader, expected=None): return loader(io.BytesIO(png))
        def verify_unchanged(self): pass

    mutations = [
        ('terminal geometry score', 'post_geometry_final.npz', lambda x: x['d'].__setitem__((0, 0), -1e-8)),
        ('episode score', 'episodes.json', lambda x: x[0].__setitem__('violation', True)),
        ('camera center vs cell', 'state.json', lambda x: x['sphere_centers_world_m'][0].__setitem__(0, .001)),
        ('camera receipt frame', 'state.json', lambda x: x.__setitem__('step', 0)),
        ('camera FIFO6', 'state.json', lambda x: x['pending_actuator_targets'][0]['F_L'].__setitem__(0, .8)),
        ('render native changed', 'after.npz', lambda x: x['F_L_q'].__setitem__((0, -1), .125)),
        ('camera clipping', 'state.json', lambda x: x['views']['front'].__setitem__('actual_clipping_range_m', [2., 10.])),
        ('forecast coverage hole', 'forecast_receipts.json', lambda x: x['chunks'][0].__setitem__('start', 1)),
    ]
    caught = []
    with patch.object(Path, 'is_file', lambda p: p.name == 'visual_protocol.json'), patch.dict(globals(), RawReader=MemoryReader):
        result = audit('/synthetic-in-memory-only')
        require(result['frames'] == 9 and result['PNG'] == 9 and result['strict_windows'] == 0,
                'complete synthetic traversal')
        for label, file, change in mutations:
            original = copy.deepcopy(objects[file]); change(objects[file])
            try:
                audit('/synthetic-in-memory-only')
            except InvalidEvidence:
                caught.append(label)
            else:
                raise InvalidEvidence('closed traversal negative escaped ' + label)
            finally:
                objects[file] = original
    return caught


def test_substeps():
    cell, initial, packets = synthetic_packet_fixture()
    context = dict(receipts=dict(capacity=262144), identity=dict(physics_dt_s=.05), views={})
    event = dict(frame=np.array(7), substep=np.array(1), physics_dt=np.array(.05))
    offset = 0
    for arm, width in zip(ARMS, WIDTHS):
        context['views'][arm] = dict(env_ids=list(range(64)),
            sensors=[f'/World/envs/env_{e}/{arm}/registered_hand' for e in range(64)],
            filters=[['/World/ground'] for _ in range(64)])
        event[arm+'_partner_normal'] = np.zeros((64, 1, 3), np.float32)
        event[arm+'_net'] = np.zeros((64, 3), np.float32)
        event[arm+'_normal_counts'] = np.zeros((64, 1), np.int32)
        event[arm+'_normal_starts'] = np.zeros((64, 1), np.int32)
        for key in NATIVE: event[arm+'_'+key] = packets[7]['post_'+arm+'_'+key].copy()
        target = np.zeros_like(initial[arm+'_native_q'])
        target[:, :width] = cell['actuator_target'][7, :, offset:offset+width]
        event[arm+'_native_position_targets'] = target
        offset += width
    for sub in (0, 1):
        event['substep'] = np.array(sub)
        peaks, _, _, _ = check_contact_event(event, 7, sub, cell, initial, packets[7], context)
        equal(peaks, np.zeros((64, 4)), 'zero normal contact diagnostic')
    caught = []
    for key, index, value, sub in [
        ('F_L_native_position_targets', (0, 0), .123, 0),
        ('U_R_native_position_targets', (0, 0), .123, 1),
        ('F_R_native_qd', (0, -1), .125, 1),
        ('U_L_normal_counts', (0, 0), 262144, 1),
        ('F_L_normal_starts', (0, 0), -1, 1)]:
        changed = copy.deepcopy(event); changed['substep'] = np.array(sub); changed[key][index] = value
        try:
            check_contact_event(changed, 7, sub, cell, initial, packets[7], context)
        except InvalidEvidence:
            caught.append('contact ' + key + ' sub' + str(sub))
        else:
            raise InvalidEvidence('contact mutation escaped ' + key)
    event['F_L_partner_normal'][0, 0, 2] = 2.
    raw, outside, _, _ = check_contact_event(event, 7, 1, cell, initial, packets[7], context)
    require(raw[0, 0] == 2 and outside[0, 0] == 2, 'ground normal contact must remain diagnostic')
    context['views']['F_L']['filters'][0][0] = context['views']['F_L']['sensors'][0]
    raw, outside, _, _ = check_contact_event(event, 7, 1, cell, initial, packets[7], context)
    require(raw[0, 0] == 2 and outside[0, 0] == 0, 'same-hand exemption changes only additional diagnostic')
    caught.append('same-hand exemption cannot erase raw normal peak or exempt ground')
    hits = np.zeros((9, 64), bool); hits[2, 4:6] = True; hits[3, 1] = True
    labels = np.full(64, -1)
    right = [dict(capture_kind='first_native_contact', env_id=4, step=2)]
    check_first_images(right, 'first_native_contact', hits, labels, 1)
    for label, wrong in [('missing contact image', []),
                         ('wrong micro-trigger macro frame', [dict(capture_kind='first_native_contact', env_id=1, step=3)]),
                         ('wrong lowest contact env', [dict(capture_kind='first_native_contact', env_id=5, step=2)])]:
        try:
            check_first_images(wrong, 'first_native_contact', hits, labels, len(wrong))
        except InvalidEvidence:
            caught.append(label)
        else:
            raise InvalidEvidence('contact image negative escaped ' + label)
    return caught


def self_test():
    cell, initial, packets = synthetic_packet_fixture()
    for t, packet in enumerate(packets):
        check_packet(packet, cell, t, initial, packets[t - 1] if t else None)
    caught = []
    for key in ('pre_F_L_applied_controlled_target', 'physics_U_L_pending_project_targets',
                'post_F_R_pending_project_targets', 'post_U_R_pending_controlled_targets',
                'physics_F_L_native_q', 'pre_F_R_native_root_velocity', 'post_U_R_issued_controlled_target',
                'pre_F_L_native_q', 'post_U_R_native_q', 'physics_U_L_native_root_xyzw'):
        changed = copy.deepcopy(packets[7])
        changed[key].flat[0 if key == 'post_U_R_native_q' else -1] += .125
        try:
            check_packet(changed, cell, 7, initial, packets[6])
        except InvalidEvidence:
            caught.append(key)
        else:
            raise InvalidEvidence('negative mutation escaped ' + key)
    changed = copy.deepcopy(packets[7]); changed['post_F_L_controlled_joint_indices'][1] = 0
    try:
        check_packet(changed, cell, 7, initial, packets[6])
    except InvalidEvidence:
        caught.append('duplicate controlled index')
    else:
        raise InvalidEvidence('duplicate joint negative escaped')
    labels = np.arange(9021) % 4
    d = np.ones((64, 9021), np.float32); ex = np.zeros(d.shape, bool)
    d[0, 0], d[1, 1], d[2, 2], d[3, 3] = -1e-8, -.005, -.0050001, -1
    ex[3, 3] = True
    margins = raw_margins(d, ex, labels)
    require(margins[0, 0] < 0 and not margins[1, 1] < -.005 and margins[2, 2] < -.005,
            'strict native-float32 scoring thresholds')
    require(margins[3, 3] == 1, 'only original exemption mask applies')
    tiny_change = margins.copy(); tiny_change[0, 0] = 0
    try:
        equal(tiny_change, margins, 'no score epsilon')
    except InvalidEvidence:
        caught.append('tiny negative threshold relaxation')
    else:
        raise InvalidEvidence('score relaxation escaped')
    # Audit incomplete roots without any file creation, using an in-memory reader.
    from unittest.mock import patch
    with patch.object(Path, 'is_file', return_value=True), patch.object(RawReader, 'json', return_value={'status': 'running'}):
        try:
            audit('/not-a-real-root')
        except IncompleteEvidence:
            caught.append('running root cannot pass')
        else:
            raise InvalidEvidence('incomplete root escaped')
    caught.extend(test_closed_traversal())
    caught.extend(test_substeps())
    return dict(status='PASS_CPU_AUDITOR_SELF_TEST_ONLY', complete_synthetic_frames=9,
                negative_controls_caught=caught, real_root_accepted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?')
    parser.add_argument('receipt', nargs='?')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--require-contacts', action='store_true')
    args = parser.parse_args()
    if not args.self_test and (args.root is None or args.receipt is None):
        parser.error('ROOT and NEW_RECEIPT_PATH are required')
    if args.receipt and Path(args.receipt).exists():
        parser.error('receipt exists; use a new path to preserve prior evidence')
    rc = 0
    try:
        result = self_test() if args.self_test else audit(args.root, args.require_contacts)
    except IncompleteEvidence as error:
        result, rc = dict(status='INCOMPLETE_NOT_ACCEPTED', error=str(error), root=args.root), 2
    except Exception as error:
        result, rc = dict(status='FAIL_RAW_EVIDENCE', error=type(error).__name__ + ': ' + str(error), root=args.root), 1
    result.update(utc=datetime.now(timezone.utc).isoformat(), auditor_sha256=digest(__file__),
                  physical_safety_certified=False, queued_future_status='UNKNOWN', exitcode=rc)
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.receipt:
        with Path(args.receipt).open('x') as stream:
            stream.write(payload)
    print(payload, end='', flush=True)
    return rc


if __name__ == '__main__':
    sys.exit(main())
