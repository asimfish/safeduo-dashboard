#!/usr/bin/env python3
"""V7 CPU oracle: two frozen plans, conditional raw with immutable resource failure.

Usage (use the installed safeduo Python, with -B):
  check_native_visible_evidence_v7.py --supervise-v7
  check_native_visible_evidence_v7.py --producer-root /absolute/registered/root
  check_native_visible_evidence_v7.py --self-test

Fork of frozen bfb oracle, which is never modified. A live or
unidentified producer yields a PENDING artifact, never a terminal verdict.
Exit 0 = evidence gate passed (NOT physical safety), 2 = closed evidence blocked,
3 = pending producer closure. No simulator, torch, renderer, or network is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import re
import sys
import time
import subprocess
import signal
import zipfile
import fcntl

import numpy as np
from PIL import Image

H = Path('/home/liyufeng/safeduo/artifacts/safety_visible_point_random_20261009_1622')
R = Path('/mnt/nas/data/lyf/double_hand/safety_visible_point_random_20261009_1622')
SSD = H/'native_execution'
BASE_ORACLE_SHA256 = 'bfb856bb698f7d8e7c16bc7b18f09f8ae5590f4ac6b2d6aebe93ffc4f5e649c0'
V4_CAMERA_SHA256 = 'c86ce750e43d6a2f22d033a39887da6263eb0c5f9973f1b44b3989c1a7a79d25'
V5_PLAN_SHA256 = 'd6fc2bf04d93e519e2fe6ddd35cdcb355be645c0b9bff0d55ffac9dabc0d32e5'
V6_ORACLE_SHA256 = '37ef81ab81824196f552360ac46f5d9507ebdc8ca262fcdb49b965ee67acf8a3'
OUTPUT_REPORT_NAMES = {m:'ASTRA_NATIVE_EVIDENCE_'+m+'_v7.json' for m in ('raw','multirow')}
V5_ORACLE_SHA256 = 'b3e4ef13c64919c35c17ff3fff8e771f21f44e34ceb7e212d783c8ade19c1436'
V6_PLAN_SHA256 = 'c3e060774848f32667264a3d9fd2ace873fe9fa3ee734a439bc1ae7ef338a96a'
RAW_EXECUTION_SHA256 = 'f322350ea417eefe4c995668970873efe9f18c4d23ac4aba87ab5b40d5185256'
ROOTS = dict(raw=str(SSD/'full_v5/raw'), multirow=str(SSD/'full_v6/multirow'))
PLAN_FILES = dict(raw=str(H/'FULL_PLAN_V5.json'), multirow=str(H/'CANDIDATE_PLAN_V6.json'))
PLAN_SHAS = dict(raw=V5_PLAN_SHA256, multirow=V6_PLAN_SHA256)
EXECUTION_FILES = dict(raw=str(H/'visible_point_full_v5_execution.json'),
                       multirow=str(H/'visible_point_candidate_v6_execution.json'))
REGISTRATION_SHAS = {'ACTIVE_ATTEMPT_V6.json':'1dd86a92cd0418eea88b8190d538653980fd3ac64a3f59d1223d10c39289084c',
                     'NATIVE_STORAGE_REGISTRATION_V6.json':'d9d194db191b962d7cc9dfa7e7e290a83a9ed71688ae13fef5ba2d5787b6add8'}
RESOURCE_LIMITS = dict(min_initial_free_MiB=11000, max_owned_process_MiB=12500,
    min_runtime_free_MiB=2500, min_initial_host_available_MiB=35000,
    min_runtime_host_available_MiB=12000, max_owned_host_RSS_MiB=46000,
    min_SSD_free_bytes=16106127360)
EXECUTED_SOURCE_BYTES = Path(__file__).read_bytes()
EXECUTED_SOURCE_SHA256 = hashlib.sha256(EXECUTED_SOURCE_BYTES).hexdigest()
CAMERA_SOURCE, TOKEN_PREFIX, MAX_ATTEMPTS = 'qualified_views_v4.py', 'qv4_hand_', 24
ARMS = ('F_L', 'F_R', 'U_L', 'U_R')
FIELDS = ('native_q', 'native_qd', 'native_root_xyzw', 'native_root_velocity',
          'native_link_transforms_xyzw', 'native_position_targets')
CHUNK_FIELDS = tuple(x for x in FIELDS if x != 'native_link_transforms_xyzw')
CLOCK = ('simulation_time_s', 'simulation_time_step_index')
STREAMS = ('instance_segmentation_fast', 'semantic_segmentation')
WIDTH, HEIGHT, GATE = 1280, 720, .1
# Isaac's recorded native clock uses float32(1/120); the force API receives the
# frozen configured .008333. They are different quantities and both are checked.
NATIVE_TICK_SECONDS = float(np.float32(1./120.))


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def validate_attempt_identity(arm, number):
    require(arm in ARMS and type(number) is int and 1 <= number <= MAX_ATTEMPTS,
            'invalid arm/attempt identity; v4 bound is 24')


def registration():
    """Do not reinterpret the failed v5 execution or relax any registered gate."""
    paths = [H/'ACTIVE_ATTEMPT_V6.json', H/'NATIVE_STORAGE_REGISTRATION_V6.json',
             H/'FULL_PLAN_V5.json', H/'CANDIDATE_PLAN_V6.json']
    raw = [p.read_bytes() for p in paths]
    active, storage, raw_plan, candidate_plan = map(json.loads, raw)
    require(all(sha(raw[i]) == REGISTRATION_SHAS[paths[i].name] for i in (0,1)), 'frozen V6 registration changed')
    require(active['primary_roots'] == ROOTS and active['primary_executions'] == EXECUTION_FILES
            and active['primary_plans'] == PLAN_FILES, 'mixed-source V6 registration mismatch')
    require(active['camera_source'] == CAMERA_SOURCE and active['reference_resource_policy_pass'] is False
            and active['reference_original_execution_verdict'] == 'failed', 'raw resource failure reclassified')
    require(active['resource_thresholds_relaxed'] is False and active['science_gate_thresholds_relaxed'] is False
            and active['new_independent_initial_states_added_by_reusing_raw'] == 0, 'changed V6 scope')
    for i, (method, plan) in enumerate((('raw',raw_plan),('multirow',candidate_plan)), 2):
        require(sha(raw[i]) == PLAN_SHAS[method], 'frozen plan changed:'+method)
        matches = [j for j in plan['jobs'] if j['id'] == method and j['out'] == ROOTS[method]]
        require(len(matches) == 1 and matches[0]['steps'] == 480, 'source plan/root/denominator mismatch')
        require(plan['resource'] == RESOURCE_LIMITS, 'resource thresholds changed')
        require(plan['sources'][str(H/CAMERA_SOURCE)] == V4_CAMERA_SHA256, 'camera source not frozen in source plan')
        suffix = '_v5' if method == 'raw' else '_v6'
        require(storage['producer_roots'][method+suffix] == ROOTS[method],
                'storage registration differs from active producer')
        require(active['astra_reports'][method] == 'ASTRA_NATIVE_EVIDENCE_'+method+'_v6.json',
                'active report naming mismatch')
    require(sha(Path(EXECUTION_FILES['raw']).read_bytes()) == RAW_EXECUTION_SHA256,
            'original resource-failed raw execution changed; never reclassify it')
    require(sha((H/'check_native_visible_evidence_v6.py').read_bytes()) == V6_ORACLE_SHA256, 'frozen v6 source changed')
    return active, {str(p): sha(data) for p, data in zip(paths, raw)}


def select_root(name, active):
    require(isinstance(name, str) and bool(name), 'root is required')
    root = Path(name)
    require(root.is_absolute() and str(root) in active['primary_roots'].values(),
            'root must be an absolute exact ACTIVE_ATTEMPT_V6 primary root')
    require(root.resolve() == root, 'registered producer root cannot resolve elsewhere')
    return root


def pair_closures(active):
    closures = {m: producer_closure(Path(p)) for m, p in active['primary_roots'].items()}
    ready = all(c['closed'] and c.get('plan_sha256') == PLAN_SHAS[m] and
                c.get('execution') == active['primary_executions'][m] for m,c in closures.items())
    return ready, closures


def resource_assessment(job, supervisor_status, limits, initial_gate):
    require(limits == RESOURCE_LIMITS, 'registered resource thresholds changed')
    violations = []
    if job.get('actual_exit') != 0:
        violations.append('producer_actual_exit_not_zero')
    if job.get('resource_abort') is not None:
        violations.append('recorded_resource_abort:'+str(job['resource_abort']))
    if supervisor_status != 'complete' or job.get('status') != 'complete':
        violations.append('supervisor_or_job_not_complete')
    samples = job.get('resource_samples', [])
    require(samples, 'missing original resource monitor samples')
    metrics = dict(max_owned_host_RSS_MiB=max(s['owned_host_RSS_MiB'] for s in samples),
        min_host_available_MiB=min(s['host_available_MiB'] for s in samples),
        max_owned_GPU_MiB=max(s['owned_GPU_MiB'] for s in samples),
        min_GPU_free_MiB=min(s['GPU_free_MiB'] for s in samples),
        first_host_available_MiB=samples[0]['host_available_MiB'])
    comparisons = [('max_owned_host_RSS_MiB','max_owned_host_RSS_MiB','max'),
        ('min_host_available_MiB','min_runtime_host_available_MiB','min'),
        ('max_owned_GPU_MiB','max_owned_process_MiB','max'),
        ('min_GPU_free_MiB','min_runtime_free_MiB','min'),
        ('first_host_available_MiB','min_initial_host_available_MiB','min')]
    for observed, threshold, direction in comparisons:
        if (metrics[observed] > limits[threshold] if direction == 'max' else metrics[observed] < limits[threshold]):
            violations.append(observed+':outside_registered_bound')
    if any(s['probe_actual_exit'] != 0 or s['free_probe_actual_exit'] != 0 for s in samples):
        violations.append('resource_monitor_probe_failed')
    if any(s['selected_gpu'] != 0 for s in samples):
        violations.append('unregistered_GPU')
    require(initial_gate['selected_gpu'] == 0 and initial_gate['required_free_MiB'] == 11000,
            'initial GPU gate not registered')
    gpu_rows = {int(x.split(',')[0]):int(x.split(',')[1]) for x in initial_gate['stdout'].splitlines()}
    if initial_gate['actual_exit'] != 0 or gpu_rows[0] < limits['min_initial_free_MiB']:
        violations.append('initial_GPU_gate_failed')
    return dict(verdict='FAILED' if violations else 'PASS', violations=violations,
        original_resource_abort=job.get('resource_abort'), observed_samples=len(samples),
        observed_metrics=metrics, registered_limits=limits,
        scope='Independent recorded GPU/host sample threshold checks plus original supervisor outcome. '
              'Initial host/SSD and runtime SSD checks were enforced by frozen supervisor; '
              'their raw numeric measurements were not persisted, so cannot be replayed numerically.')


def preserved_raw_failure(raw_bytes, doc, job):
    require(sha(raw_bytes) == RAW_EXECUTION_SHA256, 'original raw execution bytes changed')
    require(doc['status'] == 'failed' and job['actual_exit'] == 0 and job['status'] == 'child_reaped'
            and job['resource_abort'] == 'owned host RSS exceeds registered bound'
            and job['stop_signal'] == 'SIGTERM', 'original raw resource failure reclassified')


def evidence_verdict(method, evidence_ok, resource_verdict):
    require(method in ROOTS and resource_verdict in ('PASS','FAILED','UNVERIFIED'), 'invalid verdict inputs')
    eligible = evidence_ok and ((method == 'raw' and resource_verdict == 'FAILED') or
                                (method == 'multirow' and resource_verdict == 'PASS'))
    if not evidence_ok:
        label = 'BLOCKED'
    elif resource_verdict == 'PASS':
        label = 'PASS_EVIDENCE_ONLY'
    elif resource_verdict == 'FAILED':
        label = 'PASS_EVIDENCE_ONLY_WITH_RESOURCE_FAILURE'
    else:
        label = 'PASS_DATA_RESOURCE_UNVERIFIED'
    final_label = ('PASS_EVIDENCE_ONLY' if resource_verdict == 'PASS' else 'BLOCKED_RESOURCE_FAILURE') if evidence_ok else 'BLOCKED'
    return dict(final_verdict=final_label, evidence_only_verdict=label,
        data_integrity_verdict='PASS' if evidence_ok else 'FAIL',
        registered_resource_verdict=resource_verdict,
        resource_policy_verdict=resource_verdict, conditional_comparison_eligible=eligible,
        reference_resource_failure_retained=True, registered_full_run_acceptance='BLOCKED',
        physical_safety_certified=False)


def compare_npy_streams(exported, recorded, spec, count):
    """Bounded buffers, exact original bytes, no numeric tolerance or conversion."""
    headers = []
    for stream in (exported, recorded):
        version = np.lib.format.read_magic(stream)
        shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
        headers.append((shape, fortran, dtype))
    require(headers[0] == headers[1], 'export/recorded NPY dtype/shape/order differs')
    shape, fortran, dtype = headers[0]
    require(shape == (count, *spec['shape']) and dtype.str == spec['dtype'] and not dtype.hasobject,
            'export does not match closed field schema')
    expected = int(np.prod(shape, dtype=np.int64))*dtype.itemsize
    require(expected == spec['bytes'], 'closed field byte denominator differs')
    seen, h = 0, hashlib.sha256()
    while True:
        a, b = exported.read(8*1024*1024), recorded.read(8*1024*1024)
        require(a == b, 'export vs original mmap payload bytes differ')
        if not a:
            break
        seen += len(a); h.update(a)
    require(seen == expected, 'truncated or extra NPY payload')
    return dict(dtype=dtype.str, shape=list(shape), payload_bytes=seen, payload_sha256=h.hexdigest(),
                exported_vs_original_mmap_byte_exact=True)


def validate_overlay(setup, inventory):
    audit = setup['hand_descendant_instanceability_audit']
    require(setup['schema'] == 'safeduo.qualified_semantics.v4', 'wrong semantic setup version')
    require(audit['actual_paths_unchanged'] is True and audit['remaining_instance_paths'] == []
            and audit['remaining_proxy_paths'] == [] and audit['max_rounds'] == 32,
            'incomplete nested deinstance record')
    paths = audit['descendant_paths']
    path_set = set(paths)
    require(paths == sorted(path_set), 'duplicate or unordered descendant paths')
    hands = {p for p, m in inventory.items() if m['hand']}
    for path in paths:
        require(path not in hands and ancestor(path, hands) is not None,
                'deinstance descendant lacks exact native hand ancestor')
    overrides = []
    require(len(audit['rounds']) <= 32, 'deinstance round bound exceeded')
    for i, row in enumerate(audit['rounds']):
        require(row['round'] == i, 'deinstance round ordering mismatch')
        earlier_instance_write = False
        for item in row['overrides']:
            # Producer selects all candidates at round start, then refetches each
            # prim at write time. An earlier env0 edit can already deinstance its
            # inherited clones. Prior IsInstance=True is NOT editability.
            require(all(type(item[k]) is bool for k in ('is_instance','is_instance_proxy',
                    'is_instanceable','after_is_instance')), 'nonboolean deinstance metadata')
            require(item['path'] in path_set and item['is_instance_proxy'] is False and
                    item['after_is_instance'] is False, 'uneditable or ineffective deinstance override')
            require(not item['is_instance'] or item['is_instanceable'], 'inconsistent prior instanceability')
            require(item['is_instance'] or earlier_instance_write,
                    'noninstance write has no preceding composition mutation in this round')
            earlier_instance_write |= item['is_instance']
            overrides.append(item['path'])
    require(overrides == setup['hand_descendant_instanceability_overrides'] and
            len(set(overrides)) == len(overrides), 'deinstance override inventory mismatch')
    for mesh, owner in audit['render_mesh_native_hand_owners'].items():
        require(mesh in path_set and owner == ancestor(mesh, hands), 'render mesh/native hand owner mismatch')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def equal_bits(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def exact_snapshot(a, b):
    require(set(a) == set(b), 'native snapshot field set changed')
    changed = [k for k in a if not equal_bits(a[k], b[k])]
    require(not changed, 'native dtype/shape/bytes changed: ' + ','.join(changed))


def bind_microstep(baseline, native, event):
    for key in [a+'_'+f for a in ARMS for f in CHUNK_FIELDS] + list(CLOCK):
        require(equal_bits(baseline[key], native[key][event]), 'camera vs all64 exact native microstep mismatch:'+key)


def first_exceedances(peak):
    return {lane:int(np.flatnonzero(peak[:,lane] > GATE)[0]) for lane in range(64) if (peak[:,lane] > GATE).any()}


def hand_path(path):
    return ('f2_hand/' in path or any(x in path.rsplit('/', 1)[-1]
            for x in ('thumb', 'index', 'middle', 'ring', 'little', 'pinky', 'wrist_3_link')))


def ancestor(path, inventory):
    if not isinstance(path, str) or not path.startswith('/'):
        return None
    while path:
        if path in inventory:
            return path
        path = path.rstrip('/').rsplit('/', 1)[0]
    return None


def token_inventory(inventory, mode, prefix):
    result = {}
    for path, meta in inventory.items():
        if not meta['hand']:
            continue
        token = prefix + (sha(path.encode()) if mode == 'sha256' else path.lstrip('/'))
        if mode == 'unique-casefold':
            token = token.casefold()
        require(token not in result, 'semantic token collision between native rigid paths')
        result[token] = path
    return result


def mask_oracle(outputs, info, inventory, slot, arm, mode='exact', prefix='qv2_hand/'):
    """Require per-pixel agreement on the SAME native rigid link, not just arm.

    Unique casefold/digest tokens can be requested for a new audit. Instance
    paths remain case sensitive and clone-specific in EVERY mode.
    """
    errors, records, arrays, owners = [], {}, {}, {}
    desired = {p for p, m in inventory.items() if m['hand'] and
               (m['env_id'], m['arm']) == (slot, arm)}
    allpaths = sorted(inventory)
    codes = {p: i + 1 for i, p in enumerate(allpaths)}
    token_map = token_inventory(inventory, mode, prefix)
    folded = {}
    for p in allpaths:
        folded.setdefault(p.casefold(), []).append(p)
    for stream in STREAMS:
        raw = np.asarray(outputs[stream])
        require(raw.shape == (HEIGHT, WIDTH, 1) and raw.dtype.kind in 'iu' and
                raw.dtype.itemsize == 4, stream + ': original int32/uint32 HxWx1 required')
        ids = raw[..., 0]
        if ids.dtype.kind == 'i':
            ids = ids.view(np.dtype(ids.dtype.str.replace('i', 'u')))
        arrays[stream] = ids
        labels = info[stream]['idToLabels']
        require(isinstance(labels, dict), stream + ': idToLabels missing')
        converted = {}
        for k, v in labels.items():
            require(re.fullmatch(r'[0-9]+', str(k)) is not None, 'noninteger segmentation ID')
            n = int(k)
            require(n <= 2**32-1 and n not in converted, 'duplicate/out of range segmentation ID')
            converted[n] = v
        present, counts = np.unique(ids, return_counts=True)
        unknown = [int(x) for x in present if int(x) not in converted]
        if unknown:
            errors.append(stream + ':unknown_ids:' + str(unknown))
        selected, mapping, mismatches = {}, [], []
        for idx, label in converted.items():
            paths = []
            if stream == STREAMS[0]:
                if isinstance(label, str) and label.startswith('/'):
                    p = ancestor(label, inventory)
                    if p:
                        paths = [p]
                elif label not in ('BACKGROUND', 'UNLABELLED'):
                    errors.append('instance_label_not_native_path:' + str(idx))
            else:
                if not isinstance(label, dict):
                    errors.append('semantic_label_not_type_dict:' + str(idx))
                    continue
                value = label.get('class', '')
                if not isinstance(value, str):
                    errors.append('semantic_class_not_string:' + str(idx))
                    continue
                tokens = [t.strip() for t in value.split(',') if t.strip().startswith(prefix)]
                if mode == 'unique-casefold':
                    tokens = [t.casefold() for t in tokens]
                paths = [token_map[t] for t in tokens if t in token_map]
                bad = [t for t in tokens if t not in token_map]
                for token in bad:
                    p = '/' + token[len(prefix):]
                    candidates = folded.get(p.casefold(), [])
                    mismatches.append(dict(id=idx, token=p, unique_casefold_native_path=
                                           candidates[0] if len(candidates) == 1 else None))
                if bad:
                    errors.append('semantic_exact_native_path_unverified:' + str(idx))
                    paths = []
                if len(set(paths)) > 1:
                    errors.append('semantic_ambiguous_native_rigid:' + str(idx))
                    paths = []
            if idx in present and any(inventory[p]['env_id'] != slot for p in paths):
                errors.append(stream + ':visible_foreign_clone:' + str(idx))
            accepted = [p for p in paths if p in desired]
            if accepted:
                selected[idx] = accepted
            if int(idx) in present:
                mapping.append(dict(id=idx, label=label, native_paths=paths,
                                    desired_hand_paths=accepted))
        mask = np.isin(ids, list(selected))
        records[stream] = dict(observed_id_pixel_counts={str(int(k)): int(v) for k, v in zip(present, counts)},
                               unknown_ids=unknown, hand_pixels=int(mask.sum()),
                               present_mapping=mapping, exact_path_mismatches=mismatches)
        owners[stream] = selected
    instance, semantic = arrays[STREAMS[0]], arrays[STREAMS[1]]
    instance_codes = np.zeros((HEIGHT, WIDTH), np.int32)
    for idx, paths in owners[STREAMS[0]].items():
        instance_codes[instance == idx] = codes[paths[0]]
    exact = np.zeros((HEIGHT, WIDTH), bool)
    arm_intersection = (instance_codes != 0) & np.isin(semantic, list(owners[STREAMS[1]]))
    for idx, paths in owners[STREAMS[1]].items():
        exact |= (semantic == idx) & np.isin(instance_codes, [codes[p] for p in paths])
    count = int(exact.sum())
    if count <= 200:
        errors.append('same_native_hand_pixel_intersection_not_gt_200')
    diagnostic_count = 0
    for row in records[STREAMS[1]]['exact_path_mismatches']:
        p = row['unique_casefold_native_path']
        if p in desired:
            diagnostic_count += int(np.count_nonzero((semantic == row['id']) & (instance_codes == codes[p])))
    return dict(pass_gate=not errors, errors=errors, observed_hand_pixels=count,
                same_arm_pixel_intersection=int(arm_intersection.sum()),
                casefold_only_same_link_pixels_diagnostic=diagnostic_count,
                semantic_mode=mode, token_map_entries=len(token_map), streams=records)


def frustum(points, camera):
    world = np.asarray(camera['actual_camera_to_world_row_matrix'], dtype=np.float64)
    k = np.asarray(camera['actual_intrinsic_matrix'], dtype=np.float64)
    near, far = camera['actual_clipping_range_m']
    require(world.shape == (4, 4) and k.shape == (3, 3) and np.isfinite(world).all()
            and np.isfinite(k).all() and 0 < near < far, 'invalid camera matrix/clipping')
    require(np.allclose(world[:, 3], [0, 0, 0, 1], atol=1e-9, rtol=0) and
            np.allclose(world[:3, :3] @ world[:3, :3].T, np.eye(3), atol=1e-6, rtol=0),
            'camera row transform is not rigid')
    require(np.linalg.det(world[:3, :3]) > 0, 'camera transform reflected')
    optics = camera['actual_usd_optics']
    require(optics['horizontal_aperture_offset'] == 0 and optics['vertical_aperture_offset'] == 0,
            'unsupported non-centered USD optics')
    expected = np.array([[WIDTH*optics['focal_length']/optics['horizontal_aperture'], 0, WIDTH/2],
                         [0, HEIGHT*optics['focal_length']/optics['vertical_aperture'], HEIGHT/2], [0, 0, 1]])
    require(np.allclose(k, expected, atol=1e-10, rtol=1e-12), 'K does not reconstruct from USD optics')
    require(np.allclose(k, camera['sdk_intrinsic_matrix'], atol=2e-4, rtol=1e-6), 'SDK K mismatch')
    require(camera['image_size'] == [WIDTH, HEIGHT], 'camera image size mismatch')
    # Independently transform native centers to camera space and evaluate each
    # inward plane as a normalized signed distance, subtracting the .06m radius.
    local = np.column_stack([points, np.ones(len(points))]) @ np.linalg.inv(world)
    x, y, z = local[:, :3].T
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    margins = np.column_stack([(fx*x-cx*z)/np.hypot(fx, cx),
        (-fx*x-(WIDTH-cx)*z)/np.hypot(fx, WIDTH-cx),
        (-fy*y-cy*z)/np.hypot(fy, cy), (fy*y-(HEIGHT-cy)*z)/np.hypot(fy, HEIGHT-cy),
        -z-near, z+far]) - .06
    center = (points.min(0) + points.max(0)) / 2
    direction = world[3, :3] - center
    require(np.linalg.norm(direction[:2]) > 1e-9, 'azimuth undefined')
    return dict(pass_gate=bool(np.all(margins >= .02)), native_hand_links=len(points),
                minimum_plane_margin_m=float(margins.min()),
                minimum_by_plane_m=margins.min(0).tolist(), radius_m=.06, clearance_m=.02,
                azimuth_deg=float(np.degrees(np.arctan2(direction[1], direction[0])) % 360),
                scope='native link sphere framing only; no mesh visibility or physical safety proof')


def scalar_points(counts, starts, point_ids, sensor, partner, forces, capacity):
    require(counts.ndim == 2 and counts.dtype.kind in 'iu' and starts.shape == counts.shape
            and starts.dtype.kind in 'iu', 'invalid counts/starts dtype or shape')
    c, st = counts.astype(np.int64), starts.astype(np.int64)
    require((c >= 0).all() and (st >= 0).all() and (st + c <= capacity).all()
            and ((c == 0) | (st+c < capacity)).all()
            and c.sum() < capacity, 'negative counts/starts or capacity exhausted')
    n = int(c.sum())
    for name, value in [('point', point_ids), ('sensor', sensor), ('partner', partner)]:
        require(value.dtype == np.dtype('int64') and value.shape == (n,), name + ' index layout')
    require(forces.shape == (n, 1) and forces.dtype.kind == 'f' and np.isfinite(forces).all(), 'raw normal forces')
    require(((sensor >= 0) & (sensor < c.shape[0])).all() and
            ((partner >= 0) & (partner < c.shape[1])).all(), 'point owner index outside matrix')
    require(len(np.unique(point_ids)) == n, 'duplicate native point index')
    require(((point_ids >= st[sensor, partner]) & (point_ids < st[sensor, partner] + c[sensor, partner])).all(),
            'point index outside original owner interval')
    flat = sensor * c.shape[1] + partner
    reconstructed = np.bincount(flat, minlength=c.size).reshape(c.shape)
    require(np.array_equal(reconstructed, c), 'raw valid indices do not reconstruct counts')
    result = np.bincount(flat, weights=np.abs(forces[:, 0].astype(np.float64)), minlength=c.size).astype(np.float64, copy=False).reshape(c.shape)
    require(np.isfinite(result).all(), 'nonfinite sumabs')
    return result


class Audit:
    def __init__(self, root):
        self.root = root
        self.ledger = {}
        self.errors = []
        self.checks = Counter()
        self.array_cache = {}
        self.body = {}
        self.joints = {}
        self.identity = None

    def issue(self, category, where, message):
        self.errors.append(dict(category=category, where=str(where), message=str(message)))

    def check(self, ok, category, where, message):
        self.checks[category] += 1
        if not bool(ok):
            self.issue(category, where, message)
        return bool(ok)

    def data(self, path):
        path = Path(path)
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'file changed during read: ' + str(path))
        row = dict(sha256=sha(data), size=after.st_size, mtime_ns=after.st_mtime_ns)
        old = self.ledger.get(str(path))
        require(old is None or old == row, 'file changed across reads: ' + str(path))
        self.ledger[str(path)] = row
        return data

    def js(self, path):
        return json.loads(self.data(path))

    def stream_digest(self, path):
        before = path.stat(); h = hashlib.sha256()
        with path.open('rb') as f:
            for data in iter(lambda:f.read(8*1024*1024), b''):
                h.update(data)
        after = path.stat()
        require((before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns), 'streamed input changed')
        row = dict(sha256=h.hexdigest(),size=after.st_size,mtime_ns=after.st_mtime_ns)
        require(self.ledger.get(str(path),row) == row, 'streamed input changed across reads')
        self.ledger[str(path)] = row
        return row['sha256']

    def complete_exports(self):
        exports = []
        for directory, name in [('response_fields','response_stream.npz'),('dynamics_fields','dynamics_stream.npz')]:
            closed = self.js(self.root/directory/'closed.json')
            require(closed['status'] == 'complete' and closed['filled'] == closed['registered'] == 480,
                    'original field record denominator incomplete')
            path = self.root/name
            require(closed['output'] == str(path), 'closed export path differs')
            schema, fields = closed['schema'], []
            before = path.stat()
            with zipfile.ZipFile(path) as z:
                require(len(z.namelist()) == len(schema) and set(z.namelist()) == {k+'.npy' for k in schema},
                        'missing/duplicate/extra NPZ members')
                for key, spec in schema.items():
                    require(re.fullmatch(r'[A-Za-z0-9_]+',key), 'unsafe original field name')
                    original = self.root/directory/(key+'.npy')
                    original_before = original.stat()
                    with z.open(key+'.npy') as exported, original.open('rb') as recorded:
                        row = compare_npy_streams(exported,recorded,spec,480)
                    row.update(field=key,zip_crc32=z.getinfo(key+'.npy').CRC,
                               original_file_sha256=self.stream_digest(original))
                    require((original.stat().st_size,original.stat().st_mtime_ns) ==
                            (original_before.st_size,original_before.st_mtime_ns), 'original field changed while comparing/hashing')
                    fields.append(row)
                    self.checks['complete_export_field_original_bytes_and_CRC'] += 1
            require((path.stat().st_size,path.stat().st_mtime_ns) == (before.st_size,before.st_mtime_ns),
                    'export archive changed while validating')
            digest = self.stream_digest(path)
            require((path.stat().st_size,path.stat().st_mtime_ns) == (before.st_size,before.st_mtime_ns),
                    'export archive changed between validation and digest')
            exports.append(dict(path=str(path),sha256=digest,fields=fields,
                                complete_payload_CRC_checked=True,controls=480))
            print('COMPLETE_EXPORT',name,'fields',len(fields),flush=True)
        return exports

    def ref(self, record):
        require(isinstance(record, dict) and set(record) >= {'path', 'sha256'}, 'missing artifact reference')
        path = (self.root / record['path']).resolve()
        require(path.is_relative_to(self.root), 'reference escapes root')
        data = self.data(path)
        require(sha(data) == record['sha256'], 'artifact SHA mismatch: ' + str(path))
        return data

    def npz(self, path_or_ref):
        data = self.ref(path_or_ref) if isinstance(path_or_ref, dict) else self.data(path_or_ref)
        digest = sha(data)
        if digest in self.array_cache:
            return self.array_cache[digest]
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            result = {k: z[k] for k in z.files}
        # Identical held snapshots recur thousands of times. Cache only small
        # states, by content hash after reading EACH file and verifying its SHA.
        if sum(v.nbytes for v in result.values()) < 1000000 and len(self.array_cache) < 128:
            self.array_cache[digest] = result
        return result

    def snapshot(self, ref):
        state = self.npz(ref)
        expected = {'environment_origins', *CLOCK, *(a+'_'+f for a in ARMS for f in FIELDS)}
        require(set(state) == expected, 'missing/extra native snapshot fields')
        require(state['environment_origins'].shape == (64, 3), 'origins must cover all64')
        require(state[CLOCK[0]].shape == () and state[CLOCK[0]].dtype.kind == 'f' and
                state[CLOCK[1]].shape == () and state[CLOCK[1]].dtype.kind in 'iu', 'native scalar clock layout')
        for a in ARMS:
            for f in FIELDS:
                shape = (64, self.joints[a])
                if f == 'native_root_xyzw':
                    shape = (64, 7)
                elif f == 'native_root_velocity':
                    shape = (64, 6)
                elif f == 'native_link_transforms_xyzw':
                    shape = (64, len(self.body[a]), 7)
                require(state[a+'_'+f].shape == shape, a+'_'+f+': incomplete native layout')
        require(all(x.dtype.kind in 'fiu' and np.isfinite(x).all() for x in state.values()), 'nonfinite/non-numeric native state')
        self.checks['native_snapshot_complete_all64'] += 1
        return state

    def load_inventory(self, ref):
        inventory = json.loads(self.ref(ref))
        expected = {}
        sensors = {p for s in self.identity['views'] for p in s['sensors']}
        for path, m in inventory.items():
            match = re.fullmatch(r'/World/envs/env_(\d+)/(F_L|F_R|U_L|U_R)/(.+)', path)
            require(match is not None, 'invalid native rigid path')
            slot, arm = int(match[1]), match[2]
            idx = m['body_index']
            require(0 <= slot < 64 and type(idx) is int and 0 <= idx < len(self.body[arm]), 'native path index out of range')
            require((m['env_id'], m['arm'], m['body_name'], m['hand']) ==
                    (slot, arm, self.body[arm][idx], hand_path(path)), 'native path metadata mismatch')
            require(path.rsplit('/', 1)[-1] == self.body[arm][idx], 'native path/body name mismatch')
            key = (slot, arm, idx)
            require(key not in expected, 'duplicate native index')
            expected[key] = path
        require(set(inventory) == sensors, 'native mapping and all rigid contact sensors are not bijective')
        require(len(expected) == 64 * sum(len(x) for x in self.body.values()), 'missing native link inventory')
        return inventory

    def setup(self):
        body = self.npz(self.root / 'body_identity.npz')
        parameters = self.npz(self.root / 'resolved_native_parameters.npz')
        for a in ARMS:
            self.body[a] = body[a].tolist()
            self.joints[a] = len(parameters[a+'_native_joint_names'])
        self.identity = self.js(self.root / 'native_contact_identity.json')
        require(self.identity['environment_count'] == 64, 'contact identity all64 required')
        views = self.identity['views']
        require(len(views) == 4 and {x['arm'] for x in views} == set(ARMS), 'four contact views required')
        owners = {x['rigid_owner'] for x in self.identity['inventory'] if x['collision_enabled'] and
                  x['rigid_owner'] and any('/'+a+'/' in x['rigid_owner'] for a in ARMS)}
        require(len(owners) == 82, 'enabled arm collider owner count is not 82')
        partners = self.identity['partners_env0']
        require(len(partners) == len(set(partners)) and owners <= set(partners), 'partner inventory missing enabled owner')
        for x in self.identity['inventory']:
            if x['collision_enabled']:
                require((x['rigid_owner'] or x['path']) in partners, 'enabled collider absent from partner filters')
        observed = set()
        for spec in views:
            sensors, filters, ids = spec['sensors'], spec['filters'], spec['env_ids']
            a = spec['arm']
            require(len(sensors) == len(filters) == len(ids) == 64 * len(self.body[a]), 'contact view shape incomplete')
            require(len(set(sensors)) == len(sensors), 'duplicate native contact sensor')
            require(spec['capacity'] == 262144, 'unregistered point capacity')
            for sensor, row, slot in zip(sensors, filters, ids):
                require(type(slot) is int and 0 <= slot < 64 and sensor.startswith(f'/World/envs/env_{slot}/{a}/'), 'sensor owner/slot mismatch')
                require(row == [p.replace('/env_0/', f'/env_{slot}/') for p in partners], 'partner order/clone mismatch')
                observed.add(sensor)
        expected = {p.replace('/env_0/', f'/env_{slot}/') for p in owners for slot in range(64)}
        require(observed == expected, 'all82 enabled owners not covered exactly in all64')
        self.checks['enabled_owner_coverage_82x64'] += 1
        return dict(enabled_owners_per_environment=82, environments=64, sensors=len(observed),
                    exact_clone_partner_filters=True, identity_scope='env0 composed collider inventory plus all64 actual native sensor/filter paths')

    def forces(self, steps, constructor=False):
        receipt = self.js(self.root / 'point_contact_receipts.json')
        require(receipt['control_steps'] == steps and receipt['physics_events'] == steps*2, 'point receipt window mismatch')
        require(receipt['identity_sha256'] == self.ledger[str(self.root / 'native_contact_identity.json')]['sha256'], 'contact identity SHA mismatch')
        for field, name in [('point_recorder_sha256', 'native_all_body_point_contacts.py'),
                            ('point_helper_sha256', 'point_contact_data.py')]:
            require(receipt[field] == sha(self.data(H/name)), 'point recorder/helper source SHA mismatch')
        require(self.identity['source_sha256'] == sha(self.data(H/'native_all_body_contacts.py')), 'contact source SHA mismatch')
        peaks, clocks, indices, frames, subs, states = [], [], [], [], [], {}
        points_total, cursor, maximum = 0, 0, None
        chunk_paths = []
        for chunk in receipt['chunks']:
            require(chunk['start'] == cursor and cursor < chunk['stop'] <= steps*2, 'chunk coverage not contiguous')
            data = self.ref(chunk)
            chunk_paths.append(chunk['path'])
            with np.load(io.BytesIO(data), allow_pickle=False) as z:
                f, s, t, ti, dt = [z[k] for k in ('frame', 'substep', *CLOCK, 'physics_dt')]
                size = chunk['stop']-cursor
                require(all(v.shape == (size,) for v in (f, s, t, ti, dt)), 'chunk timeline layout')
                expected_frames = np.full(size, -1) if constructor else np.arange(cursor, cursor+size)//2
                require(np.array_equal(f, expected_frames) and
                        np.array_equal(s, np.arange(cursor, cursor+size)%2), 'wrong native frame/microstep')
                require(np.all(dt == self.identity['physics_dt_s']), 'native point API dt mismatch')
                peak = np.zeros((size, 64), np.float64)
                for arm in ARMS:
                    spec = next(x for x in self.identity['views'] if x['arm'] == arm)
                    env_ids = np.asarray(spec['env_ids'])
                    pre = arm+'_points_'
                    cap = int(z[pre+'capacity'])
                    require(cap == spec['capacity'] and int(z[pre+'event_count']) == size, 'point capacity/event count mismatch')
                    offsets = z[pre+'event_offsets']
                    require(offsets.dtype == np.dtype('int64') and offsets.shape == (size+1,)
                            and offsets[0] == 0 and (np.diff(offsets) >= 0).all(), 'invalid point event offsets')
                    count, start = z[pre+'counts'], z[pre+'starts']
                    require(count.shape == (size, len(env_ids), len(spec['filters'][0])), 'point count matrix shape')
                    require(equal_bits(count, z[arm+'_normal_counts']) and equal_bits(start, z[arm+'_normal_starts']), 'raw/stored counts or starts differ')
                    pi, si, fi = [z[pre+k] for k in ('point_indices', 'sensor_indices', 'partner_indices')]
                    forces = z[pre+'normal_forces']
                    require(all(x.shape == (int(offsets[-1]),) for x in (pi, si, fi)), 'point compact index lengths')
                    for field, width in [('normal_forces', 1), ('points', 3), ('normals', 3), ('separations', 1)]:
                        raw = z[pre+field]
                        require(raw.shape == (int(offsets[-1]), width) and raw.dtype.kind == 'f'
                                and np.isfinite(raw).all(), 'missing/nonfinite raw valid point field:' + field)
                    stored, duplicate = z[pre+'partner_abs_normal_sum_N'], z[arm+'_partner_scalar_abs_N']
                    require(stored.dtype == np.dtype('float64') and equal_bits(stored, duplicate), 'point scalar duplicate differs')
                    for e in range(size):
                        lo, hi = map(int, offsets[e:e+2])
                        scalar = scalar_points(count[e], start[e], pi[lo:hi], si[lo:hi], fi[lo:hi], forces[lo:hi], cap)
                        require(equal_bits(scalar, stored[e]), 'raw sumabs differs from recorded partner scalar')
                        lane = np.zeros(64, np.float64)
                        np.maximum.at(lane, env_ids, scalar.max(axis=1))
                        peak[e] = np.maximum(peak[e], lane)
                        points_total += hi-lo
                        sensor, partner = np.unravel_index(scalar.argmax(), scalar.shape)
                        value = float(scalar[sensor, partner])
                        if maximum is None or value > maximum['scalar_N']:
                            maximum = dict(scalar_N=value, arm=arm, env_id=int(env_ids[sensor]),
                                frame=int(f[e]), substep=int(s[e]), native_clock_s=float(t[e]),
                                sensor_path=spec['sensors'][sensor], partner_path=spec['filters'][sensor][partner],
                                valid_point_count=int(count[e, sensor, partner]), chunk=chunk['path'])
                        self.checks['raw_point_sumabs_event_arm'] += 1
                    for field in CHUNK_FIELDS:
                        key = arm+'_'+field
                        v = z[key]
                        expected = (size, 64, self.joints[arm])
                        if field == 'native_root_xyzw':
                            expected = (size, 64, 7)
                        elif field == 'native_root_velocity':
                            expected = (size, 64, 6)
                        require(v.shape == expected and np.isfinite(v).all(), 'chunk incomplete native state:' + key)
                        states.setdefault(key, []).append(v)
                peaks.append(peak); clocks.append(t); indices.append(ti); frames.append(f); subs.append(s)
            cursor += size
            print('POINT_CHUNK', chunk['path'], 'events', cursor, flush=True)
        require(cursor == steps*2, 'missing final native point events')
        require(set(chunk_paths) == {str(p.relative_to(self.root)) for p in (self.root/'native_contacts').glob('*.npz')}, 'orphan/missing native point chunk')
        peak, clock, index = map(np.concatenate, (peaks, clocks, indices))
        require(np.all(np.diff(index) == 1) and np.allclose(np.diff(clock), NATIVE_TICK_SECONDS, atol=1e-10, rtol=0), 'native clock is not consecutive 120Hz microsteps')
        if constructor:
            return dict(events=cursor, controller_events=0, valid_point_observations=points_total,
                        maximum=maximum, scope='explicit two-step constructor probe only; not hidden constructor events or controller windows',
                        raw_sumabs_exact=True, native_clock_tick_s=NATIVE_TICK_SECONDS,
                        force_api_dt_s=self.identity['physics_dt_s'],
                        scalar_alarm_lanes=np.flatnonzero((peak > GATE).any(0)).tolist())
        with np.load(io.BytesIO(self.data(self.root/'response_stream.npz')), allow_pickle=False) as z:
            require(equal_bits(peak.reshape(steps, 2, 64).max(1), z['scalar_normal_max_N']), 'macro scalar does not reconstruct from raw microsteps')
            post = z['post_d']
            require(post.shape == (steps, 64, 9021) and np.isfinite(post).all(), 'full9021 raw geometry layout')
            first_geometry = {lane: int(np.flatnonzero(post[:, lane].min(-1) < 0)[0])
                              for lane in range(64) if (post[:, lane].min(-1) < 0).any()}
        states = {k: np.concatenate(v) for k, v in states.items()}
        states[CLOCK[0]], states[CLOCK[1]] = clock, index
        first = first_exceedances(peak)
        events = self.js(self.root/'event_camera_links.json')
        reported = [e for e in events if e['trigger'] == 'all_arm_partner_scalar_gt_0p1N']
        require(len(reported) == len(first) and {e['lane'] for e in reported} == set(first), 'first scalar event lane coverage mismatch')
        for event in reported:
            e = first[event['lane']]
            require((event['step'], event['substep']) == (e//2, e%2) and
                    event['scalar_N'] == float(peak[e, event['lane']]), 'first scalar event not exact native microstep')
        geo = [e for e in events if e['trigger'] == 'any9021_raw_geometry_negative']
        require(len(geo) == len(first_geometry) and {e['lane']: e['step'] for e in geo} == first_geometry, 'first raw geometry event mismatch')
        self.checks['independent_first_scalar_and_geometry_events'] += 1
        result = dict(events=cursor, lanes=64, valid_point_observations=points_total, maximum=maximum,
            scalar_formula='per microstep, max over arm/sensor/partner of sum(abs(original valid point normal force)); no vector resultant',
            first_scalar_events=[dict(env_id=l, microstep=e, frame=e//2, substep=e%2,
                scalar_N=float(peak[e, l]), simulation_time_s=float(clock[e]), simulation_time_step_index=int(index[e])) for l, e in first.items()],
            no_scalar_alarm_lanes=[l for l in range(64) if l not in first],
            chunk_native_binding_fields=list(CHUNK_FIELDS)+list(CLOCK),
            chunk_fields_not_recorded=['native_link_transforms_xyzw', 'environment_origins'])
        return result, states, first, first_geometry

    def attempt(self, rec, baseline, inventory, slot, mode, prefix):
        arm, number = rec['arm'], rec['attempt']
        validate_attempt_identity(arm, number)
        where = rec['native_before_all64']['path']
        out = dict(arm=arm, attempt=number, path=where, accepted=False, issues=[])
        try:
            before, after = self.snapshot(rec['native_before_all64']), self.snapshot(rec['native_after_all64'])
            exact_snapshot(baseline, before); exact_snapshot(before, after)
            self.checks['attempt_native_freeze_all64'] += 1
        except Exception as exc:
            self.issue('render_native_freeze', where, exc); out['issues'].append(str(exc))
        held = rec.get('held_renders', [])
        for i, row in enumerate(held):
            try:
                require(row['index'] == i, 'held render ordering mismatch')
                b, a = self.snapshot(row['native_before_all64']), self.snapshot(row['native_after_all64'])
                exact_snapshot(baseline, b); exact_snapshot(b, a)
                self.checks['individual_held_render_freeze_all64'] += 1
            except Exception as exc:
                self.issue('render_native_freeze', where, exc); out['issues'].append(str(exc))
        rendered = bool(held) or 'original_outputs' in rec
        out['actual_render_records'] = len(held)
        if not rendered:
            out['issues'].append('no_actual_render_buffers')
            return out
        try:
            require(len(held) == 3 and rec['render_calls'] == 3, 'every candidate requires three recorded held renders')
            f0, f1 = np.asarray(rec['sensor_frame_before']), np.asarray(rec['sensor_frame_after'])
            ok = (f0.shape == f1.shape == (1,) and f0.dtype.kind in 'iu' and f1.dtype.kind in 'iu'
                  and int(f1[0]) == int(f0[0])+1 and rec['sensor_update_dt_s'] == 0.)
            require(self.check(ok, 'sensor_frame_exact_plus_one', where, 'frame delta is not exactly +1 at dt=0'), 'sensor refresh mismatch')
            out.update(sensor_frame_before=int(f0[0]), sensor_frame_after=int(f1[0]))
            outputs = self.npz(rec['original_outputs'])
            info = json.loads(self.ref(rec['original_info']))
            rgb = outputs['rgb']
            require(rgb.shape in [(HEIGHT, WIDTH, 3), (HEIGHT, WIDTH, 4)] and rgb.dtype == np.uint8, 'original RGB layout')
            require(sha(rgb.tobytes()) == rec['raw_rgb_sha256'], 'raw RGB pixel SHA mismatch')
            with Image.open(io.BytesIO(self.ref(rec['rgb']))) as picture:
                require(equal_bits(np.asarray(picture), rgb), 'PNG differs from original RGB bytes')
            out['rgb_original_verified'] = True
            rows = [(p, m) for p, m in inventory.items() if m['hand'] and (m['env_id'], m['arm']) == (slot, arm)]
            require(bool(rows), 'no native hand centers')
            points = baseline[arm+'_native_link_transforms_xyzw'][slot, [m['body_index'] for _, m in rows], :3].astype(np.float64)
            # JSON positions cannot supply the oracle geometry; native NPZ does.
            claimed_points = np.asarray(rec['hand_link_positions_world_m'])
            require(np.array_equal(points, claimed_points), 'recorded hand centers differ from native links')
            framing = frustum(points, rec['camera'])
            out['framing'] = framing
            if not framing['pass_gate']:
                out['issues'].append('six_plane_native_sphere_clearance_below_0p02m')
            mask = mask_oracle(outputs, info, inventory, slot, arm, mode, prefix)
            out['mask'] = mask
            out['issues'].extend(mask['errors'])
            out['accepted'] = not out['issues']
            self.checks['actual_original_rgb_segmentation_pairs'] += 1
        except Exception as exc:
            out['issues'].append(type(exc).__name__ + ':' + str(exc))
            self.issue('render_evidence_integrity', where, exc)
        return out

    def cameras(self, steps, native, first, geometry_first, mode, prefix, camera_source, closed):
        receipt_path = self.root/'qualified_camera_receipts.json'
        receipts = self.js(receipt_path)['receipts'] if receipt_path.exists() else []
        disk = sorted((self.root/'qualified_views').glob('*/*/env_*/state.json'))
        refs = {row['state']: row for row in receipts}
        if closed:
            self.check(len(refs) == len(receipts) and set(refs) == {str(p.relative_to(self.root)) for p in disk},
                       'camera_group_inventory', receipt_path, 'missing/duplicate/unreceipted camera groups')
        initial = self.npz(self.root/'native_initial_all64.npz')
        results, keys, observed_attempts, origins = [], [], set(), None
        for path in disk:
            relative = str(path.relative_to(self.root))
            state = self.js(path)
            result = dict(path=relative, env_id=state['env_id'], kind=state['capture_kind'],
                          step=state['step'], substep=state['substep'], arms={}, accepted=False)
            start_errors = len(self.errors)
            try:
                if relative in refs:
                    require(refs[relative]['sha256'] == self.ledger[str(path)]['sha256'], 'camera state receipt SHA mismatch')
                slot, kind, step, sub = state['env_id'], state['capture_kind'], state['step'], state['substep']
                require(type(slot) is int and 0 <= slot < 64 and path.parent.name == f'env_{slot:03d}', 'camera slot identity mismatch')
                require(path.parent.parent.parent.name == kind, 'camera kind path mismatch')
                require(path.parent.parent.name.startswith(f'step_{step:04d}_sub_{sub if sub is not None else "none"}_capture_'), 'camera native frame path mismatch')
                key = (kind, slot, step, sub)
                require(key not in keys, 'duplicate camera event group')
                keys.append(key)
                require(state['source_sha256'] == sha(self.data(H/camera_source)) and
                        state['inherited_hand_views_sha256'] == sha(self.data(H/'hand_views.py')), 'camera source SHA mismatch')
                require(state['max_attempts_per_arm'] == MAX_ATTEMPTS and state['required_images_per_arm'] == 3,
                        'unregistered candidate/image count')
                require(state['environment_count'] == 64 and state['physics_advanced_by_collector'] is False,
                        'collector layout or physics declaration mismatch')
                baseline = self.snapshot(state['native_before_all64'])
                exact_snapshot(baseline, self.snapshot(state['native_final_all64']))
                require(state[CLOCK[0]] == float(baseline[CLOCK[0]]) and state[CLOCK[1]] == int(baseline[CLOCK[1]]), 'camera JSON clock differs from raw snapshot')
                if origins is None:
                    origins = baseline['environment_origins']
                require(equal_bits(origins, baseline['environment_origins']), 'environment origins changed between camera groups')
                if kind == 'initial':
                    require(step == -1 and sub is None, 'initial event mismatch')
                    mapping = {'native_q':'q', 'native_qd':'qd', 'native_position_targets':'targets',
                               'native_root_xyzw':'root', 'native_root_velocity':'root_velocity', 'native_link_transforms_xyzw':'links'}
                    for a in ARMS:
                        for f, k in mapping.items():
                            require(equal_bits(baseline[a+'_'+f], initial[a+'_'+k]), 'initial native state binding mismatch:'+a+'_'+f)
                    for f in CLOCK:
                        require(equal_bits(baseline[f], initial[f]), 'initial native clock mismatch')
                    result['event_binding'] = 'initial_full_native_state_exact'
                elif native is not None:
                    if kind == 'first_scalar_alarm':
                        require(slot in first and (step, sub) == (first[slot]//2, first[slot]%2), 'camera does not bind first >0.1N exact microstep')
                        e = first[slot]
                    elif kind == 'first_raw_geometry_alarm':
                        require(slot in geometry_first and step == geometry_first[slot] and sub is None, 'geometry camera event mismatch')
                        e = step*2+1
                    elif kind == 'final':
                        require(step == steps-1 and sub is None, 'final camera event mismatch')
                        e = steps*2-1
                    else:
                        raise ValueError('unregistered capture kind:' + kind)
                    bind_microstep(baseline, native, e)
                    result['event_binding'] = 'all64_native_chunk_q_qd_roots_rootvel_fulltargets_and_clock_exact'
                    result['microstep'] = e
                    self.checks['exact_camera_native_microstep_binding'] += 1
                else:
                    result['event_binding'] = 'UNABLE_TO_VERIFY_NATIVE_POINT_CHUNKS_UNAVAILABLE'
                    self.issue('event_binding', relative, result['event_binding'])
                inventory = self.load_inventory(state['native_path_mapping'])
                if mode == 'sha256':
                    setup = json.loads(self.ref(state['semantic_setup']))
                    require(setup['semantic_token_to_native_path'] == token_inventory(inventory, mode, prefix), 'saved SHA token/native path mapping is not independent exact reconstruction')
                    validate_overlay(setup, inventory)
                    self.checks['v4_nested_deinstance_records_consistent'] += 1
                attempts = []
                seen = set()
                for embedded in state['attempts']:
                    rec = json.loads(self.ref(embedded['attempt_record']))
                    require(rec == {k:v for k,v in embedded.items() if k != 'attempt_record'}, 'embedded/file attempt mismatch')
                    require((rec['arm'], rec['attempt']) not in seen, 'duplicate arm attempt')
                    seen.add((rec['arm'], rec['attempt']))
                    observed_attempts.add(embedded['attempt_record']['path'])
                    attempts.append(self.attempt(rec, baseline, inventory, slot, mode, prefix))
                for a in ARMS:
                    arm_attempts = [x for x in attempts if x['arm'] == a]
                    require([x['attempt'] for x in arm_attempts] == list(range(1, len(arm_attempts)+1)),
                            'candidate attempts not contiguous from one')
                    good = [x for x in arm_attempts if x['accepted']]
                    angles = [x['framing']['azimuth_deg'] for x in good]
                    spread = max((abs((x-y+180)%360-180) for x,y in itertools.combinations(angles, 2)), default=0.)
                    passed = len(good) >= 3 and spread >= 30.
                    self.check(passed, 'three_images_and_azimuth', relative+'/'+a,
                               f'accepted {len(good)}/3; azimuth spread {spread:.6f} deg; target >=30')
                    result['arms'][a] = dict(accepted=passed, accepted_images=len(good),
                        azimuths_from_actual_camera_deg=angles, maximum_separation_deg=spread,
                        attempts=arm_attempts)
                result['accepted'] = len(self.errors) == start_errors and all(x['accepted'] for x in result['arms'].values())
            except Exception as exc:
                self.issue('camera_group_integrity', relative, type(exc).__name__+':'+str(exc))
            results.append(result)
            print('CAMERA_GROUP', relative, 'accepted', result['accepted'], flush=True)
        scheduled = self.js(H/'DYNAMIC_REGISTRATION.json')['camera']['scheduled_slots']
        if closed:
            expected = {('initial', l, -1, None) for l in scheduled}
            if steps:
                expected |= {('final', l, steps-1, None) for l in scheduled}
            if native is not None:
                expected |= {('first_scalar_alarm', l, e//2, e%2) for l,e in first.items()}
                expected |= {('first_raw_geometry_alarm', l, e, None) for l,e in geometry_first.items()}
            self.check(set(keys) == expected, 'expected_camera_events', self.root,
                       'missing='+repr(sorted(expected-set(keys)))+' extra='+repr(sorted(set(keys)-expected)))
            on_disk = {str(p.relative_to(self.root)) for p in (self.root/'qualified_views').glob('*/*/env_*/*.attempt.json')}
            self.check(on_disk == observed_attempts, 'all_actual_attempts_retained', self.root,
                       f'orphan attempts={len(on_disk-observed_attempts)}; missing={len(observed_attempts-on_disk)}')
        return results


def producer_closure(root):
    methods = [m for m,p in ROOTS.items() if str(root) == p]
    require(len(methods) == 1, 'unregistered producer root')
    method = methods[0]; path = Path(EXECUTION_FILES[method])
    if not path.exists():
        return dict(closed=False,reason='registered supervisor receipt not yet present')
    raw = path.read_bytes(); doc = json.loads(raw)
    found = [(path,doc,j) for j in doc.get('jobs',[]) if j.get('out') == str(root)]
    if len(found) != 1:
        return dict(closed=False, reason='no unique supervisor execution job for exact root', matches=len(found))
    path, doc, job = found[0]
    pid = job.get('pid')
    active = False
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        active = str(stat[19]) == str(job.get('start_ticks')) and stat[0] != 'Z'
    except FileNotFoundError:
        pass
    except OSError:
        active = True
    terminal = type(job.get('actual_exit')) is int and bool(job.get('closed_utc')) and bool(doc.get('closed_utc'))
    return dict(closed=terminal and not active, process_active=active, execution=str(path),
                execution_snapshot_sha256=sha(raw), pid=pid, start_ticks=job.get('start_ticks'),
                actual_exit=job.get('actual_exit'), closed_utc=job.get('closed_utc'),
                supervisor_status=doc.get('status'), supervisor_closed_utc=doc.get('closed_utc'),
                resource_abort=job.get('resource_abort'), job_status=job.get('status'),
                plan_sha256=doc.get('plan_sha256'), argv=job.get('argv'),
                reason='supervisor terminal exit plus exact PID/start_ticks absence required')


def source_check(audit, closure):
    matches = []
    for path in H.glob('*PLAN*.json'):
        data = path.read_bytes()
        if sha(data) == closure.get('plan_sha256'):
            matches.append(path)
    require(len(matches) == 1, 'cannot bind one frozen execution plan by SHA')
    plan = audit.js(matches[0])
    for path, expected in plan['sources'].items():
        require(sha(audit.data(path)) == expected, 'frozen source changed:'+path)
    require(any(j['out'] == closure.get('producer_root', str(audit.root)) and j['argv'] == closure['argv'] for j in plan['jobs']), 'execution job differs from frozen plan')
    return dict(plan=str(matches[0]), plan_sha256=closure['plan_sha256'], verified_sources=len(plan['sources']))


def resource_check(audit, closure):
    method = audit.root.name
    path = Path(EXECUTION_FILES[method]); raw = audit.data(path); doc = json.loads(raw)
    require(sha(raw) == closure['execution_snapshot_sha256'], 'execution changed since closure')
    job, = [j for j in doc['jobs'] if j.get('out') == str(audit.root)]
    plan = audit.js(PLAN_FILES[method])
    initial = audit.js(H/(doc['tag']+'_'+method+'_resource_gate.json'))
    if method == 'raw':
        preserved_raw_failure(raw,doc,job)
    result = resource_assessment(job,doc['status'],plan['resource'],initial)
    if method == 'raw':
        require(result['verdict'] == 'FAILED', 'raw resource failure cannot be promoted')
    result.update(execution=str(path),execution_sha256=sha(raw),original_supervisor_status=doc['status'],
                  original_job_status=job['status'],producer_actual_exit=job['actual_exit'],
                  original_raw_failure_immutable_verified=(method == 'raw'))
    return result


def write_report(name, report):
    require(re.fullmatch(r'ASTRA_NATIVE_EVIDENCE[A-Za-z0-9_.-]*\.json', name) is not None, 'output outside allowed namespace')
    path = H/name
    require(not path.is_symlink(), 'output cannot be symlink')
    # Audit history is append-only. Never replace a prior verdict or its SHA.
    data = json.dumps(report, indent=2, allow_nan=False)+'\n'
    for version in itertools.count(1):
        candidate = path if version == 1 else path.with_name(path.stem+f'_v{version}.json')
        require(not candidate.is_symlink(), 'output cannot be symlink')
        try:
            with candidate.open('x') as stream:
                stream.write(data)
            return candidate
        except FileExistsError:
            continue


def supervise_v7():
    """One lightweight owner, identity-bound terminal and actual wait per child."""
    lock = (H/'ASTRA_NATIVE_EVIDENCE_V7_SUPERVISOR_LOCK.log').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        print('ANOTHER_OWNED_V7_SUPERVISOR_ALREADY_HOLDS_LOCK',flush=True)
        return 2
    active, reg_sha = registration()
    source_archive = write_report('ASTRA_NATIVE_EVIDENCE_V7_EXECUTED_SOURCE.json',
        dict(sha256=EXECUTED_SOURCE_SHA256,source_utf8=EXECUTED_SOURCE_BYTES.decode(),
             base_v5_oracle_sha256=V5_ORACLE_SHA256, base_v6_oracle_sha256=V6_ORACLE_SHA256,
        deinstance_contract_correction='Prior flags are fresh write-time state, not round-start selection. Editable already-deinstanced inherited clones are allowed; proxy and ineffective changes remain rejected.',registration_sha256=reg_sha))
    pid = os.getpid()
    ticks = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    launch = write_report('ASTRA_NATIVE_EVIDENCE_V7_WORKERS_LAUNCH.json',
        dict(pid=pid,start_ticks=ticks,argv=sys.argv,oracle_sha256=EXECUTED_SOURCE_SHA256,
             source_archive=str(source_archive),registration_sha256=reg_sha,
             primary_roots=active['primary_roots'],started_utc=datetime.now(timezone.utc).isoformat(),
             status='WAITING_BOTH_REGISTERED_PRODUCERS_CLOSED',gpu_used=False,
             primary_full_run_acceptance='BLOCKED_REFERENCE_RESOURCE_FAILURE'))
    print('CPU_SUPERVISOR_LAUNCH',launch,flush=True)
    stopped = []
    def stop(signum, frame):
        stopped.append(signum)
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    children, closures, code, reason = [], {}, 3, 'waiting'
    deadline = time.monotonic()+14400.
    try:
        ready = False
        while not stopped and time.monotonic()<deadline:
            require(registration()[1] == reg_sha,'registration changed while waiting')
            ready, closures = pair_closures(active)
            if ready:
                break
            terminal_missing = []
            for method,path in EXECUTION_FILES.items():
                if Path(path).exists():
                    doc = json.loads(Path(path).read_bytes())
                    if doc.get('closed_utc') and not closures[method]['closed']:
                        terminal_missing.append(method)
            if terminal_missing:
                code, reason = 2, 'producer_terminal_without_exact_closed_job:'+','.join(terminal_missing)
                break
            time.sleep(10.)
        if ready and not stopped:
            code, reason = 0, 'both_sequential_evidence_workers_actually_reaped'
            for method in ('raw','multirow'):
                require(pair_closures(active)[0],'both producer closures required before child')
                require(sha(Path(__file__).read_bytes()) == EXECUTED_SOURCE_SHA256,'v6 oracle source changed')
                require(sha((H/'check_native_visible_evidence_v5.py').read_bytes()) == V5_ORACLE_SHA256,
                        'frozen b3 oracle changed')
                args = [sys.executable,'-B',str(Path(__file__).resolve()),'--producer-root',ROOTS[method],
                        '--output',OUTPUT_REPORT_NAMES[method],'--expected-oracle-sha',EXECUTED_SOURCE_SHA256]
                for version in itertools.count(1):
                    logpath = H/f'ASTRA_NATIVE_EVIDENCE_{method}_v7_WORKER_{version}.log'
                    try:
                        stream = logpath.open('xb'); break
                    except FileExistsError:
                        continue
                env = dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
                           OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
                with stream:
                    child = subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,env=env,cwd=H)
                    row = dict(method=method,pid=child.pid,start_ticks=None,argv=args,log=str(logpath),
                               oracle_sha256=EXECUTED_SOURCE_SHA256,source_archive=str(source_archive),
                               supervisor_launch_manifest=str(launch),started_utc=datetime.now(timezone.utc).isoformat())
                    try:
                        row['start_ticks'] = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
                        row['worker_launch_receipt'] = str(write_report(f'ASTRA_NATIVE_EVIDENCE_{method}_v7_WORKER_LAUNCH.json',row))
                        while True:
                            try:
                                child.wait(timeout=1.); break
                            except subprocess.TimeoutExpired:
                                if stopped:
                                    child.terminate()
                                    try:
                                        child.wait(timeout=60.)
                                    except subprocess.TimeoutExpired:
                                        child.kill(); child.wait()
                                    break
                    except Exception as exc:
                        row['supervision_error'] = type(exc).__name__+':'+str(exc)
                        code = 2
                    finally:
                        if child.poll() is None:
                            child.terminate()
                            try:
                                child.wait(timeout=60.)
                            except subprocess.TimeoutExpired:
                                child.kill(); child.wait()
                        actual = child.wait()
                source_after_wait = sha(Path(__file__).read_bytes())
                row.update(actual_wait_exit=actual,closed_utc=datetime.now(timezone.utc).isoformat(),
                           wait_source='subprocess.Popen.wait()',log_sha256=sha(logpath.read_bytes()),
                           oracle_source_sha256_after_wait=source_after_wait,
                           source_stable_after_wait=source_after_wait == EXECUTED_SOURCE_SHA256)
                # Persist the actual wait even if report parsing/identity later fails.
                receipt = write_report(f'ASTRA_NATIVE_EVIDENCE_{method}_v7_WORKER_WAIT_EXIT.json',row)
                reports = []
                for line in logpath.read_text().splitlines():
                    try:
                        item = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(item,dict) and item.get('phase') == 'FINAL_CLOSED_PRODUCER_AUDIT':
                        reports.append(Path(item['report']))
                try:
                    require(len(reports) == 1, 'missing or duplicate exact terminal report')
                    data = reports[0].read_bytes(); final = json.loads(data)
                    require(final['oracle_sha256'] == EXECUTED_SOURCE_SHA256 and final['root'] == ROOTS[method],
                            'child report identity mismatch')
                    row.update(report=str(reports[0]),report_sha256=sha(data),final_verdict=final['final_verdict'],
                        data_integrity_verdict=final['data_integrity_verdict'],
                        resource_policy_verdict=final['resource_policy_verdict'],
                        registered_resource_verdict=final['registered_resource_verdict'],
                        conditional_comparison_eligible=final['conditional_comparison_eligible'])
                except Exception as exc:
                    row['report_binding_error'] = type(exc).__name__+':'+str(exc)
                if not row['source_stable_after_wait']:
                    row['conditional_comparison_eligible'] = False
                    code = 2
                row['report_binding_receipt'] = str(write_report(
                    f'ASTRA_NATIVE_EVIDENCE_{method}_v7_WORKER_REPORT_BINDING.json',dict(row,actual_wait_receipt=str(receipt))))
                children.append(dict(row,actual_wait_receipt=str(receipt)))
                print('CPU_WORKER_REAPED',method,actual,receipt,flush=True)
                if actual != 0 or not row.get('conditional_comparison_eligible'):
                    code = 2
                if stopped:
                    reason = 'owned_stop_requested_child_actually_reaped'; break
        elif stopped:
            reason = 'owned_stop_requested_no_heavy_arrays_read'
        elif reason == 'waiting':
            reason = 'bounded_closure_wait_expired'
    except Exception as exc:
        code, reason = 2, type(exc).__name__+':'+str(exc)
    both = len(children) == 2 and {r['method'] for r in children} == {'raw','multirow'}
    evidence_pass = both and all(r.get('data_integrity_verdict') == 'PASS' for r in children)
    eligible = both and all(r.get('actual_wait_exit') == 0 and r.get('conditional_comparison_eligible') for r in children)
    aggregate = write_report('ASTRA_NATIVE_EVIDENCE_V7_OVERALL.json',
        dict(launch_manifest=str(launch),oracle_sha256=EXECUTED_SOURCE_SHA256,children=children,
             data_integrity_verdict='PASS' if evidence_pass else 'FAIL',
             registered_resource_verdict='FAILED',
             conditional_comparison_verdict='CONDITIONAL' if eligible else 'BLOCKED',
             registered_full_run_acceptance='BLOCKED',raw_resource_policy_verdict='FAILED',
             raw_original_execution_sha256=RAW_EXECUTION_SHA256,reason=reason,
             physical_safety_certified=False,new_independent_initial_states_added_by_reusing_raw=0,
             scope='Conditional data comparison only; original resource failure and scientific thresholds retained.',
             finished_utc=datetime.now(timezone.utc).isoformat()))
    if not eligible:
        code = 2 if code == 0 else code
    terminal = write_report('ASTRA_NATIVE_EVIDENCE_V7_WORKERS_TERMINAL.json',
        dict(launch_manifest=str(launch),children=children,producer_closures=closures,
             overall_report=str(aggregate),overall_report_sha256=sha(aggregate.read_bytes()),
             reason=reason,intended_supervisor_return_code=code,actual_supervisor_wait_exit=None,
             registered_full_run_acceptance='BLOCKED',finished_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(supervisor_terminal=str(terminal),overall_report=str(aggregate),intended_exit=code)),flush=True)
    lock.close()
    return code


def self_test():
    """In-memory contract tests only; no simulated render is scientific evidence."""
    passed = []
    p = '/World/envs/env_0/F_L/index_link'
    q = '/World/envs/env_0/F_L/thumb_link'
    foreign = '/World/envs/env_9/F_L/index_link'
    inv = {path:dict(env_id=slot, arm='F_L', hand=True) for path,slot in [(p,0),(q,0),(foreign,9)]}
    def pair(n, path=p, semantic_path=p, mode='exact', prefix='qv2_hand/'):
        arr = np.zeros((HEIGHT, WIDTH, 1), np.uint32)
        arr.reshape(-1)[:n] = 2
        token = prefix+(sha(semantic_path.encode()) if mode == 'sha256' else semantic_path.lstrip('/'))
        if mode == 'unique-casefold':
            token = token.casefold()
        info = {STREAMS[0]:dict(idToLabels={'0':'BACKGROUND','2':path}),
                STREAMS[1]:dict(idToLabels={'0':{'class':'BACKGROUND'}, '2':{'class':token}})}
        return {s:arr.copy() for s in STREAMS}, info
    for n, expected in [(0,False), (200,False), (201,True)]:
        data, info = pair(n)
        require(mask_oracle(data, info, inv, 0, 'F_L')['pass_gate'] == expected, 'pixel boundary regression')
        passed.append('strict_pixel_boundary_'+str(n))
    data, info = pair(300, semantic_path=q)
    result = mask_oracle(data, info, inv, 0, 'F_L')
    require(not result['pass_gate'] and result['observed_hand_pixels'] == 0 and result['same_arm_pixel_intersection'] == 300,
            'same-arm different-rigid mismatch was accepted')
    passed.append('same_arm_different_rigid_300_rejected')
    data, info = pair(300, mode='unique-casefold')
    require(mask_oracle(data, info, inv, 0, 'F_L', 'unique-casefold')['pass_gate'], 'unique normalized token fails')
    require(not mask_oracle(data, info, inv, 0, 'F_L')['pass_gate'], 'strict legacy mode accepts altered case')
    passed.append('casefold_unique_native_anchor_and_legacy_failure')
    data, info = pair(300, mode='sha256', prefix='qv3_hand_')
    require(mask_oracle(data, info, inv, 0, 'F_L', 'sha256', 'qv3_hand_')['pass_gate'], 'digest native anchor fails')
    data[STREAMS[0]].reshape(-1)[300] = 3
    data[STREAMS[1]].reshape(-1)[300] = 3
    info[STREAMS[0]]['idToLabels']['3'] = foreign
    info[STREAMS[1]]['idToLabels']['3'] = {'class':'qv3_hand_'+sha(foreign.encode())}
    result = mask_oracle(data, info, inv, 0, 'F_L', 'sha256', 'qv3_hand_')
    require(not result['pass_gate'] and result['observed_hand_pixels'] == 300, 'foreign clone passed alongside own300')
    passed.append('sha256_foreign_clone_even_with_target300_rejected')
    data, info = pair(300, path=foreign, semantic_path=foreign, mode='unique-casefold')
    require(not mask_oracle(data, info, inv, 0, 'F_L', 'unique-casefold')['pass_gate'], 'env9 treated as env0')
    passed.append('casefold_preserves_foreign_env_identity')
    data, info = pair(201)
    data[STREAMS[0]].reshape(-1)[202] = 4294967295
    data[STREAMS[0]] = data[STREAMS[0]].view(np.int32)
    require(not mask_oracle(data, info, inv, 0, 'F_L')['pass_gate'], 'unsigned unknown ID accepted')
    passed.append('signed_id_bits_unknown_rejected')
    collision = dict(inv)
    collision[p.lower()] = inv[p]
    try:
        token_inventory(collision, 'unique-casefold', 'qv2_hand/')
    except ValueError:
        passed.append('normalized_token_collision_rejected')
    else:
        raise ValueError('case collision accepted')
    a, b = np.zeros((64, 7), np.float32), np.zeros((64, 7), np.float32)
    b[63, 6] = -0.
    require(not equal_bits(a, b) and not equal_bits(a, a.astype(np.float64)) and not equal_bits(a, a.reshape(7,64)), 'native bit/shape/dtype check insufficient')
    passed.append('unrendered_env63_signed_zero_shape_dtype_rejected')
    peaks = np.zeros((4,64),np.float64);peaks[0,0]=.1;peaks[1,0]=.10001;peaks[3,0]=10.;peaks[2,63]=.2
    require(first_exceedances(peaks) == {0:1,63:2}, 'first scalar trigger must be per-env earliest strictly >0.1')
    passed.append('per_env_earliest_strict_0p1_trigger_not_later_peak')
    baseline = {a+'_'+f:np.zeros((64,7),np.float32) for a in ARMS for f in CHUNK_FIELDS}
    baseline.update(simulation_time_s=np.array(.5),simulation_time_step_index=np.array(60))
    native = {k:np.stack([v,v]) for k,v in baseline.items()}
    bind_microstep(baseline,native,0)
    for key, row in [('F_L_native_q',(1,63,0)),('simulation_time_step_index',(1,)),('simulation_time_s',(1,))]:
        changed = dict(native);changed[key] = changed[key].copy();changed[key][row] += 1
        try:
            bind_microstep(baseline,changed,1)
        except ValueError:
            passed.append('native_microstep_binding_rejects_'+key)
        else:
            raise ValueError('changed all64 microstep binding accepted')
    count, start = np.array([[2]], np.int32), np.array([[5]], np.int32)
    ids = np.array([5,6], np.int64); owners = np.zeros(2, np.int64)
    forces = np.array([[4.],[-3.]], np.float32)
    scalar = scalar_points(count, start, ids, owners, owners, forces, 20)
    require(scalar[0,0] == 7., 'sumabs replaced by resultant')
    passed.append('opposing_forces_sumabs7_not_resultant1')
    empty = scalar_points(np.zeros((2,3), np.int32), np.zeros((2,3), np.int32),
                          np.empty(0,np.int64), np.empty(0,np.int64), np.empty(0,np.int64),
                          np.empty((0,1),np.float32), 20)
    require(equal_bits(empty, np.zeros((2,3), np.float64)), 'zero valid contacts must reconstruct float64 zero sumabs')
    passed.append('empty_owned_point_event_float64_zero')
    for bad, label in [(np.array([5,5], np.int64),'duplicate'), (np.array([5,7], np.int64),'outside_interval')]:
        try:
            scalar_points(count, start, bad, owners, owners, forces, 20)
        except ValueError:
            passed.append('raw_point_'+label+'_rejected')
        else:
            raise ValueError('invalid point ownership accepted')
    camera = dict(actual_camera_to_world_row_matrix=np.eye(4).tolist(),
        actual_intrinsic_matrix=[[640,0,640],[0,360,360],[0,0,1]],
        sdk_intrinsic_matrix=[[640,0,640],[0,360,360],[0,0,1]],
        actual_clipping_range_m=[.1,10.],image_size=[1280,720],
        actual_usd_optics=dict(focal_length=1.,horizontal_aperture=2.,vertical_aperture=2.,
                               horizontal_aperture_offset=0.,vertical_aperture_offset=0.))
    good = frustum(np.array([[.01,0,-1.]]), camera)
    bad = frustum(np.array([[.01,0,-.15]]), camera)
    require(good['pass_gate'] and not bad['pass_gate'] and abs(bad['minimum_by_plane_m'][4]+.01)<1e-12,
            'native six-plane sphere radius check failed')
    passed.append('native_near_plane_sphere_radius_and_clearance')
    active = {'primary_roots': dict(ROOTS)}
    for name in active['primary_roots'].values():
        require(str(select_root(name, active)) == name, 'registered nested root rejected')
    for name in [str(SSD/'full_v6/raw'), str(SSD/'raw'), 'raw_v5', str(R/'raw_v5')]:
        try:
            select_root(name, active)
        except ValueError:
            pass
        else:
            raise ValueError('unregistered or mirror root accepted:'+name)
    passed.append('strict_registered_nested_roots_and_unregistered_rejection')
    for n in (1,6,7,24):
        validate_attempt_identity('F_L', n)
    for n in (0,25,True,1.0):
        try:
            validate_attempt_identity('F_L', n)
        except ValueError:
            pass
        else:
            raise ValueError('invalid candidate bound accepted')
    passed.append('v4_candidate_bound24_includes7_rejects25_and_bool')
    data, info = pair(300, mode='sha256', prefix=TOKEN_PREFIX)
    require(mask_oracle(data, info, inv, 0, 'F_L', 'sha256', TOKEN_PREFIX)['pass_gate'], 'v4 digest rejected')
    require(not mask_oracle(data, info, inv, 0, 'F_L', 'sha256', 'qv3_hand_')['pass_gate'], 'wrong v3 prefix accepted')
    passed.append('v4_correct_prefix_and_wrong_v3_prefix_rejected')
    info[STREAMS[1]]['idToLabels']['2']['class'] += ','+TOKEN_PREFIX+sha(q.encode())
    result = mask_oracle(data, info, inv, 0, 'F_L', 'sha256', TOKEN_PREFIX)
    require(not result['pass_gate'] and result['observed_hand_pixels'] == 0,
            'same-arm multi-rigid semantic ambiguity accepted')
    passed.append('v4_same_arm_two_rigid_semantic_tokens_rejected')

    # Fully in-memory parser fixture. It is never written as render evidence.
    import copy
    data, info = pair(300, mode='sha256', prefix=TOKEN_PREFIX)
    data['rgb'] = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    png = io.BytesIO(); Image.fromarray(data['rgb']).save(png, format='PNG')
    state = {'environment_origins':np.zeros((64,3),np.float32),
             CLOCK[0]:np.asarray(.5), CLOCK[1]:np.asarray(60)}
    for arm in ARMS:
        for field in FIELDS:
            shape = (64,1,7) if field == 'native_link_transforms_xyzw' else ((64,7) if field == 'native_root_xyzw' else ((64,6) if field == 'native_root_velocity' else (64,1)))
            state[arm+'_'+field] = np.zeros(shape, np.float32)
    state['F_L_native_link_transforms_xyzw'][0,0,:3] = [.01,0,-1.]
    state['F_L_native_link_transforms_xyzw'][0,0,6] = 1.
    class MemoryAudit(Audit):
        def __init__(self):
            super().__init__(H)
            self.body = {a:['index_link'] for a in ARMS}; self.joints = {a:1 for a in ARMS}
            self.states = {'held':state}
        def npz(self, ref):
            return data if ref['path'] == 'outputs' else self.states[ref['path']]
        def ref(self, ref):
            return png.getvalue() if ref['path'] == 'png' else json.dumps(info).encode()
    native_ref = {'path':'held','sha256':'fixture'}
    rec = dict(arm='F_L', attempt=24, native_before_all64=native_ref, native_after_all64=native_ref,
        held_renders=[dict(index=i,native_before_all64=native_ref,native_after_all64=native_ref) for i in range(3)],
        render_calls=3, sensor_frame_before=[10], sensor_frame_after=[11], sensor_update_dt_s=0.,
        original_outputs={'path':'outputs'}, original_info={'path':'info'}, rgb={'path':'png'},
        raw_rgb_sha256=sha(data['rgb'].tobytes()), camera=camera,
        hand_link_positions_world_m=state['F_L_native_link_transforms_xyzw'][0,:,:3].astype(np.float64).tolist(),
        table_ray_aabb=[{'path':'/World/envs/env_0/table','blocked_link_indices':[0]}])
    one_inv = {p:dict(env_id=0,arm='F_L',hand=True,body_index=0)}
    audit = MemoryAudit()
    require(audit.attempt(rec,state,one_inv,0,'sha256',TOKEN_PREFIX)['accepted'],
            'partial AABB proxy diagnostic vetoes valid original mask fixture or attempt24 rejected')
    require(audit.checks['individual_held_render_freeze_all64'] == 3, 'missing held render checks')
    passed.append('attempt24_table_proxy_diagnostic_actual_mask_parser_and_all3_held_records')
    changed = copy.deepcopy(rec); changed['held_renders'][1]['native_after_all64'] = {'path':'mutated'}
    audit = MemoryAudit(); audit.states['mutated'] = {k:v.copy() for k,v in state.items()}
    audit.states['mutated']['U_R_native_q'][63,0] = -0.
    result = audit.attempt(changed,state,one_inv,0,'sha256',TOKEN_PREFIX)
    require(not result['accepted'] and any(x['category']=='render_native_freeze' for x in audit.errors),
            'intermediate render non-target env63 signed-zero drift accepted')
    passed.append('held_render_env63_byte_drift_rejected')
    for key in CLOCK:
        audit = MemoryAudit(); audit.states['mutated'] = {k:v.copy() for k,v in state.items()}
        audit.states['mutated'][key] += 1
        require(not audit.attempt(changed,state,one_inv,0,'sha256',TOKEN_PREFIX)['accepted'],
                'held render native clock drift accepted')
        passed.append('held_render_'+key+'_drift_rejected')
    bad_frame = copy.deepcopy(rec); bad_frame['sensor_frame_after'] = [12]
    require(not MemoryAudit().attempt(bad_frame,state,one_inv,0,'sha256',TOKEN_PREFIX)['accepted'],
            'frame+2 accepted')
    passed.append('v4_actual_buffer_frame_plus2_rejected')
    no_buffers = copy.deepcopy(rec)
    for key in ('original_outputs','original_info','rgb'):
        del no_buffers[key]
    require(not MemoryAudit().attempt(no_buffers,state,one_inv,0,'sha256',TOKEN_PREFIX)['accepted'],
            'held render summary without original buffers accepted')
    passed.append('v4_missing_original_buffers_rejected')
    mesh = p+'/visual/nested/mesh'
    setup = dict(schema='safeduo.qualified_semantics.v4',
        hand_descendant_instanceability_overrides=[p+'/visual'],
        hand_descendant_instanceability_audit=dict(actual_paths_unchanged=True, max_rounds=32,
            remaining_instance_paths=[], remaining_proxy_paths=[],
            descendant_paths=[p+'/visual',p+'/visual/nested',mesh],
            render_mesh_native_hand_owners={mesh:p},
            rounds=[dict(round=0,overrides=[dict(path=p+'/visual',is_instance=True,
                is_instance_proxy=False,is_instanceable=True,after_is_instance=False)])]))
    validate_overlay(setup, one_inv)
    for field in ('remaining_instance_paths','remaining_proxy_paths'):
        bad = copy.deepcopy(setup); bad['hand_descendant_instanceability_audit'][field] = [mesh]
        try:
            validate_overlay(bad, one_inv)
        except ValueError:
            pass
        else:
            raise ValueError('remaining nested instance/proxy accepted')
        passed.append('v4_'+field+'_rejected')
    bad = copy.deepcopy(setup)
    bad['hand_descendant_instanceability_audit']['render_mesh_native_hand_owners'][mesh] = foreign
    try:
        validate_overlay(bad, inv)
    except ValueError:
        passed.append('v4_wrong_clone_render_mesh_owner_rejected')
    else:
        raise ValueError('foreign clone render mesh owner accepted')
    # Closure state-machine boundary: a completed raw method cannot start heavy
    # audit work while multirow is missing or still live.
    global producer_closure
    original_closure = producer_closure
    active['primary_executions'] = dict(EXECUTION_FILES)
    samples = {m:dict(closed=False) for m in ('raw','multirow')}
    try:
        producer_closure = lambda root: samples[root.name]
        samples['raw'] = dict(closed=True,plan_sha256=PLAN_SHAS['raw'],execution=active['primary_executions']['raw'])
        require(not pair_closures(active)[0], 'raw closure alone releases heavy audits')
        samples['multirow'] = dict(closed=True,plan_sha256=PLAN_SHAS['multirow'],execution=active['primary_executions']['multirow'])
        require(pair_closures(active)[0], 'both exact closures not released')
        samples['multirow']['plan_sha256'] = 'wrong'
        require(not pair_closures(active)[0], 'wrong producer plan closure accepted')
    finally:
        producer_closure = original_closure
    passed.append('both_exact_plan_producer_closures_required_before_arrays')
    # V6 additions exercise resource/data separation without production array I/O.
    sample = dict(owned_host_RSS_MiB=46000,host_available_MiB=35000,
        owned_GPU_MiB=12000,GPU_free_MiB=3000,probe_actual_exit=0,free_probe_actual_exit=0,selected_gpu=0)
    job = dict(actual_exit=0,resource_abort=None,status='complete',resource_samples=[sample])
    initial_gate = dict(selected_gpu=0,required_free_MiB=11000,actual_exit=0,stdout='0, 18566, 32607, 78\n')
    require(resource_assessment(job,'complete',RESOURCE_LIMITS,initial_gate)['verdict'] == 'PASS',
            'exact registered resource boundary rejected')
    passed.append('v6_resource_exact_RSS46000_boundary_pass')
    high = copy.deepcopy(job);high['resource_samples'][0]['owned_host_RSS_MiB'] = 46001
    require(resource_assessment(high,'complete',RESOURCE_LIMITS,initial_gate)['verdict'] == 'FAILED',
            'RSS threshold silently relaxed')
    passed.append('v6_RSS46001_fails_even_child0_without_abort_summary')
    aborted = copy.deepcopy(job);aborted['resource_abort'] = 'owned host RSS exceeds registered bound'
    require(resource_assessment(aborted,'complete',RESOURCE_LIMITS,initial_gate)['verdict'] == 'FAILED',
            'child0 promoted resource-aborted run')
    passed.append('v6_actual_exit0_never_erases_resource_abort')
    require(resource_assessment(job,'failed',RESOURCE_LIMITS,initial_gate)['verdict'] == 'FAILED',
            'failed supervisor reclassified')
    passed.append('v6_failed_supervisor_never_resource_pass')
    limits = dict(RESOURCE_LIMITS,max_owned_host_RSS_MiB=46143)
    try:
        resource_assessment(job,'complete',limits,initial_gate)
    except ValueError:
        passed.append('v6_resource_threshold_relaxation_rejected')
    else:
        raise ValueError('relaxed resource plan accepted')
    raw = Path(EXECUTION_FILES['raw']).read_bytes(); doc = json.loads(raw); original_job = doc['jobs'][0]
    preserved_raw_failure(raw,doc,original_job)
    try:
        preserved_raw_failure(raw+b' ',doc,original_job)
    except ValueError:
        passed.append('v6_original_resource_failed_raw_execution_byte_mutation_rejected')
    else:
        raise ValueError('rewritten raw execution accepted')
    changed = dict(doc,status='complete')
    try:
        preserved_raw_failure(raw,changed,original_job)
    except ValueError:
        passed.append('v6_raw_FAILED_to_complete_reclassification_rejected')
    else:
        raise ValueError('raw failure reclassification accepted')
    raw_verdict = evidence_verdict('raw',True,'FAILED')
    require(raw_verdict['data_integrity_verdict'] == 'PASS' and
            raw_verdict['registered_resource_verdict'] == 'FAILED' and
            raw_verdict['registered_full_run_acceptance'] == 'BLOCKED' and
            raw_verdict['final_verdict'] == 'BLOCKED_RESOURCE_FAILURE' and raw_verdict['conditional_comparison_eligible'],
            'raw conditional data pass conflated with run acceptance')
    passed.append('v6_raw_data_PASS_resource_FAILED_full_run_BLOCKED_separation')
    require(evidence_verdict('raw',False,'FAILED')['data_integrity_verdict'] == 'FAIL' and
            not evidence_verdict('raw',False,'FAILED')['conditional_comparison_eligible'],
            'bad raw data promoted because resource failure allowed')
    passed.append('v6_any_raw_data_failure_blocks_conditional_reuse')
    require(not evidence_verdict('multirow',True,'FAILED')['conditional_comparison_eligible'] and
            not evidence_verdict('multirow',True,'UNVERIFIED')['conditional_comparison_eligible'] and
            evidence_verdict('multirow',True,'PASS')['conditional_comparison_eligible'],
            'candidate does not independently require resource PASS')
    passed.append('v6_candidate_requires_independent_resource_PASS')
    def npy_bytes(array):
        buffer = io.BytesIO();np.save(buffer,array,allow_pickle=False);return buffer.getvalue()
    a = np.zeros((2,3),np.float32);a[0,0] = -0.
    spec = dict(dtype=a.dtype.str,shape=[3],bytes=a.nbytes)
    require(compare_npy_streams(io.BytesIO(npy_bytes(a)),io.BytesIO(npy_bytes(a)),spec,2)
            ['exported_vs_original_mmap_byte_exact'], 'stream export identical bytes rejected')
    passed.append('v6_streamed_original_export_exact_bytes')
    b = a.copy();b[0,0] = 0.
    for bad,label in [(npy_bytes(b),'signed_zero'),(npy_bytes(a.astype(np.float64)),'dtype'),
                      (npy_bytes(a)[:-1],'truncated'),(npy_bytes(a)+b'X','extra_bytes')]:
        try:
            compare_npy_streams(io.BytesIO(bad),io.BytesIO(npy_bytes(a)),spec,2)
        except ValueError:
            passed.append('v6_streamed_export_'+label+'_rejected')
        else:
            raise ValueError('changed/truncated export accepted:'+label)
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped,'w',compression=zipfile.ZIP_STORED) as z:
        z.writestr('field.npy',npy_bytes(a))
    corrupted = bytearray(zipped.getvalue())
    offset = corrupted.index(npy_bytes(a))
    corrupted[offset+len(npy_bytes(a))-1] ^= 1
    try:
        with zipfile.ZipFile(io.BytesIO(corrupted)) as z, z.open('field.npy') as exported:
            compare_npy_streams(exported,io.BytesIO(npy_bytes(a)),spec,2)
    except zipfile.BadZipFile:
        passed.append('v6_full_export_payload_CRC_corruption_rejected')
    else:
        raise ValueError('corrupted original export ZIP CRC accepted')
    active_real, _ = registration()
    require(active_real['primary_roots'] == ROOTS and PLAN_SHAS['raw'] != PLAN_SHAS['multirow'],
            'V6 must bind both distinct source plans')
    passed.append('v6_real_frozen_registration_two_distinct_source_plans')
    history_path = H/'ASTRA_NATIVE_EVIDENCE_V7_FALSE_POSITIVE_HISTORY.json'
    history = json.loads(history_path.read_bytes())
    for path, expected in history['original_immutable_sha256'].items():
        require(sha(Path(path).read_bytes()) == expected, 'original v6 failure history changed:'+path)
    passed.append('v7_all_v6_sources_reports_actualwait_and_camera_immutable')
    fixtures = []
    for fixture in history['fixture_rows']:
        root = Path(ROOTS[fixture['method']]); state_path = Path(fixture['state'])
        require(sha(state_path.read_bytes()) == fixture['state_sha256'], 'original native fixture state changed')
        original = {}
        for key in ('semantic_setup','native_path_mapping'):
            ref = fixture[key]; raw = (root/ref['path']).read_bytes()
            require(sha(raw) == ref['sha256'], 'original native metadata fixture changed')
            original[key] = json.loads(raw)
        setup, inventory = original['semantic_setup'], original['native_path_mapping']
        validate_overlay(setup,inventory)
        rows = [x for r in setup['hand_descendant_instanceability_audit']['rounds'] for x in r['overrides']]
        require(len(rows) == 6656 and sum(x['is_instance'] for x in rows) == 104 and
                sum(not x['is_instance'] and not x['is_instanceable'] for x in rows) == 6552,
                'actual inherited-clone fixture does not reproduce recorded metadata')
        passed.append('v7_actual_'+fixture['method']+'_104_true_6552_already_false_all_editable_GREEN')
        fixtures.append((setup,inventory))
    original, inventory = fixtures[0]
    def overlay_negative(label, mutate):
        fixture = copy.deepcopy(original);mutate(fixture)
        try:
            validate_overlay(fixture,inventory)
        except ValueError:
            passed.append('v7_native_fixture_'+label+'_rejected')
        else:
            raise ValueError('invalid original metadata mutation accepted:'+label)
    def change_flag(field,value):
        return lambda s:s['hand_descendant_instanceability_audit']['rounds'][0]['overrides'][104].update({field:value})
    overlay_negative('real_proxy',change_flag('is_instance_proxy',True))
    overlay_negative('ineffective_mutation',change_flag('after_is_instance',True))
    overlay_negative('nonboolean_prior_flag',change_flag('is_instance',0))
    overlay_negative('inconsistent_prior_instanceability',change_flag('is_instance',True))
    audit = original['hand_descendant_instanceability_audit']
    some_path = audit['rounds'][0]['overrides'][104]['path']
    overlay_negative('remaining_proxy',lambda s:s['hand_descendant_instanceability_audit'].update(remaining_proxy_paths=[some_path]))
    overlay_negative('remaining_instance',lambda s:s['hand_descendant_instanceability_audit'].update(remaining_instance_paths=[some_path]))
    overlay_negative('changed_actual_paths',lambda s:s['hand_descendant_instanceability_audit'].update(actual_paths_unchanged=False))
    overlay_negative('round_out_of_order',lambda s:s['hand_descendant_instanceability_audit']['rounds'][0].update(round=1))
    mesh, owner = next(iter(audit['render_mesh_native_hand_owners'].items()))
    overlay_negative('foreign_clone_mesh_owner',lambda s:s['hand_descendant_instanceability_audit']['render_mesh_native_hand_owners'].update({mesh:owner.replace('/env_0/','/env_9/')}))
    def duplicate(s):
        a=s['hand_descendant_instanceability_audit'];a['rounds'][0]['overrides'].append(dict(a['rounds'][0]['overrides'][-1]))
        s['hand_descendant_instanceability_overrides'].append(s['hand_descendant_instanceability_overrides'][-1])
    overlay_negative('duplicate_override',duplicate)
    def unexplained(s):
        s['hand_descendant_instanceability_audit']['rounds'][0]['overrides'][0].update(is_instance=False,is_instanceable=False)
    overlay_negative('noninstance_without_earlier_round_edit',unexplained)
    return dict(status='PASS_SOFTWARE_CONTRACT_TESTS_ONLY', tests=len(passed), passed=passed,
                actual_render_evidence=False, gpu_used=False, physical_safety_certified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root_name', nargs='?')
    parser.add_argument('--root', dest='root_option')
    parser.add_argument('--producer-root', help='exact registered mixed-source V6 root; may be used without --root')
    parser.add_argument('--supervise-v7', action='store_true', help='lightweight pair closure wait, then sequential CPU audits with actual wait exits')
    parser.add_argument('--expected-oracle-sha', help='require this exact launch source SHA')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--wait-closed', type=float, default=0., metavar='SECONDS')
    parser.add_argument('--semantic-mode', choices=['exact','unique-casefold','sha256'])
    parser.add_argument('--token-prefix')
    parser.add_argument('--camera-source', default=CAMERA_SOURCE, help='frozen v4 camera only')
    parser.add_argument('--expected-controls', type=int, choices=[0,8,480])
    parser.add_argument('--output', help='ASTRA_NATIVE_EVIDENCE*.json basename under H')
    args = parser.parse_args()
    require(sha(Path(__file__).read_bytes()) == EXECUTED_SOURCE_SHA256, 'oracle changed since process load')
    if args.expected_oracle_sha:
        require(args.expected_oracle_sha == EXECUTED_SOURCE_SHA256, 'launch oracle SHA mismatch')
    if args.supervise_v7:
        return supervise_v7()
    if args.self_test:
        result = self_test()
        result['oracle_sha256'] = EXECUTED_SOURCE_SHA256
        path = write_report('ASTRA_NATIVE_EVIDENCE_V7_SELFTEST.json', result)
        print(json.dumps(dict(status=result['status'], tests=result['tests'], report=str(path))), flush=True)
        return 0
    active, registration_sha = registration()
    require(not (args.root_option and args.root_name), 'specify one root argument')
    name = args.root_option or args.root_name or args.producer_root
    root = select_root(name, active)
    producer_root = select_root(args.producer_root, active) if args.producer_root else root
    require(producer_root == root, 'v6 primary oracle does not substitute a mirror root')
    require(args.camera_source == CAMERA_SOURCE and args.semantic_mode in (None, 'sha256') and
            args.token_prefix in (None, TOKEN_PREFIX), 'v6 requires frozen camera v4, sha256, qv4_hand_')
    require(args.expected_controls in (None, 480), 'v6 requires complete 480 controls per method')
    deadline = time.monotonic()+args.wait_closed
    ready, all_closures = pair_closures(active)
    while not ready and time.monotonic() < deadline:
        print('PENDING_BOTH_PRODUCERS_CLOSED', str(root), flush=True)
        time.sleep(min(10., max(0., deadline-time.monotonic())))
        ready, all_closures = pair_closures(active)
    closure = all_closures[root.name]
    closed = ready
    if not closed:
        report = dict(schema='safeduo.independent_native_visible_evidence.v7', root=str(root),
                      producer_closure=closure, phase='PROVISIONAL_PRODUCER_NOT_CLOSED', final_verdict=None,
                      raw_evidence_read=False, all_producer_closures=all_closures,
                      registration_sha256=registration_sha, physical_safety_certified=False, gpu_used=False,
                      oracle_sha256=EXECUTED_SOURCE_SHA256, data_integrity_verdict=None,
                      registered_resource_verdict=None, registered_full_run_acceptance='BLOCKED')
        filename = (args.output or OUTPUT_REPORT_NAMES[root.name]).removesuffix('.json')+'_PENDING.json'
        print(write_report(filename, report), flush=True)
        return 3
    closure['producer_root'] = str(producer_root)
    audit = Audit(root)
    report = dict(schema='safeduo.independent_native_visible_evidence.v7',
        started_utc=datetime.now(timezone.utc).isoformat(), root=str(root), producer_closure=closure,
        oracle_sha256=EXECUTED_SOURCE_SHA256, base_oracle_sha256=BASE_ORACLE_SHA256,
        base_v5_oracle_sha256=V5_ORACLE_SHA256, base_v6_oracle_sha256=V6_ORACLE_SHA256,
        deinstance_contract_correction='Prior flags are fresh write-time state, not round-start selection. Editable already-deinstanced inherited clones are allowed; proxy and ineffective changes remain rejected.',
        all_producer_closures=all_closures, registration_sha256=registration_sha,
        independent_of_producer_helper_imports=True,
        physical_safety_certified=False, gpu_used=False, published=False,
        limits=['Evidence sufficiency is not physical safety, mesh completeness or lack of occlusion.',
                'Offline audit authenticates recorded bytes and frozen source binding; it cannot attest a live renderer beyond producer recordings.',
                'Point chunks record full q/qd/root/root velocity/full targets/clock, but no link transforms or environment origins; those fields are independently checked across every held render.',
                'Collider inventory is composed env0 plus native sensor/filter paths for all64; not a separate per-clone USD collider dump.',
                'Raw measurements are reused from immutable resource-failed V5; registered full-run acceptance remains BLOCKED even when original data pass.',
                'V4 deinstancing is conditional renderer/physics setup; its numerical physics footprint is not certified.',
                'Azimuth contract requires three images and at least one pair separated by >=30 degrees, as registered (two distinct azimuths).'])
    source_bytes = EXECUTED_SOURCE_BYTES
    source_receipt = H/('ASTRA_NATIVE_EVIDENCE_SOURCE_'+sha(source_bytes)[:16]+'.json')
    if not source_receipt.exists():
        write_report(source_receipt.name, dict(sha256=sha(source_bytes), source_utf8=source_bytes.decode(),
                                              scope='sidecar source snapshot only; not a producer verdict'))
    report['oracle_source_archive'] = str(source_receipt)
    try:
        report['source_integrity'] = source_check(audit, closure)
    except Exception as exc:
        audit.issue('source_integrity', root, str(exc))
    resource = dict(verdict='FAILED' if root.name == 'raw' else 'UNVERIFIED')
    resource_errors = []
    try:
        resource = resource_check(audit,closure)
    except Exception as exc:
        resource_errors.append(type(exc).__name__+':'+str(exc))
        if root.name == 'raw':
            audit.issue('raw_resource_failure_immutable_binding',root,str(exc))
    report['resource_evidence'] = resource
    report['resource_verification_errors'] = resource_errors
    try:
        report['complete_export_byte_integrity'] = audit.complete_exports()
    except Exception as exc:
        audit.issue('complete_original_export_bytes',root,type(exc).__name__+':'+str(exc))
    argv = closure.get('argv') or []
    controls = args.expected_controls
    if controls is None:
        controls = 0 if '--initial-camera-only' in argv else (int(argv[argv.index('--controls')+1]) if '--controls' in argv else (8 if 'integration' in root.name else 480))
    require(controls == 480, 'v6 registered denominator is exactly 480 controls')
    camera_source, mode, prefix = CAMERA_SOURCE, 'sha256', TOKEN_PREFIX
    report.update(declared_control_steps=controls, camera_source=camera_source,
                  semantic_mode=mode, semantic_token_prefix=prefix)
    protocol_path = root/'response_protocol.json'
    if protocol_path.exists():
        protocol = audit.js(protocol_path)
        report['producer_protocol_claim'] = {k:protocol.get(k) for k in ['status','completed_steps','physics_events','error']}
        if closed:
            audit.check(protocol.get('status') == 'complete' and protocol.get('completed_steps') == controls and protocol.get('physics_events') == 960,
                        'complete_registered_window', protocol_path, 'producer did not complete declared control window')
    elif closed:
        audit.issue('complete_registered_window', protocol_path, 'missing final producer protocol; completed controls unverified')
    if closed:
        audit.check(closure['actual_exit'] == 0, 'producer_exit', root, 'producer exited '+str(closure['actual_exit']))
    native, first, geometry_first = None, {}, {}
    try:
        report['owner_coverage'] = audit.setup()
        constructor = Audit(root/'constructor_probe')
        try:
            constructor.body, constructor.joints = audit.body, audit.joints
            constructor.identity = constructor.js(constructor.root/'native_contact_identity.json')
            require(constructor.identity['views'] == audit.identity['views'] and constructor.identity['inventory'] == audit.identity['inventory'], 'constructor contact identity differs from main all82 inventory')
            report['constructor_explicit_probe'] = constructor.forces(1, constructor=True)
        except Exception as exc:
            audit.issue('constructor_explicit_probe', root, type(exc).__name__+':'+str(exc))
        audit.ledger.update(constructor.ledger)
        audit.checks.update(constructor.checks)
        if controls:
            try:
                report['raw_points'], native, first, geometry_first = audit.forces(controls)
            except Exception as exc:
                audit.issue('raw_points_and_trigger', root, type(exc).__name__+':'+str(exc))
        else:
            empty = audit.js(root/'point_contact_receipts.json')
            require(empty['control_steps'] == 0 and empty['physics_events'] == 0 and empty['chunks'] == [], 'initial-only probe contains controller point events')
            report['initial_only_zero_controller_events_verified'] = True
        report['camera_groups'] = audit.cameras(controls, native, first, geometry_first, mode, prefix, camera_source, closed)
    except Exception as exc:
        audit.issue('raw_evidence_availability', root, type(exc).__name__+':'+str(exc))
    groups = report.get('camera_groups', [])
    report['observed_denominators'] = dict(closed_state_groups=len(groups), independently_qualified_groups=sum(x['accepted'] for x in groups),
        declared_control_steps=controls, verified_main_microsteps=0 if report.get('initial_only_zero_controller_events_verified') else report.get('raw_points', {}).get('events'),
        controls_complete_verified=bool('raw_points' in report and not any(x['category']=='complete_registered_window' for x in audit.errors)),
        state_groups_by_kind=dict(Counter(x['kind'] for x in groups)))
    # Orphans are partial records, never promoted to a completed camera group.
    all_attempts = list((root/'qualified_views').glob('*/*/env_*/*.attempt.json'))
    group_dirs = {str((root/x['path']).parent) for x in groups}
    report['partial_attempt_files_without_closed_state'] = [str(x.relative_to(root)) for x in all_attempts if str(x.parent) not in group_dirs]
    end_closure = producer_closure(producer_root)
    if closed:
        audit.check(end_closure['closed'] and end_closure.get('pid') == closure.get('pid') and
                    end_closure.get('start_ticks') == closure.get('start_ticks') and
                    end_closure.get('actual_exit') == closure.get('actual_exit'), 'producer_still_closed', root, 'closure changed during audit')
        if producer_root != root:
            for path, recorded in list(audit.ledger.items()):
                p = Path(path)
                if p.is_relative_to(root):
                    try:
                        actual = sha(audit.data(producer_root/p.relative_to(root)))
                        audit.check(actual == recorded['sha256'], 'mirror_sha_binding', p, 'NAS consumed evidence differs from closed SSD producer')
                    except OSError as exc:
                        audit.issue('mirror_sha_binding', p, exc)
        for path, recorded in audit.ledger.items():
            try:
                s = Path(path).stat()
                audit.check((s.st_size, s.st_mtime_ns) == (recorded['size'], recorded['mtime_ns']),
                            'input_stability', path, 'evidence changed during audit')
            except OSError as exc:
                audit.issue('input_stability', path, exc)
    audit.check(sha(Path(__file__).read_bytes()) == EXECUTED_SOURCE_SHA256, 'oracle_source_stable',
                __file__, 'source changed after process load')
    _, final_registration_sha = registration()
    audit.check(final_registration_sha == registration_sha, 'registration_stable', root, 'v6 registration changed')
    report.update(evidence_verdict(root.name,not audit.errors,resource['verdict']))
    report.update(finished_utc=datetime.now(timezone.utc).isoformat(), checks=dict(audit.checks),
        failures=audit.errors, artifact_sha256=audit.ledger,
        phase='FINAL_CLOSED_PRODUCER_AUDIT' if closed else 'PROVISIONAL_PRODUCER_NOT_CLOSED',
        new_independent_initial_states_added_by_reusing_raw=0,
        original_resource_failure_execution_sha256=RAW_EXECUTION_SHA256)
    filename = args.output or OUTPUT_REPORT_NAMES[root.name]
    if not closed:
        filename = filename.removesuffix('.json')+'_PENDING.json'
    path = write_report(filename, report)
    print(json.dumps(dict(report=str(path), final_verdict=report['final_verdict'], phase=report['phase'],
                          failures=len(audit.errors), data_integrity_verdict=report['data_integrity_verdict'],
                          registered_resource_verdict=report['registered_resource_verdict'], denominators=report['observed_denominators'])), flush=True)
    return (0 if report['conditional_comparison_eligible'] else 2) if closed else 3


if __name__ == '__main__':
    sys.exit(main())
