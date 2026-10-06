"""Independent NumPy scoring primitives. No parent analysis/model imports."""
import hashlib
import json

import numpy as np

CLASSES = ('cross', 'self_F', 'self_U', 'table')
MODES = ('joint_reference', 'velocity_admission', 'pd_admission', 'motion_admission')
STRICT = np.float32(0.)
DEEP = np.float32(-.005)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(array):
    """Hash shape, dtype and exact contiguous bytes, not rounded JSON numbers."""
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(json.dumps([a.dtype.str, list(a.shape)]).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def native_flags(margins):
    a = np.asarray(margins)
    require(a.dtype == np.float32 and a.ndim == 3 and a.shape[-1] == 4,
            'native float32 margins must be (steps,envs,4)')
    require(a.shape[0] > 0 and a.shape[1] > 0, 'empty trajectory is unavailable')
    require(not np.isnan(a).any() and not np.isneginf(a).any(), 'invalid native margins')
    # +inf means no retained row for that class, subject to raw-geometry binding.
    return a < STRICT, a < DEEP


def score_windows(margins):
    strict, deep = native_flags(margins)
    output = []
    for env in range(margins.shape[1]):
        record = {}
        for name, flags in (('strict', strict[:, env]), ('deep', deep[:, env])):
            any_step = flags.any(axis=1)
            indices = np.flatnonzero(any_step)
            record[name] = dict(failed=bool(any_step.any()), steps=int(any_step.sum()),
                first_step=int(indices[0]) if len(indices) else None,
                class_failed=dict(zip(CLASSES, flags.any(axis=0).tolist())),
                class_steps=dict(zip(CLASSES, flags.sum(axis=0).astype(int).tolist())))
        minima = margins[:, env].min(axis=0)
        record['minimum_nonexempt_margin_m'] = {
            name: float(value) if np.isfinite(value) else None
            for name, value in zip(CLASSES, minima)}
        record['positive_infinite_margin_class_frames'] = np.isposinf(margins[:, env]).sum(axis=0).tolist()
        output.append(record)
    return output


def aggregate(windows):
    valid = [w for w in windows if w['status'] == 'VERIFIED']
    out = dict(expected_windows=len(windows), verified_windows=len(valid),
               unavailable_or_invalid_windows=len(windows)-len(valid))
    for kind in ('strict', 'deep'):
        out[kind] = dict(failed_windows=sum(w['metrics'][kind]['failed'] for w in valid),
            env_steps=sum(w['metrics'][kind]['steps'] for w in valid),
            class_failed_windows={c: sum(w['metrics'][kind]['class_failed'][c] for w in valid) for c in CLASSES},
            class_env_steps={c: sum(w['metrics'][kind]['class_steps'][c] for w in valid) for c in CLASSES})
    out['minimum_nonexempt_margin_m'] = {}
    for c in CLASSES:
        values = [w['metrics']['minimum_nonexempt_margin_m'][c] for w in valid
                  if w['metrics']['minimum_nonexempt_margin_m'][c] is not None]
        out['minimum_nonexempt_margin_m'][c] = min(values) if values else None
    return out


def paired_comparison(cases, candidate):
    """Missing, invalid, or mismatched pairs never enter the safe group."""
    require(candidate in MODES[1:], 'unknown comparison')
    result = dict(reference=MODES[0], candidate=candidate, expected_pairs=len(cases),
                  verified_pairs=0, unavailable_or_invalid=[], strict={}, deep={})
    groups = ('rescue', 'new_failure', 'both_fail', 'both_safe')
    for kind in ('strict', 'deep'):
        result[kind] = {g: [] for g in groups}
        result[kind]['by_class'] = {c: {g: [] for g in groups} for c in CLASSES}
    def label(r, c):
        return ('both_fail' if c else 'rescue') if r else ('new_failure' if c else 'both_safe')
    for case in cases:
        r, c = case['windows'][MODES[0]], case['windows'][candidate]
        if r['status'] != 'VERIFIED' or c['status'] != 'VERIFIED':
            result['unavailable_or_invalid'].append(dict(case_id=case['case_id'], reason='window not verified'))
            continue
        identity_keys = ('q0_sha256', 'qd0_sha256', 'initial_target_sha256',
                         'tape_sha256', 'full_tape_sha256', 'bank_sha256')
        if any(r.get(k) is None or r.get(k) != c.get(k) for k in identity_keys):
            result['unavailable_or_invalid'].append(dict(case_id=case['case_id'], reason='q0/qd0/initial_target/tape/bank pairing mismatch'))
            continue
        result['verified_pairs'] += 1
        for kind in ('strict', 'deep'):
            rr, cc = r['metrics'][kind], c['metrics'][kind]
            result[kind][label(rr['failed'], cc['failed'])].append(case['case_id'])
            for cl in CLASSES:
                result[kind]['by_class'][cl][label(rr['class_failed'][cl], cc['class_failed'][cl])].append(case['case_id'])
    for kind in ('strict', 'deep'):
        result[kind]['counts'] = {g: len(result[kind][g]) for g in groups}
        result[kind]['class_counts'] = {c: {g: len(result[kind]['by_class'][c][g]) for g in groups} for c in CLASSES}
        count = result[kind]['counts']
        require(sum(count.values()) == result['verified_pairs'], 'pair partition error')
        result[kind]['reference_failed'] = count['rescue'] + count['both_fail']
        result[kind]['candidate_failed'] = count['new_failure'] + count['both_fail']
    return result


def evaluation_classes(identity):
    cls = np.asarray(identity['class_id'])
    pair = np.asarray(identity['pair_sphere_idx'])
    arms = np.asarray(identity['sphere_arm_id'])
    require(cls.ndim == 1 and pair.shape == (len(cls), 2), 'bad row identity shape')
    require(pair.dtype.kind in 'iu' and arms.dtype.kind in 'iu', 'noninteger row identity')
    require(np.isin(cls, [0, 1, 2]).all(), 'unknown source class')
    require(((pair[:, 0] >= 0) & (pair[:, 0] < len(arms))).all(), 'bad sphere index')
    require(np.isin(arms, [0, 1, 2, 3]).all(), 'unknown arm identity')
    require(np.array_equal(np.asarray(identity['pair_id']), np.arange(len(cls))), 'pair ID order changed')
    classes = np.empty(len(cls), dtype=np.int8)
    for i, code in enumerate(cls):
        classes[i] = 0 if code == 0 else (3 if code == 2 else (1 if arms[pair[i, 0]] < 2 else 2))
    return classes


def unpack_exempt(packed, rows):
    a = np.asarray(packed)
    require(a.dtype == np.uint8 and a.shape[-1] == (rows+7)//8, 'bad packed mask')
    bits = np.unpackbits(a, axis=-1)
    require(not bits[..., rows:].any(), 'nonzero packed-mask padding')
    return bits[..., :rows].astype(bool)


def reduce_geometry(distances, exemptions, classes):
    """Reconstruct four nonexempt minima from native full geometry."""
    d = np.asarray(distances)
    require(d.dtype == np.float32 and d.shape == exemptions.shape, 'bad full geometry dtype/shape')
    require(np.isfinite(d).all(), 'nonfinite full measured geometry')
    require(d.shape[-1] == len(classes), 'class/row mismatch')
    out = np.full(d.shape[:-1]+(4,), np.float32(np.inf), dtype=np.float32)
    for c in range(4):
        rows = np.flatnonzero(classes == c)
        if len(rows):
            out[..., c] = np.where(exemptions[..., rows], np.float32(np.inf), d[..., rows]).min(axis=-1)
    return out
