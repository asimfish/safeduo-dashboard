"""Passive original-stream forecast audit; no controller imports or mutations."""
import hashlib
import json
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent
REG = json.loads((P / 'REGISTRATION.json').read_text())
ROOT = Path(REG['root'])
HORIZONS = REG['horizons_steps']
DT = REG['control_dt_s']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def summarize_error(predicted, actual, eligible, distance):
    output = {}
    bins = [('all', eligible), ('below10mm', eligible & (distance < .01)),
            ('10to50mm', eligible & (distance >= .01) & (distance < .05)),
            ('above50mm', eligible & (distance >= .05))]
    for label, mask in bins:
        errors = (predicted - actual)[mask]
        count = len(errors)
        bad = actual[mask] < 0
        alarm = predicted[mask] < 0
        output[label] = dict(observations=count, absolute_error_sum_m=float(np.abs(errors).sum()),
                             max_gap_overestimate_m=float(max(0., errors.max())) if count else 0.,
                             max_gap_underestimate_m=float(max(0., -errors.min())) if count else 0.,
                             future_negative=int(bad.sum()), predicted_negative=int(alarm.sum()),
                             missed_negative=int((bad & ~alarm).sum()),
                             false_alarms=int((~bad & alarm).sum()))
    return output


def merge(destination, source):
    for label, values in source.items():
        target = destination.setdefault(label, {k: 0 for k in values})
        for key, value in values.items():
            target[key] = max(target[key], value) if key.startswith('max_') else target[key] + value


def main():
    for name, expected in REG['source_receipts'].items():
        assert sha(ROOT / name) == expected, name
    receipt = json.loads((ROOT / 'forecast_receipts.json').read_text())
    identity = json.loads((ROOT / 'full_row_identity.json').read_text())
    assert receipt['steps'] == 960 and identity['rows'] == 9021
    cell = load(ROOT / 'cell_001.npz')
    assert float(json.loads(str(cell['meta_json']))['dt']) == DT
    classes = np.asarray(identity['class_id']).astype(int)
    pairs = np.asarray(identity['pair_sphere_idx'])
    arms = np.asarray(identity['sphere_arm_id'])
    labels = classes.copy()
    labels[(classes == 1) & (arms[pairs[:, 0]] >= 2)] = 2
    labels[classes == 2] = 3
    firsts = []
    for env in range(64):
        steps = np.flatnonzero((cell['official_margins'][:, env] < 0).any(-1))
        if len(steps):
            firsts.append((env, int(steps[0])))
    event_receipt = json.loads((ROOT / 'first_failure_receipts.json').read_text())
    event_rows = {}
    for env, step in firsts:
        entry = next(x for x in event_receipt['receipts'] if x['step'] == step)
        path = ROOT / entry['path']
        assert sha(path) == entry['sha256']
        z = load(path)
        i = int(np.flatnonzero(z['env_ids'] == env)[0])
        bad = np.flatnonzero((z['post_full_d'][i] < 0) & ~z['post_full_exempt'][i])
        event_rows[env] = dict(step=step, bad_rows=bad.tolist(), snapshot=z, index=i)

    history = None
    total = {str(h): {} for h in HORIZONS}
    by_class = {str(h): {} for h in HORIZONS}
    events = []
    sources = []
    seen = 0
    trace_d, trace_v = [], []
    for entry in receipt['chunks']:
        path = ROOT / entry['path']
        assert sha(path) == entry['sha256']
        with np.load(path, allow_pickle=False) as z:
            d = z['measured_d']
            v = z['velocity']
            ex = np.unpackbits(z['exempt'], axis=-1, count=9021).astype(bool)
        assert entry['start'] == seen and d.shape == v.shape == ex.shape
        assert np.isfinite(d).all() and np.isfinite(v).all()
        trace_d.append(d[:, 49, [8886, 8887]].copy())
        trace_v.append(v[:, 49, [8886, 8887]].copy())
        # Reconstruct every original post-frame score independently from raw rows.
        for i in range(len(d)):
            margins = np.stack([np.where(ex[i] | (labels[None] != k), np.inf, d[i]).min(-1)
                                for k in range(4)], -1)
            step = seen + i
            if step == 0:
                assert np.array_equal((margins < 0).any(-1), cell['initial_violation'])
            else:
                assert np.array_equal(margins, cell['official_margins'][step - 1])
        old_len = 0 if history is None else len(history[0])
        combined = [d, v, ex] if history is None else [np.concatenate([a, b]) for a, b in zip(history, [d, v, ex])]
        D, V, EX = combined
        offset = seen - old_len
        for h in HORIZONS:
            origin = np.arange(max(0, old_len - h), len(D) - h)
            if not len(origin):
                continue
            actual = D[origin + h]
            distance = D[origin]
            predicted = distance.astype(np.float64) + h * DT * V[origin]
            eligible = (~EX[origin]) & (~EX[origin + h]) & (distance >= 0)
            merge(total[str(h)], summarize_error(predicted, actual, eligible, distance))
            for label in range(4):
                mask = eligible & (labels[None, None] == label)
                merge(by_class[str(h)].setdefault(str(label), {}),
                      summarize_error(predicted, actual, mask, distance))
            for env, info in event_rows.items():
                target_step = info['step'] + 1
                if info['step'] == 0 or not (seen <= target_step < seen + len(d)):
                    continue
                o = target_step - h - offset
                if o < 0:
                    continue
                for row in info['bad_rows']:
                    gap = float(D[o, env, row])
                    rate = float(V[o, env, row])
                    actual_gap = float(D[target_step - offset, env, row])
                    pred = gap + h * DT * rate
                    events.append(dict(env=env, first_failure_step=info['step'], row=row,
                                       origin_pre_step=target_step - h, target_pre_step=target_step,
                                       horizon_steps=h, origin_gap_m=gap, signed_rate_mps=rate,
                                       predicted_gap_m=pred, actual_gap_m=actual_gap,
                                       gap_overestimate_m=pred - actual_gap,
                                       predicts_negative=pred < 0, predicts_below20mm=pred < .02))
        history = [a[-max(HORIZONS):].copy() for a in combined]
        seen += len(d)
        sources.append(entry)
        print('AUDIT', seen, '/960', flush=True)
    assert seen == 960
    terminal = load(ROOT / 'post_geometry_final.npz')
    margins = np.stack([np.where(terminal['exempt'] | (labels[None] != k), np.inf,
                                terminal['d']).min(-1) for k in range(4)], -1)
    assert np.array_equal(margins, cell['official_margins'][-1])
    # Diagnostics deliberately use pre-boundary pairs t,t+h up to959 only;
    # terminal960 is checked for scoring, but has no stored velocity channel.
    for reports in [total] + list(by_class.values()):
        def average(node):
            if 'observations' in node:
                node['mean_absolute_error_m'] = node['absolute_error_sum_m'] / node['observations'] if node['observations'] else None
            else:
                for value in node.values():
                    average(value)
        average(reports)
    result = dict(status='COMPLETE_PASSIVE_KNOWN_BANK_MOTION_DIAGNOSTIC', windows=64,
                  control_steps=960, rows=9021, forecast_origins_end=959,
                  original_post_score_all_frames_exact=True,
                  first_failures=[dict(env=e, step=s) for e, s in firsts],
                  invalid_initial_cases=[e for e, s in firsts if s == 0],
                  metrics=total, metrics_by_class=by_class, first_failure_predictions=events,
                  input_chunks=sources, analysis_source_sha256=sha(Path(__file__)),
                  registration_sha256=sha(P / 'REGISTRATION.json'),
                  scope=REG['scope'], future_safety_certified=False, new_policy_trials=0,
                  limitations=['Stored signed velocities are producer observations; raw event native binding is a separate audit.',
                               'Frozen-J constant-speed estimates are not acceleration or stopping bounds.',
                               'All rows/frames are correlated; known-bank results are not independent holdout acceptance.',
                               '20mm is a passive descriptive threshold, not a reconstruction of combined controller admission.'])
    (P / 'MOTION_FORECAST_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(P / 'ENV49_TRACE.npz', d=np.concatenate(trace_d),
                        velocity=np.concatenate(trace_v), dt=DT, rows=[8886, 8887])
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
