"""Retrospective grouped identification; reads sealed inputs and never rewrites them."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'safety_feasible_guard_20261005_2100'
DT = .016666
KEYS = ['q', 'q_initial', 'pre_qd_compact', 'actuator_target',
        'controller_target', 'pre_pending_actuator_targets', 'official_margins']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(name, value):
    with (HERE / name).open('x') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load(row):
    path = Path(row['path']) / 'cell_001.npz'
    digest = sha(path)
    assert digest == row['input_sha256']['cell_001.npz']
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in KEYS}
    assert sha(path) == digest
    for k, a in data.items():
        assert np.isfinite(a).all(), k
    data['pre_q'] = np.concatenate([data['q_initial'][None], data['q'][:-1]])
    assert data['q'].shape == (960, 64, 26)
    assert np.array_equal(data['actuator_target'], data['pre_pending_actuator_targets'][:, 0])
    for k in range(6):
        assert np.array_equal(data['pre_pending_actuator_targets'][:960-k, k], data['actuator_target'][k:])
    return data, dict(path=str(path), sha256=digest, mode=row['mode'], seed=row['seed'])


def step(q, v, u, cq, cv):
    error = u - q
    return (q + cq[:, 0] * v + cq[:, 1] * error + cq[:, 2],
            cv[:, 0] * v + cv[:, 1] * error + cv[:, 2])


def metrics(error):
    assert np.isfinite(error).all()
    a = error.reshape(-1, 26)
    return dict(samples=int(a.shape[0]), rmse_rad=float(np.sqrt(np.mean(a*a))),
                maximum_abs_rad=float(np.abs(a).max()), q99_abs_rad=float(np.quantile(np.abs(a), .99)),
                per_joint_rmse_rad=np.sqrt(np.mean(a*a, axis=0)).tolist(),
                per_joint_max_abs_rad=np.abs(a).max(0).tolist(),
                per_joint_q99_abs_rad=np.quantile(np.abs(a), .99, axis=0).tolist())


def evaluate(data, cq, cv):
    q, v = data['pre_q'].astype(np.float64), data['pre_qd_compact'].astype(np.float64)
    u = data['actuator_target'].astype(np.float64)
    predicted, predicted_v = step(q, v, u, cq, cv)
    out = {'one_step': {}, 'recursive': {}, 'bounds': {}}
    errors = {'constant_velocity': q + DT*v - data['q'],
              'instant_target_catchup': u - data['q'],
              'diagonal_affine_damped_PD': predicted - data['q']}
    for name, error in errors.items():
        out['one_step'][name] = dict(all_frames=metrics(error), after_zero_prefix=metrics(error[60:]))
    failed = (data['official_margins'] < 0).any(-1)
    first = np.zeros_like(failed)
    for env in range(64):
        ids = np.flatnonzero(failed[:, env])
        if ids.size:
            first[ids[0], env] = True
    for name, error in errors.items():
        out['one_step'][name]['first_negative_cases'] = int(first.sum())
        if first.any():
            out['one_step'][name]['first_negative_error'] = metrics(error[first])
    out['post_velocity_error_rad_s'] = metrics(predicted_v[:-1] - data['pre_qd_compact'][1:])
    out['post_velocity_metric_units'] = 'rad/s; metrics function suffix retained but values have this unit'
    for horizon in [1, 6, 12, 18]:
        length = 961-horizon
        truth = data['q'][horizon-1:]
        for policy in ['actual_future_applied_ORACLE', 'known_FIFO_then_hold_current_decision']:
            state, velocity = q[:length].copy(), v[:length].copy()
            for k in range(horizon):
                target = u[k:k+length] if policy == 'actual_future_applied_ORACLE' or k < 6 else data['controller_target'][:length]
                state, velocity = step(state, velocity, target, cq, cv)
            key = f'h{horizon}_{policy}'
            error = state-truth
            out['recursive'][key] = dict(all_frames=metrics(error), after_zero_prefix=metrics(error[60:]))
            out['bounds'][key] = np.abs(error).reshape(-1, 26).max(0).tolist()
    return out


def main():
    design = json.loads((HERE/'DESIGN.json').read_text())
    scope = json.loads((HERE/'PREDICTOR_SCOPE_REGISTRATION.json').read_text())
    source_digest = sha(Path(__file__))
    results = json.loads((OLD/'holdout_results.json').read_text())
    split = {i: [r for r in results['rows'] if f'/holdout_{i}/' in r['path']] for i in range(3)}
    assert [len(split[i]) for i in range(3)] == [4, 4, 4]
    train, bindings = [], []
    for row in split[0]:
        d, b = load(row);train.append(d);bindings.append(b)
    q = np.concatenate([d['pre_q'][:-1] for d in train]).astype(np.float64)
    v = np.concatenate([d['pre_qd_compact'][:-1] for d in train]).astype(np.float64)
    u = np.concatenate([d['actuator_target'][:-1] for d in train]).astype(np.float64)
    yq = np.concatenate([d['q'][:-1] for d in train]).astype(np.float64)-q
    yv = np.concatenate([d['pre_qd_compact'][1:] for d in train]).astype(np.float64)
    x = np.stack([v, u-q, np.ones_like(q)], -1).reshape(-1, 26, 3)
    yq, yv = yq.reshape(-1, 26), yv.reshape(-1, 26)
    cq, cv, singular_values = [], [], []
    for j in range(26):
        c, _, rank, singular = np.linalg.lstsq(x[:, j], np.stack([yq[:, j], yv[:, j]], -1), rcond=None)
        assert rank == 3 and np.isfinite(c).all()
        cq.append(c[:, 0]);cv.append(c[:, 1]);singular_values.append(singular.tolist())
    cq, cv = np.array(cq), np.array(cv)
    matrix = np.stack([1-cq[:, 1], cq[:, 0], -cv[:, 1], cv[:, 0]], -1).reshape(26, 2, 2)
    radii = np.abs(np.linalg.eigvals(matrix)).max(-1)
    model = dict(status='FIT_FROZEN_BEFORE_CALIBRATION_AND_VALIDATION', dt=DT,
                 feature_order=['pre_qd_rad_s', 'applied_target_minus_pre_q_rad', 'constant'],
                 joint_order=['F_L7', 'F_R7', 'U_L6', 'U_R6'], c_q=cq.tolist(), c_v=cv.tolist(),
                 rows_per_joint=int(x.shape[0]), homogeneous_spectral_radius=radii.tolist(),
                 least_squares_singular_values=singular_values, input_bindings=bindings,
                 fit_frames=[0, 958], source_sha256=source_digest,
                 design_sha256=sha(HERE/'DESIGN.json'), scope_sha256=sha(HERE/'PREDICTOR_SCOPE_REGISTRATION.json'),
                 scope='empirical development model, no physical/conservative certification', utc=datetime.now(timezone.utc).isoformat())
    write('MODEL_FIT.json', model);fit_digest=sha(HERE/'MODEL_FIT.json')
    print('MODEL_FROZEN',fit_digest,'spectral_radius_max',radii.max(),flush=True)
    all_rows=[];calibration_bounds={}
    for block in range(3):
        for row in split[block]:
            d, binding=load(row);report=evaluate(d, cq, cv)
            all_rows.append(dict(block=block, **binding, report=report))
            if block==1:
                for key, value in report['bounds'].items():
                    calibration_bounds[key]=np.maximum(calibration_bounds.get(key, np.zeros(26)), value)
            print('PREDICTOR_EVALUATED',block,row['mode'],'PD_RMSE',report['one_step']['diagonal_affine_damped_PD']['after_zero_prefix']['rmse_rad'],flush=True)
        if block==1:
            write('MODEL_CALIBRATION.json',dict(status='FROZEN_BEFORE_RETROSPECTIVE_VALIDATION',model_sha256=fit_digest,
                  block=1,absolute_residual_max_per_joint={k:v.tolist() for k,v in calibration_bounds.items()},
                  scope='observed max residual on old calibration block, not a guaranteed bound',utc=datetime.now(timezone.utc).isoformat()))
            calibration_digest=sha(HERE/'MODEL_CALIBRATION.json')
    assert sha(HERE/'MODEL_FIT.json')==fit_digest and sha(HERE/'MODEL_CALIBRATION.json')==calibration_digest
    assert sha(Path(__file__))==source_digest
    write('PREDICTOR_RESULTS.json',dict(status='COMPLETE_GROUPED_RETROSPECTIVE_PREDICTOR_VALIDATION',rows=all_rows,
          model_sha256=fit_digest,calibration_sha256=calibration_digest,source_sha256=source_digest,
          validation_is_prior_published_data=True,new_physics_windows_executed=0,
          measured_geometry_or_thresholds_modified=False,hardware_approved=False,
          scope='one-step all960frames; velocity959; recursiveh1/6 causalqueue, h12/18 separates oraclefuturetargets/helddecision mismatch; all failures retained'))
    print('PREDICTOR_COMPLETE',len(all_rows),flush=True)


if __name__=='__main__':
    main()
