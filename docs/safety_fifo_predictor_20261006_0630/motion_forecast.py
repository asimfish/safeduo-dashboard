"""Nominal extra admission forecasts. No actuator or projection modification."""
import json
from pathlib import Path
import torch
from safeduo.safety.types import ARM_KEYS
from safeduo.baselines.base import stack_robot
from target_forecast import require_finite
from reference_envelope import reference_bounds

HORIZONS = (1, 6, 12, 18)
MODES = ('joint_reference', 'velocity_admission', 'pd_admission', 'motion_admission')
MODEL_PATH = Path(__file__).with_name('MODEL_FIT.json')
_CACHE = {}


def model_coefficients(like):
    key = (str(like.device), str(like.dtype))
    if key not in _CACHE:
        model = json.loads(MODEL_PATH.read_text())
        cq = torch.tensor(model['c_q'], dtype=like.dtype, device=like.device)
        cv = torch.tensor(model['c_v'], dtype=like.dtype, device=like.device)
        if cq.shape != (26, 3) or cv.shape != (26, 3):
            raise ValueError('frozen diagonal model joint order changed')
        require_finite('model', dict(cq=cq, cv=cv))
        _CACHE[key] = (cq, cv, float(model['dt']))
    return _CACHE[key]


def governed_proposal(state, issued, cmd, limits, box):
    bounds = {a: reference_bounds(state.q[a], issued[a], limits[a][..., 0],
              limits[a][..., 1], box, .050) for a in ARM_KEYS}
    require_finite('proposal_bounds', {a + str(i): b[i] for a,b in bounds.items() for i in (0,1)})
    proposal = {a: (issued[a] + cmd.delta_q[a].maximum(bounds[a][0]).minimum(bounds[a][1])).clamp(
                limits[a][..., 0], limits[a][..., 1]) for a in ARM_KEYS}
    require_finite('governed_proposal', proposal)
    return proposal


def motion_displacements(q, qd, pending, proposal, cq, cv, dt, horizons=HORIZONS):
    """First six targets are immutable FIFO; only h>=7 uses proposal.

    Returns (horizon,batch,joint) CV and empirical PD displacements in radians.
    PD coefficients span one control interval, so no additional dt is applied.
    """
    if len(pending) != 6 or q.shape[-1] != 26 or qd.shape != q.shape:
        raise ValueError('exact six pending targets and 26 controlled joints required')
    require_finite('motion_inputs', dict(q=q, qd=qd, pending=pending, proposal=proposal,
                   cq=cq, cv=cv, dt=torch.as_tensor(dt)))
    if not horizons or any(type(h) is not int or h < 1 for h in horizons) or dt <= 0:
        raise ValueError('invalid horizon or control interval')
    current_q, current_v = q.clone(), qd.clone()
    pd = {}
    for k in range(1, max(horizons) + 1):
        target = pending[k-1] if k <= 6 else proposal
        debt = target - current_q
        next_q = current_q + cq[:, 0] * current_v + cq[:, 1] * debt + cq[:, 2]
        next_v = cv[:, 0] * current_v + cv[:, 1] * debt + cv[:, 2]
        require_finite('pd_intermediate_' + str(k), dict(q=next_q, v=next_v, debt=debt))
        current_q, current_v = next_q, next_v
        if k in horizons:
            pd[k] = current_q - q
    cv_disp = torch.stack([qd * (dt*h) for h in horizons])
    pd_disp = torch.stack([pd[h] for h in horizons])
    require_finite('motion_displacements', dict(cv=cv_disp, pd=pd_disp))
    return cv_disp, pd_disp


def distance_forecasts(full, rows, state, issued, pending, cmd, limits, box, target_forecast, mode):
    if mode not in MODES:
        raise ValueError('unregistered motion admission mode')
    proposal = governed_proposal(state, issued, cmd, limits, box)
    cat = lambda value: torch.cat([value[a] for a in ARM_KEYS], -1)
    q, qd = cat(state.q), cat(state.qd)
    cq, cv, model_dt = model_coefficients(q)
    if abs(float(state.dt)-model_dt) > 1e-6:
        raise ValueError('model control interval differs from actual simulator')
    cv_disp, pd_disp = motion_displacements(q, qd, torch.stack([cat(t) for t in pending]),
                                          cat(proposal), cq, cv, float(state.dt))
    components = dict(target_forecast=target_forecast)
    forecasts = {}
    for name, displacements in (('cv', cv_disp), ('pd', pd_disp)):
        forecast = full.dists.clone()
        for horizon, displacement in zip(HORIZONS, displacements):
            arms = dict(zip(ARM_KEYS, displacement.split([7, 7, 6, 6], -1)))
            delta = sum(torch.einsum('nmd,nd->nm', rows.J[r], stack_robot(arms, r)) for r in ('F','U'))
            predicted = full.dists + delta
            require_finite(name+'_h'+str(horizon), dict(delta=delta, distance=predicted))
            forecast = torch.minimum(forecast, predicted)
            if horizon == 6:
                components[name+'_h6'] = predicted
        forecasts[name] = forecast
        components[name+'_forecast'] = forecast
    result = target_forecast
    if mode in ('velocity_admission', 'motion_admission'):
        result = torch.minimum(result, forecasts['cv'])
    if mode in ('pd_admission', 'motion_admission'):
        result = torch.minimum(result, forecasts['pd'])
    require_finite('combined_forecast', dict(distance=result))
    components['forecast'] = result
    components['pd_q_horizons'] = pd_disp + q
    components['cv_q_horizons'] = cv_disp + q
    return result, components
