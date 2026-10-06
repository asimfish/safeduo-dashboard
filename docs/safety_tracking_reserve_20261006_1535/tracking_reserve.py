"""Evaluation-only tracking reserve. Nominal frozen-J risk is not viability."""
import torch

from target_forecast import require_finite

ARM_KEYS = ('F_L', 'F_R', 'U_L', 'U_R')
MODES = ('joint_reference', 'tight_reference', 'delay_reserve')


def gap_from_margin(margin, mode):
    if mode not in MODES:
        raise ValueError('unregistered tracking reserve mode')
    require_finite('reserve_margin', dict(margin=margin))
    if mode == 'joint_reference':
        return torch.full_like(margin, .050)
    if mode == 'tight_reference':
        return torch.full_like(margin, .010)
    return .010 + .040 * ((margin - .010) / .040).clamp(0, 1)


def reserve_forecast(distance, dmin, exempt, jacobian, q, qd, pending, dt):
    if len(pending) != 6 or dt <= 0:
        raise ValueError('exact six actual pending targets and positive dt required')
    if distance.shape != dmin.shape or distance.shape != exempt.shape or exempt.dtype != torch.bool:
        raise ValueError('full geometry and original exemption shape required')
    if distance.ndim != 2 or not distance.shape[1]:
        raise ValueError('nonempty batch-by-row full geometry required')
    batch, width = distance.shape
    if set(jacobian) != {'F', 'U'} or set(q) != set(ARM_KEYS) or set(qd) != set(ARM_KEYS):
        raise ValueError('exact original arm and robot components required')
    for robot, joints in [('F', 14), ('U', 12)]:
        if jacobian[robot].shape != (batch, width, joints):
            raise ValueError('full signed joint Jacobian shape changed')
    for arm, joints in zip(ARM_KEYS, [7, 7, 6, 6]):
        if q[arm].shape != (batch, joints) or qd[arm].shape != q[arm].shape:
            raise ValueError('measured joint state shape changed')
    for target in pending:
        if set(target) != set(ARM_KEYS) or any(target[a].shape != q[a].shape for a in ARM_KEYS):
            raise ValueError('actual pending target shape changed')
    require_finite('reserve_inputs', dict(distance=distance, dmin=dmin, dt=torch.as_tensor(dt)))
    require_finite('reserve_J', jacobian)
    require_finite('reserve_q', q)
    require_finite('reserve_qd', qd)
    prediction = distance.clone()
    displacements = []
    for index, target in enumerate(pending):
        require_finite('reserve_pending_' + str(index), target)
        displacements.append({a: target[a] - q[a] for a in ARM_KEYS})
    displacements.append({a: qd[a] * (18 * dt) for a in ARM_KEYS})
    for index, displacement in enumerate(displacements):
        require_finite('reserve_displacement_' + str(index), displacement)
        delta = sum(torch.einsum('nmd,nd->nm', jacobian[r],
                    torch.cat([displacement[r+'_L'], displacement[r+'_R']], -1)) for r in ('F', 'U'))
        endpoint = distance + delta
        require_finite('reserve_endpoint_' + str(index), dict(delta=delta, endpoint=endpoint))
        prediction = torch.minimum(prediction, endpoint)
    risk = (prediction - dmin).masked_fill(exempt, float('inf')).amin(-1)
    require_finite('reserve_output', dict(distance=prediction, risk_margin=risk))
    return dict(distance=prediction, risk_margin=risk)
