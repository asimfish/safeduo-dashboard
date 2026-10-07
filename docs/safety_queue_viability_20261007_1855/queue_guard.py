"""Research diagnostics; no action generation and no future safety certificate.

G @ increment <= h describes an already selected CURRENT linear set.
Pending targets are immutable observations, never cleared or rebased here.
LP tolerances apply only to the numerical solver, never to geometric endpoints.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

SOLVER_TOL = 1e-9
PROJECTION_TOL = 1e-6  # Original projector tolerance, not a geometry epsilon.


def finite(name, value):
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"nonfinite {name}")
    return result


def diagnose_set(G, h, lower, upper, alpha_G=None, alpha_h=None):
    """Numerical feasibility plus a necessary row test, with an explicit UNKNOWN.

    Passing every individual row is insufficient for their intersection.
    A zero-J row is retained: 0 <= negative h is an actual contradiction.
    """
    G, h = finite('G', G), finite('h', h)
    lower, upper = finite('lower', lower), finite('upper', upper)
    if G.ndim != 2 or h.shape != (len(G),):
        raise ValueError('matrix/RHS shape')
    n = G.shape[1]
    if n == 0 or lower.shape != (n,) or upper.shape != (n,) or np.any(lower > upper):
        raise ValueError('invalid command bounds')
    if (alpha_G is None) != (alpha_h is None):
        raise ValueError('both alpha matrix and RHS required')
    if alpha_G is not None:
        alpha_G, alpha_h = finite('alpha_G', alpha_G), finite('alpha_h', alpha_h)
        if alpha_G.ndim != 2 or alpha_G.shape[1] != n or alpha_h.shape != (len(alpha_G),):
            raise ValueError('alpha shape')
        A, b = np.vstack((G, alpha_G)), np.r_[h, alpha_h]
    else:
        A, b = G, h
    row_min = np.where(A >= 0, A * lower, A * upper).sum(axis=1)
    individually_impossible = row_min > b
    norm = np.linalg.norm(A, axis=1)
    norm = np.where(norm > 0, norm, 1.0)
    lp = linprog(np.zeros(n), A_ub=A / norm[:, None] if len(b) else None,
                 b_ub=b / norm if len(b) else None, bounds=list(zip(lower, upper)),
                 method='highs', options={'primal_feasibility_tolerance': SOLVER_TOL,
                                          'dual_feasibility_tolerance': SOLVER_TOL})
    status = {0: 'CURRENT_SELECTED_SET_NUMERICALLY_FEASIBLE',
              2: 'CURRENT_SELECTED_SET_NUMERICALLY_INFEASIBLE'}.get(lp.status, 'UNKNOWN_SOLVER_RESULT')
    witness = None if lp.status != 0 else lp.x.tolist()
    residual = None if witness is None else float(np.maximum(A @ lp.x - b, 0).max(initial=0))
    return dict(status=status, highs_status=int(lp.status), individual_impossible_rows=int(individually_impossible.sum()),
                individual_lower_bound=float(np.maximum(row_min-b, 0).max(initial=0)),
                witness=witness, witness_max_residual=residual,
                zero_in_box=bool(np.all(lower <= 0) and np.all(upper >= 0)),
                zero_satisfies_current_rows=bool(np.all(b >= 0)),
                future_safety='UNKNOWN_NO_VALIDATED_DYNAMICS_BOUND', physical_safety_certified=False)


def queue_linear_probe(distance, dmin, jacobian, q, qd, pending, dt, valid):
    """Six frozen-J endpoint probes and CV6, all explicitly heuristic.

    CV6 uses measured qd; vmax is intentionally absent from this API.
    This neither simulates the servo nor certifies the path between endpoints.
    """
    d, dm, J = finite('distance', distance), finite('dmin', dmin), finite('J', jacobian)
    q, qd, pending = finite('q', q), finite('qd', qd), finite('pending', pending)
    valid = np.asarray(valid)
    dt = float(finite('dt', dt))
    if dt <= 0 or q.ndim != 1 or qd.shape != q.shape or pending.shape != (6, len(q)):
        raise ValueError('exact six targets and positive dt required')
    if d.ndim != 1 or dm.shape != d.shape or J.shape != (len(d), len(q)) or valid.shape != d.shape or valid.dtype != np.bool_:
        raise ValueError('row shape/mask')
    if not valid.any():
        return dict(status='UNKNOWN_NO_RELEVANT_ROWS', future_safety='UNKNOWN', physical_safety_certified=False)
    margins = d[:, None] + J @ (pending-q).T - dm[:, None]
    cv = d + J @ (qd * (6*dt)) - dm
    return dict(status='HEURISTIC_FROZEN_J_QUEUE_PROBE',
                pending_min_margin_m=margins[valid].min(axis=0).tolist(), cv6_min_margin_m=float(cv[valid].min()),
                max_pending_debt_rad=float(np.abs(pending-q).max()),
                future_safety='UNKNOWN_NO_VALIDATED_DYNAMICS_BOUND', physical_safety_certified=False)


def issuance_status(safety_residual, alpha_residual, bound_residual, individual_lower_bound):
    """Shadow status only. UNKNOWN never becomes a SAFE flag or a zero action.

    Units differ between safety (m) and alpha/bounds (rad); 1e-6 is the
    pre-existing numeric projector tolerance for each channel, not a margin.
    """
    arrays = [finite(k, v) for k, v in zip(('safety', 'alpha', 'bounds', 'individual'),
              (safety_residual, alpha_residual, bound_residual, individual_lower_bound))]
    if any(v.shape != arrays[0].shape for v in arrays):
        raise ValueError('residual shape mismatch')
    if any(np.any(v < 0) for v in arrays):
        raise ValueError('positive-part residuals required')
    return dict(unmet_selected_constraints=np.maximum.reduce(arrays[:3]) > PROJECTION_TOL,
                individual_box_impossibility=arrays[3] > PROJECTION_TOL,
                future_safety='UNKNOWN_NO_VALIDATED_DYNAMICS_BOUND', modifies_command=False)
