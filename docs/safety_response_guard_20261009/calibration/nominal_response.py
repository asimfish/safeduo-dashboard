"""Nominal frozen-mass implicit PD response, not a PhysX safety certificate.

Ignores contact/mimic constraints and changing M/bias. Every result carries
convergence and velocity-limit information for calibration against native data.
"""
import numpy as np


def advance(q, v, target, mass, kp, kd, effort, velocity_limit, bias, dt,
            max_iterations=40, tolerance=1e-7):
    q, v, target, mass, kp, kd, effort, velocity_limit, bias = [
        np.asarray(x, dtype=np.float64) for x in
        (q, v, target, mass, kp, kd, effort, velocity_limit, bias)]
    if q.ndim != 2 or any(x.shape != q.shape for x in
            (v, target, kp, kd, effort, velocity_limit, bias)):
        raise ValueError('all state/drive arrays must have batch,DOF shape')
    if mass.shape != (*q.shape, q.shape[-1]):
        raise ValueError('mass must have batch,DOF,DOF shape')
    if not all(np.isfinite(x).all() for x in
               (q, v, target, mass, kp, kd, effort, velocity_limit, bias)):
        raise ValueError('nonfinite native input cannot be predicted')
    if not np.isfinite(dt) or dt <= 0 or max_iterations < 1 or tolerance <= 0:
        raise ValueError('invalid integration/solver settings')
    if (kp < 0).any() or (kd < 0).any() or (effort <= 0).any() or (velocity_limit <= 0).any():
        raise ValueError('invalid drive limits')
    if not np.allclose(mass, mass.swapaxes(-1, -2), atol=1e-5, rtol=1e-5):
        raise ValueError('native mass is not symmetric')
    mass = .5 * (mass + mass.swapaxes(-1, -2))
    np.linalg.cholesky(mass)  # missing or indefinite dynamics are not safe defaults
    n, dof = q.shape
    diag = np.arange(dof)
    drive_slope = kp * dt + kd
    drive_offset = kp * (target - q)
    initial_momentum = np.einsum('nij,nj->ni', mass, v) - dt * bias

    def residual(w):
        torque = np.clip(drive_offset - drive_slope * w, -effort, effort)
        return np.einsum('nij,nj->ni', mass, w) - initial_momentum - dt * torque

    w = v.copy()
    converged = np.zeros(n, dtype=bool)
    iterations = np.zeros(n, dtype=np.int32)
    for iteration in range(max_iterations):
        res = residual(w)
        norm = np.max(np.abs(res), axis=1)
        newly = (~converged) & (norm <= tolerance)
        iterations[newly] = iteration
        converged |= newly
        if converged.all():
            break
        raw_drive = drive_offset - drive_slope * w
        free = np.abs(raw_drive) < effort
        derivative = mass.copy()
        derivative[:, diag, diag] += dt * drive_slope * free
        direction = np.linalg.solve(derivative, -res[..., None])[..., 0]
        # Saturation can change the active set. Backtracking prevents an
        # oscillating clipped-force Newton step from becoming a false success.
        scale = np.ones(n)
        for _ in range(24):
            proposal = w + scale[:, None] * direction
            proposal_norm = np.max(np.abs(residual(proposal)), axis=1)
            accepted = (proposal_norm <= norm * (1 - 1e-4 * scale)) | (proposal_norm <= tolerance) | converged
            if accepted.all():
                break
            scale[~accepted] *= .5
        w = np.where(converged[:, None], w, proposal)
    final_residual = np.max(np.abs(residual(w)), axis=1)
    converged |= final_residual <= tolerance
    iterations[~converged] = max_iterations
    velocity_clipped = np.abs(w) > velocity_limit
    next_v = np.clip(w, -velocity_limit, velocity_limit)
    return q + dt * next_v, next_v, dict(
        converged=converged, residual=final_residual,
        iterations=iterations, velocity_clipped=velocity_clipped,
        torque=np.clip(drive_offset - drive_slope * w, -effort, effort),
        safety_certificate=False)
