"""CPU/native-boundary algebra; no native state correction or physical mass edit."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MotorEffort:
    torque: np.ndarray
    predicted_free_velocity: np.ndarray
    reduced_controller_mass: np.ndarray
    relative_native_mass_asymmetry: float
    minimum_reduced_mass_eigenvalue: float


def coupled_backward_euler_torque(native_mass, mapping, target, position, velocity,
                                 stiffness, damping, dt,
                                 relative_asymmetry_tolerance=5e-6):
    """Return effort on independent motors; caller handles gravity and saturation.

    Symmetrization affects only this controller estimate, not native/source mass.
    The predicted free velocity is neither a simulator write nor a gate oracle.
    """
    mass, transform = [np.asarray(x, dtype=np.float64) for x in (native_mass, mapping)]
    vectors = [np.asarray(x, dtype=np.float64) for x in
               (target, position, velocity, stiffness, damping)]
    if mass.ndim != 2 or mass.shape[0] != mass.shape[1] or transform.ndim != 2:
        raise ValueError('Square native mass and rectangular original mapping required')
    n, k = transform.shape
    if mass.shape != (n, n) or not 0 < k <= n or any(x.shape != (k,) for x in vectors):
        raise ValueError('Exact mass/motor dimensions required; no broadcasting')
    if (isinstance(dt, (bool, np.bool_)) or not np.isscalar(dt)
            or not np.isfinite(dt) or dt <= 0):
        raise ValueError('Positive finite physical timestep required')
    if (not np.isscalar(relative_asymmetry_tolerance)
            or not np.isfinite(relative_asymmetry_tolerance)
            or not 0 <= relative_asymmetry_tolerance <= 5e-6):
        raise ValueError('Registered asymmetry bound must not be relaxed')
    if not all(np.isfinite(x).all() for x in [mass, transform, *vectors]):
        raise ValueError('Finite original mapping, matrix and motor state required')
    target, position, velocity, stiffness, damping = vectors
    if (stiffness < 0).any() or (damping < 0).any():
        raise ValueError('Nonnegative original nominal gains required')
    scale = np.max(np.abs(mass))
    if scale <= 0:
        raise ValueError('Positive-definite native generalized inertia required')
    asymmetry = float(np.max(np.abs(mass - mass.T)) / scale)
    if asymmetry > relative_asymmetry_tolerance:
        raise ValueError('Native matrix asymmetry exceeds registered controller bound')
    symmetric_estimate = (mass + mass.T) * .5
    # Fail instead of flooring singular/negative physical modes or dropping rows.
    np.linalg.cholesky(symmetric_estimate)
    if np.linalg.matrix_rank(transform) != k:
        raise ValueError('Original motor mapping must have full column rank')
    reduced = transform.T @ symmetric_estimate @ transform
    np.linalg.cholesky(reduced)
    system = reduced + np.diag(dt * damping + dt * dt * stiffness)
    rhs = reduced @ velocity + dt * stiffness * (target - position)
    next_velocity = np.linalg.solve(system, rhs)
    torque = stiffness * (target - position - dt * next_velocity) - damping * next_velocity
    if not np.isfinite(torque).all() or not np.isfinite(next_velocity).all():
        raise ValueError('Nonfinite controller result')
    return MotorEffort(torque, next_velocity, reduced, asymmetry,
                       float(np.linalg.eigvalsh(reduced).min()))
