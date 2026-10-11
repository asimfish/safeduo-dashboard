"""Diagonal backward-Euler PD effort on independent motor coordinates only.

The supplied inertia is a controller estimate, not a modified physical mass.
The free one-coordinate energy result does not certify contacts, coupled arms,
model constraints or saturated multi-body dynamics. The caller adds gravity
feedforward and applies the original native motor force limits afterwards.
"""
import numpy as np


def stable_motor_torque(target, position, velocity, inertia, stiffness, damping, dt):
    arrays = [np.asarray(value, dtype=np.float64) for value in
              [target, position, velocity, inertia, stiffness, damping]]
    target, position, velocity, inertia, stiffness, damping = np.broadcast_arrays(*arrays)
    if isinstance(dt, (bool, np.bool_)) or not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Positive finite physical timestep required')
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError('Finite controller data required')
    if (inertia <= 0).any() or (stiffness < 0).any() or (damping < 0).any():
        raise ValueError('Positive inertia and nonnegative gains required')
    # Compute the free backward-Euler velocity; stiffness and damping retain
    # their nominal SI force gains. No contact force or future state is fitted.
    next_velocity = (inertia * velocity + dt * stiffness * (target - position)) / (
        inertia + dt * damping + dt * dt * stiffness)
    torque = inertia * (next_velocity - velocity) / dt
    if not np.isfinite(torque).all():
        raise ValueError('Nonfinite controller result')
    return torque
