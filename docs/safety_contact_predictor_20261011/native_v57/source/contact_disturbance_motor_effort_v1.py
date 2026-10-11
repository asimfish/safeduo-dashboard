"""Implicit motor prediction with an observed generalized disturbance.

This is controller algebra, never a native state or physical parameter write.
The disturbance is a previous-step dynamics residual, not a measured contact
force. Its usefulness under native contact must be tested independently.
"""
from dataclasses import dataclass
import numpy as np
from fourhand_native_probe_v55.coupled_implicit_motor_effort_v1 import coupled_backward_euler_torque


@dataclass(frozen=True)
class DisturbedMotorEffort:
    torque: np.ndarray
    predicted_velocity: np.ndarray
    reduced_controller_mass: np.ndarray
    disturbance: np.ndarray


def observed_disturbance(previous_reduced_mass, previous_velocity,
                         current_velocity, previous_actual_motor_torque,
                         previous_effective_gravity, previous_coriolis, dt,
                         selected_motors):
    """Use actual clipped torque and previous-step inputs; mask unselected roots."""
    M = np.asarray(previous_reduced_mass, dtype=np.float64)
    arrays = [np.asarray(x, dtype=np.float64) for x in
              (previous_velocity, current_velocity, previous_actual_motor_torque,
               previous_effective_gravity, previous_coriolis)]
    mask = np.asarray(selected_motors)
    n = len(arrays[0])
    if M.shape != (n,n) or any(x.shape != (n,) for x in arrays):
        raise ValueError('Exact reduced motor dimensions required')
    if mask.shape != (n,) or mask.dtype != np.bool_:
        raise ValueError('Explicit boolean independent motor mask required')
    if isinstance(dt, (bool,np.bool_)) or not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Positive finite previous physical timestep required')
    if not all(np.isfinite(x).all() for x in [M,*arrays]):
        raise ValueError('Finite captured dynamics inputs required')
    v0,v1,tau,g,c = arrays
    residual = M @ ((v1-v0)/dt) - tau + g + c
    return np.where(mask,residual,0.)


def disturbance_backward_euler_torque(native_mass,mapping,target,position,
                                     velocity,stiffness,damping,dt,disturbance,
                                     relative_asymmetry_tolerance=5e-6):
    """Predict with the residual while preserving nominal gains and native mass."""
    free = coupled_backward_euler_torque(native_mass,mapping,target,position,
        velocity,stiffness,damping,dt,relative_asymmetry_tolerance)
    f = np.asarray(disturbance,dtype=np.float64)
    if f.shape != free.torque.shape or not np.isfinite(f).all():
        raise ValueError('Finite independent motor disturbance required')
    K,D = np.asarray(stiffness,dtype=np.float64),np.asarray(damping,dtype=np.float64)
    H = free.reduced_controller_mass + np.diag(dt*D + dt*dt*K)
    correction = np.linalg.solve(H,dt*f)
    v1 = free.predicted_free_velocity + correction
    tau = free.torque - (D+dt*K)*correction
    if not np.isfinite(v1).all() or not np.isfinite(tau).all():
        raise ValueError('Nonfinite disturbance-aware result')
    return DisturbedMotorEffort(tau,v1,free.reduced_controller_mass,f.copy())
