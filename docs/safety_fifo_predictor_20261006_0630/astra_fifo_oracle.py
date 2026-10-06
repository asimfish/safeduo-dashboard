"""Independent NumPy FIFO/affine oracle; no fitting, policy, or simulator imports.

Index convention: q0/qd0 precede control step t. pending[0] is applied at t;
pending[D-1] at t+D-1. The target issued at t is applied at t+D and can
first affect post[t+D], i.e. horizon D+1 measured from q0.

All coefficients describe ONE control interval at model.dt_seconds. No extra
dt is inserted. Inputs use radians and seconds. This empirical model and a
frozen geometry Jacobian are not safety certificates.
"""
from dataclasses import dataclass

import numpy as np


def _finite(value, name, shape=None):
    value = np.asarray(value, dtype=np.float64)
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name}: expected {shape}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name}: nonfinite value")
    return value


@dataclass(frozen=True)
class DiagonalDynamics:
    """Per-joint rows, columns [pre_qd, applied_target-pre_q, constant].

    dq_coeff units: [s, 1, rad]; qd_coeff units: [1, 1/s, rad/s].
    q_next = q + dq_coeff @ features; qd_next = qd_coeff @ features.
    The two regressions need not obey dq = dt * qd_next.
    """

    dq_coeff: np.ndarray
    qd_coeff: np.ndarray
    dt_seconds: float

    def __post_init__(self):
        dq = _finite(self.dq_coeff, "dq_coeff")
        if dq.ndim != 2 or dq.shape[0] == 0 or dq.shape[1] != 3:
            raise ValueError("dq_coeff must have shape (joints, 3)")
        vel = _finite(self.qd_coeff, "qd_coeff", dq.shape)
        if not np.isfinite(self.dt_seconds) or self.dt_seconds <= 0:
            raise ValueError("dt_seconds must be positive and finite")
        for name, value in (("dq_coeff", dq), ("qd_coeff", vel)):
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def joints(self):
        return self.dq_coeff.shape[0]

    @classmethod
    def from_centered(cls, c_q, c_v, dt_seconds, feature_means,
                      displacement_mean, velocity_mean):
        """Optional exact conversion from y=coeff*(features-means)+y_mean.

        Means have shapes (joints,3), (joints,), (joints,). These are supplied
        metadata, never estimated here. Raw-feature fits use the constructor.
        The constant feature is 1; its mean is handled explicitly as supplied.
        """
        model = cls(c_q, c_v, dt_seconds)
        means = _finite(feature_means, "feature_means", model.dq_coeff.shape)
        qm = _finite(displacement_mean, "displacement_mean", (model.joints,))
        vm = _finite(velocity_mean, "velocity_mean", (model.joints,))
        cq, cv = model.dq_coeff.copy(), model.qd_coeff.copy()
        cq[:, 2] += qm - np.sum(cq * means, axis=1)
        cv[:, 2] += vm - np.sum(cv * means, axis=1)
        return cls(cq, cv, dt_seconds)


def fifo_targets(pending, issued):
    """Return applied targets and remaining FIFO after len(issued) pushes.

    Both inputs have shape (time, joints), including an explicit (0, joints)
    empty queue. The returned arrays own their data; inputs are never changed.
    """
    issued = _finite(issued, "issued")
    if issued.ndim != 2 or issued.shape[1] == 0:
        raise ValueError("issued must have shape (time, joints)")
    pending = _finite(pending, "pending")
    if pending.ndim != 2 or pending.shape[1] != issued.shape[1]:
        raise ValueError("pending must have shape (delay, joints)")
    stream = np.concatenate((pending, issued), axis=0)
    return stream[:len(issued)].copy(), stream[len(issued):].copy()


def rollout(model, q0, qd0, applied_targets):
    """Direct motion recurrence; returns post-step q and qd arrays."""
    shape = (model.joints,)
    q = _finite(q0, "q0", shape).copy()
    vel = _finite(qd0, "qd0", shape).copy()
    targets = _finite(applied_targets, "applied_targets")
    if targets.ndim != 2 or targets.shape[1] != model.joints:
        raise ValueError("applied_targets must have shape (time, joints)")
    qs, vs = np.empty_like(targets), np.empty_like(targets)
    with np.errstate(over="raise", invalid="raise"):
        for k, target in enumerate(targets):
            features = np.stack((vel, target - q, np.ones_like(q)), axis=1)
            q = q + np.sum(model.dq_coeff * features, axis=1)
            vel = np.sum(model.qd_coeff * features, axis=1)
            qs[k], vs[k] = q, vel
    return qs, vs


@dataclass(frozen=True)
class AffineMotion:
    horizons: tuple
    displacement_base: np.ndarray
    displacement_sensitivity: np.ndarray
    velocity_base: np.ndarray
    velocity_sensitivity: np.ndarray

    def evaluate(self, new_target):
        """Return displacement in rad and velocity in rad/s at each horizon."""
        target = _finite(new_target, "new_target", self.displacement_base.shape[1:])
        with np.errstate(over="raise", invalid="raise"):
            return (self.displacement_base + self.displacement_sensitivity * target,
                    self.velocity_base + self.velocity_sensitivity * target)


def affine_horizons(model, q0, qd0, pending, horizons=(1, 6, 12, 18), *,
                    future_target_offsets=None, future_target_gains=None):
    """Derive displacement = base + sensitivity * new_target (elementwise).

    Default future assumption: hold the new absolute target for every subsequent
    issue. Explicit alternatives use issued[j] = offsets[j] + gains[j]*new_target
    for j=0..max(horizons)-delay-1. The first issue MUST be new_target, with
    offset=0/gain=1. For a single issue with fixed later targets, set later
    gains=0; for fixed later incremental commands, use cumulative offsets and
    gains=1. Future policy decisions and soft-limit clipping are not affine.
    """
    horizons = tuple(horizons)
    if not horizons or any(isinstance(h, (bool, np.bool_)) or
                           not isinstance(h, (int, np.integer)) or h < 1
                           for h in horizons):
        raise ValueError("horizons must be positive integers")
    shape = (model.joints,)
    q0 = _finite(q0, "q0", shape)
    v0 = _finite(qd0, "qd0", shape)
    pending = _finite(pending, "pending")
    if pending.ndim != 2 or pending.shape[1] != model.joints:
        raise ValueError("pending must have shape (delay, joints)")
    steps, delay = max(horizons), len(pending)
    count = max(0, steps - delay)
    if (future_target_offsets is None) != (future_target_gains is None):
        raise ValueError("supply both future offsets and gains")
    if future_target_offsets is None:
        offsets = np.zeros((count, model.joints))
        gains = np.ones_like(offsets)
    else:
        offsets = _finite(future_target_offsets, "offsets", (count, model.joints))
        gains = _finite(future_target_gains, "gains", offsets.shape)
        if count and (np.any(offsets[0] != 0) or np.any(gains[0] != 1)):
            raise ValueError("first future issue must equal new_target")

    # For each joint x=[q,qd], x_next=A*x+B*target+c.
    a, b, c = model.dq_coeff.T
    d, e, f = model.qd_coeff.T
    A = np.empty((model.joints, 2, 2))
    A[:, 0, 0], A[:, 0, 1] = 1 - b, a
    A[:, 1, 0], A[:, 1, 1] = -e, d
    B, bias = np.stack((b, e), axis=1), np.stack((c, f), axis=1)
    base = np.stack((q0, v0), axis=1)
    sensitivity = np.zeros_like(base)
    bases, sensitivities = [], []
    with np.errstate(over="raise", invalid="raise"):
        for k in range(steps):
            offset = pending[k] if k < delay else offsets[k-delay]
            gain = np.zeros(model.joints) if k < delay else gains[k-delay]
            base = np.einsum("nij,nj->ni", A, base) + B * offset[:, None] + bias
            sensitivity = np.einsum("nij,nj->ni", A, sensitivity) + B * gain[:, None]
            # einsum does not consistently honor NumPy's floating-point flags.
            if not (np.isfinite(base).all() and np.isfinite(sensitivity).all()):
                raise FloatingPointError("nonfinite affine recurrence")
            bases.append(base.copy())
            sensitivities.append(sensitivity.copy())
    idx = np.asarray(horizons, dtype=int) - 1
    bases, sensitivities = np.asarray(bases)[idx], np.asarray(sensitivities)[idx]
    return AffineMotion(horizons, bases[:, :, 0] - q0, sensitivities[:, :, 0],
                        bases[:, :, 1], sensitivities[:, :, 1])


def displacement_halfspaces(J, distance, required_distance, base, sensitivity,
                           target_origin):
    """Frozen-J endpoint constraint for target=target_origin+increment.

    q_h-q0 = base + sensitivity*(target_origin+u).
    d_h ~= distance + J@(q_h-q0) >= required_distance iff G@u <= h.
    G=-J* sensitivity, h=distance-required_distance+J@(base+sensitivity*origin).
    J and G are m/rad, u/base/origin are rad, sensitivity is dimensionless,
    h/distance/required_distance are m. No gamma, dt, slack, or safety epsilon.
    Required distances, row masks, and geometry exemptions belong to the caller.
    """
    J = _finite(J, "J")
    if J.ndim != 2 or J.shape[1] == 0:
        raise ValueError("J must have shape (rows, joints)")
    distance = _finite(distance, "distance", (J.shape[0],))
    floor = _finite(required_distance, "required_distance", distance.shape)
    base = _finite(base, "base", (J.shape[1],))
    sensitivity = _finite(sensitivity, "sensitivity", base.shape)
    origin = _finite(target_origin, "target_origin", base.shape)
    with np.errstate(over="raise", invalid="raise"):
        return -J * sensitivity[None, :], distance - floor + J @ (base + sensitivity * origin)
