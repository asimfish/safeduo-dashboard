"""CPU-only numerical calibration of recorded full74 native macro responses.

The public entry point is teacher_forced_two_micro(Macro74, Variant).
It resets q/qd at each native control boundary, then propagates TWO model
microsteps with the SAME actual full74 target and frozen macro dynamics.
No simulator, geometry, actual torque estimator, safety gate, or fitted bound.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np

PREDICTOR = Path(__file__).resolve().parent.parent / 'next_mechanism/queue74_predictor_admission_v3.py'
PREDICTOR_SHA = 'd96d6e051bb08f15aedac1858a2439d8b2ac1ba11264f82899503d841bd5375a'
if hashlib.sha256(PREDICTOR.read_bytes()).hexdigest() != PREDICTOR_SHA:
    raise ValueError('frozen numerical predictor source SHA mismatch')
_spec = importlib.util.spec_from_file_location('_calibration_frozen_pd_v3', PREDICTOR)
_pd = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _pd
_spec.loader.exec_module(_pd)

ARMS = ('F_L', 'F_R', 'U_L', 'U_R')
WIDTHS = (19, 19, 18, 18)
UNKNOWN = dict(CONTACT='UNKNOWN', ACTUAL_DRIVE_TORQUE='UNKNOWN', TERMINAL_SET='UNKNOWN',
               GLOBAL_MODEL_ERROR_BOUND='UNKNOWN', physical_safety_certified=False)


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def array(value, shape, name, dtype=None):
    a = np.asarray(value)
    require(a.shape == shape and a.dtype.kind in 'fiu' and np.isfinite(a).all(), name + ': shape/nonfinite/type')
    require(dtype is None or a.dtype == dtype, name + ': exact native dtype required')
    return np.array(a, copy=True)


def same_bytes(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def array_sha(a):
    a = np.asarray(a)
    h = hashlib.sha256()
    h.update(repr((a.dtype.str, a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class Macro74:
    q: np.ndarray
    qd: np.ndarray
    actual_targets: np.ndarray  # exactly (2,74), float32, native-applied
    native_q: np.ndarray  # (2,74), post-native-physics at the two micro boundaries
    native_qd: np.ndarray
    mass: np.ndarray
    armature: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    gravity: np.ndarray
    coriolis: np.ndarray
    gravity_disabled: np.ndarray  # per native coordinate, actual config flags
    velocity_target: np.ndarray
    explicit_actuation: np.ndarray  # command only; NOT actual implicit drive torque
    effort_limits: np.ndarray
    dt_model: float
    pre_native_step: int
    native_steps: np.ndarray
    pre_native_time: float
    native_times: np.ndarray
    native_tick: float

    def validate(self):
        for name in ('q', 'qd'):
            array(getattr(self, name), (74,), name, np.float32)
        for name in ('actual_targets', 'native_q', 'native_qd'):
            array(getattr(self, name), (2, 74), name, np.float32)
        require(same_bytes(self.actual_targets[0], self.actual_targets[1]),
                'actual applied full74 targets differ within macro2micro')
        m = array(self.mass, (74, 74), 'mass')
        require(np.allclose(m, m.T, rtol=0, atol=1e-7), 'mass asymmetric')
        np.linalg.cholesky(m)
        for name in ('armature', 'kp', 'kd', 'gravity', 'coriolis', 'velocity_target',
                     'explicit_actuation', 'effort_limits'):
            array(getattr(self, name), (74,), name)
        for name in ('armature', 'kp', 'kd'):
            require(np.all(getattr(self, name) >= 0), 'negative ' + name)
        require(np.all(self.effort_limits > 0), 'invalid effort limits')
        require(self.gravity_disabled.shape == (74,) and self.gravity_disabled.dtype == np.bool_,
                'gravity flags must be explicit native bool74')
        require(np.isfinite([self.dt_model, self.native_tick, self.pre_native_time]).all()
                and self.dt_model > 0 and self.native_tick > 0, 'invalid dt/clock')
        require(isinstance(self.pre_native_step, (int, np.integer)), 'pre step must be integer')
        require(self.native_steps.dtype.kind in 'iu'
                and np.array_equal(self.native_steps, self.pre_native_step + np.arange(1, 3)),
                'native step alignment must be consecutive two-micro post boundaries')
        array(self.native_times, (2,), 'native times')
        require(np.allclose(self.native_times, self.pre_native_time + self.native_tick * np.arange(1, 3),
                            rtol=0, atol=1e-10), 'native time alignment mismatch')


@dataclass(frozen=True)
class Variant:
    name: str
    armature_mode: str = 'add_diagonal'
    include_coriolis: bool = True
    gravity_mode: str = 'recorded_flags'
    drive_mode: str = 'implicit_saturated'
    clock_mode: str = 'model_dt'
    mass_epoch: str = 'current_macro'
    bias_epoch: str = 'current_macro'


VARIANTS = (
    Variant('implicit_add_armature'),
    Variant('implicit_mass_as_recorded', armature_mode='as_recorded'),
    Variant('implicit_no_coriolis', include_coriolis=False),
    Variant('implicit_gravity_off', gravity_mode='off'),
    Variant('implicit_native_tick', clock_mode='native_tick'),
    Variant('implicit_initial_mass', mass_epoch='initial_macro'),
    Variant('implicit_initial_bias', bias_epoch='initial_macro'),
    Variant('explicit_saturated_diagnostic', drive_mode='explicit_saturated'),
)


def teacher_forced_two_micro(sample: Macro74, variant: Variant, *, initial_mass=None, initial_bias=None):
    """Return predictions/errors74; positive signed errors mean model minus native.

    Freezing M,C,g within each macro is unavoidable with the available capture.
    Initial-M/bias variants require explicitly supplied control0 values and still
    reset q/qd every macro. No post-step teacher reset occurs at substep1.
    """
    sample.validate()
    require(variant.armature_mode in ('as_recorded', 'add_diagonal'), 'explicit armature interpretation required')
    require(variant.gravity_mode in ('recorded_flags', 'off'), 'invalid gravity mode')
    require(variant.drive_mode in ('implicit_saturated', 'explicit_saturated'), 'invalid drive mode')
    require(variant.clock_mode in ('model_dt', 'native_tick'), 'invalid clock mode')
    require(variant.mass_epoch in ('current_macro', 'initial_macro')
            and variant.bias_epoch in ('current_macro', 'initial_macro'), 'invalid dynamics epoch')
    mass = sample.mass.astype(np.float64).copy()
    if variant.mass_epoch == 'initial_macro':
        mass = array(initial_mass, (74, 74), 'explicit initial mass').astype(np.float64)
    if variant.armature_mode == 'add_diagonal':
        mass += np.diag(sample.armature)
    gravity = sample.gravity.astype(np.float64) * ~sample.gravity_disabled
    if variant.gravity_mode == 'off':
        gravity = np.zeros(74)
    bias = gravity + (sample.coriolis if variant.include_coriolis else 0.)
    if variant.bias_epoch == 'initial_macro':
        bias = array(initial_bias, (74,), 'explicit initial bias').astype(np.float64)
    dt = sample.dt_model if variant.clock_mode == 'model_dt' else sample.native_tick
    dyn = _pd.DynamicsSample(mass, sample.kp, sample.kd, bias, sample.velocity_target,
                            sample.explicit_actuation, '0' * 64)
    q, v = sample.q.astype(np.float64), sample.qd.astype(np.float64)
    qs, vs, taus, diagnostics = [], [], [], []
    for sub in range(2):
        if variant.drive_mode == 'implicit_saturated':
            q, v, tau, d = _pd.implicit_saturated_step(q, v, sample.actual_targets[sub], dyn,
                                                      sample.effort_limits, dt)
            require(d['converged'], 'implicit saturated solve failed; errors must not be accepted')
        else:
            demand = sample.kp * (sample.actual_targets[sub].astype(float) - q) + sample.kd * (sample.velocity_target - v) + sample.explicit_actuation
            tau = np.clip(demand, -sample.effort_limits, sample.effort_limits)
            v = v + dt * np.linalg.solve(mass, tau - bias)
            q = q + dt * v
            d = dict(converged=True, iterations=1, residual_max=0.,
                     unsaturated_demand=demand, saturation_mask=np.abs(demand) > sample.effort_limits,
                     residual_meaning='explicit equation by construction, not implicit convergence')
        require(np.isfinite(q).all() and np.isfinite(v).all(), 'nonfinite result')
        qs.append(q.copy()); vs.append(v.copy()); taus.append(tau.copy()); diagnostics.append(d)
    qs, vs = np.array(qs), np.array(vs)
    return dict(predicted_q=qs, predicted_qd=vs, q_error=qs - sample.native_q,
                qd_error=vs - sample.native_qd, model_torque=np.array(taus),
                numerical=diagnostics, actual_target_sha256=array_sha(sample.actual_targets),
                teacher_forcing='each macro pre-native q/qd; propagate two model microsteps',
                **UNKNOWN)


def future_target_pairing(planned_micro74, actual_micro74):
    """A divergence invalidates that micro AND all later open-loop state errors.

    A full36 comparison requires at least36 observed native microsteps and
    bitwise-equal float32 targets at ALL36. Later coincidence cannot repair a
    divergent state history. Missing future observations are not zero error.
    """
    plan = array(planned_micro74, (36, 74), 'planned36', np.float32)
    actual = np.asarray(actual_micro74)
    require(actual.ndim == 2 and actual.shape[1] == 74 and actual.dtype == np.float32
            and np.isfinite(actual).all(), 'actual future target schema')
    n = min(36, len(actual))
    matches = [same_bytes(plan[i], actual[i]) for i in range(n)]
    first = next((i for i, ok in enumerate(matches) if not ok), None)
    valid = n if first is None else first
    return dict(full36_direct_error_enabled=n == 36 and first is None,
                observed_microsteps=n, first_target_mismatch_micro=first,
                paired_prefix_microsteps=valid,
                direct_error_valid_micro_mask=[i < valid for i in range(36)],
                unavailable_microsteps=list(range(n, 36)),
                reason='ALL36_BYTEEXACT' if n == 36 and first is None else
                'DISABLED_TARGET_MISMATCH_OR_INCOMPLETE_NATIVE_FUTURE')
