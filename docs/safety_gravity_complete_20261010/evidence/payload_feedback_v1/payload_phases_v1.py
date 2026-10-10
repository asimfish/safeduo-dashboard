"""Explicit ground-truth payload task observer; no physical safety certificate.

This is isolated development code. It returns task intent and failure reasons;
stopping intent does not prove that a delayed physical actuator has stopped.
Each instance controls one two-hand payload, not a four-hand common-object task.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Observation:
    object_id: str
    sequence: int
    state_time_s: float
    now_s: float
    valid: bool
    position: tuple
    quaternion_wxyz: tuple
    linear_velocity: tuple
    angular_velocity: tuple
    size_m: tuple
    paired_hand_normal_N: tuple
    table_normal_N: float
    hand_open_error_rad: tuple


@dataclass(frozen=True)
class Intent:
    phase: str
    object_target: tuple
    open_hands: bool
    freeze_pair: bool
    failure: str | None


class PayloadPhases:
    """Close -> lift -> carry -> supported place -> open -> detach -> retreat.

    The caller supplies native post-physics observations and actual hand posture.
    Forces are supplementary observations, never a substitute for a friction
    wrench or an independent full-scene physics oracle.
    """
    PHASES = ('CLOSE', 'LIFT', 'CARRY', 'PLACE', 'RELEASE', 'RETREAT', 'DONE', 'FAILED')

    def __init__(self, object_id, initial_position, goal_xy, size_m, mass_kg,
                 max_age_s=.025, phase_timeout_s=12.):
        values = (*initial_position, *goal_xy, *size_m, mass_kg, max_age_s, phase_timeout_s)
        if not all(math.isfinite(x) for x in values) or min(*size_m, mass_kg, max_age_s, phase_timeout_s) <= 0:
            raise ValueError('invalid registered payload geometry or timing')
        if len(initial_position) != 3 or len(goal_xy) != 2 or len(size_m) != 3:
            raise ValueError('invalid registered dimensions')
        self.object_id, self.initial = object_id, tuple(initial_position)
        self.goal_xy, self.size = tuple(goal_xy), tuple(size_m)
        self.weight, self.max_age, self.timeout = mass_kg * 9.81, max_age_s, phase_timeout_s
        self.phase = 'CLOSE'
        self.phase_started = None
        self.last_sequence, self.last_time = -1, None
        self.continuous_start = None
        self.contact_lost_start = None
        self.failure = None
        self.transitions = []

    def _fail(self, reason, now):
        self.failure = reason
        self._transition('FAILED', now)

    def _transition(self, phase, now):
        self.transitions.append(dict(from_phase=self.phase, to_phase=phase, state_time_s=now))
        self.phase = phase
        self.phase_started = now
        self.continuous_start = self.contact_lost_start = None

    def _continuous(self, predicate, now, duration):
        if not predicate:
            self.continuous_start = None
            return False
        if self.continuous_start is None:
            self.continuous_start = now
        return now - self.continuous_start >= duration - 1e-9

    def _intent(self):
        z = self.initial[2]
        xy = self.initial[:2] if self.phase in ('CLOSE', 'LIFT') else self.goal_xy
        if self.phase in ('LIFT', 'CARRY'):
            z += .14  # task target reserve; acceptance height stays100mm/2s
        return Intent(self.phase, (*xy, z), self.phase in ('RELEASE', 'RETREAT', 'DONE'),
                      self.phase in ('FAILED', 'DONE'), self.failure)

    def update(self, obs):
        if self.phase in ('FAILED', 'DONE'):
            return self._intent()
        values = (*obs.position, *obs.quaternion_wxyz, *obs.linear_velocity,
                  *obs.angular_velocity, *obs.size_m, *obs.paired_hand_normal_N,
                  obs.table_normal_N, *obs.hand_open_error_rad, obs.state_time_s, obs.now_s)
        reason = None
        if (len(obs.position), len(obs.quaternion_wxyz), len(obs.linear_velocity),
            len(obs.angular_velocity), len(obs.size_m), len(obs.paired_hand_normal_N),
            len(obs.hand_open_error_rad)) != (3, 4, 3, 3, 3, 2, 2):
            reason = 'invalid_observation_shape'
        elif not obs.valid or obs.object_id != self.object_id:
            reason = 'invalid_or_wrong_object'
        elif not all(math.isfinite(x) for x in values):
            reason = 'nonfinite_observation'
        elif obs.sequence <= self.last_sequence or (self.last_time is not None and obs.state_time_s <= self.last_time):
            reason = 'duplicate_or_nonmonotonic_observation'
        elif self.last_time is not None and obs.state_time_s - self.last_time > self.max_age + 1e-9:
            reason = 'observation_gap'
        elif not 0 <= obs.now_s - obs.state_time_s <= self.max_age:
            reason = 'stale_or_future_observation'
        elif any(abs(x - y) > 1e-6 for x, y in zip(obs.size_m, self.size)):
            reason = 'geometry_identity_changed'
        elif abs(sum(x*x for x in obs.quaternion_wxyz) - 1.) > 2e-4:
            reason = 'invalid_quaternion'
        elif min(*obs.paired_hand_normal_N, obs.table_normal_N, *obs.hand_open_error_rad) < 0:
            reason = 'invalid_contact_or_posture_measurement'
        now = obs.state_time_s
        if reason:
            self._fail(reason, now)
            return self._intent()
        self.last_sequence, self.last_time = obs.sequence, now
        if self.phase_started is None:
            self.phase_started = now
        if now - self.phase_started >= self.timeout:
            self._fail('phase_timeout_' + self.phase, now)
            return self._intent()
        if self.transitions and now - self.transitions[0]['state_time_s'] >= 60.:
            self._fail('task_timeout', now)
            return self._intent()
        lift = obs.position[2] - self.initial[2]
        both = min(obs.paired_hand_normal_N) > .1
        tilt = math.degrees(math.acos(max(-1., min(1., 1 - 2*(obs.quaternion_wxyz[1]**2 + obs.quaternion_wxyz[2]**2)))))
        speed = math.sqrt(sum(v*v for v in obs.linear_velocity))
        spin = math.sqrt(sum(v*v for v in obs.angular_velocity))
        rest = speed <= .01 and spin <= .1 and tilt <= 10.
        xy_error = math.hypot(obs.position[0] - self.goal_xy[0], obs.position[1] - self.goal_xy[1])
        supported = .8*self.weight <= obs.table_normal_N <= 1.5*self.weight and -.006 < lift < .012
        if self.phase in ('LIFT', 'CARRY') and lift >= .05 and not both:
            if self.contact_lost_start is None:
                self.contact_lost_start = now
            if now - self.contact_lost_start >= .2 - 1e-9:
                self._fail('lost_paired_contact_while_airborne', now)
                return self._intent()
        else:
            self.contact_lost_start = None
        if self.phase == 'CLOSE' and self._continuous(both, now, .25):
            self._transition('LIFT', now)
        elif self.phase == 'LIFT' and self._continuous(lift >= .100 and both and rest and obs.table_normal_N <= .1, now, 2.):
            self._transition('CARRY', now)
        elif self.phase == 'CARRY' and self._continuous(xy_error <= .02 and lift >= .100 and both and rest, now, .5):
            self._transition('PLACE', now)
        elif self.phase == 'PLACE' and self._continuous(supported and rest and xy_error <= .02, now, .5):
            self._transition('RELEASE', now)
        elif self.phase == 'RELEASE':
            detached = max(obs.paired_hand_normal_N) <= .1 and max(obs.hand_open_error_rad) <= .05
            if self._continuous(detached and supported and rest and xy_error <= .02, now, .5):
                self._transition('RETREAT', now)
        elif self.phase == 'RETREAT':
            if max(obs.paired_hand_normal_N) > .1:
                self._fail('recontact_during_retreat', now)
            elif self._continuous(supported and rest and xy_error <= .02, now, 1.):
                self._transition('DONE', now)
        return self._intent()
