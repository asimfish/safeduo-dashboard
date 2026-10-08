"""Opt-in experimental routing between two validated hand-release schedules.

This routes a task command before arm safety processing. It is not a contact,
force-closure or support detector, and it does not change collision thresholds.
Thresholds are exploratory for the two-pair beam experiment, not safety limits.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Literal


class ReleaseStateUnavailable(RuntimeError):
    """No trustworthy recent object state is available to route a release."""


@dataclass(frozen=True)
class ReleaseRoutingConfig:
    tilt_max_deg: float = 3.0
    linear_speed_max_m_s: float = 0.1
    angular_speed_max_rad_s: float = 2.0
    dwell_s: float = 0.2
    max_age_s: float = 0.05
    thumb_lead_s: float = 0.1

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if self.tilt_max_deg >= 90:
            raise ValueError('tilt_max_deg must be less than 90')


@dataclass(frozen=True)
class ReleaseDecision:
    mode: Literal['thumb_lead', 'synchronous']
    thumb_lead_s: float
    selection_time_s: float
    latest_state_time_s: float
    stable_span_s: float
    trailing_samples: int
    tilt_deg: float
    linear_speed_m_s: float
    angular_speed_rad_s: float


class ReleaseRouter:
    """Latch one schedule using recent, finite state from a single task object."""

    def __init__(self, config: ReleaseRoutingConfig | None = None):
        self.config = config if config is not None else ReleaseRoutingConfig()
        self._history = deque()
        self._decision: ReleaseDecision | None = None

    def observe(self, t: float, *, quat_wxyz, linear_velocity, angular_velocity):
        vectors = (tuple(quat_wxyz), tuple(linear_velocity), tuple(angular_velocity))
        if tuple(map(len, vectors)) != (4, 3, 3):
            raise ReleaseStateUnavailable('object pose/velocity shape is invalid')
        if not math.isfinite(t) or not all(math.isfinite(x) for v in vectors for x in v):
            raise ReleaseStateUnavailable('object state is not finite')
        if self._history and t <= self._history[-1][0]:
            raise ReleaseStateUnavailable('object state timestamps must increase')
        q, linear, angular = vectors
        q_norm = math.hypot(*q)
        if q_norm <= 0 or not math.isfinite(q_norm):
            raise ReleaseStateUnavailable('object quaternion has zero norm')
        q = tuple(x/q_norm for x in q)
        tilt = math.degrees(math.acos(max(-1., min(1., 1-2*(q[1]*q[1]+q[2]*q[2])))))
        speed = math.hypot(*linear)
        omega = math.hypot(*angular)
        if not all(math.isfinite(v) for v in (tilt, speed, omega)):
            raise ReleaseStateUnavailable('derived object metrics are not finite')
        cfg = self.config
        quiet = tilt <= cfg.tilt_max_deg and speed <= cfg.linear_speed_max_m_s and omega <= cfg.angular_speed_max_rad_s
        self._history.append((t, quiet, tilt, speed, omega))
        while self._history and self._history[0][0] < t - (cfg.dwell_s + cfg.max_age_s + .1):
            self._history.popleft()

    def choose(self, now: float) -> ReleaseDecision:
        if self._decision is not None:
            return self._decision
        if not self._history or not math.isfinite(now):
            raise ReleaseStateUnavailable('release requires recent object state')
        last = self._history[-1]
        if now + 1e-9 < last[0] or now-last[0] > self.config.max_age_s:
            raise ReleaseStateUnavailable('object state is stale or from the future')
        trailing = []
        for row in reversed(self._history):
            if not row[1] or (trailing and trailing[-1][0]-row[0] > self.config.max_age_s):
                break
            trailing.append(row)
        span = trailing[0][0]-trailing[-1][0] if trailing else 0.
        mode = 'thumb_lead' if span >= self.config.dwell_s else 'synchronous'
        decision = ReleaseDecision(
            mode=mode, thumb_lead_s=self.config.thumb_lead_s if mode=='thumb_lead' else 0.,
            selection_time_s=now, latest_state_time_s=last[0], stable_span_s=span,
            trailing_samples=len(trailing), tilt_deg=last[2], linear_speed_m_s=last[3], angular_speed_rad_s=last[4],
        )
        self._decision = decision
        return decision
