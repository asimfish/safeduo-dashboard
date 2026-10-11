"""Admit closing only after fresh measured arm and delayed-target settling."""
from dataclasses import dataclass
import math


@dataclass
class GraspSettleGate:
    position_error_max_rad: float = .005
    speed_max_rad_s: float = .02
    applied_error_max_rad: float = 1e-6
    dwell_s: float = .3
    timeout_s: float = 8.
    physics_dt_s: float = .008333
    since: float | None = None
    last_time: float | None = None

    def observe(self, time_s, elapsed_s, nominal_done, position_error,
                speed, applied_error, physical_safe, fresh):
        values = (time_s, elapsed_s, position_error, speed, applied_error)
        finite = all(math.isfinite(v) for v in values)
        contiguous = (self.last_time is None or
                      abs(time_s-self.last_time-self.physics_dt_s) <= 1e-9)
        valid = (finite and fresh and contiguous and nominal_done and physical_safe
                 and 0 <= position_error <= self.position_error_max_rad
                 and 0 <= speed <= self.speed_max_rad_s
                 and 0 <= applied_error <= self.applied_error_max_rad)
        if not valid:
            self.since = None
        elif self.since is None:
            self.since = time_s
        self.last_time = time_s if finite else None
        dwell = 0. if self.since is None else time_s-self.since
        ready = valid and dwell >= self.dwell_s-1e-12
        # A single late good sample cannot override a bounded phase timeout.
        timed_out = finite and elapsed_s >= self.timeout_s and not ready
        return dict(ready=bool(ready), timed_out=bool(timed_out),
                    valid=bool(valid), continuous_settled_s=float(dwell))
