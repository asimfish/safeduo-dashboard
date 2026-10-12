"""Development input only: streaming full50 operator requests, never a safety label.

No simulator or robot is connected. State and RNG persist across chunk boundaries.
Position intent may be adversarial; velocity intent is an independent input channel
for IID processes and a finite-difference command velocity for the other processes.
Joint coupling and Cartesian collision qualification remain execution obligations.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np

SOURCE = Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613/full_motor_operating_domain_v2/DEVELOPMENT_FULL50_PROPOSALS_V2.npz')
FAMILIES = ('iid_every_control', 'iid_held_targets', 'rate_bounded_waypoints',
            'persistent_velocity', 'multisine_spectrum', 'bursts_reversals',
            'stop_restart', 'staggered_arms', 'correlated_opposed', 'boundary_traversal')
ARM_SLICES = ((0, 7), (13, 20), (26, 32), (38, 44))
HAND_SLICES = ((7, 13), (20, 26), (32, 38), (44, 50))


@dataclass
class Domain:
    names: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    cap: np.ndarray
    source_sha256: str = ''

    def __post_init__(self):
        if any(np.shape(a) != (50,) for a in (self.names, self.lower, self.upper, self.cap)):
            raise ValueError('full50 domain required')
        if (len(set(self.names.tolist())) != 50 or
                not all(np.isfinite(a).all() for a in (self.lower, self.upper, self.cap)) or
                np.any(self.upper <= self.lower) or np.any(self.cap <= 0)):
            raise ValueError('invalid domain')


def load_domain():
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    meta = json.loads((SOURCE.parent / 'FULL_MOTOR_DOMAIN_V2.json').read_text())
    if digest != meta['proposal_SHA256']:
        raise ValueError('source domain bytes changed')
    with np.load(SOURCE, allow_pickle=False) as z:
        arrays = [z[k].copy() for k in ('independent_names', 'independent_lower_rad',
                  'independent_upper_rad', 'independent_speed_cap_rad_s')]
    return Domain(*arrays, digest)


def reflect(x):
    y = np.mod(x, 2.)
    return np.where(y <= 1., y, 2. - y)


class MotionStream:
    def __init__(self, domain, family, seed, duration_s, hz=60):
        if family not in FAMILIES or not isinstance(seed, int) or seed < 0:
            raise ValueError('unknown process or invalid seed')
        if not np.isfinite(duration_s) or duration_s <= 0 or hz != 60:
            raise ValueError('positive duration and original60Hz clock required')
        n = duration_s * hz
        if abs(n - round(n)) > 1e-8:
            raise ValueError('duration must contain whole controls')
        self.domain, self.family, self.seed = domain, family, seed
        self.hz, self.total, self.index = hz, int(round(n)) + 1, 0
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.span = domain.upper - domain.lower
        self.step_cap = domain.cap / self.span / hz
        self.unit = self.rng.uniform(0., 1., 50)
        self.goal = self.rng.uniform(0., 1., 50)
        # These scales are independent per coordinate, rather than one row speed.
        self.speed_fraction = self.rng.uniform(.05, 1., 50)
        self.velocity = self.rng.uniform(-1., 1., 50) * self.speed_fraction
        self.next_change = self.rng.integers(1, 121, 50)
        self.phase = self.rng.uniform(-np.pi, np.pi, (3, 50))
        # From periods100s down to0.5s; bounds are intent, not feasible derivatives.
        self.frequency = np.exp(self.rng.uniform(np.log(.01), np.log(2.), (3, 50)))
        self.arm_delay = self.rng.integers(0, 121, 4)
        self.sign = self.rng.choice([-1., 1.], 50)
        self.previous_unit = self.unit.copy()

    def _unit_sample(self, i):
        r, f = self.rng, self.family
        if f == 'iid_every_control':
            self.unit = r.uniform(0., 1., 50)
        elif f == 'iid_held_targets':
            change = i >= self.next_change
            count = int(change.sum())
            if count:
                self.unit[change] = r.uniform(0., 1., count)
                self.next_change[change] = i + r.integers(1, 121, count)
        elif f in ('rate_bounded_waypoints', 'stop_restart', 'staggered_arms'):
            reached = np.abs(self.goal - self.unit) <= self.step_cap * self.speed_fraction
            count = int(reached.sum())
            if count:
                self.goal[reached] = r.uniform(0., 1., count)
                self.speed_fraction[reached] = r.uniform(.05, 1., count)
            step = np.clip(self.goal - self.unit, -self.step_cap * self.speed_fraction,
                           self.step_cap * self.speed_fraction)
            if f == 'stop_restart' and (i // self.hz) % 17 in (8, 9, 10):
                step[:] = 0.
            elif f == 'staggered_arms':
                for a, ((lo, hi), (hl, hh)) in enumerate(zip(ARM_SLICES, HAND_SLICES)):
                    if ((i + self.arm_delay[a]) // self.hz) % 11 < 3:
                        step[lo:hi] = 0.; step[hl:hh] = 0.
            self.unit = np.clip(self.unit + step, 0., 1.)
        elif f == 'persistent_velocity':
            change = i >= self.next_change
            count = int(change.sum())
            if count:
                self.velocity[change] = r.uniform(-1., 1., count)
                self.next_change[change] = i + r.integers(12, 181, count)
            # Reflecting boundary changes direction, rather than stalling/clipping.
            raw = self.unit + self.step_cap * self.velocity
            self.velocity[(raw < 0.) | (raw > 1.)] *= -1.
            self.unit = reflect(raw)
        elif f in ('multisine_spectrum', 'correlated_opposed'):
            t = i / self.hz
            wave = np.sin(2*np.pi*t*self.frequency + self.phase).mean(0)
            if f == 'correlated_opposed':
                wave[13:20] = -wave[0:7]
                wave[38:44] = -wave[26:32]
                # Hands are included; joint correlation is NOT Cartesian collision proof.
                wave[20:26] = -wave[7:13]; wave[44:50] = -wave[32:38]
            self.unit = .5 + .5 * wave
        elif f == 'bursts_reversals':
            if i and i % 12 == 0:
                self.sign *= -1.
                if i % 120 == 0:
                    self.speed_fraction = r.uniform(.05, 1., 50)
            self.unit = reflect(self.unit + self.step_cap * self.sign * self.speed_fraction)
        elif f == 'boundary_traversal':
            # A correlated range sweep is a separate stress stratum, not IID.
            raw = self.unit + self.step_cap * self.sign * self.speed_fraction
            self.sign[(raw < 0.) | (raw > 1.)] *= -1.
            self.unit = reflect(raw)
        return self.unit

    def read(self, count):
        if not isinstance(count, int) or count <= 0:
            raise ValueError('positive chunk size required')
        count = min(count, self.total - self.index)
        ids = np.arange(self.index, self.index + count, dtype=np.int64)
        q, v = np.empty((count, 50)), np.empty((count, 50))
        for row, i in enumerate(ids):
            unit = self._unit_sample(int(i)).copy()
            q[row] = self.domain.lower + unit*self.span
            if self.family in ('iid_every_control', 'iid_held_targets'):
                v[row] = self.rng.uniform(-1., 1., 50) * self.domain.cap
            else:
                v[row] = ((unit-self.previous_unit)*self.span*self.hz) if i else 0.
            self.previous_unit = unit
        self.index += count
        return {'sample_index': ids, 'time_s': ids/self.hz, 'requested_q': q,
                'requested_velocity': v}

    def chunks(self, count=1024):
        while self.index < self.total:
            yield self.read(count)
