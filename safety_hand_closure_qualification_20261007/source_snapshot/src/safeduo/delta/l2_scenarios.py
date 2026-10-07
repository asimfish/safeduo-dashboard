"""L2 scripted conflict scenario library (doubles as the eval stress suite).

Eight parameterized scenario families, each a DeltaSource subclass with a
parameter distribution that yields >=100 variants. Train/eval variants are
hard-separated by disjoint seed ranges (see TRAIN_SEED_RANGE / EVAL_SEED_RANGE);
the split can never be polluted because a variant's RNG seed is a pure function
of (scenario, split, variant_id) and the two ranges do not overlap.

Scenarios steer end-effectors via the same IntentMapper abstraction as L1, so
when the deployed retarget/IK pipeline lands, the identical scripts replay
through it. Workspace geometry (center point, table height, per-robot boxes)
comes from WorkspaceSpec; defaults match the local toy system, Agent A's env
passes the calibrated real-scene values.

Each class documents the failure mode it attacks (`attacks` attribute).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import torch

from safeduo.delta._contract_stub import (
    ARM_KEYS,
    DeltaCmd,
    DeltaSource,
    SceneState,
)
from safeduo.delta.l1_random import IntentMapper, JointSpaceMapper, L1Params, L1RandomDelta

# facing arm pairs: (F-side arm, U-side arm); index stored in params as pair_idx
PAIRS = (("F_L", "U_R"), ("F_R", "U_L"), ("F_L", "U_L"), ("F_R", "U_R"))

TRAIN_SEED_RANGE = (0, 50_000_000)
EVAL_SEED_RANGE = (50_000_000, 100_000_000)
_SCENARIO_STRIDE = 200_000  # max variants per scenario per split


@dataclass
class WorkspaceSpec:
    """Shared-workspace geometry; defaults match tests/toy_system.py.
    Agent A's env passes the calibrated real-scene values here."""
    center: tuple = (0.65, 0.0, 0.35)
    table_z: float = 0.0
    box_F: tuple = ((0.20, -0.40, 0.10), (0.65, 0.40, 0.55))
    box_U: tuple = ((0.65, -0.40, 0.10), (1.10, 0.40, 0.55))

    def box_of(self, robot: str) -> tuple:
        return self.box_F if robot == "F" else self.box_U


def _u(g: torch.Generator, lo: float, hi: float) -> float:
    return lo + (hi - lo) * torch.rand(1, generator=g).item()

def _choice(g: torch.Generator, n: int) -> int:
    return int(torch.randint(n, (1,), generator=g).item())


def variant_seed(scenario_name: str, split: str, variant_id: int) -> int:
    """Pure function (scenario, split, id) -> RNG seed; ranges are disjoint by construction."""
    assert split in ("train", "eval"), split
    assert 0 <= variant_id < _SCENARIO_STRIDE, variant_id
    names = list(SCENARIOS.keys())
    base = TRAIN_SEED_RANGE[0] if split == "train" else EVAL_SEED_RANGE[0]
    return base + names.index(scenario_name) * _SCENARIO_STRIDE + variant_id


def make_variants(scenario_name: str, split: str, n: int) -> list:
    cls = SCENARIOS[scenario_name]
    out = []
    for vid in range(n):
        g = torch.Generator().manual_seed(variant_seed(scenario_name, split, vid))
        out.append(cls.Params.sample(g))
    return out


def build_scenario(scenario_name: str, n_envs: int, mapper: IntentMapper,
                   split: str = "train", n_variants: int = 100,
                   ws: "WorkspaceSpec | None" = None,
                   device: "str | torch.device" = "cpu",
                   dof_of: "dict | None" = None):
    variants = make_variants(scenario_name, split, n_variants)
    return SCENARIOS[scenario_name](n_envs, variants, mapper, ws or WorkspaceSpec(),
                                    device, dof_of)


def scenario_catalog() -> list:
    """[(name, attacked failure mode)] for docs and reports."""
    return [(name, cls.attacks) for name, cls in SCENARIOS.items()]


class ScriptedScenario(DeltaSource):
    """Base: param tiling across envs, per-env clock, role masks, EE steering."""

    Params: type  # set by subclass
    attacks: str = ""

    def __init__(self, n_envs: int, variants: list, mapper: IntentMapper,
                 ws: WorkspaceSpec, device: "str | torch.device" = "cpu",
                 dof_of: "dict | None" = None):
        self.n = n_envs
        self.mapper = mapper
        self.ws = ws
        self.dof_of = dof_of  # None -> contract DOF_OF (inner sources); out shapes follow state
        self.device = torch.device(device)
        self.gen: "torch.Generator | None" = None
        self.variants = variants
        # tile scalar params over envs: env i uses variants[i % V]
        self.pv = {}
        for f in dataclasses.fields(self.Params):
            vals = [float(getattr(v, f.name)) for v in variants]
            t = torch.tensor(vals, dtype=torch.float32, device=self.device)
            self.pv[f.name] = t[torch.arange(n_envs, device=self.device) % len(vals)]
        self.t = torch.zeros(n_envs, device=self.device)
        self._start_ee = {a: torch.full((n_envs, 3), float("nan"), device=self.device)
                          for a in ARM_KEYS}
        # role masks per arm key if params carry pair_idx
        if "pair_idx" in self.pv:
            pidx = self.pv["pair_idx"].long()
            self.m_F = {a: torch.zeros(n_envs, dtype=torch.bool, device=self.device)
                        for a in ARM_KEYS}
            self.m_U = {a: torch.zeros(n_envs, dtype=torch.bool, device=self.device)
                        for a in ARM_KEYS}
            for i, (fa, ua) in enumerate(PAIRS):
                self.m_F[fa] |= pidx == i
                self.m_U[ua] |= pidx == i

    def reset(self, env_ids: torch.Tensor, generator: "torch.Generator | None" = None) -> None:
        if generator is not None:
            self.gen = generator
        ids = env_ids.to(self.device)
        self.t[ids] = 0.0
        for a in ARM_KEYS:
            self._start_ee[a][ids] = float("nan")
        self._reset_extra(ids)

    def _reset_extra(self, ids: torch.Tensor) -> None:
        pass

    def _capture_start(self, state: SceneState) -> None:
        for a in ARM_KEYS:
            nan = torch.isnan(self._start_ee[a]).any(dim=-1)
            if nan.any():
                self._start_ee[a] = torch.where(nan.unsqueeze(-1), state.ee_pos[a],
                                                self._start_ee[a])

    def _ee_step(self, state: SceneState, arm: str, mask: torch.Tensor,
                 target: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        """Joint delta steering `arm` toward `target` at `speed`, zero outside mask."""
        tgt = torch.where(mask.unsqueeze(-1), target, state.ee_pos[arm])
        d = self.mapper.map(arm, state, tgt, state.dt, speed=speed)
        return torch.where(mask.unsqueeze(-1), d, torch.zeros_like(d))

    def _center(self, state: SceneState) -> torch.Tensor:
        return torch.tensor(self.ws.center, device=state.device,
                            dtype=torch.float32).expand(self.n, 3)

    def sample(self, state: SceneState) -> DeltaCmd:
        self._capture_start(state)
        out = {a: torch.zeros_like(state.q[a]) for a in ARM_KEYS}
        self._compute(state, out)
        self.t = self.t + state.dt
        return DeltaCmd(delta_q=out)

    def _compute(self, state: SceneState, out: dict) -> None:
        raise NotImplementedError


class HeadOnCrossing(ScriptedScenario):
    """Both facing arms drive straight through the shared center to mirrored goals."""

    attacks = ("symmetric head-on closing with ambiguous right-of-way -> myopic filters "
               "either deadlock face-to-face or oscillate braking, never committing a yield")

    @dataclass
    class Params:
        pair_idx: int
        speed_F: float
        speed_U: float
        lateral_off: float   # lateral separation of the two crossing paths (m)
        phase_delay_U: float  # U starts late by this many seconds
        depth: float          # how far past center the goal sits (fraction)

        @staticmethod
        def sample(g):
            return HeadOnCrossing.Params(
                pair_idx=_choice(g, 4), speed_F=_u(g, 0.15, 0.45), speed_U=_u(g, 0.15, 0.45),
                lateral_off=_u(g, 0.0, 0.08), phase_delay_U=_u(g, 0.0, 0.6),
                depth=_u(g, 0.8, 1.3))

    def _compute(self, state, out):
        cx = self.ws.center[0]
        for side, m_of, spd_key, delay in (
            ("F", self.m_F, "speed_F", None), ("U", self.m_U, "speed_U", "phase_delay_U")):
            for arm in ARM_KEYS:
                mask = m_of[arm]
                if not mask.any():
                    continue
                if delay is not None:
                    mask = mask & (self.t >= self.pv[delay])
                start = self._start_ee[arm]
                tgt = start.clone()
                # reflect across the center plane x = cx, overshoot by depth
                tgt[:, 0] = cx + (cx - start[:, 0]) * self.pv["depth"]
                off = self.pv["lateral_off"] * (1.0 if side == "F" else -1.0)
                tgt[:, 1] = start[:, 1] + off
                out[arm] = out[arm] + self._ee_step(state, arm, mask, tgt, self.pv[spd_key])


class CenterGrab(ScriptedScenario):
    """Both arms rush the same central point simultaneously, re-jittered periodically."""

    attacks = ("contested single resource: both intents are legitimate, so distance-only "
               "filters brake both arms symmetrically and neither ever wins the center")

    @dataclass
    class Params:
        pair_idx: int
        speed_F: float
        speed_U: float
        jitter: float         # radius of target re-jitter around center (m)
        regrab_period: float  # re-jitter interval (s)

        @staticmethod
        def sample(g):
            return CenterGrab.Params(
                pair_idx=_choice(g, 4), speed_F=_u(g, 0.2, 0.5), speed_U=_u(g, 0.2, 0.5),
                jitter=_u(g, 0.0, 0.06), regrab_period=_u(g, 1.0, 3.0))

    def _reset_extra(self, ids):
        if not hasattr(self, "_jit"):
            self._jit = torch.zeros(self.n, 3, device=self.device)
            self._t_grab = torch.zeros(self.n, device=self.device)
        self._jit[ids] = 0.0
        self._t_grab[ids] = 0.0

    def _compute(self, state, out):
        if not hasattr(self, "_jit"):
            self._reset_extra(torch.arange(self.n, device=self.device))
        renew = self.t - self._t_grab >= self.pv["regrab_period"]
        if renew.any():
            raw = torch.rand(self.n, 3, device=self.device, generator=self.gen) * 2.0 - 1.0
            self._jit = torch.where(renew.unsqueeze(-1),
                                    raw * self.pv["jitter"].unsqueeze(-1), self._jit)
            self._t_grab = torch.where(renew, self.t, self._t_grab)
        tgt = self._center(state) + self._jit
        for m_of, spd in ((self.m_F, "speed_F"), (self.m_U, "speed_U")):
            for arm in ARM_KEYS:
                if m_of[arm].any():
                    out[arm] = out[arm] + self._ee_step(state, arm, m_of[arm], tgt,
                                                        self.pv[spd])


class HandoverApproach(ScriptedScenario):
    """Receiver closes to a sub-d_soft standoff from the giver EE and hovers there."""

    attacks = ("intentional sustained close proximity (cooperative handover): filters that "
               "cannot separate cooperation from collision course fire false E-stops and "
               "wreck demonstration quality")

    @dataclass
    class Params:
        pair_idx: int
        giver_is_F: int       # 1: F gives (holds), U approaches; 0: reverse
        approach_speed: float
        offset_dist: float    # standoff distance, deliberately below d_soft (m)
        drift_speed: float    # giver slow drift speed (m/s)
        drift_period: float   # giver drift direction flip period (s)

        @staticmethod
        def sample(g):
            return HandoverApproach.Params(
                pair_idx=_choice(g, 4), giver_is_F=_choice(g, 2),
                approach_speed=_u(g, 0.08, 0.3), offset_dist=_u(g, 0.02, 0.06),
                drift_speed=_u(g, 0.0, 0.05), drift_period=_u(g, 1.5, 4.0))

    def _compute(self, state, out):
        giver_F = self.pv["giver_is_F"] > 0.5
        for arm in ARM_KEYS:
            m_recv = (self.m_U[arm] & giver_F) | (self.m_F[arm] & ~giver_F)
            m_give = (self.m_F[arm] & giver_F) | (self.m_U[arm] & ~giver_F)
            if m_recv.any():
                # locate this receiver's giver EE (its paired arm)
                giver_ee = torch.zeros(self.n, 3, device=self.device)
                for i, (fa, ua) in enumerate(PAIRS):
                    sel = self.pv["pair_idx"].long() == i
                    other = fa if arm == ua else ua
                    giver_ee = torch.where(sel.unsqueeze(-1), state.ee_pos[other], giver_ee)
                gap = state.ee_pos[arm] - giver_ee
                dirn = gap / gap.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                tgt = giver_ee + dirn * self.pv["offset_dist"].unsqueeze(-1)
                out[arm] = out[arm] + self._ee_step(state, arm, m_recv, tgt,
                                                    self.pv["approach_speed"])
            if m_give.any():
                # slow square-wave drift in z around the start point (visible on the toy too)
                phase = torch.floor(self.t / self.pv["drift_period"].clamp_min(1e-3)) % 2.0
                sgn = torch.where(phase < 1.0, 1.0, -1.0)
                tgt = self._start_ee[arm].clone()
                tgt[:, 2] = tgt[:, 2] + sgn * 0.15
                out[arm] = out[arm] + self._ee_step(state, arm, m_give, tgt,
                                                    self.pv["drift_speed"])


class TableSlam(ScriptedScenario):
    """One arm strokes fast down to just above the table, retracts, repeats."""

    attacks = ("high approach speed against a static surface: tests backstop timing and "
               "latency compensation; late braking clips the table, early braking kills "
               "legitimate fast motions")

    @dataclass
    class Params:
        pair_idx: int
        actor_is_F: int
        slam_speed: float
        retract_speed: float
        period: float        # full down+up cycle (s)
        z_clearance: float   # target height above table on the down stroke (m)
        xy_jitter: float

        @staticmethod
        def sample(g):
            return TableSlam.Params(
                pair_idx=_choice(g, 4), actor_is_F=_choice(g, 2),
                slam_speed=_u(g, 0.3, 0.8), retract_speed=_u(g, 0.15, 0.4),
                period=_u(g, 1.0, 2.5), z_clearance=_u(g, 0.0, 0.03),
                xy_jitter=_u(g, 0.0, 0.05))

    def _compute(self, state, out):
        actor_F = self.pv["actor_is_F"] > 0.5
        phase = (self.t / self.pv["period"].clamp_min(1e-3)) % 1.0
        down = phase < 0.5
        for arm in ARM_KEYS:
            mask = (self.m_F[arm] & actor_F) | (self.m_U[arm] & ~actor_F)
            if not mask.any():
                continue
            start = self._start_ee[arm]
            cyc = torch.floor(self.t / self.pv["period"].clamp_min(1e-3))
            jit = torch.stack([
                torch.sin(cyc * 12.9898), torch.cos(cyc * 78.233),
                torch.zeros_like(cyc)], dim=-1) * self.pv["xy_jitter"].unsqueeze(-1)
            tgt = start + jit
            tgt[:, 2] = torch.where(
                down, self.ws.table_z + self.pv["z_clearance"], start[:, 2])
            spd = torch.where(down, self.pv["slam_speed"], self.pv["retract_speed"])
            out[arm] = out[arm] + self._ee_step(state, arm, mask, tgt, spd)


class SweepAcross(ScriptedScenario):
    """One arm sweeps laterally back and forth straight through the opponent's box."""

    attacks = ("a fast moving obstacle crossing the opponent workspace: purely reactive "
               "distance filters brake too late or freeze the victim arm for the whole sweep; "
               "prediction (intent extrapolation) is required to pass efficiently")

    @dataclass
    class Params:
        pair_idx: int
        actor_is_F: int
        speed: float
        z_height: float      # sweep altitude above table (m)
        axis: int            # 0: sweep along x (toy plane), 1: along y
        period: float        # full back-and-forth period (s)

        @staticmethod
        def sample(g):
            return SweepAcross.Params(
                pair_idx=_choice(g, 4), actor_is_F=_choice(g, 2), speed=_u(g, 0.3, 0.7),
                z_height=_u(g, 0.1, 0.4), axis=_choice(g, 2), period=_u(g, 1.5, 4.0))

    def _compute(self, state, out):
        actor_F = self.pv["actor_is_F"] > 0.5
        # triangle wave 0..1..0 over one period
        frac = (self.t / self.pv["period"].clamp_min(1e-3)) % 1.0
        tri = 1.0 - (2.0 * frac - 1.0).abs()
        for arm in ARM_KEYS:
            mask = (self.m_F[arm] & actor_F) | (self.m_U[arm] & ~actor_F)
            if not mask.any():
                continue
            opp_box_lo = torch.where(
                actor_F.unsqueeze(-1),
                torch.tensor(self.ws.box_U[0], device=self.device).expand(self.n, 3),
                torch.tensor(self.ws.box_F[0], device=self.device).expand(self.n, 3))
            opp_box_hi = torch.where(
                actor_F.unsqueeze(-1),
                torch.tensor(self.ws.box_U[1], device=self.device).expand(self.n, 3),
                torch.tensor(self.ws.box_F[1], device=self.device).expand(self.n, 3))
            tgt = (opp_box_lo + opp_box_hi) * 0.5
            ax = self.pv["axis"].long().clamp(0, 1)
            lo = torch.gather(opp_box_lo, 1, ax.unsqueeze(-1)).squeeze(-1)
            hi = torch.gather(opp_box_hi, 1, ax.unsqueeze(-1)).squeeze(-1)
            sweep = lo + tri * (hi - lo)
            tgt = tgt.scatter(1, ax.unsqueeze(-1), sweep.unsqueeze(-1))
            tgt[:, 2] = self.ws.table_z + self.pv["z_height"]
            out[arm] = out[arm] + self._ee_step(state, arm, mask, tgt, self.pv["speed"])


class Chase(ScriptedScenario):
    """Pursuer tracks the evader EE while the evader keeps retreating."""

    attacks = ("persistent closing velocity that never self-resolves: without an explicit "
               "right-of-way decision the pair livelocks -- the pursuer keeps re-closing and "
               "the filter keeps re-braking, racking up oscillations")

    @dataclass
    class Params:
        pair_idx: int
        chaser_is_F: int
        chase_speed: float
        evade_speed: float
        standoff: float     # pursuer aims this far short of the evader EE (m)

        @staticmethod
        def sample(g):
            return Chase.Params(
                pair_idx=_choice(g, 4), chaser_is_F=_choice(g, 2),
                chase_speed=_u(g, 0.2, 0.5), evade_speed=_u(g, 0.05, 0.3),
                standoff=_u(g, 0.0, 0.04))

    def _compute(self, state, out):
        chaser_F = self.pv["chaser_is_F"] > 0.5
        for arm in ARM_KEYS:
            m_chase = (self.m_F[arm] & chaser_F) | (self.m_U[arm] & ~chaser_F)
            m_evade = (self.m_U[arm] & chaser_F) | (self.m_F[arm] & ~chaser_F)
            other_ee = torch.zeros(self.n, 3, device=self.device)
            for i, (fa, ua) in enumerate(PAIRS):
                sel = self.pv["pair_idx"].long() == i
                if arm in (fa, ua):
                    other = fa if arm == ua else ua
                    other_ee = torch.where(sel.unsqueeze(-1), state.ee_pos[other], other_ee)
            if m_chase.any():
                gap = other_ee - state.ee_pos[arm]
                dirn = gap / gap.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                tgt = other_ee - dirn * self.pv["standoff"].unsqueeze(-1)
                out[arm] = out[arm] + self._ee_step(state, arm, m_chase, tgt,
                                                    self.pv["chase_speed"])
            if m_evade.any():
                away = state.ee_pos[arm] - other_ee
                dirn = away / away.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                tgt = state.ee_pos[arm] + dirn * 0.5
                # evader stays inside its own workspace box
                own_lo = torch.where(
                    chaser_F.unsqueeze(-1),
                    torch.tensor(self.ws.box_U[0], device=self.device).expand(self.n, 3),
                    torch.tensor(self.ws.box_F[0], device=self.device).expand(self.n, 3))
                own_hi = torch.where(
                    chaser_F.unsqueeze(-1),
                    torch.tensor(self.ws.box_U[1], device=self.device).expand(self.n, 3),
                    torch.tensor(self.ws.box_F[1], device=self.device).expand(self.n, 3))
                tgt = torch.maximum(torch.minimum(tgt, own_hi), own_lo)
                out[arm] = out[arm] + self._ee_step(state, arm, m_evade, tgt,
                                                    self.pv["evade_speed"])


class FreezeOne(ScriptedScenario):
    """One robot's stream dies mid-episode while the opponent works right next to it."""

    attacks = ("asymmetric responsibility: the frozen robot has zero motion budget, so "
               "filters that split braking bilaterally waste half the avoidance capacity "
               "and stall the live arm")

    @dataclass
    class Params:
        pair_idx: int
        frozen_is_F: int
        t_freeze: float      # freeze onset (s)
        freeze_dur: float    # freeze duration (s)
        approach_speed: float  # live arm approach speed toward the frozen EE

        @staticmethod
        def sample(g):
            return FreezeOne.Params(
                pair_idx=_choice(g, 4), frozen_is_F=_choice(g, 2),
                t_freeze=_u(g, 0.5, 2.0), freeze_dur=_u(g, 1.0, 4.0),
                approach_speed=_u(g, 0.1, 0.4))

    def __init__(self, n_envs, variants, mapper, ws, device="cpu", dof_of=None):
        super().__init__(n_envs, variants, mapper, ws, device, dof_of)
        self.inner = L1RandomDelta(n_envs, L1Params(ou_mix=0.7), device=device,
                                   dof_of=dof_of)

    def _reset_extra(self, ids):
        self.inner.reset(ids, self.gen)

    def _compute(self, state, out):
        inner_cmd = self.inner.sample(state).delta_q
        frozen_F = self.pv["frozen_is_F"] > 0.5
        in_freeze = (self.t >= self.pv["t_freeze"]) & \
                    (self.t < self.pv["t_freeze"] + self.pv["freeze_dur"])
        for arm in ARM_KEYS:
            is_F_arm = arm.startswith("F")
            frozen_here = in_freeze & (frozen_F == is_F_arm)
            d = torch.where(frozen_here.unsqueeze(-1),
                            torch.zeros_like(inner_cmd[arm]), inner_cmd[arm])
            # the live paired arm steers toward the frozen robot's EE during the window
            m_live = (self.m_U[arm] & frozen_F) | (self.m_F[arm] & ~frozen_F)
            m_live = m_live & in_freeze
            if m_live.any():
                other_ee = torch.zeros(self.n, 3, device=self.device)
                for i, (fa, ua) in enumerate(PAIRS):
                    sel = self.pv["pair_idx"].long() == i
                    if arm in (fa, ua):
                        other = fa if arm == ua else ua
                        other_ee = torch.where(sel.unsqueeze(-1), state.ee_pos[other],
                                               other_ee)
                steer = self._ee_step(state, arm, m_live, other_ee,
                                      self.pv["approach_speed"])
                d = torch.where(m_live.unsqueeze(-1), steer, d)
            out[arm] = out[arm] + d


class LatencySpike(ScriptedScenario):
    """Deltas of one robot stall in a queue during a latency spike, then burst out stale."""

    attacks = ("stale intent arriving compressed in time: extrapolation-based prediction is "
               "briefly wrong and the burst re-excites conflicts the filter thought resolved; "
               "tests delay compensation and oscillation damping")

    @dataclass
    class Params:
        laggy_is_F: int
        t_spike: float      # spike onset (s)
        spike_dur: float    # how long packets are held (s)
        burst_rate: int     # queued packets released per step after the spike

        @staticmethod
        def sample(g):
            return LatencySpike.Params(
                laggy_is_F=_choice(g, 2), t_spike=_u(g, 0.5, 2.0),
                spike_dur=_u(g, 0.2, 1.0), burst_rate=2 + _choice(g, 3))

    def __init__(self, n_envs, variants, mapper, ws, device="cpu", dof_of=None):
        super().__init__(n_envs, variants, mapper, ws, device, dof_of)
        self.inner = L1RandomDelta(n_envs, L1Params(ou_mix=0.4), device=device,
                                   dof_of=dof_of)
        self._buf: dict = {}      # arm -> (N, cap, dof) ring buffer, lazy alloc
        self._w = {}              # arm -> (N,) write counters
        self._r = {}              # arm -> (N,) read counters

    def _reset_extra(self, ids):
        self.inner.reset(ids, self.gen)
        for a in self._buf:
            self._buf[a][ids] = 0.0
            self._w[a][ids] = 0
            self._r[a][ids] = 0

    def _ensure_buffers(self, state):
        if self._buf:
            return
        cap = int(math.ceil(1.0 / state.dt)) + 8  # >= max spike_dur (1s) backlog
        for a in ARM_KEYS:
            self._buf[a] = torch.zeros(self.n, cap, state.q[a].shape[-1],
                                       device=self.device)
            self._w[a] = torch.zeros(self.n, dtype=torch.long, device=self.device)
            self._r[a] = torch.zeros(self.n, dtype=torch.long, device=self.device)

    def _compute(self, state, out):
        self._ensure_buffers(state)
        inner_cmd = self.inner.sample(state).delta_q
        laggy_F = self.pv["laggy_is_F"] > 0.5
        holding = (self.t >= self.pv["t_spike"]) & \
                  (self.t < self.pv["t_spike"] + self.pv["spike_dur"])
        for a in ARM_KEYS:
            is_F_arm = a.startswith("F")
            laggy_here = laggy_F == is_F_arm
            buf, w, r = self._buf[a], self._w[a], self._r[a]
            cap = buf.shape[1]
            # push this step's delta into the ring
            buf[torch.arange(self.n, device=self.device), w % cap] = inner_cmd[a]
            w = w + 1
            backlog = w - r
            # pops: 0 while holding; burst_rate while backlogged; else 1
            pops = torch.where(holding & laggy_here, torch.zeros_like(r),
                               torch.where(laggy_here & (backlog > 1),
                                           self.pv["burst_rate"].long(),
                                           torch.ones_like(r)))
            pops = torch.minimum(pops, backlog)
            # sum the popped packets (vectorized over the small ring capacity)
            k = torch.arange(cap, device=self.device).unsqueeze(0)          # (1, cap)
            take = (k >= (r % cap).unsqueeze(1)) & (k < (r % cap + pops).unsqueeze(1))
            wrap = (k + cap >= (r % cap).unsqueeze(1)) & \
                   (k + cap < (r % cap + pops).unsqueeze(1))
            sel = (take | wrap).float().unsqueeze(-1)                        # (N, cap, 1)
            out[a] = out[a] + (buf * sel).sum(dim=1)
            self._w[a], self._r[a] = w, r + pops


SCENARIOS = {
    "head_on_crossing": HeadOnCrossing,
    "center_grab": CenterGrab,
    "handover_approach": HandoverApproach,
    "table_slam": TableSlam,
    "sweep_across": SweepAcross,
    "chase": Chase,
    "freeze_one": FreezeOne,
    "latency_spike": LatencySpike,
}
