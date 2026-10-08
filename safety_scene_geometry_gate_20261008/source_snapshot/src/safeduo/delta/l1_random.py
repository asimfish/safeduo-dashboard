"""L1 random intent stream: OU noise + workspace-waypoint goal-directed segments.

Design (task book C2-L1):
- OU component: bandwidth-limited joint-space jitter, mimics aimless glove motion.
- Waypoint component: piecewise goal-directed motion. Intent is formed in EE
  space (velocity toward a waypoint) and mapped to joint deltas through an
  IntentMapper. In the deployed pipeline that mapper is replaced by the real
  retarget/IK stack (interface kept); locally we ship two approximations:
    * JointSpaceMapper -- waypoint sampled directly in joint space,
      delta = k*(q_star - q); zero geometry dependence.
    * JacobianMapper   -- damped pseudo-inverse using an injected jacobian_fn
      (toy tests provide the analytic J of the planar arms).
- Segment state machine per (env, arm): WAYPOINT segment (sampled duration)
  -> PAUSE with probability pause_prob -> resample waypoint.
- All randomness goes through torch.Generator; reset(env_ids, generator) is
  reproducible.

Parameters: amplitude cap amp_max, OU bandwidth bandwidth_hz, OU magnitude
ou_sigma, segment duration seg_dur, pause probability pause_prob, pause
duration pause_dur, waypoint gain (inside mapper), optional output low-pass.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from safeduo.delta._contract_stub import (
    ARM_KEYS,
    DOF_OF,
    DeltaCmd,
    DeltaSource,
    SceneState,
)


class IntentMapper(ABC):
    """EE-space intent -> joint delta mapping; future: real retarget/IK pipeline."""

    @abstractmethod
    def map(self, arm: str, state: SceneState, tgt: torch.Tensor, dt: float,
            speed: "torch.Tensor | float | None" = None) -> torch.Tensor:
        """tgt: waypoint target (semantics defined by impl); returns (N, dof) delta."""

    @abstractmethod
    def sample_waypoint(self, arm: str, state: SceneState,
                        generator: "torch.Generator | None") -> torch.Tensor:
        """Sample a fresh waypoint for this arm (same shape map() expects as tgt)."""


class JointSpaceMapper(IntentMapper):
    """Joint-space approximation: waypoint is a joint target q*, delta = clip(k*(q*-q))."""

    def __init__(self, wp_gain: float = 2.0, q_range: float = 1.2):
        self.wp_gain = wp_gain
        self.q_range = q_range  # waypoint offset range around current q (rad)

    def map(self, arm, state, tgt, dt, speed=None):
        d = self.wp_gain * (tgt - state.q[arm]) * dt
        if speed is not None:  # interpret speed as joint-space rate cap (rad/s)
            cap = torch.as_tensor(speed, dtype=d.dtype, device=d.device).reshape(-1, 1) * dt
            norm = d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            d = d * torch.clamp(cap / norm, max=1.0)
        return d

    def sample_waypoint(self, arm, state, generator):
        n, dof = state.q[arm].shape
        off = (
            torch.rand(n, dof, device=state.q[arm].device, generator=generator) * 2.0 - 1.0
        ) * self.q_range
        return state.q[arm] + off


class JacobianMapper(IntentMapper):
    """Simplified-jacobian mapping: v_ee points at the EE waypoint, damped
    pseudo-inverse maps it to joint space.

    jacobian_fn(arm, state) -> (N, 3, dof). Toy tests inject the analytic J;
    the Isaac pipeline can inject batched GPU jacobians or swap the whole
    mapper for the deployed retarget/IK.
    """

    def __init__(self, jacobian_fn, workspace_lo, workspace_hi,
                 ee_speed: float = 0.25, damping: float = 1e-2):
        self.jacobian_fn = jacobian_fn
        self.lo = torch.as_tensor(workspace_lo, dtype=torch.float32)
        self.hi = torch.as_tensor(workspace_hi, dtype=torch.float32)
        self.ee_speed = ee_speed
        self.damping = damping

    def map(self, arm, state, tgt, dt, speed=None):
        err = tgt - state.ee_pos[arm]  # (N, 3)
        dist = err.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        cap = self.ee_speed if speed is None else torch.as_tensor(
            speed, dtype=err.dtype, device=err.device).reshape(-1, 1)
        v_ee = err / dist * torch.minimum(dist / dt, cap * torch.ones_like(dist))
        J = self.jacobian_fn(arm, state)  # (N, 3, dof)
        JJt = J @ J.transpose(-1, -2)
        eye = torch.eye(J.shape[1], device=J.device, dtype=J.dtype).expand_as(JJt)
        qd = J.transpose(-1, -2) @ torch.linalg.solve(JJt + self.damping * eye,
                                                      v_ee.unsqueeze(-1))
        return qd.squeeze(-1) * dt

    def sample_waypoint(self, arm, state, generator):
        n = state.ee_pos[arm].shape[0]
        dev = state.ee_pos[arm].device
        lo, hi = self.lo.to(dev), self.hi.to(dev)
        u = torch.rand(n, lo.numel(), device=dev, generator=generator)
        return lo + u * (hi - lo)


@dataclass
class L1Params:
    amp_max: float = 0.06          # per-step per-joint delta cap (rad); calibrate to glove data later
    ou_sigma: float = 0.015        # OU stationary magnitude (rad/step scale)
    bandwidth_hz: float = 1.5      # OU bandwidth (theta = 2*pi*bw)
    ou_mix: float = 0.5            # OU weight (0 = pure waypoint, 1 = pure OU)
    seg_dur: tuple = (1.0, 4.0)    # waypoint segment duration range (s)
    pause_prob: float = 0.25       # probability of a pause after each segment
    pause_dur: tuple = (0.3, 1.5)  # pause duration range (s)
    smooth_hz: float = 0.0         # >0 applies first-order low-pass (EMA) to output


class L1RandomDelta(DeltaSource):
    """Batched OU + waypoint-segment random intent stream (all arms, N envs)."""

    def __init__(self, n_envs: int, params: "L1Params | None" = None,
                 mapper: "IntentMapper | None" = None,
                 device: "str | torch.device" = "cpu",
                 arm_keys: tuple = ARM_KEYS,
                 dof_of: "dict | None" = None):
        self.n = n_envs
        self.p = params or L1Params()
        self.mapper = mapper or JointSpaceMapper()
        self.device = torch.device(device)
        self.arms = arm_keys
        dof_of = dof_of or DOF_OF  # toy tests override (3-DoF planar arms)
        self.gen: "torch.Generator | None" = None
        self._ou = {a: torch.zeros(n_envs, dof_of[a], device=self.device) for a in self.arms}
        self._ema = {a: torch.zeros(n_envs, dof_of[a], device=self.device) for a in self.arms}
        self._wp: dict = {a: None for a in self.arms}  # lazy init: needs a state
        self._t_left = {a: torch.zeros(n_envs, device=self.device) for a in self.arms}
        self._paused = {a: torch.zeros(n_envs, dtype=torch.bool, device=self.device)
                        for a in self.arms}

    def reset(self, env_ids: torch.Tensor, generator: "torch.Generator | None" = None) -> None:
        if generator is not None:
            self.gen = generator
        ids = env_ids.to(self.device)
        for a in self.arms:
            self._ou[a][ids] = 0.0
            self._ema[a][ids] = 0.0
            self._t_left[a][ids] = 0.0   # forces segment + waypoint resample on next sample()
            self._paused[a][ids] = False
            if self._wp[a] is not None:
                self._wp[a][ids] = float("nan")  # mark for resample

    def _rand(self, *shape) -> torch.Tensor:
        return torch.rand(*shape, device=self.device, generator=self.gen)

    def _randn(self, *shape) -> torch.Tensor:
        return torch.randn(*shape, device=self.device, generator=self.gen)

    def _advance_segments(self, arm: str, state: SceneState, dt: float) -> None:
        p = self.p
        t = self._t_left[arm] - dt
        expired = t <= 0.0
        if expired.any():
            # expired envs flip pause/move state and resample a duration
            go_pause = (self._rand(self.n) < p.pause_prob) & ~self._paused[arm] & expired
            self._paused[arm] = torch.where(expired, go_pause, self._paused[arm])
            dur_move = p.seg_dur[0] + self._rand(self.n) * (p.seg_dur[1] - p.seg_dur[0])
            dur_pause = p.pause_dur[0] + self._rand(self.n) * (p.pause_dur[1] - p.pause_dur[0])
            new_dur = torch.where(self._paused[arm], dur_pause, dur_move)
            t = torch.where(expired, new_dur, t)
            # envs entering a move segment get a fresh waypoint
            need_wp = expired & ~self._paused[arm]
            wp_new = self.mapper.sample_waypoint(arm, state, self.gen)
            if self._wp[arm] is None:
                self._wp[arm] = wp_new
            else:
                stale = need_wp | torch.isnan(self._wp[arm]).any(dim=-1)
                self._wp[arm] = torch.where(stale.unsqueeze(-1), wp_new, self._wp[arm])
        self._t_left[arm] = t

    def sample(self, state: SceneState) -> DeltaCmd:
        p, dt = self.p, state.dt
        theta = 2.0 * math.pi * p.bandwidth_hz
        out = {}
        for a in self.arms:
            self._advance_segments(a, state, dt)
            # OU: dx = -theta*x*dt + sigma*sqrt(2*theta*dt)*N -> stationary std ~= ou_sigma
            ou = self._ou[a]
            ou = ou + (-theta * ou * dt
                       + p.ou_sigma * math.sqrt(2.0 * theta * dt) * self._randn(*ou.shape))
            self._ou[a] = ou
            # waypoint goal-directed component
            wp_delta = self.mapper.map(a, state, self._wp[a], dt)
            wp_delta = torch.where(self._paused[a].unsqueeze(-1),
                                   torch.zeros_like(wp_delta), wp_delta)
            d = p.ou_mix * ou + (1.0 - p.ou_mix) * wp_delta
            if p.smooth_hz > 0.0:
                alpha = 1.0 - math.exp(-2.0 * math.pi * p.smooth_hz * dt)
                self._ema[a] = self._ema[a] + alpha * (d - self._ema[a])
                d = self._ema[a]
            out[a] = d.clamp(-p.amp_max, p.amp_max)
        return DeltaCmd(delta_q=out)
