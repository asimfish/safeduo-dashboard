"""L3 constrained adversarial intent stream: design + trainable skeleton.

Design (task book C2-L3 + FINAL_PROPOSAL round-1/2 constraints), frozen here
so the M2 build starts from a reviewed spec:

  ACTION SPACE = intent-stream parameters, NOT raw joint deltas. Every
  `seg_steps` control steps the adversary policy emits, per arm:
      waypoint (3)   EE-space target, decoded into that robot's share of the
                     shared-workspace box (same boxes the L2 scenarios use)
      speed (1)      EE approach speed in [0, v_max]
      pause (1)      probability-like gate; > 0.5 freezes the arm this
                     segment (lets the adversary craft freeze_one-style and
                     phase-offset attacks)
  The decoded intent goes through the SAME IntentMapper pipeline as L1/L2
  (deployment-consistent retarget/IK path -- CachedEEMapper on the env, toy
  jacobian mapper in tests), so the adversary can only express commands a
  human glove stream could express.

  REALISM CONSTRAINTS are structural (built into the source), not just
  penalties: (a) per-joint per-step amplitude clamp amp_max (glove scale,
  same knob as coordinator.delta_amp_max); (b) first-order low-pass at
  bandwidth_hz caps the emitted spectrum (human-band); (c) waypoints cannot
  leave the workspace boxes. A StreamStatsMonitor additionally tracks
  amplitude / high-frequency ratio / pause fraction against human reference
  stats (HumanRefStats; calibrate from L4 glove recordings at M1) -- its
  distance feeds W&B monitoring and an optional reward penalty, per the
  proposal's "??????".

  ADVERSARY REWARD (wiring contract for the A-line env hook):
      r_adv = w_margin * margin_cost + w_tube * tube
            + w_viol * violation - w_real * stream_distance
  margin_cost = algo/reward_shaping.shaped_margin_cost* (the coordinator's
  own proximity cost -- the adversary maximizes exactly what the coordinator
  is trained to minimize, in a zero-sum-on-safety sense); realism distance
  keeps it inside the human envelope. Provided as `adversary_reward`.

  ALTERNATION: coordinator trains K_c iters against a FROZEN adversary
  (mixed <= 30% of envs, rest L1/L2 curriculum), then the adversary trains
  K_a iters against the FROZEN coordinator. Frozen adversary checkpoints are
  registered for the held-out attack evaluation (train/eval split discipline:
  eval attacks come only from checkpoints never trained against).

  This module is deliberately Isaac-free: AdversarialDeltaSource follows the
  DeltaSource contract, the trainer skeleton takes opaque callbacks. Actual
  PPO plumbing lands with the M2 GPU slot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from safeduo.delta._contract_stub import ARM_KEYS, DeltaCmd, DeltaSource, SceneState
from safeduo.delta.l1_random import IntentMapper
from safeduo.delta.l2_scenarios import WorkspaceSpec

PARAMS_PER_ARM = 5           # waypoint(3) + speed(1) + pause(1)
ADV_ACTION_DIM = PARAMS_PER_ARM * len(ARM_KEYS)   # 20


@dataclass
class AdvConfig:
    seg_steps: int = 30          # policy refresh period (0.5 s @ 60 Hz)
    v_max: float = 0.5           # EE speed cap (m/s), matches L2 scenario range
    amp_max: float = 0.015      # per-joint per-step clamp (rad), glove scale
    bandwidth_hz: float = 1.5    # output low-pass (human band, matches L1Params)
    pause_threshold: float = 0.5


def decode_actions(actions: torch.Tensor, ws: WorkspaceSpec,
                   cfg: AdvConfig) -> dict:
    """(N, 20) normalized actions in [-1, 1] -> per-arm segment parameters.

    waypoint: affine map into the arm's robot workspace box (F arms -> box_F,
    U arms -> box_U); speed: [0, v_max]; pause: bool gate."""
    if actions.ndim != 2 or actions.shape[1] != ADV_ACTION_DIM:
        raise ValueError(f"actions must have shape [N, {ADV_ACTION_DIM}]")
    a = actions.clamp(-1.0, 1.0)
    out = {}
    for i, arm in enumerate(ARM_KEYS):
        blk = a[:, i * PARAMS_PER_ARM:(i + 1) * PARAMS_PER_ARM]
        lo, hi = ws.box_of(arm[0])
        lo = torch.tensor(lo, dtype=a.dtype, device=a.device)
        hi = torch.tensor(hi, dtype=a.dtype, device=a.device)
        wp = lo + (blk[:, :3] * 0.5 + 0.5) * (hi - lo)
        speed = (blk[:, 3] * 0.5 + 0.5) * cfg.v_max
        pause = blk[:, 4] > (cfg.pause_threshold * 2.0 - 1.0)
        out[arm] = {"waypoint": wp, "speed": speed, "pause": pause}
    return out


class AdversarialDeltaSource(DeltaSource):
    """DeltaSource driven by an external adversary policy.

    Contract with the training loop: check `needs_actions` each step; when
    True (segment boundary, per env after reset or seg_steps elapsed), query
    the adversary policy on its observation and push the (N, 20) normalized
    actions via `set_actions(actions, env_mask)`. Envs never fed actions
    emit zeros (fail-safe: an unwired adversary is a no-op stream, not a
    crash)."""

    def __init__(self, n_envs: int, mapper: IntentMapper, ws: WorkspaceSpec,
                 cfg: "AdvConfig | None" = None,
                 device: "str | torch.device" = "cpu",
                 dof_of: "dict | None" = None):
        from safeduo.delta._contract_stub import DOF_OF

        self.n = n_envs
        self.mapper = mapper
        self.ws = ws
        self.cfg = cfg or AdvConfig()
        self.device = torch.device(device)
        dof_of = dof_of or DOF_OF
        self._wp = {a: torch.zeros(n_envs, 3, device=self.device)
                    for a in ARM_KEYS}
        self._speed = {a: torch.zeros(n_envs, device=self.device)
                       for a in ARM_KEYS}
        self._pause = {a: torch.ones(n_envs, dtype=torch.bool,
                                     device=self.device) for a in ARM_KEYS}
        self._armed = torch.zeros(n_envs, dtype=torch.bool, device=self.device)
        self._steps_left = torch.zeros(n_envs, dtype=torch.long,
                                       device=self.device)
        self._ema = {a: torch.zeros(n_envs, dof_of[a], device=self.device)
                     for a in ARM_KEYS}
        self.monitor = StreamStatsMonitor(n_envs, device=self.device)

    @property
    def needs_actions(self) -> torch.Tensor:
        """(N,) bool: envs whose segment expired (or never armed)."""
        return (self._steps_left <= 0) | ~self._armed

    def set_actions(self, actions: torch.Tensor,
                    env_mask: "torch.Tensor | None" = None) -> None:
        dec = decode_actions(actions.to(self.device), self.ws, self.cfg)
        mask = torch.ones(self.n, dtype=torch.bool, device=self.device) \
            if env_mask is None else env_mask.to(self.device)
        for arm in ARM_KEYS:
            d = dec[arm]
            self._wp[arm][mask] = d["waypoint"][mask]
            self._speed[arm][mask] = d["speed"][mask]
            self._pause[arm][mask] = d["pause"][mask]
        self._armed |= mask
        self._steps_left[mask] = self.cfg.seg_steps

    def reset(self, env_ids: torch.Tensor,
              generator: "torch.Generator | None" = None) -> None:
        ids = env_ids.to(self.device)
        self._armed[ids] = False
        self._steps_left[ids] = 0
        for a in ARM_KEYS:
            self._ema[a][ids] = 0.0
            self._pause[a][ids] = True
        self.monitor.reset(ids)

    def sample(self, state: SceneState) -> DeltaCmd:
        cfg = self.cfg
        import math

        lp = 1.0 - math.exp(-2.0 * math.pi * cfg.bandwidth_hz * state.dt)
        out = {}
        for arm in ARM_KEYS:
            raw = self.mapper.map(arm, state, self._wp[arm], state.dt,
                                  speed=self._speed[arm])
            gate = (self._armed & ~self._pause[arm]).unsqueeze(-1)
            raw = torch.where(gate, raw, torch.zeros_like(raw))
            # order matters for the realism guarantee: clamp FIRST, then
            # low-pass, and emit the EMA itself. Clamping after the filter
            # would re-inject high frequency (clipped flat-tops) and leave
            # the per-step output jump unbounded (raw is only speed-capped,
            # not amplitude-capped). This order yields both structural
            # bounds: |out| <= amp_max and |out_t - out_{t-1}| <=
            # lp * 2 * amp_max (tests pin the latter).
            raw = raw.clamp(-cfg.amp_max, cfg.amp_max)
            self._ema[arm] = self._ema[arm] + lp * (raw - self._ema[arm])
            out[arm] = self._ema[arm]
        self._steps_left = (self._steps_left - 1).clamp_min(0)
        cmd = DeltaCmd(delta_q=out)
        self.monitor.observe(cmd)
        return cmd


# ------------------------------------------------------------------ realism

@dataclass
class HumanRefStats:
    """Per-stream reference statistics; placeholders until the M1 glove
    recordings calibrate them (from_recording)."""
    amp_mean: float = 0.004      # mean |delta| per joint (rad/step)
    hf_ratio: float = 0.15       # E|d_t - d_{t-1}|^2 / E|d_t|^2 (spectral proxy)
    pause_frac: float = 0.15     # fraction of near-zero steps

    @classmethod
    def from_recording(cls, deltas: torch.Tensor,
                       zero_tol: float = 1e-4) -> "HumanRefStats":
        """deltas: (T, dof) recorded human joint-delta stream."""
        amp = deltas.abs().mean().item()
        num = (deltas[1:] - deltas[:-1]).pow(2).mean().item()
        den = max(deltas.pow(2).mean().item(), 1e-12)
        pause = (deltas.norm(dim=-1) < zero_tol).float().mean().item()
        return cls(amp_mean=amp, hf_ratio=num / den, pause_frac=pause)


class StreamStatsMonitor:
    """Running EMA stats of an emitted delta stream + distance to reference.

    Feeds the proposal's "??????" (W&B scalar) and the optional realism
    penalty in adversary_reward. All stats are per-env scalars aggregated
    over arms; EMA horizon ~ tau steps."""

    def __init__(self, n_envs: int, tau: float = 200.0,
                 device: "str | torch.device" = "cpu"):
        self.n = n_envs
        self.k = 1.0 / tau
        self.device = torch.device(device)
        self._amp = torch.zeros(n_envs, device=self.device)
        self._hf_num = torch.zeros(n_envs, device=self.device)
        self._hf_den = torch.zeros(n_envs, device=self.device)
        self._pause = torch.zeros(n_envs, device=self.device)
        self._prev: "torch.Tensor | None" = None

    def reset(self, env_ids: torch.Tensor) -> None:
        ids = env_ids.to(self.device)
        for buf in (self._amp, self._hf_num, self._hf_den, self._pause):
            buf[ids] = 0.0
        if self._prev is not None:
            self._prev[ids] = 0.0

    def observe(self, cmd: DeltaCmd, zero_tol: float = 1e-4) -> None:
        d = cmd.stacked()
        if self._prev is None:
            self._prev = torch.zeros_like(d)
        self._amp += self.k * (d.abs().mean(dim=-1) - self._amp)
        self._hf_num += self.k * ((d - self._prev).pow(2).mean(dim=-1)
                                  - self._hf_num)
        self._hf_den += self.k * (d.pow(2).mean(dim=-1) - self._hf_den)
        self._pause += self.k * ((d.norm(dim=-1) < zero_tol).float()
                                 - self._pause)
        self._prev = d.clone()

    def stats(self) -> dict:
        hf = self._hf_num / self._hf_den.clamp_min(1e-12)
        return {"amp_mean": self._amp.clone(), "hf_ratio": hf,
                "pause_frac": self._pause.clone()}

    def distance_to(self, ref: HumanRefStats) -> torch.Tensor:
        """(N,) normalized stat distance; 0 = matches the human envelope."""
        s = self.stats()
        terms = [
            (s["amp_mean"] - ref.amp_mean) / max(ref.amp_mean, 1e-6),
            (s["hf_ratio"] - ref.hf_ratio) / max(ref.hf_ratio, 1e-6),
            (s["pause_frac"] - ref.pause_frac) / max(ref.pause_frac, 1e-6),
        ]
        return torch.stack(terms, dim=-1).pow(2).mean(dim=-1).sqrt()


def adversary_reward(margin_cost: torch.Tensor, tube: torch.Tensor,
                     violation: torch.Tensor, stream_distance: torch.Tensor,
                     w_margin: float = 1.0, w_tube: float = 0.5,
                     w_viol: float = 10.0, w_real: float = 1.0
                     ) -> torch.Tensor:
    """Adversary maximizes the coordinator's safety pressure inside the
    human envelope. Signs: all inputs (N,) >= 0 except stream_distance."""
    return (w_margin * margin_cost + w_tube * tube.float()
            + w_viol * violation.float() - w_real * stream_distance)


# -------------------------------------------------------------- alternation

def sample_adv_mask(n_envs: int, adv_frac: float,
                    generator: "torch.Generator | None" = None,
                    device: "str | torch.device" = "cpu") -> torch.Tensor:
    """(N,) bool env assignment for the adversarial stream; proposal caps the
    mix at 30% (rest stays on the L1/L2 curriculum)."""
    if not 0.0 <= adv_frac <= 0.3:
        raise ValueError("adversarial mix is capped at 30% by the proposal")
    return torch.rand(n_envs, device=torch.device(device),
                      generator=generator) < adv_frac


@dataclass
class AlternatingConfig:
    n_cycles: int = 4
    coord_iters_per_cycle: int = 200
    adv_iters_per_cycle: int = 50
    adv_frac: float = 0.2        # <= 0.3 enforced by sample_adv_mask
    freeze_every_cycle: bool = True


@dataclass
class AlternatingTrainer:
    """Orchestration skeleton for coordinator/adversary alternation.

    train_coordinator(n_iters, frozen_adv) -> checkpoint-ish handle
    train_adversary(n_iters, frozen_coord) -> checkpoint-ish handle
    Both callbacks own all PPO/env plumbing (A-line + GPU slot, M2); this
    skeleton owns only the schedule and the frozen-checkpoint registry that
    the held-out attack eval draws from.
    """

    cfg: AlternatingConfig
    train_coordinator: "object" = None
    train_adversary: "object" = None
    frozen_adversaries: list = field(default_factory=list)
    frozen_coordinators: list = field(default_factory=list)

    def run(self) -> dict:
        if self.train_coordinator is None or self.train_adversary is None:
            raise ValueError("wire train_coordinator/train_adversary first")
        sample_adv_mask(1, self.cfg.adv_frac)  # validates the cap eagerly
        history = []
        adv_ckpt = None
        for cycle in range(self.cfg.n_cycles):
            coord_ckpt = self.train_coordinator(
                self.cfg.coord_iters_per_cycle, adv_ckpt)
            adv_ckpt = self.train_adversary(
                self.cfg.adv_iters_per_cycle, coord_ckpt)
            if self.cfg.freeze_every_cycle:
                self.frozen_adversaries.append(adv_ckpt)
                self.frozen_coordinators.append(coord_ckpt)
            history.append({"cycle": cycle, "coord": coord_ckpt,
                            "adv": adv_ckpt})
        return {"history": history,
                "frozen_adversaries": list(self.frozen_adversaries)}
