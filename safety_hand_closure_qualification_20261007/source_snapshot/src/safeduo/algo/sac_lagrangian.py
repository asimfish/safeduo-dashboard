"""Self-contained SAC-Lagrangian trainer (R3 elimination-race batch 1, C15).

Role (MASTER_REPORT sec.6/sec.9 R3): the off-policy comparison arm against the
PPO-Lagrangian mainline. This file is deliberately SELF-CONTAINED -- it
touches no mainline file; shared utilities are COPIED with attribution
(cost_channels_from_obs / ObsLayout / cost_sync_from_env_cfg / _backbone)
so later PPO-mainline changes can never ripple into the race arm. The only
imports from the Lagrangian stack are the constraint controllers
(PIDLagrangian / DualGradientLagrangian, algo/lagrangian.py) which the task
explicitly prescribes to reuse.

Action contract (identical semantics to the PPO arm, beta_actor.py):
  alpha in [0,1] x 4 (per-arm pass-through gates) + p in [-1,1] (right of
  way). Actor = tanh-squashed diagonal Gaussian: raw = tanh(u) in (-1,1)^5
  IS ALREADY the duo_env action layout (env re-derives alpha=(raw+1)/2 for
  dims 0..3 and p=raw_4, recovering our values exactly) -- to_env_action is
  the identity. log_prob carries the exact tanh Jacobian correction
  log(1-t^2) computed in the numerically stable softplus form.

Why tanh-Gaussian and not the PPO arm's Beta (decision record):
  1. SAC's actor gradient flows THROUGH Q(s, a~) via the reparameterized
     sample. tanh-Gaussian rsample is the exact pathwise derivative with a
     closed-form Jacobian; Beta.rsample exists in torch but rides implicit
     (Dirichlet) reparameterization whose gradient variance blows up as
     concentrations chase boundary mass -- exactly where alpha's useful
     yielding behavior lives (full stop / full pass).
  2. The v3 freeze disease Beta fixed in the PPO arm was a property of the
     clipped-ratio objective (clip-corner attractors). SAC has no ratio and
     no clip; the auto-tuned entropy temperature actively pushes the policy
     off deterministic saturation, attacking the same failure by SAC's own
     mechanism.
  3. Keeping the literature-standard SAC actor makes the elimination race
     measure the ALGORITHM, not an exotic actor variant.

Constraint side:
  critics   = twin Q_r (reward, min-target vs overestimation) + per-channel
              twin Q_c for cross/self/table. Cost targets and the actor
              penalty take the MAX of the twins: the policy MINIMIZES cost,
              so the exploitable bias direction flips -- max keeps the
              constraint estimate safety-pessimistic (WCSAC practice).
  lambda    = PIDLagrangian reused as-is, updated ONCE per iteration on the
              FRESH rollout's mean per-step cost (same cadence + same units
              as the PPO arm; never from stale replay).
  actor obj = maximize Q_r - sum_c lambda_c * (1-gamma) * Q_c
              - alpha_temp * log pi.
              The (1-gamma) factor converts discounted cost-to-go into a
              PER-STEP average-cost estimate, i.e. the same units the PID
              controller regulates against cost_limits and the same units
              lambda multiplies in the PPO arm (per-step cost advantages).
              Without it a lambda tuned by the shared PID gains would press
              ~1/(1-gamma)=200x harder here than in the PPO arm and the
              cross-arm comparison would be measuring units, not algorithms.
  alpha_temp = standard automatic temperature tuning against a configurable
              target entropy (default -ACTION_DIM). Coupling note (sec.6 risk):
              lambda shifts the actor objective by a BOUNDED amount (PID
              integral_limit caps lambda; (1-gamma)Q_c is per-step scale),
              so temperature adaptation sees a bounded reward-shift, not a
              runaway -- both adaptations are additionally observable per
              iteration in stats.jsonl/wandb for the server race.

Replay freshness (sec.6 flagged risk -- 4096 parallel envs make staleness the
enemy, not sample starvation):
  GPU ring buffer, default capacity 524288 transitions. At server scale
  (4096 envs x 24 steps/iter = 98304 fresh transitions per iteration) that
  is ~5.3 iterations of history: old enough to give the off-policy arm its
  sample-reuse advantage (>5x an on-policy batch), fresh enough that with
  tau=0.005 soft targets the behavior-policy lag stays in the low-single-
  digit-iteration range. Classic SAC's 1e6 capacity would mean 10+
  iterations of lag here and is deliberately NOT the default. fp32 footprint
  at 275-dim obs: ~1.2 GB (obs+next_obs dominate) -- fits the 5090 next to
  the env. Both capacity and update intensity are config knobs.

Local full-chain smoke (CPU):
  PYTHONPATH=src .venv/bin/python -m safeduo.algo.sac_lagrangian --smoke
Checklist mirrors the PPO arm's seven gates, adapted to SAC semantics where
the estimator differs (critic gate rides the dense self cost Q head's TD
loss instead of the GAE value loss; reward_mean stays report-only for the
same 16-env variance reason), plus two SAC-specific gates:
temperature_adapting (the sec.6 alpha_temp x lambda coupling is exercised) and
replay_warmed (updates actually ran). Summary lands in
artifacts/analysis/c15_sac_lagrangian_smoke/.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from safeduo.algo.lagrangian import DualGradientLagrangian, PIDLagrangian
from safeduo.algo.reward_shaping import margin_cost_by_class
from safeduo.safety.semantics import semantics_with_safety_overrides
from safeduo.safety.types import (
    ARM_KEYS,
    CLASS_CROSS,
    CLASS_SELF,
    CLASS_TABLE,
    MAX_ACTIVE_PAIRS,
    PAIR_FEATURE_DIM,
    TOTAL_DOF,
)

N_ALPHA = 4
ACTION_DIM = 5  # alpha(4) + p(1), the duo_env coordinator contract
COST_HEADS = ("cross", "self", "table")
LOG_2 = math.log(2.0)


# --------------------------------------------------------------------------
# obs-side cost channels -- COPIED from algo/lagrangian_ppo.py (C14 state)
# so the race arm is isolated from future PPO-mainline refactors; the math
# must stay bit-identical to the env-side channels (margin_cost_by_class).
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ObsLayout:
    """Slice map of the duo_env coordinator observation (P0, 275 @ M=32):
    [q(26), qd(26), pair_feats(M*5), mask(M), cmd(26), alpha_prev(4), p_prev(1)].
    (copy of lagrangian_ppo.ObsLayout)"""
    max_pairs: int = MAX_ACTIVE_PAIRS
    pair_dim: int = PAIR_FEATURE_DIM
    total_dof: int = TOTAL_DOF

    @property
    def pair_start(self) -> int:
        return 2 * self.total_dof

    @property
    def mask_start(self) -> int:
        return self.pair_start + self.max_pairs * self.pair_dim

    @property
    def obs_dim(self) -> int:
        return self.mask_start + self.max_pairs + self.total_dof + 5


def cost_channels_from_obs(obs: Tensor, layout: "ObsLayout | None" = None,
                           d_warn: float = 0.05,
                           d_min_by_class: "dict | None" = None) -> Tensor:
    """(N, obs_dim) -> (N, 3) [cross, self, table] worst-pair proximity cost.
    (copy of lagrangian_ppo.cost_channels_from_obs incl. the C14 d_min
    plumbing; see that docstring for the padding-mask and per-link caveats)"""
    lay = layout or ObsLayout()
    if obs.shape[-1] != lay.obs_dim:
        raise ValueError(f"obs dim {obs.shape[-1]} != layout {lay.obs_dim} "
                         "(M drifted? pass the right ObsLayout)")
    n = obs.shape[0]
    feats = obs[:, lay.pair_start:lay.mask_start].reshape(
        n, lay.max_pairs, lay.pair_dim)
    mask = obs[:, lay.mask_start:lay.mask_start + lay.max_pairs] > 0.5
    cls = feats[..., 2:5].argmax(dim=-1).to(feats.dtype)
    pairs = torch.stack([feats[..., 0], feats[..., 1], cls,
                         torch.zeros_like(cls)], dim=-1)
    return margin_cost_by_class(pairs, mask, d_warn=d_warn,
                                d_min_by_class=d_min_by_class)


def cost_sync_from_env_cfg(safety_cfg: dict, semantics_yaml: str) -> dict:
    """env yaml safety section -> trainer cost-channel params (single source
    with the env-side shaping band). (copy of
    lagrangian_ppo.cost_sync_from_env_cfg, C14)"""
    sem = semantics_with_safety_overrides(semantics_yaml, safety_cfg)
    return {"d_warn": float(sem.d_warn),
            "d_min_by_class": {CLASS_CROSS: float(sem.d_min["cross"]),
                               CLASS_SELF: float(sem.d_min["self"]),
                               CLASS_TABLE: float(sem.d_min["table"])}}


# --------------------------------------------------------------------------
# networks
# --------------------------------------------------------------------------

def _backbone(input_dim: int, hidden: tuple) -> "tuple[nn.Module, int]":
    """(copy of beta_actor._backbone: orthogonal init + tanh MLP)"""
    layers: "list[nn.Module]" = []
    prev = input_dim
    for width in hidden:
        lin = nn.Linear(prev, width)
        nn.init.orthogonal_(lin.weight, gain=2.0 ** 0.5)
        nn.init.zeros_(lin.bias)
        layers.extend((lin, nn.Tanh()))
        prev = width
    return nn.Sequential(*layers), prev


@dataclass(frozen=True)
class SACSample:
    raw: Tensor        # (N, 5) in (-1,1)^5 = duo_env action layout
    alpha: Tensor      # (N, 4) in (0,1)
    p: Tensor          # (N,)  in (-1,1)
    log_prob: Tensor   # (N,)  joint, tanh-space (Jacobian-corrected)


class TanhGaussianCoordinatorActor(nn.Module):
    """Tanh-squashed diagonal Gaussian over (alpha_1..4, p); see module
    docstring for the distribution-choice decision record."""

    def __init__(self, obs_dim: int, hidden: tuple = (256, 256),
                 log_std_min: float = -5.0, log_std_max: float = 2.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.backbone, last = _backbone(obs_dim, hidden)
        self.parameter_head = nn.Linear(last, 2 * ACTION_DIM)
        nn.init.orthogonal_(self.parameter_head.weight, gain=0.01)
        nn.init.zeros_(self.parameter_head.bias)

    def distribution_params(self, obs: Tensor) -> "tuple[Tensor, Tensor]":
        if obs.ndim != 2 or obs.shape[-1] != self.obs_dim:
            raise ValueError(f"obs must have shape [N, {self.obs_dim}]")
        mean, log_std = self.parameter_head(self.backbone(obs)).chunk(2, -1)
        return mean, log_std.clamp(self.log_std_min, self.log_std_max)

    @staticmethod
    def _log_prob_from_u(u: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
        """Joint log-density of t = tanh(u) under N(mean, exp(log_std)^2).
        Jacobian term log(1 - tanh(u)^2) in the softplus form
        2*(log 2 - u - softplus(-2u)) -- exact and stable for |u| large."""
        std = log_std.exp()
        normal_lp = (-0.5 * ((u - mean) / std).pow(2) - log_std
                     - 0.5 * math.log(2.0 * math.pi))
        jac = 2.0 * (LOG_2 - u - F.softplus(-2.0 * u))
        return (normal_lp - jac).sum(dim=-1)

    @staticmethod
    def split(raw: Tensor) -> "tuple[Tensor, Tensor]":
        """(N,5) raw tanh sample -> (alpha (N,4) in (0,1), p (N,))."""
        return (raw[:, :N_ALPHA] + 1.0) * 0.5, raw[:, N_ALPHA]

    @staticmethod
    def to_env_action(raw: Tensor) -> Tensor:
        """Raw tanh sample IS the duo_env layout (env re-derives
        alpha=(a+1)/2, p=a4, recovering split() exactly) -- identity."""
        return raw

    def act(self, obs: Tensor, deterministic: bool = False) -> SACSample:
        """Reparameterized sample (grad flows through raw for the actor
        update; wrap in no_grad on the rollout path)."""
        mean, log_std = self.distribution_params(obs)
        u = mean if deterministic else \
            mean + log_std.exp() * torch.randn_like(mean)
        log_prob = self._log_prob_from_u(u, mean, log_std)
        # keep strictly inside (-1,1) so atanh(raw) is finite on replay
        raw = torch.tanh(u).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        alpha, p = self.split(raw)
        return SACSample(raw=raw, alpha=alpha, p=p, log_prob=log_prob)

    def log_prob_of(self, obs: Tensor, raw: Tensor) -> Tensor:
        """Density of a stored raw action under the current policy (test /
        diagnostics path; SAC's actor update uses fresh rsamples instead)."""
        if raw.shape != (obs.shape[0], ACTION_DIM):
            raise ValueError(f"raw must have shape [N, {ACTION_DIM}]")
        if bool((raw.abs() >= 1.0).any().item()):
            raise ValueError("raw actions must lie strictly inside (-1, 1)")
        mean, log_std = self.distribution_params(obs)
        return self._log_prob_from_u(torch.atanh(raw), mean, log_std)


class TwinQ(nn.Module):
    """Twin Q(obs, action) heads (clipped-double-Q; the trainer decides the
    min/max combination direction per stream, see module docstring)."""

    def __init__(self, obs_dim: int, act_dim: int = ACTION_DIM,
                 hidden: tuple = (256, 256)):
        super().__init__()
        self.nets = nn.ModuleList()
        for _ in range(2):
            body, last = _backbone(obs_dim + act_dim, hidden)
            head = nn.Linear(last, 1)
            nn.init.orthogonal_(head.weight, gain=1.0)
            nn.init.zeros_(head.bias)
            self.nets.append(nn.Sequential(body, head))

    def forward(self, obs: Tensor, act: Tensor) -> "tuple[Tensor, Tensor]":
        x = torch.cat([obs, act], dim=-1)
        return self.nets[0](x).squeeze(-1), self.nets[1](x).squeeze(-1)


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------

class ReplayBuffer:
    """Preallocated ring buffer living on the training device (GPU on the
    server). Batch insertion with wraparound; uniform sampling."""

    def __init__(self, capacity: int, obs_dim: int, n_cost: int = 3,
                 act_dim: int = ACTION_DIM,
                 device: "str | torch.device" = "cpu"):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.device = torch.device(device)
        self._ptr = 0
        self._size = 0
        d = self.device
        self.obs = torch.empty(capacity, obs_dim, device=d)
        self.raw = torch.empty(capacity, act_dim, device=d)
        self.reward = torch.empty(capacity, device=d)
        self.cost = torch.empty(capacity, n_cost, device=d)
        self.next_obs = torch.empty(capacity, obs_dim, device=d)
        self.terminated = torch.empty(capacity, dtype=torch.bool, device=d)
        self.truncated = torch.empty(capacity, dtype=torch.bool, device=d)

    @property
    def size(self) -> int:
        return self._size

    def add_batch(self, obs: Tensor, raw: Tensor, reward: Tensor,
                  cost: Tensor, next_obs: Tensor, terminated: Tensor,
                  truncated: Tensor) -> None:
        n = obs.shape[0]
        if n > self.capacity:  # keep only the newest capacity rows
            obs, raw, reward = obs[-self.capacity:], raw[-self.capacity:], \
                reward[-self.capacity:]
            cost, next_obs = cost[-self.capacity:], next_obs[-self.capacity:]
            terminated = terminated[-self.capacity:]
            truncated = truncated[-self.capacity:]
            n = self.capacity
        idx = (self._ptr + torch.arange(n, device=self.device)) % self.capacity
        self.obs[idx] = obs.to(self.device)
        self.raw[idx] = raw.to(self.device)
        self.reward[idx] = reward.to(self.device)
        self.cost[idx] = cost.to(self.device)
        self.next_obs[idx] = next_obs.to(self.device)
        self.terminated[idx] = terminated.to(self.device)
        self.truncated[idx] = truncated.to(self.device)
        self._ptr = (self._ptr + n) % self.capacity
        self._size = min(self._size + n, self.capacity)

    def sample(self, batch_size: int) -> dict:
        if self._size == 0:
            raise RuntimeError("cannot sample from an empty buffer")
        idx = torch.randint(self._size, (batch_size,), device=self.device)
        return {"obs": self.obs[idx], "raw": self.raw[idx],
                "reward": self.reward[idx], "cost": self.cost[idx],
                "next_obs": self.next_obs[idx],
                "terminated": self.terminated[idx],
                "truncated": self.truncated[idx]}


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

@dataclass
class SACLagrangianConfig:
    # rollout: same fresh-batch shape as the PPO arm so the shared PID gains
    # see statistically comparable episode-mean costs each update cycle
    num_steps_per_env: int = 24
    gamma: float = 0.995
    # replay (defaults argued in the module docstring: ~5.3 iterations of
    # staleness at 4096 envs x 24 steps; classic 1e6 deliberately avoided)
    replay_capacity: int = 524_288
    learning_starts: int = 4096        # transitions before the first update
    batch_size: int = 4096
    updates_per_iteration: int = 32
    # sac
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    temp_lr: float = 3.0e-4
    tau: float = 0.005
    init_temperature: float = 1.0
    target_entropy: "float | None" = None   # None -> -ACTION_DIM
    max_grad_norm: float = 1.0
    # nets
    hidden: tuple = (256, 256)
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    # constraint controller (PPO-arm parity numbers, algo/lagrangian.py)
    cost_limits: tuple = (0.05, 0.05, 0.05)
    pid_kp: float = 0.5
    pid_ki: float = 0.05
    pid_kd: float = 0.1
    pid_integral_limit: float = 20.0
    controller: str = "pid"                 # pid | dual | off
    dual_lr: float = 0.05
    d_warn: float = 0.05
    d_min_by_class: "dict | None" = None    # None = v4 defaults (C14 parity)
    # freeze signature gates (same thresholds as the PPO arm)
    freeze_alpha_mean: float = 0.15
    freeze_p_pin: float = 0.95
    # io
    save_interval: int = 50
    log_every: int = 1


# --------------------------------------------------------------------------
# trainer
# --------------------------------------------------------------------------

class SACLagrangian:
    """SAC with twin reward Q + per-channel twin cost Q, automatic entropy
    temperature and a PID-controlled Lagrangian objective. Interface mirrors
    algo/lagrangian_ppo.LagrangianPPO (learn/save/load/stats.jsonl/wandb)."""

    def __init__(self, env, cfg: "SACLagrangianConfig | None" = None,
                 layout: "ObsLayout | None" = None,
                 log_dir: "str | Path | None" = None,
                 device: "str | torch.device" = "cpu", seed: int = 0,
                 wandb_logger=None):
        self.env = env
        self.wandb = wandb_logger
        # optional env hooks, same contract as the PPO arm (train_lagrangian
        # _EnvAdapter): step_telemetry() feeds the wandb safety/geometry
        # panels, cost_channels() serves the exemption-aware env-side cost
        tel = getattr(env, "step_telemetry", None)
        self._telemetry = tel if (wandb_logger is not None
                                  and callable(tel)) else None
        ch = getattr(env, "cost_channels", None)
        self._cost_channels = ch if callable(ch) else None
        self.cfg = cfg or SACLagrangianConfig()
        self.layout = layout or ObsLayout()
        self.device = torch.device(device)
        torch.manual_seed(seed)

        obs_dim = self.layout.obs_dim
        self.actor = TanhGaussianCoordinatorActor(
            obs_dim, hidden=tuple(self.cfg.hidden),
            log_std_min=self.cfg.log_std_min,
            log_std_max=self.cfg.log_std_max).to(self.device)
        self.q_reward = TwinQ(obs_dim, hidden=tuple(self.cfg.hidden)).to(self.device)
        self.q_cost = nn.ModuleDict({
            h: TwinQ(obs_dim, hidden=tuple(self.cfg.hidden)).to(self.device)
            for h in COST_HEADS})
        self.q_reward_target = copy.deepcopy(self.q_reward).requires_grad_(False)
        self.q_cost_target = copy.deepcopy(self.q_cost).requires_grad_(False)

        self.log_temp = torch.tensor(
            [math.log(self.cfg.init_temperature)], device=self.device,
            requires_grad=True)
        self.target_entropy = (self.cfg.target_entropy
                               if self.cfg.target_entropy is not None
                               else -float(ACTION_DIM))

        if self.cfg.controller == "pid":
            self.controller = PIDLagrangian(
                len(COST_HEADS), kp=self.cfg.pid_kp, ki=self.cfg.pid_ki,
                kd=self.cfg.pid_kd,
                integral_limit=self.cfg.pid_integral_limit).to(self.device)
        elif self.cfg.controller == "dual":
            self.controller = DualGradientLagrangian(
                len(COST_HEADS), learning_rate=self.cfg.dual_lr).to(self.device)
        elif self.cfg.controller == "off":
            self.controller = None
        else:
            raise ValueError(f"unknown controller {self.cfg.controller}")

        critic_params = list(self.q_reward.parameters())
        for h in COST_HEADS:
            critic_params += list(self.q_cost[h].parameters())
        self._critic_params = critic_params
        self.actor_opt = torch.optim.Adam(self.actor.parameters(),
                                          lr=self.cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(critic_params,
                                           lr=self.cfg.critic_lr)
        self.temp_opt = torch.optim.Adam([self.log_temp],
                                         lr=self.cfg.temp_lr)

        self.replay = ReplayBuffer(self.cfg.replay_capacity, obs_dim,
                                   n_cost=len(COST_HEADS), device=self.device)
        self.cost_limits = torch.tensor(self.cfg.cost_limits,
                                        device=self.device)
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.iteration = 0
        self._obs = None
        self._multipliers = torch.zeros(len(COST_HEADS), device=self.device)

    # ---- rollout ----------------------------------------------------------
    @torch.no_grad()
    def collect(self) -> dict:
        """Step the env T times with the current stochastic policy, push
        transitions into replay, return a fresh-rollout dict for the lambda
        update / freeze stats / telemetry. Cost source priority and done-row
        zeroing follow lagrangian_ppo.collect exactly (see its comments):
        cost of action t is the post-step proximity read from obs_{t+1};
        done rows are post-reset states so their cost is zeroed."""
        T, N = self.cfg.num_steps_per_env, self.env.num_envs
        lay = self.layout
        raw_buf = torch.empty(T, N, ACTION_DIM, device=self.device)
        rew_buf = torch.empty(T, N, device=self.device)
        cost_buf = torch.empty(T, N, len(COST_HEADS), device=self.device)
        term_buf = torch.empty(T, N, dtype=torch.bool, device=self.device)
        trunc_buf = torch.empty(T, N, dtype=torch.bool, device=self.device)

        if self._obs is None:
            obs_d, _ = self.env.reset()
            self._obs = obs_d["policy"].to(self.device)
        obs = self._obs
        tel_frames = [] if self._telemetry is not None else None
        for t in range(T):
            sample = self.actor.act(obs)
            action = TanhGaussianCoordinatorActor.to_env_action(sample.raw)
            obs_d, reward, terminated, truncated, _extras = self.env.step(action)
            nxt = obs_d["policy"].to(self.device)
            raw_buf[t] = sample.raw
            rew_buf[t] = reward.to(self.device)
            term_buf[t] = terminated.to(self.device)
            trunc_buf[t] = truncated.to(self.device)
            done = term_buf[t] | trunc_buf[t]
            cost = self._cost_channels() if self._cost_channels else None
            if cost is None:
                cost = cost_channels_from_obs(nxt, lay, self.cfg.d_warn,
                                              self.cfg.d_min_by_class)
            else:
                cost = cost.to(self.device)
            cost = torch.where(done.unsqueeze(-1), torch.zeros_like(cost),
                               cost)
            cost_buf[t] = cost
            self.replay.add_batch(obs, raw_buf[t], rew_buf[t], cost, nxt,
                                  term_buf[t], trunc_buf[t])
            if tel_frames is not None:
                frame = self._telemetry()
                if frame:
                    tel_frames.append(frame)
            obs = nxt
        self._obs = obs
        roll = {"raw": raw_buf, "reward": rew_buf, "cost": cost_buf,
                "terminated": term_buf, "truncated": trunc_buf}
        if tel_frames:
            roll["telemetry"] = tel_frames
        return roll

    # ---- critic targets ---------------------------------------------------
    def _critic_targets(self, batch: dict,
                        temp: Tensor) -> "tuple[Tensor, dict]":
        """Soft Bellman targets. Episode-end semantics mirror the PPO arm's
        _targets: true termination = no bootstrap; time-out = bootstrap, but
        the stored next_obs is the AUTO-RESET state of a fresh episode, so
        truncated rows bootstrap from the CURRENT obs instead (the same
        V(s_t)~=V(s_{t+1}) approximation lagrangian_ppo applies). Cost
        targets take the MAX of the twins (safety-pessimistic; the policy
        minimizes cost so underestimation is the exploitable bias) and carry
        no entropy term (entropy belongs to the reward objective only)."""
        with torch.no_grad():
            boot = (batch["truncated"] & ~batch["terminated"]).unsqueeze(-1)
            boot_obs = torch.where(boot, batch["obs"], batch["next_obs"])
            nxt = self.actor.act(boot_obs)
            mask = (~batch["terminated"]).float()
            q1, q2 = self.q_reward_target(boot_obs, nxt.raw)
            v_r = torch.min(q1, q2) - temp * nxt.log_prob
            y_r = batch["reward"] + self.cfg.gamma * mask * v_r
            y_c = {}
            for i, h in enumerate(COST_HEADS):
                c1, c2 = self.q_cost_target[h](boot_obs, nxt.raw)
                y_c[h] = (batch["cost"][:, i]
                          + self.cfg.gamma * mask * torch.max(c1, c2))
        return y_r, y_c

    def _soft_update(self) -> None:
        with torch.no_grad():
            tau = self.cfg.tau
            pairs = [(self.q_reward, self.q_reward_target)]
            pairs += [(self.q_cost[h], self.q_cost_target[h])
                      for h in COST_HEADS]
            for net, tgt in pairs:
                for p, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(1.0 - tau).add_(p, alpha=tau)

    def _gradient_step(self, batch: dict, multipliers: Tensor) -> dict:
        cfg = self.cfg
        temp = self.log_temp.exp().detach()
        y_r, y_c = self._critic_targets(batch, temp)
        obs, act = batch["obs"], batch["raw"]

        # critics (one optimizer over reward + all cost twins)
        q1, q2 = self.q_reward(obs, act)
        q_reward_loss = F.mse_loss(q1, y_r) + F.mse_loss(q2, y_r)
        qc_loss = {}
        for h in COST_HEADS:
            c1, c2 = self.q_cost[h](obs, act)
            qc_loss[h] = F.mse_loss(c1, y_c[h]) + F.mse_loss(c2, y_c[h])
        self.critic_opt.zero_grad()
        (q_reward_loss + sum(qc_loss.values())).backward()
        torch.nn.utils.clip_grad_norm_(self._critic_params, cfg.max_grad_norm)
        self.critic_opt.step()

        # actor: maximize Q_r - sum lambda*(1-gamma)*Q_c - temp*logpi
        # ((1-gamma) converts cost-to-go to per-step units, module docstring)
        sample = self.actor.act(obs)
        qr = torch.min(*self.q_reward(obs, sample.raw))
        penalty = torch.zeros_like(qr)
        for i, h in enumerate(COST_HEADS):
            qc = torch.max(*self.q_cost[h](obs, sample.raw))
            penalty = penalty + multipliers[i] * (1.0 - cfg.gamma) * qc
        actor_loss = (temp * sample.log_prob - qr + penalty).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                       cfg.max_grad_norm)
        self.actor_opt.step()

        # temperature: drive E[-logpi] toward target_entropy
        temp_loss = -(self.log_temp
                      * (sample.log_prob + self.target_entropy).detach()).mean()
        self.temp_opt.zero_grad()
        temp_loss.backward()
        self.temp_opt.step()

        self._soft_update()
        # consumed-estimate means (max twin, what the actor penalty sees):
        # the smoke's critic gate watches these RISE toward the true
        # discounted cost-to-go -- the TD loss itself chases a moving
        # bootstrap target and is not monotone (see run_smoke docstring)
        with torch.no_grad():
            qc_pred = {h: float(torch.max(*self.q_cost[h](obs, act))
                                .mean().item()) for h in COST_HEADS}
        return {"actor_loss": float(actor_loss.item()),
                "q_reward_loss": float(q_reward_loss.item()),
                "qc_loss": {h: float(v.item()) for h, v in qc_loss.items()},
                "qc_pred": qc_pred,
                "entropy_est": float(-sample.log_prob.mean().item())}

    # ---- update -----------------------------------------------------------
    def update(self, roll: dict) -> dict:
        """One policy-update cycle: lambda from the FRESH rollout's mean
        per-step cost (PPO-arm cadence and units; never from stale replay),
        then cfg.updates_per_iteration gradient steps off replay. Before
        learning_starts the gradient phase is skipped (the PID still runs:
        it regulates the OBSERVED behavior cost, which exists from step 0;
        its integral_limit bounds any pre-learning wind-up)."""
        observed = roll["cost"].mean(dim=(0, 1))
        if self.controller is not None:
            self._multipliers = self.controller.update(observed,
                                                       self.cost_limits)
        else:
            self._multipliers = torch.zeros(len(COST_HEADS),
                                            device=self.device)
        stats = {"actor_loss": 0.0, "q_reward_loss": 0.0,
                 "entropy_est": 0.0, "updates": 0}
        qc_sum = {h: 0.0 for h in COST_HEADS}
        qc_pred = {h: 0.0 for h in COST_HEADS}
        if self.replay.size >= max(self.cfg.learning_starts,
                                   self.cfg.batch_size):
            n_upd = self.cfg.updates_per_iteration
            for _ in range(n_upd):
                out = self._gradient_step(self.replay.sample(self.cfg.batch_size),
                                          self._multipliers)
                stats["actor_loss"] += out["actor_loss"]
                stats["q_reward_loss"] += out["q_reward_loss"]
                stats["entropy_est"] += out["entropy_est"]
                for h in COST_HEADS:
                    qc_sum[h] += out["qc_loss"][h]
                    qc_pred[h] += out["qc_pred"][h]
            for k in ("actor_loss", "q_reward_loss", "entropy_est"):
                stats[k] /= n_upd
            qc_sum = {h: v / n_upd for h, v in qc_sum.items()}
            qc_pred = {h: v / n_upd for h, v in qc_pred.items()}
            stats["updates"] = n_upd
        stats["q_cost_loss_by_head"] = {h: round(v, 5)
                                        for h, v in qc_sum.items()}
        stats["q_cost_pred_by_head"] = {h: round(v, 5)
                                        for h, v in qc_pred.items()}
        stats["alpha_temp"] = round(float(self.log_temp.exp().item()), 6)
        stats["multipliers"] = [round(float(v), 5) for v in self._multipliers]
        stats["cost_means"] = [round(float(v), 5) for v in observed]
        stats["buffer_size"] = self.replay.size
        return stats

    # ---- freeze signature (PPO-arm fields, SAC raw semantics) --------------
    @staticmethod
    def freeze_stats(roll: dict, cfg: SACLagrangianConfig) -> dict:
        """Same fields/thresholds as LagrangianPPO.freeze_stats so the peak
        harvester and sentinel tooling read SAC runs unchanged; only the raw
        decoding differs (tanh space: alpha=(raw+1)/2, p=raw_4)."""
        raw = roll["raw"]
        alpha = (raw[..., :N_ALPHA].reshape(-1, N_ALPHA) + 1.0) * 0.5
        p = raw[..., N_ALPHA].reshape(-1)
        arm_means = alpha.mean(dim=0)
        a_mean = float(alpha.mean().item())
        p_pin = float((p.abs() > 0.9).float().mean().item())
        return {
            "alpha_mean": round(a_mean, 4),
            "alpha_arm_means": [round(float(v), 4) for v in arm_means],
            "alpha_std": round(float(alpha.std().item()), 4),
            "alpha_sat_lo": round(float((alpha < 0.05).float().mean().item()), 4),
            "p_pinned_frac": round(p_pin, 4),
            "p_frac_pos": round(float((p > 0.05).float().mean().item()), 4),
            "p_frac_neg": round(float((p < -0.05).float().mean().item()), 4),
            "reward_mean": round(float(roll["reward"].mean().item()), 4),
            "frozen": bool(a_mean < cfg.freeze_alpha_mean
                           or p_pin > cfg.freeze_p_pin),
        }

    # ---- checkpointing ------------------------------------------------------
    def save(self, tag: "int | str | None" = None,
             stats: "dict | None" = None) -> Path:
        """Learner state only -- replay is NOT serialized (0.5M x 275-dim
        transitions ~= GBs per checkpoint); a resumed run refills its buffer
        within a few iterations, an accepted off-policy resume approximation."""
        assert self.log_dir is not None, "log_dir required for save()"
        tag = self.iteration if tag is None else tag
        path = self.log_dir / f"model_{tag}.pt"
        torch.save({
            "format": "sac_lagrangian_v1",
            "iteration": self.iteration,
            "actor": self.actor.state_dict(),
            "q_reward": self.q_reward.state_dict(),
            "q_cost": self.q_cost.state_dict(),
            "q_reward_target": self.q_reward_target.state_dict(),
            "q_cost_target": self.q_cost_target.state_dict(),
            "log_temp": self.log_temp.detach().cpu(),
            "controller": (self.controller.state_dict()
                           if self.controller is not None else None),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "temp_opt": self.temp_opt.state_dict(),
            "config": asdict(self.cfg),
            "obs_dim": self.layout.obs_dim,
            "stats": stats or {},
        }, path)
        return path

    def load(self, path: "str | Path") -> dict:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.q_reward.load_state_dict(ck["q_reward"])
        self.q_cost.load_state_dict(ck["q_cost"])
        self.q_reward_target.load_state_dict(ck["q_reward_target"])
        self.q_cost_target.load_state_dict(ck["q_cost_target"])
        with torch.no_grad():
            self.log_temp.copy_(ck["log_temp"].to(self.device))
        if self.controller is not None and ck.get("controller"):
            self.controller.load_state_dict(ck["controller"])
        for key, opt in (("actor_opt", self.actor_opt),
                         ("critic_opt", self.critic_opt),
                         ("temp_opt", self.temp_opt)):
            if key in ck:
                opt.load_state_dict(ck[key])
        self.iteration = int(ck.get("iteration", 0))
        return ck

    # ---- main loop ----------------------------------------------------------
    def learn(self, iterations: int, quiet: bool = False) -> list:
        history = []
        stats_file = (self.log_dir / "stats.jsonl").open("a") \
            if self.log_dir else None
        for _ in range(iterations):
            t0 = time.time()
            roll = self.collect()
            upd = self.update(roll)
            fz = self.freeze_stats(roll, self.cfg)
            rec = {"iter": self.iteration, **upd, **fz,
                   "wall_s": round(time.time() - t0, 2)}
            history.append(rec)
            if stats_file:
                stats_file.write(json.dumps(rec) + "\n")
                stats_file.flush()
            if not quiet and self.iteration % self.cfg.log_every == 0:
                print("SAC_LAG " + json.dumps(rec), flush=True)
            if self.wandb is not None and self.wandb.active:
                self.wandb.log(self.iteration, self._wandb_metrics(rec, roll))
            self.iteration += 1
            if self.log_dir and self.iteration % self.cfg.save_interval == 0:
                path = self.save(stats=rec)
                if self.wandb is not None:
                    self.wandb.log_checkpoint(self.iteration, path,
                                              score=rec.get("reward_mean"))
        if stats_file:
            stats_file.close()
        if self.log_dir:
            path = self.save(tag="last", stats=history[-1] if history else None)
            if self.wandb is not None and history:
                self.wandb.log_checkpoint(self.iteration, path,
                                          score=history[-1].get("reward_mean"))
        return history

    # ---- wandb panels (same sec.5 groups as the PPO arm) -----------------------
    def _wandb_metrics(self, rec: dict, roll: dict) -> dict:
        """PPO-arm panel groups adapted to SAC: loss/* carries the SAC
        losses, health/* adds alpha_temp + buffer freshness. Env-side
        telemetry passthrough copied from lagrangian_ppo._wandb_metrics."""
        m = {
            "task/reward_mean": rec["reward_mean"],
            "task/reward_std": float(roll["reward"].std().item()),
            "health/alpha_mean": rec["alpha_mean"],
            "health/alpha_std": rec["alpha_std"],
            "health/p_pinned_frac": rec["p_pinned_frac"],
            "health/p_frac_pos": rec["p_frac_pos"],
            "health/p_frac_neg": rec["p_frac_neg"],
            "health/entropy": rec["entropy_est"],
            "health/alpha_temp": rec["alpha_temp"],
            "health/frozen": float(rec["frozen"]),
            "sys/iter_wall_s": rec["wall_s"],
            "sys/fps_ctrl_steps": round(
                self.cfg.num_steps_per_env * self.env.num_envs
                / max(rec["wall_s"], 1e-9), 1),
            "sys/buffer_size": rec["buffer_size"],
            "loss/actor": rec["actor_loss"],
            "loss/q_reward": rec["q_reward_loss"],
        }
        for i, h in enumerate(COST_HEADS):
            m[f"loss/q_cost_{h}"] = rec["q_cost_loss_by_head"][h]
            m[f"safety/cost_{h}"] = rec["cost_means"][i]
            m[f"safety/limit_{h}"] = float(self.cfg.cost_limits[i])
            m[f"safety/lambda_{h}"] = rec["multipliers"][i]
        raw = roll["raw"]
        alpha = (raw[..., :N_ALPHA].reshape(-1, N_ALPHA) + 1.0) * 0.5
        qs = torch.quantile(
            alpha.float(), torch.tensor([0.1, 0.9], device=alpha.device), dim=0)
        for i, arm in enumerate(ARM_KEYS):
            m[f"health/alpha_{arm}_mean"] = rec["alpha_arm_means"][i]
            m[f"health/alpha_{arm}_p10"] = round(float(qs[0, i].item()), 4)
            m[f"health/alpha_{arm}_p90"] = round(float(qs[1, i].item()), 4)
        m["health/p_mean"] = round(
            float(raw[..., N_ALPHA].reshape(-1).mean().item()), 4)
        tel = roll.get("telemetry")
        if tel:
            cat = lambda k: torch.cat([f[k].reshape(f[k].shape[0], -1).float()
                                       for f in tel if k in f])
            if "violation" in tel[0]:
                m["safety/violation_rate"] = round(
                    float(cat("violation").mean().item()), 5)
            if "tube" in tel[0]:
                m["safety/tube_fraction"] = round(
                    float(cat("tube").mean().item()), 5)
            if "bs_active" in tel[0]:
                bs = cat("bs_active").reshape(-1, len(ARM_KEYS)).mean(dim=0)
                for i, arm in enumerate(ARM_KEYS):
                    m[f"safety/backstop_rate_{arm}"] = round(
                        float(bs[i].item()), 5)
            for k in ("mm_cross", "mm_table", "mm_self_f", "mm_self_u"):
                if k in tel[0]:
                    v = cat(k).reshape(-1)
                    pq = torch.quantile(
                        v, torch.tensor([0.01, 0.5], device=v.device))
                    m[f"geom/{k}_p1"] = round(float(pq[0].item()), 5)
                    m[f"geom/{k}_p50"] = round(float(pq[1].item()), 5)
        else:
            m["safety/violation_rate"] = round(
                float(roll["terminated"].float().mean().item()), 5)
        if self.device.type == "cuda" and torch.cuda.is_available():
            m["sys/gpu_mem_gb"] = round(
                torch.cuda.memory_allocated(self.device) / 2 ** 30, 3)
        return m


# --------------------------------------------------------------------------
# local full-chain smoke
# --------------------------------------------------------------------------

def smoke_config() -> SACLagrangianConfig:
    """Smoke-scale knobs: 16 envs x 24 steps = 384 transitions/iter, so
    learning_starts 512 means updates begin at iteration 2 and 23 of the 25
    iterations exercise the full gradient path."""
    return SACLagrangianConfig(
        replay_capacity=65_536, learning_starts=512, batch_size=256,
        updates_per_iteration=32, save_interval=10)


def run_smoke(n_envs: int = 16, iterations: int = 25, seed: int = 0,
              out_dir: "str | Path | None" = None, quiet: bool = False,
              cfg: "SACLagrangianConfig | None" = None) -> dict:
    """KinematicDuoEnv full chain: real geometry + conflict traffic +
    tanh-Gaussian actor + twin Q stacks + PID-Lagrangian off replay.

    Gate mapping vs the PPO arm's seven checks (none silently dropped):
    - loss_finite / lambda_responded / lambda_regulating /
      no_freeze_signature / alpha_in_band_late / p_two_sided: identical
      semantics (freeze stats decode tanh raw instead of Beta raw).
    - critic_learning (ADAPTED, reason on record): the PPO gate rode the
      SELF cost head's GAE value loss decreasing -- a regression onto
      FIXED per-rollout targets. SAC's TD targets are themselves
      bootstrapped and keep MOVING for thousands of updates (Q_c grows
      from init ~0 toward cost-per-step x effective horizon), so within a
      25-iteration window the TD loss chasing that moving target may rise
      while learning is perfectly healthy (measured: 0.013 -> 0.071 on a
      passing run). The equivalent observable is bootstrap PROPAGATION:
      the SELF head's consumed Q_c estimate (max twin, what the actor
      penalty sees) must RISE from its zero init toward the true
      discounted cost-to-go, medianed over the update-active iterations'
      thirds. Same stream choice as PPO (self = dense spike-free at this
      traffic); divergence-without-fit is covered by loss_finite.
    - reward_mean stays REPORTED, not gated (PPO-arm reasoning: one
      terminal violation swings a 16-env batch mean ~0.26).
    SAC-specific additions: temperature_adapting (alpha_temp moved off its
    init and stayed finite -- the sec.6 temp x lambda coupling is actually
    exercised) and replay_warmed (gradient phase ran; guards against a
    silently skipped learning path passing the other gates)."""
    from statistics import median

    from safeduo.algo.kinematic_env import KinematicDuoEnv, KinematicEnvConfig

    env = KinematicDuoEnv(
        n_envs, cfg=KinematicEnvConfig(episode_length=120), seed=seed)
    out = Path(out_dir) if out_dir else None
    cfg = cfg or smoke_config()
    trainer = SACLagrangian(env, cfg, log_dir=out, seed=seed)
    history = trainer.learn(iterations, quiet=quiet)

    k = max(3, iterations // 5)
    late = history[-k:]
    lam_series = [h["multipliers"] for h in history]
    cost_series = [h["cost_means"] for h in history]
    lam_max = [max(l[c] for l in lam_series) for c in range(3)]
    responded = any(m > 1e-4 for m in lam_max)
    ended_regulated = all(
        cost_series[-1][c] <= trainer.cfg.cost_limits[c] * 1.5
        or lam_series[-1][c] > 1e-4 for c in range(3))
    upd_iters = [h for h in history if h["updates"] > 0]
    ku = max(3, len(upd_iters) // 3) if upd_iters else 0
    qc_self_loss = [h["q_cost_loss_by_head"]["self"] for h in upd_iters]
    qc_self_pred = [h["q_cost_pred_by_head"]["self"] for h in upd_iters]
    early_pred = median(qc_self_pred[:ku]) if upd_iters else 0.0
    late_pred = median(qc_self_pred[-ku:]) if upd_iters else 0.0
    late_alpha = sum(h["alpha_mean"] for h in late) / k
    loss_keys = ("actor_loss", "q_reward_loss")
    finite = all(
        math.isfinite(h[k2]) for h in history for k2 in loss_keys) and all(
        math.isfinite(v) for h in history
        for v in h["q_cost_loss_by_head"].values())
    temps = [h["alpha_temp"] for h in history]
    checks = {
        "loss_finite": finite,
        "critic_learning": bool(upd_iters) and late_pred > early_pred
                           and late_pred > 0.0,
        "lambda_responded": responded,
        "lambda_regulating": ended_regulated,
        "no_freeze_signature": not any(h["frozen"] for h in late),
        "alpha_in_band_late": 0.3 <= late_alpha <= 0.9,
        "p_two_sided": late[-1]["p_frac_pos"] >= 0.01
                       and late[-1]["p_frac_neg"] >= 0.01,
        "temperature_adapting": bool(upd_iters)
                                and abs(temps[-1] - cfg.init_temperature) > 1e-4
                                and math.isfinite(temps[-1]) and temps[-1] > 0,
        "replay_warmed": bool(upd_iters),
    }
    summary = {
        "n_envs": n_envs, "iterations": iterations, "seed": seed,
        "qc_self_pred_median_early_vs_late": [round(early_pred, 4),
                                              round(late_pred, 4)],
        "qc_self_loss_median_early_vs_late": [
            round(median(qc_self_loss[:ku]), 4) if upd_iters else None,
            round(median(qc_self_loss[-ku:]), 4) if upd_iters else None],
        "reward_trend_first_last": [history[0]["reward_mean"],
                                    history[-1]["reward_mean"]],
        "alpha_temp_first_last": [temps[0], temps[-1]],
        "final": history[-1], "lambda_max": lam_max,
        "checks": checks, "pass": all(checks.values()),
    }
    if out:
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def print_smoke_summary(summary: dict) -> None:
    """Per-gate true/false lines + the PPO-arm style JSON block."""
    for name, ok in summary["checks"].items():
        print(f"CHECK {name} {str(bool(ok)).lower()}", flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "final"},
                     indent=2), flush=True)
    print("SMOKE " + ("PASS" if summary["pass"] else "FAIL"), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="local kinematic full-chain smoke (CPU)")
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--iterations", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/analysis/c15_sac_lagrangian_smoke")
    args = ap.parse_args()
    if not args.smoke:
        ap.error("only --smoke is wired here; server training goes through "
                 "safeduo.algo.train_sac_lagrangian")
    summary = run_smoke(args.n_envs, args.iterations, args.seed, args.out)
    print_smoke_summary(summary)
    raise SystemExit(0 if summary["pass"] else 1)


if __name__ == "__main__":
    main()
