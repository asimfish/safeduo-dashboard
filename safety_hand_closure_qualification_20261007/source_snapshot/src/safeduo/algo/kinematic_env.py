"""Local closed-loop kinematic DuoEnv stand-in (CPU, no Isaac).

Why this exists (C5-W7): the v4 anti-freeze recipe (PPO-Lagrangian + Beta
actor, algo/lagrangian_ppo.py) must be smoke-tested END TO END locally before
any GPU time -- the v3 collapse was a reward-landscape disease that only
shows up in closed loop. bc_smoke_check.collect_closedloop_kinematic already
proved the ingredients (real geometry + conflict traffic + alpha-scaled
execution) catch closed-loop pathologies the holdout gates miss; this module
lifts that inline loop into a reusable env with the DuoEnv step contract so
the same trainer code drives BOTH this and the live Isaac env.

Contract (mirrors isaaclab DirectRLEnv usage in train_ppo):
    obs, _extras = env.reset()
    obs, reward, terminated, truncated, extras = env.step(action)
obs is {"policy": (N, 275)} -- bit-compatible layout with
DuoEnv._get_observations via the shared obs_features_duo_env (P0 fixed:
[q, qd, pair_feats(M*5), mask(M), cmd(26), alpha_prev(4), p_prev(1)]).
action is the duo_env coordinator layout (N, 5): env re-derives
alpha = (clamp(a,-1,1)+1)/2 per arm, p = clamp(a4,-1,1).

Fidelity notes (deliberate, documented):
- Execution is the first-order budget effect exec = alpha * cmd -- no PD lag,
  no backstop QP, so p has NO dynamic effect here (in duo_env p steers the
  backstop's right-of-way split). Freeze diagnostics therefore key on alpha;
  p health is still observable through the policy distribution itself.
- violation = any true sphere-pair margin < 0 (same zero point as duo_env's
  sphere layer), terminated on violation, truncated on time-out -- exactly
  the two episode-end channels validity_aware_gae distinguishes.
- Rewards keep duo_env's sign conventions on the subset that is meaningful
  kinematically: w_track tracking penalty, optional w_margin proximity
  shaping (OFF in the Lagrangian arm -- the same signal rides the constraint
  channels instead), w_violation terminal penalty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from safeduo.algo.reward_shaping import margin_cost_by_class, shaped_margin_cost_from_pairs
from safeduo.algo.warmstart_oracle import obs_features_duo_env
from safeduo.safety.types import ARM_KEYS, DOF_OF

# duo_env prep poses (v3-era defaults; v4 callers pass their own)
FR3_PREP_Q = (0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741)
UR5E_PREP_Q = (0.0, -2.2, 1.9, -1.383, -1.57, 0.0)
DEFAULT_INIT_Q = {"F_L": FR3_PREP_Q, "F_R": FR3_PREP_Q,
                  "U_L": UR5E_PREP_Q, "U_R": UR5E_PREP_Q}


@dataclass
class KinematicEnvConfig:
    episode_length: int = 240
    dt: float = 1.0 / 60.0
    init_jitter: float = 0.02
    # duo_env reward subset (duo_env_v4.yaml coordinator.reward signs)
    w_track: float = 10.0
    w_margin: float = 0.0          # Lagrangian arm: costs ride channels, not reward
    w_violation: float = 100.0
    d_warn: float = 0.05
    terminate_on_violation: bool = True
    init_q: dict = field(default_factory=lambda: dict(DEFAULT_INIT_Q))


def default_conflict_source(n_envs: int, provider, device: str = "cpu",
                            env_yaml: str = "duo_env.yaml",
                            amp_max: float = 0.015):
    """Conflict-rich ConflictMixSource on the provider's geometry (the
    bc_smoke_check kinematic-backend recipe, factored out)."""
    from safeduo.delta.l1_random import JacobianMapper
    from safeduo.delta.l2_env_source import ConflictMixSource, RealScenePoses

    poses = RealScenePoses(env_yaml)
    f_sign = 1.0 if provider.layout.base_pose("F_L")[0][0] > 0 else -1.0
    ws = poses.workspace_spec(f_sign=f_sign)
    mapper = JacobianMapper(provider.ee_jacobian, poses.ws_lo, poses.ws_hi)
    cfg = {"mix": {"l1": 0.1, "l2": 0.9}, "n_variants": 8, "split": "train",
           "scenarios": {"head_on_crossing": 2.0, "center_grab": 2.0,
                         "chase": 2.0, "handover_approach": 1.0}}
    return ConflictMixSource(n_envs, cfg, device=device, mapper=mapper,
                             ws=ws, amp_max=amp_max)


class KinematicDuoEnv:
    """See module docstring. provider/source injectable so the same env runs
    the v0/v3 layout (RealGeometryProvider default) or the v4 real layout
    (baselines/real_geometry make_v4_provider)."""

    def __init__(self, n_envs: int, provider=None, source=None,
                 cfg: "KinematicEnvConfig | None" = None,
                 device: str = "cpu", seed: int = 0,
                 env_yaml: str = "duo_env.yaml"):
        self.num_envs = n_envs
        self.device = torch.device(device)
        self.cfg = cfg or KinematicEnvConfig()
        if provider is None:
            from safeduo.baselines.real_geometry import RealGeometryProvider, SceneLayout

            provider = RealGeometryProvider(n_envs, device=device,
                                            layout=SceneLayout(base_x=0.70))
        self.provider = provider
        self.source = source or default_conflict_source(n_envs, provider,
                                                        device=device,
                                                        env_yaml=env_yaml)
        self.gen = torch.Generator().manual_seed(seed)
        self._q = {a: torch.zeros(n_envs, DOF_OF[a], device=self.device)
                   for a in ARM_KEYS}
        self._qd = {a: torch.zeros_like(self._q[a]) for a in ARM_KEYS}
        self.episode_length_buf = torch.zeros(n_envs, dtype=torch.long,
                                              device=self.device)
        self._alpha_prev = torch.ones(n_envs, 4, device=self.device)
        self._p_prev = torch.zeros(n_envs, device=self.device)
        self._pending_cmd = None
        self._state = None

    # ---- episode plumbing ------------------------------------------------
    def _reset_envs(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        for arm in ARM_KEYS:
            base = torch.tensor(self.cfg.init_q[arm], dtype=torch.float32,
                                device=self.device).expand(n, -1).clone()
            if self.cfg.init_jitter > 0:
                base += (torch.rand(n, DOF_OF[arm], generator=self.gen)
                         * 2 - 1).to(self.device) * self.cfg.init_jitter
            self._q[arm][env_ids] = base
            self._qd[arm][env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self._alpha_prev[env_ids] = 1.0
        self._p_prev[env_ids] = 0.0
        self.source.reset(env_ids, self.gen)

    def _observe(self) -> dict:
        self._state = self.provider.scene_state(self._q, self._qd,
                                                dt=self.cfg.dt)
        self._pending_cmd = self.source.sample(self._state)
        obs = obs_features_duo_env(self._state, self._pending_cmd,
                                   self._alpha_prev, self._p_prev)
        return {"policy": obs}

    def reset(self) -> tuple:
        self._reset_envs(torch.arange(self.num_envs, device=self.device))
        return self._observe(), {}

    # ---- step ---------------------------------------------------------------
    def step(self, action: torch.Tensor) -> tuple:
        a = action.clamp(-1.0, 1.0)
        alpha = (a[:, :4] + 1.0) * 0.5
        cmd = self._pending_cmd
        exec_q, cmd_s = {}, cmd.stacked()
        for i, arm in enumerate(ARM_KEYS):
            exec_q[arm] = alpha[:, i:i + 1] * cmd.delta_q[arm]
            self._q[arm] = self._q[arm] + exec_q[arm]
            self._qd[arm] = exec_q[arm] / self.cfg.dt
        self.episode_length_buf += 1

        # margins/violation on the post-step configuration (duo_env order)
        mm = self.provider.min_margin_by_class(self._q)
        min_margin = torch.minimum(torch.minimum(mm["cross"], mm["self"]),
                                   mm["table"])
        violation = min_margin < 0.0
        state_post = self.provider.scene_state(self._q, self._qd,
                                               dt=self.cfg.dt)

        exec_s = torch.cat([exec_q[a] for a in ARM_KEYS], dim=-1)
        track = -self.cfg.w_track * (exec_s - cmd_s).pow(2).sum(-1)
        reward = track - self.cfg.w_violation * violation.float()
        if self.cfg.w_margin > 0.0:
            reward = reward - self.cfg.w_margin * shaped_margin_cost_from_pairs(
                state_post.active_pairs, state_post.active_mask,
                d_warn=self.cfg.d_warn)
        cost3 = margin_cost_by_class(state_post.active_pairs,
                                     state_post.active_mask,
                                     d_warn=self.cfg.d_warn)

        terminated = violation if self.cfg.terminate_on_violation else \
            torch.zeros_like(violation)
        truncated = self.episode_length_buf >= self.cfg.episode_length
        extras = {"cost_by_class": cost3, "min_margin": min_margin,
                  "alpha": alpha, "p": a[:, 4],
                  "time_outs": truncated}

        self._alpha_prev = alpha
        self._p_prev = a[:, 4]
        done = terminated | truncated
        if done.any():
            self._reset_envs(done.nonzero(as_tuple=False).squeeze(-1))
        return self._observe(), reward, terminated, truncated, extras
