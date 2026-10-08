"""One-step analytic warm-start oracle: (alpha*, p*) supervision for BC pretrain.

ROUND2_DELTAS item C-2: before PPO, the coordinator is behavior-cloned from an
analytic oracle so it starts from "sane yielding behavior" instead of random
gating. The oracle reuses the strengthened CBF-QP core (short-horizon intent-
preserving planning):

  p*     = the filter's fixed-rule hysteretic priority (aggressor yields,
           deadband + dwell + deterministic symmetric tiebreak).
  alpha* = per-arm progress that SURVIVES the safety QP under p*, i.e.
           effective_alpha(cmd, qp_exec): 1 when the arm's commanded motion is
           feasible, shrinking exactly as much as safety requires. This is the
           minimal-intervention gate consistent with the constraint set --
           what we want the learned coordinator to reproduce before RL
           sharpens it on liveness.

Labels are therefore *feasibility-aware*, not distance heuristics; no-conflict
states give alpha* ~= 1 / p* = 0 by construction (tests pin this).

BC dataset: rollouts of any DeltaSource through a GeometryProvider; features
are deployment observations only (q, qd, delta history, active pairs) --
matching the actor's observation contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from safeduo.baselines.base import GeometryProvider
from safeduo.baselines.strong_cbf_qp import StrongCBFQPConfig, StrongCBFQPFilter
from safeduo.delta._contract_stub import (
    ARM_KEYS,
    DeltaCmd,
    DeltaSource,
    SceneState,
    pair_obs_features,
)


class WarmStartOracle:
    """Wraps the strengthened CBF-QP core as a label generator."""

    def __init__(self, n_envs: int, provider: GeometryProvider,
                 cfg: "StrongCBFQPConfig | None" = None,
                 device: "str | torch.device" = "cpu",
                 dof_of: "dict | None" = None):
        self.filt = StrongCBFQPFilter(n_envs, provider, cfg, device=device,
                                      dof_of=dof_of)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.filt.reset(env_ids)

    def labels_and_exec(self, state: SceneState, cmd: DeltaCmd) -> tuple:
        """-> (alpha_star (N,4), p_star (N,), exec delta dict).

        Single filter call per step: the CBF-QP filter is stateful (delta
        history, priority dwell), so labels and the safe rollout step must
        come from the same invocation."""
        out = self.filt.filter(state, cmd)
        return out.alpha, out.priority_p, out.delta_exec

    def labels(self, state: SceneState, cmd: DeltaCmd) -> tuple:
        alpha, p, _ = self.labels_and_exec(state, cmd)
        return alpha, p


@dataclass
class BCDatasetConfig:
    steps: int = 200
    hist_h: int = 5           # delta history depth in the observation
    dt: float = 0.02
    seed: int = 0
    # spawn-distribution alignment (A3 W3 sec.8 r1 autopsy: BC data born at
    # real_geometry's stale constants sat 1.3 rad away from the fixed env
    # spawn -> extrapolation saturation). init_q overrides provider.default_q
    # (arm -> (dof,) tensor/tuple); jitter mirrors the env reset (duo_env
    # _reset_idx writes exact default_joint_pos -> 0.0, was hardcoded 0.1).
    init_q: "dict | None" = None
    init_jitter: float = 0.0


def obs_features(state: SceneState, cmd: DeltaCmd, hist: torch.Tensor) -> torch.Tensor:
    """Deployment-observable feature vector (actor side, no privileged info):
    [q(26), qd(26), delta_cmd(26), delta_hist(26*H), active_pairs(M*4)]."""
    parts = [torch.cat([state.q[a] for a in ARM_KEYS], dim=-1),
             torch.cat([state.qd[a] for a in ARM_KEYS], dim=-1),
             cmd.stacked(),
             hist.flatten(1)]
    if state.active_pairs is not None:
        parts.append(state.active_pairs.flatten(1))
    return torch.cat(parts, dim=-1)


def obs_features_duo_env(state: SceneState, cmd: DeltaCmd,
                         alpha_prev: torch.Tensor,
                         p_prev: torch.Tensor) -> torch.Tensor:
    """duo_env coordinator policy-observation layout, bit-compatible with
    DuoEnv._get_observations (v3 P0): [q, qd, pair_obs_features(active_pairs),
    active_mask, delta_cmd, alpha_prev, p_prev] -- with M=32 that is 275 dims.
    Pair rows are featurized to [dist, closing_vel, class onehot(3)] by the
    shared safety.types.pair_obs_features (the raw pair_id column caused the
    r2 birth saturation; single source of truth prevents layout drift). BC
    nets pretrained on this layout can initialize the A5 PPO actor directly
    (alpha_prev/p_prev = previous step's oracle labels, matching deployment
    where they are the policy's own previous action)."""
    assert state.active_pairs is not None and state.active_mask is not None, \
        "duo_env layout requires the active-pair set in SceneState"
    return torch.cat([
        torch.cat([state.q[a] for a in ARM_KEYS], dim=-1),
        torch.cat([state.qd[a] for a in ARM_KEYS], dim=-1),
        pair_obs_features(state.active_pairs, state.active_mask),
        state.active_mask.float(),
        cmd.stacked(),
        alpha_prev,
        p_prev.unsqueeze(-1),
    ], dim=-1)


def build_bc_dataset(source: DeltaSource, provider, n_envs: int,
                     cfg: "BCDatasetConfig | None" = None,
                     oracle_cfg: "StrongCBFQPConfig | None" = None,
                     dof_of: "dict | None" = None,
                     scene_state_fn=None,
                     obs_layout: str = "legacy") -> dict:
    """Roll `source` through the provider's world, label with the oracle.

    scene_state_fn(q, qd) -> SceneState; defaults to provider.scene_state.
    Executes the ORACLE-FILTERED delta (labels correspond to the state
    distribution the safe policy visits, not the unfiltered crash states).

    obs_layout: "legacy" = W2 feature vector (q,qd,cmd,hist,pairs);
    "duo_env" = the A5 coordinator policy observation (obs_features_duo_env),
    so the pretrained trunk transfers into PPO without feature remapping.
    """
    assert obs_layout in ("legacy", "duo_env"), obs_layout
    cfg = cfg or BCDatasetConfig()
    g = torch.Generator().manual_seed(cfg.seed)
    state_fn = scene_state_fn or (lambda q, qd: provider.scene_state(q, qd, dt=cfg.dt))
    oracle = WarmStartOracle(n_envs, provider, oracle_cfg, dof_of=dof_of)
    oracle.reset(torch.arange(n_envs))
    source.reset(torch.arange(n_envs), g)
    if cfg.init_q is not None:
        q = {}
        for a, v in cfg.init_q.items():
            base = torch.as_tensor(v, dtype=torch.float32).expand(
                n_envs, -1).clone()
            if cfg.init_jitter > 0.0:
                base += (torch.rand(base.shape, generator=g) * 2 - 1) \
                    * cfg.init_jitter
            q[a] = base
    else:
        q = provider.default_q(jitter=cfg.init_jitter, generator=g)
    qd = {a: torch.zeros_like(q[a]) for a in ARM_KEYS}
    hist = None
    alpha_prev = torch.ones(n_envs, 4)
    p_prev = torch.zeros(n_envs)
    X, A, P = [], [], []
    for _ in range(cfg.steps):
        state = state_fn(q, qd)
        cmd = source.sample(state)
        stacked = cmd.stacked()
        if hist is None:
            hist = torch.zeros(n_envs, cfg.hist_h, stacked.shape[-1])
        alpha_star, p_star, exec_ = oracle.labels_and_exec(state, cmd)
        if obs_layout == "duo_env":
            X.append(obs_features_duo_env(state, cmd, alpha_prev, p_prev))
        else:
            X.append(obs_features(state, cmd, hist))
        A.append(alpha_star)
        P.append(p_star)
        alpha_prev, p_prev = alpha_star, p_star
        hist = torch.cat([hist[:, 1:], stacked.unsqueeze(1)], dim=1)
        q = {a: q[a] + exec_[a] for a in ARM_KEYS}
        qd = {a: exec_[a] / cfg.dt for a in ARM_KEYS}
    return {
        "obs": torch.cat(X),          # (T*N, F)
        "alpha_star": torch.cat(A),   # (T*N, 4)
        "p_star": torch.cat(P),       # (T*N,)
    }


class CoordinatorMLP(torch.nn.Module):
    """The pi_safe head shape used for BC pretrain (PPO fine-tune reuses it)."""

    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden), torch.nn.ELU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ELU(),
        )
        self.head_alpha = torch.nn.Linear(hidden, 4)
        self.head_p = torch.nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple:
        z = self.body(x)
        return torch.sigmoid(self.head_alpha(z)), torch.tanh(self.head_p(z)).squeeze(-1)


def bc_pretrain(dataset: dict, epochs: int = 20, lr: float = 3e-4,
                batch_size: int = 512, hidden: int = 256,
                seed: int = 0, device: "str | torch.device" = "cpu") -> tuple:
    """Minimal BC loop -> (model, loss_history). W2 skeleton: MSE on both heads;
    PPO fine-tune entry (RSL-RL) lands with the C3 training recipe."""
    torch.manual_seed(seed)
    obs, a_star, p_star = (dataset["obs"].to(device), dataset["alpha_star"].to(device),
                           dataset["p_star"].to(device))
    model = CoordinatorMLP(obs.shape[-1], hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    n = obs.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        ep_loss, n_batches = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            a_hat, p_hat = model(obs[idx])
            loss = torch.nn.functional.mse_loss(a_hat, a_star[idx]) \
                + torch.nn.functional.mse_loss(p_hat, p_star[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_batches += 1
        losses.append(ep_loss / max(n_batches, 1))
    return model, losses
