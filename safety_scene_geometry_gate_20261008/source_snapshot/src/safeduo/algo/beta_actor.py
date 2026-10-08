"""Bounded Beta policy head for the coordinator action contract alpha(4)+p(1).

PROVENANCE: ported from the double_hand workspace on bjxy_5090 NAS
(/mnt/nas/data/lyf/double_hand policies/models.py CentralizedAlphaActor +
CentralizedCritics; pure torch). Port adaptations for the SafeDuo contract:
  - double_hand's action space was 4 per-arm alpha gates; SafeDuo adds the
    right-of-way scalar p in [-1, 1]. The joint distribution stays a
    5-dimensional Independent Beta in (0,1)^5; dims 0..3 map to alpha
    directly, dim 4 maps affinely to p = 2*b - 1. The affine map's constant
    log-Jacobian (log 2) cancels in PPO ratios; log_prob is reported in raw
    Beta space and the RAW action is what must be stored/replayed for
    evaluate_actions (same discipline as double_hand's alpha-only actor).
  - backbone init (orthogonal, tanh), softplus + min_concentration
    parameterization and deterministic-mode = distribution mean kept exactly.
  - CentralizedCritics generalized to n named heads (SafeDuo: reward +
    cross/self/table cost critics for PPO-Lagrangian; heads stay independent
    MLPs so cost estimates cannot bleed into the reward stream).

Why Beta (vs the rsl_rl Gaussian): alpha lives on a closed interval and the
useful yielding behavior sits at the boundary (full stop / full pass). A
squashed Gaussian concentrates density mid-interval and clips gradients at
the walls; Beta puts mass exactly on [0,1] with well-defined boundary limits.
Wiring (@A2): duo_env maps incoming actions via alpha=(a+1)/2, p=a4, so
`to_env_action` converts a raw Beta sample into that layout -- the env needs
no change; train_ppo swaps its policy class for this actor + MultiHeadCritic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Beta, Independent
from torch.nn import functional as F

N_ALPHA = 4
ACTION_DIM = 5  # alpha(4) + p(1)


def _backbone(input_dim: int, hidden: tuple) -> tuple[nn.Module, int]:
    layers: list[nn.Module] = []
    prev = input_dim
    for width in hidden:
        lin = nn.Linear(prev, width)
        nn.init.orthogonal_(lin.weight, gain=2.0 ** 0.5)
        nn.init.zeros_(lin.bias)
        layers.extend((lin, nn.Tanh()))
        prev = width
    return nn.Sequential(*layers), prev


@dataclass(frozen=True)
class CoordinatorSample:
    raw: Tensor        # (N, 5) in (0,1)^5 -- store THIS for PPO replay
    alpha: Tensor      # (N, 4) in (0,1)
    p: Tensor          # (N,)  in (-1,1)
    log_prob: Tensor   # (N,)  joint, raw-space
    entropy: Tensor    # (N,)


class BetaCoordinatorActor(nn.Module):
    """Joint bounded Beta policy over (alpha_1..4, p)."""

    def __init__(self, obs_dim: int, hidden: tuple = (256, 256),
                 min_concentration: float = 1.0, hazard_head: bool = False):
        super().__init__()
        if min_concentration <= 0.0:
            raise ValueError("min_concentration must be positive")
        self.obs_dim = obs_dim
        self.min_concentration = min_concentration
        self.backbone, last = _backbone(obs_dim, hidden)
        self.parameter_head = nn.Linear(last, 2 * ACTION_DIM)
        nn.init.orthogonal_(self.parameter_head.weight, gain=0.01)
        nn.init.zeros_(self.parameter_head.bias)
        # a25/P3（R22-F2 修复）：独立的逐臂危险 logit 头。开启后 hazard BCE
        # 训练这里（共享 backbone 做表征塑形），不再把 Beta 参数头的均值往
        # 标签上拽——那条旧路径让 BCE 与 RL 目标在同一组参数上打架，是熵
        # 坍缩（-0.4 -> -5.7）的直接推手。默认 False = 结构与旧检查点逐位
        # 兼容（module 不存在，state_dict 键集不变）。
        self.hazard_head = (nn.Linear(last, N_ALPHA) if hazard_head else None)
        if self.hazard_head is not None:
            nn.init.orthogonal_(self.hazard_head.weight, gain=0.01)
            nn.init.zeros_(self.hazard_head.bias)

    def distribution(self, obs: Tensor) -> Independent:
        if obs.ndim != 2 or obs.shape[-1] != self.obs_dim:
            raise ValueError(f"obs must have shape [N, {self.obs_dim}]")
        params = self.parameter_head(self.backbone(obs))
        c1_raw, c0_raw = params.chunk(2, dim=-1)
        c1 = F.softplus(c1_raw) + self.min_concentration
        c0 = F.softplus(c0_raw) + self.min_concentration
        return Independent(Beta(c1, c0, validate_args=False), 1)

    @staticmethod
    def split(raw: Tensor) -> tuple[Tensor, Tensor]:
        """(N,5) raw Beta sample -> (alpha (N,4), p (N,))."""
        return raw[:, :N_ALPHA], raw[:, N_ALPHA] * 2.0 - 1.0

    @staticmethod
    def to_env_action(raw: Tensor) -> Tensor:
        """Raw sample -> duo_env action layout (env re-derives alpha=(a+1)/2,
        p=a4, recovering our values exactly)."""
        alpha, p = BetaCoordinatorActor.split(raw)
        return torch.cat([alpha * 2.0 - 1.0, p.unsqueeze(-1)], dim=-1)

    def act(self, obs: Tensor, deterministic: bool = False) -> CoordinatorSample:
        dist = self.distribution(obs)
        raw = dist.mean if deterministic else dist.sample()
        # Beta samples can touch 0/1 in float32; keep strictly inside for
        # replayable log_prob (same epsilon discipline as double_hand ingress)
        raw = raw.clamp(1e-6, 1.0 - 1e-6)
        alpha, p = self.split(raw)
        return CoordinatorSample(raw=raw, alpha=alpha, p=p,
                                 log_prob=dist.log_prob(raw),
                                 entropy=dist.entropy())

    def hazard_logits(self, obs: Tensor) -> Tensor:
        """(N, obs_dim) -> (N, 4) 逐臂危险 logit（sigmoid 后 = P(0.5s 内危险)）。
        只在 hazard_head=True 的构造下可用；调用侧（lagrangian_ppo BCE 块）
        用 with_logits 版 BCE，数值上比先 sigmoid 再 clamp 稳定。"""
        if self.hazard_head is None:
            raise RuntimeError("actor was built without hazard_head")
        return self.hazard_head(self.backbone(obs))

    def evaluate_actions(self, obs: Tensor, raw: Tensor) -> tuple[Tensor, Tensor]:
        """Replay stored RAW actions -> (log_prob, entropy) for PPO ratios."""
        if raw.shape != (obs.shape[0], ACTION_DIM):
            raise ValueError(f"raw must have shape [N, {ACTION_DIM}]")
        if bool((raw <= 0.0).any().item()) or bool((raw >= 1.0).any().item()):
            raise ValueError("raw actions must lie strictly inside (0, 1)")
        dist = self.distribution(obs)
        return dist.log_prob(raw), dist.entropy()


class MultiHeadCritic(nn.Module):
    """Independent value MLPs per stream (double_hand CentralizedCritics,
    generalized from the fixed [reward, robot_cost, table_cost] triple)."""

    def __init__(self, obs_dim: int, heads: tuple = ("reward", "cross", "self", "table"),
                 hidden: tuple = (256, 256)):
        super().__init__()
        if len(heads) != len(set(heads)):
            raise ValueError("critic head names must be unique")
        self.head_names = tuple(heads)
        self.nets = nn.ModuleDict()
        for name in heads:
            body, last = _backbone(obs_dim, hidden)
            head = nn.Linear(last, 1)
            nn.init.orthogonal_(head.weight, gain=1.0)
            nn.init.zeros_(head.bias)
            self.nets[name] = nn.Sequential(body, head)

    def forward(self, obs: Tensor) -> dict:
        return {name: net(obs).squeeze(-1) for name, net in self.nets.items()}

    def stacked(self, obs: Tensor) -> Tensor:
        """(N, H) in head order -- cost columns feed validity_aware_gae."""
        vals = self.forward(obs)
        return torch.stack([vals[n] for n in self.head_names], dim=-1)
