"""Speed-scaling baseline (Chong 2002 style): per-arm linear slowdown with margin.

alpha = clip((margin - d_min) / (d_slow - d_min), floor, 1) applied to the raw
command. Smoother than E-stop, still purely reactive to current distance:
no prediction, no yielding decision, so both arms slow symmetrically and
tight-coupling tasks crawl.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from safeduo.baselines.base import (
    BaselineFilter,
    FilterOutput,
    GeometryProvider,
    arm_min_margin,
    effective_alpha,
)
from safeduo.delta._contract_stub import ARM_KEYS, DeltaCmd, SceneState


@dataclass
class SpeedScalingConfig:
    d_min: float = 0.03      # margin at which alpha hits the floor (m)
    d_slow: float = 0.15     # margin at which slowdown starts (m)
    alpha_floor: float = 0.0
    ema_hz: float = 0.0      # >0 low-passes alpha to soften chatter


class SpeedScalingFilter(BaselineFilter):
    def __init__(self, n_envs: int, provider: GeometryProvider,
                 cfg: "SpeedScalingConfig | None" = None,
                 device: "str | torch.device" = "cpu"):
        self.n = n_envs
        self.provider = provider
        self.cfg = cfg or SpeedScalingConfig()
        self.device = torch.device(device)
        self._alpha_ema = torch.ones(n_envs, 4, device=self.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self._alpha_ema[env_ids.to(self.device)] = 1.0

    def filter(self, state: SceneState, cmd: DeltaCmd) -> FilterOutput:
        import math

        cfg = self.cfg
        rows = self.provider.rows(state)
        margin = arm_min_margin(rows)
        alpha = ((margin - cfg.d_min) / max(cfg.d_slow - cfg.d_min, 1e-9)).clamp(
            cfg.alpha_floor, 1.0)
        if cfg.ema_hz > 0.0:
            k = 1.0 - math.exp(-2.0 * math.pi * cfg.ema_hz * state.dt)
            self._alpha_ema = self._alpha_ema + k * (alpha - self._alpha_ema)
            alpha = self._alpha_ema
        exec_ = {a: cmd.delta_q[a] * alpha[:, i].unsqueeze(-1)
                 for i, a in enumerate(ARM_KEYS)}
        return FilterOutput(
            delta_exec=exec_,
            alpha=effective_alpha(cmd.delta_q, exec_),
            priority_p=torch.zeros(self.n, device=self.device),
            backstop_active=alpha < 0.999,
            info={"margin": margin, "alpha_raw": alpha},
        )
