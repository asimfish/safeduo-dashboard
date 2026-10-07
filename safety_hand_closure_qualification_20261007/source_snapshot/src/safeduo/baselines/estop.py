"""E-stop baseline: per-arm hard stop below a margin threshold, hysteretic resume.

The classic teleop safety rule and the weakest informative baseline: it is
safe by construction (an arm that stops cannot close distance) but pays with
full-stop interventions, oscillation at the threshold (mitigated by the
hysteresis band), and zero notion of who should yield.
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
class EStopConfig:
    d_stop: float = 0.05     # stop when arm min margin drops below (m)
    d_resume: float = 0.08   # resume only above this margin (hysteresis)


class EStopFilter(BaselineFilter):
    def __init__(self, n_envs: int, provider: GeometryProvider,
                 cfg: "EStopConfig | None" = None,
                 device: "str | torch.device" = "cpu"):
        self.n = n_envs
        self.provider = provider
        self.cfg = cfg or EStopConfig()
        self.device = torch.device(device)
        self.stopped = torch.zeros(n_envs, 4, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.stopped[env_ids.to(self.device)] = False

    def filter(self, state: SceneState, cmd: DeltaCmd) -> FilterOutput:
        rows = self.provider.rows(state)
        margin = arm_min_margin(rows)                      # (N, 4)
        self.stopped = torch.where(margin < self.cfg.d_stop,
                                   torch.ones_like(self.stopped), self.stopped)
        self.stopped = torch.where(margin > self.cfg.d_resume,
                                   torch.zeros_like(self.stopped), self.stopped)
        alpha = (~self.stopped).float()
        exec_ = {a: cmd.delta_q[a] * alpha[:, i].unsqueeze(-1)
                 for i, a in enumerate(ARM_KEYS)}
        return FilterOutput(
            delta_exec=exec_,
            alpha=effective_alpha(cmd.delta_q, exec_),
            priority_p=torch.zeros(self.n, device=self.device),
            backstop_active=self.stopped.clone(),
            info={"margin": margin},
        )
