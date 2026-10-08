"""纯随机噪声 DeltaSource——Agent A 的冒烟实现，只用于管线/吞吐自测。

真正的 L1(OU+waypoint)/L2(脚本冲突)/L3(对抗)/L4(回放) 由 Agent C 实现。
"""

from __future__ import annotations

import torch

from safeduo.safety.types import ARM_KEYS, DOF_OF, DeltaCmd, DeltaSource, SceneState


class SmokeNoiseDelta(DeltaSource):
    """每步均匀噪声 delta in [-amp, amp]，可复现（跟随传入 generator）。"""

    def __init__(self, n_envs: int, amp: float = 0.05, device: torch.device | str = "cpu"):
        self.n = n_envs
        self.amp = amp
        self.device = torch.device(device)
        self.gen: torch.Generator | None = None

    def reset(self, env_ids: torch.Tensor, generator: torch.Generator | None = None) -> None:
        if generator is not None:
            self.gen = generator  # 无内部状态，只接管随机源

    def sample(self, state: SceneState) -> DeltaCmd:
        out = {}
        for arm in ARM_KEYS:
            u = torch.rand(self.n, DOF_OF[arm], device=self.device, generator=self.gen)
            out[arm] = (u * 2.0 - 1.0) * self.amp
        return DeltaCmd(delta_q=out)
