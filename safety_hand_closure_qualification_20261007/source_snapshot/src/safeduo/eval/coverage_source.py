"""State-feedback source for pair-stratified safety coverage windows."""

from __future__ import annotations

import torch

from safeduo.delta._contract_stub import ARM_KEYS, DeltaCmd, DeltaSource, SceneState
from safeduo.delta.l2_env_source import CachedEEMapper, EnvEEBackend, RealScenePoses
from safeduo.delta.l1_workspace_v7 import EnvEEBackendV7
from safeduo.eval.coverage_design import ARM_PAIRS, CoverageCell, build_schedule


def _gather_arms(values: dict[str, torch.Tensor], index: torch.Tensor) -> torch.Tensor:
    """Gather one batched arm tensor per environment from a six-pair index."""
    return torch.stack([values[arm] for arm in ARM_KEYS], dim=1)[
        torch.arange(index.shape[0], device=index.device), index]


class PairStratifiedDelta(DeltaSource):
    """Drive exactly one scheduled arm pair through a distance/direction cell.

    The source is intentionally a pressure generator, not an avoidance policy.
    It uses the same analytic EE mapper as the existing L2 source and leaves
    the coordinator/backstop responsible for safety.  Same-row pairs are
    supported explicitly, so exposure is not left to chance.
    """

    def __init__(self, n_envs: int, amp_max: float, seed: int,
                 env_yaml: str = "duo_env_a31.yaml",
                 device: str | torch.device = "cpu",
                 cells: list[CoverageCell] | None = None):
        self.n = int(n_envs)
        self.device = torch.device(device)
        self.amp_max = float(amp_max)
        self.seed = int(seed)
        self.cells = cells or build_schedule(self.n, self.amp_max, self.seed)
        if len(self.cells) != self.n:
            raise ValueError("one coverage cell is required per environment")
        self.pair_index = torch.tensor([c.pair_index for c in self.cells],
                                       dtype=torch.long, device=self.device)
        self.target_gap = torch.tensor([c.target_gap_m for c in self.cells],
                                       dtype=torch.float32, device=self.device)
        self.direction = tuple(c.direction for c in self.cells)
        self._direction_code = torch.tensor(
            [{"approach": -1.0, "recede": 1.0, "tangent": 0.0}[c.direction]
             for c in self.cells], dtype=torch.float32, device=self.device)
        self.backend: EnvEEBackend = EnvEEBackendV7(
            RealScenePoses(env_yaml), device=self.device)
        self.mapper = CachedEEMapper(self.backend, ee_speed=0.35)
        self._origin = {a: torch.full((self.n, 3), float("nan"), device=self.device)
                        for a in ARM_KEYS}
        self._t = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        self._token = 0
        self.gen: torch.Generator | None = None
        self.last_targets: dict[str, torch.Tensor] = {}

    def coverage_metadata(self) -> list[dict]:
        return [cell.as_dict() for cell in self.cells]

    def reset(self, env_ids: torch.Tensor,
              generator: torch.Generator | None = None) -> None:
        if generator is not None:
            self.gen = generator
        ids = env_ids.to(self.device)
        for arm in ARM_KEYS:
            self._origin[arm][ids] = float("nan")
        self._t[ids] = 0.0

    def _capture_origin(self, state: SceneState) -> None:
        for arm in ARM_KEYS:
            stale = torch.isnan(self._origin[arm]).any(dim=-1)
            self._origin[arm] = torch.where(
                stale.unsqueeze(-1), state.ee_pos[arm], self._origin[arm])

    def _pair_geometry(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos = self._origin
        left = {i: ARM_PAIRS[i][0] for i in range(len(ARM_PAIRS))}
        right = {i: ARM_PAIRS[i][1] for i in range(len(ARM_PAIRS))}
        pa = torch.stack([pos[ARM_KEYS[left[i]]] for i in range(len(ARM_PAIRS))], dim=1)
        pb = torch.stack([pos[ARM_KEYS[right[i]]] for i in range(len(ARM_PAIRS))], dim=1)
        rows = torch.arange(self.n, device=self.device)
        pa = pa[rows, self.pair_index]
        pb = pb[rows, self.pair_index]
        vec = pa - pb
        norm = vec.norm(dim=-1, keepdim=True)
        fallback = torch.zeros_like(vec)
        fallback[:, 0] = 1.0
        normal = torch.where(norm > 1e-6, vec / norm.clamp_min(1e-6), fallback)
        midpoint = (pa + pb) * 0.5
        return midpoint, normal, norm.squeeze(-1)

    def sample(self, state: SceneState) -> DeltaCmd:
        self._token += 1
        self.backend.refresh(state, self._token)
        self._capture_origin(state)
        midpoint, normal, initial_gap = self._pair_geometry()
        t = self._t
        # Four seconds is long enough to observe approach/recede trends while
        # keeping a 10 s window capable of repeating the pressure pass.
        progress = 1.0 - torch.exp(-t / 4.0)
        sign = self._direction_code
        gap = torch.where(
            sign < 0,
            self.target_gap + (torch.maximum(initial_gap, self.target_gap + 0.12) - self.target_gap) * (1.0 - progress),
            torch.where(sign > 0,
                        self.target_gap + 0.12 * progress,
                        self.target_gap),
        )
        # Tangent cells translate the midpoint along a stable perpendicular
        # direction while keeping the requested pair separation.
        tangent = torch.stack((-normal[:, 1], normal[:, 0], torch.zeros_like(normal[:, 0])), dim=-1)
        tangent_scale = 0.06 * torch.sin(t * 1.2)
        midpoint = midpoint + tangent * tangent_scale.unsqueeze(-1) * (sign == 0).float().unsqueeze(-1)
        target_a = midpoint + normal * (gap * 0.5).unsqueeze(-1)
        target_b = midpoint - normal * (gap * 0.5).unsqueeze(-1)
        out = {arm: torch.zeros_like(state.q[arm]) for arm in ARM_KEYS}
        for pair_idx, (ai, bi) in enumerate(ARM_PAIRS):
            mask = self.pair_index == pair_idx
            if not bool(mask.any()):
                continue
            a, b = ARM_KEYS[ai], ARM_KEYS[bi]
            da = self.mapper.map(a, state, torch.where(mask.unsqueeze(-1), target_a, state.ee_pos[a]),
                                 state.dt, speed=0.35)
            db = self.mapper.map(b, state, torch.where(mask.unsqueeze(-1), target_b, state.ee_pos[b]),
                                 state.dt, speed=0.35)
            out[a] = torch.where(mask.unsqueeze(-1), da, out[a])
            out[b] = torch.where(mask.unsqueeze(-1), db, out[b])
        self.last_targets = {"a": target_a.detach(), "b": target_b.detach(),
                             "initial_gap": initial_gap.detach(), "target_gap": gap.detach()}
        self._t = self._t + state.dt
        return DeltaCmd(delta_q={a: d.clamp(-self.amp_max, self.amp_max)
                                 for a, d in out.items()})
