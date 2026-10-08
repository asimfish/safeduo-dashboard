"""Shared baseline filter interface and constraint-geometry contract.

All baselines (E-stop, speed scaling, strengthened CBF-QP) implement
BaselineFilter so the eval runner treats them interchangeably with SafeDuo's
L1+L2 stack. They consume constraint geometry through GeometryProvider -- the
same rows Agent A's L2 backstop consumes -- so the comparison is apples to
apples (alignment point with A, logged in STATUS_C.md).

Row convention (one row per active sphere pair):
  d      : (N, M) center distance minus radii (margin, meters)
  J      : dict robot ("F"/"U") -> (N, M, dof_R) with dof_R = stacked arm dofs
           (F = F_L + F_R, U = U_L + U_R); d_dot = sum_R J_R @ qd_R (m/s).
           Rows not involving robot R carry zeros in J[R].
  cls    : (N, M) class id (CLASS_CROSS=0 / CLASS_SELF=1 / CLASS_TABLE=2)
  arm_mask: (N, M, 4) bool, which of ARM_KEYS the row involves
  valid  : (N, M) bool (padding mask)
  d_min  : optional (N, M) per-row braking boundary (A's semantics-driven
           pair_dmin / active_dmin); None -> consumer falls back to its config
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch

from safeduo.delta._contract_stub import ARM_KEYS, ARMS_OF_ROBOT, DeltaCmd, SceneState


@dataclass
class ConstraintRows:
    d: torch.Tensor
    J: dict
    cls: torch.Tensor
    arm_mask: torch.Tensor
    valid: torch.Tensor
    d_min: "torch.Tensor | None" = None


class GeometryProvider(ABC):
    """Produces constraint rows from a SceneState. Toy tests use the analytic
    planar-arm provider; in Isaac, Agent A backs this with GPU sphere data."""

    @abstractmethod
    def rows(self, state: SceneState) -> ConstraintRows:
        ...


@dataclass
class FilterOutput:
    delta_exec: dict                    # arm -> (N, dof)
    alpha: torch.Tensor                 # (N, 4) effective per-arm passthrough in [0, 1]
    priority_p: torch.Tensor            # (N,) in [-1, 1]; +1 = U yields to F
    backstop_active: torch.Tensor       # (N, 4) bool, command was modified for this arm
    info: dict = field(default_factory=dict)


class BaselineFilter(ABC):
    @abstractmethod
    def reset(self, env_ids: torch.Tensor) -> None:
        ...

    @abstractmethod
    def filter(self, state: SceneState, cmd: DeltaCmd) -> FilterOutput:
        ...


def active_pair_set(d: torch.Tensor, closing: torch.Tensor, cls: torch.Tensor,
                    pair_id: torch.Tensor, max_active: int, d_soft: float = 0.05,
                    tau_ttc: float = 0.5) -> tuple:
    """Contract observation extraction, mirroring A's SphereDistanceModule v2:
    rows eligible when margin < d_soft or 0 < TTC < tau_ttc; margin-ascending
    cap at max_active; zero-filled features with pair_id = -1 on padding.

    d/closing/cls/pair_id: (N, P). Returns (active_pairs (N, M, 4), mask (N, M))."""
    ttc = torch.where(closing > 1e-6, d / closing.clamp_min(1e-6),
                      torch.full_like(d, torch.inf))
    eligible = (d < d_soft) | ((ttc > 0) & (ttc < tau_ttc))
    score = torch.where(eligible, d, torch.full_like(d, torch.inf))
    m = min(max_active, d.shape[1])
    top_v, top_i = torch.topk(score, m, dim=1, largest=False)
    mask = torch.isfinite(top_v)
    feat = torch.stack([
        d.gather(1, top_i), closing.gather(1, top_i),
        cls.gather(1, top_i), pair_id.gather(1, top_i),
    ], dim=-1)
    feat = torch.where(mask.unsqueeze(-1), feat, torch.zeros_like(feat))
    feat[..., 3] = torch.where(mask, feat[..., 3], torch.full_like(feat[..., 3], -1.0))
    return feat, mask


def arm_min_margin(rows: ConstraintRows, big: float = 1e6) -> torch.Tensor:
    """(N, 4) min margin over valid rows involving each arm; `big` where none."""
    d = rows.d.unsqueeze(-1).expand(-1, -1, 4)
    involved = rows.arm_mask & rows.valid.unsqueeze(-1)
    d = torch.where(involved, d, torch.full_like(d, big))
    return d.min(dim=1).values


def effective_alpha(cmd: dict, exec_: dict) -> torch.Tensor:
    """(N, 4) measured passthrough <exec, cmd> / ||cmd||^2, clipped to [0, 1].

    Equals the applied gate for pure scaling filters; for the QP it reports the
    command-direction component that survived (used by the intervention metric)."""
    cols = []
    for a in ARM_KEYS:
        num = (exec_[a] * cmd[a]).sum(dim=-1)
        den = (cmd[a] * cmd[a]).sum(dim=-1).clamp_min(1e-12)
        cols.append((num / den).clamp(0.0, 1.0))
    return torch.stack(cols, dim=-1)


def stack_robot(delta_q: dict, robot: str) -> torch.Tensor:
    """Stack a robot's two arm tensors: F -> (N, dof_FL+dof_FR), U likewise."""
    a1, a2 = ARMS_OF_ROBOT[robot]
    return torch.cat([delta_q[a1], delta_q[a2]], dim=-1)


def unstack_robot(u: torch.Tensor, robot: str, dof_of: dict) -> dict:
    a1, a2 = ARMS_OF_ROBOT[robot]
    n1 = dof_of[a1]
    return {a1: u[..., :n1], a2: u[..., n1:]}
