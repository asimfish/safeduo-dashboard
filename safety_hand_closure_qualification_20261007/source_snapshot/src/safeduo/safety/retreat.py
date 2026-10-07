"""R30 retreat pass-through (pure torch, Isaac-free).

A clutch-locked arm (alpha ~ 0) whose OWN commanded joint motion is non-closing
on every retained constraint row within the warn band is released for the
current step. Motivation: hysteresis clutches freeze an arm in a static
near-hazard pose where the policy's alpha never rises again (tray_relay F_L,
2026-09-03); the operator's retreat (e.g. the lift) is itself the safe move and
must not be held. Manual-recovery mode already assumes a 1-step unlock on
retreat -- this makes the hysteresis path behave the same.
"""
from __future__ import annotations

import torch

from safeduo.baselines.base import ConstraintRows, stack_robot
from safeduo.safety.types import ARM_KEYS, ARMS_OF_ROBOT, DeltaCmd


def retreat_release(alpha: torch.Tensor, cmd: DeltaCmd, rows: ConstraintRows,
                    viol_exempt: "torch.Tensor | None", d_warn: float,
                    eps: float = 1e-7,
                    lock_steps: "torch.Tensor | None" = None,
                    dwell: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (alpha_exec (N,4), released (N,4) bool).

    released[n,k] = locked (alpha<1e-6) and no retained row within d_warn that
    involves arm k has the arm's own motion closing it (J_k . u_k < -eps).
    Rows the cost side exempts (legal near-table parking) are ignored.
    R30b dwell: with lock_steps (N,4) = consecutive steps the arm has been
    locked, release additionally requires lock_steps >= dwell, so the first
    dwell steps of a brake are never softened (transient stop response intact);
    dwell <= 0 or lock_steps None = R30 behaviour."""
    band = rows.valid & (rows.d < d_warn)
    if viol_exempt is not None:
        band = band & ~viol_exempt
    release = torch.zeros_like(alpha, dtype=torch.bool)
    for robot in ("F", "U"):
        a1, a2 = ARMS_OF_ROBOT[robot]
        J = rows.J[robot]                                       # (N, M, dof)
        for arm in (a1, a2):
            u = {a1: torch.zeros_like(cmd.delta_q[a1]),
                 a2: torch.zeros_like(cmd.delta_q[a2])}
            u[arm] = cmd.delta_q[arm]
            ddot = torch.einsum("nmd,nd->nm", J, stack_robot(u, robot))
            k = ARM_KEYS.index(arm)
            closing = band & rows.arm_mask[..., k] & (ddot < -eps)
            release[:, k] = ~closing.any(dim=-1)
    released = (alpha < 1e-6) & release
    if dwell > 0 and lock_steps is not None:
        released = released & (lock_steps >= dwell)
    return torch.where(released, torch.ones_like(alpha), alpha), released
