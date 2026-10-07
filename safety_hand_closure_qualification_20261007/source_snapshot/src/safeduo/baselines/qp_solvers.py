"""Batched QP solvers for the projection-type safety QP.

Problem (per env, robot):
    min_u  1/2 ||u - c||^2  +  rho/2 ||s||^2
    s.t.   G u <= h + s,  s >= 0,  lo <= u <= hi

Eliminating s pointwise (s* = relu(G u - h)) gives the exactly equivalent
smooth program
    min_u  1/2 ||u - c||^2  +  rho/2 ||relu(G u - h)||^2   s.t. lo <= u <= hi,
a box-constrained convex piecewise-quadratic problem. We solve it with
projected FISTA to tolerance -- this IS the slacked QP optimum, not an
approximation (verified against proxsuite in tests/test_qp_solvers.py).

Solver menu:
  * solve_slack_qp_pgd  -- default; pure torch, batched, GPU-ready.
  * solve_slack_qp_qpth -- optional; qpth PDIPM on the lifted (u, s) QP,
    batched. Kept as a cross-check; PDIPM is slower and less robust at these
    sizes, see artifacts/baseline_matrix.md for the selection rationale.
  * proxsuite (tests only) -- per-env dense reference solutions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class QPResult:
    u: torch.Tensor            # (B, n)
    slack: torch.Tensor        # (B, M) relu(Gu - h) at the solution
    iters: int
    converged: bool


def _spectral_norm_sq(G: torch.Tensor, valid: torch.Tensor, iters: int = 12) -> torch.Tensor:
    """(B,) upper estimate of sigma_max(G_valid)^2 via power iteration."""
    Gm = G * valid.unsqueeze(-1)
    v = torch.ones(G.shape[0], G.shape[2], 1, device=G.device, dtype=G.dtype)
    for _ in range(iters):
        w = Gm @ v
        v = Gm.transpose(-1, -2) @ w
        n = v.norm(dim=-2, keepdim=True).clamp_min(1e-12)
        v = v / n
    w = Gm @ v
    return (w * w).sum(dim=(-2, -1)) * 1.05 + 1e-9  # 5% headroom


def solve_slack_qp_pgd(
    c: torch.Tensor,          # (B, n) desired solution (raw command)
    G: torch.Tensor,          # (B, M, n)
    h: torch.Tensor,          # (B, M)
    valid: torch.Tensor,      # (B, M) bool
    lo: torch.Tensor,         # (B, n) or broadcastable
    hi: torch.Tensor,
    rho: float = 1e3,
    max_iters: int = 300,
    tol: float = 1e-7,  # absolute step-size stop; 1e-7 is the fp32 floor
) -> QPResult:
    vm = valid.to(c.dtype)
    L = 1.0 + rho * _spectral_norm_sq(G, valid)      # (B,) Lipschitz bound
    step = (1.0 / L).unsqueeze(-1)
    u = c.clamp(min=lo, max=hi)
    y, t_acc = u.clone(), 1.0
    it, converged = 0, False
    for it in range(1, max_iters + 1):
        viol = ((G @ y.unsqueeze(-1)).squeeze(-1) - h).clamp_min(0.0) * vm
        grad = (y - c) + rho * (G.transpose(-1, -2) @ viol.unsqueeze(-1)).squeeze(-1)
        u_new = (y - step * grad).clamp(min=lo, max=hi)
        t_new = 0.5 * (1.0 + (1.0 + 4.0 * t_acc * t_acc) ** 0.5)
        y = u_new + ((t_acc - 1.0) / t_new) * (u_new - u)
        moved = (u_new - u).abs().max()
        u, t_acc = u_new, t_new
        if moved.item() < tol:
            converged = True
            break
    slack = ((G @ u.unsqueeze(-1)).squeeze(-1) - h).clamp_min(0.0) * vm
    return QPResult(u=u, slack=slack, iters=it, converged=converged)


def solve_slack_qp_qpth(c, G, h, valid, lo, hi, rho: float = 1e3) -> QPResult:
    """Lifted (u, s) QP via qpth PDIPM; requires qpth installed."""
    from qpth.qp import QPFunction

    B, M, n = G.shape
    dev, dt_ = c.device, c.dtype
    nz = n + M
    Q = torch.zeros(B, nz, nz, device=dev, dtype=dt_)
    Q[:, :n, :n] = torch.eye(n, device=dev, dtype=dt_)
    Q[:, n:, n:] = rho * torch.eye(M, device=dev, dtype=dt_)
    p = torch.cat([-c, torch.zeros(B, M, device=dev, dtype=dt_)], dim=-1)
    big = 1e8  # disable padded rows by pushing their bound far away
    h_eff = torch.where(valid, h, torch.full_like(h, big))
    G_eff = G * valid.unsqueeze(-1).to(dt_)
    # rows: [G u - s <= h; -s <= 0; u <= hi; -u <= -lo]
    I_m = torch.eye(M, device=dev, dtype=dt_).expand(B, M, M)
    I_n = torch.eye(n, device=dev, dtype=dt_).expand(B, n, n)
    Z_nm = torch.zeros(B, n, M, device=dev, dtype=dt_)
    A1 = torch.cat([G_eff, -I_m], dim=-1)
    A2 = torch.cat([torch.zeros(B, M, n, device=dev, dtype=dt_), -I_m], dim=-1)
    A3 = torch.cat([I_n, Z_nm], dim=-1)
    A4 = torch.cat([-I_n, Z_nm], dim=-1)
    Gq = torch.cat([A1, A2, A3, A4], dim=1)
    hq = torch.cat([h_eff, torch.zeros(B, M, device=dev, dtype=dt_),
                    hi.expand(B, n), -lo.expand(B, n)], dim=1)
    e = torch.empty(0, device=dev, dtype=dt_)
    z = QPFunction(verbose=-1, eps=1e-10, maxIter=50)(Q, p, Gq, hq, e, e)
    u = z[:, :n]
    slack = ((G @ u.unsqueeze(-1)).squeeze(-1) - h).clamp_min(0.0) * valid.to(dt_)
    return QPResult(u=u, slack=slack, iters=-1, converged=True)


def solve_reference_proxsuite(c, G, h, valid, lo, hi, rho: float = 1e3):
    """Per-env dense proxsuite reference on the lifted QP (tests only, CPU)."""
    import numpy as np
    import proxsuite

    B, M, n = G.shape
    out = torch.zeros_like(c)
    for b in range(B):
        m_v = int(valid[b].sum().item())
        idx = valid[b].nonzero(as_tuple=True)[0].cpu().numpy()
        nz = n + m_v
        H = np.eye(nz)
        H[n:, n:] *= rho
        g = np.concatenate([-c[b].cpu().numpy(), np.zeros(m_v)])
        # inequalities C z <= u_b : [G -I] z <= h ; -s <= 0 ; box on u
        C = np.zeros((m_v + m_v + 2 * n, nz))
        ub = np.zeros(m_v + m_v + 2 * n)
        Gb = G[b].cpu().numpy()[idx]
        C[:m_v, :n] = Gb
        C[:m_v, n:] = -np.eye(m_v)
        ub[:m_v] = h[b].cpu().numpy()[idx]
        C[m_v:2 * m_v, n:] = -np.eye(m_v)
        ub[m_v:2 * m_v] = 0.0
        C[2 * m_v:2 * m_v + n, :n] = np.eye(n)
        ub[2 * m_v:2 * m_v + n] = hi.expand_as(c)[b].cpu().numpy()
        C[2 * m_v + n:, :n] = -np.eye(n)
        ub[2 * m_v + n:] = -lo.expand_as(c)[b].cpu().numpy()
        qp = proxsuite.proxqp.dense.QP(nz, 0, C.shape[0])
        qp.init(H, g, None, None, C, np.full(C.shape[0], -1e20), ub)
        qp.settings.eps_abs = 1e-10
        qp.solve()
        out[b] = torch.as_tensor(qp.results.x[:n], dtype=c.dtype)
    return out
