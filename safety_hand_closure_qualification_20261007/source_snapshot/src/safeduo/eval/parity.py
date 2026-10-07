"""G0 consistency parity: A's training-time sequential projection vs the
deployment-grade exact QP, on hard-conflict states (tail-quantile report).

Why this exists (round-2 delta #3 for A, G0 gate): the training L2
(VelocityDamperBackstop, POCS-style sequential projection) converges to *a*
feasible point of the constraint polyhedron, not necessarily the closest one;
the deployment L2 re-solves the projection exactly (QP). The paper's safety
claim shares L2 across train/deploy, so the behavioral gap must be measured,
not assumed: we report rewrite-difference percentiles (P50/P95/P99) between
the two on states harvested from real conflicts.

Both sides consume the IDENTICAL constraint set:
  - safety rows from a GeometryProvider (same -J, same cap with per-row d_min,
    same (1+p)/2:(1-p)/2 cross-row budget),
  - alpha progress rows  <u_a, dir_a> <= alpha_a * ||c_a||  per arm,
  - the vmax box.
The QP side solves min ||u - c||^2 s.t. all rows, via the slacked exact solver
with a stiff slack penalty (documented; slack ~ 0 on feasible instances).

Usage:
    .venv/bin/python src/safeduo/eval/parity.py          # full local run
    from safeduo.eval.parity import run_parity           # programmatic
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from safeduo.baselines.base import GeometryProvider, stack_robot
from safeduo.baselines.qp_solvers import solve_slack_qp_pgd
from safeduo.delta._contract_stub import (
    ARM_KEYS,
    ARMS_OF_ROBOT,
    CLASS_CROSS,
    DOF_OF,
    DeltaCmd,
)
from safeduo.safety.backstop import BackstopConfig, VelocityDamperBackstop

_ARM_IDX = {a: i for i, a in enumerate(ARM_KEYS)}


@dataclass
class ParityConfig:
    gamma: float = 4.0
    vmax: float = 1.5
    dt: float = 0.02
    d_soft: float = 0.05          # hard-conflict harvest threshold
    slack_rho: float = 1e6        # stiff -> near-hard constraints on the QP side
    qp_iters: int = 3000
    qp_tol: float = 1e-9
    a_tol: float = 1e-5           # A-side residual considered "converged"


def _alpha_rows(c: torch.Tensor, alpha: torch.Tensor, robot: str):
    """A's progress-budget half-spaces, replicated 1:1 (see backstop._alpha_rows)."""
    G, h, rel = [], [], []
    i = 0
    for a in ARMS_OF_ROBOT[robot]:
        d = DOF_OF[a]
        ca = c[:, i:i + d]
        norm = ca.norm(dim=-1)
        dir_full = torch.zeros_like(c)
        dir_full[:, i:i + d] = ca / norm.clamp_min(1e-9).unsqueeze(-1)
        G.append(dir_full)
        h.append(alpha[:, _ARM_IDX[a]] * norm)
        rel.append(norm > 1e-9)
        i += d
    return torch.stack(G, dim=1), torch.stack(h, dim=1), torch.stack(rel, dim=1)


def qp_reference_exec(rows, cmd: DeltaCmd, alpha: torch.Tensor, p: torch.Tensor,
                      cfg: ParityConfig) -> dict:
    """Deployment-grade exact projection onto A's constraint set, per robot."""
    dm = rows.d_min if rows.d_min is not None else torch.full_like(rows.d, 0.03)
    cap = cfg.gamma * (rows.d - dm) * cfg.dt
    is_cross = rows.cls == CLASS_CROSS
    box = cfg.vmax * cfg.dt
    out = {}
    for r in ("F", "U"):
        budget = (1.0 + p) * 0.5 if r == "F" else (1.0 - p) * 0.5
        c = stack_robot(cmd.delta_q, r).clamp(-box, box)
        involves = rows.arm_mask[..., [_ARM_IDX[a] for a in ARMS_OF_ROBOT[r]]].any(-1)
        rel = rows.valid & involves
        G = -rows.J[r]
        h = torch.where(is_cross, budget.unsqueeze(-1) * cap, cap)
        aG, ah, arel = _alpha_rows(c, alpha, r)
        G_all = torch.cat([G, aG], dim=1)
        h_all = torch.cat([h, ah], dim=1)
        rel_all = torch.cat([rel, arel], dim=1)
        lo = torch.full_like(c, -box)
        hi = torch.full_like(c, box)
        res = solve_slack_qp_pgd(c, G_all, h_all, rel_all, lo, hi,
                                 rho=cfg.slack_rho, max_iters=cfg.qp_iters,
                                 tol=cfg.qp_tol)
        out[r] = res
    return out


def collect_hard_conflict_states(provider, n_envs: int, n_states: int,
                                 seed: int = 0, cfg: "ParityConfig | None" = None,
                                 max_steps: int = 400) -> list:
    """Ram rollout on the provider; harvest (rows, cmd, alpha, p) snapshots
    whenever any env sits inside the hard-conflict band (cross margin < d_soft)."""
    from safeduo.baselines.strong_cbf_qp import StrongCBFQPConfig, StrongCBFQPFilter
    from safeduo.delta.l1_random import JacobianMapper

    cfg = cfg or ParityConfig()
    g = torch.Generator().manual_seed(seed)
    opp = {"F_L": "U_R", "F_R": "U_L", "U_L": "F_R", "U_R": "F_L"}
    mapper = JacobianMapper(provider.ee_jacobian, (0, 0, 0), (1, 1, 1))
    # a safety filter keeps the rollout *in* the conflict band instead of
    # blasting through it (unfiltered arms interpenetrate and leave the band)
    filt = StrongCBFQPFilter(n_envs, provider,
                             StrongCBFQPConfig(d_min=0.03, gamma=cfg.gamma))
    filt.reset(torch.arange(n_envs))
    q = provider.default_q(jitter=0.15, generator=g)
    qd = {a: torch.zeros_like(q[a]) for a in ARM_KEYS}
    states = []
    for _ in range(max_steps):
        state = provider.scene_state(q, qd, dt=cfg.dt)
        cmd = DeltaCmd(delta_q={
            a: mapper.map(a, state, state.ee_pos[opp[a]], cfg.dt,
                          speed=0.4 + 0.4 * torch.rand(n_envs, generator=g))
            for a in ARM_KEYS})
        mm = provider.min_margin_by_class(q)
        in_band = mm["cross"] < cfg.d_soft
        if in_band.any():
            rows = provider.rows_from_q(q)
            alpha = 0.3 + 0.7 * torch.rand(n_envs, 4, generator=g)
            p = torch.rand(n_envs, generator=g) * 2.0 - 1.0
            states.append({"rows": rows, "cmd": cmd, "alpha": alpha, "p": p,
                           "in_band": in_band})
            if sum(s["in_band"].sum().item() for s in states) >= n_states:
                break
        out = filt.filter(state, cmd)
        q = {a: q[a] + out.delta_exec[a] for a in ARM_KEYS}
        qd = {a: out.delta_exec[a] / cfg.dt for a in ARM_KEYS}
    return states


def run_parity(provider=None, n_envs: int = 8, n_states: int = 256,
               seed: int = 0, cfg: "ParityConfig | None" = None,
               out_dir: "Path | str | None" = None) -> dict:
    """Full parity run -> report dict (+ JSON/markdown when out_dir given)."""
    cfg = cfg or ParityConfig()
    if provider is None:
        from safeduo.baselines.real_geometry import RealGeometryProvider

        provider = RealGeometryProvider(n_envs)
    backstop = VelocityDamperBackstop(BackstopConfig(
        gamma=cfg.gamma, vmax=cfg.vmax))
    states = collect_hard_conflict_states(provider, n_envs, n_states, seed, cfg)
    diffs, resid_a, slack_qp, viol_a, viol_qp = [], [], [], [], []
    for s in states:
        rows, cmd, alpha, p = s["rows"], s["cmd"], s["alpha"], s["p"]
        dmin = rows.d_min if rows.d_min is not None else None
        exec_a, _, info_a = backstop.project(cmd, rows, alpha, p, cfg.dt, dmin=dmin)
        ref = qp_reference_exec(rows, cmd, alpha, p, cfg)
        u_a = torch.cat([stack_robot(exec_a.delta_q, "F"),
                         stack_robot(exec_a.delta_q, "U")], dim=-1)
        u_q = torch.cat([ref["F"].u, ref["U"].u], dim=-1)
        band = s["in_band"]
        diffs.append((u_a - u_q).abs().amax(dim=-1)[band])
        resid_a.append(torch.maximum(info_a["residual_F"],
                                     info_a["residual_U"])[band])
        slack_qp.append((ref["F"].slack.sum(-1) + ref["U"].slack.sum(-1))[band])
    diffs = torch.cat(diffs) if diffs else torch.zeros(0)
    resid_a = torch.cat(resid_a) if len(resid_a) else torch.zeros(0)
    slack_qp = torch.cat(slack_qp) if len(slack_qp) else torch.zeros(0)
    conv = resid_a < cfg.a_tol

    def _pct(x, q):
        return float(x.double().quantile(q).item()) if x.numel() else float("nan")

    report = {
        "n_states": int(diffs.numel()),
        "n_a_converged": int(conv.sum().item()),
        "rewrite_diff_p50": _pct(diffs, 0.50),
        "rewrite_diff_p95": _pct(diffs, 0.95),
        "rewrite_diff_p99": _pct(diffs, 0.99),
        "rewrite_diff_max": float(diffs.max().item()) if diffs.numel() else float("nan"),
        "rewrite_diff_p99_converged_only": _pct(diffs[conv], 0.99),
        "a_residual_p99": _pct(resid_a, 0.99),
        "qp_slack_p99": _pct(slack_qp, 0.99),
        "units": "rad per control step (max over 26 joints)",
        "config": {k: getattr(cfg, k) for k in
                   ("gamma", "vmax", "dt", "d_soft", "slack_rho", "a_tol")},
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "parity_report.json").write_text(json.dumps(report, indent=2))
        lines = ["# L2 train-projection vs deploy-QP parity (G0)", "",
                 "| metric | value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in report.items() if k != "config"]
        (out / "parity_report.md").write_text("\n".join(lines) + "\n")
    return report


if __name__ == "__main__":
    import sys

    _repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_repo / "src"))
    rep = run_parity(n_states=512, out_dir=_repo / "artifacts" / "parity")
    print(json.dumps(rep, indent=2))
