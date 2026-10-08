"""Strengthened CBF-QP baseline (the one SafeDuo must beat honestly).

Per active sphere pair, a velocity-damper linear constraint
    d_dot >= -gamma * (d - d_min)
is enforced on the commanded joint deltas. Strengthening over a vanilla
CBF-QP (each item logged with rationale in artifacts/baseline_matrix.md):

  1. Intent extrapolation: the opponent robot is a *moving* obstacle -- its
     next delta is linearly extrapolated from the last H executed deltas and
     enters the constraint constant term (same H=5 history SafeDuo's policy
     sees, so information access is fair).
  2. Slack variables: the QP is always feasible; slack is penalized (rho) so
     it only absorbs genuinely infeasible instants instead of hiding sloppy
     tuning.
  3. Fixed-rule priority with hysteresis: a bilateral budget split
     (1+p)/2 : (1-p)/2 on cross-robot rows, p in [-1, +1], +1 = U yields to F.
     The less-aggressive robot (smaller commanded closing usage) gets right of
     way; deadband + dwell + rate limit prevent flip-flopping.
  4. Delay compensation: margins are advanced by d_dot * tau_delay before
     constraint construction, so the filter brakes for where the pair will be
     when the command lands, not where it was.

Solved per robot (F: QP over its 14 stacked dofs, U: 12), batched over envs in
pure torch (projected FISTA on the exact slacked QP; optional qpth cross-check).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from safeduo.baselines.base import (
    BaselineFilter,
    ConstraintRows,
    FilterOutput,
    GeometryProvider,
    effective_alpha,
    stack_robot,
    unstack_robot,
)
from safeduo.baselines.qp_solvers import solve_slack_qp_pgd
from safeduo.delta._contract_stub import (
    ARM_KEYS,
    ARMS_OF_ROBOT,
    CLASS_CROSS,
    DeltaCmd,
    SceneState,
)

_BIG = 1e6


@dataclass
class StrongCBFQPConfig:
    d_min: float = 0.03        # hard margin floor (m)
    # Constraint activation caliber (STATUS.md Round-17 arbitration): the
    # default now matches A's active-pair eligibility -- a row enters the QP
    # iff margin < d_soft OR time-to-contact < tau_ttc -- so every method
    # activates L2 constraints under the same rule. d_act (the W1 fixed
    # 0.25 m distance gate) stays as a legacy/explicit override: set a float
    # to reproduce the old behavior (used by safety/consistency_parity.py).
    d_act: "float | None" = None
    d_soft: float = 0.05       # active-set distance gate (contact_semantics)
    tau_ttc: float = 0.5       # active-set TTC gate (s), duo_env safety.tau_ttc
    gamma: float = 4.0         # damper gain (1/s): max closing speed = gamma*(d-d_min)
    slack_rho: float = 1e4     # slack penalty
    vmax: float = 1.5          # per-joint speed bound (rad/s), boxes the QP
    horizon_h: int = 5         # opponent delta history for extrapolation
    extrap: str = "linear"     # linear | hold | zero (zero = vanilla CBF-QP)
    borrow_frac: float = 0.5   # fraction of predicted opponent retreat we may
    # spend as extra closing budget; caps exposure to prediction error
    tau_delay: float = 0.0     # delay compensation (s); set to measured latency
    prio_deadband: float = 2e-4  # aggression-score deadband (m/step)
    prio_dwell_steps: int = 10   # min steps between priority flips
    prio_rate: float = 0.2       # max |dp| per step (smooth budget shifts)
    prio_tiebreak: float = 1.0   # symmetric-conflict tiebreak (+1 = U yields);
    # fixed arbitrary convention, breaks the symmetric deadlock deterministically
    solver_iters: int = 300
    solver_tol: float = 1e-7
    debug_qp: bool = False     # stash per-robot QP matrices in info (tests)


class StrongCBFQPFilter(BaselineFilter):
    def __init__(self, n_envs: int, provider: GeometryProvider,
                 cfg: "StrongCBFQPConfig | None" = None,
                 device: "str | torch.device" = "cpu",
                 dof_of: "dict | None" = None):
        from safeduo.delta._contract_stub import DOF_OF

        self.n = n_envs
        self.provider = provider
        self.cfg = cfg or StrongCBFQPConfig()
        self.device = torch.device(device)
        self.dof_of = dof_of or DOF_OF
        self.dof_r = {r: sum(self.dof_of[a] for a in ARMS_OF_ROBOT[r]) for r in ("F", "U")}
        self._hist = {r: torch.zeros(n_envs, self.cfg.horizon_h, self.dof_r[r],
                                     device=self.device) for r in ("F", "U")}
        self._hist_len = {r: torch.zeros(n_envs, dtype=torch.long, device=self.device)
                          for r in ("F", "U")}
        self.p = torch.zeros(n_envs, device=self.device)
        # start with dwell satisfied so the first engagement is not delayed
        self._dwell = torch.full((n_envs,), self.cfg.prio_dwell_steps,
                                 dtype=torch.long, device=self.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        ids = env_ids.to(self.device)
        for r in ("F", "U"):
            self._hist[r][ids] = 0.0
            self._hist_len[r][ids] = 0
        self.p[ids] = 0.0
        self._dwell[ids] = self.cfg.prio_dwell_steps

    # ---- opponent intent extrapolation -------------------------------------
    def _predict_next(self, robot: str) -> torch.Tensor:
        cfg, hist = self.cfg, self._hist[robot]
        H = cfg.horizon_h
        if cfg.extrap == "zero":
            return torch.zeros_like(hist[:, 0])
        if cfg.extrap == "hold" or H < 2:
            return hist[:, -1]
        t = torch.arange(1, H + 1, device=hist.device, dtype=hist.dtype)
        t_bar = t.mean()
        w = (t - t_bar) / ((t - t_bar) ** 2).sum()           # slope weights
        slope = (hist * w.view(1, H, 1)).sum(dim=1)
        pred = hist.mean(dim=1) + slope * (H + 1 - t_bar)
        # cold start: not enough history -> hold last (avoids wild slopes)
        cold = (self._hist_len[robot] < H).unsqueeze(-1)
        return torch.where(cold, hist[:, -1], pred)

    def _push_hist(self, robot: str, delta: torch.Tensor) -> None:
        self._hist[robot] = torch.cat([self._hist[robot][:, 1:], delta.unsqueeze(1)], dim=1)
        self._hist_len[robot] = (self._hist_len[robot] + 1).clamp(max=self.cfg.horizon_h)

    # ---- fixed-rule priority with hysteresis --------------------------------
    def _update_priority(self, rows: ConstraintRows, cmd_stacked: dict,
                         active_cross: torch.Tensor) -> None:
        cfg = self.cfg
        chi = {}
        for r in ("F", "U"):
            closing = (-(rows.J[r] @ cmd_stacked[r].unsqueeze(-1)).squeeze(-1)).clamp_min(0.0)
            chi[r] = (closing * active_cross.float()).sum(dim=-1)   # commanded closing usage
        # aggressor yields: F much more aggressive -> F yields -> p = -1
        want_neg = chi["F"] > chi["U"] + cfg.prio_deadband
        want_pos = chi["U"] > chi["F"] + cfg.prio_deadband
        target = torch.where(want_pos, 1.0, torch.where(want_neg, -1.0, self.p.sign()))
        # symmetric conflict with no incumbent priority: deterministic tiebreak
        in_conflict = active_cross.any(dim=-1)
        contested = (chi["F"] + chi["U"]) > cfg.prio_deadband
        tie = in_conflict & contested & ~want_pos & ~want_neg & (self.p.sign() == 0)
        target = torch.where(tie, torch.full_like(target, cfg.prio_tiebreak), target)
        target = torch.where(in_conflict, target, torch.zeros_like(target))
        flip = (target.sign() != self.p.sign()) & (target != 0)
        allowed = self._dwell >= cfg.prio_dwell_steps
        target = torch.where(flip & ~allowed, self.p.sign(), target)
        self._dwell = torch.where(flip & allowed, torch.zeros_like(self._dwell),
                                  self._dwell + 1)
        self.p = self.p + (target - self.p).clamp(-cfg.prio_rate, cfg.prio_rate)

    # ---- main entry ----------------------------------------------------------
    def filter(self, state: SceneState, cmd: DeltaCmd) -> FilterOutput:
        cfg, dt = self.cfg, state.dt
        rows = self.provider.rows(state)
        n, m = rows.d.shape

        # delay compensation: advance margins by current d_dot * tau
        qd_s = {r: stack_robot(state.qd, r) for r in ("F", "U")}
        d_dot = sum((rows.J[r] @ qd_s[r].unsqueeze(-1)).squeeze(-1) for r in ("F", "U"))
        d_eff = rows.d + d_dot * cfg.tau_delay

        if cfg.d_act is not None:
            # legacy fixed distance gate (explicit override only)
            active = rows.valid & (d_eff < cfg.d_act)
        else:
            # arbitrated active-set rule: distance branch keeps the delay-
            # compensated margin (a strengthening knob, off at tau_delay=0);
            # TTC branch mirrors A's eligibility exactly (closing = -d_dot).
            ttc_hit = (d_dot < 0) & (rows.d < -d_dot * cfg.tau_ttc)
            active = rows.valid & ((d_eff < cfg.d_soft) | ttc_hit)
        is_cross = rows.cls == CLASS_CROSS
        active_cross = active & is_cross

        cmd_stacked = {r: stack_robot(cmd.delta_q, r) for r in ("F", "U")}
        self._update_priority(rows, cmd_stacked, active_cross)

        # per-row braking boundary when the provider supplies it (A's semantics)
        d_min = rows.d_min if rows.d_min is not None else \
            torch.full_like(rows.d, cfg.d_min)
        cap = cfg.gamma * (d_eff - d_min) * dt            # per-step closing allowance (m)
        budget = {"F": (1.0 + self.p) * 0.5, "U": (1.0 - self.p) * 0.5}

        involves = {
            r: rows.arm_mask[..., [ARM_KEYS.index(a) for a in ARMS_OF_ROBOT[r]]].any(-1)
            for r in ("F", "U")
        }
        pred = {r: self._predict_next(r) for r in ("F", "U")}

        exec_stacked, slack_mag, iters, dbg = {}, {}, {}, {}
        for r, o in (("F", "U"), ("U", "F")):
            # opponent's predicted contribution to d_dot*dt on each row
            c_hat = (rows.J[o] @ pred[o].unsqueeze(-1)).squeeze(-1)
            bud_r = budget[r].unsqueeze(-1)
            bud_o = budget[o].unsqueeze(-1)
            h_cross = bud_r * cap + cfg.borrow_frac * c_hat.clamp_min(0.0) \
                - (-c_hat - bud_o * cap).clamp_min(0.0)
            h = torch.where(is_cross, h_cross, cap)
            rel = active & involves[r]
            G = -rows.J[r]
            box = cfg.vmax * dt
            lo = torch.full_like(cmd_stacked[r], -box)
            hi = torch.full_like(cmd_stacked[r], box)
            c_qp = cmd_stacked[r].clamp(-box, box)
            res = solve_slack_qp_pgd(c_qp, G, h, rel, lo, hi,
                                     rho=cfg.slack_rho, max_iters=cfg.solver_iters,
                                     tol=cfg.solver_tol)
            exec_stacked[r] = res.u
            slack_mag[r] = res.slack.sum(dim=-1)
            iters[r] = res.iters
            if cfg.debug_qp:
                dbg[r] = {"c": c_qp, "G": G, "h": h, "valid": rel,
                          "lo": lo, "hi": hi, "u": res.u}

        exec_ = {}
        for r in ("F", "U"):
            exec_.update(unstack_robot(exec_stacked[r], r, self.dof_of))
            self._push_hist(r, exec_stacked[r])

        dev = torch.stack([(exec_[a] - cmd.delta_q[a]).norm(dim=-1) for a in ARM_KEYS],
                          dim=-1)
        return FilterOutput(
            delta_exec=exec_,
            alpha=effective_alpha(cmd.delta_q, exec_),
            priority_p=self.p.clone(),
            backstop_active=dev > 1e-6,
            info={
                "margin_min": torch.where(rows.valid, rows.d,
                                          torch.full_like(rows.d, _BIG)).min(-1).values,
                "slack_F": slack_mag["F"], "slack_U": slack_mag["U"],
                "iters_F": iters["F"], "iters_U": iters["U"],
                "d_eff": d_eff, "active_rows": active,
                **({"qp_debug": dbg} if cfg.debug_qp else {}),
            },
        )
