"""R17 S8: clutch execution semantics -- four-metric evaluation battery.

Owner product correction (2026-08-20): the s0 model is a safety FILTER for
teleop data collection. Execution semantics = eval/clutch.py hysteresis
binarization (ENGAGED -> bitwise passthrough, LOCKED -> frozen arm; the
learned alpha is a lookahead danger detector, the analytic stack still
bottoms out). This battery measures the four R17 acceptance metrics per
(theta_hi, theta_lo) grid cell on the v7 scene:

  1. passthrough fidelity   regular random stream (l1_ws): among ACTUAL-SAFE
     steps (per-channel SAFE_LINES -- the spec's blanket 30 mm line is
     structurally empty on v7, see the constant's comment; the blanket
     number is still reported as safe_frac_blanket_30mm), the fraction whose
     executed delta equals the commanded delta bitwise (L-inf < 1e-6), plus
     the mean attenuation and its source split (alpha gating vs analytic-
     stack correction -- tube/damper/backstop legitimately modify actions
     too and are reported, not hidden). Target >= 99 %.
  2. brake performance      conflict-injected stream (directed
     head_on_crossing): threat trigger (cross margin drops below the
     TRIGGER_MARGIN d_warn line) -> steps until the at-risk arms' commanded
     delta zeroes (and until physical joint velocity zeroes), whole-window
     min margins, violation rate. Target: 100 % braked, zero violations.
     n_locked_untriggered counts lookahead brakes (locked while the margin
     never even reached the trigger line).
  3. unlock-recovery        scripted "user" (RecoveryScript): after a freeze
     one pair arm's input stream is replaced by a retreat-to-home action,
     the other arm keeps its original stream. Verifies the retreat arm
     re-engages (hysteresis mode) or moves under an explicit user unlock
     (manual mode), the margin recovers, and the other arm resumes
     passthrough. Deadlock = both pair arms frozen and the user has been
     retreating > 2 s with no resolution.
  4. false-brake rate       regular stream: fraction of LOCKED arm-steps
     among actual-safe steps (a false brake interrupts data collection).
     Target <= 5 %.

The pure-torch metric functions (fidelity_stats / false_brake_stats /
brake_rows / recovery_aggregate) are Mac-importable and unit-tested in
tests/test_clutch.py; only main() and the runner touch Isaac.

Server usage (GPU1 only, proxies unset, v7 scene, a21 checkpoint):
  source ~/safeduo_setup/env.sh && cd ~/safeduo && \
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy && \
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m safeduo.eval.clutch_eval \
      --headless --device cpu --ckpt artifacts/runs/a21_v7_r15_s46/model_2000.pt \
      --env-yaml duo_env_v7.yaml --geometry v7 --num-envs 25 --duration-s 60 \
      --theta-pairs 0.7:0.4 --reference --out artifacts/clutch/grid_smoke
Grid lanes split cells via --theta-pairs; afterwards
  python -m safeduo.eval.clutch_eval --summarize-only --out artifacts/clutch/<dir>
(no Isaac import) merges cell_*.json into grid_summary.{json,md}.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from safeduo.delta._contract_stub import ARM_KEYS, DOF_OF

CLASS_KEYS = ("cross", "self_F", "self_U", "table")
SAFE_MARGIN = 0.030          # m -- spec safe line (cross / self_U channels)
# Per-channel "actual safe step" lines, calibrated on the v7 geometry
# (s8_smoke2 diagnostics, 2026-08-20): the spec's blanket 30 mm line is
# structurally EMPTY on v7 -- self_F rests at a CONSTANT 23.08 mm (the S7/R16
# static fat-finger shell) and the table channel works at ~1.6 mm by design
# (hand-on-own-table is the exempted-legal state the near_table semantics
# exists for). cross/self_U keep the spec 30 mm; self_F uses 10 mm (above
# the official violation 0, below the static floor so the line is
# informative); table uses the OFFICIAL exempt-aware violation flag when the
# caller provides it (margin > 0 raw fallback otherwise).
SAFE_LINES = {"cross": 0.030, "self_F": 0.010, "self_U": 0.030}
# Threat trigger for the brake metric: the v7 cross d_warn (0.08). The
# 30 mm line never fires under the clutch -- s8_smoke2 measured the clutch
# freezing every head-on env before the cross margin got below 62 mm.
TRIGGER_MARGIN = 0.080       # m
RECOVER_MARGIN = 0.100       # m -- recovery declared when cross re-exceeds
PASSTHROUGH_TOL = 1e-6       # rad -- bitwise passthrough L-inf tolerance
QD_EPS = 0.05                # rad/s -- physical-stop readout threshold

_ARM_SLICES: list = []
_off = 0
for _arm in ARM_KEYS:
    _ARM_SLICES.append(slice(_off, _off + DOF_OF[_arm]))
    _off += DOF_OF[_arm]


def safe_mask(margins: dict, table_viol: "torch.Tensor | None" = None,
              lines: "dict | None" = None) -> torch.Tensor:
    """(T, N) bool "actual safe step": per-channel margin lines (SAFE_LINES,
    v7-calibrated -- see the constant's comment) and, for the table channel,
    the official exempt-aware violation flag when provided (raw margin > 0
    fallback for margin-only callers)."""
    lines = SAFE_LINES if lines is None else lines
    ok = torch.ones_like(margins["cross"], dtype=torch.bool)
    for k, line in lines.items():
        ok &= margins[k] > line
    if table_viol is not None:
        ok &= ~table_viol.to(torch.bool)
    else:
        ok &= margins["table"] > 0.0
    return ok


def _arm_linf(x: torch.Tensor) -> torch.Tensor:
    """(T, N, 26) -> (T, N, 4) per-arm L-inf."""
    return torch.stack([x[..., sl].abs().amax(dim=-1) for sl in _ARM_SLICES],
                       dim=-1)


def fidelity_stats(cmd: torch.Tensor, exe: torch.Tensor, margins: dict,
                   alpha_exec: torch.Tensor, tol: float = PASSTHROUGH_TOL,
                   table_viol: "torch.Tensor | None" = None) -> dict:
    """Metric 1. cmd/exe (T, N, 26) commanded vs executed delta, margins
    {class: (T, N)}, alpha_exec (T, N, 4) the gate actually applied (binary
    under the clutch, continuous on the reference row).

    Fidelity = fraction of safe steps where ALL 26 dims match bitwise.
    Attenuation sources on non-exact safe arm-steps: "alpha" when the applied
    gate < 1 (clutch lock / continuous attenuation), "stack" when the
    executed delta deviates from alpha*cmd (tube/damper/backstop correction).
    A step can carry both; counts are reported per category and overlapped.
    """
    safe = safe_mask(margins, table_viol)                         # (T, N)
    n_safe = int(safe.sum())
    err_inf = _arm_linf(exe - cmd)                                # (T, N, 4)
    exact_arm = err_inf < tol
    exact_step = exact_arm.all(dim=-1)
    alpha_src = alpha_exec < 1.0 - 1e-9
    gated = torch.cat([alpha_exec[..., i:i + 1] * cmd[..., sl]
                       for i, sl in enumerate(_ARM_SLICES)], dim=-1)
    stack_src = _arm_linf(exe - gated) >= tol

    # effective per-arm attenuation on delta-nonzero safe arm-steps
    att_sum, att_cnt = 0.0, 0
    per_arm_fid = {}
    for i, (arm, sl) in enumerate(zip(ARM_KEYS, _ARM_SLICES)):
        c, e = cmd[..., sl], exe[..., sl]
        den = (c * c).sum(-1)
        nz = (den > 1e-12) & safe
        eff = ((e * c).sum(-1) / den.clamp_min(1e-12)).clamp(0.0, 1.0)
        att_sum += float((1.0 - eff)[nz].sum())
        att_cnt += int(nz.sum())
        per_arm_fid[arm] = (float(exact_arm[..., i][safe].float().mean())
                            if n_safe else 0.0)

    imperfect = ~exact_arm & safe.unsqueeze(-1)                   # (T, N, 4)
    n_imp = int(imperfect.sum())
    # materiality curve: bitwise 1e-6 is the spec caliber; the looser tols
    # separate material trims from numerical dust (cmd scale ~amp rad/step)
    fid_at = {}
    for t in (1e-6, 1e-4, 1e-3):
        fid_at[f"{t:.0e}"] = (float((err_inf < t).all(-1)[safe].float()
                                    .mean()) if n_safe else 0.0)
    dev = err_inf[imperfect].float()
    dev_q = ({"p50": round(float(dev.quantile(0.5)), 6),
              "p90": round(float(dev.quantile(0.9)), 6),
              "p99": round(float(dev.quantile(0.99)), 6),
              "max": round(float(dev.max()), 6)} if n_imp else {})
    all_engaged = (alpha_exec >= 1.0 - 1e-9).all(-1)
    return {
        "fidelity_at_tol": fid_at,
        "imperfect_dev_linf_quantiles": dev_q,
        # clutch-only fidelity: safe steps where NO arm was alpha-attenuated
        # (the stack may still trim; this is the pure clutch interruption
        # view, complement of false_brake_any)
        "clutch_engaged_frac_safe": (float(all_engaged[safe].float().mean())
                                     if n_safe else 0.0),
        "n_steps": int(safe.numel()),
        "n_safe_steps": n_safe,
        "safe_step_frac": float(safe.float().mean()),
        "safe_frac_by_class": {
            k: float((margins[k] > SAFE_LINES.get(k, 0.0)).float().mean())
            for k in CLASS_KEYS},
        "safe_frac_blanket_30mm": float(
            torch.stack([margins[k] > SAFE_MARGIN for k in CLASS_KEYS])
            .all(0).float().mean()),
        "fidelity_bitwise": (float(exact_step[safe].float().mean())
                             if n_safe else 0.0),
        "fidelity_per_arm": per_arm_fid,
        "mean_attenuation_safe": (att_sum / att_cnt) if att_cnt else 0.0,
        "imperfect_arm_steps": n_imp,
        "imperfect_alpha_source": int((imperfect & alpha_src).sum()),
        "imperfect_stack_source": int((imperfect & stack_src).sum()),
        "imperfect_both_sources": int((imperfect & alpha_src & stack_src).sum()),
        "tol": tol,
        "safe_lines": dict(SAFE_LINES),
    }


def bypass_attribution(cmd: torch.Tensor, exe: torch.Tensor, margins: dict,
                       alpha_exec: torch.Tensor, bypass: torch.Tensor,
                       table_viol: "torch.Tensor | None" = None,
                       tol: float = PASSTHROUGH_TOL,
                       blocked_cls: "torch.Tensor | None" = None) -> dict:
    """R19: residual-distortion attribution on safe arm-steps under the
    execution-layer bypass. bypass (T, N, 4) bool = the per-step per-arm
    bypass mask the env actually applied (step_cache["bypass_arm"]).

    Splits imperfect (non-bitwise) safe arm-steps into:
      locked          -- clutch froze the arm (alpha_exec < 1);
      bypassed        -- arm was bypassed yet deviates (expected 0: bitwise
                         passthrough is the bypass contract, verification);
      engaged_blocked -- engaged but emergency-blocked -> analytic stack
                         handled the arm (the R19 residual to attribute).
    For the blocked residual, histograms the ENV-level cross margin band
    (mm) and co-flags self/table proximity (env-level channel mins -- arm
    attribution is not recoverable from channel mins; the bands still
    locate the [30,40) structural-distortion zone). Also reports the
    all-arm-bypassed step share and the fidelity inside it (contract
    check, expect 1.0). Table margins include structural per-link hover
    rows (excluded from the env-side emergency check), so no table band
    is drawn here."""
    safe = safe_mask(margins, table_viol)                          # (T, N)
    err_inf = _arm_linf(exe - cmd)                                 # (T, N, 4)
    imperfect = (err_inf >= tol) & safe.unsqueeze(-1)
    locked = alpha_exec < 1.0 - 1e-9
    engaged_blocked = ~locked & ~bypass
    n_safe_arm = int(safe.sum()) * len(ARM_KEYS)
    all_byp = bypass.all(dim=-1) & safe
    exact_step = (err_inf < tol).all(dim=-1)
    blocked_imp = imperfect & engaged_blocked                      # (T, N, 4)
    blocked_any = blocked_imp.any(dim=-1)                          # (T, N)
    cross_mm = margins["cross"] * 1e3
    bands = ((30.0, 38.0), (38.0, 40.0), (40.0, 43.0), (43.0, 50.0),
             (50.0, 80.0), (80.0, float("inf")))
    band_hist = {}
    for lo, hi in bands:
        sel = blocked_any & (cross_mm > lo) & (cross_mm <= hi)
        key = f"({lo:g},{hi:g}]mm" if hi != float("inf") else f">{lo:g}mm"
        band_hist[key] = int(sel.sum())
    co = {}
    if blocked_any.any():
        co = {"self_F<=18mm": round(float(
                  (margins["self_F"][blocked_any] <= 0.018).float().mean()), 4),
              "self_U<=18mm": round(float(
                  (margins["self_U"][blocked_any] <= 0.018).float().mean()), 4),
              "table<=25mm": round(float(
                  (margins["table"][blocked_any] <= 0.025).float().mean()), 4)}
    by_class = None
    fid_table_clear = None
    if blocked_cls is not None:
        # exact per-arm-step attribution: which class row (hazard = below
        # lock line / gray = inside emergency band) blocked the imperfect
        # engaged arm-step; counts overlap when several classes block
        by_class = {}
        for ci, cname in enumerate(("cross", "self", "table")):
            for hi, hname in enumerate(("hazard", "gray")):
                by_class[f"{cname}_{hname}"] = int(
                    (blocked_imp & blocked_cls[..., ci, hi]).sum())
        by_class["n_blocked_imperfect"] = int(blocked_imp.sum())
        # comparison caliber: step fidelity on safe steps where no arm is
        # table-blocked (cross/self blocking still counts against fidelity;
        # env-level min table margin is exempt-polluted so this is the only
        # clean way to draw the "table clear" line)
        tbl_free = safe & ~blocked_cls[..., 2, :].any(dim=(-2, -1))
        by_class["n_safe_table_clear"] = int(tbl_free.sum())
        fid_table_clear = (float(exact_step[tbl_free].float().mean())
                           if int(tbl_free.sum()) else -1.0)
    return {
        "n_safe_steps": int(safe.sum()),
        "bypass_arm_duty_safe": (float(bypass[safe.unsqueeze(-1).expand_as(
            bypass)].float().mean()) if n_safe_arm else 0.0),
        "all_bypassed_step_frac_safe": (float(all_byp[safe].float().mean())
                                        if int(safe.sum()) else 0.0),
        "fidelity_when_all_bypassed": (float(exact_step[all_byp].float()
                                             .mean()) if int(all_byp.sum())
                                       else -1.0),
        "imperfect_split": {
            "locked": int((imperfect & locked).sum()),
            "bypassed": int((imperfect & bypass).sum()),
            "engaged_blocked": int(blocked_imp.sum()),
        },
        "blocked_step_cross_band": band_hist,
        "blocked_step_coflags": co,
        "blocked_by_class": by_class,
        "fidelity_safe_table_clear": fid_table_clear,
    }


def lock_lookahead_stats(margins: dict, table_viol: torch.Tensor,
                         engaged: torch.Tensor, horizon: int = 30,
                         lock_lines: "dict | None" = None) -> dict:
    """Hindsight consistency of clutch locks on SAFE steps: fraction of
    locked-arm safe steps where some channel actually dropped below the
    LOCK line (cross 38 / self 13 mm, official table flag) within the next
    `horizon` steps -- i.e. the lock was a correct 0.5 s lookahead fire
    that the instantaneous safe-line caliber counts as a false brake.
    Env-level margins (per-arm attribution not recoverable)."""
    ll = lock_lines or {"cross": 0.038, "self_F": 0.013, "self_U": 0.013}
    hazard = torch.zeros_like(table_viol, dtype=torch.bool)
    for k, line in ll.items():
        hazard |= margins[k] < line
    hazard |= table_viol.to(torch.bool)
    T = hazard.shape[0]
    fut = torch.zeros_like(hazard)
    inf = horizon + 1
    # backward O(T) scan; before iteration t, `dist` = steps from t+1 to the
    # next hazard at >= t+1 (0 when hazard[t+1], saturated at `inf`)
    dist = torch.full(hazard.shape[1:], inf, dtype=torch.long,
                      device=hazard.device)
    for t in range(T - 1, -1, -1):
        fut[t] = dist <= horizon - 1          # hazard within (t, t+horizon]
        dist = torch.where(hazard[t], torch.zeros_like(dist),
                           (dist + 1).clamp_max(inf))
    safe = safe_mask(margins, table_viol)
    locked_any = (~engaged).any(dim=-1) & safe
    n = int(locked_any.sum())
    return {
        "n_locked_safe_steps": n,
        "lookahead_consistent_frac": (float(fut[locked_any].float().mean())
                                      if n else -1.0),
        "horizon_steps": horizon,
    }


CALIB_BINS_MM = (0.0, 30.0, 38.0, 43.0, 50.0, 65.0, 80.0, 120.0, 200.0,
                 float("inf"))
CALIB_THETAS = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)


def calibration_stats(margins: dict, alpha_policy: torch.Tensor,
                      thetas: tuple = CALIB_THETAS) -> dict:
    """Hazard-head calibration curve: per cross-margin band (mm), the
    min-over-arms detector alpha quantiles and P(min alpha < theta) --
    the direct empirical basis for picking theta_lo (any-arm lock trigger)
    against the margin the env is actually at."""
    cross_mm = (margins["cross"] * 1e3).flatten()
    a_min = alpha_policy.min(dim=-1).values.flatten().float()
    rows = []
    for lo, hi in zip(CALIB_BINS_MM[:-1], CALIB_BINS_MM[1:]):
        sel = (cross_mm > lo) & (cross_mm <= hi)
        n = int(sel.sum())
        row = {"band_mm": f"({lo:g},{hi:g}]" if hi != float("inf")
               else f">{lo:g}", "n": n}
        if n:
            a = a_min[sel]
            row["alpha_min_p10"] = round(float(a.quantile(0.10)), 4)
            row["alpha_min_p50"] = round(float(a.quantile(0.50)), 4)
            row.update({f"P(a<{th:g})": round(float((a < th).float().mean()), 4)
                        for th in thetas})
        rows.append(row)
    return {"bins": rows, "note": "min-over-arms detector alpha vs env "
                                  "cross-margin band"}


def false_brake_stats(margins: dict, engaged: torch.Tensor,
                      table_viol: "torch.Tensor | None" = None) -> dict:
    """Metric 4: LOCKED share inside actual-safe steps. engaged (T, N, 4)."""
    safe = safe_mask(margins, table_viol)
    locked = ~engaged
    n_safe = int(safe.sum())
    per_arm = {}
    for i, arm in enumerate(ARM_KEYS):
        per_arm[arm] = (float(locked[..., i][safe].float().mean())
                        if n_safe else 0.0)
    return {
        "n_safe_steps": n_safe,
        "false_brake_arm_rate": (float(locked[safe].float().mean())
                                 if n_safe else 0.0),
        "false_brake_any_rate": (float(locked.any(-1)[safe].float().mean())
                                 if n_safe else 0.0),
        "false_brake_per_arm": per_arm,
        "safe_lines": dict(SAFE_LINES),
    }


def stream_diagnostics(margins: dict, engaged: torch.Tensor,
                       alpha_policy: torch.Tensor) -> dict:
    """Calibration readouts per stream: per-class margin quantiles, per-class
    frac above the safe line, detector-alpha quantiles and lock duty per
    arm (unconditioned). Feeds the report's threshold-calibration section."""
    qs = (0.01, 0.05, 0.50)
    diag = {"margin_quantiles": {}, "margin_frac_above_line": {}}
    for k in CLASS_KEYS:
        m = margins[k].flatten().float()
        diag["margin_quantiles"][k] = {
            f"p{int(q * 100)}": round(float(m.quantile(q)), 5) for q in qs}
        diag["margin_frac_above_line"][k] = round(
            float((m > SAFE_LINES.get(k, 0.0)).float().mean()), 4)
    ap = alpha_policy.reshape(-1, alpha_policy.shape[-1]).float()
    lk = (~engaged).reshape(-1, engaged.shape[-1]).float()
    diag["alpha_policy_quantiles"] = {
        arm: {"p10": round(float(ap[:, i].quantile(0.10)), 4),
              "p50": round(float(ap[:, i].quantile(0.50)), 4),
              "p90": round(float(ap[:, i].quantile(0.90)), 4)}
        for i, arm in enumerate(ARM_KEYS)}
    diag["locked_frac_per_arm"] = {
        arm: round(float(lk[:, i].mean()), 4)
        for i, arm in enumerate(ARM_KEYS)}
    return diag


def brake_rows(margins: dict, engaged: torch.Tensor, exe: torch.Tensor,
               qd: torch.Tensor, cross_arms: torch.Tensor,
               table_viol: torch.Tensor, dt: float,
               trigger: float = TRIGGER_MARGIN, tol: float = PASSTHROUGH_TOL,
               qd_eps: float = QD_EPS) -> list:
    """Metric 2 per-env rows. cross_arms (T, N, 2) = per-step argmin cross
    pair arm ids; the at-risk arms of an env are the argmin pair at its
    FIRST trigger step. table_viol (T, N) = exempt-aware table flag so the
    official any-class violation count matches the endurance caliber.
    """
    T, N = margins["cross"].shape
    exe_arm = _arm_linf(exe)                                       # (T, N, 4)
    qd_arm = _arm_linf(qd)
    locked = ~engaged
    stack = torch.stack([margins[k] for k in CLASS_KEYS])          # (C, T, N)
    viol = stack < 0.0
    viol[CLASS_KEYS.index("table")] = table_viol.to(torch.bool)
    rows = []
    for e in range(N):
        cross = margins["cross"][:, e]
        below = cross < trigger
        row = {"env_idx": e, "triggered": bool(below.any()),
               "locked_any": bool(locked[:, e].any()),
               "min_margin_by_class": {
                   k: round(float(stack[i, :, e].min()), 5)
                   for i, k in enumerate(CLASS_KEYS)},
               "violation_steps_any": int(viol.any(0)[:, e].sum()),
               "violation_steps_cross": int(
                   viol[CLASS_KEYS.index("cross")][:, e].sum())}
        if not row["triggered"]:
            rows.append(row)
            continue
        t0 = int(below.float().argmax())
        arms = sorted({int(a) for a in cross_arms[t0, e].tolist() if a >= 0})
        row.update({"trigger_step": t0,
                    "at_risk_arms": [ARM_KEYS[a] for a in arms]})

        def _first(cond: torch.Tensor) -> int:
            idx = cond[t0:].nonzero(as_tuple=True)[0]
            return int(idx[0]) if idx.numel() else -1

        # two stop levels: "any" = at least one pair arm stopped (the p-head
        # semantics: one party yields while the other legitimately continues
        # or retreats), "all" = both hands still (the owner's example). The
        # headline braked/brake_success use ANY; ALL is reported alongside.
        lock_any = _first(locked[:, e, arms].any(-1))
        lock_all = _first(locked[:, e, arms].all(-1))
        cmd_stop = _first((exe_arm[:, e, arms] < tol).any(-1))
        cmd_stop_all = _first((exe_arm[:, e, arms] < tol).all(-1))
        phys_stop = _first((qd_arm[:, e, arms] < qd_eps).any(-1))
        phys_stop_all = _first((qd_arm[:, e, arms] < qd_eps).all(-1))
        row.update({
            "lock_delay_steps": lock_any,
            "lock_all_delay_steps": lock_all,
            "cmd_stop_delay_steps": cmd_stop,
            "cmd_stop_all_delay_steps": cmd_stop_all,
            "phys_stop_delay_steps": phys_stop,
            "phys_stop_all_delay_steps": phys_stop_all,
            "cmd_stop_delay_ms": round(cmd_stop * dt * 1e3, 1)
            if cmd_stop >= 0 else -1.0,
            "phys_stop_delay_ms": round(phys_stop * dt * 1e3, 1)
            if phys_stop >= 0 else -1.0,
            "braked": bool(cmd_stop >= 0),
            "brake_success": bool(cmd_stop >= 0
                                  and row["violation_steps_cross"] == 0),
        })
        rows.append(row)
    return rows


def _delay_stats(vals: list) -> dict:
    if not vals:
        return {"mean": -1.0, "p50": -1.0, "p95": -1.0, "max": -1}
    t = torch.tensor([float(v) for v in vals])
    return {"mean": round(float(t.mean()), 2),
            "p50": round(float(t.median()), 1),
            "p95": round(float(t.quantile(0.95)), 1),
            "max": int(t.max())}


def brake_aggregate(rows: list) -> dict:
    trig = [r for r in rows if r["triggered"]]
    braked = [r for r in trig if r["braked"]]
    return {
        "n_runs": len(rows),
        "n_triggered": len(trig),
        "n_braked": len(braked),
        "brake_rate": (len(braked) / len(trig)) if trig else 0.0,
        "n_brake_success": sum(1 for r in trig if r["brake_success"]),
        # lookahead brakes: the clutch locked while the margin never even
        # crossed the trigger line (early detector fire, no threat exposure)
        "n_locked_untriggered": sum(1 for r in rows if not r["triggered"]
                                    and r.get("locked_any", False)),
        "violation_free_rate": (sum(1 for r in trig
                                    if r["violation_steps_cross"] == 0)
                                / len(trig)) if trig else 0.0,
        "violation_free_any_class_rate": (
            sum(1 for r in trig if r["violation_steps_any"] == 0)
            / len(trig)) if trig else 0.0,
        "n_cmd_stop_all": sum(1 for r in trig
                              if r.get("cmd_stop_all_delay_steps", -1) >= 0),
        "lock_delay_steps": _delay_stats(
            [r["lock_delay_steps"] for r in trig
             if r.get("lock_delay_steps", -1) >= 0]),
        "cmd_stop_delay_steps": _delay_stats(
            [r["cmd_stop_delay_steps"] for r in braked]),
        "cmd_stop_all_delay_steps": _delay_stats(
            [r["cmd_stop_all_delay_steps"] for r in trig
             if r.get("cmd_stop_all_delay_steps", -1) >= 0]),
        "phys_stop_delay_steps": _delay_stats(
            [r["phys_stop_delay_steps"] for r in braked
             if r["phys_stop_delay_steps"] >= 0]),
        "phys_stop_all_delay_steps": _delay_stats(
            [r["phys_stop_all_delay_steps"] for r in trig
             if r.get("phys_stop_all_delay_steps", -1) >= 0]),
        "min_cross_margin": (round(min(r["min_margin_by_class"]["cross"]
                                       for r in rows), 5) if rows else 0.0),
    }


def recovery_aggregate(rows: list) -> dict:
    froze = [r for r in rows if r["froze"]]
    ret = [r for r in froze if r["retreat_start"] >= 0]
    deadlocks = [r for r in ret if r["deadlock"]]

    def _delays(key: str, base: str = "retreat_start") -> dict:
        return _delay_stats([r[key] - r[base] for r in ret
                             if r[key] >= 0 and r[base] >= 0])

    return {
        "n_runs": len(rows),
        "n_froze": len(froze),
        "n_retreat": len(ret),
        "n_deadlock": len(deadlocks),
        "froze_frac": (len(froze) / len(rows)) if rows else 0.0,
        "no_deadlock_rate": (1.0 - len(deadlocks) / len(ret)) if ret else 1.0,
        "retreat_moved_rate": (sum(1 for r in ret
                                   if r["retreat_moved_step"] >= 0)
                               / len(ret)) if ret else 0.0,
        "re_engage_rate": (sum(1 for r in ret if r["re_engage_step"] >= 0)
                           / len(ret)) if ret else 0.0,
        "margin_recovered_rate": (sum(1 for r in ret
                                      if r["margin_recover_step"] >= 0)
                                  / len(ret)) if ret else 0.0,
        "other_resumed_rate": (sum(1 for r in ret
                                   if r["other_resume_step"] >= 0)
                               / len(ret)) if ret else 0.0,
        "done_rate": (sum(1 for r in ret if r["done"]) / len(ret))
        if ret else 0.0,
        "re_engage_delay_steps": _delays("re_engage_step"),
        "margin_recover_delay_steps": _delays("margin_recover_step"),
        "other_resume_delay_steps": _delays("other_resume_step"),
    }


# --------------------------------------------------------------------------
# Isaac-side pieces (constructed only inside main/runner)
# --------------------------------------------------------------------------

class CrossArgminTracker:
    """Per-step argmin cross pair arm ids (N, 2), same centers-based readout
    family as endurance_eval.SelfUNegativePairRecorder but for the cross
    slice and arm-level attribution only."""

    def __init__(self, sph) -> None:
        pc = int(sph._slice_cross.stop)
        if pc == 0:
            raise ValueError("sphere module has no cross pairs")
        self._sph = sph
        self._pairs = sph.pairs_cross
        self._radius_sum = (sph.radii[self._pairs[:, 0]]
                            + sph.radii[self._pairs[:, 1]])
        self._pair_arms = sph.pair_arms[:pc]                     # (P, 2)
        self.steps: list = []

    def capture(self) -> None:
        centers = self._sph.last_centers
        assert centers is not None, "compute() must precede capture()"
        ci = centers[:, self._pairs[:, 0]]
        cj = centers[:, self._pairs[:, 1]]
        margins = (ci - cj).norm(dim=-1) - self._radius_sum.unsqueeze(0)
        argmin = margins.argmin(dim=1)                           # (N,)
        self.steps.append(self._pair_arms[argmin].detach().cpu().clone())

    def stacked(self) -> torch.Tensor:
        return torch.stack(self.steps)                           # (T, N, 2)


class RetreatOverrideSource:
    """Delta-source wrapper: per-(env, arm) retreat mask replaces the inner
    stream with a retreat-to-home delta (clamp(q_home - q, +-amp)) -- the
    scripted 'user pulls the arm back' model. q_home is captured lazily at
    the first sample() after reset (= the birth pose the window starts
    from). Everything else delegates to the inner source (assignment/names
    duck-typing included)."""

    def __init__(self, inner, num_envs: int, amp: float,
                 device: "str | torch.device" = "cpu"):
        self._inner = inner
        self._amp = float(amp)
        self.retreat_mask = {
            arm: torch.zeros(num_envs, dtype=torch.bool,
                             device=torch.device(device))
            for arm in ARM_KEYS}
        self.q_home: "dict | None" = None

    def sample(self, state):
        cmd = self._inner.sample(state)
        if self.q_home is None:
            self.q_home = {arm: state.q[arm].detach().clone()
                           for arm in ARM_KEYS}
        for arm in ARM_KEYS:
            m = self.retreat_mask[arm]
            if bool(m.any()):
                back = (self.q_home[arm] - state.q[arm]).clamp(
                    -self._amp, self._amp)
                cmd.delta_q[arm] = torch.where(
                    m.unsqueeze(-1), back.to(cmd.delta_q[arm].dtype),
                    cmd.delta_q[arm])
        return cmd

    def __getattr__(self, name):
        return getattr(self._inner, name)


class RecoveryScript:
    """Metric-3 scripted user, one state machine per env.

    WAIT_FREEZE: cross margin < trigger AND >= 1 argmin-pair arm LOCKED
                 -> record freeze, wait react_steps (user reaction time).
    RETREAT:     retreat arm = locked pair arm with the LOWER alpha_policy;
                 its stream is replaced by retreat-to-home; manual mode
                 additionally holds the arm ENGAGED (explicit user unlock).
                 The other pair arm keeps its original stream.
    DONE:        cross margin > recover_margin AND the other arm re-engaged.

    Deadlock (spec): both pair arms frozen and the user has been retreating
    > deadlock_steps (2 s) with the retreat arm never re-engaged.
    """

    WAIT_FREEZE, REACT, RETREAT, DONE = 0, 1, 2, 3

    def __init__(self, driver, src: RetreatOverrideSource, num_envs: int,
                 mode: str = "hysteresis", trigger: float = TRIGGER_MARGIN,
                 recover_margin: float = RECOVER_MARGIN,
                 react_steps: int = 30, deadlock_steps: int = 120,
                 tol: float = PASSTHROUGH_TOL):
        assert mode in ("hysteresis", "manual"), mode
        self.driver = driver
        self.src = src
        self.mode = mode
        self.trigger = float(trigger)
        self.recover_margin = float(recover_margin)
        self.react_steps = int(react_steps)
        self.deadlock_steps = int(deadlock_steps)
        self.tol = float(tol)
        n = num_envs
        self._pair: dict = {}
        self.phase = [self.WAIT_FREEZE] * n
        self.freeze_step = [-1] * n
        self.retreat_start = [-1] * n
        self.retreat_arm = [-1] * n
        self.other_arm = [-1] * n
        self.re_engage_step = [-1] * n
        self.retreat_moved_step = [-1] * n
        self.margin_recover_step = [-1] * n
        self.other_resume_step = [-1] * n
        self.deadlock = [False] * n

    def step(self, t: int, mm_cross: torch.Tensor, engaged: torch.Tensor,
             alpha_policy: torch.Tensor, cross_arms: torch.Tensor,
             exe: torch.Tensor) -> None:
        """Called after env.step with this step's post-step readouts:
        mm_cross (N,), engaged/alpha_policy (N, 4), cross_arms (N, 2),
        exe (N, 26) executed delta."""
        exe_arm = _arm_linf(exe.unsqueeze(0))[0]                  # (N, 4)
        for e in range(len(self.phase)):
            ph = self.phase[e]
            if ph == self.DONE:
                continue
            locked = ~engaged[e]
            if ph == self.WAIT_FREEZE:
                pair = sorted({int(a) for a in cross_arms[e].tolist()
                               if a >= 0})
                if (float(mm_cross[e]) < self.trigger and pair
                        and bool(locked[pair].any())):
                    self.freeze_step[e] = t
                    self._pair[e] = pair          # pair identity at freeze
                    self.phase[e] = self.REACT
                continue
            pair = self._pair[e]
            if ph == self.REACT:
                if t - self.freeze_step[e] < self.react_steps:
                    continue
                locked_pair = [a for a in pair if bool(locked[a])]
                if not locked_pair:
                    # freeze resolved itself before the user reacted
                    self.phase[e] = self.WAIT_FREEZE
                    self.freeze_step[e] = -1
                    continue
                ra = min(locked_pair, key=lambda a: float(alpha_policy[e, a]))
                others = [a for a in pair if a != ra]
                self.retreat_arm[e] = ra
                self.other_arm[e] = others[0] if others else -1
                self.retreat_start[e] = t
                self.src.retreat_mask[ARM_KEYS[ra]][e] = True
                if self.mode == "manual":
                    self.driver.set_hold(torch.tensor([e]), ra, True)
                self.phase[e] = self.RETREAT
                continue
            # RETREAT
            ra, oa = self.retreat_arm[e], self.other_arm[e]
            if self.re_engage_step[e] < 0 and bool(engaged[e, ra]):
                self.re_engage_step[e] = t
            if (self.retreat_moved_step[e] < 0
                    and float(exe_arm[e, ra]) > self.tol):
                self.retreat_moved_step[e] = t
            if (self.margin_recover_step[e] < 0
                    and float(mm_cross[e]) > self.recover_margin):
                self.margin_recover_step[e] = t
            if (not self.deadlock[e] and self.re_engage_step[e] < 0
                    and t - self.retreat_start[e] > self.deadlock_steps
                    and bool(locked[pair].all())):
                self.deadlock[e] = True
            if self.margin_recover_step[e] >= 0 and oa >= 0 \
                    and self.other_resume_step[e] < 0 \
                    and bool(engaged[e, oa]):
                self.other_resume_step[e] = t
            if self.margin_recover_step[e] >= 0 and (
                    oa < 0 or self.other_resume_step[e] >= 0):
                self.src.retreat_mask[ARM_KEYS[ra]][e] = False
                if self.mode == "manual":
                    self.driver.set_hold(torch.tensor([e]), ra, False)
                self.phase[e] = self.DONE

    def rows(self) -> list:
        out = []
        for e in range(len(self.phase)):
            out.append({
                "env_idx": e,
                "mode": self.mode,
                "froze": self.freeze_step[e] >= 0
                or self.retreat_start[e] >= 0,
                "freeze_step": self.freeze_step[e],
                "retreat_start": self.retreat_start[e],
                "retreat_arm": (ARM_KEYS[self.retreat_arm[e]]
                                if self.retreat_arm[e] >= 0 else None),
                "other_arm": (ARM_KEYS[self.other_arm[e]]
                              if self.other_arm[e] >= 0 else None),
                "re_engage_step": self.re_engage_step[e],
                "retreat_moved_step": self.retreat_moved_step[e],
                "margin_recover_step": self.margin_recover_step[e],
                "other_resume_step": self.other_resume_step[e],
                "deadlock": self.deadlock[e],
                "done": self.phase[e] == self.DONE,
            })
        return out


def run_traced_window(env, driver, steps: int, script: "RecoveryScript | None" = None,
                      cross_tracker: "CrossArgminTracker | None" = None,
                      log_every: int = 600) -> dict:
    """Reset-free traced window: per-step margins, exempt-aware table flags,
    cmd/exec (T, N, 26), applied alpha, clutch engagement, detector alpha,
    joint velocities, argmin cross arms. Reference (non-clutch) drivers get
    engaged := alpha > 1 - 1e-6 and alpha_policy := applied alpha."""
    from safeduo.eval.endurance_eval import table_flags_from_module

    margins = {k: [] for k in CLASS_KEYS}
    table_viol, table_exempt = [], []
    cmds, exes, alphas, engageds, alpha_pols, qds = [], [], [], [], [], []
    bypasses, blocked_clss = [], []
    obs, _ = env.reset()
    driver.reset(torch.arange(env.num_envs, device=env.device))
    t0 = time.time()
    for t in range(steps):
        action = driver.act(env, obs)
        obs, _, term, trunc, _ = env.step(action)
        if bool((term | trunc).any()):
            raise RuntimeError(
                f"env reset inside a clutch window at step {t} -- check "
                "terminate_on_violation=False and episode_length_s")
        mm = env._last_out.min_margin
        for k in CLASS_KEYS:
            margins[k].append(mm[k].detach().cpu().clone())
        tv, te = table_flags_from_module(env._sph)
        table_viol.append(tv.detach().cpu().clone())
        table_exempt.append(te.detach().cpu().clone())
        c = env._step_cache
        cmds.append(c["cmd"].stacked().detach().cpu().clone())
        exes.append(c["exec"].stacked().detach().cpu().clone())
        alpha = c["alpha"].detach().cpu().clone()
        alphas.append(alpha)
        eng = getattr(driver, "engaged", None)
        engageds.append(eng.detach().cpu().clone() if eng is not None
                        else alpha > 1.0 - 1e-6)
        ap = getattr(driver, "last_alpha_policy", None)
        alpha_pols.append(ap.detach().cpu().clone() if ap is not None
                          else alpha.clone())
        bp = c.get("bypass_arm")                # R19: absent = bypass off
        bypasses.append(bp.detach().cpu().clone() if bp is not None
                        else torch.zeros(alpha.shape[0], len(ARM_KEYS),
                                         dtype=torch.bool))
        bc = c.get("bypass_blocked_cls")        # (N, 4, 3, 2) [hazard, gray]
        blocked_clss.append(bc.detach().cpu().clone() if bc is not None
                            else torch.zeros(alpha.shape[0], len(ARM_KEYS),
                                             3, 2, dtype=torch.bool))
        qds.append(torch.cat(
            [env._arms[a].data.joint_vel[:, env._joint_idx[a]]
             for a in ARM_KEYS], dim=-1).detach().cpu().clone())
        if cross_tracker is not None:
            cross_tracker.capture()
        if script is not None:
            script.step(t, margins["cross"][-1], engageds[-1],
                        alpha_pols[-1],
                        cross_tracker.steps[-1] if cross_tracker else None,
                        exes[-1])
        if t % log_every == 0:
            rate = (t + 1) / max(time.time() - t0, 1e-9)
            print(f"[clutch] step {t}/{steps} ({rate:.1f} steps/s, eta "
                  f"{(steps - t) / max(rate, 1e-9):.0f}s)", flush=True)
    return {
        "margins": {k: torch.stack(v) for k, v in margins.items()},
        "table_viol": torch.stack(table_viol),
        "table_exempt": torch.stack(table_exempt),
        "cmd": torch.stack(cmds),
        "exec": torch.stack(exes),
        "alpha": torch.stack(alphas),
        "engaged": torch.stack(engageds),
        "alpha_policy": torch.stack(alpha_pols),
        "bypass": torch.stack(bypasses),
        "blocked_cls": torch.stack(blocked_clss),
        "qd": torch.stack(qds),
        "cross_arms": (cross_tracker.stacked()
                       if cross_tracker is not None else None),
        "wall_s": time.time() - t0,
    }


def make_headon_flow(env, amp: float, env_yaml: str, geometry: str):
    """Directed conflict-injection stream: head_on_crossing family only
    (spec metric 2), eval split, same ConflictMixSource plumbing as
    endurance_eval's directed flow."""
    from safeduo.delta.l2_env_source import ConflictMixSource

    cfg = {"mix": {"l1": 0.0, "l2": 1.0},
           "scenarios": {"head_on_crossing": 1.0},
           "split": "eval", "n_variants": 100, "stages": None,
           "geometry": geometry}
    return ConflictMixSource(env.num_envs, cfg, device=env.device,
                             amp_max=float(amp), env_yaml=env_yaml)


# --------------------------------------------------------------------------
# grid summary (pure; also reachable via --summarize-only, no Isaac import)
# --------------------------------------------------------------------------

def render_grid(cells: list) -> str:
    """cells: list of cell payload dicts -> markdown grid table."""
    lines = [
        "# R17 clutch theta grid",
        "",
        "| theta_hi | theta_lo | fidelity(safe) | mean atten | false-brake "
        "arm | false-brake any | brake rate | viol-free(cross) | cmd-stop "
        "p50/p95 (steps) | no-deadlock(hys) | done(hys) | done(manual) | froze-cov(hys/man) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in sorted(cells, key=lambda x: (x["theta_hi"], x["theta_lo"])):
        fid = c["fidelity"]
        fb = c["false_brake"]
        br = c["brake"]["aggregate"]
        rh = c.get("recovery_hysteresis", {}).get("aggregate", {})
        rm = c.get("recovery_manual", {}).get("aggregate", {})
        cs = br["cmd_stop_delay_steps"]
        lines.append(
            f"| {c['theta_hi']} | {c['theta_lo']} "
            f"| {fid['fidelity_bitwise']:.4f} "
            f"| {fid['mean_attenuation_safe']:.4f} "
            f"| {fb['false_brake_arm_rate']:.4f} "
            f"| {fb['false_brake_any_rate']:.4f} "
            f"| {br['brake_rate']:.3f} "
            f"| {br['violation_free_rate']:.3f} "
            f"| {cs['p50']}/{cs['p95']} "
            f"| {rh.get('no_deadlock_rate', float('nan')):.3f} "
            f"| {rh.get('done_rate', float('nan')):.3f} "
            f"| {rm.get('done_rate', float('nan')):.3f} "
            f"| {_froze_cov(rh)}/{_froze_cov(rm)} |")
    return "\n".join(lines)


def _froze_cov(agg: dict) -> str:
    """Recovery-gate coverage: how many runs actually froze (older cells lack
    froze_frac -> derive from n_froze/n_runs; both absent -> 'n/a')."""
    if "froze_frac" in agg:
        return f"{agg['froze_frac']:.2f}"
    if agg.get("n_runs"):
        return f"{agg.get('n_froze', 0) / agg['n_runs']:.2f}"
    return "n/a"


def summarize(out_dir: "str | Path") -> dict:
    out_dir = Path(out_dir)
    cells = [json.loads(p.read_text())
             for p in sorted(out_dir.glob("cell_*.json"))]
    ref_p = out_dir / "reference.json"
    ref = json.loads(ref_p.read_text()) if ref_p.exists() else None
    summary = {"cells": cells, "reference": ref,
               "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "grid_summary.json").write_text(json.dumps(summary, indent=1))
    md = render_grid(cells)
    if ref is not None:
        fid = ref["fidelity"]
        md += ("\n\nreference (clutch OFF, continuous alpha): fidelity "
               f"{fid['fidelity_bitwise']:.4f}, mean attenuation "
               f"{fid['mean_attenuation_safe']:.4f}\n")
    (out_dir / "grid_summary.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    return summary


# --------------------------------------------------------------------------
# main (Isaac)
# --------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize-only", action="store_true",
                        help="merge existing cell_*.json in --out into the "
                             "grid summary (no Isaac)")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--env-yaml", type=str, default="duo_env_v7.yaml")
    parser.add_argument("--geometry", type=str, choices=("v5", "v7"),
                        default="v7")
    parser.add_argument("--num-envs", type=int, default=25)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--theta-pairs", type=str, default="0.7:0.4",
                        help="comma list of theta_hi:theta_lo cells, e.g. "
                             "'0.6:0.3,0.7:0.4' (lane splitting)")
    parser.add_argument("--hazard-source", choices=("beta", "head"),
                        default="beta",
                        help="loop-4 detector source: beta = Beta policy mean "
                             "(legacy, bit-identical default); head = "
                             "1-sigmoid(hazard_logits/T) from the a25+ "
                             "independent hazard head. Applies to clutch "
                             "cells only; reference cells always use beta")
    parser.add_argument("--hazard-temp", type=float, default=1.0,
                        help="hazard head temperature T (logits/T before "
                             "sigmoid). Theta-equivalent for the four gates "
                             "(theta'=sigmoid(logit(theta)/T)); only spreads "
                             "saturated logits. Different T must use "
                             "different --out (cell filenames lack T)")
    parser.add_argument("--retreat-passthrough", action="store_true",
                        help="R30: release locked arms whose command is "
                             "non-closing on all rows within d_warn")
    parser.add_argument("--retreat-dwell", type=int, default=10,
                        help="R30b: consecutive locked steps before retreat "
                             "pass-through may release (0 = R30 immediate)")
    parser.add_argument("--reference", action="store_true",
                        help="also run the clutch-OFF reference cell "
                             "(fidelity/false-brake/brake, no recovery)")
    parser.add_argument("--regular-amps", nargs="+", type=float,
                        default=[0.015, 0.03],
                        help="l1_ws window amps for metrics 1/4")
    parser.add_argument("--directed-amps", nargs="+", type=float,
                        default=[0.03, 0.06],
                        help="head-on window amps for metric 2")
    parser.add_argument("--bypass-mm", type=float, default=None,
                        help="R19 execution-layer bypass: emergency band "
                             "width in mm above the per-row LOCK line "
                             "(engaged arms with all retained rows clear of "
                             "the band pass through bitwise, analytic stack "
                             "fully bypassed; default None = off = "
                             "bit-identical d5 caliber)")
    parser.add_argument("--recovery-amp", type=float, default=0.03)
    parser.add_argument("--recovery-modes", nargs="+",
                        default=["hysteresis", "manual"],
                        choices=("hysteresis", "manual"))
    parser.add_argument("--metrics", type=str, default="all",
                        choices=("all", "regular", "regular+directed"),
                        help="cell scope: 'regular' = metrics 1/4 only "
                             "(cheap theta scan), 'regular+directed' adds "
                             "the brake metric, 'all' (default) = full "
                             "battery including recovery")
    parser.add_argument("--out", type=str, required=True)

    if "--summarize-only" in __import__("sys").argv:
        args, _ = parser.parse_known_args()
        summarize(args.out)
        return

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app  # noqa: F841

    from safeduo.envs.duo_env import DuoEnv, make_duo_env_cfg
    from safeduo.eval.block1_harness import PolicyDriver
    from safeduo.eval.clutch import wrap_clutch
    from safeduo.eval.endurance_eval import (ckpt_arm_aware, ckpt_p2_obs,
                                             make_endurance_flow)

    if not args.ckpt:
        raise SystemExit("--ckpt is required")
    theta_cells = []
    for tok in args.theta_pairs.split(","):
        hi, lo = tok.strip().split(":")
        theta_cells.append((float(hi), float(lo)))

    arm_aware = ckpt_arm_aware(args.ckpt)
    p2 = ckpt_p2_obs(args.ckpt)  # R22 P2: 547-dim obs ckpts need the env tail
    cfg = make_duo_env_cfg(num_envs=args.num_envs, device=args.device,
                           yaml_name=args.env_yaml, coordinator=True,
                           arm_aware_obs=arm_aware, p2_obs=p2)
    cfg.coordinator["terminate_on_violation"] = False
    if args.retreat_passthrough:
        cfg.coordinator["retreat_passthrough"] = True
        cfg.coordinator["retreat_dwell_steps"] = int(args.retreat_dwell)
    ctrl_dt = cfg.sim.dt * cfg.decimation
    cfg.episode_length_s = float(args.duration_s) + 5.0
    cfg.seed = args.seed
    env = DuoEnv(cfg)
    if args.bypass_mm is not None:
        env.set_r19_bypass_mm(args.bypass_mm)
        print(f"[clutch] R19 bypass ON, emergency band = {args.bypass_mm} mm "
              "above per-row lock line", flush=True)
    steps = int(round(args.duration_s / ctrl_dt))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def new_window(amp: float, w: int, flow: str):
        env._gen.manual_seed(args.seed + w)
        env._pending_cmd = None
        if flow == "regular":
            env._delta_src = make_endurance_flow(
                "l1_ws", env, amp, args.env_yaml, wild=True,
                geometry=args.geometry)
        else:
            env._delta_src = make_headon_flow(env, amp, args.env_yaml,
                                              args.geometry)

    def merge_fidelity(traces_list: list) -> tuple:
        """Concatenate windows along T for the step-level metrics 1/4."""
        cat = {
            "margins": {k: torch.cat([t["margins"][k] for t in traces_list])
                        for k in CLASS_KEYS},
            "cmd": torch.cat([t["cmd"] for t in traces_list]),
            "exec": torch.cat([t["exec"] for t in traces_list]),
            "alpha": torch.cat([t["alpha"] for t in traces_list]),
            "engaged": torch.cat([t["engaged"] for t in traces_list]),
            "table_viol": torch.cat([t["table_viol"] for t in traces_list]),
            "bypass": torch.cat([t["bypass"] for t in traces_list]),
            "blocked_cls": torch.cat([t["blocked_cls"] for t in traces_list]),
        }
        alpha_pol = torch.cat([t["alpha_policy"] for t in traces_list])
        fid = fidelity_stats(cat["cmd"], cat["exec"], cat["margins"],
                             cat["alpha"], table_viol=cat["table_viol"])
        fb = false_brake_stats(cat["margins"], cat["engaged"],
                               table_viol=cat["table_viol"])
        diag = stream_diagnostics(cat["margins"], cat["engaged"], alpha_pol)
        # R19 additive diagnostics (cheap, computed regardless of bypass)
        extras = {
            "bypass_attr": bypass_attribution(
                cat["cmd"], cat["exec"], cat["margins"], cat["alpha"],
                cat["bypass"], table_viol=cat["table_viol"],
                blocked_cls=cat["blocked_cls"]),
            "lock_lookahead": lock_lookahead_stats(
                cat["margins"], cat["table_viol"], cat["engaged"]),
            "calibration_regular": calibration_stats(cat["margins"],
                                                     alpha_pol),
        }
        return fid, fb, diag, extras

    def run_cell(theta_hi: "float | None", theta_lo: "float | None") -> dict:
        clutch_on = theta_hi is not None
        label = (f"hi{theta_hi}_lo{theta_lo}" if clutch_on else "reference")
        use_head = clutch_on and args.hazard_source == "head"
        cell: dict = {
            "theta_hi": theta_hi, "theta_lo": theta_lo,
            "clutch": clutch_on, "ckpt": args.ckpt,
            "env_yaml": args.env_yaml, "geometry": args.geometry,
            "num_envs": env.num_envs, "duration_s": args.duration_s,
            "steps_per_window": steps, "dt": ctrl_dt, "seed": args.seed,
            "regular_amps": args.regular_amps,
            "directed_amps": args.directed_amps,
            "bypass_mm": args.bypass_mm,
            "hazard_source": ("head" if use_head else "beta"),
            "hazard_temp": (args.hazard_temp if use_head else None),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        base = PolicyDriver(args.ckpt, device=str(env.device),
                            hazard_detector=use_head,
                            hazard_temp=args.hazard_temp)

        def make_driver():
            # fresh clutch state per window over the shared stateless actor
            return wrap_clutch(base, clutch_on, theta_hi or 0.0,
                               theta_lo or 0.0, env.num_envs, env.device)

        # metrics 1 + 4: regular stream
        reg_traces = []
        for w, amp in enumerate(args.regular_amps):
            print(f"[clutch] === {label} regular amp={amp} "
                  f"window {w + 1}/{len(args.regular_amps)} ===", flush=True)
            new_window(amp, w, "regular")
            reg_traces.append(run_traced_window(env, make_driver(), steps))
        (cell["fidelity"], cell["false_brake"], cell["diag_regular"],
         extras) = merge_fidelity(reg_traces)
        cell.update(extras)
        del reg_traces
        if args.metrics == "regular":
            return cell

        # metric 2: head-on conflict injection
        brows, dir_traces = [], []
        for w, amp in enumerate(args.directed_amps):
            print(f"[clutch] === {label} head-on amp={amp} "
                  f"window {w + 1}/{len(args.directed_amps)} ===", flush=True)
            new_window(amp, 100 + w, "directed")
            tr = run_traced_window(env, make_driver(), steps,
                                   cross_tracker=CrossArgminTracker(env._sph))
            rows = brake_rows(tr["margins"], tr["engaged"], tr["exec"],
                              tr["qd"], tr["cross_arms"], tr["table_viol"],
                              ctrl_dt)
            for r in rows:
                r["amp"] = amp
                r["window"] = w
            brows.extend(rows)
            dir_traces.append({k: tr[k] for k in
                               ("margins", "engaged", "alpha_policy")})
            del tr
        cell["brake"] = {"aggregate": brake_aggregate(brows), "rows": brows}
        dir_margins = {k: torch.cat([t["margins"][k] for t in dir_traces])
                       for k in CLASS_KEYS}
        dir_alpha_pol = torch.cat([t["alpha_policy"] for t in dir_traces])
        cell["diag_directed"] = stream_diagnostics(
            dir_margins, torch.cat([t["engaged"] for t in dir_traces]),
            dir_alpha_pol)
        cell["calibration_directed"] = calibration_stats(dir_margins,
                                                         dir_alpha_pol)
        del dir_traces

        # metric 3: scripted unlock-recovery (clutch cells only)
        if clutch_on and args.metrics == "all":
            for mode in args.recovery_modes:
                print(f"[clutch] === {label} recovery mode={mode} ===",
                      flush=True)
                new_window(args.recovery_amp, 200, "directed")
                src = RetreatOverrideSource(env._delta_src, env.num_envs,
                                            args.recovery_amp,
                                            device=env.device)
                env._delta_src = src
                driver = make_driver()
                script = RecoveryScript(driver, src, env.num_envs, mode=mode)
                tr = run_traced_window(env, driver, steps, script=script,
                                       cross_tracker=CrossArgminTracker(
                                           env._sph))
                rows = script.rows()
                # merge official violation dose per env into the rows
                stack = torch.stack([tr["margins"][k] for k in CLASS_KEYS])
                viol = stack < 0.0
                viol[CLASS_KEYS.index("table")] = tr["table_viol"].bool()
                for r in rows:
                    e = r["env_idx"]
                    r["violation_steps_any"] = int(viol.any(0)[:, e].sum())
                cell[f"recovery_{mode}"] = {
                    "aggregate": recovery_aggregate(rows), "rows": rows}
                del tr
        return cell

    for hi, lo in theta_cells:
        cell = run_cell(hi, lo)
        path = out_dir / f"cell_hi{hi}_lo{lo}.json"
        path.write_text(json.dumps(cell, indent=1))
        print(f"[clutch] wrote {path}", flush=True)
    if args.reference:
        cell = run_cell(None, None)
        path = out_dir / "reference.json"
        path.write_text(json.dumps(cell, indent=1))
        print(f"[clutch] wrote {path}", flush=True)
    print("[clutch] CELLS_DONE", flush=True)


if __name__ == "__main__":
    main()
    import os

    os._exit(0)  # skip kit close (known 15-30 min hang on this box)
