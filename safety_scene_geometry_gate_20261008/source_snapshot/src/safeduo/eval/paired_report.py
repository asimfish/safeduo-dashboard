"""Paired cross-method report layer: anticipation lead time (ROUND2 item 4).

The C1 narrative needs evidence that the learned coordinator yields BEFORE the
conflict becomes geometrically acute, while reactive filters yield after
(non-myopic coordination, FINAL_PROPOSAL claim scope). This is a PAIRED
metric: both methods replay the same scenario variants (same DeltaSource
construction + same reset seed -> env i sees the same script in both runs;
trajectories diverge only through the filters' own interventions).

Definitions (per env pair):
  t_yield    first step the method visibly yields: any-arm hysteresis
             GATE_ON edge (alpha < gate_on) OR first priority engagement
             (|p| crosses p_deadband from below).
  t_tube     first step the method's own rollout enters the conflict tube.
  lead_vs_conflict = (t_tube - t_yield) * dt   per method; positive =
             yielded before its own conflict onset (anticipation).
  paired_lead_s    = (t_yield_baseline - t_yield_ours) * dt; positive =
             ours yields earlier on the same scenario.

Censoring is explicit, never silently pooled:
  both_yield        pairs where both methods yield -> paired_lead_s stats
  conflict_avoided  method yields and never enters the tube at all (best
                    outcome; lead_vs_conflict undefined -> counted, excluded
                    from the mean)
  no_conflict       neither yield nor tube (scenario never became relevant
                    for that method)
  never_yield_in_conflict  tube entered but no yield ever registered
"""

from __future__ import annotations

import torch

from safeduo.eval.metrics import (
    EpisodeBatch,
    MetricsConfig,
    _gate_events,
    _priority_flips,
    bootstrap_ci,
)


def first_yield_step(batch: EpisodeBatch, cfg: "MetricsConfig | None" = None
                     ) -> torch.Tensor:
    """(N,) first yielding step per env; -1 = never yields.

    Yield evidence = hysteresis GATE_ON on any arm, or the priority leaving
    the deadband for the first time (initial |p| already outside the deadband
    counts as t=0 -- a policy that starts yielded has zero yield latency).
    """
    cfg = cfg or MetricsConfig()
    on_e, _ = _gate_events(batch.alpha, cfg.gate_on, cfg.gate_off)
    gate_any = on_e.any(dim=-1)                       # (T, N)
    p_engaged = batch.priority_p.abs() > cfg.p_deadband
    yielding = gate_any | p_engaged
    return _first_true_step(yielding)


def first_tube_step(batch: EpisodeBatch) -> torch.Tensor:
    """(N,) first conflict-tube step per env; -1 = never in tube."""
    return _first_true_step(batch.in_tube)


def _first_true_step(mask: torch.Tensor) -> torch.Tensor:
    T = mask.shape[0]
    idx = torch.arange(T, device=mask.device).unsqueeze(-1).expand_as(mask)
    big = torch.full_like(idx, T)
    first = torch.where(mask.bool(), idx, big).amin(dim=0)
    return torch.where(first >= T, torch.full_like(first, -1), first)


def _lead_vs_conflict(t_yield: torch.Tensor, t_tube: torch.Tensor,
                      dt: float) -> torch.Tensor:
    """(t_tube - t_yield)*dt where both defined, else nan."""
    ok = (t_yield >= 0) & (t_tube >= 0)
    lead = (t_tube - t_yield).float() * dt
    return torch.where(ok, lead, torch.full_like(lead, float("nan")))


def anticipation_report(batch_ours: EpisodeBatch, batch_base: EpisodeBatch,
                        cfg: "MetricsConfig | None" = None,
                        names: tuple = ("safeduo", "strong_cbf_qp")) -> dict:
    """Paired anticipation report. Env i in the two batches MUST correspond to
    the same scenario variant + reset seed (runner construction guarantee)."""
    cfg = cfg or MetricsConfig()
    if batch_ours.N != batch_base.N:
        raise ValueError("paired batches must have identical env counts")
    if abs(batch_ours.dt - batch_base.dt) > 1e-12:
        raise ValueError("paired batches must share dt")
    dt = batch_ours.dt
    y_a, y_b = first_yield_step(batch_ours, cfg), first_yield_step(batch_base, cfg)
    t_a, t_b = first_tube_step(batch_ours), first_tube_step(batch_base)

    both_yield = (y_a >= 0) & (y_b >= 0)
    paired_lead = (y_b - y_a).float() * dt            # + = ours earlier
    paired_lead = paired_lead[both_yield]

    out = {
        "n_pairs": int(batch_ours.N),
        "n_both_yield": int(both_yield.sum().item()),
        "paired_lead_mean_s": float(paired_lead.mean().item())
        if len(paired_lead) else float("nan"),
        "paired_lead_median_s": float(paired_lead.median().item())
        if len(paired_lead) else float("nan"),
        "paired_lead_frac_ours_earlier": float((paired_lead > 0).float()
                                               .mean().item())
        if len(paired_lead) else float("nan"),
        "paired_lead_values_s": paired_lead,
    }
    if len(paired_lead):
        lo, hi = bootstrap_ci(paired_lead, conf=cfg.conf)
        out["paired_lead_ci95_s"] = (lo, hi)
    else:
        out["paired_lead_ci95_s"] = (float("nan"), float("nan"))

    for name, y, t in ((names[0], y_a, t_a), (names[1], y_b, t_b)):
        lead = _lead_vs_conflict(y, t, dt)
        valid = ~torch.isnan(lead)
        out[f"{name}_lead_vs_conflict_mean_s"] = float(
            lead[valid].mean().item()) if valid.any() else float("nan")
        out[f"{name}_n_conflict_avoided"] = int(
            ((y >= 0) & (t < 0)).sum().item())
        out[f"{name}_n_no_conflict"] = int(((y < 0) & (t < 0)).sum().item())
        out[f"{name}_n_never_yield_in_conflict"] = int(
            ((y < 0) & (t >= 0)).sum().item())
    return out


def anticipation_table_row(report: dict, names: tuple = ("safeduo",
                                                         "strong_cbf_qp")
                           ) -> str:
    """One markdown row for the paired section of the main table."""
    lo, hi = report["paired_lead_ci95_s"]
    return (f"| {names[0]} vs {names[1]} "
            f"| {report['n_both_yield']}/{report['n_pairs']} "
            f"| {report['paired_lead_mean_s']:.3f} [{lo:.3f},{hi:.3f}] "
            f"| {report['paired_lead_median_s']:.3f} "
            f"| {report['paired_lead_frac_ours_earlier']:.2f} "
            f"| {report[f'{names[0]}_n_conflict_avoided']}"
            f"/{report[f'{names[1]}_n_conflict_avoided']} |")
