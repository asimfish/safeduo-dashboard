"""Scenario-library eval runner: DeltaSource x BaselineFilter -> step records
(parquet, ACCEPTANCE section-1 schema) + EpisodeBatch -> metrics -> main table.

Conflict-tube determination uses UNCAPPED provider margins/closing rather than
the M-capped active_pairs observation: with B's current (deliberately fat) UR
tube spheres the 32-row active set can be fully occupied by tight self rows,
starving cross rows out of the observation (documented @A in STATUS_C W2).
Evaluation must not inherit that blind spot.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from safeduo.baselines.base import BaselineFilter
from safeduo.delta._contract_stub import ARM_KEYS, CLASS_CROSS, DeltaSource
from safeduo.eval.metrics import (
    EpisodeBatch,
    MetricsConfig,
    _gate_events,
    _priority_flips,
    bootstrap_ci,
    compute_all_metrics,
)


@dataclass
class EvalConfig:
    steps: int = 200
    dt: float = 0.02
    d_soft: float = 0.05
    d_warn: float = 0.05
    seed: int = 0
    init_jitter: float = 0.05


def _conflict_tube_uncapped(provider, q: dict, qd: dict, d_soft: float) -> torch.Tensor:
    """(N,) bool from full pair tables (provider-specific fast paths)."""
    if hasattr(provider, "conflict_tube"):
        return provider.conflict_tube(q, qd, d_soft)
    rows = provider.rows_from_q(q) if hasattr(provider, "rows_from_q") else None
    if rows is None:  # toy provider path: recompute rows with gradients
        rows = provider.world.rows_from(q, qd)
    qd_f = torch.cat([qd["F_L"], qd["F_R"]], dim=-1)
    qd_u = torch.cat([qd["U_L"], qd["U_R"]], dim=-1)
    ddot = (torch.einsum("nmj,nj->nm", rows.J["F"], qd_f)
            + torch.einsum("nmj,nj->nm", rows.J["U"], qd_u))
    closing = -ddot
    hit = (rows.cls == CLASS_CROSS) & rows.valid & (rows.d < d_soft) & (closing > 0)
    return hit.any(dim=-1)


def run_eval(source: DeltaSource, filt: BaselineFilter, provider, n_envs: int,
             cfg: "EvalConfig | None" = None, method: str = "",
             delta_source_tag: str = "") -> tuple:
    """Rollout -> (EpisodeBatch, records DataFrame)."""
    import pandas as pd

    cfg = cfg or EvalConfig()
    g = torch.Generator().manual_seed(cfg.seed)
    source.reset(torch.arange(n_envs), g)
    filt.reset(torch.arange(n_envs))
    q = provider.default_q(jitter=cfg.init_jitter, generator=g)
    qd = {a: torch.zeros_like(q[a]) for a in ARM_KEYS}
    log = {k: [] for k in ("alpha", "p", "in_tube", "violation", "backstop",
                           "mm_cross", "mm_self", "mm_table")}
    cmd_log = {a: [] for a in ARM_KEYS}
    exec_log = {a: [] for a in ARM_KEYS}
    for _ in range(cfg.steps):
        state = provider.scene_state(q, qd, dt=cfg.dt)
        cmd = source.sample(state)
        out = filt.filter(state, cmd)
        mm = provider.min_margin_by_class(q)
        log["alpha"].append(out.alpha)
        log["p"].append(out.priority_p)
        log["in_tube"].append(_conflict_tube_uncapped(provider, q, qd, cfg.d_soft))
        log["violation"].append(
            (mm["cross"] < 0) | (mm["self"] < 0) | (mm["table"] < 0))
        log["backstop"].append(out.backstop_active)
        for key in ("cross", "self", "table"):
            log[f"mm_{key}"].append(mm[key])
        for a in ARM_KEYS:
            cmd_log[a].append(cmd.delta_q[a])
            exec_log[a].append(out.delta_exec[a])
        q = {a: q[a] + out.delta_exec[a] for a in ARM_KEYS}
        qd = {a: out.delta_exec[a] / cfg.dt for a in ARM_KEYS}

    batch = EpisodeBatch(
        delta_cmd={a: torch.stack(cmd_log[a]) for a in ARM_KEYS},
        delta_exec={a: torch.stack(exec_log[a]) for a in ARM_KEYS},
        alpha=torch.stack(log["alpha"]),
        priority_p=torch.stack(log["p"]),
        in_tube=torch.stack(log["in_tube"]),
        violation=torch.stack(log["violation"]),
        backstop_active=torch.stack(log["backstop"]),
        dt=cfg.dt,
    )
    records = _records_dataframe(pd, batch, log, method, delta_source_tag)
    return batch, records


def _records_dataframe(pd, batch: EpisodeBatch, log: dict, method: str,
                       tag: str):
    """Flatten to the ACCEPTANCE section-1 step_record schema (compact form)."""
    mcfg = MetricsConfig()
    T, N = batch.T, batch.N
    on_e, off_e = _gate_events(batch.alpha, mcfg.gate_on, mcfg.gate_off)
    flips = _priority_flips(batch.priority_p, mcfg.p_deadband)
    prev = torch.cat([torch.zeros_like(batch.backstop_active[:1]),
                      batch.backstop_active[:-1]], dim=0)
    bs_trig = (batch.backstop_active & ~prev).any(-1)
    t_idx, env_idx = torch.meshgrid(torch.arange(T), torch.arange(N), indexing="ij")
    cols = {
        "t": (t_idx.flatten() * batch.dt).numpy(),
        "env_id": env_idx.flatten().numpy(),
        "method": method,
        "delta_source": tag,
        "priority_p": batch.priority_p.flatten().numpy(),
        "in_conflict_tube": batch.in_tube.flatten().numpy(),
        "violation": batch.violation.flatten().numpy(),
        "backstop_trigger": bs_trig.flatten().numpy(),
        "gate_on": on_e.any(-1).flatten().numpy(),
        "gate_off": off_e.any(-1).flatten().numpy(),
        "priority_flip": flips.flatten().numpy(),
        "min_margin_cross": torch.stack(log["mm_cross"]).flatten().numpy(),
        "min_margin_self": torch.stack(log["mm_self"]).flatten().numpy(),
        "min_margin_table": torch.stack(log["mm_table"]).flatten().numpy(),
        "warn": (torch.stack(log["mm_cross"]) < mcfg.d_soft).flatten().numpy(),
    }
    for i, a in enumerate(ARM_KEYS):
        cols[f"alpha_{a}"] = batch.alpha[..., i].flatten().numpy()
        cols[f"cmd_norm_{a}"] = batch.delta_cmd[a].norm(dim=-1).flatten().numpy()
        cols[f"exec_norm_{a}"] = batch.delta_exec[a].norm(dim=-1).flatten().numpy()
        cols[f"backstop_{a}"] = batch.backstop_active[..., i].flatten().numpy()
    return pd.DataFrame(cols)


def save_records_parquet(records, path: "Path | str") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    records.to_parquet(path, index=False)


def per_episode_summaries(batch: EpisodeBatch) -> dict:
    """Per-episode arrays for bootstrap CIs in the main table."""
    interv = (1.0 - batch.alpha).mean(dim=(0, 2))          # (N,)
    dwell = batch.in_tube.float().sum(dim=0) * batch.dt    # (N,) tube seconds
    return {"intervention": interv, "tube_dwell_s": dwell}


def main_table(results: dict, out_md: "Path | str | None" = None,
               conf: float = 0.95) -> str:
    """results: method -> {"metrics": dict, "batch": EpisodeBatch}. Renders the
    C1 main comparison table with Clopper-Pearson bound and bootstrap CIs."""
    header = ("| method | collisions (UB95) | intervention [CI] | tracking RMSE | "
              "time-to-clear mean/p95 (s) | oscill./ep | proj. progress | "
              "backstop/ep | recovery (s) | stalls |")
    rows = [header, "|" + "---|" * 10]
    for m, r in results.items():
        met, batch = r["metrics"], r["batch"]
        eps = per_episode_summaries(batch)
        lo, hi = bootstrap_ci(eps["intervention"], conf=conf)
        rows.append(
            f"| {m} "
            f"| {met['collision_episodes']}/{met['episodes']} "
            f"({met['collision_rate_ub95']:.4f}) "
            f"| {met['intervention_rate']:.4f} [{lo:.4f},{hi:.4f}] "
            f"| {met['tracking_rmse']:.5f} "
            f"| {met['time_to_clear_mean']:.2f}/{met['time_to_clear_p95']:.2f} "
            f"| {met['oscillations_per_episode']:.2f} "
            f"| {met['projected_progress_conflict']:.3f} "
            f"| {met['backstop_triggers_per_episode']:.2f} "
            f"| {met['recovery_latency_mean']:.2f} "
            f"| {met['stall_events']} |")
    text = "\n".join(rows) + "\n"
    if out_md is not None:
        Path(out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(out_md).write_text(text)
    return text


def evaluate_method(source_fn, filt_fn, provider_fn, n_envs: int,
                    cfg: "EvalConfig | None" = None, method: str = "",
                    tag: str = "") -> dict:
    """Convenience: fresh source/filter/provider -> batch + records + metrics."""
    provider = provider_fn()
    batch, records = run_eval(source_fn(provider), filt_fn(provider), provider,
                              n_envs, cfg, method, tag)
    return {"batch": batch, "records": records,
            "metrics": compute_all_metrics(batch)}
