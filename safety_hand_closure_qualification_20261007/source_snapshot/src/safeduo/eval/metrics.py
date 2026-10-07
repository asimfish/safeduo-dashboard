"""SafeDuo evaluation metrics -- exact implementation of ACCEPTANCE_AND_VIZ.md section 3.

Metric table (spec -> function):
  collision count      0/N episode ratio + Clopper-Pearson 95% upper bound
                       -> collision_stats
  intervention rate    mean(1 - alpha) over steps/arms   -> intervention_rate
  tracking RMSE        RMS of ||executed - commanded||, overall / per arm /
                       outside-conflict                   -> tracking_rmse
  time-to-clear        duration from conflict-tube entry to exit
                       -> conflict_segments / time_to_clear
  oscillation count    (alpha hysteresis crossings + priority flips) per episode
                       -> oscillation_count
  projected progress   inside the tube: <exec, cmd>/||cmd|| accumulated over
                       ||cmd||, i.e. fraction of commanded motion that survived
                       in the commanded direction          -> projected_progress
  backstop frequency   BACKSTOP_TRIGGER rising edges per episode and per second
                       -> backstop_stats
  recovery latency     GATE_ON -> GATE_OFF duration        -> recovery_latency
  stall events         tube dwell > T_stall with ~zero progress -> stall_events

Conflict tube (round-1 review item 7): a cross-robot sphere pair exists with
margin < d_soft AND the pair is approaching -> conflict_tube_mask. NOTE the
contract convention (safety/types.py): closing_vel is POSITIVE when
approaching, so "approaching" means closing_vel > 0.

Conventions:
  time-major tensors, T = steps, N = envs (episodes), A = 4 arms.
  Segments still open at episode end are censored and reported separately,
  never silently mixed into the cleared-time statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from safeduo.delta._contract_stub import ARM_KEYS, CLASS_CROSS


@dataclass
class MetricsConfig:
    d_soft: float = 0.08        # conflict-tube margin threshold (m)
    gate_on: float = 0.9        # GATE_ON when alpha drops below
    gate_off: float = 0.95      # GATE_OFF when alpha recovers above (hysteresis)
    p_deadband: float = 0.1     # priority flip counted when p crosses +-deadband
    t_stall: float = 2.0        # STALL when tube dwell exceeds this (s)
    stall_progress: float = 0.05  # ...with projected progress below this fraction
    conf: float = 0.95          # Clopper-Pearson confidence


@dataclass
class EpisodeBatch:
    """Time-major recording of one eval batch (T steps, N envs)."""
    delta_cmd: dict               # arm -> (T, N, dof)
    delta_exec: dict              # arm -> (T, N, dof)
    alpha: torch.Tensor           # (T, N, 4)
    priority_p: torch.Tensor      # (T, N)
    in_tube: torch.Tensor         # (T, N) bool
    violation: torch.Tensor       # (T, N) bool
    backstop_active: torch.Tensor  # (T, N, 4) bool
    dt: float
    extras: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return self.alpha.shape[0]

    @property
    def N(self) -> int:
        return self.alpha.shape[1]


# ---------------------------------------------------------------- conflict tube

def conflict_tube_mask(active_pairs: torch.Tensor, d_soft: float,
                       cross_class: float = CLASS_CROSS,
                       active_mask: "torch.Tensor | None" = None) -> torch.Tensor:
    """(..., M, 4) active-pair set -> (...,) bool: any valid cross-robot row
    with margin < d_soft that is approaching (closing_vel > 0 per types.py).
    Zero-filled padding rows are inert (closing_vel 0 is not > 0), so
    active_mask is optional. Works on (N, M, 4) or (T, N, M, 4)."""
    dist, cvel, cls = (active_pairs[..., 0], active_pairs[..., 1],
                       active_pairs[..., 2])
    hit = (cls == cross_class) & (dist < d_soft) & (cvel > 0.0)
    if active_mask is not None:
        hit = hit & active_mask
    return hit.any(dim=-1)


# ------------------------------------------------------------------- collisions

def clopper_pearson_upper(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided exact binomial upper bound for k failures in n trials."""
    from scipy.stats import beta

    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta.ppf(conf, k + 1, n - k))


def clopper_pearson_interval(k: int, n: int, conf: float = 0.95) -> tuple:
    """Two-sided exact (Clopper-Pearson) binomial CI for k events in n trials.

    This is the interval the teleop-safety eval spec (2026-08-13, section 2B)
    asks to report next to every x/N success rate: at 100/100 passes the
    97.5% upper tail on the failure rate is ~3.6% (the spec's reference
    number), vs ~3.0% for the one-sided bound above. Tail split is
    (1-conf)/2 per side; k=0 pins the lower bound at 0, k=n pins the upper
    bound at 1 (standard CP convention).
    """
    from scipy.stats import beta

    if n <= 0:
        return (0.0, 1.0)
    a = (1.0 - conf) / 2.0
    lo = 0.0 if k <= 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k >= n else float(beta.ppf(1.0 - a, k + 1, n - k))
    return (lo, hi)


def collision_stats(batch: EpisodeBatch, cfg: MetricsConfig) -> dict:
    ep_viol = batch.violation.any(dim=0)
    k, n = int(ep_viol.sum().item()), batch.N
    return {
        "collision_episodes": k,
        "episodes": n,
        "collision_rate": k / max(n, 1),
        "collision_rate_ub95": clopper_pearson_upper(k, n, cfg.conf),
    }


# ----------------------------------------- damaging / near-contact split (G1)

MARGIN_CLASSES = ("cross", "self_f", "self_u", "table")


@dataclass
class CollisionSplitConfig:
    """G1 Round-46 arbitration: a DAMAGING collision is a mesh-level event --
    sphere-layer penetration beyond the conservative inflation band, OR a
    sustained contact force over threshold. Shallow in-band grazes are
    NEAR-CONTACT (reported as their own main-table column, not a gate item).

    band_m       inflation-band edge (A4 knife: SafeDuo's 50 collisions all
                 <= 3.27 mm, CBF/passthrough carry 12-13 mm tails; 5 mm is
                 the conservative-sphere band boundary)
    force_n      sustained-contact-force threshold, N (G1 initial value 20 N;
                 calibration owned by the A/B lines before the G1 gate)
    force_steps  consecutive steps the force must exceed force_n
    """
    band_m: float = 0.005
    force_n: float = 20.0
    force_steps: int = 2
    conf: float = 0.95


def sustained_over(x: torch.Tensor, thresh: float, steps: int) -> torch.Tensor:
    """(T, N) trace -> (N,) bool: any run of >= `steps` CONSECUTIVE entries
    strictly above `thresh` (the "sustained force" criterion shape)."""
    over = x > thresh
    if steps <= 1:
        return over.any(dim=0)
    run = torch.zeros(x.shape[1], dtype=torch.long, device=x.device)
    hit = torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)
    for t in range(x.shape[0]):
        run = (run + 1) * over[t].long()
        hit |= run >= steps
    return hit


def episode_max_depth(margins: dict, valid: "torch.Tensor | None" = None) -> dict:
    """Per-episode deepest sphere-layer penetration (m), per class + overall.

    margins: class name -> (T, N) min-margin trace (episode-major, i.e. one
    column per episode as produced by chop_episodes / the offline re-cut).
    Padding rows must be inert; pass `valid` when padding margins could be
    negative (harness pads +1.0 so it never is, but the mask keeps the
    semantics explicit)."""
    out = {}
    stack = []
    for k in MARGIN_CLASSES:
        if k not in margins:
            continue
        depth = (-margins[k]).clamp_min(0.0)
        if valid is not None:
            depth = depth * valid.float()
        out[k] = depth.max(dim=0).values
        stack.append(out[k])
    if not stack:
        raise ValueError("no margin traces given")
    all_cls = torch.stack(stack)                     # (C, N)
    out["overall"] = all_cls.max(dim=0).values
    present = [k for k in MARGIN_CLASSES if k in margins]
    out["argmax_class"] = [present[i] for i in all_cls.argmax(dim=0).tolist()]
    return out


def classify_collision_severity(max_depth: torch.Tensor,
                                violation_any: torch.Tensor,
                                cfg: "CollisionSplitConfig | None" = None,
                                sustained_force: "torch.Tensor | None" = None
                                ) -> tuple:
    """-> (damaging (N,) bool, near_contact (N,) bool).

    damaging     violation AND (depth STRICTLY > band  OR  sustained force)
    near_contact violation AND NOT damaging (in-band shallow graze)
    Depth exactly at the band edge is near-contact (strict >, pinned in
    tests). Without force data the split is depth-only, which can only be
    conservative in the near-contact direction (a hypothetical in-band
    sustained-force episode would be under-classified, never over)."""
    cfg = cfg or CollisionSplitConfig()
    over_band = max_depth > cfg.band_m
    if sustained_force is not None:
        over_band = over_band | sustained_force
    damaging = violation_any & over_band
    return damaging, violation_any & ~damaging


def collision_split_stats(margins: dict, violation: torch.Tensor,
                          cfg: "CollisionSplitConfig | None" = None,
                          valid: "torch.Tensor | None" = None,
                          contact_force: "torch.Tensor | None" = None) -> dict:
    """Full G1-split report from episode-major traces.

    margins: class -> (T, N) min margins; violation: (T, N) bool;
    contact_force: optional (T, N) max contact force (records to date do NOT
    carry it -- the force criterion then reports inactive)."""
    cfg = cfg or CollisionSplitConfig()
    viol_any = violation.any(dim=0)
    depth = episode_max_depth(margins, valid=valid)
    sustained = None
    if contact_force is not None:
        sustained = sustained_over(contact_force, cfg.force_n, cfg.force_steps)
    damaging, near = classify_collision_severity(
        depth["overall"], viol_any, cfg, sustained_force=sustained)
    n = int(viol_any.numel())
    k_dmg, k_near = int(damaging.sum().item()), int(near.sum().item())
    d_mm = depth["overall"][viol_any] * 1e3
    cls_of = [c for c, v in zip(depth["argmax_class"], viol_any.tolist()) if v]
    dmg_cls = [c for c, v in zip(depth["argmax_class"], damaging.tolist()) if v]
    dmg_mm = sorted((depth["overall"][damaging] * 1e3).tolist(), reverse=True)
    out = {
        "episodes": n,
        "damaging_episodes": k_dmg,
        "damaging_rate": k_dmg / max(n, 1),
        "damaging_rate_ub95": clopper_pearson_upper(k_dmg, n, cfg.conf),
        "near_contact_episodes": k_near,
        "near_contact_rate": k_near / max(n, 1),
        "near_contact_rate_ub95": clopper_pearson_upper(k_near, n, cfg.conf),
        "band_mm": cfg.band_m * 1e3,
        "force_criterion_active": contact_force is not None,
        "collision_class_counts": {c: cls_of.count(c) for c in sorted(set(cls_of))},
        "damaging_class_counts": {c: dmg_cls.count(c) for c in sorted(set(dmg_cls))},
        "damaging_depths_mm": [round(v, 3) for v in dmg_mm],
    }
    if d_mm.numel():
        out.update({
            "collision_depth_mm_p50": float(d_mm.quantile(0.5).item()),
            "collision_depth_mm_p90": float(d_mm.quantile(0.9).item()),
            "collision_depth_mm_max": float(d_mm.max().item()),
            "collision_depth_frac_lt2mm": float((d_mm < 2.0).float().mean().item()),
            "collision_depth_frac_gt_band": float(
                (d_mm > cfg.band_m * 1e3).float().mean().item()),
        })
    else:
        out.update({"collision_depth_mm_p50": 0.0, "collision_depth_mm_p90": 0.0,
                    "collision_depth_mm_max": 0.0,
                    "collision_depth_frac_lt2mm": 0.0,
                    "collision_depth_frac_gt_band": 0.0})
    return out


# ----------------------------------------------------------------- intervention

def intervention_rate(batch: EpisodeBatch) -> dict:
    one_minus = 1.0 - batch.alpha
    per_arm = one_minus.mean(dim=(0, 1))
    return {
        "intervention_rate": float(one_minus.mean().item()),
        **{f"intervention_rate_{a}": float(per_arm[i].item())
           for i, a in enumerate(ARM_KEYS)},
    }


def tracking_rmse(batch: EpisodeBatch) -> dict:
    out, sq_all, cnt_all = {}, 0.0, 0
    sq_free, cnt_free = 0.0, 0
    free = ~batch.in_tube  # (T, N)
    for i, a in enumerate(ARM_KEYS):
        err = batch.delta_exec[a] - batch.delta_cmd[a]
        sq = (err * err).sum(dim=-1)                      # (T, N)
        out[f"tracking_rmse_{a}"] = float(sq.mean().sqrt().item())
        sq_all += sq.sum().item()
        cnt_all += sq.numel()
        sq_free += (sq * free.float()).sum().item()
        cnt_free += int(free.sum().item())
    out["tracking_rmse"] = (sq_all / max(cnt_all, 1)) ** 0.5
    out["tracking_rmse_outside_conflict"] = (sq_free / max(cnt_free, 1)) ** 0.5
    return out


# ------------------------------------------------------------ conflict segments

def _segments_from_mask(mask: torch.Tensor) -> list:
    """(T, N) bool -> list of (env, t_start, t_end_exclusive, censored)."""
    T, N = mask.shape
    m = mask.to(torch.int8)
    pad = torch.zeros(1, N, dtype=torch.int8, device=mask.device)
    diff = torch.cat([m, pad], dim=0) - torch.cat([pad, m], dim=0)  # (T+1, N)
    segs = []
    for env in range(N):
        starts = (diff[:, env] == 1).nonzero(as_tuple=True)[0].tolist()
        ends = (diff[:, env] == -1).nonzero(as_tuple=True)[0].tolist()
        for s, e in zip(starts, ends):
            segs.append((env, s, e, e >= T))
    return segs


def conflict_segments(batch: EpisodeBatch) -> list:
    return _segments_from_mask(batch.in_tube)


def time_to_clear(batch: EpisodeBatch) -> dict:
    segs = conflict_segments(batch)
    cleared = [(e - s) * batch.dt for _, s, e, cens in segs if not cens]
    censored = [(e - s) * batch.dt for _, s, e, cens in segs if cens]
    t = torch.tensor(cleared, dtype=torch.float64) if cleared else torch.zeros(0)
    return {
        "n_conflicts": len(segs),
        "n_conflicts_cleared": len(cleared),
        "n_conflicts_censored": len(censored),
        "time_to_clear_mean": float(t.mean().item()) if len(t) else 0.0,
        "time_to_clear_median": float(t.median().item()) if len(t) else 0.0,
        "time_to_clear_p95": float(t.quantile(0.95).item()) if len(t) else 0.0,
        "time_censored_mean": (sum(censored) / len(censored)) if censored else 0.0,
    }


# ------------------------------------------------------------------ oscillation

def _gate_events(alpha: torch.Tensor, gate_on: float, gate_off: float):
    """Hysteresis gate state per (T, N, A) alpha; returns (on_edges, off_edges)
    as (T, N, A) bool. Gate turns ON when alpha < gate_on while off, OFF when
    alpha > gate_off while on."""
    T = alpha.shape[0]
    gated = torch.zeros_like(alpha[0], dtype=torch.bool)
    on_edges, off_edges = [], []
    for t in range(T):
        turn_on = (alpha[t] < gate_on) & ~gated
        turn_off = (alpha[t] > gate_off) & gated
        gated = (gated | turn_on) & ~turn_off
        on_edges.append(turn_on)
        off_edges.append(turn_off)
    return torch.stack(on_edges), torch.stack(off_edges)


def _priority_flips(p: torch.Tensor, deadband: float) -> torch.Tensor:
    """(T, N) priority trace -> (T, N) flip events: sign changes between
    excursions beyond +-deadband."""
    T, N = p.shape
    last_sign = torch.zeros(N, dtype=torch.long, device=p.device)
    flips = torch.zeros(T, N, dtype=torch.bool, device=p.device)
    for t in range(T):
        s = torch.where(p[t] > deadband, 1, torch.where(p[t] < -deadband, -1, 0))
        flips[t] = (s != 0) & (last_sign != 0) & (s != last_sign)
        last_sign = torch.where(s != 0, s, last_sign)
    return flips


def oscillation_count(batch: EpisodeBatch, cfg: MetricsConfig) -> dict:
    on_e, off_e = _gate_events(batch.alpha, cfg.gate_on, cfg.gate_off)
    flips = _priority_flips(batch.priority_p, cfg.p_deadband)
    per_ep = (on_e.sum(dim=(0, 2)) + off_e.sum(dim=(0, 2))
              + flips.sum(dim=0)).float()
    return {
        "oscillations_per_episode": float(per_ep.mean().item()),
        "gate_on_per_episode": float(on_e.sum(dim=(0, 2)).float().mean().item()),
        "priority_flips_per_episode": float(flips.sum(dim=0).float().mean().item()),
    }


# ------------------------------------------------------- progress under conflict

def projected_progress(batch: EpisodeBatch) -> dict:
    """Inside the tube: sum_t <exec, cmd>/||cmd||  /  sum_t ||cmd||, aggregated
    over all arms; 1.0 = full commanded motion survived, 0 = fully blocked."""
    tube = batch.in_tube.float()
    num, den = 0.0, 0.0
    for a in ARM_KEYS:
        cmd, ex = batch.delta_cmd[a], batch.delta_exec[a]
        cn = cmd.norm(dim=-1).clamp_min(1e-12)             # (T, N)
        proj = (ex * cmd).sum(dim=-1) / cn
        num += (proj * tube).sum().item()
        den += (cn * tube).sum().item()
    return {"projected_progress_conflict": num / den if den > 0 else 1.0}


def _per_segment_progress(batch: EpisodeBatch, seg) -> float:
    env, s, e, _ = seg
    num, den = 0.0, 0.0
    for a in ARM_KEYS:
        cmd = batch.delta_cmd[a][s:e, env]
        ex = batch.delta_exec[a][s:e, env]
        cn = cmd.norm(dim=-1).clamp_min(1e-12)
        num += ((ex * cmd).sum(dim=-1) / cn).sum().item()
        den += cn.sum().item()
    return num / den if den > 0 else 1.0


# ---------------------------------------------------------------------- backstop

def backstop_stats(batch: EpisodeBatch) -> dict:
    act = batch.backstop_active
    prev = torch.cat([torch.zeros_like(act[:1]), act[:-1]], dim=0)
    rising = act & ~prev
    per_ep = rising.sum(dim=(0, 2)).double()
    total_s = batch.T * batch.dt
    return {
        "backstop_triggers_per_episode": float(per_ep.mean().item()),
        "backstop_triggers_per_second": float((per_ep / total_s).mean().item()),
        "backstop_duty_cycle": float(act.double().mean().item()),
    }


def recovery_latency(batch: EpisodeBatch, cfg: MetricsConfig) -> dict:
    """Mean GATE_ON -> GATE_OFF duration per (env, arm) gate episode."""
    on_e, off_e = _gate_events(batch.alpha, cfg.gate_on, cfg.gate_off)
    durs = []
    T = batch.T
    open_t = torch.full_like(on_e[0], -1, dtype=torch.long)
    for t in range(T):
        open_t = torch.where(on_e[t] & (open_t < 0),
                             torch.full_like(open_t, t), open_t)
        closing = off_e[t] & (open_t >= 0)
        if closing.any():
            durs.extend(((t - open_t[closing]).double() * batch.dt).tolist())
            open_t = torch.where(closing, torch.full_like(open_t, -1), open_t)
    n_open = int((open_t >= 0).sum().item())
    return {
        "recovery_latency_mean": (sum(durs) / len(durs)) if durs else 0.0,
        "recovery_latency_p95": (float(torch.tensor(durs, dtype=torch.float64)
                                       .quantile(0.95).item()) if durs else 0.0),
        "gates_unrecovered": n_open,
    }


# ------------------------------------------------------------------------ stalls

def stall_events(batch: EpisodeBatch, cfg: MetricsConfig) -> dict:
    """STALL = conflict segment longer than t_stall whose projected progress is
    below stall_progress (liveness failure: stuck in the tube, nobody moving)."""
    segs = conflict_segments(batch)
    n_stall = 0
    for seg in segs:
        env, s, e, _ = seg
        if (e - s) * batch.dt <= cfg.t_stall:
            continue
        if _per_segment_progress(batch, seg) < cfg.stall_progress:
            n_stall += 1
    return {"stall_events": n_stall,
            "stall_events_per_episode": n_stall / max(batch.N, 1)}


# -------------------------------------------------------------------- aggregate

def bootstrap_ci(values: torch.Tensor, n_boot: int = 2000, conf: float = 0.95,
                 seed: int = 0) -> tuple:
    """Percentile bootstrap CI of the mean of per-episode values."""
    if values.numel() == 0:
        return (0.0, 0.0)
    g = torch.Generator().manual_seed(seed)
    n = values.numel()
    idx = torch.randint(n, (n_boot, n), generator=g)
    means = values.flatten()[idx].float().mean(dim=1)
    lo = (1.0 - conf) / 2.0
    return (float(means.quantile(lo).item()), float(means.quantile(1.0 - lo).item()))


def compute_all_metrics(batch: EpisodeBatch, cfg: "MetricsConfig | None" = None) -> dict:
    cfg = cfg or MetricsConfig()
    out = {}
    out.update(collision_stats(batch, cfg))
    out.update(intervention_rate(batch))
    out.update(tracking_rmse(batch))
    out.update(time_to_clear(batch))
    out.update(oscillation_count(batch, cfg))
    out.update(projected_progress(batch))
    out.update(backstop_stats(batch))
    out.update(recovery_latency(batch, cfg))
    out.update(stall_events(batch, cfg))
    return out
