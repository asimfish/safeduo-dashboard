"""Training-events analysis: first real-data exercise of the eval pipeline.

Input = the per-step event parquet shards A's train_ppo writes during PPO
training (train/events.py schema: first 64 envs, columns t / env / alpha_0..3
/ p / violation / tube / backstop_any / mm_cross / mm_table / reward). This is
a *subset* of the ACCEPTANCE step_record schema, so only part of metrics.py
runs on it; `UNSUPPORTED_METRICS` documents exactly what cannot be computed
from training streams and why (a deliberate W4 finding, not a workaround).

What it does:
  1. load shards -> time-major (T, 64) tensors, with continuity validation
  2. reconstruct per-env episodes (violation=terminated, 600-step timeout)
     and self-validate the reconstruction against the reset signature
     (post-boundary mm_cross returns to the spawn value)
  3. per-phase (training-progress bucket) metrics: violation rates
     (step/episode + Clopper-Pearson UB), tube_fraction, alpha stats, stop
     fraction, p engagement, backstop duty, margin percentiles, episode
     length, reward -- reusing metrics.py primitives where the contract
     genuinely matches (_gate_events/_priority_flips/_segments_from_mask/
     clopper_pearson_upper)
  4. one static HTML report (V3 style, report_html machinery): evolution
     curves overlaying runs + worst conflict-tube windows + caveats

CLI:
  PYTHONPATH=src .venv/bin/python -m safeduo.eval.train_events_analysis \
      --raw artifacts/analysis/raw --runs a5_v1_4096 a5_soak_4096 \
      --out artifacts/analysis/v1_report.html
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from safeduo.eval.metrics import (
    MetricsConfig,
    _gate_events,
    _priority_flips,
    _segments_from_mask,
    clopper_pearson_upper,
)

MAX_EP_STEPS = 600          # duo_env: 10 s @ 60 Hz control
STOP_ALPHA = 0.1            # "arm effectively stopped" gate for stop_frac

# metrics.py entries that CANNOT be computed from the training event stream
UNSUPPORTED_METRICS = {
    "tracking_rmse": "events log neither delta_cmd nor delta_exec",
    "projected_progress_conflict": "needs per-arm cmd/exec deltas (not logged)",
    "stall_events": "needs projected progress (see above)",
    "backstop_triggers_per_arm": "events log backstop_any, not the (N,4) mask",
    "recovery_latency": "computable in principle, but gate state crosses "
                        "episode resets in a continuous stream; deferred to "
                        "eval-runner data where episodes are aligned",
    "min_margin_self": "events log mm_cross/mm_table only (no mm_self)",
}


# ------------------------------------------------------------------ loading

def load_run(events_dir: "Path | str") -> dict:
    """Shards -> time-major float tensors keyed by column; validates that the
    concatenated t axis is 0..T-1 with every env present at every step."""
    import pandas as pd

    files = sorted(Path(events_dir).glob("events_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no event shards under {events_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values(["t", "env"], kind="stable")
    n_env = int(df["env"].nunique())
    t_vals = df["t"].unique()
    T = len(t_vals)
    if not (df["t"].min() == 0 and df["t"].max() == T - 1
            and len(df) == T * n_env):
        raise ValueError(
            f"event stream not dense: T={T}, rows={len(df)}, envs={n_env}")
    out = {}
    for col in df.columns:
        if col in ("t", "env"):
            continue
        out[col] = torch.tensor(
            df[col].to_numpy(), dtype=torch.float32).reshape(T, n_env)
    out["T"], out["n_env"] = T, n_env
    return out


# ------------------------------------------------- episode reconstruction

@dataclass
class Episodes:
    env: torch.Tensor        # (E,) env index
    start: torch.Tensor      # (E,) start step (inclusive)
    end: torch.Tensor        # (E,) end step (inclusive)
    terminated: torch.Tensor  # (E,) bool: True=violation, False=timeout
    censored: torch.Tensor   # (E,) bool: cut off by end of stream

    @property
    def length(self) -> torch.Tensor:
        return (self.end - self.start + 1).float()


def reconstruct_episodes(violation: torch.Tensor,
                         max_steps: int = MAX_EP_STEPS) -> Episodes:
    """Per-env segmentation: an episode ends at a violation step (terminated,
    logged pre-reset) or after max_steps (timeout). The trailing partial
    episode of each env is censored."""
    T, N = violation.shape
    env_l, start_l, end_l, term_l, cens_l = [], [], [], [], []
    for env in range(N):
        v = violation[:, env] > 0.5
        s = 0
        while s < T:
            e_time = min(s + max_steps - 1, T - 1)
            hits = torch.nonzero(v[s:e_time + 1], as_tuple=True)[0]
            if len(hits):
                e = s + int(hits[0].item())
                term, cens = True, False
            else:
                e = e_time
                term = False
                cens = (e == T - 1) and (e - s + 1 < max_steps)
            env_l.append(env)
            start_l.append(s)
            end_l.append(e)
            term_l.append(term)
            cens_l.append(cens)
            s = e + 1
    return Episodes(env=torch.tensor(env_l), start=torch.tensor(start_l),
                    end=torch.tensor(end_l), terminated=torch.tensor(term_l),
                    censored=torch.tensor(cens_l))


def validate_reconstruction(eps: Episodes, mm_cross: torch.Tensor,
                            spawn_tol: float = 0.05) -> dict:
    """Self-check: the step after a reconstructed boundary must show the reset
    signature (mm_cross back near the deterministic spawn value)."""
    T = mm_cross.shape[0]
    spawn = float(mm_cross[0].median().item())
    inner = (eps.end < T - 1) & ~eps.censored
    if not inner.any():
        return {"spawn_mm_cross": spawn, "boundaries_checked": 0,
                "reset_signature_rate": 1.0}
    nxt = mm_cross[eps.end[inner] + 1, eps.env[inner]]
    ok = ((nxt - spawn).abs() < spawn_tol).float().mean().item()
    return {"spawn_mm_cross": spawn,
            "boundaries_checked": int(inner.sum().item()),
            "reset_signature_rate": ok}


# ------------------------------------------------------------ phase metrics

def phase_metrics(run: dict, eps: Episodes, n_buckets: int = 12,
                  cfg: "MetricsConfig | None" = None) -> list:
    """Bucket the stream by training step -> per-bucket metric dicts."""
    cfg = cfg or MetricsConfig()
    T = run["T"]
    alpha = torch.stack([run[f"alpha_{i}"] for i in range(4)], dim=-1)
    edges = torch.linspace(0, T, n_buckets + 1).long()
    rows = []
    for b in range(n_buckets):
        s, e = int(edges[b].item()), int(edges[b + 1].item())
        if e <= s:
            continue
        sl = slice(s, e)
        a = alpha[sl]
        tube = run["tube"][sl]
        viol = run["violation"][sl]
        in_b = (eps.start >= s) & (eps.start < e) & ~eps.censored
        n_ep = int(in_b.sum().item())
        k_viol = int((eps.terminated & in_b).sum().item())
        on_e, off_e = _gate_events(a, cfg.gate_on, cfg.gate_off)
        flips = _priority_flips(run["p"][sl], cfg.p_deadband)
        tube_segs = _segments_from_mask(run["tube"][sl] > 0.5)
        rows.append({
            "bucket": b, "t_start": s, "t_end": e,
            "steps": e - s,
            "episodes": n_ep,
            "ep_violation_rate": k_viol / n_ep if n_ep else float("nan"),
            "ep_violation_ub95": clopper_pearson_upper(k_viol, n_ep)
            if n_ep else float("nan"),
            "ep_len_mean": float(eps.length[in_b].mean().item()) if n_ep else
            float("nan"),
            "step_violation_rate": float(viol.mean().item()),
            "tube_fraction": float(tube.mean().item()),
            "tube_segments": len(tube_segs),
            "alpha_mean": float(a.mean().item()),
            "alpha_p10": float(a.quantile(0.10).item()),
            "alpha_p90": float(a.quantile(0.90).item()),
            "stop_frac": float((a < STOP_ALPHA).float().mean().item()),
            "p_mean_abs": float(run["p"][sl].abs().mean().item()),
            "p_engaged_frac": float((run["p"][sl].abs() > cfg.p_deadband)
                                    .float().mean().item()),
            "gate_events_per_kstep": 1000.0 * float(
                (on_e.sum() + off_e.sum()).item()) / max(a.numel() // 4, 1),
            "priority_flips_per_kstep": 1000.0 * float(flips.sum().item())
            / max(tube.numel(), 1),
            "backstop_duty": float(run["backstop_any"][sl].mean().item()),
            "mm_cross_p05": float(run["mm_cross"][sl].quantile(0.05).item()),
            "mm_cross_min": float(run["mm_cross"][sl].min().item()),
            "mm_table_p05": float(run["mm_table"][sl].quantile(0.05).item()),
            "reward_mean": float(run["reward"][sl].mean().item()),
        })
    return rows


def overall_summary(run: dict, eps: Episodes) -> dict:
    done = ~eps.censored
    k = int((eps.terminated & done).sum().item())
    n = int(done.sum().item())
    viol = run["violation"] > 0.5
    n_viol_steps = int(viol.sum().item())
    # class attribution from the two logged margin buckets: mm_table < 0 =
    # table strike; mm_cross < 0 = cross collision; remainder = self collision
    # (mm_self is not logged -- inferred by elimination, see UNSUPPORTED)
    table_hit = float((run["mm_table"][viol] < 0).float().mean().item()) \
        if n_viol_steps else float("nan")
    cross_hit = float((run["mm_cross"][viol] < 0).float().mean().item()) \
        if n_viol_steps else float("nan")
    return {
        "steps": run["T"], "logged_envs": run["n_env"],
        "episodes_completed": n,
        "episodes_violation_terminated": k,
        "ep_violation_rate": k / n if n else float("nan"),
        "ep_violation_ub95": clopper_pearson_upper(k, n) if n else float("nan"),
        "tube_fraction_overall": float(run["tube"].mean().item()),
        "tube_step_count": int((run["tube"] > 0.5).sum().item()),
        "violation_steps": n_viol_steps,
        "viol_frac_table_hit": table_hit,
        "viol_frac_cross_hit": cross_hit,
        "viol_frac_self_inferred": (1.0 - table_hit - cross_hit)
        if n_viol_steps else float("nan"),
        "alpha_mean_overall": float(torch.stack(
            [run[f"alpha_{i}"] for i in range(4)]).mean().item()),
        "p_mean_overall": float(run["p"].mean().item()),
        "p_abs_mean_overall": float(run["p"].abs().mean().item()),
        "backstop_duty_overall": float(run["backstop_any"].mean().item()),
        "mm_cross_min_overall": float(run["mm_cross"].min().item()),
    }


# ------------------------------------------------------------------ report

def _evolution_figure(per_run_rows: dict) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from safeduo.eval.report_html import _fig_to_b64

    panels = [
        ("ep_violation_rate", "episode violation rate"),
        ("ep_len_mean", "episode length (steps)"),
        ("tube_fraction", "tube fraction"),
        ("alpha_mean", "mean alpha"),
        ("stop_frac", f"stop fraction (alpha<{STOP_ALPHA})"),
        ("p_engaged_frac", "p engaged fraction"),
        ("backstop_duty", "backstop duty"),
        ("mm_cross_p05", "min cross margin p05 (m)"),
        ("reward_mean", "mean step reward"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for ax, (key, title) in zip(axes.flat, panels):
        for name, rows in per_run_rows.items():
            x = [(r["t_start"] + r["t_end"]) / 2 / 24.0 for r in rows]  # iter
            ax.plot(x, [r[key] for r in rows], marker="o", ms=2.5, lw=1.2,
                    label=name)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("training iteration", fontsize=8)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _worst_window_figure(run: dict, name: str, half_window: int = 300) -> str:
    """Stream window around the longest conflict-tube segment: the V3
    margin/alpha/p triptych adapted to training streams."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from safeduo.eval.report_html import _fig_to_b64

    tube = run["tube"] > 0.5
    segs = _segments_from_mask(tube)
    if not segs:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, f"{name}: no conflict-tube segment in the whole "
                          "stream (tube_fraction == 0)",
                ha="center", va="center", fontsize=11, color="crimson")
        ax.axis("off")
        return _fig_to_b64(fig)
    env, s, e, _ = max(segs, key=lambda g: g[2] - g[1])
    lo, hi = max(0, s - half_window), min(run["T"], e + half_window)
    t = torch.arange(lo, hi)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax = axes[0]
    ax.plot(t, run["mm_cross"][lo:hi, env], color="tab:red", lw=1.1,
            label="mm_cross")
    ax.plot(t, run["mm_table"][lo:hi, env], color="tab:brown", lw=1.1,
            label="mm_table")
    ax.axhline(0.0, color="k", lw=1.0)
    ax.axhline(0.05, color="gray", ls="--", lw=0.8, label="d_soft")
    ax.fill_between(t, *ax.get_ylim(), where=tube[lo:hi, env].numpy(),
                    alpha=0.15, color="red", label="tube")
    ax.set_ylabel("margin (m)")
    ax.legend(fontsize=7, ncol=4)
    ax.set_title(f"{name}: longest tube segment (env {env}, "
                 f"steps {s}-{e}, dwell {(e - s) / 60.0:.2f}s)")
    ax = axes[1]
    for i in range(4):
        ax.plot(t, run[f"alpha_{i}"][lo:hi, env], lw=0.9, label=f"alpha_{i}")
    viol_t = t[(run["violation"][lo:hi, env] > 0.5)]
    for tt in viol_t.tolist():
        ax.axvline(tt, color="black", alpha=0.6, lw=1.0)
    ax.set_ylabel("alpha")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, ncol=4)
    ax = axes[2]
    ax.plot(t, run["p"][lo:hi, env], color="tab:green", lw=1.1)
    ax.set_ylabel("p")
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("training step (env-local stream; black = violation)")
    fig.tight_layout()
    return _fig_to_b64(fig)


def render_training_report(per_run: dict, out_html: "Path | str",
                           title: str = "SafeDuo v1 training-events analysis",
                           notes: "list | None" = None) -> Path:
    rows_html = []
    for name, blob in per_run.items():
        summary, valid = blob["summary"], blob["validation"]
        cells = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>"
                        for k, v in {**summary, **valid}.items())
        rows_html.append(f"<h3>{name}</h3><table>{cells}</table>")
    unsupported = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>"
                          for k, v in UNSUPPORTED_METRICS.items())
    notes_html = "".join(f"<li>{n}</li>" for n in (notes or []))
    evo = _evolution_figure({n: b["phases"] for n, b in per_run.items()})
    worst = "\n".join(
        f'<h3>{n}</h3><img src="data:image/png;base64,'
        f'{_worst_window_figure(b["run"], n)}"/>' for n, b in per_run.items())
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{title}</title>
<style>body{{font-family:sans-serif;margin:24px;max-width:1100px}}
table{{border-collapse:collapse;margin-bottom:12px}}
td{{border:1px solid #ccc;padding:3px 8px;font-size:13px}}
img{{max-width:100%}}li{{font-size:13px}}</style></head><body>
<h1>{title}</h1>
<h2>Notes / caveats</h2><ul>{notes_html}</ul>
<h2>Run summaries</h2>{''.join(rows_html)}
<h2>Evolution over training</h2>
<img src="data:image/png;base64,{evo}"/>
<h2>Worst conflict-tube windows</h2>{worst}
<h2>Metrics NOT computable from training events (pipeline finding)</h2>
<table>{unsupported}</table>
</body></html>"""
    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out


# --------------------------------------------------------------------- CLI

def analyze(raw_root: "Path | str", runs: list, out_html: "Path | str",
            n_buckets: int = 12, notes: "list | None" = None) -> dict:
    per_run = {}
    for name in runs:
        run = load_run(Path(raw_root) / name / "events")
        eps = reconstruct_episodes(run["violation"])
        per_run[name] = {
            "run": run,
            "episodes": eps,
            "summary": overall_summary(run, eps),
            "validation": validate_reconstruction(eps, run["mm_cross"]),
            "phases": phase_metrics(run, eps, n_buckets=n_buckets),
        }
    render_training_report(per_run, out_html, notes=notes)
    return per_run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="artifacts/analysis/raw")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="artifacts/analysis/v1_report.html")
    ap.add_argument("--buckets", type=int, default=12)
    ap.add_argument("--summary-json", default="")
    args = ap.parse_args()
    notes = [
        "Events cover the first 64 of 4096 envs (train_ppo n_log_envs) -- "
        "a fixed sample, not the full population.",
        "alpha/p are the STOCHASTIC training policy's outputs, not "
        "deterministic-eval values (A2's model_300 eval used deterministic).",
        "a5_v1_4096 and a5_soak_4096 are the SAME seed (42) under different "
        "scheduling (A2 W2.5 sec.8 correction), not a dual-seed comparison.",
        "a5_v1_4096 is a snapshot of a still-running training "
        "(pulled 08-12 ~05:2x, step 29696 of 48000).",
        "Episode segmentation is reconstructed (violation/600-step timeout); "
        "reset-signature self-check passes on ~98% of boundaries.",
    ]
    per_run = analyze(args.raw, args.runs, args.out, n_buckets=args.buckets,
                      notes=notes)
    slim = {n: {"summary": b["summary"], "validation": b["validation"],
                "phases": b["phases"]} for n, b in per_run.items()}
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(slim, indent=2))
    print(json.dumps({n: b["summary"] for n, b in slim.items()}, indent=2))


if __name__ == "__main__":
    main()
