"""Anticipation narrative material (PAPER_NOTES Fig.4 feed, Round-38 G0 door
by-product): turn A3's parity v3.1 alarm follow-through statistics into
paper-quality figures and join them with the behavioral lead-time metric from
eval/paired_report.py.

Two levels of "anticipation", one story:
  L2 safety layer   sphere-shell alarms fire BEFORE contact force develops
                    (parity v3.1 K-step force-tracking window: every alarm is
                    either an instant TP, an early TP that led the force by
                    1..K control steps, or a sustained tangential near-track
                    -- the recall=1 design point's signature, 85.3% on the
                    rollout distribution; NOT a defect, see STATUS Round 38).
  L1 coordinator    the learned policy yields BEFORE its own conflict-tube
                    onset (paired_report.lead_vs_conflict > 0) and earlier
                    than the reactive strong CBF-QP on the same scenarios
                    (paired_lead_s > 0).

Inputs:
  --parity   artifacts/parity/a3_contact_20260812_v3_rollout_eps1.json
             (stats.ft + stats.alarm_delay_hist_steps)
  --paired   optional anticipation_paired.json exported by
             eval/block1_harness.py (adds the behavioral panel)
Outputs (default artifacts/analysis/anticipation/):
  alarm_lead_time.png/.pdf   figure (1-2 panels)
  anticipation_summary.json  pinned numbers with provenance

dt note: the parity rollout logs at the duo_env control rate (sim dt 1/120 x
decimation 2 = 60 Hz); --dt overrides if the protocol changes.
"""

from __future__ import annotations

import json
from pathlib import Path

# palette: Okabe-Ito (colorblind-safe)
C_TP_INSTANT = "#0072B2"
C_TP_EARLY = "#009E73"
C_OURS = "#0072B2"


def parse_parity_ft(report: dict, dt: float) -> dict:
    """Extract the v3.1 follow-through block into plot-ready numbers."""
    stats = report["stats"]
    ft = stats["ft"]
    hist = list(stats["alarm_delay_hist_steps"])
    n_confirmed = ft["tp_instant"] + ft["tp_early"]
    assert sum(hist) == n_confirmed, \
        "alarm_delay_hist must cover exactly the confirmed alarms"
    lead_steps = []
    for step, count in enumerate(hist):
        lead_steps.extend([step] * int(count))
    out = {
        "alarms": ft["alarms"],
        "tp_instant": ft["tp_instant"],
        "tp_early": ft["tp_early"],
        "fp_confirmed": ft["fp_confirmed"],
        "censored_by_reset": ft.get("censored_by_reset", 0),
        "false_rate_followthrough": stats.get("false_rate_followthrough"),
        "window_steps": len(hist) - 1,
        "lead_steps": lead_steps,
        "lead_ms": [s * dt * 1e3 for s in lead_steps],
        "dt": dt,
        "protocol": report.get("protocol", {}),
        "date": report.get("date"),
    }
    return out


def lead_cdf(lead_ms: list) -> tuple:
    """Empirical CDF over confirmed-alarm lead times; returns (x_ms, F)."""
    xs = sorted(lead_ms)
    n = len(xs)
    return xs, [(i + 1) / n for i in range(n)]


def plot_alarm_material(ft: dict, paired: "dict | None" = None,
                        out_base: "Path | str" = "alarm_lead_time") -> list:
    """Render the figure; returns the list of written paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = 2 if paired else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 3.2))
    if n_panels == 1:
        axes = [axes]

    # panel A: safety-layer alarm -> contact-force lead time
    ax = axes[0]
    ms = ft["lead_ms"]
    win_ms = ft["window_steps"] * ft["dt"] * 1e3
    if ms:
        import numpy as np

        step_ms = ft["dt"] * 1e3
        bins = np.arange(-0.5, ft["window_steps"] + 1.5) * step_ms
        ax.hist(ms, bins=bins, color=C_TP_INSTANT, alpha=0.75,
                edgecolor="white", label="confirmed alarms")
        xs, F = lead_cdf(ms)
        ax2 = ax.twinx()
        ax2.step([0] + xs, [0] + F, where="post", color=C_TP_EARLY, lw=2,
                 label="CDF")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("CDF", color=C_TP_EARLY)
    ax.set_xlabel("alarm lead over contact onset (ms)")
    ax.set_ylabel("alarms")
    n_conf = ft["tp_instant"] + ft["tp_early"]
    n_resolved = n_conf + ft["fp_confirmed"]
    fp_frac = (ft["false_rate_followthrough"]
               if ft.get("false_rate_followthrough") is not None
               else ft["fp_confirmed"] / max(n_resolved, 1))
    ax.set_title(
        f"L2 alarm lead time (n={n_conf} confirmed; "
        f"{ft['fp_confirmed']}/{n_resolved} sustained near-track "
        f"= {100 * fp_frac:.1f}%, window {win_ms:.0f} ms)", fontsize=8)

    # panel B: behavioral (coordinator) lead, when paired data exists
    if paired:
        ax = axes[1]
        vals = paired.get("paired_lead_values_s") or []
        if vals:
            xs, F = lead_cdf([v * 1e3 for v in vals])
            ax.step(xs, F, where="post", color=C_OURS, lw=2,
                    label="paired lead (ours - baseline)")
            ax.axvline(0.0, color="gray", lw=0.8, ls="--")
        note = []
        for key in list(paired.keys()):
            if key.endswith("_lead_vs_conflict_mean_s"):
                method = key.replace("_lead_vs_conflict_mean_s", "")
                v = paired[key]
                if v == v:  # not-nan
                    note.append(f"{method}: yield {v * 1e3:+.0f} ms before tube")
        ax.set_xlabel("paired yield-lead vs baseline (ms)")
        ax.set_ylabel("CDF")
        ax.set_title("coordinator anticipation (paired, episode-1)\n"
                     + "; ".join(note), fontsize=8)

    fig.tight_layout()
    written = []
    for ext in ("png", "pdf"):
        p = Path(f"{out_base}.{ext}")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200 if ext == "png" else None,
                    bbox_inches="tight")
        written.append(str(p))
    plt.close(fig)
    return written


def build_summary(ft: dict, paired: "dict | None", parity_path: str,
                  paired_path: str = "") -> dict:
    n_conf = ft["tp_instant"] + ft["tp_early"]
    ms = ft["lead_ms"]
    summary = {
        "source_parity": parity_path,
        "source_paired": paired_path or None,
        "protocol": ft["protocol"],
        "dt_s": ft["dt"],
        "alarms_total": ft["alarms"],
        "alarms_confirmed": n_conf,
        "alarms_sustained_near_track": ft["fp_confirmed"],
        "sustained_near_track_frac": ft["fp_confirmed"] / max(ft["alarms"], 1),
        "false_rate_followthrough": ft["false_rate_followthrough"],
        "lead_ms_values": ms,
        "lead_ms_mean": sum(ms) / len(ms) if ms else None,
        "lead_ms_max": max(ms) if ms else None,
        "confirmed_led_frac": (ft["tp_early"] / n_conf) if n_conf else None,
        "narrative": (
            "Every judged contact was preceded or met by a sphere-layer alarm "
            "(chargeable misses 0); alarms without follow-through force are "
            "sustained tangential near-tracks -- the recall=1 shell's "
            "signature, priced in G1 as intervention cost, not as detector "
            "error (STATUS Round 38 arbitration)."),
        "caveat": (
            "Confirmed-alarm sample is small (violations are razor-rare in "
            "the filtered regime); regenerate from the Block-1 main-run "
            "parity report for the camera-ready figure."),
    }
    if paired:
        summary["paired"] = {k: v for k, v in paired.items()
                             if not isinstance(v, (list, dict)) or k == "paired_lead_ci95_s"}
    return summary


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parity", default="artifacts/parity/"
                    "a3_contact_20260812_v3_rollout_eps1.json")
    ap.add_argument("--paired", default="",
                    help="anticipation_paired.json from block1_harness")
    ap.add_argument("--dt", type=float, default=1.0 / 60.0,
                    help="control dt of the parity rollout (60 Hz duo_env)")
    ap.add_argument("--out", default="artifacts/analysis/anticipation")
    args = ap.parse_args()

    report = json.loads(Path(args.parity).read_text())
    ft = parse_parity_ft(report, dt=args.dt)
    paired = json.loads(Path(args.paired).read_text()) if args.paired else None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = plot_alarm_material(ft, paired, out_dir / "alarm_lead_time")
    summary = build_summary(ft, paired, args.parity, args.paired)
    sp = out_dir / "anticipation_summary.json"
    sp.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("wrote:", *written, str(sp), sep="\n  ")


if __name__ == "__main__":
    main()
