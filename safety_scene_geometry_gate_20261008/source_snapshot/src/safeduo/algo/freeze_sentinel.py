"""Freeze-sentinel sidecar: kill-switch instrumentation for the v3 disease.

A6 handover item (STATUS_A W6 sec.5.1c, ~1h ticket): both v3 seeds slid into
the freeze attractor MID-training (alpha 0.49 healthy at birth -> 0.0005 by
iter ~450) while every first-shard gate was green -- birth checks cannot see
a mid-run collapse, and the reward CANNOT be watched naively because freezing
RAISES it (-1.3 -> -0.91, the landscape disease itself). This sidecar watches
the event shards a training run drops every `flush_every` steps and applies
per-shard freeze gates; on FAIL it writes a verdict file the launch guard can
act on (kill the run, keep the peak checkpoint -- see select_peak_checkpoint
in algo/lagrangian_ppo.py).

Gates (per newest complete shard; thresholds vs measured v3 trajectories in
artifacts/analysis/p_head_v3_*/p_head_diagnosis.json):
  alpha_alive        shard alpha mean >= 0.15 (A6 sentinel spec; healthy
                     shards sit at 0.46-0.54, frozen at 0.001-0.03 -- the
                     gate has an order of magnitude of separation each way)
  p_not_pinned       modal-extreme fraction < 0.95 (s43 froze at p == +1.000;
                     healthy v3 shards reach pin_lo 0.51, so 0.95 keeps a
                     wide false-positive margin)
  no_inversion       EARLY WARNING while alpha is still above the hard gate:
                     alpha fell >= 0.15 from its running shard peak AND the
                     current shard alpha < 0.35 AND reward rose vs the peak
                     shard -- the exact v3 signature (reward climbing while
                     the policy dies). Requires the reward column (train_ppo
                     events carry it); silently inactive on older runs.
Per-arm means are REPORTED in the verdict but not gated: r2's U_R sat at
alpha == 1.0 from an init lock and v3's tube-conditioned U arms idled at
0.01 while the run was still globally healthy -- arm-level kills would have
false-positived every historical healthy run.

C6 additions (s43 autopsy, artifacts/analysis/s43_collapse_autopsy/) --
NON-GATING: neither changes "pass", so kill behaviour is byte-identical:
  collapse_type      verdict payload label once a hard gate has failed.
                     "overconservative_pin" = alpha dead (<0.05) AND p
                     pinned at an extreme (>=0.95) -- the s43 terminal
                     (p==+1.000, replay fires shard 15/iter~341);
                     "freeze_drift" = alpha dead, p wandering -- the
                     v3_main terminal (pin_lo peaked 0.64, never >=0.95);
                     "p_disease" = p pinned while alpha alive (r2's birth
                     defect); "inversion_only" = landscape signature only.
  warnings (advisory, log + verdict only, never kill):
    inversion_early  relaxed inversion (drop >= 0.12, alpha < 0.42, any
                     reward rise). Replay: fires s43 shard 11 (iter ~255)
                     and main shard 13 (~298), i.e. 2 shards before the
                     kill-grade inversion gate on BOTH v3 seeds; healthy
                     plateaus (alpha >= 0.45) can't reach it.
    p_corner_capture p modal-extreme pin >= 0.5 for 2 consecutive shards
                     while alpha slipped >= 0.05 off its running peak.
                     Replay: s43 shard 12 (~277, pin 0.534->0.657 during
                     the slide), main shard 12 (~277, its true slide
                     onset); healthy v3 plateau peaked at pin 0.47 with
                     alpha AT peak -> no fire; r2 fires it, but r2 is
                     already a p_not_pinned hard FAIL from shard 0.

The sidecar is a separate OS process (plain python, no kit): its exit code
is trustworthy, unlike anything running inside Isaac (the A4/C4 lesson --
kit teardown eats SystemExit). The verdict FILE stays the authoritative
channel for the guard regardless.

Usage:
  # one-shot (manual / tests / guard polling):
  PYTHONPATH=src python -m safeduo.algo.freeze_sentinel \
      --events-dir ~/safeduo/artifacts/runs/<run>/events
  # sidecar watch (guard launches next to training):
  PYTHONPATH=src python -m safeduo.algo.freeze_sentinel \
      --events-dir .../events --watch --interval 120 --max-minutes 720
Exit 0 = healthy so far / clean stop, 1 = freeze verdict FAILED (verdict
JSON at <run>/freeze_sentinel_verdict.json either way).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

ALPHA_ALIVE_MIN = 0.15          # A6 sentinel threshold
P_PIN_FRAC_MAX = 0.95           # same zero point as bc_smoke_check
INVERSION_ALPHA_DROP = 0.15     # fall from running peak that arms the check
INVERSION_ALPHA_BELOW = 0.35    # only meaningful once alpha is this low
INVERSION_REWARD_RISE = 0.0     # any reward rise during the fall = disease
# C6 (non-gating) -- thresholds replay-calibrated on both v3 seeds + r2:
ALPHA_DEAD_MAX = 0.05           # "alpha channel pinned at zero" for typing
WARN_INV_ALPHA_DROP = 0.12      # relaxed inversion (advisory only)
WARN_INV_ALPHA_BELOW = 0.42     # v3 healthy plateaus never dip below 0.45
WARN_P_CORNER_PIN = 0.5         # healthy v3 plateau max pin = 0.47
WARN_P_CORNER_ALPHA_SLIP = 0.05  # and only while alpha is off its peak


def shard_stats(path: "str | Path") -> dict:
    """One event shard -> the per-shard window stats the gates consume.
    torch + pyarrow only (server canonical env has no pandas)."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    have = set(pf.schema_arrow.names)
    cols = [f"alpha_{i}" for i in range(4)] + ["p"]
    if "reward" in have:
        cols.append("reward")
    tab = pf.read(columns=["t"] + cols)
    t = tab.column("t").to_numpy(zero_copy_only=False)
    alpha = torch.stack([
        torch.tensor(tab.column(f"alpha_{i}").to_numpy(zero_copy_only=False),
                     dtype=torch.float32) for i in range(4)], dim=-1)
    p = torch.tensor(tab.column("p").to_numpy(zero_copy_only=False),
                     dtype=torch.float32)
    pin = max(float((p >= p.max() - 1e-6).float().mean().item()),
              float((p <= p.min() + 1e-6).float().mean().item()))
    out = {
        "shard": Path(path).name,
        "t_min": int(t.min()), "t_max": int(t.max()),
        "alpha_mean": round(float(alpha.mean().item()), 4),
        "alpha_arm_means": [round(float(v), 4) for v in alpha.mean(dim=0)],
        "alpha_std": round(float(alpha.std().item()), 4),
        "p_pinned_frac": round(pin, 4),
        "reward_mean": (round(float(torch.tensor(
            tab.column("reward").to_numpy(zero_copy_only=False),
            dtype=torch.float32).mean().item()), 4)
            if "reward" in have else None),
    }
    return out


def collapse_type(cur: dict, checks: dict) -> "str | None":
    """Label the collapse species once any hard gate failed (C6 taxonomy:
    both v3 seeds share ONE behavioural attractor -- penalty-dominated full
    stop -- and differ only in the p-channel terminal microstate)."""
    if all(checks.values()):
        return None
    if cur["alpha_mean"] < ALPHA_DEAD_MAX:
        return ("overconservative_pin" if cur["p_pinned_frac"] >= P_PIN_FRAC_MAX
                else "freeze_drift")
    if not checks.get("p_not_pinned", True):
        return "p_disease"
    return "inversion_only"


def sentinel_warnings(history: list) -> list:
    """Advisory-only early signals (never gate). Replay-validated to lead
    the kill gates by 1-3 shards on both v3 seeds; see module docstring."""
    cur = history[-1]
    warns = []
    peak_i = max(range(len(history)), key=lambda i: history[i]["alpha_mean"])
    peak = history[peak_i]
    drop = peak["alpha_mean"] - cur["alpha_mean"]
    rewards = [h.get("reward_mean") for h in history]
    if all(r is not None for r in rewards) and len(history) >= 2:
        rise = cur["reward_mean"] - peak["reward_mean"]
        if (drop >= WARN_INV_ALPHA_DROP
                and cur["alpha_mean"] < WARN_INV_ALPHA_BELOW
                and rise > INVERSION_REWARD_RISE):
            warns.append("inversion_early")
    if (len(history) >= 2
            and cur["p_pinned_frac"] >= WARN_P_CORNER_PIN
            and history[-2]["p_pinned_frac"] >= WARN_P_CORNER_PIN
            and drop >= WARN_P_CORNER_ALPHA_SLIP):
        warns.append("p_corner_capture")
    return warns


def sentinel_gates(history: list) -> dict:
    """history = shard_stats dicts in shard order; gates ride the newest."""
    cur = history[-1]
    checks = {
        "alpha_alive": cur["alpha_mean"] >= ALPHA_ALIVE_MIN,
        "p_not_pinned": cur["p_pinned_frac"] < P_PIN_FRAC_MAX,
    }
    inversion = {"active": False}
    rewards = [h.get("reward_mean") for h in history]
    if all(r is not None for r in rewards) and len(history) >= 2:
        peak_i = max(range(len(history)),
                     key=lambda i: history[i]["alpha_mean"])
        peak = history[peak_i]
        drop = peak["alpha_mean"] - cur["alpha_mean"]
        rise = cur["reward_mean"] - peak["reward_mean"]
        inversion = {
            "active": True,
            "alpha_peak": peak["alpha_mean"], "alpha_peak_shard": peak["shard"],
            "alpha_drop": round(drop, 4), "reward_rise_vs_peak": round(rise, 4),
        }
        checks["no_inversion"] = not (
            drop >= INVERSION_ALPHA_DROP
            and cur["alpha_mean"] < INVERSION_ALPHA_BELOW
            and rise > INVERSION_REWARD_RISE)
    return {
        "newest": cur,
        "n_shards": len(history),
        "inversion": inversion,
        "checks": checks,
        "pass": all(checks.values()),
        # C6 additions: advisory payload only -- "pass" is untouched above.
        "warnings": sentinel_warnings(history),
        "collapse_type": collapse_type(cur, checks),
    }


# ---- R13 stats-based rolling gates (G3 RCA 2026-08-18) ---------------------
# The shard gates above ride raw event windows (alpha/p/reward only). The
# v6.1 inversion collapse showed the earliest reliable signals live in the
# per-iteration training stats instead: alpha saturation-at-zero fraction,
# policy entropy and p sign imbalance. Replayed on v6.1 these gates fire
# WARN at iter ~1354 and STOP at ~1366 -- 54/42 iterations before the shard
# hard-FAIL at ~1408.
R13_WINDOW = 20
R13_WARN = {"alpha_sat_lo": 0.08, "entropy": -5.0, "sign_imbalance": 0.75}
R13_STOP = {"alpha_sat_lo": 0.10, "entropy": -5.25, "sign_imbalance": 0.80}


def stats_rolling_gates(rows: list, window: int = R13_WINDOW) -> dict:
    """rows = parsed stats.jsonl dicts (iteration order). Evaluates the R13
    rolling-window WARN/STOP gates on the newest `window` rows. Rows missing
    any required key (older formats, smoke stubs) deactivate the gate."""
    need = ("alpha_sat_lo", "entropy", "p_frac_pos", "p_frac_neg")
    rows = [r for r in rows if all(k in r for k in need)]
    if len(rows) < window:
        return {"active": False, "n_rows": len(rows)}
    win = rows[-window:]
    mean = lambda k: sum(float(r[k]) for r in win) / window
    sat = mean("alpha_sat_lo")
    ent = mean("entropy")
    imb = max(mean("p_frac_pos"), mean("p_frac_neg"))
    warn = sat >= R13_WARN["alpha_sat_lo"] and (
        ent <= R13_WARN["entropy"] or imb >= R13_WARN["sign_imbalance"])
    stop = sat >= R13_STOP["alpha_sat_lo"] and (
        ent <= R13_STOP["entropy"] or imb >= R13_STOP["sign_imbalance"])
    return {"active": True, "n_rows": len(rows), "window": window,
            "alpha_sat_lo": round(sat, 4), "entropy": round(ent, 4),
            "sign_imbalance": round(imb, 4),
            "warn": warn, "stop": stop}


def read_stats_jsonl(path: "str | Path") -> list:
    """Tolerant stats.jsonl reader (skips partial trailing lines)."""
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def select_peak_from_events(run_dir: "str | Path", steps_per_iter: int = 24,
                            alpha_band: tuple = (0.3, 0.9),
                            p_pin_max: float = 0.95) -> "dict | None":
    """Peak-harvest discipline for rsl_rl runs (the m200 lesson mechanized
    for the A-line layout): pick the newest model_<iter>.pt whose iteration
    falls inside the run's HEALTHY window, judged from the event shards.

    v3 ground truth this reproduces: main(s42) shards healthy (alpha
    0.46-0.54) through iter ~277, collapse after -> eligible checkpoints
    {model_200} -> picks model_200, exactly A6's manual harvest that beat
    the frozen model_1999 (protection 0.45 vs 0.0). Health bands = the
    bc_smoke_check run-events bands, stricter than the sentinel's 0.15
    stop-gate on purpose: stopping a run needs conservatism, harvesting
    from a finished one does not.

    Returns {"path", "iter", "healthy_up_to_iter", "shard"} or None when no
    checkpoint is inside the healthy window (all-frozen run)."""
    run_dir = Path(run_dir)
    shards = sorted((run_dir / "events").glob("events_*.parquet"))
    if not shards:
        return None
    healthy_up_to = -1
    last_healthy_shard = None
    for f in shards:
        st = shard_stats(f)
        if alpha_band[0] <= st["alpha_mean"] <= alpha_band[1] \
                and st["p_pinned_frac"] <= p_pin_max:
            it = st["t_max"] // steps_per_iter
            if it > healthy_up_to:
                healthy_up_to, last_healthy_shard = it, st["shard"]
    if healthy_up_to < 0:
        return None
    best = None
    for ck in run_dir.glob("model_*.pt"):
        tag = ck.stem.split("_", 1)[1]
        if tag.isdigit() and int(tag) <= healthy_up_to \
                and (best is None or int(tag) > best[0]):
            best = (int(tag), ck)
    if best is None:
        return None
    return {"path": str(best[1]), "iter": best[0],
            "healthy_up_to_iter": int(healthy_up_to),
            "shard": last_healthy_shard}


class FreezeSentinel:
    """Incremental shard reader + gate evaluator with a verdict file."""

    def __init__(self, events_dir: "str | Path",
                 verdict_path: "str | Path | None" = None,
                 stats_path: "str | Path | None" = None):
        self.events_dir = Path(events_dir)
        self.verdict_path = (Path(verdict_path) if verdict_path else
                             self.events_dir.parent / "freeze_sentinel_verdict.json")
        # R13 stats gates: default to the run dir's stats.jsonl when present.
        self.stats_path = (Path(stats_path) if stats_path else
                           self.events_dir.parent / "stats.jsonl")
        self._stats: dict[str, dict] = {}
        # R13 latch: a FAIL verdict is append-only truth. The v6.1 postmortem
        # showed the rolling event archiver can leave only tail shards, so a
        # later one-shot recompute flipped the kill-time FAIL back to PASS.
        # Once FAIL is observed (including one already on disk from a prior
        # sentinel process), every subsequent write stays FAIL.
        self._latched_fail: "dict | None" = None
        if self.verdict_path.exists():
            try:
                prev = json.loads(self.verdict_path.read_text())
                # only a verdict for THIS run's events dir may latch -- a
                # stale file from another run must not poison a fresh one
                if (prev.get("pass") is False
                        and prev.get("events_dir") == str(self.events_dir)):
                    self._latched_fail = {
                        "checked_at": prev.get("checked_at"),
                        "checks": prev.get("checks"),
                        "collapse_type": prev.get("collapse_type"),
                        "newest": prev.get("newest"),
                        "stats_gates": prev.get("stats_gates"),
                    }
            except Exception:
                pass

    def poll(self) -> "dict | None":
        """Read new complete shards; None when no shard is readable yet.
        A shard mid-write raises inside pyarrow -> skipped until next poll."""
        names = sorted(f.name for f in self.events_dir.glob("events_*.parquet"))
        for n in names:
            if n in self._stats:
                continue
            try:
                self._stats[n] = shard_stats(self.events_dir / n)
            except Exception as e:  # partial flush -- retry next round
                print(f"[sentinel] shard {n} unreadable this round: {e}",
                      flush=True)
                break
        sg = stats_rolling_gates(read_stats_jsonl(self.stats_path))
        if not self._stats and not sg.get("active"):
            return None
        history = [self._stats[n] for n in sorted(self._stats)]
        if history:
            verdict = sentinel_gates(history)
        else:  # stats-only mode (no readable shard yet)
            verdict = {"newest": None, "n_shards": 0, "inversion": None,
                       "checks": {}, "pass": True, "warnings": [],
                       "collapse_type": None}
        # R13 stats gates ride on top of the shard gates: STOP is a hard
        # gate, WARN is advisory.
        verdict["stats_gates"] = sg
        if sg.get("active"):
            verdict["checks"]["stats_no_stop"] = not sg["stop"]
            verdict["pass"] = verdict["pass"] and not sg["stop"]
            if sg["warn"] and not sg["stop"]:
                verdict["warnings"] = list(verdict.get("warnings", []))
                verdict["warnings"].append("r13_stats_warn")
            if sg["stop"] and not verdict.get("collapse_type"):
                verdict["collapse_type"] = "r13_stats_stop"
        # R13 latch: FAIL is terminal.
        if self._latched_fail is not None:
            verdict["pass"] = False
            verdict["latched"] = True
            verdict["first_fail"] = self._latched_fail
        elif not verdict["pass"]:
            self._latched_fail = {
                "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "checks": verdict.get("checks"),
                "collapse_type": verdict.get("collapse_type"),
                "newest": verdict.get("newest"),
                "stats_gates": sg if sg.get("active") else None,
            }
            verdict["latched"] = True
            verdict["first_fail"] = self._latched_fail
        verdict["checked_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        verdict["events_dir"] = str(self.events_dir)
        self.verdict_path.write_text(json.dumps(verdict, indent=2))
        return verdict

    def watch(self, interval_s: float = 120.0,
              max_minutes: float = 720.0) -> dict:
        """Poll until FAIL or the time budget runs out. Returns the last
        verdict (guards should still judge the FILE, not our exit code,
        even though a plain-python exit code is in fact trustworthy)."""
        deadline = time.time() + max_minutes * 60.0
        verdict = None
        while time.time() < deadline:
            v = self.poll()
            if v is not None:
                verdict = v
                tag = "PASS" if v["pass"] else "FAIL"
                warn = (" WARN=" + ",".join(v["warnings"])
                        if v.get("warnings") else "")
                cur = v.get("newest") or {}
                sg = v.get("stats_gates") or {}
                sgs = (f" sat={sg.get('alpha_sat_lo')} ent={sg.get('entropy')}"
                       f" imb={sg.get('sign_imbalance')}"
                       if sg.get("active") else "")
                print(f"[sentinel] {tag} shards={v['n_shards']} "
                      f"alpha={cur.get('alpha_mean')} "
                      f"p_pin={cur.get('p_pinned_frac')}{sgs}{warn}", flush=True)
                if not v["pass"]:
                    return v
            time.sleep(interval_s)
        return verdict or {"pass": True, "n_shards": 0,
                           "note": "no shards seen before time budget"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events-dir", required=True)
    ap.add_argument("--verdict", default="",
                    help="verdict JSON path (default <run>/freeze_sentinel_verdict.json)")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=float, default=120.0)
    ap.add_argument("--max-minutes", type=float, default=720.0)
    args = ap.parse_args()
    s = FreezeSentinel(args.events_dir, args.verdict or None)
    if args.watch:
        verdict = s.watch(args.interval, args.max_minutes)
    else:
        verdict = s.poll()
        if verdict is None:
            print("[sentinel] no readable shards yet", flush=True)
            raise SystemExit(0)
    print(json.dumps(verdict, indent=2), flush=True)
    raise SystemExit(0 if verdict.get("pass", False) else 1)


if __name__ == "__main__":
    main()
