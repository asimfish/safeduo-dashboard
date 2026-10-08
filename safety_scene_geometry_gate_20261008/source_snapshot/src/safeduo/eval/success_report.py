"""Success-rate report skeleton (teleop-safety eval spec sections 2A/2B).

Two report surfaces, both md + json:

  A. skill-bucket success matrix (spec 2A) -- per-task x per-noise-level
     success rates over the cuRobo skill library replay. The data source is
     T1's SkillReplayDelta (landed 2026-08-13; wired into endurance_eval
     --flow skill by C9), which follows the ConflictMixSource duck-typed
     contract:
       - DeltaSource protocol: reset(env_ids, generator) / sample(state);
       - .names (list of task names) + .assignment ((N,) long, per-env task
         index) so endurance_eval logs a `family` per run for free;
       - the T2 noise tier is one per window (--skill-tier or amp-derived),
         logged as `skill_tier` on every run row.
     skill_matrix_from_runs() turns such rows straight into the matrix;
     skill_matrix_skeleton()/fill_skill_cell() stay as the manual fill API.

  B. pure-random tier report (spec 2B) -- rows per (method, tier, duration):
     n runs, successes, success rate with two-sided Clopper-Pearson 95% CI
     (spec: never a bare x/N), violation-step dose stats, per-class
     breakdown. Tiers: regular amp 0.015-0.03 (glove -> brisk human),
     adversarial amp 0.03-0.06 + directed-approach flows, always labeled.

Everything here is pure python/torch-free -- Mac-importable, unit-tested in
tests/test_endurance_eval.py. endurance_eval.py calls render_endurance_report
at the end of every batch.
"""

from __future__ import annotations

import json

from safeduo.eval.metrics import clopper_pearson_interval

CLASS_KEYS = ("cross", "self_F", "self_U", "table")

# spec section 1 amp calibration anchors (rad/step @60Hz)
TIER_DEFS = {
    "regular": {"amp_lo": 0.015, "amp_hi": 0.03,
                "note": "glove-scale to brisk-human random roaming"},
    "adversarial": {"amp_lo": 0.03, "amp_hi": 0.06,
                    "note": "deliberate-aggression amps + directed-approach "
                            "flows (head_on_crossing / handover_approach), "
                            "always labeled as adversarial tier"},
    # T2-C beyond-braking scale; must mirror endurance_eval.TIER_AMPS keys or
    # the md report render KeyErrors after a full battery cell (G1 finding,
    # d1_battery3 extreme 2026-08-16).
    "extreme": {"amp_lo": 0.06, "amp_hi": 0.10,
                "note": "beyond-braking 'deliberate strike' amps"},
}

PENDING = "pending T1 (SkillReplayDelta)"


# --------------------------------------------------------------------------
# B. pure-random tier report
# --------------------------------------------------------------------------

def _fmt_ci(ci) -> str:
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def tier_label(payload: dict) -> str:
    if payload.get("tier"):
        d = TIER_DEFS[payload["tier"]]
        return f"{payload['tier']} (amp {d['amp_lo']}-{d['amp_hi']})"
    amps = payload.get("amps") or []
    if len(set(amps)) == 1:
        return f"amp {amps[0]}"
    return f"amps {sorted(set(amps))}"


def endurance_row(payload: dict) -> str:
    agg = payload["aggregate"]
    vs = agg["violation_steps"]
    per_cls = agg["per_class"]
    cls_txt = " / ".join(
        f"{k}:{per_cls[k]['runs_with_violation']}" for k in CLASS_KEYS)
    return (f"| {payload['method']} | {payload['flow']} "
            f"| {tier_label(payload)} | {payload['duration_s']:.0f}s "
            f"| {agg['n_success']}/{agg['n_runs']} "
            f"| {agg['success_rate']:.3f} {_fmt_ci(agg['success_ci95'])} "
            f"| {agg['n_success_gt_verified']}/{agg['n_runs']} "
            f"{_fmt_ci(agg['success_gt_ci95'])} "
            f"| {vs['mean']:.1f}/{vs['median']:.0f}/{vs['max']} "
            f"| {cls_txt} |")


ENDURANCE_HEADER = (
    "| method | flow | tier | duration | success | rate [CP95] "
    "| GT-verified [CP95] | viol steps mean/med/max "
    "| runs w/ violation by class (cross/self_F/self_U/table) |")


def render_endurance_report(payloads: list) -> str:
    """Markdown report for a batch of endurance payloads (one per method)."""
    lines = [
        "# Pure-random endurance success report (spec section 2B)",
        "",
        "success = zero violation steps over the whole continuous run "
        "(sphere-margin caliber, conservative under the +5mm inflation "
        "adjudication); CI = two-sided Clopper-Pearson 95%.",
        "GT-verified column counts cross/self_F/table only (diagnostic: "
        "splits the v4 self_U conservative-sphere artifact out; on the v5 "
        "sphere model the two columns should converge).",
        "table channel is exempt-aware (2026-08-15 C11 caliber fix): "
        "near_table low-speed hand-on-own-table contact is designed-legal "
        "(violation_on_contact=false) and no longer counts as a violation "
        "step; the old exemption-blind counts stay in the JSON under *_raw "
        "and table_exempt_steps is the per-run transparency column.",
        "",
        ENDURANCE_HEADER,
        "|" + "---|" * 9,
    ]
    for p in payloads:
        lines.append(endurance_row(p))
    lines.append("")
    for p in payloads:
        agg = p["aggregate"]
        if "n_success_raw" not in agg:
            continue                    # legacy payload (pre-fix caliber)
        ex = agg.get("table_exempt_steps", {})
        lines.append(
            f"- {p['method']}: raw (exemption-blind) caliber would read "
            f"{agg['n_success_raw']}/{agg['n_runs']}; exempted table-contact "
            f"steps total {ex.get('total_steps', 0)} across "
            f"{ex.get('runs_with_exempt_contact', 0)} runs "
            f"(table_caliber={agg.get('table_caliber', '?')})")
    lines.append("")
    env_yamls = sorted({p.get("env_yaml", "?") for p in payloads})
    durs = sorted({p["duration_s"] for p in payloads})
    n_by_dur = {d: max((p["aggregate"]["n_runs"] for p in payloads
                        if p["duration_s"] == d), default=0) for d in durs}
    lines += [
        f"scene: {', '.join(env_yamls)}; runs per cell: "
        + ", ".join(f"{n}x{d:.0f}s" for d, n in n_by_dur.items()),
        "",
        "sample-size note (spec): 100/100 passes -> CP95 upper bound on the "
        "violation rate ~3.6%; pushing below 1% needs ~300 runs.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# A. skill-bucket matrix skeleton (T1 pending)
# --------------------------------------------------------------------------

# template default = the spec's example common four-arm tasks; real batches
# use the observed families instead (skill_matrix_from_runs)
DEFAULT_TASKS = ("handover", "dual_arm_carry", "cross_pick_place",
                 "center_grab")
DEFAULT_NOISE_LEVELS = ("none", "low", "high")


def skill_matrix_skeleton(tasks: "list | None" = None,
                          noise_levels: "list | None" = None) -> dict:
    """Empty skill-bucket matrix (spec 2A). Cells fill via fill_skill_cell
    as SkillReplayDelta batches come in; unfilled cells render as pending."""
    tasks = list(tasks or DEFAULT_TASKS)
    noise_levels = list(noise_levels or DEFAULT_NOISE_LEVELS)
    return {
        "schema": "skill_success_matrix_v1",
        "status": PENDING,
        "tasks": tasks,
        "noise_levels": noise_levels,
        "cells": {t: {nl: None for nl in noise_levels} for t in tasks},
        "notes": [
            "success = zero violation steps over the replayed trajectory "
            "(same caliber as the endurance report)",
            "coverage: cuRobo skill library x start/goal randomization x "
            "noise tiers (spec section 1 T1/T2)",
            "data source: T1 SkillReplayDelta (record -> replay + noise); "
            "wire-in: endurance_eval --flow skill, family per run comes from "
            "the .assignment/.names duck-typed protocol",
        ],
    }


def fill_skill_cell(matrix: dict, task: str, noise_level: str,
                    n: int, k_success: int, conf: float = 0.95) -> dict:
    """Fill one cell with n runs / k successes (+ CP CI); returns the cell."""
    assert task in matrix["cells"], f"unknown task {task}"
    assert noise_level in matrix["cells"][task], f"unknown noise {noise_level}"
    cell = {
        "n": int(n), "k_success": int(k_success),
        "rate": (k_success / n) if n else 0.0,
        "ci95": list(clopper_pearson_interval(k_success, n, conf)),
    }
    matrix["cells"][task][noise_level] = cell
    if all(c is not None for row in matrix["cells"].values()
           for c in row.values()):
        matrix["status"] = "complete"
    return cell


def skill_matrix_from_runs(rows: list, conf: float = 0.95,
                           success_key: str = "success") -> dict:
    """Spec-2A matrix straight from endurance_eval --flow skill run rows.

    Rows must carry ``family`` (skill name via the .names/.assignment
    protocol) and ``skill_tier`` (T2 noise tier logged per window); tasks and
    noise levels are the observed sets, each cell aggregates the strict
    success flag over its runs (fill_skill_cell -> CP CI). Cells no run
    landed on stay pending: the per-env skill draw is random, so small
    batches do not guarantee full coverage.

    ``success_key`` (T4 R5, additive; default keeps the safety caliber
    bit-identical): which per-run bool to aggregate -- "success" = zero
    violation steps (safety), "task_success" = EE waypoint task-completion
    caliber (eval/skill_success.py). Rows missing the key count as failure.
    """
    tagged = [r for r in rows if "family" in r and "skill_tier" in r]
    if not tagged:
        raise ValueError(
            "no rows carry family + skill_tier -- need endurance_eval "
            "--flow skill output (SkillReplayDelta .names/.assignment)")
    tasks = sorted({str(r["family"]) for r in tagged})
    tiers = sorted({int(r["skill_tier"]) for r in tagged})
    matrix = skill_matrix_skeleton(
        tasks=tasks, noise_levels=[f"tier{t}" for t in tiers])
    matrix["success_key"] = success_key
    for task in tasks:
        for t in tiers:
            sub = [r for r in tagged
                   if str(r["family"]) == task and int(r["skill_tier"]) == t]
            if sub:
                fill_skill_cell(
                    matrix, task, f"tier{t}", n=len(sub),
                    k_success=sum(1 for r in sub
                                  if bool(r.get(success_key, False))),
                    conf=conf)
    # T1 is live here by construction -- unfilled cells just mean the random
    # per-env skill draw never landed on them in this batch
    matrix["pending_label"] = "no runs yet"
    if matrix["status"] == PENDING:
        matrix["status"] = "partial: unobserved cells need more runs"
    return matrix


def render_skill_matrix(matrix: dict) -> str:
    """Markdown table: tasks x noise levels; the point of the matrix is that
    the most dangerous cooperation family is visible at a glance (spec 2A)."""
    nls = matrix["noise_levels"]
    pending = matrix.get("pending_label", PENDING)
    skey = matrix.get("success_key", "success")
    caliber = "" if skey == "success" else \
        f"\n\ncaliber: {skey} (EE waypoint task-completion, " \
        "eval/skill_success.py)"
    lines = [
        "# Skill-bucket success matrix (spec section 2A)" + caliber,
        "",
        f"status: {matrix['status']}",
        "",
        "| task | " + " | ".join(nls) + " |",
        "|" + "---|" * (len(nls) + 1),
    ]
    for task in matrix["tasks"]:
        cells = []
        for nl in nls:
            c = matrix["cells"][task][nl]
            if c is None:
                cells.append(pending)
            else:
                cells.append(f"{c['k_success']}/{c['n']} "
                             f"({c['rate']:.2f} {_fmt_ci(c['ci95'])})")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    lines += [""] + [f"- {n}" for n in matrix["notes"]]
    return "\n".join(lines)


def write_report_templates(out_dir) -> None:
    """Emit the empty md+json skeleton pair (deliverable for the work order;
    also handy as the paper-facing template before real batches land)."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    matrix = skill_matrix_skeleton()
    (out / "skill_matrix_template.json").write_text(json.dumps(matrix, indent=2))
    (out / "skill_matrix_template.md").write_text(
        render_skill_matrix(matrix), encoding="utf-8")
    demo = {
        "method": "<safeduo|estop|speed|cbf|passthrough|raw>",
        "flow": "<l1|directed>", "tier": None,
        "amps": [0.015, 0.02, 0.025, 0.03], "duration_s": 60.0,
        "env_yaml": "duo_env_v5.yaml",
        "aggregate": {
            "n_runs": 0, "n_success": 0, "success_rate": 0.0,
            "success_ci95": [0.0, 1.0], "n_success_gt_verified": 0,
            "success_rate_gt_verified": 0.0, "success_gt_ci95": [0.0, 1.0],
            "conf": 0.95,
            "violation_steps": {"mean": 0.0, "max": 0, "median": 0.0},
            "per_class": {k: {"runs_with_violation": 0,
                              "total_violation_steps": 0}
                          for k in CLASS_KEYS},
        },
        "runs": [],
    }
    (out / "endurance_report_template.json").write_text(json.dumps(demo, indent=2))
    (out / "endurance_report_template.md").write_text(
        render_endurance_report([demo]), encoding="utf-8")


if __name__ == "__main__":
    import sys

    write_report_templates(sys.argv[1] if len(sys.argv) > 1
                           else "artifacts/analysis/c8_eval_protocol")
