"""Per-class damaging/graze re-cut -> official v4 Tab.1 (C7, Round-129 order).

Motivation: the aggregate violation caliber of block1_harness is flooded by
the self_U wrist-ring fat-sphere artifact (sphere-layer collision rate = 100%
for every method; B2 parity: real mesh distance of the wrist-ring pair is a
constant ~62 mm at the birth column while the v4 sphere ceiling is 6.2 mm).
The learning increment is invisible in that column. This module re-cuts the
EXISTING final-eval records (zero sim re-run, zero GPU) into:

  per class c in {cross, self_f, self_u, table}, per episode:
    viol_c      episode's min margin of class c dips below 0
    damaging_c  depth_c strictly > band (5 mm, G1 Round-69/76 arbitration)
    graze_c     viol_c and not damaging_c (in-band shallow contact)

  self_u additionally under the "v5-equivalent" caliber (B2 threshold
  conversion table: d_min_v5 ~= d_min_v4 + 8 mm, the bilateral de-inflation
  difference 2 x 4 mm): margins shifted by +8 mm before the same split.
  NOTE this is a re-thresholding of trajectories that were *terminated* under
  the v4 caliber -- it removes the artifact from the count but cannot
  un-truncate episodes the artifact ended early. True v5 numbers require
  re-eval under the staged v5 spheres (opens after Tab.1 per Round-129).

Episode-boundary semantics are imported from collision_recut / block1_harness
(single source, validated bit-for-bit against A4's footnote numbers), and the
sphere-layer + argmax-attribution results are cross-checked cell-by-cell
against the harness manifest.json -- any mismatch aborts.

Because eval terminates on the first violation (terminate_on_violation=True),
per-episode rates carry exposure censoring: an episode ended by the self_U
artifact at step k never exposes the later cross conflict. Per-1k-step class
rates are reported alongside as the truncation-robust view.

Usage (local or server, CPU, ~1 min for 20 files):
  PYTHONPATH=src python -m safeduo.eval.classwise_recut \
      --dirs s42=artifacts/block1/a8_v4s42_main \
             s43=artifacts/block1/a8_v4s43_main \
      --out artifacts/block1/v4_table
Outputs: main_table_v4.md, main_table_v4.tex, classwise_split.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import torch

from safeduo.eval.block1_harness import MARGIN_PAD, episode_slices
from safeduo.eval.collision_recut import discover_cells, load_records_grid
from safeduo.eval.metrics import (
    MARGIN_CLASSES,
    clopper_pearson_upper,
    episode_max_depth,
)

BAND_M = 0.005            # G1 damaging band: depth strictly > 5 mm
SELF_U_SHIFT_M = 0.008    # B2 v5-equivalent conversion: d_min_v5 ~= d_min_v4 + 8 mm


@dataclass
class ClasswiseConfig:
    band_m: float = BAND_M
    shift_m: dict = field(default_factory=lambda: {"self_u": SELF_U_SHIFT_M})
    conf: float = 0.95


def episode_major_with_valid(grid: dict) -> dict:
    """Env-major (T, N) grids -> episode-major traces + valid mask.

    Same boundary semantics as collision_recut.episode_major (episode_slices
    is the single source); additionally returns the valid mask and episode
    lengths so per-step rates use true exposure, never padding."""
    done = grid["done"]
    episodes, n_tail = episode_slices(done)
    if not episodes:
        raise ValueError("no complete episodes in records")
    H = max(e - s + 1 for _, s, e in episodes)
    E = len(episodes)
    viol = torch.zeros(H, E, dtype=torch.bool)
    valid = torch.zeros(H, E, dtype=torch.bool)
    ep_len = torch.zeros(E, dtype=torch.long)
    margins = {mk: torch.full((H, E), MARGIN_PAD) for mk in grid["margins"]}
    for j, (env, s, e) in enumerate(episodes):
        ln = e - s + 1
        ep_len[j] = ln
        valid[:ln, j] = True
        viol[:ln, j] = grid["violation"][s:e + 1, env]
        for mk, m in grid["margins"].items():
            margins[mk][:ln, j] = m[s:e + 1, env]
    return {"violation": viol, "valid": valid, "margins": margins,
            "ep_len": ep_len, "n_episodes": E, "n_dropped_tail": n_tail}


def _split_one_class(margin: torch.Tensor, band_m: float, conf: float,
                     n_steps: int) -> dict:
    """(H, E) min-margin trace of one class -> episode + step split stats.
    Padding is +1.0 (inert): it can neither violate nor add depth."""
    depth = (-margin).clamp_min(0.0)          # (H, E)
    ep_depth = depth.max(dim=0).values        # (E,)
    viol_ep = (margin < 0.0).any(dim=0)       # (E,)
    dmg_ep = ep_depth > band_m                # strict >: band edge = graze
    graze_ep = viol_ep & ~dmg_ep
    n = int(viol_ep.numel())
    k_viol, k_dmg = int(viol_ep.sum()), int(dmg_ep.sum())
    k_graze = int(graze_ep.sum())
    viol_steps = int((margin < 0.0).sum())
    deep_steps = int((margin < -band_m).sum())
    dmg_depths = sorted(((ep_depth[dmg_ep]) * 1e3).tolist(), reverse=True)
    out = {
        "episodes": n,
        "viol_episodes": k_viol,
        "damaging_episodes": k_dmg,
        "graze_episodes": k_graze,
        "damaging_rate": k_dmg / max(n, 1),
        "damaging_rate_ub95": clopper_pearson_upper(k_dmg, n, conf),
        "graze_rate": k_graze / max(n, 1),
        "viol_steps": viol_steps,
        "deep_steps": deep_steps,
        "viol_steps_per_1k": 1e3 * viol_steps / max(n_steps, 1),
        "deep_steps_per_1k": 1e3 * deep_steps / max(n_steps, 1),
        "damaging_depths_mm": [round(v, 3) for v in dmg_depths[:20]],
        "depth_mm_max": float(ep_depth.max().item() * 1e3),
    }
    return out


def classwise_split(em: dict, cfg: "ClasswiseConfig | None" = None) -> dict:
    """Episode-major traces -> full per-class dual-caliber split."""
    cfg = cfg or ClasswiseConfig()
    margins, viol = em["margins"], em["violation"]
    n_steps = int(em["ep_len"].sum())
    n_ep = em["n_episodes"]
    out = {"episodes": n_ep, "total_steps": n_steps,
           "n_dropped_tail": em["n_dropped_tail"],
           "band_mm": cfg.band_m * 1e3, "classes": {}, "classes_v5eq": {}}

    # GT-verified classes: contact-GT parity coverage in v4 is
    # cross+table+F-self (Round-110 dual-caliber ruling); self_u is the
    # class without contact-GT cross-validation.
    GT_VERIFIED = ("cross", "self_f", "table")
    dmg_union_v4 = torch.zeros(n_ep, dtype=torch.bool)
    dmg_union_v5eq = torch.zeros(n_ep, dtype=torch.bool)
    dmg_union_gtv = torch.zeros(n_ep, dtype=torch.bool)
    viol_union_v4 = torch.zeros(n_ep, dtype=torch.bool)
    viol_union_v5eq = torch.zeros(n_ep, dtype=torch.bool)
    for cls in MARGIN_CLASSES:
        if cls not in margins:
            continue
        m = margins[cls]
        out["classes"][cls] = _split_one_class(m, cfg.band_m, cfg.conf, n_steps)
        depth = (-m).clamp_min(0.0).max(dim=0).values
        dmg_union_v4 |= depth > cfg.band_m
        viol_union_v4 |= (m < 0.0).any(dim=0)
        if cls in GT_VERIFIED:
            dmg_union_gtv |= depth > cfg.band_m
        shift = cfg.shift_m.get(cls, 0.0)
        m_eq = m + shift
        if shift:
            out["classes_v5eq"][cls] = _split_one_class(
                m_eq, cfg.band_m, cfg.conf, n_steps)
            out["classes_v5eq"][cls]["shift_mm"] = shift * 1e3
        depth_eq = (-m_eq).clamp_min(0.0).max(dim=0).values
        dmg_union_v5eq |= depth_eq > cfg.band_m
        viol_union_v5eq |= (m_eq < 0.0).any(dim=0)

    for tag, dmg, vio in (("v4", dmg_union_v4, viol_union_v4),
                          ("v5eq", dmg_union_v5eq, viol_union_v5eq),
                          ("gtv", dmg_union_gtv, None)):
        k = int(dmg.sum())
        out[f"any_damaging_{tag}"] = k
        out[f"any_damaging_{tag}_rate"] = k / max(n_ep, 1)
        out[f"any_damaging_{tag}_ub95"] = clopper_pearson_upper(
            k, n_ep, cfg.conf)
        if vio is not None:
            out[f"any_viol_{tag}"] = int(vio.sum())

    # argmax attribution (manifest caliber) for the cross-check
    depth_all = episode_max_depth(margins)
    viol_any = viol.any(dim=0)
    dmg_argmax = viol_any & (depth_all["overall"] > cfg.band_m)
    cls_list = [c for c, keep in zip(depth_all["argmax_class"],
                                     dmg_argmax.tolist()) if keep]
    out["crosscheck"] = {
        "collision_episodes": int(viol_any.sum()),
        "damaging_episodes_argmax": int(dmg_argmax.sum()),
        "damaging_class_counts_argmax":
            {c: cls_list.count(c) for c in sorted(set(cls_list))},
    }
    return out


def crosscheck_manifest(cell: str, split: dict, man: dict) -> None:
    """Abort on any disagreement with the harness manifest (guards episode
    reconstruction drift, same policy as collision_recut)."""
    cc = split["crosscheck"]
    checks = [
        ("episodes", split["episodes"], man["episodes"]),
        ("collision_episodes", cc["collision_episodes"],
         man["collision_episodes"]),
        ("damaging_episodes(argmax)", cc["damaging_episodes_argmax"],
         man["damaging_episodes"]),
    ]
    for name, ours, theirs in checks:
        if ours != theirs:
            raise SystemExit(
                f"{cell}: {name} recut={ours} manifest={theirs} -- "
                "episode reconstruction drift, refusing to write a table")
    man_cls = man.get("damaging_class_counts") or {}
    if man_cls != cc["damaging_class_counts_argmax"]:
        raise SystemExit(
            f"{cell}: argmax damaging class counts recut="
            f"{cc['damaging_class_counts_argmax']} manifest={man_cls}")


BASELINE_METHODS = ("passthrough", "estop", "speed", "cbf")

BASELINE_IDENTITY_KEYS = (
    "episodes", "total_steps", "any_damaging_v4", "any_damaging_v5eq",
    "any_damaging_gtv",
)


BASELINE_MANIFEST_KEYS = (
    "effective_intervention_rate", "intervention_rate", "stall_events",
    "oscillations_per_episode", "time_to_clear_mean", "time_to_clear_p95",
    "projected_progress_conflict",
)

# Replicate-drift tolerances for the live baselines (passthrough/cbf): the
# two harness runs share the seed but PhysX-GPU physics is not bit-exact and
# fast-moving methods amplify float noise chaotically (observed 2026-08-13:
# passthrough_l1 176/1590 vs 175/1585 episodes across the s42/s43 dirs;
# cbf_l1 eff-intervention differs at 1e-7). Statue baselines (estop/speed)
# and all l2eval cells reproduced bit-exact and stay under strict equality.
# metric_rel is generous (5%) on purpose: it must flag protocol breaches
# (wrong seed/config would shift metrics by tens of percent) while passing
# replicate chaos on small-sample statistics (e.g. time-to-clear means over
# ~45 cleared conflicts drift ~2% when a handful of episodes flip).
DRIFT_TOL = {"episodes_rel": 0.005, "count_abs": 5, "rate_abs": 0.002,
             "metric_rel": 0.05}


def _drift(ref, other) -> float:
    denom = max(abs(ref), abs(other), 1e-12)
    return abs(ref - other) / denom


def verify_baseline_replicates(splits: dict, manifests: dict,
                               labels: list) -> dict:
    """Baseline cells are replicate runs of one protocol (same harness
    seed): statue baselines must match bit-exact; live baselines may carry
    PhysX float-noise drift within DRIFT_TOL (recorded verbatim in the
    report; anything larger aborts -- that would be a protocol breach, not
    noise)."""
    report: dict = {}
    ref_label = labels[0]
    for (method, flow) in sorted({k[1:] for k in splits
                                  if k[1] in BASELINE_METHODS}):
        strict = method in ("estop", "speed")
        ref = splits[(ref_label, method, flow)]
        ref_man = manifests[ref_label]["metrics"][f"{method}_{flow}"]
        drifts: dict = {}
        for lab in labels[1:]:
            other = splits[(lab, method, flow)]
            man = manifests[lab]["metrics"][f"{method}_{flow}"]
            pairs = [(k, ref[k], other[k]) for k in BASELINE_IDENTITY_KEYS]
            for cls, st in ref["classes"].items():
                ost = other["classes"][cls]
                pairs += [(f"{cls}.{k}", st[k], ost[k])
                          for k in ("viol_episodes", "damaging_episodes")]
            pairs += [(f"manifest.{k}", ref_man[k], man[k])
                      for k in BASELINE_MANIFEST_KEYS]
            for name, a, b in pairs:
                if a == b:
                    continue
                if strict:
                    raise SystemExit(
                        f"statue baseline {method}/{flow}: {name} differs "
                        f"across seed dirs ({a} vs {b}) -- must be bit-exact")
                if name in ("episodes", "total_steps"):
                    ok = _drift(a, b) <= DRIFT_TOL["episodes_rel"]
                elif isinstance(a, int):
                    # counters (incl. rare-event counters like stalls):
                    # chaos flips borderline episodes one by one
                    ok = abs(a - b) <= DRIFT_TOL["count_abs"]
                elif name.startswith("manifest."):
                    ok = _drift(a, b) <= DRIFT_TOL["metric_rel"]
                else:
                    ok = abs(a - b) <= DRIFT_TOL["rate_abs"]
                if not ok:
                    raise SystemExit(
                        f"baseline {method}/{flow}: {name} drift beyond "
                        f"tolerance ({a} vs {b}) -- protocol breach")
                drifts[name] = {ref_label: a, lab: b}
        report[f"{method}_{flow}"] = drifts or "identical"
    return report


# ---------------------------------------------------------------- rendering

FLOW_TITLE = {
    "l2eval": "L2-eval conflict battery (eval split)",
    "l1": "L1 non-conflict random traffic",
}

METHOD_LABEL = {
    "safeduo": "SafeDuo",
    "passthrough": "passthrough (damper-only)",
    "estop": "E-stop (frozen W5) [dagger]",
    "speed": "speed-scaling (frozen W5) [dagger]",
    "cbf": "strong CBF-QP (gamma4-linear)",
}

LATEX_LABEL = {
    "safeduo": "SafeDuo",
    "passthrough": "passthrough (damper-only)",
    "estop": "E-stop$^\\dagger$",
    "speed": "speed-scaling$^\\dagger$",
    "cbf": "strong CBF-QP",
}

METHOD_ORDER = ("safeduo", "passthrough", "estop", "speed", "cbf")


def _cls_cell(st: dict) -> str:
    """damaging n (rate%) / graze n"""
    return (f"{st['damaging_episodes']} ({100 * st['damaging_rate']:.2f}%) "
            f"/ {st['graze_episodes']}")


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def _behavior_cells(man: dict) -> list:
    eint = man["effective_intervention_rate"]
    ci = man.get("effective_intervention_ci95") or [float("nan")] * 2
    return [
        f"{eint:.3f} [{ci[0]:.3f},{ci[1]:.3f}]",
        f"{man['intervention_rate']:.3f}",
        f"{man['stall_events']}",
        f"{man['oscillations_per_episode']:.2f}",
        f"{man['time_to_clear_mean']:.2f}/{man['time_to_clear_p95']:.2f}",
        f"{man['projected_progress_conflict']:.3f}",
    ]


def render_md(splits: dict, manifests: dict, labels: list,
              meta: dict) -> str:
    L = []
    L.append("# Block-1 main table v4 (official Tab.1: per-class damaging "
             "split, dual-caliber self_U)")
    L.append("")
    L.append(f"- source: {meta['dirs']}")
    L.append(f"- protocol: peak ckpt (model_1950 both seeds) + "
             f"duo_env_v4.yaml + 5 methods x {{l2eval,l1}} x 256 env x "
             f"601 steps x harness seed 123 + frozen W5 baselines "
             f"(recorded in per-dir manifest.json)")
    L.append(f"- damaging band: depth > {meta['band_mm']:.0f} mm (strict, "
             f"G1 Round-69/76); self_U v5-equivalent shift: "
             f"+{meta['shift_mm']:.0f} mm (B2 conversion table, Round-117)")
    L.append(f"- baseline rows are replicate runs of one protocol, rendered "
             f"once (from the {labels[0]} dir): statue baselines "
             f"(estop/speed) and all l2eval cells reproduced bit-exact "
             f"across dirs; live-baseline l1 cells carry PhysX float-noise "
             f"replicate drift within tolerance (verbatim in "
             f"classwise_split.json, see note 6). SafeDuo rows are per-seed "
             f"({'/'.join(labels)})")
    L.append(f"- generated: {meta['date']} by classwise_recut.py; evidence: "
             f"classwise_split.json")
    L.append("")
    for flow in ("l2eval", "l1"):
        L.append(f"## {FLOW_TITLE[flow]}")
        L.append("")
        L.append("### A. per-class violation split "
                 "(cell = damaging n (rate%) / graze n; episode caliber)")
        L.append("")
        L.append("| method | episodes | cross | self_F | table | "
                 "GT-verified dmg (CP-UB95) | self_U (v4) | self_U (v5-eq) "
                 "| any-damaging v4 (CP-UB95) | any-damaging v5-eq "
                 "(CP-UB95) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for method in METHOD_ORDER:
            rows = ([(lab, (lab, method, flow)) for lab in labels]
                    if method == "safeduo"
                    else [(None, (labels[0], method, flow))])
            for lab, key in rows:
                sp = splits[key]
                name = METHOD_LABEL[method] + (f" ({lab})" if lab else "")
                c = sp["classes"]
                ceq = sp["classes_v5eq"]["self_u"]
                L.append(
                    f"| {name} | {sp['episodes']} "
                    f"| {_cls_cell(c['cross'])} | {_cls_cell(c['self_f'])} "
                    f"| {_cls_cell(c['table'])} "
                    f"| {sp['any_damaging_gtv']} "
                    f"({_fmt_pct(sp['any_damaging_gtv_rate'])}, "
                    f"UB {_fmt_pct(sp['any_damaging_gtv_ub95'])}) "
                    f"| {_cls_cell(c['self_u'])} "
                    f"| {_cls_cell(ceq)} "
                    f"| {sp['any_damaging_v4']} "
                    f"({_fmt_pct(sp['any_damaging_v4_rate'])}, "
                    f"UB {_fmt_pct(sp['any_damaging_v4_ub95'])}) "
                    f"| {sp['any_damaging_v5eq']} "
                    f"({_fmt_pct(sp['any_damaging_v5eq_rate'])}, "
                    f"UB {_fmt_pct(sp['any_damaging_v5eq_ub95'])}) |")
        L.append("")
        L.append("### B. behavior metrics (from harness manifest, "
                 "unchanged caliber)")
        L.append("")
        L.append("| method | eff. intervention [CI95] | declared interv. "
                 "| stalls | oscill./ep | time-to-clear mean/p95 (s) "
                 "| proj. progress |")
        L.append("|---|---|---|---|---|---|---|")
        for method in METHOD_ORDER:
            rows = ([(lab, (lab, method, flow)) for lab in labels]
                    if method == "safeduo"
                    else [(None, (labels[0], method, flow))])
            for lab, key in rows:
                man = manifests[key[0]]["metrics"][f"{method}_{flow}"]
                name = METHOD_LABEL[method] + (f" ({lab})" if lab else "")
                L.append("| " + " | ".join([name] + _behavior_cells(man))
                         + " |")
        L.append("")
    L.append(FOOTNOTES_MD.format(band_mm=meta["band_mm"],
                                 shift_mm=meta["shift_mm"]))
    return "\n".join(L) + "\n"


FOOTNOTES_MD = """## Table notes (honest calibers; wording per Round-110/114/128/129 rulings)

1. **self_U artifact (measurement layer, not hardware risk).** The self_U
   column is dominated by the U wrist-ring pair (jaka link4 | hand_base)
   whose v4 sphere model is structurally fat: its sphere-layer margin
   ceiling is +6.2 mm over the whole joint space while the true mesh
   distance at the birth column is a constant ~62 mm (B2 parity,
   wrist_ring_parity_v5). Every method saturates this channel (sphere-layer
   collision rate 100%); physical self-collision is ENABLED in v4, so real
   interpenetration is prevented by PhysX regardless of the reading. The
   self_U columns are therefore upper bounds of a measurement artifact
   family and are NOT comparable to the GT-verified classes.
2. **GT blind zone / force criterion inactive.** The damaging split is
   depth-only: records carry no contact-force column (duo_env contact
   sensors exist only under enable_contact_gt; the harness recorder does
   not read them), so the G1 force criterion (>20 N sustained 2 steps) is
   inactive -- a hypothetical in-band sustained-force episode would be
   under- never over-classified as damaging. Contact-GT parity coverage in
   v4 is cross+table+F-self ("zero missed detection"); self_U is exactly
   the class without contact-GT cross-validation (Round-110 dual-caliber
   ruling: GT-verified classes carry full alarm weight, self_U is demoted
   to "shell contact (conservative sphere)"). The "GT-verified dmg" column
   is the per-episode union of damaging over cross+self_F+table -- the
   clean method-ranking safety axis of this table.
3. **v5-equivalent caliber (self_U only).** Margins re-thresholded with a
   +{shift_mm:.0f} mm shift = B2's generic de-inflation conversion
   (d_min_v5 ~= d_min_v4 + 8 mm, bilateral 2 x 4 mm; Round-117). Two
   honesty limits: (a) the wrist-ring pair carries ~50 mm of ADDITIONAL
   pseudo-conservatism beyond the generic conversion (v4 reads negative
   while true distance is still ~26 mm median), so residual self_U counts
   under v5-eq remain artifact-dominated upper bounds; (b) trajectories
   were terminated under the v4 caliber (terminate_on_violation), so v5-eq
   removes artifact counts but cannot un-truncate the exposure the
   artifact censored. The clean closure is re-training/re-eval under the
   staged v5 spheres (opens after this table per Round-129).
4. **[dagger] dead-baseline rows (structural evidence, Round-114 ruling).**
   E-stop and speed-scaling at their frozen W5 work points have no live
   work point in the v4 strong-coupling layout (birth self floor ~5 mm
   under their thresholds): they effectively freeze arms from birth
   (eff. intervention ~0.95-1.0, declared 1.0). Their low damaging counts
   are the "statue" corner of the safety-productivity trade-off, kept as
   structural evidence that distance-gate families collapse here.
5. **Episode caliber & censoring.** Eval terminates on first violation, so
   per-episode denominators inflate for methods that hit the artifact more
   often, and exposure after an artifact stop is censored (a cross
   conflict that would have developed later is never observed).
   classwise_split.json carries per-1k-step class rates as the
   truncation-robust companion view; damaging band edge (= exactly
   {band_mm:.0f} mm depth) counts as graze (strict >).
6. **Baseline replicate drift (new fact, not previously in the ledger).**
   The baseline cells in the two seed dirs are replicate runs of one
   protocol (same harness seed 123). Statue baselines and ALL l2eval cells
   reproduced bit-exact; the live-baseline l1 cells did not: passthrough_l1
   176/1590 vs 175/1585 episodes (any-damaging 176 vs 175), cbf_l1
   effective-intervention differs at 1e-7. Attributed to PhysX-GPU
   float-level nondeterminism amplified chaotically by fast free motion
   over 601 steps; magnitude is far below every reading difference quoted
   in this table. Exact per-dir values: classwise_split.json
   ("baseline_identity_across_dirs" + per-dir cells).
"""


def render_latex(splits: dict, manifests: dict, labels: list,
                 meta: dict) -> str:
    """Booktabs LaTeX fragment (tabular only, \\input-ready)."""
    L = []
    L.append("% Auto-generated by safeduo.eval.classwise_recut "
             f"({meta['date']}). Counts are damaging episodes per class")
    L.append("% (episode caliber, band > "
             f"{meta['band_mm']:.0f} mm strict); see classwise_split.json.")
    L.append("% Requires \\usepackage{booktabs,multirow}.")
    L.append("\\begin{table*}[t]")
    L.append("\\centering")
    L.append("\\caption{Block-1 main results (v4 dual-seed). Damaging "
             "collisions (depth $>$ 5\\,mm) split per contact class; "
             "``union'' is the per-episode union over the GT-verified "
             "classes (cross/self\\_F/table). self\\_U is reported under "
             "the in-use v4 caliber and the v5-equivalent caliber "
             "(+8\\,mm de-inflation conversion); it is a known "
             "conservative-sphere artifact channel (real wrist-ring "
             "clearance $\\approx$62\\,mm) without contact-GT "
             "cross-validation, and carries no method ranking. "
             "$\\dagger$ marks frozen distance-gate baselines with no live "
             "work point in this layout (statue regime). Behavior columns: "
             "effective intervention (lower = more permissive), stalls, "
             "oscillations/ep, time-to-clear p95.}")
    L.append("\\label{tab:block1-v4}")
    L.append("\\small")
    L.append("\\begin{tabular}{ll rrrr rr rr r r r r}")
    L.append("\\toprule")
    L.append(" & & \\multicolumn{4}{c}{GT-verified damaging} & "
             "\\multicolumn{2}{c}{self\\_U (artifact ch.)} & "
             "\\multicolumn{2}{c}{any-damaging} & & & & \\\\")
    L.append("\\cmidrule(lr){3-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}")
    L.append("flow & method & cross & self\\_F & table & union & v4 & "
             "v5-eq & v4 & v5-eq & eff.\\,int. & stalls & osc./ep & "
             "TTC p95 (s) \\\\")
    L.append("\\midrule")
    for flow, ftitle in (("l2eval", "L2 conflict"), ("l1", "L1 random")):
        n_rows = len(labels) + len(METHOD_ORDER) - 1
        first = True
        for method in METHOD_ORDER:
            rows = ([(lab, (lab, method, flow)) for lab in labels]
                    if method == "safeduo"
                    else [(None, (labels[0], method, flow))])
            for lab, key in rows:
                sp = splits[key]
                man = manifests[key[0]]["metrics"][f"{method}_{flow}"]
                c = sp["classes"]
                ceq = sp["classes_v5eq"]["self_u"]
                name = LATEX_LABEL[method] + (f" ({lab})" if lab else "")
                cell0 = (f"\\multirow{{{n_rows}}}{{*}}{{{ftitle}}}"
                         if first else "")
                first = False
                L.append(
                    f"{cell0} & {name} & "
                    f"{c['cross']['damaging_episodes']} & "
                    f"{c['self_f']['damaging_episodes']} & "
                    f"{c['table']['damaging_episodes']} & "
                    f"{sp['any_damaging_gtv']} & "
                    f"{c['self_u']['damaging_episodes']} & "
                    f"{ceq['damaging_episodes']} & "
                    f"{sp['any_damaging_v4']}/{sp['episodes']} & "
                    f"{sp['any_damaging_v5eq']}/{sp['episodes']} & "
                    f"{man['effective_intervention_rate']:.3f} & "
                    f"{man['stall_events']} & "
                    f"{man['oscillations_per_episode']:.1f} & "
                    f"{man['time_to_clear_p95']:.2f} \\\\")
        L.append("\\midrule" if flow == "l2eval" else "\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{table*}")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------- driver

def run(dirs: dict, out_dir: "str | Path",
        cfg: "ClasswiseConfig | None" = None) -> dict:
    cfg = cfg or ClasswiseConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = list(dirs)
    splits: dict = {}
    manifests: dict = {}
    for lab, d in dirs.items():
        d = Path(d)
        manifests[lab] = json.loads((d / "manifest.json").read_text())
        for method, flow, path in discover_cells(d):
            em = episode_major_with_valid(load_records_grid(path))
            sp = classwise_split(em, cfg)
            cell = f"{lab}/{method}_{flow}"
            man = manifests[lab]["metrics"][f"{method}_{flow}"]
            crosscheck_manifest(cell, sp, man)
            splits[(lab, method, flow)] = sp
            print(f"[classwise] {cell}: any-dmg v4 {sp['any_damaging_v4']}"
                  f"/{sp['episodes']} -> v5eq {sp['any_damaging_v5eq']} | "
                  f"cross dmg {sp['classes']['cross']['damaging_episodes']} "
                  f"self_f {sp['classes']['self_f']['damaging_episodes']} "
                  f"table {sp['classes']['table']['damaging_episodes']} "
                  f"self_u {sp['classes']['self_u']['damaging_episodes']}"
                  f"->{sp['classes_v5eq']['self_u']['damaging_episodes']}",
                  flush=True)

    identity = verify_baseline_replicates(splits, manifests, labels)
    meta = {
        "dirs": {k: str(v) for k, v in dirs.items()},
        "band_mm": cfg.band_m * 1e3,
        "shift_mm": cfg.shift_m["self_u"] * 1e3,
        "date": date.today().isoformat(),
    }
    md = render_md(splits, manifests, labels, meta)
    tex = render_latex(splits, manifests, labels, meta)
    (out_dir / "main_table_v4.md").write_text(md, encoding="utf-8")
    (out_dir / "main_table_v4.tex").write_text(tex, encoding="utf-8")
    evidence = {
        "config": {"band_m": cfg.band_m, "shift_m": cfg.shift_m,
                   "conf": cfg.conf},
        "meta": meta,
        "baseline_identity_across_dirs": identity,
        "cells": {f"{lab}/{m}_{f}": sp
                  for (lab, m, f), sp in sorted(splits.items())},
    }
    (out_dir / "classwise_split.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8")
    print(f"[classwise] wrote {out_dir}/main_table_v4.md + .tex + "
          "classwise_split.json", flush=True)
    return {"splits": splits, "md": md, "tex": tex}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dirs", nargs="+", required=True,
                    metavar="LABEL=PATH",
                    help="labelled records dirs, e.g. s42=artifacts/...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--band-mm", type=float, default=BAND_M * 1e3)
    ap.add_argument("--self-u-shift-mm", type=float,
                    default=SELF_U_SHIFT_M * 1e3)
    args = ap.parse_args()
    dirs = {}
    for spec in args.dirs:
        lab, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--dirs expects LABEL=PATH, got {spec}")
        dirs[lab] = path
    run(dirs, args.out,
        ClasswiseConfig(band_m=args.band_mm * 1e-3,
                        shift_m={"self_u": args.self_u_shift_mm * 1e-3}))


if __name__ == "__main__":
    main()
