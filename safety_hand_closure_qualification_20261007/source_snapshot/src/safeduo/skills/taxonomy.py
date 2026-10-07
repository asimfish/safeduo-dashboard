"""Registry of every trajectory / stream family SafeDuo can produce, filed
under the five owner categories, with the coupling structure the safety filter
has to respect and the last verified status.

Pure data + a coverage report; no Isaac import.  Status strings are the
MASTER_REPORT ledger as of 2026-09-08 (rounds R24-R35); update them together
with the ledger.  `python -m safeduo.skills.taxonomy` prints the report.
"""
from __future__ import annotations

from dataclasses import dataclass, field

ARMS_F = ("F_L", "F_R")
ARMS_U = ("U_L", "U_R")
ARMS_ALL = ARMS_F + ARMS_U

CATEGORIES = {
    1: "dual_independent",     # two arms, each on its own object
    2: "dual_coordinated",     # two arms coupled through one object
    3: "quad_independent",     # four arms, pairs independent
    4: "quad_cooperative",     # four arms in one task chain
    5: "random",               # unscripted operator streams (safety evidence)
}

# coupling kinds the gate must treat as one unit (a31 item 4: group min-alpha)
COUPLING = ("none", "handover", "colift", "relay", "assembly", "contested")


@dataclass(frozen=True)
class Family:
    name: str
    category: int
    arms: tuple                  # arms that move
    coupling: str                # one of COUPLING
    coupled_groups: tuple = ()   # arm groups sharing one object at some phase
    objects: int = 0             # task objects (bins/totes excluded)
    grasp: str = ""              # physics_grasp | kinematic | none
    source: str = ""             # module that builds it
    status: str = ""             # last verified outcome
    notes: str = ""
    tags: tuple = field(default_factory=tuple)

    def __post_init__(self):
        assert self.category in CATEGORIES, self.category
        assert self.coupling in COUPLING, self.coupling


FAMILIES: tuple = (
    # ---------------------------------------------------------------- category 1
    Family("grasp", 1, ("F_L",), "none", objects=1, grasp="physics_grasp",
           source="delta.task_record_s9", status="raw+s0 success (R27 physical grasp)",
           notes="single arm; the R17 clutch smoke task"),
    Family("pick_place", 1, ("F_L",), "none", objects=1, grasp="physics_grasp",
           source="delta.task_record_s9", status="raw success; s0 depends on theta (self_F brake)"),
    Family("bottle_pick", 1, ARMS_F, "none", objects=2, grasp="physics_grasp",
           source="delta.task_library_r26", status="raw+s0 success (v26; F_R close knock = known B-grade)"),
    Family("rod_pick", 1, ARMS_F + ARMS_U, "none", objects=4, grasp="physics_grasp",
           source="delta.task_library_r26",
           status="raw success after gravity comp (R33); s0 table brake steps = low pick geometry",
           notes="F pair one rod each, U pair one cube each -- independent, so category 1/3 border"),
    # ---------------------------------------------------------------- category 2
    Family("handover", 2, ("F_L", "U_R"), "handover", coupled_groups=(("F_L", "U_R"),),
           objects=1, grasp="physics_grasp", source="delta.task_record_s9",
           status="s0 success only with R29 structural-row exemption; physical two-hand hold needs R32"),
    Family("tray_relay", 2, ARMS_F, "handover", coupled_groups=(ARMS_F,), objects=1,
           grasp="physics_grasp", source="delta.task_library_r26",
           status="s0 success with R30/R30b retreat passthrough"),
    Family("box_colift", 2, ARMS_F, "colift", coupled_groups=(ARMS_F,), objects=1,
           grasp="physics_grasp", source="delta.task_library_r26",
           status="raw+s0 success (table brake steps = low palm clamp)"),
    Family("rack_colift", 2, ARMS_F, "colift", coupled_groups=(ARMS_F,), objects=3,
           grasp="physics_grasp", source="delta.task_library_r34",
           status="raw+s0 success; U place fix verified 09-08 (beaker 1.6 / crucible 1.3 cm)",
           notes="F pair palm-clamps the rack while U pair does independent picks"),
    Family("beaker_relay", 2, ARMS_ALL, "relay", coupled_groups=(("F_L", "U_R"), ("U_L", "F_R")),
           objects=2, grasp="physics_grasp", source="delta.task_library_r34",
           status="raw+s0 success (09-07)"),
    Family("pillow_sheath", 2, ARMS_ALL, "colift", coupled_groups=(ARMS_F,), objects=1,
           grasp="kinematic", source="delta.task_library_r24",
           status="s0 success (alpha 1.0, never engaged)"),
    Family("profile_colift", 2, ARMS_ALL, "colift", coupled_groups=(ARMS_F,), objects=3,
           grasp="physics_grasp", source="delta.task_library_r35",
           status="raw per-object PASS (repeatable); s0 lifts+carries but topples at release (a31 asymmetry)",
           notes="two-hand carry of a 70 cm 4080 profile + U pair connector blocks"),
    # ---------------------------------------------------------------- category 3
    Family("reagent_pick", 3, ARMS_ALL, "none", objects=4, grasp="physics_grasp",
           source="delta.task_library_r34",
           status="raw+s0 success; U place fix verified 09-08 (1.3 / 1.7 cm)"),
    Family("pipe_sorting", 3, ARMS_ALL, "none", objects=4, grasp="physics_grasp",
           source="delta.task_library_r35",
           status="raw+s0 per-object PASS (video 53 / r35_pipe8, claw squeeze hints)"),
    Family("carton_packing", 3, ARMS_ALL, "none", objects=5, grasp="physics_grasp",
           source="delta.task_library_r35",
           status="cans version raw+s0 per-object PASS (video 55); pipe version retired (pinch-hinge swing)",
           notes="F_L packs two cans into the KLT tote, F_R and U pair sort in parallel"),
    Family("bin_packing", 3, ARMS_ALL, "colift", coupled_groups=(ARMS_F,), objects=3,
           grasp="physics_grasp", source="delta.task_library_r34",
           status="raw PASS (U place fix 09-08); s0 blocked by structural false brake (training side)",
           notes="F pair pack bottles then palm-clamp co-carry the bin -> category 3/2 mixed"),
    Family("tube_racking", 3, ARMS_ALL, "none", objects=4, grasp="physics_grasp",
           source="delta.task_library_r34",
           status="PARKED: fine insertion fails (pad opening 1.9 cm vs 2.6 cm tube)"),
    # ---------------------------------------------------------------- category 4
    Family("relay_chain", 4, ARMS_ALL, "relay",
           coupled_groups=(("F_L", "U_R"), ("U_R", "U_L"), ("U_L", "F_R")), objects=1,
           grasp="kinematic", source="delta.task_library_r24",
           status="raw+s0 success (kinematic attach relay); physical relay needs R32"),
    Family("four_lift", 4, ARMS_ALL, "colift", coupled_groups=(ARMS_F,), objects=3,
           grasp="kinematic", source="delta.task_library_r24",
           status="s0 success viol 4 (raw 8)"),
    Family("dual_pick_swap", 4, ARMS_ALL, "none", objects=4, grasp="kinematic",
           source="delta.task_library_r24",
           status="PARKED: U pair cross-line = DLS wrist-fold basin (needs joint spline / cuRobo)"),
    # v7 kinematic skill library (no objects) -- used in the training mix and the skill eval flow
    Family("mirror_sync", 4, ARMS_ALL, "none", grasp="none", source="skill_library v7",
           status="tier-0 pass", tags=("kinematic",)),
    Family("perimeter_sweep", 4, ARMS_ALL, "none", grasp="none", source="skill_library v7",
           status="tier-0 pass", tags=("kinematic",)),
    Family("tool_pass_chain", 4, ARMS_ALL, "relay", grasp="none", source="skill_library v7",
           status="tier-0 pass (1-2 arms idle per audit)", tags=("kinematic",)),
    Family("dual_carry_both", 4, ARMS_ALL, "colift", grasp="none", source="skill_library v7",
           status="tier-0 pass", tags=("kinematic",)),
    Family("sequential_center_grab", 4, ARMS_ALL, "contested", grasp="none", source="skill_library v7",
           status="tier-0 pass", tags=("kinematic",)),
    Family("bimanual_lift_rotate", 2, ARMS_F, "colift", grasp="none", source="skill_library v7",
           status="tier-0 pass", tags=("kinematic",)),
    Family("center_grab_cross", 2, ("F_L", "U_R"), "contested", grasp="none", source="skill_library v7",
           status="tier-0 pass", tags=("kinematic",)),
    Family("cross_pick_place_FU", 2, ("F_L", "U_R"), "handover", grasp="none", source="skill_library v7",
           status="tier-0 pass", tags=("kinematic",)),
    Family("cross_pick_place_UF", 2, ("U_L", "F_R"), "handover", grasp="none", source="skill_library v7",
           status="tier-0 pass", tags=("kinematic",)),
    Family("handover_FL_UR", 2, ("F_L", "U_R"), "handover", grasp="none", source="skill_library v7",
           status="tier-0 pass", tags=("kinematic",)),
    Family("handover_UR_FL", 2, ("U_R", "F_L"), "handover", grasp="none", source="skill_library v7",
           status="tier-0 pass", tags=("kinematic",)),
    Family("handover_high", 2, ("F_L", "U_R"), "handover", grasp="none", source="skill_library v7",
           status="tier-0 pass", tags=("kinematic",)),
    Family("handover_low", 2, ("F_L", "U_R"), "handover", grasp="none", source="skill_library v7",
           status="tier-0 pass", tags=("kinematic",)),
    # ---------------------------------------------------------------- category 5
    Family("l1", 5, ARMS_ALL, "none", grasp="none", source="delta.l1_random",
           status="training mix 0.15; OU joint jitter, no workspace intent", tags=("stream",)),
    Family("l1_ws", 5, ARMS_ALL, "none", grasp="none", source="delta.l1_workspace_v7",
           status="four-gate regular stream: centre-band roaming, 0.45 m/s, 2-5 s segments",
           notes="box volume ~8-10 % of the reach; wrist drifts in the position null space", tags=("stream",)),
    Family("l1_full", 5, ARMS_ALL, "none", grasp="none", source="delta.l1_coverage",
           status="training mix 0.25 (v7_cov); NOT in the four-gate battery",
           notes="octant LRU coverage + band 0.3 / deep 0.1 + 6D orientation + 3 speed tiers", tags=("stream",)),
    Family("directed", 5, ARMS_ALL, "contested", grasp="none", source="delta.l2_scenarios",
           status="four-gate brake metric uses head_on_crossing + handover_approach only",
           notes="8 scripted families: head_on_crossing, center_grab, handover_approach, table_slam, "
                 "sweep_across, chase, freeze_one, latency_spike", tags=("stream",)),
    Family("skill_replay", 5, ARMS_ALL, "none", grasp="none", source="delta.skill_replay",
           status="training mix 0.2; tempo/noise perturbed replays of the 13 kinematic skills", tags=("stream",)),
)

BY_NAME = {f.name: f for f in FAMILIES}
assert len(BY_NAME) == len(FAMILIES), "duplicate family name"


def by_category(cat: int) -> list:
    return [f for f in FAMILIES if f.category == cat]


def coverage_gaps() -> dict:
    """Owner-facing gap list per category (SKILL_LIBRARY_PLAN_20260908.md section 1)."""
    return {
        1: ["cross-row independent pairs (F_L + U_R on separate objects)",
            "mixed object shapes per pair (bottle + can, pipe + block)",
            "place targets deliberately near the opposite pair's box (near-but-no-conflict samples)"],
        2: ["U pair (DFX claw) co-lift",
            "physical cross-row handover (needs R32 (a)/(b) ruling)",
            "co-lift with an attitude constraint (level / upright) as the phase-sync eval case"],
        3: ["independent tasks with intentionally overlapping workspaces"],
        4: ["profile assembly: F pair holds the 4080 profile, U pair inserts connector blocks",
            "two pairs co-lifting one large item",
            "relay chain with physical handovers"],
        5: ["episode length 60 s -> 300-600 s",
            "all 8 scripted scenarios + l1_full in the battery (gate uses 2/8 and l1_ws)",
            "speed tiers up to the 0.06-0.08 rad/step command clip",
            "seed sweep (>=10) instead of the fixed 20260820",
            "pairwise collision-opportunity audit (6 arm pairs, margin < d_warn count per episode)",
            ">=1e3 episodes per config with 0/N + Clopper-Pearson bound",
            "real operator recordings (L4) for spectrum matching",
            "adversarial operator pi_adv (G2)"],
    }


def coverage_report() -> str:
    lines = []
    for cat, cname in CATEGORIES.items():
        fams = by_category(cat)
        lines.append(f"\n== {cat} {cname}: {len(fams)} families")
        for f in fams:
            grp = " groups=" + "|".join("+".join(g) for g in f.coupled_groups) if f.coupled_groups else ""
            lines.append(f"  - {f.name:22s} arms={','.join(f.arms):17s} coupling={f.coupling:9s}"
                         f" objs={f.objects} grasp={f.grasp or '-':14s}{grp}")
            lines.append(f"      status: {f.status}")
        for g in coverage_gaps()[cat]:
            lines.append(f"  GAP: {g}")
    n_phys = sum(1 for f in FAMILIES if f.grasp == "physics_grasp")
    n_kin = sum(1 for f in FAMILIES if f.grasp == "kinematic" or "kinematic" in f.tags)
    n_stream = sum(1 for f in FAMILIES if "stream" in f.tags)
    lines.append(f"\ntotal {len(FAMILIES)} families: {n_phys} physics-grasp tasks, "
                 f"{n_kin} kinematic, {n_stream} random streams")
    return "\n".join(lines)


if __name__ == "__main__":
    print(coverage_report())
