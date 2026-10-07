"""v7 skill trajectory recording (R14 S5): the T1/T2 13-family library
re-choreographed and re-planned for the UR5+RH56DFX scene.

Same three-stage pipeline as skill_record.py (design -> plan -> compose), same
storage format (SkillTrajectory npz, absolute joints @ 60 Hz), same replay /
noise-tier machinery -- only the geometry provider (make_v7_provider), the
choreography numbers and the floors change. v6 artifacts are untouched; output
lands in artifacts/skill_trajs_v7/.

v7 choreography retarget rules (probe-verified, see S5 report):
  - own-side targets shift with the base rows: F x += 0.198 (base 0.55 ->
    0.7482), U x -= 0.198; U staging y pulled inboard ~0.06 (pair spacing
    1.00 -> 0.8717).
  - meet / cross points stay near the x=0 seam. v7's shared band is only
    ~0.20 m wide but both arms reach it comfortably: flange-gap 0.30 meets
    audit at cross ~ +0.10-0.14 m (v5's wrist bottleneck read +0.033 at the
    same gap -- the 1.4964 m row spacing makes cross the EASY channel; the
    binding constraints in v7 are reachability and self/table).
  - low-point discipline: F flange z >= 1.00, U flange z >= 1.035 (DFX hand
    spans ~0.21 m past the wrist_3 flange), planar >= 0.33 m from the arm's
    own base disc (v5 lesson, re-verified: closer points fold the wrist into
    self-collision).

Floors: the v5 design floor table=0.003 is unattainable in v7 -- the ur5e-
reused shoulder-end sphere of upper_arm_link sits at a STRUCTURAL +1.7 mm
above the U table at every arm pose (UR5 d1=0.089 vs UR5e 0.1625; the S1
sphere fix targeted the wrist, not this shell). The hard deliverable gate
stays margin > 0 at every step; the v7 table floor drops to 0.0012 and the
supplemental audit reports the table channel with and without that
structural pair so real hand/forearm lows stay visible.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from safeduo.baselines.real_geometry_v7 import (
    make_v7_provider,
    min_margin_by_class4,
)
from safeduo.delta.skill_record import (
    CONTROL_DT,
    DELTA_HARD_CAP,
    DELTA_SOFT_CAP,
    Phase,
    SkillSpec,
    _margins_ok,
    compose_skill,
    solve_waypoint,
    validate_trajectory,
)
from safeduo.delta.skill_replay import SkillTrajectory
from safeduo.safety.types import ARM_KEYS

# structural facts of the v7 birth/scene (see module docstring)
V7_DESIGN_FLOORS = {"cross": 0.020, "self": 0.008, "table": 0.0012}
V7_IK_FLOORS = {"cross": 0.002, "self": 0.002, "table": 0.0010}

_MIRROR_ARM = {"F_L": "F_R", "F_R": "F_L", "U_L": "U_R", "U_R": "U_L"}
# R27 (2026-09-05): multi-seed IK. Default off = bit-identical legacy; the
# physical-grasp task builders opt in. The guarded DLS-IK is local: from the
# birth pose the F_R rod grasp folds the wrist onto the forearm and stalls at
# self ~1 mm although the mirrored F_L reaches the same pose with 23 mm
# (r27_rod_fr_seed.py). Waypoints are joint configurations and the phase
# motion is a joint-space cosine ease between them, so re-seeding the solve
# is legitimate as long as the ease from the previous waypoint stays safe.
IK_MULTISEED = {"on": False, "samples": 24}


def _seed_variants(q_prev: dict, arm: str) -> list:
    """Alternative start configurations for `arm`: the mirror arm's current
    joints as-is and with the odd joints sign-flipped (mirror-symmetric
    bases), plus a half-way blend toward them from the arm's own pose."""
    mirror = _MIRROR_ARM[arm]
    own = q_prev[arm]
    cand = []
    if mirror in q_prev and q_prev[mirror].shape == own.shape:
        m = q_prev[mirror]
        sign = torch.ones_like(m)
        sign[:, 0::2] = -1.0
        for seed in (m, m * sign):
            cand.append(seed.clone())
            cand.append(0.5 * (own + seed))
    return cand


def _ease_ok(provider, q_from: dict, q_to: dict, floors: dict, samples: int) -> tuple:
    """Every sample of the joint-space cosine ease q_from -> q_to clears floors."""
    worst = None
    for k in range(1, samples):
        s = 0.5 - 0.5 * math.cos(math.pi * k / samples)
        q = {a: q_from[a] + s * (q_to[a] - q_from[a]) for a in q_from}
        ok, mm = _margins_ok(provider, q, floors)
        if not ok:
            return False, mm
        worst = mm if worst is None else {c: min(worst[c], mm[c]) for c in mm}
    return True, worst


def solve_waypoint_multiseed(provider, q: dict, moving: dict, floors: dict = V7_IK_FLOORS,
                             ori_targets: "dict | None" = None, **kw) -> tuple:
    """solve_waypoint, then -- when IK_MULTISEED is on and an arm is blocked --
    retry that arm from _seed_variants and keep the first solution whose
    cosine ease from `q` is safe. info["seeded"] names the arm/variant used."""
    q_solved, info = solve_waypoint(provider, q, moving, floors=floors,
                                    ori_targets=ori_targets, **kw)
    if info.get("ok") or not IK_MULTISEED["on"]:
        return q_solved, info
    blocked = list((info.get("blocked") or {}).keys()) or list(moving.keys())
    verbose = IK_MULTISEED.get("verbose", False)
    oris = ori_targets or {}
    for arm in blocked:
        others = {a: t for a, t in moving.items() if a != arm}
        seeds = _seed_variants(q, arm)
        try:
            # the birth configuration: most designed poses were first solved
            # from it (R34 bin clamp after a load sequence fails from every
            # mirror seed but not from birth)
            birth = provider.default_q()[arm]
            seeds.append(birth.to(q[arm].device, dtype=q[arm].dtype).clone())
            seeds.append(0.5 * (q[arm] + seeds[-1]))
        except Exception:  # noqa: BLE001
            pass
        for k, seed in enumerate(seeds):
            q_seed = {a: v.clone() for a, v in q.items()}
            q_seed[arm] = seed
            # the blocked arm alone first (its guard then only sees its own
            # channels plus the frozen others), then the rest from that state
            q_try, info_try = solve_waypoint(provider, q_seed, {arm: moving[arm]}, floors=floors,
                                             ori_targets=({arm: oris[arm]} if arm in oris else None), **kw)
            if info_try.get("ok") and others:
                q_try, info_try = solve_waypoint(provider, q_try, others, floors=floors,
                                                 ori_targets=({a: oris[a] for a in others if a in oris} or None),
                                                 **kw)
            if not info_try.get("ok"):
                if verbose:
                    print(f"    multiseed {arm}:variant{k} solve failed "
                          f"{info_try.get('blocked') or info_try.get('residuals')}", flush=True)
                continue
            ease_ok, mm = _ease_ok(provider, q, q_try, floors, IK_MULTISEED["samples"])
            if not ease_ok:
                info_try["ease_blocked"] = mm
                if verbose:
                    print(f"    multiseed {arm}:variant{k} ease blocked {mm}", flush=True)
                continue
            info_try["seeded"] = f"{arm}:variant{k}"
            info_try["ease_worst"] = mm
            return q_try, info_try
    # last resort: sequential per-arm re-seeding -- each moving arm in turn
    # is solved alone from [current, birth, mirror variants, flipped birth]
    # with the other arms frozen at their already-solved configurations, then
    # the whole waypoint is ease-audited from q (a multi-arm regroup after a
    # long sequence: no single joint seed set solves every arm at once)
    try:
        birth = provider.default_q()
        q_acc = {a: v.clone() for a, v in q.items()}
        ok_all = True
        for arm in moving:
            sign = torch.ones_like(q[arm])
            sign[:, 0::2] = -1.0
            b = birth[arm].to(q[arm].device, dtype=q[arm].dtype)
            seeds = [q_acc[arm], b.clone(), b * sign] + _seed_variants(q_acc, arm)
            solved = False
            for seed in seeds:
                q_seed = {a: v.clone() for a, v in q_acc.items()}
                q_seed[arm] = seed.clone()
                q_try, info_try = solve_waypoint(provider, q_seed, {arm: moving[arm]}, floors=floors,
                                                 ori_targets=({arm: oris[arm]} if arm in oris else None), **kw)
                if info_try.get("ok"):
                    q_acc[arm] = q_try[arm]
                    solved = True
                    break
            if not solved:
                ok_all = False
                if verbose:
                    print(f"    multiseed sequential: {arm} unsolved {info_try.get('blocked') or info_try.get('residuals')}", flush=True)
                break
        if ok_all:
            ease_ok, mm = _ease_ok(provider, q, q_acc, floors, IK_MULTISEED["samples"])
            if ease_ok:
                info_seq = {"ok": True, "residuals": {}, "seeded": "sequential", "ease_worst": mm}
                _, info_seq["margins"] = _margins_ok(provider, q_acc, floors)
                return q_acc, info_seq
            if verbose:
                print(f"    multiseed sequential ease blocked {mm}", flush=True)
    except Exception as ex:  # noqa: BLE001
        if verbose:
            print(f"    multiseed sequential error {ex}", flush=True)
    return q_solved, info


# ---------------------------------------------------------------------------
# v7 choreography: the same 13 families, retargeted
# ---------------------------------------------------------------------------


def skill_specs_v7() -> dict:
    hold_all = {"F_L": "hold", "F_R": "hold", "U_L": "hold", "U_R": "hold"}

    # 1) forward handover F_L gives, U_R receives (lane y<0)
    handover = SkillSpec("handover_FL_UR", [
        Phase("approach", 3.5,
              targets={"F_L": (0.18, -0.40, 1.10), "U_R": (-0.16, -0.34, 1.10)},
              joint_moves={"F_R": {1: -0.06, 3: 0.05}, "U_L": {1: 0.05, 3: -0.05}},
              mode={"F_L": "plan", "U_R": "plan", "F_R": "lerp", "U_L": "lerp"}),
        Phase("meet", 2.0,
              targets={"F_L": (0.15, -0.40, 1.08), "U_R": (-0.15, -0.33, 1.08)},
              joint_moves={"F_R": {1: 0.03}, "U_L": {1: -0.03}},
              mode={"F_L": "plan", "U_R": "plan", "F_R": "lerp", "U_L": "lerp"}),
        Phase("transfer", 1.5, mode=dict(hold_all)),
        Phase("retreat", 3.0,
              targets={"F_L": (0.42, -0.48, 1.15), "U_R": (-0.38, -0.30, 1.12)},
              joint_moves={"F_R": {1: 0.03, 3: -0.05}, "U_L": {1: -0.02, 3: 0.05}},
              mode={"F_L": "plan", "U_R": "plan", "F_R": "lerp", "U_L": "lerp"}),
    ])

    # 2) U yields, F dual-arm carries
    dual_carry = SkillSpec("dual_carry_both", [
        Phase("yield_U", 2.5,
              targets={"U_L": (-0.42, 0.42, 1.16), "U_R": (-0.42, -0.42, 1.16)},
              mode={"U_L": "plan", "U_R": "plan", "F_L": "hold", "F_R": "hold"}),
        Phase("reach_F", 2.5,
              targets={"F_L": (0.55, -0.20, 1.01), "F_R": (0.55, 0.20, 1.01)},
              mode={"F_L": "plan", "F_R": "plan", "U_L": "hold", "U_R": "hold"}),
        Phase("grip", 1.0, mode=dict(hold_all)),
        Phase("lift", 1.5,
              targets={"F_L": (0.55, -0.20, 1.09), "F_R": (0.55, 0.20, 1.09)},
              mode={"F_L": "plan", "F_R": "plan", "U_L": "hold", "U_R": "hold"}),
        Phase("carry", 2.5,
              targets={"F_L": (0.36, -0.20, 1.09), "F_R": (0.36, 0.20, 1.09)},
              mode={"F_L": "plan", "F_R": "plan", "U_L": "hold", "U_R": "hold"}),
        Phase("setdown", 1.5,
              targets={"F_L": (0.36, -0.20, 1.03), "F_R": (0.36, 0.20, 1.03)},
              mode={"F_L": "plan", "F_R": "plan", "U_L": "hold", "U_R": "hold"}),
        Phase("release_up", 1.5,
              targets={"F_L": (0.36, -0.20, 1.16), "F_R": (0.36, 0.20, 1.16)},
              mode={"F_L": "plan", "F_R": "plan", "U_L": "hold", "U_R": "hold"}),
        Phase("retreat_return", 2.5,
              targets={"F_L": (0.46, -0.44, 1.28), "F_R": (0.46, 0.44, 1.28),
                       "U_L": (-0.32, 0.29, 1.12), "U_R": (-0.32, -0.29, 1.12)},
              mode={a: "plan" for a in ARM_KEYS}),
    ])

    # 3) staggered cross-line picks at the seam (F batch, then U batch)
    center_grab = SkillSpec("center_grab_cross", [
        Phase("yield_U", 2.5,
              targets={"U_L": (-0.44, 0.52, 1.16), "U_R": (-0.44, -0.46, 1.16)},
              mode={"U_L": "plan", "U_R": "plan", "F_L": "hold", "F_R": "hold"}),
        Phase("F_cross_reach", 3.0,
              targets={"F_L": (-0.03, -0.38, 1.06), "F_R": (-0.03, 0.28, 1.06)},
              mode={"F_L": "plan", "F_R": "plan", "U_L": "hold", "U_R": "hold"}),
        Phase("F_grab", 1.5,
              targets={"F_L": (-0.03, -0.38, 1.03), "F_R": (-0.03, 0.28, 1.03)},
              mode={"F_L": "plan", "F_R": "plan", "U_L": "hold", "U_R": "hold"}),
        Phase("F_lift", 1.2,
              # straight-z first: diagonal pull-back from the stretched cross
              # low folds the FR3 wrist (v5 lesson 4, re-hit at v7 depth)
              targets={"F_L": (-0.03, -0.38, 1.14), "F_R": (-0.03, 0.28, 1.14)},
              mode={"F_L": "plan", "F_R": "plan", "U_L": "hold", "U_R": "hold"}),
        Phase("F_return", 3.0,
              targets={"F_L": (0.42, -0.46, 1.20), "F_R": (0.42, 0.46, 1.20)},
              mode={"F_L": "plan", "F_R": "plan", "U_L": "hold", "U_R": "hold"}),
        Phase("U_cross_reach", 3.0,
              targets={"U_L": (0.03, 0.38, 1.06), "U_R": (0.03, -0.20, 1.06)},
              mode={"U_L": "plan", "U_R": "plan", "F_L": "hold", "F_R": "hold"}),
        Phase("U_grab", 1.5,
              targets={"U_L": (0.03, 0.38, 1.035), "U_R": (0.03, -0.20, 1.035)},
              mode={"U_L": "plan", "U_R": "plan", "F_L": "hold", "F_R": "hold"}),
        Phase("U_lift", 1.2,
              targets={"U_L": (0.03, 0.38, 1.12), "U_R": (0.03, -0.20, 1.12)},
              mode={"U_L": "plan", "U_R": "plan", "F_L": "hold", "F_R": "hold"}),
        Phase("U_return", 3.0,
              targets={"U_L": (-0.36, 0.49, 1.10), "U_R": (-0.36, -0.27, 1.10)},
              mode={"U_L": "plan", "U_R": "plan", "F_L": "hold", "F_R": "hold"}),
        Phase("settle", 1.0, mode=dict(hold_all)),
    ])

    # 4) reverse handover: U_R gives first, F_L receives (lane y<0)
    handover_ur_fl = SkillSpec("handover_UR_FL", [
        Phase("give_approach", 3.0,
              targets={"U_R": (-0.14, -0.34, 1.12)},
              joint_moves={"F_R": {1: -0.05, 3: 0.04}, "U_L": {1: 0.04, 3: -0.04}},
              mode={"U_R": "plan", "F_R": "lerp", "U_L": "lerp", "F_L": "hold"}),
        Phase("recv_approach", 2.5,
              targets={"F_L": (0.16, -0.40, 1.12)},
              mode={"F_L": "plan", "U_R": "hold", "F_R": "hold", "U_L": "hold"}),
        Phase("transfer", 1.5, mode=dict(hold_all)),
        Phase("give_retreat", 2.5,
              targets={"U_R": (-0.44, -0.33, 1.18)},
              mode={"U_R": "plan", "F_L": "hold", "F_R": "hold", "U_L": "hold"}),
        Phase("recv_return", 2.5,
              targets={"F_L": (0.44, -0.50, 1.30)},
              joint_moves={"F_R": {1: 0.05, 3: -0.04}, "U_L": {1: -0.04, 3: 0.04}},
              mode={"F_L": "plan", "F_R": "lerp", "U_L": "lerp", "U_R": "hold"}),
    ])

    # 5) high-layer handover on the y>0 diagonal, sequenced (U_L parks first)
    handover_high = SkillSpec("handover_high", [
        Phase("u_approach", 3.0,
              targets={"U_L": (-0.17, 0.35, 1.26)},
              joint_moves={"F_L": {1: -0.05, 3: 0.04}, "U_R": {1: -0.04, 3: 0.04}},
              mode={"U_L": "plan", "F_L": "lerp", "U_R": "lerp", "F_R": "hold"}),
        Phase("f_approach", 2.5,
              targets={"F_R": (0.13, 0.42, 1.26)},
              mode={"F_R": "plan", "F_L": "hold", "U_L": "hold", "U_R": "hold"}),
        Phase("transfer", 1.5, mode=dict(hold_all)),
        Phase("f_retreat", 2.0,
              targets={"F_R": (0.42, 0.48, 1.35)},
              mode={"F_R": "plan", "F_L": "hold", "U_L": "hold", "U_R": "hold"}),
        Phase("u_retreat", 2.5,
              targets={"U_L": (-0.36, 0.44, 1.24)},
              joint_moves={"F_L": {1: 0.05, 3: -0.04}, "U_R": {1: 0.04, 3: -0.04}},
              mode={"U_L": "plan", "F_L": "lerp", "U_R": "lerp", "F_R": "hold"}),
    ])

    # 6) low-layer handover (meet z=1.04; straight-z lift before pull-back)
    handover_low = SkillSpec("handover_low", [
        Phase("approach", 3.5,
              targets={"F_L": (0.18, -0.40, 1.10), "U_R": (-0.16, -0.33, 1.10)},
              joint_moves={"F_R": {1: -0.05}, "U_L": {1: 0.04}},
              mode={"F_L": "plan", "U_R": "plan", "F_R": "lerp", "U_L": "lerp"}),
        Phase("descend_meet", 2.5,
              targets={"F_L": (0.15, -0.40, 1.04), "U_R": (-0.16, -0.34, 1.04)},
              mode={"F_L": "plan", "U_R": "plan", "F_R": "hold", "U_L": "hold"}),
        Phase("transfer", 1.5, mode=dict(hold_all)),
        Phase("lift", 1.5,
              targets={"F_L": (0.15, -0.40, 1.14), "U_R": (-0.16, -0.34, 1.14)},
              mode={"F_L": "plan", "U_R": "plan", "F_R": "hold", "U_L": "hold"}),
        Phase("retreat", 2.5,
              targets={"F_L": (0.43, -0.48, 1.25), "U_R": (-0.40, -0.28, 1.16)},
              joint_moves={"F_R": {1: 0.05}, "U_L": {1: -0.04}},
              mode={"F_L": "plan", "U_R": "plan", "F_R": "lerp", "U_L": "lerp"}),
    ])

    # 7) F dual-arm lift + 30 deg yaw about the carried midpoint (0.53, 0)
    lift_rotate = SkillSpec("bimanual_lift_rotate", [
        Phase("yield_U", 2.5,
              targets={"U_L": (-0.42, 0.42, 1.16), "U_R": (-0.42, -0.42, 1.16)},
              mode={"U_L": "plan", "U_R": "plan", "F_L": "hold", "F_R": "hold"}),
        Phase("reach", 2.5,
              targets={"F_L": (0.53, -0.20, 1.01), "F_R": (0.53, 0.20, 1.01)},
              mode={"F_L": "plan", "F_R": "plan", "U_L": "hold", "U_R": "hold"}),
        Phase("grip", 1.0, mode=dict(hold_all)),
        Phase("lift", 1.5,
              targets={"F_L": (0.53, -0.20, 1.12), "F_R": (0.53, 0.20, 1.12)},
              mode={"F_L": "plan", "F_R": "plan", "U_L": "hold", "U_R": "hold"}),
        Phase("rotate_cw", 2.5,
              targets={"F_L": (0.63, -0.173, 1.12), "F_R": (0.43, 0.173, 1.12)},
              mode={"F_L": "plan", "F_R": "plan", "U_L": "hold", "U_R": "hold"}),
        Phase("rotate_back", 2.5,
              targets={"F_L": (0.53, -0.20, 1.12), "F_R": (0.53, 0.20, 1.12)},
              mode={"F_L": "plan", "F_R": "plan", "U_L": "hold", "U_R": "hold"}),
        Phase("setdown", 1.5,
              targets={"F_L": (0.53, -0.20, 1.03), "F_R": (0.53, 0.20, 1.03)},
              mode={"F_L": "plan", "F_R": "plan", "U_L": "hold", "U_R": "hold"}),
        Phase("release_up", 1.5,
              targets={"F_L": (0.53, -0.20, 1.14), "F_R": (0.53, 0.20, 1.14)},
              mode={"F_L": "plan", "F_R": "plan", "U_L": "hold", "U_R": "hold"}),
        Phase("retreat_return", 2.5,
              targets={"F_L": (0.46, -0.44, 1.28), "F_R": (0.46, 0.44, 1.28),
                       "U_L": (-0.32, 0.29, 1.12), "U_R": (-0.32, -0.29, 1.12)},
              mode={a: "plan" for a in ARM_KEYS}),
    ])

    # 8) cross-machine pick&place F->U (lane y<0, U_R yields; staged descend)
    cross_fu = SkillSpec("cross_pick_place_FU", [
        Phase("yield_UR", 2.5,
              targets={"U_R": (-0.44, -0.44, 1.18)},
              mode={"U_R": "plan", "F_L": "hold", "F_R": "hold", "U_L": "hold"}),
        Phase("pre_reach", 2.0,
              targets={"F_L": (0.54, -0.22, 1.15)},
              mode={"F_L": "plan", "F_R": "hold", "U_L": "hold", "U_R": "hold"}),
        Phase("pick_down", 1.5,
              targets={"F_L": (0.54, -0.22, 1.01)},
              mode={"F_L": "plan", "F_R": "hold", "U_L": "hold", "U_R": "hold"}),
        Phase("lift", 1.5,
              targets={"F_L": (0.54, -0.22, 1.12)},
              mode={"F_L": "plan", "F_R": "hold", "U_L": "hold", "U_R": "hold"}),
        Phase("carry_cross", 3.5,
              targets={"F_L": (-0.04, -0.32, 1.10)},
              mode={"F_L": "plan", "F_R": "hold", "U_L": "hold", "U_R": "hold"}),
        Phase("place", 1.5,
              targets={"F_L": (-0.04, -0.32, 1.04)},
              mode={"F_L": "plan", "F_R": "hold", "U_L": "hold", "U_R": "hold"}),
        Phase("release_up", 1.5,
              targets={"F_L": (-0.04, -0.32, 1.15)},
              mode={"F_L": "plan", "F_R": "hold", "U_L": "hold", "U_R": "hold"}),
        Phase("return", 3.0,
              targets={"F_L": (0.44, -0.48, 1.25), "U_R": (-0.38, -0.30, 1.16)},
              mode={"F_L": "plan", "U_R": "plan", "F_R": "hold", "U_L": "hold"}),
    ])

    # 9) cross-machine pick&place U->F (lane y>0, F_R yields)
    cross_uf = SkillSpec("cross_pick_place_UF", [
        Phase("yield_FR", 2.0,
              targets={"F_R": (0.50, 0.56, 1.30)},
              mode={"F_R": "plan", "F_L": "hold", "U_L": "hold", "U_R": "hold"}),
        Phase("pre_reach", 2.5,
              targets={"U_L": (-0.42, 0.40, 1.15)},
              mode={"U_L": "plan", "F_L": "hold", "F_R": "hold", "U_R": "hold"}),
        Phase("pick_down", 1.5,
              targets={"U_L": (-0.42, 0.40, 1.04)},
              mode={"U_L": "plan", "F_L": "hold", "F_R": "hold", "U_R": "hold"}),
        Phase("lift", 1.5,
              targets={"U_L": (-0.42, 0.40, 1.12)},
              mode={"U_L": "plan", "F_L": "hold", "F_R": "hold", "U_R": "hold"}),
        Phase("carry_cross", 3.5,
              targets={"U_L": (0.04, 0.40, 1.08)},
              mode={"U_L": "plan", "F_L": "hold", "F_R": "hold", "U_R": "hold"}),
        Phase("place", 1.5,
              targets={"U_L": (0.04, 0.40, 1.045)},
              mode={"U_L": "plan", "F_L": "hold", "F_R": "hold", "U_R": "hold"}),
        Phase("release_up", 1.5,
              targets={"U_L": (0.04, 0.40, 1.14)},
              mode={"U_L": "plan", "F_L": "hold", "F_R": "hold", "U_R": "hold"}),
        Phase("return", 3.0,
              targets={"U_L": (-0.36, 0.46, 1.18), "F_R": (0.42, 0.48, 1.32)},
              mode={"U_L": "plan", "F_R": "plan", "F_L": "hold", "U_R": "hold"}),
    ])

    # 10) strict turn-taking at the seam (one arm near x=0 at a time)
    def _turn(arm, reach, dip_z, back):
        others = {a: "hold" for a in ARM_KEYS if a != arm}
        return [
            Phase(f"{arm}_reach", 2.5, targets={arm: reach},
                  mode={arm: "plan", **others}),
            Phase(f"{arm}_dip", 1.0, targets={arm: (reach[0], reach[1], dip_z)},
                  mode={arm: "plan", **others}),
            Phase(f"{arm}_return", 2.5, targets={arm: back},
                  mode={arm: "plan", **others}),
        ]

    seq_center = SkillSpec("sequential_center_grab",
        [Phase("stage_U", 2.5,
               targets={"U_L": (-0.44, 0.46, 1.18), "U_R": (-0.44, -0.44, 1.18)},
               mode={"U_L": "plan", "U_R": "plan", "F_L": "hold", "F_R": "hold"})]
        + _turn("F_L", (-0.02, -0.40, 1.06), 1.03, (0.44, -0.48, 1.22))
        + _turn("U_R", (0.02, -0.24, 1.06), 1.035, (-0.38, -0.27, 1.16))
        + _turn("F_R", (-0.02, 0.32, 1.06), 1.03, (0.44, 0.46, 1.22))
        + _turn("U_L", (0.02, 0.46, 1.06), 1.035, (-0.38, 0.49, 1.16)))

    # 11) simultaneous low perimeter sweep in diverging quadrant lanes.
    #     v5 primitive discipline kept: LOW moves only along x, lateral
    #     shifts at mid z (1.18), descents pure-z, low points >= 0.33 m
    #     planar from the own base disc.
    perimeter = SkillSpec("perimeter_sweep", [
        Phase("spread", 3.0,
              targets={"F_L": (0.54, -0.20, 1.18), "F_R": (0.54, 0.20, 1.18),
                       "U_L": (-0.42, 0.40, 1.18), "U_R": (-0.44, -0.20, 1.18)},
              mode={a: "plan" for a in ARM_KEYS}),
        Phase("descend", 1.5,
              targets={"F_L": (0.54, -0.20, 1.05), "F_R": (0.54, 0.20, 1.05),
                       "U_L": (-0.42, 0.40, 1.06), "U_R": (-0.44, -0.20, 1.06)},
              mode={a: "plan" for a in ARM_KEYS}),
        Phase("sweep_seam", 3.0,
              targets={"F_L": (0.36, -0.20, 1.05), "F_R": (0.36, 0.20, 1.05),
                       "U_L": (-0.28, 0.40, 1.06), "U_R": (-0.30, -0.20, 1.06)},
              mode={a: "plan" for a in ARM_KEYS}),
        Phase("lift_up", 1.5,
              targets={"F_L": (0.36, -0.20, 1.18), "F_R": (0.36, 0.20, 1.18),
                       "U_L": (-0.28, 0.40, 1.18), "U_R": (-0.30, -0.20, 1.18)},
              mode={a: "plan" for a in ARM_KEYS}),
        Phase("shift_outer", 2.5,
              targets={"F_L": (0.42, -0.58, 1.18), "F_R": (0.42, 0.58, 1.18),
                       "U_L": (-0.40, 0.58, 1.18), "U_R": (-0.44, -0.46, 1.20)},
              mode={a: "plan" for a in ARM_KEYS}),
        Phase("dip_outer", 1.5,
              targets={"F_L": (0.42, -0.58, 1.05), "F_R": (0.42, 0.58, 1.05),
                       "U_L": (-0.40, 0.58, 1.06), "U_R": (-0.44, -0.46, 1.06)},
              mode={a: "plan" for a in ARM_KEYS}),
        Phase("corner_in", 2.0,
              targets={"F_L": (0.34, -0.58, 1.05), "F_R": (0.34, 0.58, 1.05),
                       "U_L": (-0.32, 0.58, 1.06), "U_R": (-0.32, -0.46, 1.06)},
              mode={a: "plan" for a in ARM_KEYS}),
        Phase("lift_return", 2.5,
              targets={"F_L": (0.42, -0.50, 1.25), "F_R": (0.42, 0.50, 1.25),
                       "U_L": (-0.30, 0.55, 1.18), "U_R": (-0.30, -0.31, 1.18)},
              mode={a: "plan" for a in ARM_KEYS}),
    ])

    # 12) three-arm relay F_L -> U_R -> F_R (U_R carries across y -0.32 -> 0)
    tool_chain = SkillSpec("tool_pass_chain", [
        Phase("meet1_approach", 3.0,
              targets={"F_L": (0.16, -0.38, 1.12), "U_R": (-0.15, -0.32, 1.12)},
              joint_moves={"F_R": {1: -0.04}},
              mode={"F_L": "plan", "U_R": "plan", "F_R": "lerp", "U_L": "hold"}),
        Phase("transfer1", 1.2, mode=dict(hold_all)),
        Phase("split1", 2.5,
              targets={"F_L": (0.46, -0.50, 1.26), "U_R": (-0.36, -0.08, 1.14)},
              mode={"F_L": "plan", "U_R": "plan", "F_R": "hold", "U_L": "hold"}),
        Phase("meet2_approach", 2.5,
              targets={"F_R": (0.14, 0.06, 1.14), "U_R": (-0.16, 0.00, 1.14)},
              mode={"F_R": "plan", "U_R": "plan", "F_L": "hold", "U_L": "hold"}),
        Phase("transfer2", 1.2, mode=dict(hold_all)),
        Phase("split2", 2.5,
              targets={"U_R": (-0.38, -0.26, 1.14), "F_R": (0.42, 0.30, 1.24)},
              mode={"U_R": "plan", "F_R": "plan", "F_L": "hold", "U_L": "hold"}),
        Phase("settle", 1.5,
              targets={"F_R": (0.44, 0.46, 1.32)},
              joint_moves={"F_L": {1: 0.04}},
              mode={"F_R": "plan", "F_L": "lerp", "U_L": "hold", "U_R": "hold"}),
    ])

    # 13) mirrored synchronized advance + lateral sways (four arms at once)
    mirror = SkillSpec("mirror_sync", [
        Phase("advance", 3.5,
              targets={"F_L": (0.20, -0.38, 1.14), "U_R": (-0.20, -0.36, 1.14),
                       "F_R": (0.20, 0.38, 1.14), "U_L": (-0.20, 0.36, 1.14)},
              mode={a: "plan" for a in ARM_KEYS}),
        Phase("close_in", 1.5,
              targets={"F_L": (0.15, -0.38, 1.14), "U_R": (-0.15, -0.36, 1.14),
                       "F_R": (0.15, 0.38, 1.14), "U_L": (-0.15, 0.36, 1.14)},
              mode={a: "plan" for a in ARM_KEYS}),
        Phase("sway_plus", 2.0,
              targets={"F_L": (0.15, -0.28, 1.14), "U_R": (-0.15, -0.26, 1.14),
                       "F_R": (0.15, 0.48, 1.14), "U_L": (-0.15, 0.46, 1.14)},
              mode={a: "plan" for a in ARM_KEYS}),
        Phase("sway_minus", 2.5,
              targets={"F_L": (0.15, -0.48, 1.14), "U_R": (-0.15, -0.46, 1.14),
                       "F_R": (0.15, 0.28, 1.14), "U_L": (-0.15, 0.26, 1.14)},
              mode={a: "plan" for a in ARM_KEYS}),
        Phase("recenter", 1.5,
              targets={"F_L": (0.15, -0.38, 1.14), "U_R": (-0.15, -0.36, 1.14),
                       "F_R": (0.15, 0.38, 1.14), "U_L": (-0.15, 0.36, 1.14)},
              mode={a: "plan" for a in ARM_KEYS}),
        Phase("withdraw", 2.5,
              targets={"F_L": (0.42, -0.50, 1.32), "U_R": (-0.32, -0.32, 1.18),
                       "F_R": (0.42, 0.50, 1.32), "U_L": (-0.32, 0.32, 1.18)},
              mode={a: "plan" for a in ARM_KEYS}),
    ])

    return {s.name: s for s in (
        handover, dual_carry, center_grab, handover_ur_fl, handover_high,
        handover_low, lift_rotate, cross_fu, cross_uf, seq_center,
        perimeter, tool_chain, mirror)}


# ---------------------------------------------------------------------------
# stage A/C with v7 floors
# ---------------------------------------------------------------------------


def design_skill_v7(provider, spec: SkillSpec, verbose: bool = True) -> dict:
    """skill_record.design_skill with the v7 floors injected (the v5 body
    hardcodes DESIGN_FLOORS; table 0.003 is structurally unattainable in v7,
    see module docstring)."""
    q = provider.default_q()
    wps = {a: [q[a][0].tolist()] for a in ARM_KEYS}
    phase_rows = []
    for ph in spec.phases:
        moving = dict(ph.targets)
        q_next = {a: v.clone() for a, v in q.items()}
        for a, off in ph.joint_moves.items():
            for j, dv in off.items():
                q_next[a][0, int(j)] += float(dv)
        if moving:
            # R23：相位带 ori_targets（抓取类）时走 6 维位姿 IK；不带 =
            # 纯位置路径逐位不变（13 族技能库照旧）
            q_solved, info = solve_waypoint_multiseed(provider, q_next, moving,
                                                      floors=V7_IK_FLOORS,
                                                      ori_targets=(ph.ori_targets or None))
            if not info["ok"]:
                raise RuntimeError(
                    f"{spec.name}/{ph.name}: IK blocked or residual too big: {info}")
            if verbose and info.get("seeded"):
                print(f"  [{spec.name}] {ph.name}: re-seeded {info['seeded']} "
                      f"ease_worst {info.get('ease_worst')}")
        else:
            q_solved, info = q_next, {
                "ok": True, "residuals": {},
                "margins": _margins_ok(provider, q_next, V7_DESIGN_FLOORS)[1]}
        ok, mm = _margins_ok(provider, q_solved, V7_DESIGN_FLOORS)
        if not ok:
            raise RuntimeError(f"{spec.name}/{ph.name}: waypoint below floors {mm}")
        if verbose:
            print(f"  [{spec.name}] {ph.name}: margins {mm} "
                  f"residuals {info['residuals']}")
        for a in ARM_KEYS:
            wps[a].append(q_solved[a][0].tolist())
        phase_rows.append({"name": ph.name, "dur": ph.dur,
                           "mode": {a: ph.mode.get(a, "hold") for a in ARM_KEYS},
                           "margins_at_boundary": mm})
        q = q_solved
    return {"skill": spec.name, "dt": CONTROL_DT, "phases": phase_rows,
            "waypoints": wps,
            "meta": {"layout": "v7", "birth_pose": "duo_env_v7",
                     "floors": V7_DESIGN_FLOORS}}


def audit_v7_channels(provider, traj: SkillTrajectory) -> dict:
    """Supplemental four-channel margin sweep (cross / self_F / self_U /
    table) + table with the structural U upper_arm shoulder-shell pairs
    excluded (they read a constant ~+1.7 mm at ANY pose and would mask real
    hand/forearm lows)."""
    d_all_rows = []
    T = traj.n_steps
    for t in range(T + 1):
        q = {a: torch.tensor(traj.q[a][t: t + 1], dtype=torch.float32)
             for a in ARM_KEYS}
        d_all, _, _ = provider._all_margins(provider.fk_all(q)["centers"])
        d_all_rows.append(d_all[0])
    D = torch.stack(d_all_rows)                       # (T+1, P)
    names = provider.sph.qualified_names
    is_self = provider.pair_class == 1.0
    is_u = provider.pair_arm_i >= 2
    is_table = provider.pair_class == 2.0
    ua_shell = torch.tensor(
        [names[int(i)].endswith("/upper_arm_link") for i in provider.pair_sph_i])
    masks = {
        "cross": provider.pair_class == 0.0,
        "self_F": is_self & ~is_u,
        "self_U": is_self & is_u,
        "table": is_table,
        "table_excl_shoulder": is_table & ~(ua_shell & is_u),
    }
    out = {}
    for k, m in masks.items():
        if not m.any():
            out[k] = None
            continue
        vals = D[:, m].amin(dim=1)
        out[k] = {"min_mm": round(float(vals.min()) * 1000, 1),
                  "argmin_step": int(vals.argmin()),
                  "rest_mm": round(float(vals[0]) * 1000, 1)}
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: "list | None" = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["design", "compose", "validate", "manifest"])
    ap.add_argument("--out", default="artifacts/skill_trajs_v7")
    ap.add_argument("--skill", default=None)
    ap.add_argument("--segments", default=None,
                    help="segments_<skill>.npz dir (compose); omit for fallback")
    ap.add_argument("--traj", default=None)
    args = ap.parse_args(argv)

    provider = make_v7_provider(1, device="cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = skill_specs_v7()
    names = [args.skill] if args.skill else list(specs)

    if args.stage == "design":
        for name in names:
            doc = design_skill_v7(provider, specs[name])
            p = out / f"waypoints_{name}.json"
            p.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
            print(f"wrote {p}")
    elif args.stage == "compose":
        n_fail = 0
        for name in names:
            doc = json.loads((out / f"waypoints_{name}.json").read_text())
            seg = None
            if args.segments:
                sp = Path(args.segments) / f"segments_{name}.npz"
                if sp.exists():
                    seg = dict(np.load(sp))
            traj = compose_skill(doc, seg)
            traj.meta["layout"] = "v7"
            traj.meta["birth_pose"] = "duo_env_v7"
            rep = validate_trajectory(provider, traj, floors=V7_DESIGN_FLOORS)
            rep["channels4"] = audit_v7_channels(provider, traj)
            traj.meta["validation"] = rep
            fp = out / f"skill_{name}.npz"
            traj.save(fp)
            (out / f"{name}_report.json").write_text(
                json.dumps(rep, indent=1, ensure_ascii=False))
            print(f"{name}: {json.dumps(rep, ensure_ascii=False)}")
            if "FAIL" in rep:
                n_fail += 1
        return 1 if n_fail else 0
    elif args.stage == "validate":
        traj = SkillTrajectory.load(args.traj)
        rep = validate_trajectory(provider, traj, floors=V7_DESIGN_FLOORS)
        rep["channels4"] = audit_v7_channels(provider, traj)
        print(json.dumps(rep, indent=1))
    else:  # manifest
        entries = []
        for name in sorted(specs):
            fp = out / f"skill_{name}.npz"
            if not fp.exists():
                print(f"SKIP {name}: no npz")
                continue
            traj = SkillTrajectory.load(fp)
            rep = traj.meta.get("validation", {})
            entries.append({
                "skill": name, "file": fp.name,
                "steps": traj.n_steps, "duration_s": round(traj.duration, 3),
                "hard_gate_margin_gt0": rep.get("hard_gate_margin_gt0"),
                "min_margin_mm": {k: round(v * 1000, 1)
                                  for k, v in rep.get("min_margin", {}).items()},
                "argmin_step": rep.get("argmin_step"),
                "max_delta_per_arm": rep.get("max_delta_per_arm"),
                "planner": traj.meta.get("planner"),
            })
        man = {
            "schema": "skill_library_manifest_v1",
            "date": "2026-08-19",
            "layout": "v7",
            "birth_pose": "duo_env_v7",
            "dt": CONTROL_DT,
            "n_skills": len(entries),
            "hard_gate": "sphere margin > 0 at every grid step "
                         "(make_v7_provider audit, real_v7 spheres)",
            "noise_tiers": "SkillNoiseParams.tier(0-3); start/goal "
                           "randomization via start_frac/end_frac",
            "consumers": "v7 skill flow: endurance_eval --flow skill; "
                         "skill_success with V7SkillFK + skill_success_v7.yaml",
            "skills": entries,
        }
        (out / "skills_manifest.json").write_text(
            json.dumps(man, indent=1, ensure_ascii=False))
        print(f"wrote {out / 'skills_manifest.json'} ({len(entries)} skills)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
