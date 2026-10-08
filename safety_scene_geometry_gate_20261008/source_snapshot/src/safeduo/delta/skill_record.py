"""Skill trajectory recording pipeline (T1 layer, spec S1).

Three stages -- CPU design and validation are local & authoritative, GPU
smoothing is a server fragment job (see artifacts/skill_trajs/README_T1.md):

  design   (local CPU)  choreography EE targets -> per-phase joint waypoints
                        via damped-least-squares IK on RealGeometryProvider
                        (v5 scene), with a margin guard at every IK step.
                        Emits waypoints_<skill>.json.
  plan     (server GPU) cuRobo plan_cspace connects consecutive waypoints per
                        arm into smooth self-collision-checked joint segments
                        (skill_plan_server.py, venv curobo_t1). Emits
                        segments_<skill>.npz. Arms/phases marked "lerp"/"hold"
                        skip planning (cosine-eased interpolation instead).
  compose  (local CPU)  segments -> uniform control-dt grid retimed to the
                        choreography phase durations, concatenated, margin-
                        validated per step (min_margin_by_class > 0 hard gate
                        + design floors as soft report), delta stats audited,
                        written as SkillTrajectory npz.

Design floors (nominal trajectory should not merely avoid touching, it should
stay clear of the braking boundary so the safety layer is quiet on clean
replay): cross >= 25 mm, self >= 12 mm, table >= 12 mm. The hard deliverable
gate from the spec is margin > 0 at every step; floors are reported per skill.

Frame conventions: EE targets are world-frame flange positions (the provider's
t_flange); orientation is left free (position-only DLS-IK) -- intent streams
do not constrain tool orientation, and cuRobo receives joint-space goals so
no tool-frame agreement between provider and cuRobo is ever needed. Provider
FK and cuRobo FK were cross-checked bit-close at the v4.1 birth pose (JAKA EE
base-frame [0.564, 0.202, 0.389] both sides; see T1_PROGRESS 19:30).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from safeduo.delta.skill_replay import SkillTrajectory
from safeduo.safety.types import ARM_KEYS, DOF_OF, SceneState

CONTROL_DT = 1.0 / 60.0          # duo_env control step (0.008333 * decimation 2)
# Floors calibrated to the v5 scene's own birth readings (cross 129.9 / self
# 23.1 / table 19.0 mm, vs baked d_min cross/self 13 / table 20 mm): table
# proximity below d_min_table is a birth-pose fact of life, so the floors for
# self/table are "stay clearly positive", not "stay above braking". The hard
# deliverable gate stays margin > 0; time spent below floors is reported.
DESIGN_FLOORS = {"cross": 0.020, "self": 0.008, "table": 0.003}
DELTA_SOFT_CAP = 0.02            # rad/step design target (glove ~0.015, demo 0.03)
DELTA_HARD_CAP = 0.03            # composer refuses above this


# ---------------------------------------------------------------------------
# choreography spec
# ---------------------------------------------------------------------------


@dataclass
class Phase:
    """One synchronized phase: every arm reaches its waypoint at the boundary.

    mode per arm: "plan" (cuRobo segment), "lerp" (cosine-eased joint interp,
    for small sways), "hold" (stay at previous waypoint)."""

    name: str
    dur: float
    targets: dict = field(default_factory=dict)   # arm -> world EE xyz (plan/lerp)
    joint_moves: dict = field(default_factory=dict)  # arm -> joint offset (sway)
    mode: dict = field(default_factory=dict)      # arm -> plan|lerp|hold
    # R23 姿态目标（可选）：arm -> 3x3 世界系 flange 旋转（嵌套 list/tuple）。
    # 为空 = 纯位置 IK，历史 13 族技能行为逐位不变；抓取类相位给了它才会
    # 走 6 维位姿 IK（穿模根因二：position-only IK 手掌姿态不对齐物体）。
    ori_targets: dict = field(default_factory=dict)


@dataclass
class SkillSpec:
    name: str
    phases: list


def _sway(arm_off: dict) -> dict:
    """Helper: joint-space sway targets for idle arms (offsets from current)."""
    return dict(arm_off)


def skill_specs() -> dict:
    """The three prototype skills (spec-named families).

    EE numbers are v5-world coordinates chosen off the v4.1 birth EEs
    (F_L [.20,-.52,1.40] F_R [.20,.48,1.40] U_L [.01,.70,1.21] U_R [.01,-.30,1.21]);
    tables top z=0.80 spanning x in [-0.8, 0.8] gap=0.
    """
    hold_all = {"F_L": "hold", "F_R": "hold", "U_L": "hold", "U_R": "hold"}

    # Z conventions learned from the margin diagnostics (T1_PROGRESS 20:0x):
    # the F2 hand hangs ~0.19 m below the flange. JAKA (U) arms keep a
    # permanent 19 mm link2-to-table floor and their hand points down, so U
    # flange stays >= 1.00 in low phases; FR3 (F) hand pose differs and its
    # permanent floor is fr3_link1 at 104 mm, so F may reach lower.
    handover = SkillSpec("handover_FL_UR", [
        Phase("approach", 3.0,
              targets={"F_L": (0.16, -0.40, 1.10), "U_R": (-0.14, -0.34, 1.10)},
              joint_moves={"F_R": {1: -0.06, 3: 0.05}, "U_L": {1: 0.05, 3: -0.05}},
              mode={"F_L": "plan", "U_R": "plan", "F_R": "lerp", "U_L": "lerp"}),
        Phase("meet", 2.0,
              # 0.30 m flange gap = wrist pair (fr3_link6<->jaka_link5, the
              # true cross bottleneck, NOT the hands) at ~33 mm margin
              targets={"F_L": (0.12, -0.40, 1.08), "U_R": (-0.18, -0.33, 1.08)},
              joint_moves={"F_R": {1: 0.03}, "U_L": {1: -0.03}},
              mode={"F_L": "plan", "U_R": "plan", "F_R": "lerp", "U_L": "lerp"}),
        Phase("transfer", 1.5, mode=dict(hold_all)),
        Phase("retreat", 3.0,
              targets={"F_L": (0.22, -0.48, 1.15), "U_R": (-0.18, -0.30, 1.12)},
              joint_moves={"F_R": {1: 0.03, 3: -0.05}, "U_L": {1: -0.02, 3: 0.05}},
              mode={"F_L": "plan", "U_R": "plan", "F_R": "lerp", "U_L": "lerp"}),
    ])

    # Choreography lesson (margin diagnostics, T1_PROGRESS): simultaneous
    # four-arm convergence on the center band self/cross-collides -- REAL
    # cooperation yields lanes. dual_carry = U yields, F dual-arm carries, U
    # returns; center_grab = staggered cross-line picks (F first, then U).
    # Simultaneous head-on convergence is deliberately L2's conflict job,
    # not the T1 safe-skill layer.
    dual_carry = SkillSpec("dual_carry_both", [
        Phase("yield_U", 2.0,
              targets={"U_L": (-0.30, 0.45, 1.15), "U_R": (-0.30, -0.45, 1.15)},
              mode={"U_L": "plan", "U_R": "plan", "F_L": "hold", "F_R": "hold"}),
        Phase("reach_F", 2.5,
              targets={"F_L": (0.35, -0.20, 1.00), "F_R": (0.35, 0.20, 1.00)},
              mode={"F_L": "plan", "F_R": "plan", "U_L": "hold", "U_R": "hold"}),
        Phase("grip", 1.0, mode=dict(hold_all)),
        Phase("lift", 1.5,
              targets={"F_L": (0.35, -0.20, 1.08), "F_R": (0.35, 0.20, 1.08)},
              mode={"F_L": "plan", "F_R": "plan", "U_L": "hold", "U_R": "hold"}),
        Phase("carry", 2.5,
              targets={"F_L": (0.16, -0.20, 1.08), "F_R": (0.16, 0.20, 1.08)},
              mode={"F_L": "plan", "F_R": "plan", "U_L": "hold", "U_R": "hold"}),
        Phase("setdown", 1.5,
              targets={"F_L": (0.16, -0.20, 1.02), "F_R": (0.16, 0.20, 1.02)},
              mode={"F_L": "plan", "F_R": "plan", "U_L": "hold", "U_R": "hold"}),
        Phase("release_up", 1.5,
              # straight-z lift first: diagonal pull-back from the extended
              # low pose folds the FR3 wrist onto its forearm (self 2 mm)
              targets={"F_L": (0.16, -0.20, 1.15), "F_R": (0.16, 0.20, 1.15)},
              mode={"F_L": "plan", "F_R": "plan", "U_L": "hold", "U_R": "hold"}),
        Phase("retreat_return", 2.5,
              targets={"F_L": (0.26, -0.44, 1.28), "F_R": (0.26, 0.44, 1.28),
                       "U_L": (-0.12, 0.35, 1.12), "U_R": (-0.12, -0.35, 1.12)},
              mode={a: "plan" for a in ARM_KEYS}),
    ])

    center_grab = SkillSpec("center_grab_cross", [
        Phase("yield_U", 2.0,
              targets={"U_L": (-0.30, 0.62, 1.15), "U_R": (-0.30, -0.55, 1.15)},
              mode={"U_L": "plan", "U_R": "plan", "F_L": "hold", "F_R": "hold"}),
        Phase("F_cross_reach", 2.5,
              targets={"F_L": (-0.06, -0.38, 1.05), "F_R": (-0.06, 0.28, 1.05)},
              mode={"F_L": "plan", "F_R": "plan", "U_L": "hold", "U_R": "hold"}),
        Phase("F_grab", 1.5,
              targets={"F_L": (-0.06, -0.38, 1.02), "F_R": (-0.06, 0.28, 1.02)},
              mode={"F_L": "plan", "F_R": "plan", "U_L": "hold", "U_R": "hold"}),
        Phase("F_return", 2.5,
              targets={"F_L": (0.24, -0.46, 1.18), "F_R": (0.24, 0.46, 1.18)},
              mode={"F_L": "plan", "F_R": "plan", "U_L": "hold", "U_R": "hold"}),
        Phase("U_cross_reach", 2.5,
              targets={"U_L": (-0.03, 0.38, 1.06), "U_R": (-0.03, -0.20, 1.06)},
              mode={"U_L": "plan", "U_R": "plan", "F_L": "hold", "F_R": "hold"}),
        Phase("U_grab", 1.5,
              targets={"U_L": (-0.03, 0.38, 1.02), "U_R": (-0.03, -0.20, 1.02)},
              mode={"U_L": "plan", "U_R": "plan", "F_L": "hold", "F_R": "hold"}),
        Phase("U_return", 2.5,
              targets={"U_L": (-0.16, 0.55, 1.10), "U_R": (-0.16, -0.30, 1.10)},
              mode={"U_L": "plan", "U_R": "plan", "F_L": "hold", "F_R": "hold"}),
        Phase("settle", 1.0, mode=dict(hold_all)),
    ])

    specs = {s.name: s for s in (handover, dual_carry, center_grab)}
    specs.update(skill_specs_t2())
    return specs


def skill_specs_t2() -> dict:
    """T2 expansion wave (owner 20:58: more base motions): 10 more four-arm
    cooperation families on top of the T1 trio, designed under the four T1
    geometry lessons (STATUS_T): wrist pair is the cross bottleneck (meet
    flange gap >= 0.30 m), the F2 hand hangs ~0.19 m below the flange (low
    phases: U flange z >= 1.00, F >= 0.96, +3-5 cm when stretched across the
    line), four-arm simultaneous convergence self+cross collides (use yield /
    staggered lanes -- simultaneous motion is fine when lanes diverge), and
    extended-low retreats lift straight in z first.

    Families (real-rig representativeness order, per the T2 work order):
      handover_UR_FL        reverse direction handover (U gives, F receives)
      handover_high/low     handover height-layer variants (z 1.26 / 1.04)
      bimanual_lift_rotate  F dual-arm lift + in-air yaw about the midpoint
      cross_pick_place_FU   F picks own side, carries across, places U side
      cross_pick_place_UF   the U->F mirror direction
      sequential_center_grab four arms take strict turns at the center band
      perimeter_sweep       simultaneous low sweep along the table perimeter
                            (diverging quadrant lanes -- large-range low pose)
      tool_pass_chain       three-arm relay F_L -> U_R -> F_R (two staggered
                            meets, the carry crosses the U_R/U_L half-line)
      mirror_sync           all four advance/sway in mirrored lanes at once
                            (the p-head yield-semantics stressor)
    """
    hold_all = {"F_L": "hold", "F_R": "hold", "U_L": "hold", "U_R": "hold"}

    # 1) reverse handover: sequenced (giver arrives first), lane y<0
    handover_ur_fl = SkillSpec("handover_UR_FL", [
        Phase("give_approach", 2.5,
              targets={"U_R": (-0.10, -0.34, 1.12)},
              joint_moves={"F_R": {1: -0.05, 3: 0.04}, "U_L": {1: 0.04, 3: -0.04}},
              mode={"U_R": "plan", "F_R": "lerp", "U_L": "lerp", "F_L": "hold"}),
        Phase("recv_approach", 2.5,
              # meet gap: dx 0.30 / dy 0.06 -> 0.306 m flange distance
              targets={"F_L": (0.20, -0.40, 1.12)},
              mode={"F_L": "plan", "U_R": "hold", "F_R": "hold", "U_L": "hold"}),
        Phase("transfer", 1.5, mode=dict(hold_all)),
        Phase("give_retreat", 2.0,
              targets={"U_R": (-0.24, -0.38, 1.18)},
              mode={"U_R": "plan", "F_L": "hold", "F_R": "hold", "U_L": "hold"}),
        Phase("recv_return", 2.5,
              targets={"F_L": (0.24, -0.50, 1.30)},
              joint_moves={"F_R": {1: 0.05, 3: -0.04}, "U_L": {1: -0.04, 3: 0.04}},
              mode={"F_L": "plan", "F_R": "lerp", "U_L": "lerp", "U_R": "hold"}),
    ])

    # 2) high-layer handover on the OTHER diagonal (F_R gives, U_L receives,
    #    lane y>0) -- library covers both diagonals and a second height layer.
    #    SEQUENCED approach (T2 lesson): the y>0 diagonal is the fragile one
    #    (asymmetric birth parks U_L far outboard at y=0.70; its big -y swing
    #    bulges the elbow ~3 cm toward the center line mid-path, and cuRobo
    #    plans each arm blind to the other robot -- the simultaneous version
    #    audited at cross -0.8 mm). U_L parks first, F_R then descends.
    handover_high = SkillSpec("handover_high", [
        Phase("u_approach", 2.5,
              targets={"U_L": (-0.18, 0.37, 1.26)},
              joint_moves={"F_L": {1: -0.05, 3: 0.04}, "U_R": {1: -0.04, 3: 0.04}},
              mode={"U_L": "plan", "F_L": "lerp", "U_R": "lerp", "F_R": "hold"}),
        Phase("f_approach", 2.5,
              targets={"F_R": (0.13, 0.42, 1.26)},
              mode={"F_R": "plan", "F_L": "hold", "U_L": "hold", "U_R": "hold"}),
        Phase("transfer", 1.5, mode=dict(hold_all)),
        Phase("f_retreat", 2.0,
              targets={"F_R": (0.22, 0.48, 1.35)},
              mode={"F_R": "plan", "F_L": "hold", "U_L": "hold", "U_R": "hold"}),
        Phase("u_retreat", 2.5,
              targets={"U_L": (-0.16, 0.50, 1.24)},
              joint_moves={"F_L": {1: 0.05, 3: -0.04}, "U_R": {1: 0.04, 3: -0.04}},
              mode={"U_L": "plan", "F_L": "lerp", "U_R": "lerp", "F_R": "hold"}),
    ])

    # 3) low-layer handover (meet z=1.04: U low floor 1.00 + stretched-cross
    #    allowance; F2 hand bottom ~0.85, table clear) with a straight-z lift
    #    phase before the pull-back (T1 lesson 4)
    handover_low = SkillSpec("handover_low", [
        Phase("approach", 3.0,
              targets={"F_L": (0.18, -0.40, 1.10), "U_R": (-0.16, -0.33, 1.10)},
              joint_moves={"F_R": {1: -0.05}, "U_L": {1: 0.04}},
              mode={"F_L": "plan", "U_R": "plan", "F_R": "lerp", "U_L": "lerp"}),
        Phase("descend_meet", 2.5,
              targets={"F_L": (0.14, -0.40, 1.04), "U_R": (-0.18, -0.34, 1.04)},
              mode={"F_L": "plan", "U_R": "plan", "F_R": "hold", "U_L": "hold"}),
        Phase("transfer", 1.5, mode=dict(hold_all)),
        Phase("lift", 1.5,
              targets={"F_L": (0.14, -0.40, 1.14), "U_R": (-0.18, -0.34, 1.14)},
              mode={"F_L": "plan", "U_R": "plan", "F_R": "hold", "U_L": "hold"}),
        Phase("retreat", 2.5,
              targets={"F_L": (0.23, -0.48, 1.25), "U_R": (-0.20, -0.31, 1.16)},
              joint_moves={"F_R": {1: 0.05}, "U_L": {1: -0.04}},
              mode={"F_L": "plan", "U_R": "plan", "F_R": "lerp", "U_L": "lerp"}),
    ])

    # 4) F dual-arm lift + in-air yaw: EEs orbit the carried midpoint
    #    (0.33, 0, z) by 30 deg at radius 0.20 -> the EE pair gap stays 0.40 m
    #    through the rotation; U yields first (T1 lesson 3)
    lift_rotate = SkillSpec("bimanual_lift_rotate", [
        Phase("yield_U", 2.0,
              targets={"U_L": (-0.30, 0.48, 1.16), "U_R": (-0.30, -0.48, 1.16)},
              mode={"U_L": "plan", "U_R": "plan", "F_L": "hold", "F_R": "hold"}),
        Phase("reach", 2.5,
              targets={"F_L": (0.33, -0.20, 1.00), "F_R": (0.33, 0.20, 1.00)},
              mode={"F_L": "plan", "F_R": "plan", "U_L": "hold", "U_R": "hold"}),
        Phase("grip", 1.0, mode=dict(hold_all)),
        Phase("lift", 1.5,
              targets={"F_L": (0.33, -0.20, 1.12), "F_R": (0.33, 0.20, 1.12)},
              mode={"F_L": "plan", "F_R": "plan", "U_L": "hold", "U_R": "hold"}),
        Phase("rotate_cw", 2.5,
              targets={"F_L": (0.43, -0.173, 1.12), "F_R": (0.23, 0.173, 1.12)},
              mode={"F_L": "plan", "F_R": "plan", "U_L": "hold", "U_R": "hold"}),
        Phase("rotate_back", 2.5,
              targets={"F_L": (0.33, -0.20, 1.12), "F_R": (0.33, 0.20, 1.12)},
              mode={"F_L": "plan", "F_R": "plan", "U_L": "hold", "U_R": "hold"}),
        Phase("setdown", 1.5,
              targets={"F_L": (0.33, -0.20, 1.02), "F_R": (0.33, 0.20, 1.02)},
              mode={"F_L": "plan", "F_R": "plan", "U_L": "hold", "U_R": "hold"}),
        Phase("release_up", 1.5,
              targets={"F_L": (0.33, -0.20, 1.14), "F_R": (0.33, 0.20, 1.14)},
              mode={"F_L": "plan", "F_R": "plan", "U_L": "hold", "U_R": "hold"}),
        Phase("retreat_return", 2.5,
              targets={"F_L": (0.26, -0.44, 1.28), "F_R": (0.26, 0.44, 1.28),
                       "U_L": (-0.12, 0.35, 1.12), "U_R": (-0.12, -0.35, 1.12)},
              mode={a: "plan" for a in ARM_KEYS}),
    ])

    # 5) cross-machine pick&place, F->U direction (lane y<0, U_R yields).
    #    Pick approach is STAGED (planar at safe z, then straight-z descend):
    #    the one-shot diagonal descend to a near-base low point folds the FR3
    #    wrist (IK guard hit self 2mm) -- lesson-4 mirror image; pick point
    #    kept at planar >= 0.33 m from the F_L base (proven dual_carry band).
    cross_fu = SkillSpec("cross_pick_place_FU", [
        Phase("yield_UR", 2.0,
              targets={"U_R": (-0.30, -0.52, 1.18)},
              mode={"U_R": "plan", "F_L": "hold", "F_R": "hold", "U_L": "hold"}),
        Phase("pre_reach", 2.0,
              targets={"F_L": (0.34, -0.24, 1.15)},
              mode={"F_L": "plan", "F_R": "hold", "U_L": "hold", "U_R": "hold"}),
        Phase("pick_down", 1.5,
              targets={"F_L": (0.34, -0.24, 1.01)},
              mode={"F_L": "plan", "F_R": "hold", "U_L": "hold", "U_R": "hold"}),
        Phase("lift", 1.5,
              targets={"F_L": (0.34, -0.24, 1.12)},
              mode={"F_L": "plan", "F_R": "hold", "U_L": "hold", "U_R": "hold"}),
        Phase("carry_cross", 3.0,
              targets={"F_L": (-0.06, -0.32, 1.10)},
              mode={"F_L": "plan", "F_R": "hold", "U_L": "hold", "U_R": "hold"}),
        Phase("place", 1.5,
              targets={"F_L": (-0.06, -0.32, 1.03)},
              mode={"F_L": "plan", "F_R": "hold", "U_L": "hold", "U_R": "hold"}),
        Phase("release_up", 1.5,
              targets={"F_L": (-0.06, -0.32, 1.15)},
              mode={"F_L": "plan", "F_R": "hold", "U_L": "hold", "U_R": "hold"}),
        Phase("return", 2.5,
              targets={"F_L": (0.24, -0.48, 1.25), "U_R": (-0.18, -0.32, 1.16)},
              mode={"F_L": "plan", "U_R": "plan", "F_R": "hold", "U_L": "hold"}),
    ])

    # 6) cross-machine pick&place, U->F direction (lane y>0, F_R yields;
    #    U stretched low across the line keeps z >= 1.04). Same staged
    #    approach discipline as FU (planar first, then straight-z down).
    cross_uf = SkillSpec("cross_pick_place_UF", [
        Phase("yield_FR", 2.0,
              targets={"F_R": (0.30, 0.56, 1.30)},
              mode={"F_R": "plan", "F_L": "hold", "U_L": "hold", "U_R": "hold"}),
        Phase("pre_reach", 2.0,
              targets={"U_L": (-0.26, 0.42, 1.15)},
              mode={"U_L": "plan", "F_L": "hold", "F_R": "hold", "U_R": "hold"}),
        Phase("pick_down", 1.5,
              # z 1.035 not 1.02: at planar 0.30 from the U_L base the JAKA
              # hand hangs deeper -- 1.02 grazed the table floor (2.3 mm)
              targets={"U_L": (-0.26, 0.42, 1.035)},
              mode={"U_L": "plan", "F_L": "hold", "F_R": "hold", "U_R": "hold"}),
        Phase("lift", 1.5,
              targets={"U_L": (-0.26, 0.42, 1.12)},
              mode={"U_L": "plan", "F_L": "hold", "F_R": "hold", "U_R": "hold"}),
        Phase("carry_cross", 3.0,
              targets={"U_L": (0.10, 0.40, 1.09)},
              mode={"U_L": "plan", "F_L": "hold", "F_R": "hold", "U_R": "hold"}),
        Phase("place", 1.5,
              targets={"U_L": (0.10, 0.40, 1.04)},
              mode={"U_L": "plan", "F_L": "hold", "F_R": "hold", "U_R": "hold"}),
        Phase("release_up", 1.5,
              targets={"U_L": (0.10, 0.40, 1.14)},
              mode={"U_L": "plan", "F_L": "hold", "F_R": "hold", "U_R": "hold"}),
        Phase("return", 2.5,
              targets={"U_L": (-0.16, 0.52, 1.18), "F_R": (0.22, 0.48, 1.32)},
              mode={"U_L": "plan", "F_R": "plan", "F_L": "hold", "U_R": "hold"}),
    ])

    # 7) strict per-arm turn-taking at the center band (the time-sliced
    #    counterpart of center_grab_cross: at most ONE arm near x=0 at a time)
    def _turn(arm, reach, dip_z, back):
        others = {a: "hold" for a in ARM_KEYS if a != arm}
        return [
            Phase(f"{arm}_reach", 2.0, targets={arm: reach},
                  mode={arm: "plan", **others}),
            Phase(f"{arm}_dip", 1.0, targets={arm: (reach[0], reach[1], dip_z)},
                  mode={arm: "plan", **others}),
            Phase(f"{arm}_return", 2.0, targets={arm: back},
                  mode={arm: "plan", **others}),
        ]

    # prologue: U arms step back to staging posts first -- the v4.1 birth
    # parks the U EEs at x ~= 0.01, i.e. ON the center band, so the first F
    # turn would otherwise brush a parked U wrist (IK guard hit cross 1.5mm)
    seq_center = SkillSpec("sequential_center_grab",
        [Phase("stage_U", 2.0,
               targets={"U_L": (-0.30, 0.52, 1.18), "U_R": (-0.30, -0.48, 1.18)},
               mode={"U_L": "plan", "U_R": "plan", "F_L": "hold", "F_R": "hold"})]
        + _turn("F_L", (-0.02, -0.40, 1.06), 1.03, (0.24, -0.48, 1.22))
        + _turn("U_R", (-0.02, -0.24, 1.06), 1.025, (-0.18, -0.30, 1.16))
        + _turn("F_R", (-0.02, 0.32, 1.06), 1.03, (0.24, 0.46, 1.22))
        + _turn("U_L", (-0.02, 0.46, 1.06), 1.025, (-0.18, 0.55, 1.16)))

    # 8) simultaneous low perimeter sweep in DIVERGING quadrant lanes (large
    #    y-range low motion; inner endpoints keep the 0.32 m cross x-gap)
    # Perimeter coverage rebuilt from motion primitives the probes proved
    # safe: LOW moves only along x (a lateral y-move at low z folds the FR3
    # wrist to <2 mm no matter the height 1.01-1.05 or the arc shape --
    # probed pure_-y / diag / y_then all blocked), lateral shifts happen at
    # mid z (1.18), descents are pure-z after a planar spread, and every low
    # point keeps >= 0.33 m planar distance from the arm's own base disc.
    # Pattern = wipe seam band -> lift -> shift to outer corner -> dip:
    # exactly how a human wipes two zones of a table.
    perimeter = SkillSpec("perimeter_sweep", [
        Phase("spread", 2.5,
              targets={"F_L": (0.34, -0.20, 1.18), "F_R": (0.34, 0.20, 1.18),
                       "U_L": (-0.28, 0.44, 1.18), "U_R": (-0.30, -0.20, 1.18)},
              mode={a: "plan" for a in ARM_KEYS}),
        Phase("descend", 1.5,
              targets={"F_L": (0.34, -0.20, 1.05), "F_R": (0.34, 0.20, 1.05),
                       "U_L": (-0.28, 0.44, 1.06), "U_R": (-0.30, -0.20, 1.06)},
              mode={a: "plan" for a in ARM_KEYS}),
        Phase("sweep_seam", 3.0,
              # pure x toward the table seam, cross x-gap held at 0.32
              targets={"F_L": (0.16, -0.20, 1.05), "F_R": (0.16, 0.20, 1.05),
                       "U_L": (-0.16, 0.44, 1.06), "U_R": (-0.16, -0.20, 1.06)},
              mode={a: "plan" for a in ARM_KEYS}),
        Phase("lift_up", 1.5,
              targets={"F_L": (0.16, -0.20, 1.18), "F_R": (0.16, 0.20, 1.18),
                       "U_L": (-0.16, 0.44, 1.18), "U_R": (-0.16, -0.20, 1.18)},
              mode={a: "plan" for a in ARM_KEYS}),
        Phase("shift_outer", 2.5,
              targets={"F_L": (0.22, -0.58, 1.18), "F_R": (0.22, 0.58, 1.18),
                       "U_L": (-0.22, 0.58, 1.18), "U_R": (-0.22, -0.55, 1.18)},
              mode={a: "plan" for a in ARM_KEYS}),
        Phase("dip_outer", 1.5,
              targets={"F_L": (0.22, -0.58, 1.05), "F_R": (0.22, 0.58, 1.05),
                       "U_L": (-0.22, 0.58, 1.06), "U_R": (-0.22, -0.55, 1.06)},
              mode={a: "plan" for a in ARM_KEYS}),
        Phase("corner_in", 2.0,
              # short pure-x hop covering the outer corner band
              targets={"F_L": (0.16, -0.58, 1.05), "F_R": (0.16, 0.58, 1.05),
                       "U_L": (-0.16, 0.58, 1.06), "U_R": (-0.16, -0.55, 1.06)},
              mode={a: "plan" for a in ARM_KEYS}),
        Phase("lift_return", 2.5,
              targets={"F_L": (0.22, -0.50, 1.25), "F_R": (0.22, 0.50, 1.25),
                       "U_L": (-0.10, 0.62, 1.18), "U_R": (-0.10, -0.35, 1.18)},
              mode={a: "plan" for a in ARM_KEYS}),
    ])

    # 9) three-arm relay F_L -> U_R -> F_R: two staggered meets, U_R carries
    #    the piece across its own half-line (y -0.32 -> +0.00)
    tool_chain = SkillSpec("tool_pass_chain", [
        Phase("meet1_approach", 2.5,
              targets={"F_L": (0.15, -0.38, 1.12), "U_R": (-0.16, -0.32, 1.12)},
              joint_moves={"F_R": {1: -0.04}},
              mode={"F_L": "plan", "U_R": "plan", "F_R": "lerp", "U_L": "hold"}),
        Phase("transfer1", 1.2, mode=dict(hold_all)),
        Phase("split1", 2.5,
              targets={"F_L": (0.26, -0.50, 1.26), "U_R": (-0.20, -0.08, 1.14)},
              mode={"F_L": "plan", "U_R": "plan", "F_R": "hold", "U_L": "hold"}),
        Phase("meet2_approach", 2.5,
              targets={"F_R": (0.11, 0.06, 1.14), "U_R": (-0.19, 0.00, 1.14)},
              mode={"F_R": "plan", "U_R": "plan", "F_L": "hold", "U_L": "hold"}),
        Phase("transfer2", 1.2, mode=dict(hold_all)),
        Phase("split2", 2.5,
              targets={"U_R": (-0.18, -0.28, 1.14), "F_R": (0.22, 0.30, 1.24)},
              mode={"U_R": "plan", "F_R": "plan", "F_L": "hold", "U_L": "hold"}),
        Phase("settle", 1.5,
              targets={"F_R": (0.24, 0.46, 1.32)},
              joint_moves={"F_L": {1: 0.04}},
              mode={"F_R": "plan", "F_L": "lerp", "U_L": "hold", "U_R": "hold"}),
    ])

    # 10) mirrored synchronized advance + lateral sways: all four arms move AT
    #     ONCE in lane-separated mirror symmetry (cross x-gap fixed at 0.32 m,
    #     sways translate both sides of each lane together) -- the skill-layer
    #     stressor for the p-head yield semantics under tier noise
    mirror = SkillSpec("mirror_sync", [
        # advance lands at a WIDE 0.40 m x-gap: all four arms plan at once
        # and the U_L elbow bulge (see handover_high note) eats ~3 cm of
        # margin mid-path -- with the 0.32 m endpoint the audit grazed
        # 0.4 mm. The tight gap is reached by the short low-curvature
        # close_in hop instead (6 cm straight pulls barely bulge).
        Phase("advance", 3.0,
              targets={"F_L": (0.21, -0.38, 1.14), "U_R": (-0.19, -0.36, 1.14),
                       "F_R": (0.21, 0.38, 1.14), "U_L": (-0.19, 0.36, 1.14)},
              mode={a: "plan" for a in ARM_KEYS}),
        Phase("close_in", 1.5,
              targets={"F_L": (0.17, -0.38, 1.14), "U_R": (-0.15, -0.36, 1.14),
                       "F_R": (0.17, 0.38, 1.14), "U_L": (-0.15, 0.36, 1.14)},
              mode={a: "plan" for a in ARM_KEYS}),
        Phase("sway_plus", 2.0,
              targets={"F_L": (0.17, -0.28, 1.14), "U_R": (-0.15, -0.26, 1.14),
                       "F_R": (0.17, 0.48, 1.14), "U_L": (-0.15, 0.46, 1.14)},
              mode={a: "plan" for a in ARM_KEYS}),
        Phase("sway_minus", 2.5,
              targets={"F_L": (0.17, -0.48, 1.14), "U_R": (-0.15, -0.46, 1.14),
                       "F_R": (0.17, 0.28, 1.14), "U_L": (-0.15, 0.26, 1.14)},
              mode={a: "plan" for a in ARM_KEYS}),
        Phase("recenter", 1.5,
              targets={"F_L": (0.17, -0.38, 1.14), "U_R": (-0.15, -0.36, 1.14),
                       "F_R": (0.17, 0.38, 1.14), "U_L": (-0.15, 0.36, 1.14)},
              mode={a: "plan" for a in ARM_KEYS}),
        Phase("withdraw", 2.5,
              targets={"F_L": (0.22, -0.50, 1.32), "U_R": (-0.12, -0.32, 1.18),
                       "F_R": (0.22, 0.50, 1.32), "U_L": (-0.12, 0.60, 1.18)},
              mode={a: "plan" for a in ARM_KEYS}),
    ])

    return {s.name: s for s in (
        handover_ur_fl, handover_high, handover_low, lift_rotate,
        cross_fu, cross_uf, seq_center, perimeter, tool_chain, mirror)}


# ---------------------------------------------------------------------------
# stage A: waypoint design (DLS-IK + margin guard)
# ---------------------------------------------------------------------------


# R27 (2026-09-05): design-time counterpart of the R29 structural-row
# exemption -- the UR5 upper_arm x table row hovers 0.6 mm above the v7 IK
# table floor in EVERY pose, so any U-arm motion trips the guard. Default
# False = bit-identical legacy; task builders opt in (--struct-exempt).
DESIGN_STRUCT_EXEMPT = {"on": False}


def _margins_ok(provider, q: dict, floors: dict = DESIGN_FLOORS) -> tuple:
    try:
        mm = provider.min_margin_by_class(q, exclude_structural=DESIGN_STRUCT_EXEMPT["on"])
    except TypeError:            # providers without the option
        mm = provider.min_margin_by_class(q)
    ok = all(mm[k].item() >= floors[k] for k in floors)
    return ok, {k: round(mm[k].item(), 4) for k in mm}


# during-IK guard: only forbid true proximity/penetration -- the IK path is
# NOT kept (cuRobo plans the real path; the composed trajectory is audited
# step by step), so transient shallow dips are fine. Final waypoints must
# still clear DESIGN_FLOORS (checked by design_skill).
IK_PATH_FLOORS = {"cross": 0.002, "self": 0.002, "table": 0.002}

# R35 (2026-09-07): opt-in joint-limit projection inside the DLS-IK. The design
# layer had no joint limits at all: the carton_packing F_R trajectory ran j7 to
# 3.71 rad (FR3 limit 3.016) for 860 steps and the sim arm stalled 12 cm short
# (r35_carton5). off = bit-identical historical behaviour.
IK_JOINT_LIMITS = {
    "on": False,
    # FR3 official (rad); UR5 joints are +-2 pi (effectively unconstrained)
    "F": ([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159],
          [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159]),
    "U": ([-6.2832, -6.2832, -3.1416, -6.2832, -6.2832, -6.2832], [6.2832, 6.2832, 3.1416, 6.2832, 6.2832, 6.2832]),
    "margin": 0.02,                 # keep 0.02 rad inside the hard limit
}


def clamp_joint_limits(arm: str, q: torch.Tensor) -> torch.Tensor:
    if not IK_JOINT_LIMITS.get("on"):
        return q
    lo, hi = IK_JOINT_LIMITS[arm[0]]
    m = float(IK_JOINT_LIMITS.get("margin", 0.0))
    lo_t = torch.tensor(lo, dtype=q.dtype, device=q.device) + m
    hi_t = torch.tensor(hi, dtype=q.dtype, device=q.device) - m
    return torch.maximum(torch.minimum(q, hi_t), lo_t)


def _rot_log(R: torch.Tensor) -> torch.Tensor:
    """SO(3) 对数映射：(3,3) 旋转 -> 轴角向量 (3,)。

    姿态 IK 的误差项。ang→pi 时标准公式 (R - R^T)/(2 sin) 退化，走
    对称部分分支（R23 场景确实会遇到：出生腕姿态与顶抓姿态可差 ~pi）。
    """
    cos = ((R[0, 0] + R[1, 1] + R[2, 2]) - 1.0) * 0.5
    ang = torch.arccos(cos.clamp(-1.0, 1.0))
    if float(ang) < 1e-6:
        return torch.zeros(3, dtype=R.dtype, device=R.device)
    if float(ang) > math.pi - 1e-3:
        # 轴取 (R + I)/2 的最大列（Rodrigues 在 pi 附近的稳定式）
        M = (R + torch.eye(3, dtype=R.dtype, device=R.device)) * 0.5
        col = int(M.diagonal().argmax())
        axis = M[:, col] / M[:, col].norm().clamp_min(1e-9)
        return axis * ang
    axis = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0],
                        R[1, 0] - R[0, 1]]) / (2.0 * torch.sin(ang))
    return axis * ang


def _ee_rot_jacobian(provider, arm: str, fko_arm: dict) -> torch.Tensor:
    """flange 姿态雅可比 (1, 3, dof)：串联臂第 j 列 = 世界系关节轴 z_j。

    fk() 已经把每个关节的世界轴存在 fko["z"] ((N, dof, 3))，flange 在末
    关节之后，所有关节都影响它 —— 直接转置即得。
    """
    return fko_arm["z"].transpose(-1, -2)


def solve_waypoint(provider, q: dict, moving: dict,
                   tol: float = 0.015, iters: int = 400,
                   gain: float = 0.6, damping: float = 1e-2,
                   floors: dict = IK_PATH_FLOORS,
                   ori_targets: "dict | None" = None,
                   rot_tol: float = 0.06, rot_weight: float = 0.5) -> tuple:
    """Position-only DLS-IK for the arms in `moving` (arm -> world xyz),
    margin-guarded: any step that would sink below `floors` is halved (up to
    4 times) or rejected. Non-moving arms stay fixed.

    R23 扩展：`ori_targets`（arm -> 3x3 世界系 flange 旋转）非空时，对应
    臂走 6 维位姿 DLS-IK（位置误差 + rot_weight × 轴角误差；旋转雅可比 =
    世界系关节轴）。不给 = 历史行为逐位不变（13 族技能库不受影响）。

    Returns (q_new, info). info["ok"] False => target unreachable under
    guard; caller adjusts the choreography.
    """
    q = {a: v.clone() for a, v in q.items()}
    tgts = {a: torch.tensor(p, dtype=torch.float32).view(1, 3)
            for a, p in moving.items()}
    oris = {a: torch.tensor(R, dtype=torch.float32).view(3, 3)
            for a, R in (ori_targets or {}).items()}
    assert all(a in tgts for a in oris), \
        f"ori_targets arms {list(oris)} must be a subset of moving arms"
    info = {"ok": True, "residuals": {}, "iters": 0}
    for it in range(iters):
        state = SceneState(q=q, dt=CONTROL_DT)
        done = True
        for a, tgt in tgts.items():
            fko = provider.fk_all(q)
            ee = fko[a]["t_flange"]                      # (1, 3)
            err = tgt - ee
            e_rot = None
            if a in oris:
                e_rot = _rot_log(oris[a] @ fko[a]["R_flange"][0].T)
                if err.norm() < tol and e_rot.norm() < rot_tol:
                    continue
            elif err.norm() < tol:
                continue
            done = False
            J = provider.ee_jacobian(a, state)           # (1, 3, dof)
            if e_rot is not None:
                # 位置行 + 加权姿态行拼 6 维（权重换算 rad<->m 的误差量纲）
                Jr = _ee_rot_jacobian(provider, a, fko[a])
                J = torch.cat([J, rot_weight * Jr], dim=1)      # (1, 6, dof)
                err = torch.cat([err, rot_weight * e_rot.view(1, 3)], dim=1)
            JJt = J @ J.transpose(-1, -2)
            eye = torch.eye(J.shape[1]).expand_as(JJt)
            dq = (J.transpose(-1, -2)
                  @ torch.linalg.solve(JJt + damping * eye,
                                       err.unsqueeze(-1))).squeeze(-1) * gain
            dq = dq.clamp(-0.05, 0.05)                   # rad per IK step
            scale = 1.0
            for _ in range(5):
                q_try = {k: v.clone() for k, v in q.items()}
                q_try[a] = clamp_joint_limits(a, q_try[a] + dq * scale)
                ok, mm = _margins_ok(provider, q_try, floors)
                if ok:
                    q = q_try
                    break
                scale *= 0.5
            else:
                info["ok"] = False
                info["blocked"] = {a: mm}
                return q, info
        info["iters"] = it + 1
        if done:
            break
    fko = provider.fk_all(q)
    for a, tgt in tgts.items():
        res = (tgt - fko[a]["t_flange"]).norm().item()
        info["residuals"][a] = round(res, 4)
        if res > tol * 2.0:
            info["ok"] = False
        if a in oris:
            rres = _rot_log(oris[a] @ fko[a]["R_flange"][0].T).norm().item()
            info.setdefault("rot_residuals", {})[a] = round(rres, 4)
            if rres > rot_tol * 2.0:
                info["ok"] = False
    _, info["margins"] = _margins_ok(provider, q, floors)
    return q, info


def design_skill(provider, spec: SkillSpec, verbose: bool = True) -> dict:
    """Waypoint chain for one skill: q_0 = provider birth pose; each phase
    boundary solved from the previous. Returns the waypoints doc (json-able)."""
    q = provider.default_q()
    wps = {a: [q[a][0].tolist()] for a in ARM_KEYS}
    phase_rows = []
    for ph in spec.phases:
        moving = dict(ph.targets)
        # joint-space sways resolve to explicit joint targets
        q_next = {a: v.clone() for a, v in q.items()}
        for a, off in ph.joint_moves.items():
            for j, dv in off.items():
                q_next[a][0, int(j)] += float(dv)
        if moving:
            q_solved, info = solve_waypoint(provider, q_next, moving)
            if not info["ok"]:
                raise RuntimeError(
                    f"{spec.name}/{ph.name}: IK blocked or residual too big: {info}")
        else:
            q_solved, info = q_next, {"ok": True, "residuals": {},
                                      "margins": _margins_ok(provider, q_next)[1]}
        ok, mm = _margins_ok(provider, q_solved)
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
            "meta": {"layout": "v5", "birth_pose": "v4.1",
                     "floors": DESIGN_FLOORS}}


# ---------------------------------------------------------------------------
# stage C: compose + validate
# ---------------------------------------------------------------------------


def _cosine_ease(q0: np.ndarray, q1: np.ndarray, n: int) -> np.ndarray:
    """(n+1, dof) smoothstep interpolation with zero end velocities."""
    s = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n + 1))
    return q0[None] + s[:, None] * (q1 - q0)[None]


def _resample(path: np.ndarray, n: int) -> np.ndarray:
    """Uniform time-stretch of a dense (M, dof) joint path onto n+1 points."""
    m = path.shape[0]
    src = np.linspace(0.0, 1.0, m)
    dst = np.linspace(0.0, 1.0, n + 1)
    out = np.empty((n + 1, path.shape[1]), dtype=np.float64)
    for j in range(path.shape[1]):
        out[:, j] = np.interp(dst, src, path[:, j])
    return out


def compose_skill(doc: dict, segments: "dict | None",
                  dt: float = CONTROL_DT) -> SkillTrajectory:
    """Waypoints doc (+ optional cuRobo segments npz dict) -> SkillTrajectory.

    segments keys: "<arm>_seg<k>" dense (M, dof) arrays for phases whose mode
    is "plan"; missing/None => cosine-eased fallback for every phase (pure-CPU
    contingency path, margins still validated downstream)."""
    wps = {a: np.asarray(doc["waypoints"][a], dtype=np.float64)
           for a in ARM_KEYS}
    chunks = {a: [wps[a][0:1]] for a in ARM_KEYS}
    for k, ph in enumerate(doc["phases"]):
        n = max(1, round(ph["dur"] / dt))
        for a in ARM_KEYS:
            q0, q1 = wps[a][k], wps[a][k + 1]
            mode = ph["mode"][a]
            key = f"{a}_seg{k}"
            if mode == "plan" and segments is not None and key in segments:
                seg = _resample(np.asarray(segments[key], dtype=np.float64), n)
                # planner endpoints must agree with the designed waypoints
                if (np.abs(seg[0] - q0).max() > 1e-3
                        or np.abs(seg[-1] - q1).max() > 1e-3):
                    raise ValueError(f"{key}: segment endpoints drift from waypoints")
            elif mode == "hold" or np.abs(q1 - q0).max() < 1e-9:
                seg = np.repeat(q0[None], n + 1, axis=0)
            else:
                seg = _cosine_ease(q0, q1, n)
            chunks[a].append(seg[1:])                    # drop shared boundary row
    q = {a: np.concatenate(chunks[a], axis=0).astype(np.float32)
         for a in ARM_KEYS}
    meta = {"skill": doc["skill"], "layout": "v5", "birth_pose": "v4.1",
            "phases": [{"name": p["name"], "dur": p["dur"]} for p in doc["phases"]],
            "planner": "curobo plan_cspace" if segments else "cosine_ease_fallback"}
    return SkillTrajectory(q=q, dt=dt, meta=meta)


def validate_trajectory(provider, traj: SkillTrajectory,
                        floors: dict = DESIGN_FLOORS) -> dict:
    """Authoritative per-step audit on the CPU provider.

    Hard gate (spec): min margin > 0 at EVERY grid point. Soft report: time
    below design floors, per-arm max |delta|/step, boundary continuity."""
    T = traj.n_steps
    mins = {"cross": [], "self": [], "table": []}
    for t in range(T + 1):
        q = {a: torch.tensor(traj.q[a][t: t + 1], dtype=torch.float32)
             for a in ARM_KEYS}
        mm = provider.min_margin_by_class(q)
        for k in mins:
            mins[k].append(mm[k].item())
    mins = {k: np.asarray(v) for k, v in mins.items()}
    deltas = {a: np.abs(np.diff(traj.q[a], axis=0)) for a in ARM_KEYS}
    report = {
        "skill": traj.meta.get("skill"),
        "steps": T, "duration_s": round(traj.duration, 3),
        "hard_gate_margin_gt0": bool(all(m.min() > 0.0 for m in mins.values())),
        "min_margin": {k: round(float(m.min()), 4) for k, m in mins.items()},
        "argmin_step": {k: int(m.argmin()) for k, m in mins.items()},
        "frac_below_floor": {
            k: round(float((mins[k] < floors[k]).mean()), 4) for k in mins},
        "max_delta_per_arm": {a: round(float(d.max()), 4)
                              for a, d in deltas.items()},
        "delta_soft_cap_ok": bool(all(d.max() <= DELTA_SOFT_CAP
                                      for d in deltas.values())),
    }
    if not report["hard_gate_margin_gt0"]:
        report["FAIL"] = "margin <= 0 somewhere; DO NOT SHIP this trajectory"
    for a, d in deltas.items():
        if d.max() > DELTA_HARD_CAP:
            report["FAIL"] = f"{a} delta {d.max():.4f} > hard cap {DELTA_HARD_CAP}"
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: "list | None" = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["design", "compose", "validate"])
    ap.add_argument("--out", default="artifacts/skill_trajs")
    ap.add_argument("--skill", default=None, help="one skill name; default all")
    ap.add_argument("--segments", default=None,
                    help="segments_<skill>.npz dir (compose); omit for fallback")
    ap.add_argument("--traj", default=None, help="npz path (validate)")
    args = ap.parse_args(argv)

    from safeduo.baselines.real_geometry import make_v5_provider

    provider = make_v5_provider(1, device="cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = skill_specs()
    names = [args.skill] if args.skill else list(specs)

    if args.stage == "design":
        for name in names:
            doc = design_skill(provider, specs[name])
            p = out / f"waypoints_{name}.json"
            p.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
            print(f"wrote {p}")
    elif args.stage == "compose":
        for name in names:
            doc = json.loads((out / f"waypoints_{name}.json").read_text())
            seg = None
            if args.segments:
                sp = Path(args.segments) / f"segments_{name}.npz"
                if sp.exists():
                    seg = dict(np.load(sp))
            traj = compose_skill(doc, seg)
            rep = validate_trajectory(provider, traj)
            traj.meta["validation"] = rep
            fp = out / f"skill_{name}.npz"
            traj.save(fp)
            (out / f"{name}_report.json").write_text(
                json.dumps(rep, indent=1, ensure_ascii=False))
            print(f"{name}: {json.dumps(rep, ensure_ascii=False)}")
            if "FAIL" in rep:
                return 1
    else:
        traj = SkillTrajectory.load(args.traj)
        print(json.dumps(validate_trajectory(provider, traj), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
