"""R35 (2026-09-07): industrial task families on Isaac cloud assets, inside the
"factory cell" dressing (configs/dressing_r35_factory_cell.yaml).

Families (all four arms move; validated grasp recipes only):
  pipe_sorting    -- fine sorting by colour: four exhaust pipes (6.2 cm x 18 cm,
                     black / blue) are picked from the shared centre zone and
                     stood down inside the matching black / blue parts bins
                     (F: top pinch = reagent-bottle recipe; U: DFX top claw =
                     beaker recipe).
  carton_packing  -- collaborative packaging: F_L and F_R load two pipes into the
                     open KLT tote (19.8 x 29.7 x 14.6 cm; the Isaac corrugated
                     cartons are sealed), both return home, then palm-clamp the
                     tote ends and carry it 14 cm
                     (R34 bin_packing skeleton; both-arm regroup avoids the
                     folded-wrist posture that made the a27 policy lock F_L).

usage:
  python -m safeduo.delta.task_library_r35 build --task pipe_sorting --out artifacts/task_trajs_r35 \
      --hand-calib v7 --spheres r16 --struct-exempt --multiseed
  python -m safeduo.delta.task_library_r35 env --task pipe_sorting
"""
from __future__ import annotations

import os
from pathlib import Path

from safeduo.delta import task_library_r26 as R26
from safeduo.delta import task_library_r34 as R34
from safeduo.delta.task_library_r24 import ARM_KEYS, _above, _ph, grip_station_cands
from safeduo.delta.task_library_r34 import LabObject, TABLE_TOP_Z, lab_object
from safeduo.delta.task_record_s9 import GRASP_PARAMS, S9Task
from safeduo.delta.skill_record import SkillSpec

DRESSING_YAML = "dressing_r35_factory_cell_lite.yaml"   # full (warehouse shell) variant: dressing_r35_factory_cell.yaml
R35_ENV = {"pipe_sorting": "duo_env_v7_r35_pipe_sorting.yaml",
           "carton_packing": "duo_env_v7_r35_carton_packing.yaml",
           "profile_colift": "duo_env_v7_r35_profile_colift.yaml"}

# ---------------------------------------------------------------------------
# family 1: pipe_sorting
# ---------------------------------------------------------------------------

BIN_FLOOR_DZ = 0.006          # parts-bin floor thickness: stand the pipe on it, not through it
PIPE_LAYOUT = {
    # arm: (pipe name, catalog, start xy, bin name, bin catalog, bin xy, bin yaw)
    "F_L": ("pipe_black_f", "exhaust_pipe_black", (0.50, -0.36), "bin_black_f", "sorting_bin_black", (0.48, -0.60), 90.0),
    "F_R": ("pipe_blue_f", "exhaust_pipe_blue", (0.50, 0.36), "bin_blue_f", "sorting_bin_blue", (0.48, 0.60), 90.0),
    # U arms: the DFX top claw is validated on 5-7 cm wide / <= 10 cm tall objects
    # (cubes, 100 ml beaker, crucible); it never held the 18 cm exhaust pipe
    # (r35_pipe2/3/4: U pipes stayed at their start) -> YCB soup cans (6.8 x 10 cm)
    "U_L": ("can_u_l", "soup_can", (-0.44, 0.42), "bin_black_u", "sorting_bin_black", (-0.30, 0.28), 90.0),
    "U_R": ("can_u_r", "soup_can", (-0.46, -0.20), "bin_blue_u", "sorting_bin_blue", (-0.28, -0.32), 90.0),
}


def _u_place(cand, obj, legacy_dz: float) -> tuple:
    """U-arm release flange pose that puts the GRASPED object at obj.place_xy.
    The absolute-flange form (place_xy, _place_z) inherited from the R26 cube
    tasks put the FLANGE on the target, so the object landed one pinch offset
    away: ~12 cm with the finger approach, ~7 cm with flange_z (r35_colift3-6,
    r35_carton13/14, r35_rail7 -- the blocks and cans never slipped, every
    GRASP_TELE trace shows them held from close to release)."""
    return R26._place_rel(cand.flange_grasp, obj.pos, obj.place_xy, legacy_dz)


def sort_objects() -> dict:
    objs = {}
    for arm, (pn, pk, pxy, bn, bk, bxy, byaw) in PIPE_LAYOUT.items():
        objs[arm] = lab_object(pn, pk, pxy, bxy)
        objs[f"bin:{arm}"] = lab_object(bn, bk, bxy, None, yaw=byaw)
    return objs


def _pipe_sorting_variants(provider):
    objs = sort_objects()
    gen = {a: (lambda ap, a=a: R34._top_cands(provider, a, objs[a], ap)) for a in ARM_KEYS}
    bins = [objs[f"bin:{a}"].name for a in ARM_KEYS]
    for tag, cands in R26._screened_combos(provider, "pipe_sorting", gen):
        yield tag, R34._pick_place_task("pipe_sorting", objs, cands, R35_ENV["pipe_sorting"],
                                        place_dz=BIN_FLOOR_DZ, extra_objects=bins)


# ---------------------------------------------------------------------------
# family 2: carton_packing (R34 bin_packing skeleton on a corrugated carton)
# ---------------------------------------------------------------------------

CARTON_XY = (0.53, 0.0)
CARTON_DX = 0.14
# pipes go along the carton's long (x) axis at y = 0: with the carton yawed 90 deg the
# interior is 28 x 21 cm, and a pipe at y = +-0.06 left only 2 cm between the hand's
# thumb side and the 5 mm wall -- the 0.35 kg carton got pushed over (r35_carton1)
CARTON_SLOT_DX = 0.045
CARTON_SLOT_DY = 0.045
CARTON_SLOT_Y_LAG = 0.020      # see carton_objects()
CAN_SLOT_DX = 0.060            # cans 12 cm apart along the tote's long axis
CARTON_CARRY_S = 6.0           # see the carry phase in _carton_packing_task (6 s tested, not the fix)
CARTON_PIPE_DEPTH = 0.070      # the validated sorting pinch depth. The two-pad pinch is a hinge: the pipe hangs
                               # ~3 cm off vertical at the bottom after the carry at 5.5, 7 AND 9 cm depth
                               # (r35_carton12/14/15 -- pinching at the 9 cm mid-height did not reduce the tilt and
                               # cost grasp tracking, qerr 0.01 -> 0.10), so the tote has to give the tilt room.
# Tote: yaw 90 keeps the 29.7 cm side along x -- the F_L hand sits 11-15 cm in +x of the pipe and
# with yaw 0 (19.8 cm along x) it landed on the +x rim (r35_carton16). The tilt room has to come
# from the y span instead: the tote's own 19.8 cm side is stretched 1.5x to 29.7 cm (slot to wall
# 14.4 - 4.5 - 2.1 pipe radius = 7.8 cm instead of 3.3; r35_carton14 caught the rim at y = -0.094).
CARTON_YAW = 90.0
TOTE_SCALE_Y = 1.5             # local x of the KLT = world y at yaw 90
TOTE_SCALE_Z = 0.75            # 11 cm tote: palm (pinch + ~3 cm = 0.946) clears the 0.91 rim; co-carry is parked so the clamp no longer matters
CARTON_PALM_PRESS = 0.030
# F_L loads BOTH pipes (A then B); F_R only joins for the palm-clamp co-carry.
# Every F_R loading variant sat on the j7 limit (+90 deg wrist yaw + place
# orientation) and the limit-projected IK flipped the arm mid-carry (r35_carton6/7);
# F_L's loading geometry succeeded in every run.
# F_L packs two soup cans (6.8 x 10 cm, squat): the 18 cm pipes swing 15-30 deg in
# the fingertip pinch during the carry and topple in the tote (r35_carton13-23,
# divider pockets included -- the swung pipe lands on the rails); a squat can
# released with the same tilt settles upright.
CARTON_LAYOUT_F = {
    "A": ("can_c_a", "soup_can", (0.50, -0.36)),
    "B": ("can_c_b", "soup_can", (0.40, -0.24)),
}
CARTON_LAYOUT_U = {
    "U_L": ("can_u_l", "soup_can", (-0.44, 0.42), "bin_black_u", "sorting_bin_black", (-0.30, 0.28), 90.0),
    "U_R": ("can_u_r", "soup_can", (-0.46, -0.20), "bin_blue_u", "sorting_bin_blue", (-0.28, -0.32), 90.0),
}


def carton_objects() -> dict:
    objs = {}
    # the Isaac corrugated cartons are SEALED (the "scotch" mesh is the tape) -- a
    # pipe stalled on the lid at rim height (r35_carton2/3). The open container is
    # the KLT tote (19.8 x 29.7 x 14.6) yawed 90: 29.7 cm along x, the stretched 19.8 cm side along y
    carton = lab_object("carton", "small_klt", CARTON_XY, (CARTON_XY[0] + CARTON_DX, CARTON_XY[1]), yaw=CARTON_YAW)
    carton.scale_xyz = (TOTE_SCALE_Y, 1.0, TOTE_SCALE_Z)
    for key, (pn, pk, pxy) in CARTON_LAYOUT_F.items():
        # two slots side by side in the tote (same x, 9 cm apart in y), both shifted +2 cm: the pipe
        # hangs 3-4.5 cm behind the hand in -y after the +y carry (two-pad hinge lag) and pipe A at
        # y = -0.045 kept catching the -y wall top on the way down (r35_carton14/18)
        # cans: slots along the tote's x axis (12 cm apart, 4.4 cm to the walls) -- with
        # the y-side-by-side slots the hand placing can B knocked can A over (r35_carton24)
        slot_x = CARTON_XY[0] - CAN_SLOT_DX if key == "A" else CARTON_XY[0] + CAN_SLOT_DX
        slot_y = CARTON_SLOT_Y_LAG
        o = lab_object(pn, pk, pxy, (slot_x, slot_y))
        # pinch the pipe high so the palm casing stays above the tote rim and
        # only the fingers enter the cavity (r35_carton2: with 7 cm the palm hit
        # the wall and pushed the container away)
        objs["F_L" if key == "A" else "F_L2"] = o        # catalog can pinch (depth 5 cm, squeeze 0.18)
    # F_R sorts a third pipe into the blue bin on its side (validated pipe_sorting
    # geometry) -- the palm-clamp co-carry of the loaded tote is parked (see
    # _carton_packing_task docstring)
    objs["F_R"] = lab_object("pipe_blue_f", "exhaust_pipe_blue", (0.50, 0.36), (0.48, 0.60))
    objs["bin:F_R"] = lab_object("bin_blue_f", "sorting_bin_blue", (0.48, 0.60), None, yaw=90.0)
    for arm, (pn, pk, pxy, bn, bk, bxy, byaw) in CARTON_LAYOUT_U.items():
        objs[arm] = lab_object(pn, pk, pxy, bxy)
        objs[f"bin:{arm}"] = lab_object(bn, bk, bxy, None, yaw=byaw)
    objs["carton"] = carton
    return objs


def _carton_packing_task(objs: dict, cands: dict, env_yaml: str) -> S9Task:
    """Tote packing: F_L loads pipe A then pipe B into the KLT tote (cands["F_L"],
    cands["F_L2"]); F_R and the U arms sort their pipes into the parts bins in
    parallel with pipe A. The palm-clamp co-carry of the loaded tote is PARKED:
    in raw replays the 1.2 kg tote rose 4-7 cm and then slid in the palms
    (r35_carton8/9; the R34 bin1 co-lift at 1.1 kg is the existence proof) --
    friction-only two-palm carries are marginal and need a form-closure grip."""
    tote = objs["carton"]
    fl, fl2, fr, ul, ur = cands["F_L"], cands["F_L2"], cands["F_R"], cands["U_L"], cands["U_R"]
    ori_u = {"U_L": ul.R_flange, "U_R": ur.R_flange}
    fgl, fgl2, fgr, ugl, ugr = fl.flange_grasp, fl2.flange_grasp, fr.flange_grasp, ul.flange_grasp, ur.flange_grasp
    lift_f, lift_u = 0.12, 0.11
    # KLT floor (~1.2 cm x scale) + 4 cm release height: a pipe that is already standing on the floor
    # when the pads open is swept over by the opening fingers (r35_carton17 pipe A at 5 mm, r35_carton18
    # pipe B at 2 cm -- it hangs ~2 cm lower in the hand than at the pick), while a pipe dropped from
    # 2-7 cm lands standing (r35_pipe6 bins at PLACE_CLEAR_PHYS, r35_carton13 pipe B from 6.7 cm)
    floor_dz = 0.012 * TOTE_SCALE_Z + 0.040
    oa, ob, oc = objs["F_L"], objs["F_L2"], objs["F_R"]
    pl_a = (fgl[0] + oa.place_xy[0] - oa.pos[0], fgl[1] + oa.place_xy[1] - oa.pos[1], fgl[2] + floor_dz)
    pl_b = (fgl2[0] + ob.place_xy[0] - ob.pos[0], fgl2[1] + ob.place_xy[1] - ob.pos[1], fgl2[2] + floor_dz)
    pl_c = R26._place_rel(fgr, oc.pos, oc.place_xy, -0.005 + BIN_FLOOR_DZ)
    place_ul = _u_place(ul, objs["U_L"], 0.015 + BIN_FLOOR_DZ)
    place_ur = _u_place(ur, objs["U_R"], 0.015 + BIN_FLOOR_DZ)
    home_l, home_r = (0.44, -0.30, 1.25), (0.44, 0.30, 1.25)
    o1, o2, orr = {"F_L": fl.R_flange}, {"F_L": fl2.R_flange}, {"F_R": fr.R_flange}
    phases = [
        # ---- pipe A into the tote; F_R / U arms sort in parallel ----
        _ph("approach", 3.0, targets={"F_L": fl.flange_pre, "F_R": fr.flange_pre, "U_L": ul.flange_pre, "U_R": ur.flange_pre},
            oris={**o1, **orr, **ori_u}),
        _ph("descend", 1.5, targets={"F_L": fgl, "F_R": fgr, "U_L": ugl, "U_R": ugr}, oris={**o1, **orr, **ori_u}),
        _ph("close", 1.2),
        _ph("lift", 1.5, targets={"F_L": _above(fgl, lift_f), "F_R": _above(fgr, 0.15),
                                  "U_L": _above(ugl, lift_u), "U_R": _above(ugr, lift_u)},
            oris={**o1, **orr, **ori_u}),
        # F_L's pipe rotates in the two-pad pinch during the 33 cm carry (offset to the flange wanders
        # 4-5 cm and the pipe rides 3.5 cm up the pads, r35_carton19/20) while F_R's identical pipe
        # never moves in its pinch. Slowing the carry 4 -> 6 s (r35_carton20) changed nothing: the
        # rotation is quasi-static, not inertial -> look at the F_L hand geometry / wrist orientation
        # along the carry, not the speed
        _ph("carry", CARTON_CARRY_S, targets={"F_L": _above(pl_a, lift_f), "F_R": _above(pl_c, 0.15),
                                              "U_L": (-0.10, 0.26, 1.12), "U_R": (-0.12, -0.26, 1.12)},
            oris={**o1, **orr}),
        _ph("carry_u", 1.5, targets={"U_L": _above(place_ul, 0.10), "U_R": _above(place_ur, 0.10)}, oris=dict(ori_u),
            lerps={"F_L": dict(R34._BREATH)}),
        _ph("place", 2.5, targets={"F_L": pl_a, "F_R": pl_c, "U_L": place_ul, "U_R": place_ur},
            oris={**o1, **orr, **ori_u}),
        _ph("release", 2.0),
        _ph("clear", 1.5, targets={"F_L": _above(pl_a, 0.16), "F_R": _above(pl_c, 0.14)}, oris={**o1, **orr}),
        # ---- pipe B (same arm); F_R retreats home ----
        # F_R simply stays above its bin (y = 0.60) while F_L crosses the centre
        # line with pipe B: home_r blocked F_L's carry_r (self 1.2 mm) and a park
        # pose near the F_R base folded F_R itself (self 0.6 mm), r35_build_carton10/11
        _ph("approach_r", 3.0, targets={"F_L": fl2.flange_pre}, oris=dict(o2)),
        _ph("descend_r", 1.5, targets={"F_L": fgl2}, oris=dict(o2)),
        _ph("close_r", 1.2),
        _ph("lift_r", 1.5, targets={"F_L": _above(fgl2, lift_f)}, oris=dict(o2)),
        _ph("carry_r", CARTON_CARRY_S, targets={"F_L": _above(pl_b, lift_f)}, oris=dict(o2)),
        _ph("place_r", 2.5, targets={"F_L": pl_b}, oris=dict(o2)),
        _ph("release_r", 2.0),
        _ph("clear_r", 1.5, targets={"F_L": _above(pl_b, 0.16)}, oris=dict(o2)),
        _ph("retreat", 2.5,
            targets={"F_L": home_l, "F_R": home_r, "U_L": (-0.38, 0.42, 1.16), "U_R": (-0.40, -0.36, 1.16)}),
    ]
    na, nb, nc = oa.name, ob.name, oc.name
    nu = {a: objs[a].name for a in ("U_L", "U_R")}
    return S9Task(
        family="carton_packing",
        spec=SkillSpec("s9_carton_packing", phases),
        hand_events=R26._close_events(cands, ("F_L", "F_R", "U_L", "U_R"))
        + R26._close_events({"F_L": fl2}, ("F_L",), phase="close_r")
        + [("release", 0.75, a, "open", 0.0) for a in ("U_L", "U_R")]
        + [("release", 0.60, "F_L", "open", 0.0, 1.0), ("release", 0.90, "F_R", "open", 0.0),
           ("release_r", 0.60, "F_L", "open", 0.0, 1.0)],
        object_events=[("close", 0.90, "attach", "F_L", None, na, 0.24), ("close", 0.90, "attach", "F_R", None, nc, 0.24)]
        + [("close", 0.90, "attach", a, None, nu[a], 0.24) for a in ("U_L", "U_R")]
        + [("close_r", 0.90, "attach", "F_L", None, nb, 0.24)]
        + [("release", 0.60, "release", None, None, n, None) for n in (na, nc, nu["U_L"], nu["U_R"])]
        + [("release_r", 0.60, "release", None, None, nb, None)],
        # success = the second pipe standing in the tote (the recorder reads the pipe's bottom origin)
        # rest on the tote floor: the recorder reads the prim origin (centre for the YCB can)
        success={"kind": "pick_place", "place_xy": list(ob.place_xy), "radius_m": 0.08, "min_disp_m": 0.10,
                 "rest_z": TABLE_TOP_Z + 0.012 * TOTE_SCALE_Z + (ob.pos[2] - TABLE_TOP_Z) + ob.origin_dz},
        objects=[nb, na, nc, nu["U_L"], nu["U_R"], tote.name, objs["bin:F_R"].name, objs["bin:U_L"].name, objs["bin:U_R"].name],
        env_yaml=env_yaml,
    )


def _carton_packing_variants(provider):
    objs = carton_objects()
    # F_L picks pipe A and pipe B with the same arm; the screened combos give the
    # A / F_R / U candidates, the B candidate is screened separately (first feasible)
    gen = {a: (lambda ap, a=a: R34._top_cands(provider, a, objs[a], ap)) for a in ARM_KEYS}
    fl2 = []
    for ap in R26.PHYS_APPROACHES:
        fl2 += R26._screen(provider, "F_L", R34._top_cands(provider, "F_L", objs["F_L2"], ap), max_keep=1)
    if not fl2:
        print("R35_CARTON_PIPE_B_INFEASIBLE", flush=True)
        return
    for tag, cands in R26._screened_combos(provider, "carton_packing", gen):
        yield tag, _carton_packing_task(objs, {**cands, "F_L2": fl2[0]}, R35_ENV["carton_packing"])


# ---------------------------------------------------------------------------
# family 3: profile_colift (two-hand carry of one 70 cm 4080 profile)
# ---------------------------------------------------------------------------
# A single fingertip pinch cannot hold a long object level (PhysX two-pad
# pinch = free hinge about the pinch axis; torsional patches on object and
# fingers both failed, r34_rail3-5 / r35_rail7). Two pinches 52 cm apart
# constrain the pitch geometrically -- and carrying a long profile with two
# arms is how it is done on a real assembly line.

PROFILE_XY = (0.53, 0.0)
PROFILE_HALF_STATION = 0.31      # pinch stations at y = -/+0.31: at 0.26 the two F hands met at self 1-4 mm (flanges 24 cm apart)
PROFILE_DX = -0.16               # carry away from the F bases like the rod family: +x folded the elbows (self 2 mm, colift2)
PROFILE_LIFT = 0.10
PROFILE_SQUEEZE = 0.06           # 0.10 made the pads drag the standing profile over at release (colift6); rod-family value
PROFILE_DEPTH = R26.ROD_PHYS_DEPTH   # 1.2 cm below the top. A deeper pinch would shorten the lever arm of the
                                     # pad drag that topples the standing profile at a gated release (r35_colift8/9,
                                     # hands level), but at 4 cm the flange_z hands' idle fingers come within
                                     # 0.7-0.9 mm of the table (< 1 mm IK floor) and the finger approach never
                                     # passes F_R's trajectory IK -> no buildable variant (r35_build_..._d6)
BLOCK_LAYOUT_U = {
    "U_L": ("block_u_l", "cube_connector_50", (-0.44, 0.42), (-0.30, 0.28)),
    "U_R": ("block_u_r", "cube_connector_50", (-0.46, -0.20), (-0.32, -0.28)),
}


def colift_objects() -> dict:
    prof = lab_object("profile", "alu_profile_4080_700", PROFILE_XY, (PROFILE_XY[0] + PROFILE_DX, PROFILE_XY[1]))
    objs = {"F_L": prof, "F_R": prof}
    for arm, (n, k, xy, pxy) in BLOCK_LAYOUT_U.items():
        objs[arm] = lab_object(n, k, xy, pxy)
    return objs


def _profile_colift_task(objs: dict, cands: dict, env_yaml: str) -> S9Task:
    prof = objs["F_L"]
    fl, fr, ul, ur = (cands[a] for a in ("F_L", "F_R", "U_L", "U_R"))
    ori_f = {"F_L": fl.R_flange, "F_R": fr.R_flange}
    ori_u = {"U_L": ul.R_flange, "U_R": ur.R_flange}
    fgl, fgr, ugl, ugr = fl.flange_grasp, fr.flange_grasp, ul.flange_grasp, ur.flange_grasp
    lift_u = 0.11
    place_ul = _u_place(ul, objs["U_L"], 0.015)
    place_ur = _u_place(ur, objs["U_R"], 0.015)
    # both pinches translate identically -> the profile stays level and straight;
    # no downward overshoot at the place: the standing 4 x 8 cm profile was
    # pressed into the table and tipped over when the pads opened (colift3)
    dx = (PROFILE_DX, 0.0, 0.0)
    fgl_c = tuple(a + b for a, b in zip(fgl, dx))
    fgr_c = tuple(a + b for a, b in zip(fgr, dx))
    phases = [
        _ph("approach", 3.0,
            targets={"F_L": fl.flange_pre, "F_R": fr.flange_pre, "U_L": ul.flange_pre, "U_R": ur.flange_pre},
            oris={**ori_f, **ori_u}),
        _ph("descend", 1.5, targets={"F_L": fgl, "F_R": fgr, "U_L": ugl, "U_R": ugr}, oris={**ori_f, **ori_u}),
        _ph("close", 1.2),
        _ph("lift", 2.0,
            targets={"F_L": _above(fgl, PROFILE_LIFT), "F_R": _above(fgr, PROFILE_LIFT),
                     "U_L": _above(ugl, lift_u), "U_R": _above(ugr, lift_u)},
            oris={**ori_f, **ori_u}),
        _ph("carry", 3.0,
            targets={"F_L": _above(fgl_c, PROFILE_LIFT), "F_R": _above(fgr_c, PROFILE_LIFT),
                     "U_L": (-0.10, 0.26, 1.12), "U_R": (-0.12, -0.26, 1.12)},
            oris={**ori_f, **ori_u}),
        _ph("carry_u", 1.5,
            targets={"U_L": _above(place_ul, 0.10), "U_R": _above(place_ur, 0.10)},
            oris=dict(ori_u), lerps={"F_L": dict(R34._BREATH)}),
        _ph("place", 2.5,
            targets={"F_L": fgl_c, "F_R": fgr_c, "U_L": place_ul, "U_R": place_ur},
            oris={**ori_f, **ori_u}),
        # let a gate-attenuated hand catch up before the pads open: under s0 F_R was
        # still 1.3 cm above F_L at the end of the place (2.7-3.8 cm during it), the
        # F_L end of the profile hit the table first and the profile toppled on
        # release (r35_colift8 s0; raw with both hands level stood it up)
        _ph("hold", 1.0),
        _ph("release", 2.5),
        _ph("settle", 1.0),
        _ph("clear", 1.5, targets={"F_L": _above(fgl_c, 0.14), "F_R": _above(fgr_c, 0.14)}, oris=dict(ori_f)),
        _ph("retreat", 2.5,
            targets={"F_L": (0.44, -0.30, 1.25), "F_R": (0.44, 0.30, 1.25),
                     "U_L": (-0.38, 0.42, 1.16), "U_R": (-0.40, -0.36, 1.16)}),
    ]
    names = {"U_L": objs["U_L"].name, "U_R": objs["U_R"].name}
    return S9Task(
        family="profile_colift",
        spec=SkillSpec("s9_profile_colift", phases),
        hand_events=R26._close_events(cands, ARM_KEYS)
        + [("release", 0.75, a, "open", 0.0) for a in ("U_L", "U_R")]
        # slow symmetric opening (ramp 1.0 over the 2.5 s release): the pads drag the
        # standing profile sideways when they snap open
        + [("release", 0.30, a, "open", 0.0, 1.0) for a in ("F_L", "F_R")],
        object_events=[("close", 0.90, "attach", "F_L", None, prof.name, 0.40)]
        + [("close", 0.90, "attach", a, None, names[a], 0.24) for a in ("U_L", "U_R")]
        + [("release", 0.60, "release", None, None, n, None) for n in (prof.name, names["U_L"], names["U_R"])],
        success={"kind": "pick_place", "place_xy": list(prof.place_xy),
                 "radius_m": 0.10, "min_disp_m": 0.08, "rest_z": prof.pos[2] + prof.origin_dz},
        objects=[prof.name, names["U_L"], names["U_R"]],
        env_yaml=env_yaml,
    )


def _profile_colift_variants(provider):
    objs = colift_objects()
    prof = objs["F_L"]
    size = prof.size
    st_l = (PROFILE_XY[0], PROFILE_XY[1] - PROFILE_HALF_STATION, prof.pos[2])
    st_r = (PROFILE_XY[0], PROFILE_XY[1] + PROFILE_HALF_STATION, prof.pos[2])
    gen = {
        "F_L": lambda ap: grip_station_cands(provider, "F_L", st_l, size[0],
                                             R26._params_for("F_L", size[0], GRASP_PARAMS, ap, size[2],
                                                             grasp_depth=PROFILE_DEPTH,
                                                             squeeze_extra=PROFILE_SQUEEZE)),
        "F_R": lambda ap: grip_station_cands(provider, "F_R", st_r, size[0],
                                             R26._params_for("F_R", size[0], GRASP_PARAMS, ap, size[2],
                                                             grasp_depth=PROFILE_DEPTH,
                                                             squeeze_extra=PROFILE_SQUEEZE)),
        "U_L": lambda ap: R34._top_cands(provider, "U_L", objs["U_L"], ap),
        "U_R": lambda ap: R34._top_cands(provider, "U_R", objs["U_R"], ap),
    }
    for tag, cands in R26._screened_combos(provider, "profile_colift", gen):
        yield tag, _profile_colift_task(objs, cands, R35_ENV["profile_colift"])


# ---------------------------------------------------------------------------
# registration into the R34 build machinery + dressed env yaml
# ---------------------------------------------------------------------------

R34._VARIANTS.update({"pipe_sorting": _pipe_sorting_variants, "carton_packing": _carton_packing_variants,
                      "profile_colift": _profile_colift_variants})
R34._OBJECTS.update({"pipe_sorting": sort_objects, "carton_packing": carton_objects,
                     "profile_colift": colift_objects})
R34.R34_ENV.update(R35_ENV)


def _tote_divider_props() -> list:
    """KLT divider insert (industrial practice): a 2-pocket grid of 3.5 cm rails
    around the two pipe slots. The fingertip-pinch hinge swings the pipe 15-30
    deg during the carry (r35_carton22 video) and nothing at the trajectory level
    removed it; a pocket catches the tilted pipe and it settles against the rail
    instead of falling over. Static kinematic colliders resting on the tote floor."""
    ax = CARTON_XY[0] - CARTON_SLOT_DX
    ya = -CARTON_SLOT_DY + CARTON_SLOT_Y_LAG
    yb = CARTON_SLOT_DY + CARTON_SLOT_Y_LAG
    half = 0.045                          # 9 cm pockets
    z = TABLE_TOP_Z + 0.012 * TOTE_SCALE_Z + 0.0175
    root = "assets_real/objects/industrial"
    props = []
    for i, y in enumerate((ya - half, yb - half, yb + half)):   # rails along x (pocket y-walls)
        props.append({"name": f"divider_x{i}", "usd": f"{root}/divider_rail_110/divider_rail_110.usda",
                      "pos": [round(ax, 4), round(y, 4), round(z, 4)], "yaw": 0.0, "collision": True})
    for i, x in enumerate((ax - 0.05, ax + 0.05)):               # rails along y (pocket x-walls)
        props.append({"name": f"divider_y{i}", "usd": f"{root}/divider_rail_190/divider_rail_190.usda",
                      "pos": [round(x, 4), round(0.5 * (ya + yb), 4), round(z, 4)], "yaw": 90.0, "collision": True})
    return props


# family-specific fixtures appended to the dressing static props
def _profile_jig_props() -> list:
    """Assembly jig at the profile's place position: two 70 cm stop rails forming a
    5 cm channel (3 cm tall) for the 4 cm wide profile. The two-hand carry sets the
    profile down level, but the opening pads tip the 4 x 8 cm standing profile in
    about half of the renders (colift3/6/11 vs 4/8/9); in the channel it can lean
    on a rail at most 1 cm. Fingertips (pinch 1.2 cm below the top) stay > 2 cm
    above the rails."""
    x0 = PROFILE_XY[0] + PROFILE_DX
    z = TABLE_TOP_Z + 0.0175
    root = "assets_real/objects/industrial"
    return [{"name": f"jig_rail_{i}", "usd": f"{root}/jig_rail_700/jig_rail_700.usda",
             "pos": [round(x0 + dx, 4), PROFILE_XY[1], round(z, 4)], "yaw": 90.0, "collision": True}
            for i, dx in enumerate((-0.030, 0.030))]


FAMILY_STATIC_PROPS = {"profile_colift": _profile_jig_props}
# (tote divider insert retired: the swung pipe landed on the rails, r35_carton23)


def write_env_yaml(family: str, out_path: "str | None" = None, dressing: bool = True) -> str:
    """R35 recording env: R27 physical-grasp knobs + catalog objects + the
    factory-cell dressing block + the hand torsional patch."""
    import yaml

    from safeduo.configs import CONFIG_DIR

    objs = {o.name: o for o in R34._OBJECTS[family]().values()}
    doc = {
        "extends": "duo_env_v7_r26_bottle_r27r16.yaml",
        "assets": {
            "object_catalog": R34.CATALOG_LAB,
            "table_objects": [o.yaml_entry() for o in objs.values()],
            "hand_torsional_patch": {"radius": 0.02, "min_radius": 0.01},
        },
    }
    if dressing:
        with open(os.path.join(str(CONFIG_DIR), DRESSING_YAML)) as f:
            doc["dressing"] = yaml.safe_load(f)["dressing"]
    if family in FAMILY_STATIC_PROPS:
        doc.setdefault("dressing", {}).setdefault("static_props", [])
        doc["dressing"]["static_props"] = list(doc["dressing"]["static_props"]) + FAMILY_STATIC_PROPS[family]()
    out_path = out_path or os.path.join(str(CONFIG_DIR), R35_ENV[family])
    with open(out_path, "w") as f:
        f.write(f"# R35 (auto-generated by task_library_r35.write_env_yaml): {family} recording env "
                f"(factory-cell dressing from {DRESSING_YAML})\n")
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True, width=200)
    return out_path


def main(argv: "list | None" = None) -> int:
    import argparse
    import time

    from safeduo.delta.task_record_s9 import make_s9_provider

    ap = argparse.ArgumentParser(description="R35 industrial task families (factory cell)")
    ap.add_argument("stage", choices=["build", "env"])
    ap.add_argument("--task", default="pipe_sorting", choices=list(R35_ENV) + ["all"])
    ap.add_argument("--out", default="artifacts/task_trajs_r35")
    ap.add_argument("--hand-calib", default="v7")
    ap.add_argument("--spheres", default="r16", choices=["v7", "r16"])
    ap.add_argument("--struct-exempt", action="store_true")
    ap.add_argument("--multiseed", action="store_true")
    ap.add_argument("--no-dressing", action="store_true")
    ap.add_argument("--no-joint-limits", action="store_true",
                    help="R35 default projects the design IK onto the FR3 joint limits (r35_carton5 lesson)")
    args = ap.parse_args(argv)
    fams = list(R35_ENV) if args.task == "all" else [args.task]
    if args.stage == "env":
        for fam in fams:
            print(write_env_yaml(fam, dressing=not args.no_dressing))
        return 0
    if args.struct_exempt:
        from safeduo.delta.skill_record import DESIGN_STRUCT_EXEMPT
        DESIGN_STRUCT_EXEMPT["on"] = True
    if args.multiseed:
        from safeduo.delta.skill_record_v7 import IK_MULTISEED
        IK_MULTISEED["on"] = True
        IK_MULTISEED["verbose"] = True
    if not args.no_joint_limits:
        from safeduo.delta.skill_record import IK_JOINT_LIMITS
        IK_JOINT_LIMITS["on"] = True
    R26.PHYS_GRASP["on"] = True
    R26.PHYS_GRASP["calib"] = args.hand_calib
    provider = make_s9_provider(1, device="cpu", spheres=args.spheres)
    rc = 0
    for fam in fams:
        t0 = time.time()
        try:
            rep = R34.build_task_r34(provider, fam, Path(args.out) / fam)
        except RuntimeError as e:
            print(f"R35_BUILD_FAILED {fam}: {e}", flush=True)
            rc = 1
            continue
        print(f"R35_BUILD_OK {fam} in {time.time() - t0:.1f}s gate={rep.get('hard_gate_margin_gt0')}", flush=True)
        joint_limit_report(Path(args.out) / fam / f"task_{fam}.npz")
        write_env_yaml(fam, dressing=not args.no_dressing)
    return rc


def joint_limit_report(npz_path) -> dict:
    """Design QA: count steps outside the FR3 / UR joint limits per arm."""
    import numpy as np

    from safeduo.delta.skill_record import IK_JOINT_LIMITS

    d = np.load(npz_path, allow_pickle=True)
    out = {}
    for arm in ARM_KEYS:
        q = d[f"q_{arm}"]
        lo, hi = (np.asarray(v) for v in IK_JOINT_LIMITS[arm[0]])
        viol = int(((q < lo) | (q > hi)).any(axis=1).sum())
        worst = float(np.maximum(lo - q, q - hi).max())
        out[arm] = {"steps_outside": viol, "worst_overshoot_rad": round(worst, 3)}
    print(f"R35_JOINT_LIMIT_CHECK {npz_path}: {out}", flush=True)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
