"""R34 (2026-09-06): laboratory task families on sim-ready mesh assets.

Owner direction (09-06): "their tasks are good -- build our scene properly,
collect suitable object assets; the current objects are too crude" (reference:
a four-arm MuJoCo assembly benchmark -- aluminium-profile assembly, fine
sorting, collaborative packaging). SafeDuo is a SafeLab project, so the first
asset set is the psilab sim-ready laboratory glassware (configs/
object_catalog_lab.yaml): reagent bottles, beakers, test tubes and racks,
flasks, crucibles. Every family here is built from the catalog (bounding box
= grasp-planner proxy, per-object grasp hints) on top of the R27 physical
grasp machinery (calibrated hand frames, screened candidates, multi-seed IK).

Families
  reagent_pick   F_L/F_R pick a brown / clear reagent bottle by the body and
                 stand it in their placing zones; U_L/U_R claw a beaker and a
                 crucible into theirs (four independent physical pick-places,
                 the sorting archetype)
  rack_colift    (next) two F hands carry the test-tube rack by its ends
  beaker_relay   (next) indirect relay of a beaker across the F/U seam
  tube_racking   (next) insert test tubes into the rack (fine assembly)

CLI (design-time, CPU):
  python -m safeduo.delta.task_library_r34 build --task reagent_pick \
      --out artifacts/task_trajs_r34 --hand-calib v7 --spheres r16 --struct-exempt --multiseed
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from safeduo.delta import task_library_r26 as R26
from safeduo.delta.grasp_gen import box_antipodal_grasps
from safeduo.delta.skill_record import SkillSpec
from safeduo.delta.task_library_r24 import (U_GRASP_PARAMS, _BREATH, _above, _arm_prior, _ph,
                                            audit_arm_participation)
from safeduo.delta.task_record_s9 import GRASP_PARAMS, S9Task
from safeduo.envs.task_objects import load_object_catalog
from safeduo.safety.types import ARM_KEYS

CATALOG_LAB = "object_catalog_lab.yaml"
TABLE_TOP_Z = 0.80
R34_ENV = {"reagent_pick": "duo_env_v7_r34_reagent_pick.yaml",
           "rack_colift": "duo_env_v7_r34_rack_colift.yaml",
           "beaker_relay": "duo_env_v7_r34_beaker_relay.yaml",
           "tube_racking": "duo_env_v7_r34_tube_racking.yaml",
           "bin_packing": "duo_env_v7_r34_bin_packing.yaml"}


@dataclass
class LabObject:
    """One catalog object on the table: where it starts and (optionally) where
    the arm should stand it down. z of `pos` is the object centre (bbox/2
    above the table top) unless given."""

    name: str
    catalog: str
    xy: tuple
    place_xy: "tuple | None" = None
    yaw: float = 0.0
    entry: dict = field(default_factory=dict)
    z_override: "float | None" = None      # e.g. a tube standing in a rack hole
    mass_override: "float | None" = None
    scale: float = 1.0
    scale_xyz: "tuple | None" = None        # non-uniform scale (e.g. a shallow crate)

    @property
    def size(self) -> tuple:
        sc = self.scale_xyz if self.scale_xyz is not None else (self.scale,) * 3
        return tuple(float(v) * float(k) for v, k in zip(self.entry["size"], sc))

    @property
    def pos(self) -> tuple:
        z = self.z_override if self.z_override is not None else TABLE_TOP_Z + 0.5 * self.size[2]
        return (float(self.xy[0]), float(self.xy[1]), float(z))

    @property
    def grasp(self) -> dict:
        return dict(self.entry.get("grasp") or {})

    def meta(self) -> dict:
        return {"name": self.name, "catalog": self.catalog, "size": list(self.size),
                "init_pos": list(self.pos), "usd": self.entry.get("usd"),
                "mass": self.entry.get("mass"),
                "place_xy": (list(self.place_xy) if self.place_xy else None)}

    @property
    def origin_dz(self) -> float:
        """prim origin relative to the bbox centre (catalog `origin_dz`, e.g.
        -h/2 for bottom-origin Isaac props); the grasp geometry uses the centre,
        the spawn pose uses the origin (R35)."""
        v = self.entry.get("origin_dz")          # metres, at the catalog size
        sc = self.scale_xyz[2] if self.scale_xyz is not None else self.scale
        return float(v) * float(sc) if v is not None else 0.0

    def yaml_entry(self) -> dict:
        p = self.pos
        d = {"name": self.name, "catalog": self.catalog,
             "pos": [round(p[0], 4), round(p[1], 4), round(p[2] + self.origin_dz, 4)]}
        if self.yaw:
            d["yaw"] = self.yaw
        if self.mass_override is not None:
            d["mass"] = self.mass_override
        if self.scale_xyz is not None:
            d["scale"] = [float(v) for v in self.scale_xyz]
        elif self.scale != 1.0:
            d["scale"] = self.scale
        return d


def catalog() -> dict:
    return load_object_catalog(CATALOG_LAB)


def lab_object(name: str, key: str, xy, place_xy=None, yaw: float = 0.0) -> LabObject:
    cat = catalog()
    if key not in cat:
        raise KeyError(f"{key!r} not in {CATALOG_LAB}")
    return LabObject(name, key, tuple(xy), (tuple(place_xy) if place_xy else None), yaw, dict(cat[key]))


# ---------------------------------------------------------------------------
# grasp candidates from catalog hints
# ---------------------------------------------------------------------------


def _top_cands(provider, arm: str, obj: LabObject, approach: str) -> list:
    """Top-kind antipodal candidates on the object's bounding box with the
    calibrated hand frame for its grasp width; depth from the catalog."""
    g = obj.grasp
    width = float(g.get("width", min(obj.size[0], obj.size[1])))
    legacy = GRASP_PARAMS if arm.startswith("F") else U_GRASP_PARAMS
    if arm.startswith("F"):
        prm = R26._params_for(arm, width, legacy, approach, obj.size[2],
                              grasp_depth=float(g.get("depth_from_top", 0.025)),
                              squeeze_extra=float(g.get("squeeze_extra", 0.06)),
                              f_max=(float(g["f_max"]) if "f_max" in g else None))
    else:
        # R35: wide objects (6.8 cm can) sit at the edge of the DFX claw's validated
        # 5-7 cm band and dropped mid-carry -> catalog hints can firm the claw
        prm = R26._params_for(arm, width, legacy, approach, obj.size[2],
                              grasp_depth=float(g.get("depth_from_top", 0.025)),
                              squeeze_extra=float(g.get("u_squeeze_extra", R26.DFX_CUBE_SQUEEZE)),
                              thumb_extra=float(g.get("u_thumb_extra", R26.DFX_CUBE_THUMB_EXTRA)))
    ref_R, base_xy = _arm_prior(provider, arm)
    # the planner box is the object's footprint at the grasp width (a bottle's
    # neck/body diameter), not the full bbox -- keeps the pinch on the body
    size = (width, width, obj.size[2])
    return [c for c in box_antipodal_grasps(obj.pos, size, prm, ref_R=ref_R, base_xy=base_xy)
            if c.kind == "top"]


# ---------------------------------------------------------------------------
# family 1: reagent_pick (four independent physical pick-places)
# ---------------------------------------------------------------------------

REAGENT_LAYOUT = {
    # F arms: reagent bottles where the R26 bottles stood, placed on the same
    # equal-reach arcs; U arms: beaker / crucible where the R26 cubes stood
    "F_L": ("bottle_brown", "brown_reagent_bottle_large", (0.50, -0.36), (0.48, -0.57)),
    "F_R": ("bottle_clear", "clear_reagent_bottle_large", (0.50, 0.36), (0.48, 0.57)),
    "U_L": ("beaker", "glass_beaker_100ml", (-0.44, 0.42), (-0.30, 0.28)),
    "U_R": ("crucible", "crucible", (-0.46, -0.20), (-0.32, -0.28)),
}


def reagent_objects() -> dict:
    return {arm: lab_object(n, k, xy, pxy) for arm, (n, k, xy, pxy) in REAGENT_LAYOUT.items()}


def _pick_place_task(family: str, objs: dict, cands: dict, env_yaml: str,
                     place_dz: float = 0.0, extra_objects: "list | None" = None) -> S9Task:
    """Four independent pick & places (R26 bottle_pick skeleton): F arms carry
    on a high lane and stand the object down with a 5 mm overshoot; U arms
    go through the audited centre-lane waypoints. place_dz raises every place
    target (R35: standing parts down inside a parts bin); extra_objects are
    passive scene objects listed in the task (the bins)."""
    fl, fr, ul, ur = (cands[a] for a in ("F_L", "F_R", "U_L", "U_R"))
    ori_f = {"F_L": fl.R_flange, "F_R": fr.R_flange}
    ori_u = {"U_L": ul.R_flange, "U_R": ur.R_flange}
    fgl, fgr, ugl, ugr = fl.flange_grasp, fr.flange_grasp, ul.flange_grasp, ur.flange_grasp
    lift_f, lift_u = 0.15, 0.11
    pl_l = R26._place_rel(fgl, objs["F_L"].pos, objs["F_L"].place_xy, -0.005 + place_dz)
    pl_r = R26._place_rel(fgr, objs["F_R"].pos, objs["F_R"].place_xy, -0.005 + place_dz)
    # object-relative like the F arms: the absolute-flange form put the U object one
    # pinch offset (7-12 cm) off its bin centre (R35 r35_pipe6 can against the wall,
    # r35_carton13 cans on the bin rim, r35_colift3-6 connector blocks 12 cm short)
    place_ul = R26._place_rel(ugl, objs["U_L"].pos, objs["U_L"].place_xy, 0.015 + place_dz)
    place_ur = R26._place_rel(ugr, objs["U_R"].pos, objs["U_R"].place_xy, 0.015 + place_dz)
    phases = [
        _ph("approach", 3.0,
            targets={"F_L": fl.flange_pre, "F_R": fr.flange_pre, "U_L": ul.flange_pre, "U_R": ur.flange_pre},
            oris={**ori_f, **ori_u}),
        _ph("descend", 1.5, targets={"F_L": fgl, "F_R": fgr, "U_L": ugl, "U_R": ugr}, oris={**ori_f, **ori_u}),
        _ph("close", 1.2),
        _ph("lift", 1.5,
            targets={"F_L": _above(fgl, lift_f), "F_R": _above(fgr, lift_f),
                     "U_L": _above(ugl, lift_u), "U_R": _above(ugr, lift_u)},
            oris={**ori_f, **ori_u}),
        _ph("carry", 3.0,
            targets={"F_L": _above(pl_l, lift_f), "F_R": _above(pl_r, lift_f),
                     "U_L": (-0.10, 0.26, 1.12), "U_R": (-0.12, -0.26, 1.12)},
            oris=dict(ori_f)),
        _ph("carry_u", 1.5,
            targets={"U_L": _above(place_ul, 0.10), "U_R": _above(place_ur, 0.10)},
            oris=dict(ori_u), lerps={"F_L": dict(_BREATH)}),
        _ph("place", 2.5,
            targets={"F_L": pl_l, "F_R": pl_r, "U_L": place_ul, "U_R": place_ur},
            oris={**ori_f, **ori_u}),
        _ph("release", 2.5),
        _ph("clear", 1.2, targets={"F_L": _above(pl_l, 0.14), "F_R": _above(pl_r, 0.14)}, oris=dict(ori_f)),
        _ph("retreat", 2.5,
            targets={"F_L": (0.44, -0.30, 1.25), "F_R": (0.44, 0.30, 1.25),
                     "U_L": (-0.38, 0.42, 1.16), "U_R": (-0.40, -0.36, 1.16)}),
    ]
    names = {a: objs[a].name for a in ARM_KEYS}
    return S9Task(
        family=family,
        spec=SkillSpec(f"s9_{family}", phases),
        hand_events=R26._close_events(cands, ARM_KEYS)
        + [("release", 0.75, a, "open", 0.0) for a in ("U_L", "U_R")]
        + [("release", 0.90, a, "open", 0.0) for a in ("F_L", "F_R")],
        object_events=[("close", 0.90, "attach", a, None, names[a], 0.24) for a in ARM_KEYS]
        + [("release", 0.60, "release", None, None, names[a], None) for a in ARM_KEYS],
        success={"kind": "pick_place", "place_xy": list(objs["F_L"].place_xy),
                 # the recorder reads the prim origin -> bottom-origin assets rest at centre + origin_dz
                 "radius_m": 0.10, "min_disp_m": 0.15, "rest_z": objs["F_L"].pos[2] + place_dz + objs["F_L"].origin_dz},
        objects=[names["F_L"], names["F_R"], names["U_L"], names["U_R"]] + list(extra_objects or []),
        env_yaml=env_yaml,
    )


def _reagent_pick_variants(provider):
    objs = reagent_objects()
    gen = {a: (lambda ap, a=a: _top_cands(provider, a, objs[a], ap)) for a in ARM_KEYS}
    for tag, cands in R26._screened_combos(provider, "reagent_pick", gen):
        yield tag, _pick_place_task("reagent_pick", objs, cands, R34_ENV["reagent_pick"])


# ---------------------------------------------------------------------------
# family 2: rack_colift (two F palms clamp the test-tube rack -- the box
# co-lift recipe on a real 22 cm rack; U arms carry a beaker and a crucible)
# ---------------------------------------------------------------------------

RACK_XY = (0.53, 0.0)
RACK_YAW = 90.0            # asset long axis is x; stand it along y like the R26 box
RACK_DX = 0.14             # carry towards the F bases (R26 box lesson: -x stretches the arms)
RACK_LAYOUT_U = {
    "U_L": ("beaker", "glass_beaker_100ml", (-0.44, 0.42), (-0.30, 0.28)),
    "U_R": ("crucible", "crucible", (-0.46, -0.20), (-0.32, -0.28)),
}


def rack_objects() -> dict:
    objs = {"F_L": lab_object("rack", "test_tube_rack", RACK_XY, (RACK_XY[0] + RACK_DX, RACK_XY[1]), yaw=RACK_YAW)}
    objs["F_R"] = objs["F_L"]
    for arm, (n, k, xy, pxy) in RACK_LAYOUT_U.items():
        objs[arm] = lab_object(n, k, xy, pxy)
    return objs


def _rack_clamp_task(objs: dict, cands: dict, env_yaml: str) -> S9Task:
    """Bimanual palm clamp of the rack (box_colift skeleton): press, lift,
    carry +x, set down, unclamp, rise with the orientation locked, retreat."""
    rack = objs["F_L"]
    fl, fr, ul, ur = (cands[a] for a in ("F_L", "F_R", "U_L", "U_R"))
    ori_f = {"F_L": fl.R_flange, "F_R": fr.R_flange}
    ori_u = {"U_L": ul.R_flange, "U_R": ur.R_flange}
    fgl, fgr, ugl, ugr = fl.flange_grasp, fr.flange_grasp, ul.flange_grasp, ur.flange_grasp
    lift_u = 0.11
    place_ul = R26._place_rel(ugl, objs["U_L"].pos, objs["U_L"].place_xy, 0.015)
    place_ur = R26._place_rel(ugr, objs["U_R"].pos, objs["U_R"].place_xy, 0.015)
    fgl_c = (fgl[0] + RACK_DX, fgl[1], fgl[2])
    fgr_c = (fgr[0] + RACK_DX, fgr[1], fgr[2])
    out_l = (fgl_c[0], fgl_c[1] - 0.05, fgl_c[2])
    out_r = (fgr_c[0], fgr_c[1] + 0.05, fgr_c[2])
    phases = [
        _ph("approach", 3.0,
            targets={"F_L": fl.flange_pre, "F_R": fr.flange_pre, "U_L": ul.flange_pre, "U_R": ur.flange_pre},
            oris={**ori_f, **ori_u}),
        _ph("descend", 1.5, targets={"F_L": fgl, "F_R": fgr, "U_L": ugl, "U_R": ugr}, oris={**ori_f, **ori_u}),
        _ph("close", 1.2),
        _ph("lift", 1.5,
            targets={"F_L": _above(fgl, R26.BOX_PHYS_LIFT), "F_R": _above(fgr, R26.BOX_PHYS_LIFT),
                     "U_L": _above(ugl, lift_u), "U_R": _above(ugr, lift_u)},
            oris={**ori_f, **ori_u}),
        _ph("carry", 3.0,
            targets={"F_L": _above(fgl_c, R26.BOX_PHYS_LIFT), "F_R": _above(fgr_c, R26.BOX_PHYS_LIFT),
                     "U_L": (-0.10, 0.26, 1.12), "U_R": (-0.12, -0.26, 1.12)},
            oris=dict(ori_f)),
        _ph("carry_u", 1.5,
            targets={"U_L": _above(place_ul, 0.10), "U_R": _above(place_ur, 0.10)},
            oris=dict(ori_u), lerps={"F_L": dict(_BREATH)}),
        _ph("place", 2.5,
            targets={"F_L": fgl_c, "F_R": fgr_c, "U_L": place_ul, "U_R": place_ur},
            oris={**ori_f, **ori_u}),
        _ph("release", 1.0),
        _ph("clear", 1.5, targets={"F_L": out_l, "F_R": out_r}, oris=dict(ori_f)),
        _ph("clear_up", 1.5,
            targets={"F_L": _above(out_l, 0.15), "F_R": _above(out_r, 0.15)}, oris=dict(ori_f)),
        _ph("retreat", 2.5,
            targets={"F_L": (0.44, -0.30, 1.25), "F_R": (0.44, 0.30, 1.25),
                     "U_L": (-0.38, 0.42, 1.16), "U_R": (-0.40, -0.36, 1.16)}),
    ]
    names = {"U_L": objs["U_L"].name, "U_R": objs["U_R"].name}
    return S9Task(
        family="rack_colift",
        spec=SkillSpec("s9_rack_colift", phases),
        hand_events=R26._close_events(cands, ("U_L", "U_R"))
        + [("close", 0.10, a, "close", R26.BOX_F_CURL, 0.6, 0.0) for a in ("F_L", "F_R")]
        + [("release", 0.75, a, "open", 0.0) for a in ARM_KEYS],
        object_events=[("close", 0.90, "attach", "F_L", None, rack.name, 0.20)]
        + [("close", 0.90, "attach", a, None, names[a], None) for a in ("U_L", "U_R")]
        + [("release", 0.60, "release", None, None, n, None) for n in (rack.name, names["U_L"], names["U_R"])],
        success={"kind": "pick_place", "place_xy": list(rack.place_xy),
                 "radius_m": 0.10, "min_disp_m": 0.10, "rest_z": rack.pos[2]},
        objects=[rack.name, names["U_L"], names["U_R"]],
        env_yaml=env_yaml,
    )


def _rack_colift_variants(provider):
    objs = rack_objects()
    rack = objs["F_L"]
    half_len = 0.5 * rack.size[0]          # 0.219 along +/-y after the 90 deg yaw
    # palm centre 3.5 cm above the top, but never so low that the downward
    # fingers (11.5 cm below the palm centre) reach the table (rack top 0.87
    # vs the R26 box 0.90: every palm candidate was table-blocked)
    z_palm = max(rack.pos[2] + 0.5 * rack.size[2] + 0.035, TABLE_TOP_Z + 0.135)
    face_l = (RACK_XY[0], RACK_XY[1] - half_len, z_palm)
    face_r = (RACK_XY[0], RACK_XY[1] + half_len, z_palm)
    gen = {
        "F_L": lambda ap: [R26._palm_cand("F_L", "palm_fz-", face_l, (0.0, 1.0, 0.0), (0.0, 0.0, -1.0),
                                          R26.BOX_PALM_PRE, R26.BOX_PALM_PRESS)],
        "F_R": lambda ap: [R26._palm_cand("F_R", "palm_fz-", face_r, (0.0, -1.0, 0.0), (0.0, 0.0, -1.0),
                                          R26.BOX_PALM_PRE, R26.BOX_PALM_PRESS)],
        "U_L": lambda ap: _top_cands(provider, "U_L", objs["U_L"], ap),
        "U_R": lambda ap: _top_cands(provider, "U_R", objs["U_R"], ap),
    }
    for tag, cands in R26._screened_combos(provider, "rack_colift", gen):
        yield tag, _rack_clamp_task(objs, cands, R34_ENV["rack_colift"])


# ---------------------------------------------------------------------------
# family 3: beaker_relay (indirect relay of a beaker across the F/U seam)
# ---------------------------------------------------------------------------

RELAY_START_XY = (0.52, -0.40)
RELAY_LANE_XY = (0.10, -0.40)
RELAY_U_HOME_XY = (-0.33, -0.26)
RELAY_LIFT = 0.10
# the giver's pads drag the object while closing (R27 baton: 2.8/1.7 cm); the
# receiver's claw is centred on where the object actually stands
RELAY_DRAG_XY = (0.028, 0.017)


def relay_objects() -> dict:
    o = lab_object("beaker", "glass_beaker_100ml", RELAY_START_XY, RELAY_LANE_XY)
    return {"F_L": o, "U_R": o}


def _relay_task(objs: dict, cands: dict, env_yaml: str) -> S9Task:
    obj = objs["F_L"]
    g, r = cands["F_L"], cands["U_R"]
    ori_g, ori_r = {"F_L": g.R_flange}, {"U_R": r.R_flange}
    fg, rg = g.flange_grasp, r.flange_grasp
    dxy = (RELAY_LANE_XY[0] - obj.pos[0], RELAY_LANE_XY[1] - obj.pos[1])
    f_lift = (fg[0], fg[1], fg[2] + RELAY_LIFT)
    f_carry = (fg[0] + dxy[0], fg[1] + dxy[1], fg[2] + RELAY_LIFT)
    f_place = (fg[0] + dxy[0], fg[1] + dxy[1], fg[2] - 0.005)
    f_clear = (f_place[0] + 0.10, f_place[1], f_place[2] + 0.10)
    u_lift = (rg[0], rg[1], rg[2] + RELAY_LIFT)
    u_home = (rg[0] + RELAY_U_HOME_XY[0] - RELAY_LANE_XY[0],
              rg[1] + RELAY_U_HOME_XY[1] - RELAY_LANE_XY[1], rg[2] + RELAY_LIFT)
    u_rest = (u_home[0], u_home[1], u_home[2] + 0.02)
    phases = [
        _ph("reach", 2.5,
            targets={"F_L": g.flange_pre, "U_R": (-0.28, -0.32, 1.14),
                     "F_R": (0.46, 0.34, 1.20), "U_L": (-0.32, 0.34, 1.16)},
            oris=dict(ori_g)),
        _ph("descend", 1.5, targets={"F_L": fg, "F_R": (0.46, 0.34, 1.14), "U_L": (-0.30, 0.35, 1.14)},
            oris=dict(ori_g)),
        _ph("close", 1.4, targets={"U_L": (-0.18, 0.36, 1.14)}),
        _ph("lift", 1.5, targets={"F_L": f_lift, "F_R": (0.30, 0.39, 1.14)}, oris=dict(ori_g)),
        _ph("meet", 3.0, targets={"F_L": f_carry}, oris=dict(ori_g)),
        _ph("place", 2.5, targets={"F_L": f_place}, oris=dict(ori_g)),
        _ph("settle_hold", 1.0),
        _ph("release", 1.5),
        _ph("clear", 1.5, targets={"F_L": f_clear}, oris=dict(ori_g), lerps={"U_L": dict(_BREATH)}),
        _ph("handoff", 2.5, targets={"F_L": (0.44, -0.48, 1.25), "U_R": r.flange_pre}, oris={**ori_g, **ori_r}),
        _ph("descend_u", 1.5, targets={"U_R": rg}, oris=dict(ori_r)),
        _ph("transfer", 1.4, targets={"F_R": (0.17, 0.41, 1.14)}),
        _ph("lift_u", 1.5, targets={"U_R": u_lift}, oris=dict(ori_r)),
        _ph("carry_u", 2.5, targets={"U_R": u_home}, oris=dict(ori_r), lerps={"U_L": dict(_BREATH)}),
        _ph("retreat", 2.5,
            targets={"U_R": u_rest, "F_R": (0.42, 0.48, 1.28), "U_L": (-0.36, 0.46, 1.16)},
            oris=dict(ori_r), lerps={"F_L": dict(_BREATH)}),
        _ph("settle", 1.0),
    ]
    return S9Task(
        family="beaker_relay",
        spec=SkillSpec("s9_beaker_relay", phases),
        hand_events=R26._close_events({"F_L": g}, ("F_L",), t_off=0.10, phase="close")
        + [("release", 0.10, "F_L", "open", 0.0, 1.0)]
        + R26._close_events({"U_R": r}, ("U_R",), t_off=0.10, phase="transfer"),
        object_events=[("close", 0.90, "attach", "F_L", None, obj.name, 0.13),
                       ("release", 0.60, "release", None, None, obj.name, None),
                       ("transfer", 0.90, "attach", "U_R", (0.18, 0.6), obj.name, None)],
        success={"kind": "handover", "giver": "F_L", "receiver": "U_R", "follow_dist_m": 0.45, "min_travel_m": 0.10},
        objects=[obj.name],
        env_yaml=env_yaml,
    )


def _beaker_relay_variants(provider):
    objs = relay_objects()
    obj = objs["F_L"]
    standing = LabObject(obj.name, obj.catalog,
                         (RELAY_LANE_XY[0] + RELAY_DRAG_XY[0], RELAY_LANE_XY[1] + RELAY_DRAG_XY[1]),
                         None, 0.0, dict(obj.entry))
    # full cross product of the feasible giver x receiver candidates: the
    # index-paired combos of _screened_combos missed the only reachable pair
    feas = {}
    for arm, o in (("F_L", obj), ("U_R", standing)):
        feas[arm] = []
        for ap in R26.PHYS_APPROACHES:
            feas[arm] += [(ap, c) for c in R26._screen(provider, arm, _top_cands(provider, arm, o, ap), max_keep=2)]
        print(f"R26_SCREEN_SUMMARY beaker_relay {arm}: {[(a, c.name) for a, c in feas[arm]]}", flush=True)
    for apg, g in feas["F_L"]:
        for apr, r in feas["U_R"]:
            tag = f"F_L:{apg[:2]}{g.name[-3:]}+U_R:{apr[:2]}{r.name[-3:]}"
            yield tag, _relay_task(objs, {"F_L": g, "U_R": r}, R34_ENV["beaker_relay"])


# ---------------------------------------------------------------------------
# family 4: tube_racking (fine assembly: move 20 ml test tubes between the
# holes of the rack; hole d 2.6 cm vs tube d 1.9 cm -> 3.5 mm radial clearance)
# ---------------------------------------------------------------------------

# rack geometry measured on the mesh (tools/r34_rack_holes.py), rack-local
# frame before the 90 deg yaw: 2 rows (y = +/-0.0185) x 6 holes
RACK_HOLE_XS = (-0.0858, -0.0515, -0.0174, 0.0174, 0.0515, 0.0858)
RACK_HOLE_YS = (-0.0185, 0.0185)
RACK_PLATE_TOP = 0.0325          # top plate surface (local z)
RACK_FLOOR_TOP = -0.0294         # lower plate surface the tube bottom rests on
TUBE_LIFT = 0.12                 # tube fully clears the top plate
# open-loop carry lands the tube ~2.8 cm towards +x of the target (r34_tube3/5,
# same sign both runs): pre-compensate; the recorder's insertion assist then
# servos the residual from the simulated tube pose
TUBE_CARRY_OFFSET_XY = (-0.028, 0.0)
TUBE_RACK_XY = (0.53, 0.0)


# The F2 thumb-index pad aperture bottoms out at 2.5 cm (f = 0.7, beyond
# which the fingers cross), so the 1.9 cm 20 ml tube cannot be pinched at all
# (r34_tube2: closed 6 mm short of the tube). The family therefore uses the
# 50 ml tube (d 2.6 cm, 20 cm long) in a rack scaled x1.35 (holes 3.5 cm ->
# 4.5 mm radial clearance, 4.6 cm pitch, 29.6 cm long).
TUBE_RACK_SCALE = 1.35
TUBE_CATALOG = "glass_test_tube_50ml"


def _rack_hole_world(rack: LabObject, i: int, row: int) -> tuple:
    """world xy of hole (i, row) of a rack standing at TUBE_RACK_XY, yaw 90:
    local x (long axis) -> world +y, local y -> world -x."""
    lx, ly = RACK_HOLE_XS[i] * rack.scale, RACK_HOLE_YS[row] * rack.scale
    return (rack.xy[0] - ly, rack.xy[1] + lx)


def tube_objects() -> dict:
    rack = lab_object("rack", "test_tube_rack", TUBE_RACK_XY, None, yaw=RACK_YAW)
    rack.scale = TUBE_RACK_SCALE
    rack.mass_override = 2.0       # heavy rack: does not skate when a tube is pushed in
    tube_len = float(catalog()[TUBE_CATALOG]["size"][2])
    z_tube = (TABLE_TOP_Z + 0.5 * rack.size[2] + RACK_FLOOR_TOP * rack.scale
              + 0.5 * tube_len + 0.005)   # 5 mm drop-in, no initial overlap
    objs = {"rack": rack}
    # F_L works the -y half (holes 0..2), F_R the +y half (holes 3..5)
    for arm, name, i_from, i_to in (("F_L", "tube_l", 0, 1), ("F_R", "tube_r", 5, 4)):
        start = _rack_hole_world(rack, i_from, 0)
        target = _rack_hole_world(rack, i_to, 1)
        t = lab_object(name, TUBE_CATALOG, start, target)
        t.z_override = z_tube
        objs[arm] = t
    for arm, (n, k, xy, pxy) in RACK_LAYOUT_U.items():
        objs[arm] = lab_object(n, k, xy, pxy)
    return objs


def _tube_racking_task(objs: dict, cands: dict, env_yaml: str) -> S9Task:
    """Sequential fine insertion: F_L moves its tube while the U arms pick,
    then F_R moves its tube while the U arms place (two Franka hands over
    the 22 cm rack at once sit at self 7.7 mm: the parallel version never
    passed the approach floor)."""
    fl, fr, ul, ur = (cands[a] for a in ("F_L", "F_R", "U_L", "U_R"))
    ori_f = {"F_L": fl.R_flange, "F_R": fr.R_flange}
    ori_u = {"U_L": ul.R_flange, "U_R": ur.R_flange}
    fgl, fgr, ugl, ugr = fl.flange_grasp, fr.flange_grasp, ul.flange_grasp, ur.flange_grasp
    lift_u = 0.11
    tl, tr = objs["F_L"], objs["F_R"]
    # insertion pose = pick pose translated to the target hole (same height:
    # the tube bottom returns onto the lower plate), no overshoot
    ox, oy = TUBE_CARRY_OFFSET_XY
    ins_l = (fgl[0] + tl.place_xy[0] - tl.pos[0] + ox, fgl[1] + tl.place_xy[1] - tl.pos[1] + oy, fgl[2])
    ins_r = (fgr[0] + tr.place_xy[0] - tr.pos[0] + ox, fgr[1] + tr.place_xy[1] - tr.pos[1] + oy, fgr[2])
    place_ul = R26._place_rel(ugl, objs["U_L"].pos, objs["U_L"].place_xy, 0.015)
    place_ur = R26._place_rel(ugr, objs["U_R"].pos, objs["U_R"].place_xy, 0.015)
    home_l, home_r = (0.44, -0.30, 1.25), (0.44, 0.30, 1.25)
    ol, orr = {"F_L": fl.R_flange}, {"F_R": fr.R_flange}
    phases = [
        # ---- F_L's tube (U arms pick in parallel) ----
        _ph("approach", 3.0, targets={"F_L": fl.flange_pre, "U_L": ul.flange_pre, "U_R": ur.flange_pre},
            oris={**ol, **ori_u}),
        _ph("descend", 1.5, targets={"F_L": fgl, "U_L": ugl, "U_R": ugr}, oris={**ol, **ori_u}),
        _ph("close", 1.2),
        _ph("lift", 2.0, targets={"F_L": _above(fgl, TUBE_LIFT), "U_L": _above(ugl, lift_u), "U_R": _above(ugr, lift_u)},
            oris={**ol, **ori_u}),
        # slow carry: the 2.6 cm tube sits at the F2 closure limit and slides
        # in the pinch under lateral acceleration (r34_tube3: 4 cm short)
        _ph("carry", 4.0, targets={"F_L": _above(ins_l, TUBE_LIFT), "U_L": (-0.10, 0.26, 1.12), "U_R": (-0.12, -0.26, 1.12)},
            oris=dict(ol)),
        _ph("align", 2.0, targets={"U_L": _above(place_ul, 0.10), "U_R": _above(place_ur, 0.10)}, oris=dict(ori_u)),
        # insert: the recorder's insertion assist servos the holding arm onto
        # the hole from the object's simulated pose during this phase
        _ph("insert", 3.5, targets={"F_L": ins_l}, oris=dict(ol)),
        _ph("release_l", 2.0),
        _ph("clear_l", 1.5, targets={"F_L": _above(ins_l, TUBE_LIFT)}, oris=dict(ol)),
        # ---- F_R's tube (U arms place in parallel) ----
        _ph("approach_r", 3.0, targets={"F_R": fr.flange_pre, "F_L": home_l}, oris=dict(orr)),
        _ph("descend_r", 1.5, targets={"F_R": fgr, "U_L": place_ul, "U_R": place_ur}, oris={**orr, **ori_u}),
        _ph("close_r", 1.2),
        _ph("lift_r", 2.0, targets={"F_R": _above(fgr, TUBE_LIFT)}, oris=dict(orr)),
        _ph("carry_r", 4.0, targets={"F_R": _above(ins_r, TUBE_LIFT)}, oris=dict(orr)),
        _ph("align_r", 2.0),
        _ph("insert_r", 3.5, targets={"F_R": ins_r}, oris=dict(orr)),
        _ph("release_r", 2.0),
        _ph("clear_r", 1.5, targets={"F_R": _above(ins_r, TUBE_LIFT)}, oris=dict(orr)),
        _ph("retreat", 2.5, targets={"F_R": home_r, "U_L": (-0.38, 0.42, 1.16), "U_R": (-0.40, -0.36, 1.16)}),
    ]
    names = {a: objs[a].name for a in ARM_KEYS}
    return S9Task(
        family="tube_racking",
        spec=SkillSpec("s9_tube_racking", phases),
        hand_events=R26._close_events(cands, ("F_L", "U_L", "U_R"))
        + R26._close_events(cands, ("F_R",), phase="close_r")
        + [("release_l", 0.60, "F_L", "open", 0.0, 1.0)]
        + [("release_r", 0.60, "F_R", "open", 0.0, 1.0)]
        + [("release_r", 0.60, a, "open", 0.0) for a in ("U_L", "U_R")],
        object_events=[("close", 0.90, "attach", a, None, names[a], 0.24) for a in ("F_L", "U_L", "U_R")]
        + [("close_r", 0.90, "attach", "F_R", None, names["F_R"], 0.24)]
        + [("release_l", 0.60, "release", None, None, names["F_L"], None)]
        + [("release_r", 0.60, "release", None, None, names[a], None) for a in ("F_R", "U_L", "U_R")],
        success={"kind": "pick_place", "place_xy": list(tl.place_xy),
                 "radius_m": 0.02, "min_disp_m": 0.05, "rest_z": tl.pos[2]},
        objects=[names["F_L"], names["F_R"], objs["rack"].name, names["U_L"], names["U_R"]],
        env_yaml=env_yaml,
    )


def _tube_racking_variants(provider):
    objs = tube_objects()
    gen = {a: (lambda ap, a=a: _top_cands(provider, a, objs[a], ap)) for a in ARM_KEYS}
    for tag, cands in R26._screened_combos(provider, "tube_racking", gen):
        yield tag, _tube_racking_task(objs, cands, R34_ENV["tube_racking"])


# ---------------------------------------------------------------------------
# family 5: bin_packing (collaborative packaging: the F arms load two reagent
# bottles into a KLT bin, then clamp the loaded bin between their palms and
# carry it; the U arms pick their beaker / crucible in parallel)
# ---------------------------------------------------------------------------

BIN_XY = (0.53, 0.0)
BIN_SCALE_Z = 0.6              # shallow crate (8.8 cm): the pads stay above the rim when a bottle stands on the floor
BIN_SCALE_Y = 0.75             # 22 cm long like the rack: the F_L palm folds the wrist (self 1.9 mm) at the 29.7 cm end face
BIN_SLOT_Y = 0.05              # bottle slots inside the bin (+/-y)
BIN_PALM_PRESS = 0.040         # loaded bin ~1.1 kg: 1.5 / 2.5 cm slipped under gating (the damper attenuates the
                               # inward press while an intra-arm self row sits at ~1 cm); the arms stop at contact anyway
BIN_F_CURL = 0.35
BIN_DX = 0.14                  # carry towards the F bases (R26 box lesson)
BIN_LAYOUT_F = {
    "F_L": ("bottle_brown", "brown_reagent_bottle_large", (0.50, -0.36)),
    "F_R": ("bottle_clear", "clear_reagent_bottle_large", (0.50, 0.36)),
}


def bin_objects() -> dict:
    kl = lab_object("bin", "small_klt", BIN_XY, (BIN_XY[0] + BIN_DX, BIN_XY[1]))
    kl.scale_xyz = (1.0, BIN_SCALE_Y, BIN_SCALE_Z)
    objs = {"bin": kl}
    for arm, (n, k, xy) in BIN_LAYOUT_F.items():
        objs[arm] = lab_object(n, k, xy, (BIN_XY[0], BIN_SLOT_Y if arm == "F_R" else -BIN_SLOT_Y))
    for arm, (n, k, xy, pxy) in RACK_LAYOUT_U.items():
        objs[arm] = lab_object(n, k, xy, pxy)
    return objs


def _bin_packing_task(objs: dict, cands: dict, env_yaml: str) -> S9Task:
    kl = objs["bin"]
    fl, fr, ul, ur = (cands[a] for a in ("F_L", "F_R", "U_L", "U_R"))
    pl_c, pr_c = cands["F_L_palm"], cands["F_R_palm"]
    ori_f = {"F_L": fl.R_flange, "F_R": fr.R_flange}
    ori_p = {"F_L": pl_c.R_flange, "F_R": pr_c.R_flange}
    ori_u = {"U_L": ul.R_flange, "U_R": ur.R_flange}
    fgl, fgr, ugl, ugr = fl.flange_grasp, fr.flange_grasp, ul.flange_grasp, ur.flange_grasp
    lift_f, lift_u = 0.15, 0.11
    # release the bottle ~5 mm above the bin floor (floor thickness ~1.2 cm x scale)
    floor_dz = 0.012 * BIN_SCALE_Z + 0.005
    ol, orr_ = objs["F_L"], objs["F_R"]
    pl_l = (fgl[0] + ol.place_xy[0] - ol.pos[0], fgl[1] + ol.place_xy[1] - ol.pos[1], fgl[2] + floor_dz)
    pl_r = (fgr[0] + orr_.place_xy[0] - orr_.pos[0], fgr[1] + orr_.place_xy[1] - orr_.pos[1], fgr[2] + floor_dz)
    place_ul = R26._place_rel(ugl, objs["U_L"].pos, objs["U_L"].place_xy, 0.015)
    place_ur = R26._place_rel(ugr, objs["U_R"].pos, objs["U_R"].place_xy, 0.015)
    bgl, bgr = pl_c.flange_grasp, pr_c.flange_grasp
    bgl_c = (bgl[0] + BIN_DX, bgl[1], bgl[2])
    bgr_c = (bgr[0] + BIN_DX, bgr[1], bgr[2])
    out_l = (bgl_c[0], bgl_c[1] - 0.05, bgl_c[2])
    out_r = (bgr_c[0], bgr_c[1] + 0.05, bgr_c[2])
    home_l, home_r = (0.44, -0.30, 1.25), (0.44, 0.30, 1.25)
    ol_, or_ = {"F_L": fl.R_flange}, {"F_R": fr.R_flange}
    phases = [
        # ---- F_L loads its bottle (U arms pick in parallel); two bottles
        # 10 cm apart cannot be loaded at once: self_F 6.6 mm ----
        _ph("approach", 3.0, targets={"F_L": fl.flange_pre, "U_L": ul.flange_pre, "U_R": ur.flange_pre},
            oris={**ol_, **ori_u}),
        _ph("descend", 1.5, targets={"F_L": fgl, "U_L": ugl, "U_R": ugr}, oris={**ol_, **ori_u}),
        _ph("close", 1.2),
        _ph("lift", 1.5, targets={"F_L": _above(fgl, lift_f), "U_L": _above(ugl, lift_u), "U_R": _above(ugr, lift_u)},
            oris={**ol_, **ori_u}),
        _ph("carry", 3.0, targets={"F_L": _above(pl_l, lift_f), "U_L": (-0.10, 0.26, 1.12), "U_R": (-0.12, -0.26, 1.12)},
            oris=dict(ol_)),
        _ph("carry_u", 1.5, targets={"U_L": _above(place_ul, 0.10), "U_R": _above(place_ur, 0.10)}, oris=dict(ori_u),
            lerps={"F_L": dict(_BREATH)}),
        _ph("place", 2.5, targets={"F_L": pl_l, "U_L": place_ul, "U_R": place_ur}, oris={**ol_, **ori_u}),
        _ph("release", 2.0),
        _ph("clear", 1.5, targets={"F_L": _above(pl_l, 0.16)}, oris=dict(ol_)),
        # ---- F_R loads its bottle ----
        _ph("approach_r", 3.0, targets={"F_R": fr.flange_pre, "F_L": home_l}, oris=dict(or_)),
        _ph("descend_r", 1.5, targets={"F_R": fgr}, oris=dict(or_)),
        _ph("close_r", 1.2),
        _ph("lift_r", 1.5, targets={"F_R": _above(fgr, lift_f)}, oris=dict(or_)),
        _ph("carry_r", 3.0, targets={"F_R": _above(pl_r, lift_f)}, oris=dict(or_)),
        _ph("place_r", 2.5, targets={"F_R": pl_r}, oris=dict(or_)),
        _ph("release_r", 2.0),
        _ph("clear_r", 1.5, targets={"F_R": _above(pl_r, 0.16)}, oris=dict(or_)),
        # ---- clamp and carry the loaded bin (both arms regroup at home
        # first: the palm poses are reachable from birth-like configurations,
        # not from the post-place pinch poses -- wrist folds, self 1.8 mm) ----
        _ph("regroup", 2.5, targets={"F_R": home_r}),
        # 5 s: the palm pre-pose is a large wrist reconfiguration (per-step
        # joint delta 0.035 rad at 3 s > 0.03 hard cap)
        _ph("approach_bin", 5.0, targets={"F_L": pl_c.flange_pre, "F_R": pr_c.flange_pre}, oris=dict(ori_p)),
        _ph("press", 1.5, targets={"F_L": bgl, "F_R": bgr}, oris=dict(ori_p)),
        _ph("clamp", 1.2),
        _ph("lift_bin", 1.5, targets={"F_L": _above(bgl, R26.BOX_PHYS_LIFT), "F_R": _above(bgr, R26.BOX_PHYS_LIFT)},
            oris=dict(ori_p)),
        _ph("carry_bin", 3.0, targets={"F_L": _above(bgl_c, R26.BOX_PHYS_LIFT), "F_R": _above(bgr_c, R26.BOX_PHYS_LIFT)},
            oris=dict(ori_p)),
        _ph("place_bin", 2.5, targets={"F_L": bgl_c, "F_R": bgr_c}, oris=dict(ori_p)),
        _ph("release_bin", 1.0),
        _ph("unclamp", 1.5, targets={"F_L": out_l, "F_R": out_r}, oris=dict(ori_p)),
        _ph("clear_up", 1.5, targets={"F_L": _above(out_l, 0.15), "F_R": _above(out_r, 0.15)}, oris=dict(ori_p)),
        _ph("retreat", 2.5,
            targets={"F_L": home_l, "F_R": home_r, "U_L": (-0.38, 0.42, 1.16), "U_R": (-0.40, -0.36, 1.16)}),
    ]
    names = {a: objs[a].name for a in ARM_KEYS}
    return S9Task(
        family="bin_packing",
        spec=SkillSpec("s9_bin_packing", phases),
        hand_events=R26._close_events(cands, ("F_L", "U_L", "U_R"))
        + R26._close_events(cands, ("F_R",), phase="close_r")
        + [("release", 0.75, a, "open", 0.0) for a in ("U_L", "U_R")]
        + [("release", 0.60, "F_L", "open", 0.0, 1.0), ("release_r", 0.60, "F_R", "open", 0.0, 1.0)]
        + [("clamp", 0.10, a, "close", BIN_F_CURL, 0.6, 0.0) for a in ("F_L", "F_R")]
        + [("release_bin", 0.60, a, "open", 0.0) for a in ("F_L", "F_R")],
        object_events=[("close", 0.90, "attach", a, None, names[a], 0.24) for a in ("F_L", "U_L", "U_R")]
        + [("close_r", 0.90, "attach", "F_R", None, names["F_R"], 0.24)]
        + [("release", 0.60, "release", None, None, names[a], None) for a in ("F_L", "U_L", "U_R")]
        + [("release_r", 0.60, "release", None, None, names["F_R"], None)]
        + [("clamp", 0.90, "attach", "F_L", None, kl.name, 0.30),
           ("release_bin", 0.60, "release", None, None, kl.name, None)],
        # success = the loaded bin carried to its zone (bottles ride inside)
        success={"kind": "pick_place", "place_xy": list(kl.place_xy),
                 "radius_m": 0.10, "min_disp_m": 0.10, "rest_z": kl.pos[2]},
        objects=[kl.name, names["F_L"], names["F_R"], names["U_L"], names["U_R"]],
        env_yaml=env_yaml,
    )


def _bin_packing_variants(provider):
    objs = bin_objects()
    kl = objs["bin"]
    half_len = 0.5 * kl.size[1]
    z_palm = max(kl.pos[2] + 0.5 * kl.size[2] + 0.030, TABLE_TOP_Z + 0.130)   # fingers span the whole end wall
    face_l = (BIN_XY[0], BIN_XY[1] - half_len, z_palm)
    face_r = (BIN_XY[0], BIN_XY[1] + half_len, z_palm)
    palm_l = R26._palm_cand("F_L", "palm_fz-", face_l, (0.0, 1.0, 0.0), (0.0, 0.0, -1.0), R26.BOX_PALM_PRE, BIN_PALM_PRESS)
    palm_r = R26._palm_cand("F_R", "palm_fz-", face_r, (0.0, -1.0, 0.0), (0.0, 0.0, -1.0), R26.BOX_PALM_PRE, BIN_PALM_PRESS)
    if not (R26._screen(provider, "F_L", [palm_l]) and R26._screen(provider, "F_R", [palm_r])):
        print("R34_BIN_PALM_INFEASIBLE", flush=True)
        return
    gen = {a: (lambda ap, a=a: _top_cands(provider, a, objs[a], ap)) for a in ARM_KEYS}
    for tag, cands in R26._screened_combos(provider, "bin_packing", gen):
        yield tag, _bin_packing_task(objs, {**cands, "F_L_palm": palm_l, "F_R_palm": palm_r}, R34_ENV["bin_packing"])


_VARIANTS = {"reagent_pick": _reagent_pick_variants, "rack_colift": _rack_colift_variants,
             "beaker_relay": _beaker_relay_variants, "tube_racking": _tube_racking_variants,
             "bin_packing": _bin_packing_variants}
_OBJECTS = {"reagent_pick": reagent_objects, "rack_colift": rack_objects, "beaker_relay": relay_objects,
            "tube_racking": tube_objects, "bin_packing": bin_objects}


# ---------------------------------------------------------------------------
# build (same hard gates as R24/R26)
# ---------------------------------------------------------------------------


def build_task_r34(provider, family: str, out_dir, verbose: bool = True) -> dict:
    from safeduo.delta.task_record_s9 import _compose_and_validate, resolve_events

    last_err = None
    for tag, task in _VARIANTS[family](provider):
        try:
            traj, rep = _compose_and_validate(provider, task, None)
        except RuntimeError as e:
            last_err = f"{tag}: {e}"
            print(f"R34_VARIANT_REJECTED {family} {tag}: {e}", flush=True)
            continue
        if "FAIL" in rep:
            last_err = f"{tag}: {rep['FAIL']}"
            print(f"R34_VARIANT_REJECTED {family} {tag}: {rep['FAIL']}", flush=True)
            continue
        rep["participation"] = audit_arm_participation(provider, traj)
        if not rep["participation"]["all_arms_moving"]:
            rep["FAIL"] = f"idle arm: {rep['participation']}"
            last_err = f"{tag}: {rep['FAIL']}"
            print(f"R34_VARIANT_REJECTED {family} {tag}: {rep['FAIL']}", flush=True)
            continue
        break
    else:
        raise RuntimeError(f"{family}: no R34 variant passes the hard gates, last error: {last_err}")

    phases, hand, objev, success = resolve_events(task)
    assert phases[-1]["t1"] <= traj.duration + 1e-6
    objs = {o.name: o for o in _OBJECTS[family]().values()}
    traj.meta["s9_task"] = {
        "family": task.family,
        "object": objs[task.objects[0]].meta(),
        "objects": [objs[n].meta() for n in task.objects],
        "env_yaml": task.env_yaml,
        "phases": phases, "hand_events": hand, "object_events": objev,
        "variant": tag, "success": success,
    }
    if family == "tube_racking":
        # recorder insertion assist (fine assembly): servo the holding arm so
        # the tube axis meets the target hole during the insert phases
        by_arm = _OBJECTS[family]()
        traj.meta["s9_task"]["insert_assist"] = [
            {"object": by_arm[a].name, "arm": a, "target_xy": list(by_arm[a].place_xy), "phase": ph,
             # servo the tube's BOTTOM (hangs a few degrees off vertical in the pinch)
             "half_length": 0.5 * by_arm[a].size[2]}
            for a, phases in (("F_L", ("align", "insert")), ("F_R", ("align_r", "insert_r")))
            for ph in phases
        ]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj.save(out_dir / f"task_{task.family}.npz")
    (out_dir / f"task_{task.family}_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    part = rep["participation"]
    print(f"{task.family}: variant={tag} steps={traj.n_steps} dur={traj.duration:.1f}s "
          f"gate={rep['hard_gate_margin_gt0']} min_margin={rep['min_margin']} "
          f"travel={ {a: part[a]['ee_travel_m'] for a in ARM_KEYS} }", flush=True)
    return rep


def write_env_yaml(family: str, out_path: "str | None" = None) -> str:
    """Emit the recording env yaml for a family: the R27 physical-grasp env
    knobs (extends the bottle r27r16 env) with the family's catalog objects."""
    import yaml

    from safeduo.configs import CONFIG_DIR

    objs = {o.name: o for o in _OBJECTS[family]().values()}      # arms may share an object
    doc = {
        "extends": "duo_env_v7_r26_bottle_r27r16.yaml",
        "assets": {
            "object_catalog": CATALOG_LAB,
            "table_objects": [o.yaml_entry() for o in objs.values()],
        },
    }
    out_path = out_path or os.path.join(str(CONFIG_DIR), R34_ENV[family])
    with open(out_path, "w") as f:
        f.write(f"# R34 (auto-generated by task_library_r34.write_env_yaml): {family} recording env\n")
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
    return out_path


def main(argv: "list | None" = None) -> int:
    import argparse
    import time

    from safeduo.delta.task_record_s9 import make_s9_provider

    ap = argparse.ArgumentParser(description="R34 lab task families on catalog mesh assets")
    ap.add_argument("stage", choices=["build", "env"])
    ap.add_argument("--task", default="reagent_pick", choices=list(_VARIANTS) + ["all"])
    ap.add_argument("--out", default="artifacts/task_trajs_r34")
    ap.add_argument("--hand-calib", default="v7")
    ap.add_argument("--spheres", default="r16", choices=["v7", "r16"])
    ap.add_argument("--struct-exempt", action="store_true")
    ap.add_argument("--multiseed", action="store_true")
    args = ap.parse_args(argv)
    if args.stage == "env":
        print(write_env_yaml(args.task))
        return 0
    if args.struct_exempt:
        from safeduo.delta.skill_record import DESIGN_STRUCT_EXEMPT
        DESIGN_STRUCT_EXEMPT["on"] = True
    if args.multiseed:
        from safeduo.delta.skill_record_v7 import IK_MULTISEED
        IK_MULTISEED["on"] = True
        IK_MULTISEED["verbose"] = True
    R26.PHYS_GRASP["on"] = True
    R26.PHYS_GRASP["calib"] = args.hand_calib
    provider = make_s9_provider(1, device="cpu", spheres=args.spheres)
    rc = 0
    for fam in (list(_VARIANTS) if args.task == "all" else [args.task]):
        t0 = time.time()
        try:
            rep = build_task_r34(provider, fam, Path(args.out) / fam)
        except RuntimeError as e:
            print(f"R34_BUILD_FAILED {fam}: {e}", flush=True)
            rc = 1
            continue
        print(f"R34_BUILD_OK {fam} in {time.time() - t0:.1f}s gate={rep.get('hard_gate_margin_gt0')}", flush=True)
        write_env_yaml(fam)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
