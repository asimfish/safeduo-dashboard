"""Declarative task specs -- what the planner (LLM) writes, what compile.py
turns into an S9Task programme.

A spec describes WHAT happens (objects, who grasps what, where it goes, which
arms move together); the compiler supplies HOW (the validated phase skeleton,
grasp candidates, object-relative place targets, hand/object events) and the
existing IK hard gates decide whether it is feasible.

YAML shape (SKILL_LIBRARY_PLAN_20260908.md section 2):

    family: mix_pick_center          # unique; also the env yaml / npz name
    category: 3                      # 1-5, see skills.taxonomy
    env_base: factory                # factory (R35 dressing) | lab (R34 plain)
    objects:
      - {name: can_a, catalog: soup_can, xy: [0.50, -0.36], place_xy: [0.32, -0.30]}
      - {name: bin_a, catalog: sorting_bin_black, xy: [0.32, -0.30], yaw: 90, passive: true}
    arms:
      F_L: {do: pick_place, obj: can_a, grasp: top}
      F_R: {do: pick_place, obj: bottle_b, grasp: top}
      U_L: {do: pick_place, obj: cube_c, grasp: top}
      U_R: {do: idle}
    groups:                          # optional two-arm co-lift groups
      - {arms: [F_L, F_R], obj: profile, do: colift, half_station: 0.31, carry_dxy: [-0.16, 0.0]}
    success: {object: can_a, radius_m: 0.08, min_disp_m: 0.10}
    timing: {carry: 3.0}             # overrides of the default phase durations
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

ARMS = ("F_L", "F_R", "U_L", "U_R")
ROLES = ("pick_place", "colift", "idle")
GRASPS = ("top", "station")
ENV_BASES = ("factory", "lab")

DEFAULT_TIMING = {
    "approach": 3.0, "descend": 1.5, "close": 1.2, "lift": 1.5, "carry": 3.0,
    "carry_u": 1.5, "place": 2.5, "hold": 1.0, "release": 2.5, "settle": 1.0,
    "clear": 1.5, "clear_axial": 1.5, "retreat": 2.5,
}


@dataclass
class ObjectSpec:
    name: str
    catalog: str
    xy: tuple
    place_xy: "tuple | None" = None
    yaw: float = 0.0
    passive: bool = False           # bins / totes: listed in the task, never grasped
    scale_xyz: "tuple | None" = None
    grasp_depth: "float | None" = None   # override of the catalog depth_from_top (m)


@dataclass
class ArmSpec:
    arm: str
    do: str = "idle"                # pick_place | colift | idle
    obj: "str | None" = None
    grasp: str = "top"              # top | station
    lift: "float | None" = None     # lift height (m); default by arm family
    place_dz: float = 0.0           # extra release height (bin floor etc.)
    approach: "str | None" = None   # finger | flange_z: restrict the candidate approach mode


@dataclass
class GroupSpec:
    arms: tuple
    obj: str
    do: str = "colift"
    half_station: float = 0.30      # pinch stations at y = obj.y -/+ half_station along the long axis
    carry_dxy: tuple = (-0.16, 0.0)
    lift: float = 0.10
    squeeze_extra: "float | None" = None
    grasp_depth: "float | None" = None
    approach: "str | None" = None   # finger | flange_z for both hands of the group
    # R35 profile_colift fixture: two stop rails forming a 5 cm channel along the object's
    # long axis at the place -- the opening pads tip a standing 4 x 8 cm profile in about
    # half of the renders (colift3/6/11 vs 4/8/9); in the channel it can lean <= 1 cm.
    jig: bool = False
    # After release/settle, move each hand outward along the beam at place
    # height before lifting. Zero preserves the existing phase sequence.
    clear_axial_m: float = 0.0
    # Flange z offset applied to place/hold only (m, negative = lower). GRASP_TELE of the
    # 2026-09-11/12 two_pair_profiles runs: the pinch lifts the standing profile 0.9 cm while
    # closing and it slides a further 0.6 cm down the pads during lift/carry, so with the
    # place at the grasp height the profile still hangs 1.5 cm above the table when the
    # pads open and topples (z 0.855 -> 0.820). Default 0 keeps the colift3 no-overshoot rule.
    place_dz: float = 0.0



@dataclass
class TaskSpec:
    family: str
    category: int
    objects: list
    arms: dict                      # arm -> ArmSpec
    groups: list = field(default_factory=list)
    env_base: str = "factory"
    success: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    u_lane: dict = field(default_factory=lambda: {"U_L": (-0.10, 0.26, 1.12), "U_R": (-0.12, -0.26, 1.12)})
    homes: dict = field(default_factory=lambda: {"F_L": (0.44, -0.30, 1.25), "F_R": (0.44, 0.30, 1.25),
                                                 "U_L": (-0.38, 0.42, 1.16), "U_R": (-0.40, -0.36, 1.16)})
    source_path: "str | None" = None

    # ---- derived -------------------------------------------------------------
    def object(self, name: str) -> ObjectSpec:
        for o in self.objects:
            if o.name == name:
                return o
        raise KeyError(f"{self.family}: object {name!r} not in spec")

    def duration(self, phase: str) -> float:
        return float(self.timing.get(phase, DEFAULT_TIMING[phase]))

    def group_of(self, arm: str) -> "GroupSpec | None":
        for g in self.groups:
            if arm in g.arms:
                return g
        return None

    def moving_arms(self) -> list:
        return [a for a in ARMS if self.arms[a].do != "idle" or self.group_of(a) is not None]

    def validate(self) -> None:
        assert self.category in (1, 2, 3, 4, 5), self.category
        assert self.env_base in ENV_BASES, self.env_base
        names = [o.name for o in self.objects]
        assert len(names) == len(set(names)), f"{self.family}: duplicate object names"
        for a in ARMS:
            assert a in self.arms, f"{self.family}: arm {a} missing (use do: idle)"
            s = self.arms[a]
            assert s.do in ROLES, (a, s.do)
            assert s.grasp in GRASPS, (a, s.grasp)
            assert s.approach in (None, "finger", "flange_z"), (a, s.approach)
            if s.do == "pick_place":
                o = self.object(s.obj)
                assert not o.passive, f"{a} grasps passive object {o.name}"
                assert o.place_xy is not None, f"{o.name} needs place_xy for pick_place"
        for g in self.groups:
            assert len(g.arms) == 2, "co-lift groups are pairs"
            if not math.isfinite(g.clear_axial_m) or g.clear_axial_m < 0:
                raise ValueError("clear_axial_m must be finite and nonnegative")
            for a in g.arms:
                assert self.arms[a].do in ("idle", "colift"), f"{a} is both in a group and {self.arms[a].do}"
            o = self.object(g.obj)
            assert o.place_xy is None or True
        assert self.moving_arms(), f"{self.family}: nobody moves"
        prim = self.success.get("object")
        if prim is not None:
            self.object(prim)


def _tup(v, n=None):
    t = tuple(float(x) for x in v)
    if n is not None:
        assert len(t) == n, (v, n)
    return t


def from_dict(d: dict, source_path: "str | None" = None) -> TaskSpec:
    objs = []
    for o in d["objects"]:
        objs.append(ObjectSpec(
            name=o["name"], catalog=o["catalog"], xy=_tup(o["xy"], 2),
            place_xy=_tup(o["place_xy"], 2) if o.get("place_xy") is not None else None,
            yaw=float(o.get("yaw", 0.0)), passive=bool(o.get("passive", False)),
            scale_xyz=_tup(o["scale_xyz"], 3) if o.get("scale_xyz") else None,
            grasp_depth=(float(o["grasp_depth"]) if o.get("grasp_depth") is not None else None)))
    arms = {}
    for a in ARMS:
        s = dict(d.get("arms", {}).get(a) or {"do": "idle"})
        arms[a] = ArmSpec(arm=a, do=s.get("do", "idle"), obj=s.get("obj"), grasp=s.get("grasp", "top"),
                          lift=(float(s["lift"]) if s.get("lift") is not None else None),
                          place_dz=float(s.get("place_dz", 0.0)), approach=s.get("approach"))
    groups = []
    for g in d.get("groups", []) or []:
        groups.append(GroupSpec(arms=tuple(g["arms"]), obj=g["obj"], do=g.get("do", "colift"),
                                half_station=float(g.get("half_station", 0.30)),
                                carry_dxy=_tup(g.get("carry_dxy", (-0.16, 0.0)), 2),
                                lift=float(g.get("lift", 0.10)),
                                squeeze_extra=(float(g["squeeze_extra"]) if g.get("squeeze_extra") is not None else None),
                                grasp_depth=(float(g["grasp_depth"]) if g.get("grasp_depth") is not None else None),
                                approach=g.get("approach"), jig=bool(g.get("jig", False)),
                                place_dz=float(g.get("place_dz", 0.0)),
                                clear_axial_m=float(g.get("clear_axial_m", 0.0))))
    spec = TaskSpec(family=d["family"], category=int(d["category"]), objects=objs, arms=arms, groups=groups,
                    env_base=d.get("env_base", "factory"), success=dict(d.get("success") or {}),
                    timing=dict(d.get("timing") or {}), source_path=source_path)
    if d.get("u_lane"):
        spec.u_lane = {k: _tup(v, 3) for k, v in d["u_lane"].items()}
    if d.get("homes"):
        spec.homes = {k: _tup(v, 3) for k, v in d["homes"].items()}
    spec.validate()
    return spec


def load(path) -> TaskSpec:
    import yaml

    p = Path(path)
    with open(p, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return from_dict(d, source_path=str(p))
