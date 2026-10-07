"""TaskSpec -> S9Task compiler + registration into the R34 build machinery.

The compiler owns the HOW that every hand-written family (R24-R35) repeated:
the synchronous phase skeleton (approach / descend / close / lift / carry /
carry_u / place / hold / release / settle / clear / retreat), grasp candidates
from the catalog hints, object-relative place targets (R26._place_rel -- the
absolute-flange form misplaced U objects by one pinch offset, 09-08), the
co-lift group discipline of profile_colift (both pinches translate by one
displacement, slow symmetric opening, a hold before the release so a gated
lagging hand can catch up) and the recorder's hand / object events.

Nothing below bypasses the hard gates: `R34.build_task_r34` still runs the
IK floors, the trajectory audit and the four-arm participation audit on the
compiled task.  Arms the spec leaves idle would fail that audit, so the
`patrol` role gives them an object-free excursion (near-but-no-conflict
traffic, exactly the category-1 sample the filter must not brake on).
"""
from __future__ import annotations

import math

from safeduo.delta import task_library_r26 as R26
from safeduo.delta import task_library_r34 as R34
from safeduo.delta import task_library_r35 as R35
from safeduo.delta.grasp_gen import box_antipodal_grasps
from safeduo.delta.skill_record import SkillSpec
from safeduo.delta.task_library_r24 import U_GRASP_PARAMS, _BREATH, _above, _arm_prior, _ph, grip_station_cands
from safeduo.delta.task_library_r34 import LabObject, lab_object
from safeduo.delta.task_record_s9 import GRASP_PARAMS, S9Task
from safeduo.skills.spec import ARMS, TaskSpec

LIFT_DEFAULT = {"F": 0.15, "U": 0.11}
PLACE_LEGACY_DZ = {"F": -0.005, "U": 0.015}
PATROL_OFFSET = (-0.20, 0.0, -0.06)     # object-free excursion from home (>= 0.15 m EE travel for the audit)
GATE_TCP_OBJECT = 0.24
GATE_TCP_LONG = 0.40                     # two-hand long object: TCP-to-centre distance is ~half the span


# --------------------------------------------------------------------------- objects
def objects_for(spec: TaskSpec) -> dict:
    """arm/role-keyed LabObjects (the R34 machinery dedups by .name)."""
    labs = {}
    for o in spec.objects:
        lo = lab_object(o.name, o.catalog, o.xy, o.place_xy, yaw=o.yaw)
        if o.scale_xyz is not None:
            lo.scale_xyz = tuple(o.scale_xyz)
        if o.grasp_depth is not None:
            lo.entry = {**lo.entry, "grasp": {**lo.grasp, "depth_from_top": float(o.grasp_depth)}}
        labs[o.name] = lo
    out = {}
    for a in ARMS:
        s = spec.arms[a]
        if s.do == "pick_place":
            out[a] = labs[s.obj]
    for g in spec.groups:
        lo = labs[g.obj]
        if lo.place_xy is None:
            lo = LabObject(lo.name, lo.catalog, lo.xy,
                           (lo.xy[0] + g.carry_dxy[0], lo.xy[1] + g.carry_dxy[1]), lo.yaw, dict(lo.entry),
                           lo.z_override, lo.mass_override, lo.scale, lo.scale_xyz)
            labs[g.obj] = lo
        for a in g.arms:
            out[a] = lo
    for o in spec.objects:
        key = f"passive:{o.name}" if o.passive else f"obj:{o.name}"
        if labs[o.name] not in out.values():
            out[key] = labs[o.name]
    return out


# --------------------------------------------------------------------------- candidates
def _long_axis(obj: LabObject) -> int:
    """0 = the long axis lies along world x, 1 = along y.  Long assets are
    authored with their length along y; a yaw of +/-90 deg turns them along x."""
    size = obj.size
    assert size[1] > size[0], f"{obj.name}: station grasps need a long object (size {size})"
    return 0 if abs(abs(float(obj.yaw)) - 90.0) < 1e-6 else 1


def _station_cands(provider, arm: str, obj: LabObject, approach: str, group) -> list:
    """Two-hand pinch stations on a long object: a virtual box the size of the
    cross-section at +/- half_station along the long axis, top pinches whose
    closing line crosses the short side (generalises R24.grip_station_cands,
    which only knows y-long objects).  y-long: each hand takes the station on
    its own base side; x-long: the group order decides."""
    size = obj.size
    ax = _long_axis(obj)
    _, base_xy = _arm_prior(provider, arm)
    if ax == 1:
        side = 1.0 if base_xy[1] >= 0 else -1.0
        st = (obj.pos[0], obj.pos[1] + side * group.half_station, obj.pos[2])
    else:
        side = -1.0 if arm == group.arms[0] else 1.0
        st = (obj.pos[0] + side * group.half_station, obj.pos[1], obj.pos[2])
    depth = group.grasp_depth if group.grasp_depth is not None else R26.ROD_PHYS_DEPTH
    if arm.startswith("F"):
        prm = R26._params_for(arm, size[0], GRASP_PARAMS, approach, size[2], grasp_depth=depth,
                              squeeze_extra=(group.squeeze_extra if group.squeeze_extra is not None else 0.06))
    else:
        prm = R26._params_for(arm, size[0], U_GRASP_PARAMS, approach, size[2], grasp_depth=depth,
                              squeeze_extra=(group.squeeze_extra if group.squeeze_extra is not None
                                             else R26.DFX_CUBE_SQUEEZE),
                              thumb_extra=R26.DFX_CUBE_THUMB_EXTRA)
    if ax == 1:
        return grip_station_cands(provider, arm, st, size[0], prm)
    ref_R, base_xy = _arm_prior(provider, arm)
    cands = box_antipodal_grasps(st, size[0], prm, ref_R=ref_R, base_xy=base_xy)
    # the closing line must cross the short side: along world y for an x-long object
    return [c for c in cands if c.kind == "top" and abs(c.closing[1]) > 0.9]


def variants(spec: TaskSpec, provider):
    objs = objects_for(spec)
    gen = {}
    # R26._screened_combos pairs the k-th feasible candidate of every arm, so a
    # validated mixed choice (F flange_z + U finger, profile_colift) is never tried
    # unless the spec pins the approach per arm / group
    for a in ARMS:
        s = spec.arms[a]
        g = spec.group_of(a)
        if g is not None:
            pref = g.approach
            gen[a] = (lambda ap, a=a, g=g, pref=pref:
                      [] if (pref and ap != pref) else _station_cands(provider, a, objs[a], ap, g))
        elif s.do == "pick_place":
            if s.grasp == "station":
                raise ValueError(f"{a}: station grasps are for co-lift groups; use grasp: top for pick_place")
            pref = s.approach
            gen[a] = (lambda ap, a=a, pref=pref:
                      [] if (pref and ap != pref) else R34._top_cands(provider, a, objs[a], ap))
    for tag, cands in R26._screened_combos(provider, spec.family, gen):
        yield tag, compile_task(spec, objs, cands)


# --------------------------------------------------------------------------- programme
def compile_task(spec: TaskSpec, objs: dict, cands: dict) -> S9Task:
    T = spec.duration
    tg = {k: {} for k in ("approach", "descend", "lift", "carry", "carry_u", "place", "clear", "retreat")}
    oris = {k: {} for k in tg}
    lerps_carry_u = {}
    axial_targets, axial_oris = {}, {}
    hand_events, object_events = [], []
    grasp_arms, group_arms = [], []
    attached = set()
    primary = None

    # ---- independent pick & place arms
    for a in ARMS:
        s = spec.arms[a]
        if s.do != "pick_place":
            continue
        c, o = cands[a], objs[a]
        fam = a[0]
        lift = s.lift if s.lift is not None else LIFT_DEFAULT[fam]
        fg, R = c.flange_grasp, c.R_flange
        place = R26._place_rel(fg, o.pos, o.place_xy, PLACE_LEGACY_DZ[fam] + s.place_dz)
        place = (place[0], place[1], place[2] + s.place_dz)
        tg["approach"][a] = c.flange_pre
        tg["descend"][a] = fg
        tg["lift"][a] = _above(fg, lift)
        if fam == "F":
            tg["carry"][a] = _above(place, lift)
        else:
            tg["carry"][a] = spec.u_lane[a]              # audited centre-lane waypoint
            tg["carry_u"][a] = _above(place, 0.10)
        tg["place"][a] = place
        if fam == "F":
            tg["clear"][a] = _above(place, 0.14)
        tg["retreat"][a] = spec.homes[a]
        # R34 skeleton: the U lane move is position-only, the orientation comes back in carry_u
        ori_phases = ("approach", "descend", "lift", "carry", "place", "clear") if fam == "F" \
            else ("approach", "descend", "lift", "carry_u", "place")
        for k in ori_phases:
            if a in tg[k]:
                oris[k][a] = R
        grasp_arms.append(a)
        hand_events.append(("release", 0.75 if fam == "U" else 0.90, a, "open", 0.0))
        if o.name not in attached:
            object_events.append(("close", 0.90, "attach", a, None, o.name, GATE_TCP_OBJECT))
            attached.add(o.name)
        if primary is None and fam == "F":
            primary = o
        if primary is None:
            primary = o

    # ---- co-lift groups: both pinches translate by one displacement, slow symmetric release
    for g in spec.groups:
        o = objs[g.arms[0]]
        dx = (g.carry_dxy[0], g.carry_dxy[1], 0.0)
        first = True
        for a in g.arms:
            c = cands[a]
            fg, R = c.flange_grasp, c.R_flange
            fg_c = tuple(p + q for p, q in zip(fg, dx))
            tg["approach"][a] = c.flange_pre
            tg["descend"][a] = fg
            tg["lift"][a] = _above(fg, g.lift)
            tg["carry"][a] = _above(fg_c, g.lift)
            # place = grasp height by default (colift3: no blind overshoot); GroupSpec.place_dz
            # lowers place/hold by the measured hang so the object touches down before release
            tg["place"][a] = (fg_c[0], fg_c[1], fg_c[2] + g.place_dz)
            tg["clear"][a] = _above(fg_c, 0.14)
            if g.clear_axial_m > 0:
                axis = _long_axis(o)
                # Determine the side from the selected station, not arm names:
                # the same rule covers both x-long and y-long beams.
                side = -1.0 if c.tcp_grasp[axis] < o.pos[axis] else 1.0
                target = list(tg["place"][a])
                target[axis] += side * g.clear_axial_m
                axial_targets[a] = tuple(target)
                axial_oris[a] = R
                raised = list(tg["clear"][a])
                raised[axis] = target[axis]
                tg["clear"][a] = tuple(raised)
            tg["retreat"][a] = spec.homes[a]
            for k in ("approach", "descend", "lift", "carry", "place", "clear"):
                oris[k][a] = R
            grasp_arms.append(a)
            group_arms.append(a)
            if a.startswith("F"):
                # Franka pads: slow symmetric opening so they do not drag the standing profile over
                hand_events.append(("release", 0.30, a, "open", 0.0, 1.0))
            else:
                # DFX claw: use the legacy 0.6 s opening ramp. resolve_events supplies
                # that default; the recorder clears the separate thumb goal on open.
                hand_events.append(("release", 0.30, a, "open", 0.0))
            if first:
                object_events.append(("close", 0.90, "attach", a, None, o.name, GATE_TCP_LONG))
                attached.add(o.name)
                first = False
        if primary is None:
            primary = o

    # ---- patrol / idle arms: object-free excursion so the participation audit passes
    for a in ARMS:
        s = spec.arms[a]
        if s.do != "idle" or spec.group_of(a) is not None:
            continue
        home = spec.homes[a]
        out = tuple(h + d for h, d in zip(home, PATROL_OFFSET))
        tg["approach"][a] = home
        tg["carry"][a] = out
        tg["place"][a] = home
        tg["retreat"][a] = home
        lerps_carry_u[a] = dict(_BREATH)

    phases = [
        _ph("approach", T("approach"), targets=tg["approach"], oris=oris["approach"]),
        _ph("descend", T("descend"), targets=tg["descend"], oris=oris["descend"]),
        _ph("close", T("close")),
        _ph("lift", T("lift"), targets=tg["lift"], oris=oris["lift"]),
        _ph("carry", T("carry"), targets=tg["carry"], oris=oris["carry"]),
        _ph("carry_u", T("carry_u"), targets=tg["carry_u"], oris=oris["carry_u"], lerps=lerps_carry_u),
        _ph("place", T("place"), targets=tg["place"], oris=oris["place"]),
    ]
    if group_arms:
        phases.append(_ph("hold", T("hold")))          # gated lagging hand catches up before the pads open
    phases.append(_ph("release", T("release")))
    if group_arms:
        phases.append(_ph("settle", T("settle")))
    if axial_targets:
        phases.append(_ph("clear_axial", T("clear_axial"),
                          targets=axial_targets, oris=axial_oris))
    phases += [
        _ph("clear", T("clear"), targets=tg["clear"], oris=oris["clear"]),
        _ph("retreat", T("retreat"), targets=tg["retreat"]),
    ]

    object_events += [("release", 0.60, "release", None, None, n, None) for n in sorted(attached)]
    assert primary is not None, f"{spec.family}: no grasped object"
    if spec.success.get("object"):
        primary = next(o for o in objs.values() if o.name == spec.success["object"])
    prim_dz = next((spec.arms[a].place_dz for a in ARMS
                    if spec.arms[a].do == "pick_place" and objs[a].name == primary.name), 0.0)
    success = {"kind": "pick_place", "place_xy": list(primary.place_xy),
               "radius_m": float(spec.success.get("radius_m", 0.08)),
               "min_disp_m": float(spec.success.get("min_disp_m", 0.08)),
               "rest_z": primary.pos[2] + prim_dz + primary.origin_dz}
    names = [primary.name] + [n for n in sorted(attached) if n != primary.name]
    names += [o.name for o in spec.objects if o.passive]
    return S9Task(
        family=spec.family,
        spec=SkillSpec(f"s9_{spec.family}", phases),
        hand_events=R26._close_events(cands, grasp_arms) + hand_events,
        object_events=object_events,
        success=success,
        objects=names,
        env_yaml=env_yaml_name(spec),
    )


# --------------------------------------------------------------------------- registration
def env_yaml_name(spec: TaskSpec) -> str:
    return f"duo_env_v7_spec_{spec.family}.yaml"


JIG_HALF_GAP = 0.030               # rail centres +/- 3 cm across the short axis -> 5 cm channel (R35)
JIG_RAIL_USD = "assets_real/objects/industrial/jig_rail_700/jig_rail_700.usda"


def jig_props(spec: TaskSpec) -> list:
    """Static stop rails for every co-lift group with `jig: true` (R35 _profile_jig_props
    generalised): the channel sits at the group's place (object xy + carry_dxy) and runs
    along the object's long axis. Fingertips pinching 1.2 cm below the top of an 8 cm
    profile stay > 2 cm above the 3 cm rails."""
    objs = objects_for(spec)
    props = []
    for g in spec.groups:
        if not g.jig:
            continue
        o = objs[g.arms[0]]
        cx, cy = o.place_xy                          # objects_for: group place = xy + carry_dxy
        along_x = _long_axis(o) == 0
        for i, off in enumerate((-JIG_HALF_GAP, JIG_HALF_GAP)):
            pos = [round(cx, 4), round(cy + off, 4)] if along_x else [round(cx + off, 4), round(cy, 4)]
            props.append({"name": f"jig_{o.name}_{i}", "usd": JIG_RAIL_USD,
                          "pos": pos + [round(R35.TABLE_TOP_Z + 0.0175, 4)],
                          "yaw": 0.0 if along_x else 90.0, "collision": True})
    return props


def register(spec: TaskSpec) -> str:
    """Make the spec a first-class family for R34.build_task_r34 / the env writers."""
    fam = spec.family
    R34._OBJECTS[fam] = (lambda spec=spec: objects_for(spec))
    R34._VARIANTS[fam] = (lambda provider, spec=spec: variants(spec, provider))
    R34.R34_ENV[fam] = env_yaml_name(spec)
    R35.R35_ENV[fam] = env_yaml_name(spec)
    if any(g.jig for g in spec.groups):
        R35.FAMILY_STATIC_PROPS[fam] = (lambda spec=spec: jig_props(spec))
    else:
        R35.FAMILY_STATIC_PROPS.pop(fam, None)
    return fam


def write_env_yaml(spec: TaskSpec, dressing: bool = True) -> str:
    if spec.env_base == "factory":
        return R35.write_env_yaml(spec.family, dressing=dressing)
    return R34.write_env_yaml(spec.family)


def describe(spec: TaskSpec) -> str:
    lines = [f"{spec.family} (category {spec.category}, env {spec.env_base})"]
    for a in ARMS:
        s = spec.arms[a]
        g = spec.group_of(a)
        if g is not None:
            lines.append(f"  {a}: colift {g.obj} with {'+'.join(g.arms)} carry {g.carry_dxy}")
        elif s.do == "pick_place":
            o = spec.object(s.obj)
            d = math.hypot(o.place_xy[0] - o.xy[0], o.place_xy[1] - o.xy[1])
            lines.append(f"  {a}: pick {o.name} ({o.catalog}) at {o.xy} -> {o.place_xy} ({d:.2f} m)")
        else:
            lines.append(f"  {a}: patrol")
    return "\n".join(lines)
