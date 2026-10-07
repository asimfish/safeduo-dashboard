"""v7 real-asset scene provider (R14 S5): FR3+F2 (F side unchanged from v5)
+ UR5+RH56DFX (U side), scene_layout_v7 placement, real_v7 spheres,
contact_semantics_v7 + duo_env_v7 safety overrides.

Same caliber family as make_v5_provider (the skill library's hard-gate audit):
RealGeometryProvider FK + SphereDistanceModule pair tables. U-arm kinematics
and the DFX hand zero-pose tree are parsed at build time from the R14-cleaned
URDF (assets_src/real/ur5_20260818/ur5+RH56DFX/urdf/ur5_dfx_right_v7.urdf --
left arm was normalized to the identical convention, FK parity checked in
R14 Round 169), so there is no hand-transcription drift. Base placement and
init poses quote scene_layout_v7.yaml / duo_env_v7.yaml verbatim.

v7 birth pose = duo_env_v7 init_qpos: F arms keep the v4.1 birth (pending the
R14 birth re-search), U arms sit at the UR retract pose. Sanity anchor: at
this rest pose the S1-fixed real_v7 pack reads self_U ~ +27.6 mm.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from safeduo.baselines.real_geometry import (
    FR3_FLANGE_V4,
    FR3_JOINTS,
    FR3_LINKS,
    ArmKinematics,
    RealGeometryProvider,
    SceneLayoutV4,
    _precompose_hand_spheres,
    _rpy_matrix,
)
from safeduo.safety.types import ARM_KEYS

REPO_ROOT = Path(__file__).resolve().parents[3]
UR5_DFX_URDF = (REPO_ROOT / "assets_src" / "real" / "ur5_20260818"
                / "ur5+RH56DFX" / "urdf" / "ur5_dfx_right_v7.urdf")
DUO_ENV_V7 = REPO_ROOT / "src" / "safeduo" / "configs" / "duo_env_v7.yaml"
SEMANTICS_V7 = REPO_ROOT / "src" / "safeduo" / "configs" / "contact_semantics_v7.yaml"

UR5_DFX_LINKS = ("base_link", "shoulder_link", "upper_arm_link",
                 "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link")
# EE caliber: U flange = wrist_3_link frame origin (duo_env ur5_dfx EE body;
# merge-joints put hand base/flange/palm into the wrist_3 body)
UR5_DFX_FLANGE = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

# duo_env_v7.yaml init_qpos (F = v4.1 birth carried over; U = UR retract)
FR3_INIT_Q_V7 = (0.0592, -0.6042, -0.0997, -2.2558, 0.3542, 1.4988, 1.2033)
UR_INIT_Q_V7 = (0.0, -2.2, 1.9, -1.383, -1.57, 0.0)


# ---------------------------------------------------------------------------
# URDF parsing: arm joint table + hand zero-pose tree
# ---------------------------------------------------------------------------


def _parse_urdf_joints(urdf_path: "str | Path") -> list:
    out = []
    for j in ET.parse(urdf_path).getroot().iter("joint"):
        if j.get("type") is None:      # transmission etc.
            continue
        o = j.find("origin")
        xyz = tuple(float(x) for x in (o.get("xyz") or "0 0 0").split()) \
            if o is not None else (0.0, 0.0, 0.0)
        rpy = tuple(float(x) for x in (o.get("rpy") or "0 0 0").split()) \
            if o is not None else (0.0, 0.0, 0.0)
        ax = j.find("axis")
        axis = tuple(float(x) for x in ax.get("xyz").split()) \
            if ax is not None else (0.0, 0.0, 1.0)
        out.append({"name": j.get("name"), "type": j.get("type"),
                    "parent": j.find("parent").get("link"),
                    "child": j.find("child").get("link"),
                    "xyz": xyz, "rpy": rpy, "axis": axis})
    return out


def ur5_dfx_joint_table(urdf_path: "str | Path" = UR5_DFX_URDF) -> tuple:
    """(joints, internal_yaw) for ArmKinematics from the cleaned v7 URDF.

    Chain base_link -> wrist_3_link: exactly one leading fixed joint (the UR
    base_link->base_link_inertia Rz(pi)) then six revolute z-axis joints with
    no interleaved fixed transforms -- both structural facts are asserted so
    a future URDF revision cannot silently break the folding."""
    joints = _parse_urdf_joints(urdf_path)
    by_parent: dict = {}
    for j in joints:
        by_parent.setdefault(j["parent"], []).append(j)

    def path_to(target: str, link: str) -> "list | None":
        if link == target:
            return []
        for j in by_parent.get(link, []):
            sub = path_to(target, j["child"])
            if sub is not None:
                return [j] + sub
        return None

    chain = path_to("wrist_3_link", "base_link")
    assert chain is not None, "no base_link -> wrist_3_link chain in URDF"
    rev = [j for j in chain if j["type"] == "revolute"]
    assert len(rev) == 6, f"expected 6 revolute arm joints, got {len(rev)}"
    lead = chain[: chain.index(rev[0])]
    assert all(j["type"] == "fixed" for j in lead)
    yaw = 0.0
    for j in lead:
        assert j["xyz"] == (0.0, 0.0, 0.0) and j["rpy"][:2] == (0.0, 0.0), \
            f"leading fixed joint {j['name']} is not a pure z-rotation"
        yaw += j["rpy"][2]
    assert chain[chain.index(rev[0]):] == rev, \
        "fixed joints interleaved between revolute arm joints"
    for j in rev:
        assert j["axis"] == (0.0, 0.0, 1.0), \
            f"{j['name']}: non-z axis {j['axis']} (ArmKinematics fast path)"
    return tuple((j["xyz"], j["rpy"]) for j in rev), yaw


def dfx_hand_tree(urdf_path: "str | Path" = UR5_DFX_URDF) -> dict:
    """{link: (R (3,3), t (3,))} zero-pose transforms of every wrist_3_link
    descendant, expressed IN the wrist_3_link frame (= the v7 U flange).
    Revolute finger joints evaluated at q=0 (open hand, duo_env default)."""
    joints = _parse_urdf_joints(urdf_path)
    by_parent: dict = {}
    for j in joints:
        by_parent.setdefault(j["parent"], []).append(j)
    out = {"wrist_3_link": (torch.eye(3), torch.zeros(3))}
    stack = ["wrist_3_link"]
    while stack:
        link = stack.pop()
        R, t = out[link]
        for j in by_parent.get(link, []):
            Rj = R @ _rpy_matrix(j["rpy"])
            tj = t + R @ torch.tensor(j["xyz"], dtype=torch.float32)
            out[j["child"]] = (Rj, tj)          # revolute at q = 0
            stack.append(j["child"])
    return out


# ---------------------------------------------------------------------------
# scene layout + spec assembly
# ---------------------------------------------------------------------------


@dataclass
class SceneLayoutV7(SceneLayoutV4):
    """assets_src/real/scene_layout_v7.yaml (owner calibration robot_base.json):
    rows 1.4964 m apart (v5: 1.10), FR3 pair 0.9742, UR5 pair 0.8717, bases
    z=0.80, tables two 1.0 x 1.5 boards butted at x=0, top z=0.80."""
    base_x: float = 0.7482
    base_z: float = 0.80
    f_half_spacing: float = 0.4871
    u_half_spacing: float = 0.4358
    table_centers: tuple = ((0.50, 0.0, 0.40), (-0.50, 0.0, 0.40))
    table_half_extents: tuple = ((0.50, 0.75, 0.40), (0.50, 0.75, 0.40))

    def base_pose(self, arm: str) -> tuple:
        if arm.startswith("F"):        # F on +x facing -x, F_L at -y
            y = -self.f_half_spacing if arm == "F_L" else self.f_half_spacing
            return (self.base_x, y, self.base_z), math.pi
        y = self.u_half_spacing if arm == "U_L" else -self.u_half_spacing
        return (-self.base_x, y, self.base_z), 0.0


def real_v7_specs_for_fk(repo_root=REPO_ROOT) -> dict:
    """real_v7_arm_specs with hand spheres precomposed onto each arm's flange
    frame: F side reuses the F2 v5 code path verbatim; U side composes the
    DFX finger-link offsets through the URDF zero-pose tree into the
    wrist_3_link frame (the U flange). wrist_3_link palm spheres are already
    in a chain-link frame and ride it directly (untouched)."""
    from safeduo.safety.sphere_distance import LinkSpheres
    from safeduo.safety.sphere_specs import real_v7_arm_specs

    specs = real_v7_arm_specs(repo_root)
    _precompose_hand_spheres({"F_L": specs["F_L"], "F_R": specs["F_R"]})
    tree = dfx_hand_tree()
    side_map = {"U_L": "left", "U_R": "right"}
    for arm in ("U_L", "U_R"):
        side = side_map[arm]
        new_links = []
        for ls in specs[arm].links:
            if not (ls.semantic_name or "").startswith("hand/") \
                    or ls.link in UR5_DFX_LINKS:
                new_links.append(ls)
                continue
            # right-arm URDF tree keyed right_*; mirror name for the left side
            key = ls.link if ls.link in tree else ls.link.replace(
                f"{side}_", "right_", 1)
            R, t = tree[key]
            offs = []
            for o in ls.offsets:
                v = R @ torch.tensor(o, dtype=torch.float32) + t
                offs.append((float(v[0]), float(v[1]), float(v[2])))
            new_links.append(LinkSpheres(ls.link, offs, list(ls.radii),
                                         table_check=ls.table_check,
                                         semantic_name=ls.semantic_name))
        specs[arm].links[:] = new_links
    return specs


def v7_semantics():
    """Production caliber: contact_semantics_v7 + duo_env_v7 safety overrides
    (the same construction duo_env / diag_v7_self_overlap use)."""
    from safeduo.safety.semantics import semantics_with_safety_overrides

    env = yaml.safe_load(DUO_ENV_V7.read_text())
    return semantics_with_safety_overrides(str(SEMANTICS_V7), env["safety"])


def real_v7r16_specs_for_fk(repo_root=REPO_ROOT) -> dict:
    """r16 geometry shell (S12 2026-08-21, library home since R27 2026-09-05):
    F arms = assets_src/real/spheres_r16 (fr3_link3/link6 refit + F2 fingers
    3 balls r15-18 instead of 1 ball r45) precomposed onto the flange exactly
    like real_v7_specs_for_fk; U arms = real_v7 verbatim. Same construction as
    tools/s12_r16_offline_validation.make_r16_provider and duo_env's
    spheres_mode "real_v7r16"."""
    from safeduo.safety.sphere_distance import LinkSpheres
    from safeduo.safety.sphere_specs import (_FR3_FLANGE_SPHERE, ArmSpheres,
                                             load_robot_yaml)

    specs = real_v7_specs_for_fk(repo_root)
    d = Path(repo_root) / "assets_src" / "real" / "spheres_r16"
    fr3_full = load_robot_yaml(str(d / "fr3_r16.yaml"), "fr3")
    arm_links = [ls for ls in fr3_full.links if ls.link != "fr3_hand"]
    arm_links.append(LinkSpheres(*_FR3_FLANGE_SPHERE))

    def hand(side: str) -> list:
        h = load_robot_yaml(str(d / f"inspire_rh56f2_{side}_r16.yaml"), "rh56f2")
        return [LinkSpheres(ls.link, ls.offsets, ls.radii,
                            semantic_name=f"hand/{ls.link}")
                for ls in h.links]

    f_specs = {"F_L": ArmSpheres(links=list(arm_links) + hand("left")),
               "F_R": ArmSpheres(links=list(arm_links) + hand("right"))}
    _precompose_hand_spheres(f_specs)
    specs["F_L"], specs["F_R"] = f_specs["F_L"], f_specs["F_R"]
    return specs


def v7r16_semantics():
    """contact_semantics_v7r16 + duo_env_v7_r16 safety overrides (the r16
    production caliber, mirrors v7_semantics)."""
    from safeduo.safety.semantics import semantics_with_safety_overrides

    cfg_dir = REPO_ROOT / "src" / "safeduo" / "configs"
    env16 = yaml.safe_load((cfg_dir / "duo_env_v7_r16.yaml").read_text())
    return semantics_with_safety_overrides(
        str(cfg_dir / "contact_semantics_v7r16.yaml"), env16["safety"])


def make_v7r16_provider(n_envs: int, device: "str | torch.device" = "cpu",
                        top_m: int = 64) -> RealGeometryProvider:
    """make_v7_provider with the r16 geometry shell (R27: lets the trajectory
    designer audit deep pinches against realistic finger spheres)."""
    joints, internal_yaw = ur5_dfx_joint_table()
    kin = {
        "F": ArmKinematics(FR3_JOINTS, FR3_LINKS, True, FR3_FLANGE_V4,
                           0.0, device=device),
        "U": ArmKinematics(joints, UR5_DFX_LINKS, True, UR5_DFX_FLANGE,
                           internal_yaw, device=device),
    }
    init_q = {"F_L": FR3_INIT_Q_V7, "F_R": FR3_INIT_Q_V7,
              "U_L": UR_INIT_Q_V7, "U_R": UR_INIT_Q_V7}
    return RealGeometryProvider(
        n_envs, device=device, top_m=top_m,
        layout=SceneLayoutV7(),
        specs=real_v7r16_specs_for_fk(),
        kin=kin, init_q=init_q,
        semantics=v7r16_semantics())


def make_v7_provider(n_envs: int, device: "str | torch.device" = "cpu",
                     top_m: int = 64) -> RealGeometryProvider:
    joints, internal_yaw = ur5_dfx_joint_table()
    kin = {
        "F": ArmKinematics(FR3_JOINTS, FR3_LINKS, True, FR3_FLANGE_V4,
                           0.0, device=device),
        "U": ArmKinematics(joints, UR5_DFX_LINKS, True, UR5_DFX_FLANGE,
                           internal_yaw, device=device),
    }
    init_q = {"F_L": FR3_INIT_Q_V7, "F_R": FR3_INIT_Q_V7,
              "U_L": UR_INIT_Q_V7, "U_R": UR_INIT_Q_V7}
    return RealGeometryProvider(
        n_envs, device=device, top_m=top_m,
        layout=SceneLayoutV7(),
        specs=real_v7_specs_for_fk(),
        kin=kin, init_q=init_q,
        semantics=v7_semantics())


def min_margin_by_class4(provider: RealGeometryProvider, q: dict) -> dict:
    """Four-channel rest/trajectory margins: cross / self_F / self_U / table
    (the v7 GT reporting split; provider.min_margin_by_class merges self)."""
    d_all, _, _ = provider._all_margins(provider.fk_all(q)["centers"])
    is_self = provider.pair_class == 1.0
    is_u = provider.pair_arm_i >= 2                    # ARM_KEYS order F,F,U,U
    masks = {"cross": provider.pair_class == 0.0,
             "self_F": is_self & ~is_u,
             "self_U": is_self & is_u,
             "table": provider.pair_class == 2.0}
    out = {}
    for name, m in masks.items():
        out[name] = d_all[:, m].amin(dim=1) if m.any() else \
            torch.full((provider.n,), torch.inf, device=provider.device)
    return out


class V7SkillFK:
    """Flange-position FK on the v7 scene (skill_success judge caliber for
    the v7 library; mirrors eval.skill_success.V5SkillFK's interface)."""

    def __init__(self, device: "str | torch.device" = "cpu"):
        self.device = torch.device(device)
        self.layout = SceneLayoutV7()
        joints, internal_yaw = ur5_dfx_joint_table()
        self.kin = {
            "F": ArmKinematics(FR3_JOINTS, FR3_LINKS, True, FR3_FLANGE_V4,
                               0.0, device=device),
            "U": ArmKinematics(joints, UR5_DFX_LINKS, True, UR5_DFX_FLANGE,
                               internal_yaw, device=device),
        }

    def ee_pos(self, q_by_arm: dict) -> dict:
        out = {}
        for arm, q in q_by_arm.items():
            q = torch.as_tensor(q, dtype=torch.float32, device=self.device)
            lead = q.shape[:-1]
            q2 = q.reshape(-1, q.shape[-1])
            pos, yaw = self.layout.base_pose(arm)
            fko = self.kin[arm[0]].fk(q2, pos, yaw)
            out[arm] = fko["t_flange"].reshape(*lead, 3)
        return out


def default_q_v7(n: int = 1, device: "str | torch.device" = "cpu") -> dict:
    init = {"F_L": FR3_INIT_Q_V7, "F_R": FR3_INIT_Q_V7,
            "U_L": UR_INIT_Q_V7, "U_R": UR_INIT_Q_V7}
    return {a: torch.tensor(init[a], dtype=torch.float32,
                            device=device).expand(n, -1).clone()
            for a in ARM_KEYS}
