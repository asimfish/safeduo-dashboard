"""Real 4-arm constraint geometry: FR3 + UR5e FK, B's sphere decomposition,
A's pair tables -> ConstraintRows for the baselines (and local SceneState).

Sources of truth:
- Joint frames: URDF joint origins transcribed from the archived repos
  (assets_src/third_party/curobo .../franka_panda.urdf for FR3-kinematics-
  equals-Panda, .../ur10e.urdf for the UR frame *structure* with official
  UR5e lengths substituted). tests/test_real_geometry.py re-derives these
  tables from the URDFs when the archive is present.
- Spheres: B's assets_src/spheres/{fr3,ur5e}.yaml (link-local frames), loaded
  through A's sphere_specs.from_curobo_yaml.
- Pair enumeration (cross/self/table, adjacency exemption, class_id, pair_id):
  reuses A's SphereDistanceModule static tables verbatim, so pair semantics
  are bit-identical with the runtime safety layer (fairness + G0 parity).
- Scene layout: numbers mirror configs/duo_env.yaml (bases x=+-0.40, y=+-0.30,
  z=0.75, F faces +x, U yawed pi; tables 0.8x1.2x0.75 with 0.5 gap).

Known W2 approximations (logged in STATUS_C):
- UR URDF carries an internal base_link -> base_link_inertia yaw of pi, but
  applying it puts the UR wrists 1.4 cm from the F wrists at A's init pose,
  contradicting A's stated init intent ("elbow folded over own table, no
  centerline crossing"). Default ur_internal_yaw=0.0 reproduces that intent
  (init cross margin 0.178 m); the URDF value stays available via the
  UR_INTERNAL_YAW constant. Which convention Isaac's ur5e.usd actually bakes
  is a flagged A3 server-parity item (@A in STATUS_C).
- Contact semantics: loads B's contact_semantics.yaml through A's
  ContactSemantics by default (same adjacency exemptions -- e.g. the always-
  overlapping fr3_link5/link7 pair -- and per-pair d_min as the runtime
  safety layer). semantics=None falls back to A's legacy ordinal rule.
- include_hand=True rigidly mounts B's 8 Inspire spheres on the flange frame
  (finger proximal articulation ignored; radii already cover bend envelopes).
  Off by default until B's assembly transform is confirmed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from safeduo.baselines.base import ConstraintRows, GeometryProvider
from safeduo.safety.sphere_distance import SphereDistanceModule
from safeduo.safety.sphere_specs import from_curobo_yaml
from safeduo.safety.types import ARM_KEYS, DOF_OF, SceneState

REPO_ROOT = Path(__file__).resolve().parents[3]
SPHERE_DIR = REPO_ROOT / "assets_src" / "spheres"

# ---- URDF joint tables: (xyz, rpy) of the joint origin; all axes are child z ----

# franka_panda.urdf (curobo main@8e734f3); FR3 kinematics == Panda (B's note)
FR3_JOINTS = (
    ((0.0, 0.0, 0.333), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (-math.pi / 2, 0.0, 0.0)),
    ((0.0, -0.316, 0.0), (math.pi / 2, 0.0, 0.0)),
    ((0.0825, 0.0, 0.0), (math.pi / 2, 0.0, 0.0)),
    ((-0.0825, 0.384, 0.0), (-math.pi / 2, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (math.pi / 2, 0.0, 0.0)),
    ((0.088, 0.0, 0.0), (math.pi / 2, 0.0, 0.0)),
)
FR3_LINKS = tuple(f"fr3_link{i}" for i in range(8))  # link0 = base frame
FR3_FLANGE = ((0.0, 0.0, 0.107), (0.0, 0.0, -math.pi / 4))  # joint8 + hand yaw

# ur10e.urdf structure with official UR5e lengths
# (d1=0.1625, a2=-0.425, a3=-0.3922, d4=0.1333, d5=0.0997, d6=0.0996)
UR5E_JOINTS = (
    ((0.0, 0.0, 0.1625), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (math.pi / 2, 0.0, 0.0)),
    ((-0.425, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((-0.3922, 0.0, 0.1333), (0.0, 0.0, 0.0)),
    ((0.0, -0.0997, 0.0), (math.pi / 2, 0.0, 0.0)),
    ((0.0, 0.0996, 0.0), (math.pi / 2, math.pi, math.pi)),
)
UR5E_LINKS = ("shoulder_link", "upper_arm_link", "forearm_link",
              "wrist_1_link", "wrist_2_link", "wrist_3_link")
UR5E_FLANGE = ((0.0, 0.0, 0.0), (0.0, -math.pi / 2, -math.pi / 2))
UR_INTERNAL_YAW = math.pi  # base_link -> base_link_inertia in the URDF

# links whose spheres skip the table check (base/shoulder live near the table)
TABLE_SKIP = {"fr3": ("fr3_link0", "fr3_link1"), "ur5e": ("shoulder_link",)}

# init poses mirroring A's env defaults
FR3_INIT_Q = (0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741)
UR5E_INIT_Q = (0.0, -1.9, 1.9, -1.57, -1.57, 0.0)


@dataclass
class SceneLayout:
    """Mirror of configs/duo_env.yaml scene section (env-local frame)."""
    base_x: float = 0.40        # gap/2 + inset
    base_half_spacing: float = 0.30
    base_z: float = 0.75        # table top
    table_centers: tuple = ((-0.65, 0.0, 0.375), (0.65, 0.0, 0.375))
    table_half_extents: tuple = ((0.4, 0.6, 0.375), (0.4, 0.6, 0.375))

    def base_pose(self, arm: str) -> tuple:
        x = -self.base_x if arm.startswith("F") else self.base_x
        y = self.base_half_spacing if arm in ("F_L", "U_R") else -self.base_half_spacing
        yaw = 0.0 if arm.startswith("F") else math.pi
        return (x, y, self.base_z), yaw

    def t_fu7(self) -> tuple:
        """U_L base pose in the F_L base frame, 7-vector pos+quat(wxyz).
        Generic formula (C5-W7): the old hardcoded (2bx, -2hs, 0, 0,0,0,1)
        assumed F at -x / yaw 0 -- the v4 real layout flips both."""
        (pf, yf) = self.base_pose("F_L")
        (pu, yu) = self.base_pose("U_L")
        dx, dy, dz = (pu[0] - pf[0], pu[1] - pf[1], pu[2] - pf[2])
        c, s = math.cos(-yf), math.sin(-yf)
        rel_yaw = yu - yf
        return (c * dx - s * dy, s * dx + c * dy, dz,
                math.cos(rel_yaw / 2), 0.0, 0.0, math.sin(rel_yaw / 2))


def _rpy_matrix(rpy) -> torch.Tensor:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    # URDF convention: R = Rz(y) @ Ry(p) @ Rx(r)
    return torch.tensor([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=torch.float32)


def _rz_batch(q: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(q), torch.sin(q)
    z, o = torch.zeros_like(q), torch.ones_like(q)
    return torch.stack([
        torch.stack([c, -s, z], -1),
        torch.stack([s, c, z], -1),
        torch.stack([z, z, o], -1),
    ], dim=-2)


def _axis_frame(axis) -> torch.Tensor:
    """(3,3) rotation whose third column is `axis` (any unit-ish vector).
    Rotation about `axis` by q == C @ Rz(q) @ C^T for ANY such C -- the
    choice of the first two columns cancels in the conjugation."""
    a = torch.tensor(axis, dtype=torch.float32)
    a = a / a.norm()
    ref = torch.tensor([1.0, 0.0, 0.0]) if float(a[0].abs()) < 0.9 \
        else torch.tensor([0.0, 1.0, 0.0])
    u = torch.linalg.cross(ref, a)
    u = u / u.norm()
    v = torch.linalg.cross(a, u)
    return torch.stack([u, v, a], dim=-1)


class ArmKinematics:
    """Batched serial-chain FK for one arm variant.

    W2 build assumed all revolute joints spin about the child +z (true for
    the FR3/UR5e tables). C5-W7 generalizes to per-joint axes for JAKA Zu7
    (its URDF mixes +-y and -x axes): pass `axes` = one 3-vector per joint.
    axes=None keeps the original all-z fast path BIT-EXACTLY (the legacy
    branch below is the untouched W2 code path)."""

    def __init__(self, joints, link_names, base_frame_is_link0: bool,
                 flange, internal_yaw: float = 0.0,
                 device: "str | torch.device" = "cpu",
                 axes: "tuple | None" = None):
        self.device = torch.device(device)
        self.dof = len(joints)
        self.link_names = link_names
        self.base_frame_is_link0 = base_frame_is_link0
        self.R_org = torch.stack([_rpy_matrix(rpy) for _, rpy in joints]).to(self.device)
        self.t_org = torch.tensor([xyz for xyz, _ in joints],
                                  dtype=torch.float32, device=self.device)
        self.R_int = _rpy_matrix((0.0, 0.0, internal_yaw)).to(self.device)
        self.R_fl = _rpy_matrix(flange[1]).to(self.device)
        self.t_fl = torch.tensor(flange[0], dtype=torch.float32, device=self.device)
        self.axes = axes
        if axes is not None:
            if len(axes) != self.dof:
                raise ValueError(f"axes needs {self.dof} entries")
            self.axis_vec = torch.stack([
                torch.tensor(a, dtype=torch.float32) / torch.tensor(
                    a, dtype=torch.float32).norm() for a in axes]).to(self.device)
            self.C = torch.stack([_axis_frame(a) for a in axes]).to(self.device)
            self.Ct = self.C.transpose(-1, -2)
        # frame index each named link attaches to (-1 = base frame before joint 1)
        off = -1 if base_frame_is_link0 else 0
        self.frame_of_link = {n: i + off for i, n in enumerate(link_names)}

    def fk(self, q: torch.Tensor, base_pos, base_yaw: float):
        """q: (N, dof). Returns dict with per-frame world rotation/translation,
        joint axes/origins for jacobians, and the flange frame.

        frames: R (N, dof+1, 3, 3), t (N, dof+1, 3) -- index 0 = base frame,
        index k = frame after joint k. Joint j data: axis z_j, origin o_j.
        """
        n = q.shape[0]
        dev = q.device
        Rb = (_rpy_matrix((0.0, 0.0, base_yaw)).to(dev) @ self.R_int.to(dev))
        R = Rb.expand(n, 3, 3).contiguous()
        t = torch.tensor(base_pos, dtype=torch.float32, device=dev).expand(n, 3)
        Rs, ts = [R], [t]
        z_ax, o_org = [], []
        for j in range(self.dof):
            R_pre = R @ self.R_org[j].to(dev)
            t_pre = t + (R @ self.t_org[j].to(dev))
            o_org.append(t_pre)
            if self.axes is None:
                z_ax.append(R_pre[..., :, 2])
                R = R_pre @ _rz_batch(q[:, j])
            else:
                z_ax.append((R_pre @ self.axis_vec[j].to(dev)))
                R = R_pre @ (self.C[j].to(dev) @ _rz_batch(q[:, j])
                             @ self.Ct[j].to(dev))
            t = t_pre
            Rs.append(R)
            ts.append(t)
        R_flange = R @ self.R_fl.to(dev)
        t_flange = t + (R @ self.t_fl.to(dev))
        return {
            "R": torch.stack(Rs, dim=1), "t": torch.stack(ts, dim=1),
            "z": torch.stack(z_ax, dim=1), "o": torch.stack(o_org, dim=1),
            "R_flange": R_flange, "t_flange": t_flange,
        }

    def point_jacobian(self, fkout: dict, p: torch.Tensor,
                       frame_idx: torch.Tensor) -> torch.Tensor:
        """p: (N, S, 3) world points attached to frames frame_idx: (S,).
        Returns (N, S, 3, dof); column j = z_j x (p - o_j) for j < moving joints
        of the frame (frame k moves with joints 0..k-1 -> frame_idx entries are
        'frame after joint k' indices, so joint j affects iff j <= k-1)."""
        n, s = p.shape[0], p.shape[1]
        z = fkout["z"].unsqueeze(2)            # (N, dof, 1, 3)
        o = fkout["o"].unsqueeze(2)            # (N, dof, 1, 3)
        pj = p.unsqueeze(1)                    # (N, 1, S, 3)
        col = torch.cross(z.expand(n, self.dof, s, 3),
                          pj - o, dim=-1)      # (N, dof, S, 3)
        jdx = torch.arange(self.dof, device=p.device).view(1, -1, 1)
        moving = (jdx <= (frame_idx.view(1, 1, -1) - 1)).unsqueeze(-1)
        col = torch.where(moving, col, torch.zeros_like(col))
        return col.permute(0, 2, 3, 1)         # (N, S, 3, dof)


def load_arm_specs(include_hand: bool = False, sphere_dir: "Path | None" = None) -> dict:
    """B's YAML decomposition -> per-arm ArmSpheres (A's spec dataclasses)."""
    d = Path(sphere_dir or SPHERE_DIR)
    fr3 = from_curobo_yaml(str(d / "fr3.yaml"), list(FR3_LINKS),
                           table_skip=TABLE_SKIP["fr3"])
    ur5e = from_curobo_yaml(str(d / "ur5e.yaml"), list(UR5E_LINKS),
                            table_skip=TABLE_SKIP["ur5e"])
    if include_hand:
        from safeduo.safety.sphere_distance import LinkSpheres

        import yaml

        data = yaml.safe_load((d / "inspire_hand.yaml").read_text())
        hand = data.get("collision_spheres") or data["inspire_hand"]
        offs, rads = [], []
        for entries in hand.values():
            offs.extend(tuple(e["center"]) for e in entries)
            rads.extend(float(e["radius"]) for e in entries)
        for arm_spec, flange_link in ((fr3, "hand_mount_F"), (ur5e, "hand_mount_U")):
            arm_spec.links.append(LinkSpheres(flange_link, offs, rads, table_check=True))
    return {"F_L": fr3, "F_R": fr3, "U_L": ur5e, "U_R": ur5e}


class RealGeometryProvider(GeometryProvider):
    """ConstraintRows from real FR3/UR5e kinematics + B spheres + A pair tables.

    C5-W7 injection points (all default to the W2 behavior): `specs` and
    `kin` override the sphere decomposition / kinematic chains (the v4 real
    assets swap the U arms to JAKA Zu7, see make_v4_provider), `init_q`
    overrides the default episode-start pose, and `semantics` also accepts
    an explicit YAML path (v4 uses configs/contact_semantics_v3.yaml).
    Any spec link name that is not in its arm's kinematic chain rides the
    FLANGE frame (v0 'hand_mount_*' behavior generalized -- v4 uses it for
    the fr3_link8 flange puck and the precomposed F2 hand spheres)."""

    def __init__(self, n_envs: int, device: "str | torch.device" = "cpu",
                 top_m: int = 64, include_hand: bool = False,
                 layout: "SceneLayout | None" = None,
                 adjacent_skip: int = 2, ur_internal_yaw: float = 0.0,
                 semantics: "str | None" = "auto",
                 sphere_dir: "Path | None" = None,
                 specs: "dict | None" = None,
                 kin: "dict | None" = None,
                 init_q: "dict | None" = None):
        self.n = n_envs
        self.device = torch.device(device)
        self.top_m = top_m
        self.include_hand = include_hand
        self.layout = layout or SceneLayout()
        self.specs = specs or load_arm_specs(include_hand, sphere_dir)
        self.init_q = init_q
        sem = None
        if semantics == "auto":
            sem_path = REPO_ROOT / "assets_src" / "contact_semantics.yaml"
            if sem_path.exists():
                from safeduo.safety.semantics import ContactSemantics

                sem = ContactSemantics(str(sem_path))
        elif isinstance(semantics, str) and semantics:
            from safeduo.safety.semantics import ContactSemantics

            sem = ContactSemantics(str(REPO_ROOT / semantics))
        elif semantics is not None and not isinstance(semantics, str):
            # C6-W8: a pre-built ContactSemantics instance (the v5 path --
            # thresholds are overridden from the bundle manifest before the
            # pair tables bake d_min, mirroring duo_env's d_min_override hook)
            sem = semantics
        # A's v2 module supplies the static pair tables + per-pair d_min
        # (B's semantics when available; never call compute() here)
        self.sph = SphereDistanceModule(self.specs, semantics=sem,
                                        adjacent_skip=adjacent_skip, device=device)
        self.kin = kin or {
            "F": ArmKinematics(FR3_JOINTS, FR3_LINKS, True, FR3_FLANGE,
                               0.0, device=device),
            "U": ArmKinematics(UR5E_JOINTS, UR5E_LINKS, False, UR5E_FLANGE,
                               ur_internal_yaw, device=device),
        }
        # per-sphere frame index within its arm chain (order matches A's flatten)
        self._frame_idx, self._hand_sphere = {}, {}
        for arm in ARM_KEYS:
            kin_a = self.kin[arm[0]]
            idx, is_hand = [], []
            for li, ls in enumerate(self.specs[arm].links):
                hand_link = ls.link not in kin_a.frame_of_link
                f = kin_a.dof if hand_link else kin_a.frame_of_link[ls.link] + 1
                # frame_of_link gives "after joint k" index k-? -> +1 converts
                # to fk() frames array indexing (0=base, k=after joint k)
                idx.extend([f] * len(ls.radii))
                is_hand.extend([hand_link] * len(ls.radii))
            self._frame_idx[arm] = torch.tensor(idx, dtype=torch.long, device=self.device)
            self._hand_sphere[arm] = torch.tensor(is_hand, dtype=torch.bool,
                                                  device=self.device)
        # static per-pair metadata for row assembly
        self._build_pair_meta()
        tc = torch.tensor(self.layout.table_centers, dtype=torch.float32,
                          device=self.device)
        th = torch.tensor(self.layout.table_half_extents, dtype=torch.float32,
                          device=self.device)
        self._tables = (tc, th)

    # ---- static metadata -----------------------------------------------------
    def _build_pair_meta(self) -> None:
        sph = self.sph
        arm_of_sphere = sph.arm_id                        # (S,) index into ARM_KEYS
        pc, ps = len(sph.pairs_cross), len(sph.pairs_self)
        pt = len(sph.pairs_table)
        sph_i = torch.cat([sph.pairs_cross[:, 0], sph.pairs_self[:, 0],
                           sph.pairs_table[:, 0]])
        sph_j = torch.cat([sph.pairs_cross[:, 1], sph.pairs_self[:, 1],
                           torch.zeros(pt, dtype=torch.long, device=self.device)])
        tab_j = torch.cat([torch.full((pc + ps,), -1, dtype=torch.long,
                                      device=self.device), sph.pairs_table[:, 1]])
        is_table = tab_j >= 0
        # BUG GUARD (@A, STATUS_C W2): A's semantics path in
        # SphereDistanceModule._judge_link_pair does not exempt same-link /
        # ordinal-adjacent same-arm sphere pairs, so B's deliberately
        # overlapping tube-chain spheres register as permanent -8 cm "self
        # collisions" (VIOLATION at spawn). Until A applies the ordinal
        # exemption inside the semantics branch, filter those pairs here so
        # local tables match the documented intent (legacy ordinal rule AND
        # semantics adjacency both apply).
        same_arm = arm_of_sphere[sph_i] == arm_of_sphere[sph_j.clamp(min=0)]
        ord_close = (sph.link_ord[sph_i]
                     - sph.link_ord[sph_j.clamp(min=0)]).abs() < self.sph.adjacent_skip
        keep = is_table | ~(same_arm & ord_close)
        self._n_pairs_dropped_bugguard = int((~keep).sum().item())
        sph_i, sph_j, tab_j, is_table = (sph_i[keep], sph_j[keep], tab_j[keep],
                                         is_table[keep])
        self.pair_sph_i, self.pair_sph_j, self.pair_tab = sph_i, sph_j, tab_j
        self.pair_is_table = is_table
        self.pair_class = sph.class_id[keep]              # (P,)
        self.pair_id = sph.pair_id[keep]                  # keeps A's indices
        self.pair_dmin_v = sph.pair_dmin[keep]
        # R27/R29 (2026-09-05): structural table rows = per-link d_min override
        # far below the table class tier (the UR5 upper_arm x table row hovers
        # at +1.6 mm with d_min 0.001 in every pose). Same predicate family as
        # the R29 backstop exemption; consumers opt in via
        # min_margin_by_class(exclude_structural=True).
        tab_tier = float(self.pair_dmin_v[self.pair_is_table].max()) if bool(self.pair_is_table.any()) else 0.0
        self.pair_structural = self.pair_is_table & (self.pair_dmin_v < 0.5 * tab_tier)
        arm_i = arm_of_sphere[sph_i]
        arm_j = torch.where(self.pair_is_table, arm_i, arm_of_sphere[sph_j])
        self.pair_arm_i, self.pair_arm_j = arm_i, arm_j
        am = torch.zeros(len(sph_i), 4, dtype=torch.bool, device=self.device)
        am[torch.arange(len(sph_i)), arm_i] = True
        am[torch.arange(len(sph_i)), arm_j] = True
        self.pair_arm_mask = am
        # column offsets of each arm inside the 26-dim stacked vector
        offs, o = [], 0
        for a in ARM_KEYS:
            offs.append(o)
            o += DOF_OF[a]
        self.arm_col_offset = torch.tensor(offs, dtype=torch.long, device=self.device)
        self.n_pairs = len(sph_i)

    # ---- runtime kinematics ----------------------------------------------------
    def fk_all(self, q: dict) -> dict:
        """Per-arm fk output + world sphere centers and padded point jacobians."""
        out = {}
        centers, jpt = [], []
        for arm in ARM_KEYS:
            kin = self.kin[arm[0]]
            pos, yaw = self.layout.base_pose(arm)
            fko = kin.fk(q[arm], pos, yaw)
            sl = self.sph._arm_slices[arm]
            offsets = self.sph.offsets[sl]                        # (Sa, 3)
            fidx = self._frame_idx[arm].clone()
            hand = self._hand_sphere[arm]
            R = fko["R"][:, fidx.clamp(max=kin.dof)]              # (N, Sa, 3, 3)
            t = fko["t"][:, fidx.clamp(max=kin.dof)]
            if hand.any():  # hand spheres ride the flange frame
                R = torch.where(hand.view(1, -1, 1, 1), fko["R_flange"].unsqueeze(1), R)
                t = torch.where(hand.view(1, -1, 1), fko["t_flange"].unsqueeze(1), t)
            c = t + (R @ offsets.unsqueeze(0).unsqueeze(-1).to(q[arm])).squeeze(-1)
            J = kin.point_jacobian(fko, c, fidx)                  # (N, Sa, 3, dof)
            out[arm] = fko
            centers.append(c)
            # pad dof -> 7 so all arms stack into one (N, S, 3, 7) tensor
            if J.shape[-1] < 7:
                J = torch.nn.functional.pad(J, (0, 7 - J.shape[-1]))
            jpt.append(J)
        out["centers"] = torch.cat(centers, dim=1)                # (N, S, 3)
        out["jpt"] = torch.cat(jpt, dim=1)                        # (N, S, 3, 7)
        return out

    def ee_pose(self, fkout: dict) -> tuple:
        pos = {a: fkout[a]["t_flange"] for a in ARM_KEYS}
        rot = {a: fkout[a]["R_flange"] for a in ARM_KEYS}
        return pos, rot

    # ---- margins for every pair -------------------------------------------------
    def _all_margins(self, centers: torch.Tensor) -> tuple:
        """Returns (d (N,P), n_hat (N,P,3), table_grad_point (N,P,3))."""
        ci = centers[:, self.pair_sph_i]                          # (N, P, 3)
        radii = self.sph.radii
        # sphere-sphere part
        cj = centers[:, self.pair_sph_j.clamp(min=0)]
        rel = ci - cj
        dist = rel.norm(dim=-1)
        n_hat = rel / dist.clamp_min(1e-9).unsqueeze(-1)
        d_ss = dist - (radii[self.pair_sph_i] + radii[self.pair_sph_j.clamp(min=0)])
        # sphere-table part (A's box SDF formula)
        tc, th = self._tables
        tab = self.pair_tab.clamp(min=0)
        qv = (ci - tc[tab]).abs() - th[tab]
        outside = qv.clamp_min(0.0)
        d_out = outside.norm(dim=-1)
        d_in = qv.amax(dim=-1).clamp_max(0.0)
        sdf = d_out + d_in
        d_tb = sdf - radii[self.pair_sph_i]
        sgn = torch.sign(ci - tc[tab])
        grad_out = sgn * outside / d_out.clamp_min(1e-9).unsqueeze(-1)
        inside_axis = torch.nn.functional.one_hot(qv.argmax(dim=-1), 3).to(ci.dtype)
        grad_in = sgn * inside_axis
        tgrad = torch.where((d_out > 0).unsqueeze(-1), grad_out, grad_in)
        is_t = self.pair_is_table
        d = torch.where(is_t, d_tb, d_ss)
        return d, n_hat, tgrad

    # ---- provider API -------------------------------------------------------------
    def rows(self, state: SceneState) -> ConstraintRows:
        return self.rows_from_q(state.q)

    def rows_from_q(self, q: dict) -> ConstraintRows:
        fko = self.fk_all(q)
        centers, jpt = fko["centers"], fko["jpt"]
        d_all, n_hat, tgrad = self._all_margins(centers)
        m = min(self.top_m, self.n_pairs)
        sel_d, sel = torch.topk(d_all, m, dim=1, largest=False)   # (N, M)
        n_env = torch.arange(self.n, device=self.device).unsqueeze(-1)
        sph_i, sph_j = self.pair_sph_i[sel], self.pair_sph_j[sel]  # (N, M)
        is_t = self.pair_is_table[sel]
        # gradient direction wrt point i: sphere pairs use n_hat, table rows use SDF grad
        gdir = torch.where(is_t.unsqueeze(-1), tgrad.gather(
            1, sel.unsqueeze(-1).expand(-1, -1, 3)),
            n_hat.gather(1, sel.unsqueeze(-1).expand(-1, -1, 3)))
        J_i = jpt[n_env, sph_i]                                   # (N, M, 3, 7)
        J_j = jpt[n_env, sph_j.clamp(min=0)]
        vals_i = torch.einsum("nmc,nmcj->nmj", gdir, J_i)          # (N, M, 7)
        vals_j = -torch.einsum("nmc,nmcj->nmj", gdir, J_j)
        vals_j = torch.where(is_t.unsqueeze(-1), torch.zeros_like(vals_j), vals_j)
        # scatter-add both contributions into the 26-dim stacked row
        row26 = torch.zeros(self.n, m, 26, device=self.device)
        for vals, arms in ((vals_i, self.pair_arm_i[sel]), (vals_j, self.pair_arm_j[sel])):
            off = self.arm_col_offset[arms]                        # (N, M)
            idx = off.unsqueeze(-1) + torch.arange(7, device=self.device)
            # 6-dof arms: 7th column would bleed out; value there is 0 -> clamp
            idx = idx.clamp(max=25)
            row26.scatter_add_(2, idx, vals)
        arm_mask = self.pair_arm_mask[sel]                         # (N, M, 4)
        return ConstraintRows(
            d=sel_d,
            J={"F": row26[..., :14], "U": row26[..., 14:]},
            cls=self.pair_class[sel],
            arm_mask=arm_mask,
            valid=torch.ones(self.n, m, dtype=torch.bool, device=self.device),
            d_min=self.pair_dmin_v[sel],  # A's per-pair braking boundary
        )

    # ---- full contract SceneState for local rollouts -------------------------------
    def scene_state(self, q: dict, qd: "dict | None" = None,
                    max_active: "int | None" = None, d_soft: float = 0.05,
                    tau_ttc: float = 0.5, dt: float = 0.02) -> SceneState:
        from safeduo.safety.types import MAX_ACTIVE_PAIRS

        max_active = max_active or MAX_ACTIVE_PAIRS
        qd = qd or {a: torch.zeros_like(q[a]) for a in ARM_KEYS}
        fko = self.fk_all(q)
        centers, jpt = fko["centers"], fko["jpt"]
        d_all, n_hat, tgrad = self._all_margins(centers)
        # d_dot for every pair from point jacobians and qd (approach = positive)
        qd_pad = []
        for arm in ARM_KEYS:
            v = qd[arm]
            if v.shape[-1] < 7:
                v = torch.nn.functional.pad(v, (0, 7 - v.shape[-1]))
            qd_pad.append(v)
        qd_of_arm = torch.stack(qd_pad, dim=1)                     # (N, 4, 7)
        vel_sph = torch.einsum("nscj,nsj->nsc", jpt,
                               qd_of_arm[:, self.sph.arm_id])      # (N, S, 3)
        vi = vel_sph[:, self.pair_sph_i]
        vj = vel_sph[:, self.pair_sph_j.clamp(min=0)]
        vj = torch.where(self.pair_is_table.view(1, -1, 1), torch.zeros_like(vj), vj)
        gdir = torch.where(self.pair_is_table.view(1, -1, 1), tgrad, n_hat)
        ddot = torch.einsum("npc,npc->np", gdir, vi - vj)
        closing = -ddot                                            # >0 approaching
        from safeduo.baselines.base import active_pair_set

        feat, mask = active_pair_set(
            d_all, closing, self.pair_class.unsqueeze(0).expand(self.n, -1),
            self.pair_id.unsqueeze(0).expand(self.n, -1),
            max_active, d_soft, tau_ttc)
        ee_p, ee_r = self.ee_pose(fko)
        quat = {a: _rot_to_wxyz(ee_r[a]) for a in ARM_KEYS}
        # U_L base in F_L base frame (matches A's duo_env T_FU convention;
        # generic formula so the v4 layout -- F at +x, yaw pi -- stays right)
        t_fu = torch.tensor(self.layout.t_fu7(),
                            device=self.device).expand(self.n, 7)
        return SceneState(q=q, qd=qd, ee_pos=ee_p, ee_quat=quat,
                          active_pairs=feat, active_mask=mask, T_FU=t_fu, dt=dt)

    def ee_jacobian(self, arm: str, state: SceneState) -> torch.Tensor:
        """(N, 3, dof) flange position jacobian -- plugs into JacobianMapper."""
        kin = self.kin[arm[0]]
        pos, yaw = self.layout.base_pose(arm)
        fko = kin.fk(state.q[arm], pos, yaw)
        p = fko["t_flange"].unsqueeze(1)                           # (N, 1, 3)
        fidx = torch.tensor([kin.dof], dtype=torch.long, device=p.device)
        return kin.point_jacobian(fko, p, fidx)[:, 0]              # (N, 3, dof)

    def default_q(self, jitter: float = 0.0,
                  generator: "torch.Generator | None" = None) -> dict:
        q = {}
        for arm in ARM_KEYS:
            if self.init_q is not None:
                init = self.init_q[arm]
            else:
                init = FR3_INIT_Q if arm.startswith("F") else UR5E_INIT_Q
            base = torch.tensor(init, device=self.device).expand(
                self.n, DOF_OF[arm]).clone()
            if jitter > 0.0:
                base = base + (torch.rand(self.n, DOF_OF[arm], device=self.device,
                                          generator=generator) * 2 - 1) * jitter
            q[arm] = base
        return q

    def min_margin_by_class(self, q: dict, exclude_structural: bool = False) -> dict:
        d_all, _, _ = self._all_margins(self.fk_all(q)["centers"])
        out = {}
        for name, cls in (("cross", 0.0), ("self", 1.0), ("table", 2.0)):
            mask = self.pair_class == cls
            if exclude_structural and cls == 2.0:
                mask = mask & ~self.pair_structural
            out[name] = d_all[:, mask].amin(dim=1) if mask.any() else \
                torch.full((self.n,), torch.inf, device=self.device)
        return out


def _rot_to_wxyz(R: torch.Tensor) -> torch.Tensor:
    """(N,3,3) -> (N,4) wxyz; branch-free-enough conversion for logging use."""
    w = (1.0 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]).clamp_min(1e-9).sqrt() * 0.5
    x = (R[:, 2, 1] - R[:, 1, 2]) / (4 * w)
    y = (R[:, 0, 2] - R[:, 2, 0]) / (4 * w)
    z = (R[:, 1, 0] - R[:, 0, 1]) / (4 * w)
    return torch.stack([w, x, y, z], dim=-1)


# ============================================================================
# v4 real-asset scene (C5-W7): JAKA Zu7 + F2 hands + scene_layout_v3
# ============================================================================

# ---- JAKA Zu7 joint table, transcribed from assets_src/real/
# jaka_zu7_clean.urdf (B's cleaned conversion source; axes are NOT all-z,
# hence the ArmKinematics `axes` extension). Verification: reach check in
# tests/test_v4_geometry.py against the official 0.819 m datasheet number
# scene_layout_v3.yaml quotes.
JAKA_JOINTS = (
    ((0.0068395, -0.0017473, 0.040967), (1.5708, 0.0, 1.5708)),
    ((0.06407, 0.08015, 0.0), (-3.1416, -1.5708, 1.5708)),
    ((0.36, 0.0, 0.0), (0.0, 3.1415926, -3.1415926)),
    ((0.303, -0.0054826, 0.0), (-1.5708, 0.0, -1.5708)),
    ((0.04546, -0.067529, 0.0), (3.1415926, 0.0, 1.5708)),
    ((-0.045971, -0.069408, 0.0), (0.0, -1.5708, 0.0)),
)
JAKA_AXES = ((0, 1, 0), (0, -1, 0), (0, 1, 0), (-1, 0, 0), (-1, 0, 0), (0, -1, 0))
JAKA_LINKS = ("base_link", "link1", "link2", "link3", "link4", "link5", "link6")
JAKA_FLANGE = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))   # tool frame = link6 frame

# v4 prep poses = duo_env v4 init (envs/duo_env.py _FR3_INIT_OVERRIDE j4/j6 +
# _JAKA_INIT_QPOS with A6's j2 1.35 birth-window fix)
FR3_INIT_Q_V4 = (0.0, -0.569, 0.0, -2.31, 0.0, 1.74, 0.741)
JAKA_INIT_Q_V4 = (0.0, 1.35, -1.2, 1.2, 1.57, 0.0)
# v4 F flange = fr3_link8 TRUE frame (joint7 + 0.107, no -pi/4): the F2 hand
# welds onto fr3_link8 with identity rotation (compose_robot_usd), unlike the
# v0 inspire mount that carried the panda_hand -pi/4 yaw.
FR3_FLANGE_V4 = ((0.0, 0.0, 0.107), (0.0, 0.0, 0.0))
F2_MOUNT_Z = 0.017          # flange spacer (scene_layout_v3 mount_offset_pos)

# ---- RH56F2 zero-pose link tree {child: (parent, xyz, rpy)}, transcribed
# from assets_real RH56F2_{L,R}_clean.urdf (server; extraction command in
# STATUS_C W7). Zero pose = open hand, matching duo_env's default hand
# actuator target. B's sphere placement is flexion-robust (centers on the
# *_2 joint axes / mimic-invariant), so riding the zero-pose frames rigidly
# is the same approximation class the v0 include_hand mount used.
_F2_TREE = {
    "left": {
        "left_hand_palm": (None, (0.0, 0.0, 0.0305), (0.0, 0.0, 0.0)),
        "left_thumb_1": ("left_hand_palm",
                         (0.018361, -0.027493, 0.040054), (0.0, 0.0, 0.0)),
        "left_thumb_2": ("left_thumb_1",
                         (0.0090358556, -0.0124048111, -0.0052961217),
                         (1.5707093852, 0.0, -1.4835298639)),
        "left_thumb_3": ("left_thumb_2",
                         (0.0437230821, 0.0271075946, -0.0025), (0.0, 0.0, 0.0)),
        "left_index_1": ("left_hand_palm",
                         (0.0007075284, -0.03437652, 0.1050417896),
                         (1.6318828506, 0.0, 0.0)),
        "left_middle_1": ("left_hand_palm",
                          (0.0007074581, -0.015767486, 0.1059777153),
                          (1.5707963267, 0.0, 0.0)),
        "left_ring_1": ("left_hand_palm",
                        (0.0007074624, 0.0030876019, 0.1059278653),
                        (1.5184364492, 0.0, 0.0)),
        "left_pinky_1": ("left_hand_palm",
                         (0.0007075282, 0.0210944783, 0.1048071609),
                         (1.4660765716, 0.0, 0.0)),
    },
    "right": {
        "right_hand_palm": (None, (0.0, 0.0, 0.0305), (0.0, 0.0, 0.0)),
        "right_thumb_1": ("right_hand_palm",
                          (0.018359979, 0.0272411074, 0.0400114099),
                          (0.0, 0.0, 0.0)),
        "right_thumb_2": ("right_thumb_1",
                          (0.0097343431, 0.0125953579, -0.0052540455),
                          (1.5708832684, 0.0, 1.4835298639)),
        "right_thumb_3": ("right_thumb_2",
                          (0.0437230821, 0.0271075946, 0.0018),
                          (-8.72736808e-05, 0.0872664626, 0.0)),
        "right_index_1": ("right_hand_palm",
                          (0.0007075284, 0.03437652, 0.1050417896),
                          (1.509709803, 0.0, 0.0)),
        "right_middle_1": ("right_hand_palm",
                           (0.0007074402, 0.015767486, 0.1059777142),
                           (1.5707963268, 0.0, 0.0)),
        "right_ring_1": ("right_hand_palm",
                         (0.0007074624, -0.0030876019, 0.1059278653),
                         (1.6231562044, 0.0, 0.0)),
        "right_pinky_1": ("right_hand_palm",
                          (0.0007075169, -0.0214508257, 0.1047697047),
                          (1.6755160820, 0.0, 0.0)),
    },
}


def _f2_link_transform(side: str, link: str) -> tuple:
    """Zero-pose (R, t) of a hand link frame in the hand_base frame."""
    R = torch.eye(3)
    t = torch.zeros(3)
    chain = []
    cur = link
    tree = _F2_TREE[side]
    while cur is not None and cur in tree:
        chain.append(tree[cur])
        cur = tree[cur][0]
    for parent, xyz, rpy in reversed(chain):
        Rj = _rpy_matrix(rpy)
        tj = torch.tensor(xyz, dtype=torch.float32)
        t = t + R @ tj
        R = R @ Rj
    return R, t


def _precompose_hand_spheres(specs: dict) -> dict:
    """Precompose every hand link's sphere offsets into the FLANGE frame
    (mount +0.017 z, then the zero-pose hand chain) IN PLACE -- so the whole
    hand rides the provider's flange branch while pair tables / semantic
    names stay bit-identical with the Isaac path. Extracted verbatim from
    the C5-W7 real_v3_specs_for_fk body so the v5 spec set reuses the exact
    same transform code path (zero drift between scene generations)."""
    from safeduo.safety.sphere_distance import LinkSpheres

    side_of = {"F_L": "left", "F_R": "right", "U_L": "left", "U_R": "right"}
    for arm, spec in specs.items():
        side = side_of[arm]
        new_links = []
        for ls in spec.links:
            if not (ls.semantic_name or "").startswith("hand/"):
                new_links.append(ls)
                continue
            base_name = f"{side}_hand_base"
            if ls.link == base_name:
                R, t = torch.eye(3), torch.zeros(3)
            else:
                R, t = _f2_link_transform(side, ls.link)
            offs = []
            for o in ls.offsets:
                v = R @ torch.tensor(o, dtype=torch.float32) + t
                offs.append((float(v[0]), float(v[1]), float(v[2]) + F2_MOUNT_Z))
            new_links.append(LinkSpheres(ls.link, offs, list(ls.radii),
                                         table_check=ls.table_check,
                                         semantic_name=ls.semantic_name))
        spec.links[:] = new_links
    return specs


def real_v3_specs_for_fk(repo_root=REPO_ROOT) -> dict:
    """safety.sphere_specs.real_v3_arm_specs with every hand link's sphere
    offsets precomposed into the FLANGE frame (mount +0.017 z, then the
    zero-pose hand chain) -- so the whole hand rides the provider's flange
    branch and the pair tables / semantic names stay bit-identical with the
    Isaac path (same LinkSpheres entries, same 'hand/<link>' names)."""
    from safeduo.safety.sphere_specs import real_v3_arm_specs

    return _precompose_hand_spheres(real_v3_arm_specs(repo_root))


@dataclass
class SceneLayoutV4(SceneLayout):
    """assets_src/real/scene_layout_v3.yaml numbers (strong-coupling layout):
    bases +-0.55 x / +-0.50 y at z 0.816 (0.80 table + 16 mm pedestal),
    F on the +x table facing -x (yaw pi), U (JAKA) at -x facing +x --
    NOTE x-side and yaw are BOTH flipped vs the v0 SceneLayout convention.
    Tables 0.8 x 1.2 x 0.80 boxes centered +-0.65."""
    base_x: float = 0.55
    base_half_spacing: float = 0.50
    base_z: float = 0.816
    table_centers: tuple = ((0.65, 0.0, 0.40), (-0.65, 0.0, 0.40))
    table_half_extents: tuple = ((0.4, 0.6, 0.40), (0.4, 0.6, 0.40))

    def base_pose(self, arm: str) -> tuple:
        x = self.base_x if arm.startswith("F") else -self.base_x
        # scene_layout_v3: F_L at y=-0.50 (F faces -x, its left = -y);
        # U_L at y=+0.50 (U faces +x, its left = +y)
        y = -self.base_half_spacing if arm in ("F_L", "U_R") else self.base_half_spacing
        yaw = math.pi if arm.startswith("F") else 0.0
        return (x, y, self.base_z), yaw


def make_v4_provider(n_envs: int, device: "str | torch.device" = "cpu",
                     top_m: int = 64) -> RealGeometryProvider:
    """RealGeometryProvider on the v4 real scene: JAKA Zu7 kinematics,
    real_v3 sphere decomposition (arm + flange + F2 hands), scene_layout_v3
    placement, contact_semantics_v3 pair rules, duo_env v4 prep poses."""
    kin = {
        "F": ArmKinematics(FR3_JOINTS, FR3_LINKS, True, FR3_FLANGE_V4,
                           0.0, device=device),
        "U": ArmKinematics(JAKA_JOINTS, JAKA_LINKS, True, JAKA_FLANGE,
                           0.0, device=device, axes=JAKA_AXES),
    }
    init_q = {"F_L": FR3_INIT_Q_V4, "F_R": FR3_INIT_Q_V4,
              "U_L": JAKA_INIT_Q_V4, "U_R": JAKA_INIT_Q_V4}
    return RealGeometryProvider(
        n_envs, device=device, top_m=top_m,
        layout=SceneLayoutV4(),
        specs=real_v3_specs_for_fk(),
        kin=kin, init_q=init_q,
        semantics="src/safeduo/configs/contact_semantics_v3.yaml")


# ============================================================================
# v5 scene (C6-W8): B2 bundle -- de-inflated sphere rev + gap=0 layout + new
# d_min semantics. Single entry point = assets_src/real/v5_bundle_manifest.yaml
# (supervision Round-121 open); v4 constructors above stay bit-untouched so
# grid_v4 / s42 / s43 artifacts remain reproducible with current code.
# ============================================================================

V5_MANIFEST = REPO_ROOT / "assets_src" / "real" / "v5_bundle_manifest.yaml"

# v4.1 birth pose transcribed from configs/duo_env_v4.yaml init_qpos (A7-W7
# birth re-engineering, Round-114 ruling) -- the pose the s42/s43 trainings
# actually ran and the pose the B2 v5 manifest prescribes for v5 ("no re-
# search needed, wrist-ring j6=0 stays optimal"). NOTE the v4 grid constants
# above (FR3_INIT_Q_V4 / JAKA_INIT_Q_V4) predate this pose; they are kept
# verbatim for grid_v4 reproducibility. If A9's integration lands a further
# birth tweak in STATUS_A, re-anchor here (draft-table caveat in STATUS_C W8).
FR3_INIT_Q_V41 = (0.0592, -0.6042, -0.0997, -2.2558, 0.3542, 1.4988, 1.2033)
JAKA_INIT_Q_V41 = (0.1623, 1.0319, -0.8742, 1.2292, 1.5062, 0.0)


@dataclass
class SceneLayoutV5(SceneLayoutV4):
    """assets_src/real/scene_layout_v5.yaml: tables butted (gap=0, photo
    adjudication), width 1.2 -> 1.50, per-table box 0.8 x 1.5 x 0.80 centered
    x = +-0.40. Base anchors (+-0.55 / +-0.50 / z 0.816) unchanged from v4."""
    table_centers: tuple = ((0.40, 0.0, 0.40), (-0.40, 0.0, 0.40))
    table_half_extents: tuple = ((0.4, 0.75, 0.40), (0.4, 0.75, 0.40))


def real_v5_arm_specs(repo_root=REPO_ROOT) -> dict:
    """v5 bundle sphere set (B2 manifest components.spheres): F arm =
    spheres/fr3.yaml minus fr3_hand plus flange puck (all unchanged from v4)
    + rh56f2 v5 hand (10 -> 11 spheres, vertex-level de-inflation); U arm =
    jaka_zu7_v5.yaml (16 -> 18, wrist de-fattened, flange disk axis fixed)
    + v5 hand. Mirrors safety.sphere_specs.real_v3_arm_specs structure;
    lives here because baselines/ is the C-line write boundary."""
    from safeduo.safety.sphere_distance import LinkSpheres
    from safeduo.safety.sphere_specs import (
        _FR3_FLANGE_SPHERE,
        load_robot_yaml,
    )

    d = Path(repo_root) / "assets_src"
    fr3_full = load_robot_yaml(str(d / "spheres" / "fr3.yaml"), "fr3")
    arm_links = [ls for ls in fr3_full.links if ls.link != "fr3_hand"]
    arm_links.append(LinkSpheres(*_FR3_FLANGE_SPHERE))
    jaka = load_robot_yaml(str(d / "real" / "jaka_zu7_v5.yaml"), "jaka_zu7")

    def hand_links(side: str) -> list:
        hand = load_robot_yaml(
            str(d / "real" / f"inspire_rh56f2_{side}_v5.yaml"), "rh56f2")
        return [LinkSpheres(ls.link, ls.offsets, ls.radii,
                            semantic_name=f"hand/{ls.link}")
                for ls in hand.links]

    from safeduo.safety.sphere_specs import ArmSpheres

    return {
        "F_L": ArmSpheres(links=list(arm_links) + hand_links("left")),
        "F_R": ArmSpheres(links=list(arm_links) + hand_links("right")),
        "U_L": ArmSpheres(links=list(jaka.links) + hand_links("left")),
        "U_R": ArmSpheres(links=list(jaka.links) + hand_links("right")),
    }


def real_v5_specs_for_fk(repo_root=REPO_ROOT) -> dict:
    """v5 spec set with hands precomposed onto the flange frame (identical
    transform code path as the v4 real_v3_specs_for_fk)."""
    return _precompose_hand_spheres(real_v5_arm_specs(repo_root))


def v5_semantics(dmin_caliber: str = "manifest"):
    """contact_semantics_v3 rules with v5 threshold overrides applied BEFORE
    the pair tables bake d_min (same injection point as duo_env's
    safety.d_min_override hook).

    dmin_caliber:
      "manifest"     thresholds_v5 verbatim from the B2 bundle manifest
                     (self/cross 0.013, table 0.020, d_warn 0.058). NOTE
                     (@D, STATUS_C W8): the manifest derives 0.013 from
                     "v4 0.005 + 8mm", but the v4 semantics actually brake
                     at self 0.020 / cross 0.030 -- so this caliber is a
                     real-buffer REDUCTION for cbf/backstop, not an
                     equivalence. The v5 grid measures its safety outcome.
      "equivalence"  B2's +8mm same-real-buffer conversion applied to the
                     true v4 values: self 0.028 / cross 0.038 / table 0.020
                     (sensitivity arm for the cbf intervention-tax readout).
    """
    import yaml

    from safeduo.safety.semantics import ContactSemantics

    sem = ContactSemantics(
        str(REPO_ROOT / "src" / "safeduo" / "configs"
            / "contact_semantics_v3.yaml"))
    if dmin_caliber == "manifest":
        th = yaml.safe_load(V5_MANIFEST.read_text())["thresholds_v5"]
        sem.d_min["self"] = float(th["d_min_self"])
        sem.d_min["cross"] = float(th["d_min_cross"])
        sem.d_min["table"] = float(th["d_min_table"])
        sem.d_warn = float(th["d_warn"])
    elif dmin_caliber == "equivalence":
        sem.d_min["self"] = 0.028
        sem.d_min["cross"] = 0.038
        sem.d_min["table"] = 0.020
    else:
        raise ValueError(f"unknown dmin_caliber {dmin_caliber}")
    return sem


def make_v5_provider(n_envs: int, device: "str | torch.device" = "cpu",
                     top_m: int = 64,
                     dmin_caliber: str = "manifest") -> RealGeometryProvider:
    """RealGeometryProvider on the v5 bundle: v5 spheres (de-inflated),
    scene_layout_v5 placement (gap=0 / 1.5 m tables), v4.1 birth pose
    (B2-manifest prescribed, pending A9 finalization), v5 d_min semantics."""
    kin = {
        "F": ArmKinematics(FR3_JOINTS, FR3_LINKS, True, FR3_FLANGE_V4,
                           0.0, device=device),
        "U": ArmKinematics(JAKA_JOINTS, JAKA_LINKS, True, JAKA_FLANGE,
                           0.0, device=device, axes=JAKA_AXES),
    }
    init_q = {"F_L": FR3_INIT_Q_V41, "F_R": FR3_INIT_Q_V41,
              "U_L": JAKA_INIT_Q_V41, "U_R": JAKA_INIT_Q_V41}
    return RealGeometryProvider(
        n_envs, device=device, top_m=top_m,
        layout=SceneLayoutV5(),
        specs=real_v5_specs_for_fk(),
        kin=kin, init_q=init_q,
        semantics=v5_semantics(dmin_caliber))
