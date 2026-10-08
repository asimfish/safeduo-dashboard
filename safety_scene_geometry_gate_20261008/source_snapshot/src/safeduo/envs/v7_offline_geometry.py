"""S4(R14 收尾/R15 前置): v7 场景 CPU 离线几何 -- 出生位重搜与 reach 盒重标的共用底座.

组装 RealGeometryProvider 的 v7 变体, **不改** baselines/real_geometry.py
(红线: v5/v6 行为逐位不动). 组件与真值来源:

- U 臂运动学: owner 组合 URDF ur5_dfx_left_v7.urdf 的 base_link->wrist_3_link
  串链(UR5 CB 系, 6 关节全 child-z 轴; base_link->base_link_inertia 的固定
  yaw=pi 折进 joint1 origin), flange = wrist_3_link 原点(merge-joints 后
  腕即法兰即手底座, 与 duo_env._EE_BODY["ur5_dfx"]="wrist_3_link" 同轴);
- U 手球预合成: rh56dfx_{side}_v7.yaml 的手指 link 球按 URDF 零位(手关节 0,
  与出生位手执行器默认张开一致)相对 wrist_3 变换, 烘进 flange 帧;
  wrist_3_link 键的手基座球本来就在腕帧, 直接骑链帧;
- F 臂: v5 原样(FR3 链 + FR3_FLANGE_V4 + F2 v5 手预合成, 与 make_v5_provider
  同一代码路径);
- 布局: scene_layout_v7.yaml 逐字(基座位姿; 桌 = 0.05 厚薄板, 与 duo_env
  make_duo_env_cfg 的 table_board 同一口径 -- 注意 v4/v5 CPU provider 用的是
  全高桌盒, v7 这里改薄板是为了与 Isaac 读数逐位对齐);
- 语义: contact_semantics_v7 + duo_env_v7.yaml safety 覆写(同生产/同 S1 diag).

通道 floor 一律在生产 SphereDistanceModule 的 pair 表上算(cross/self_F/
self_U/table 四桶, margin = 球面距, 违规 = margin<0), 与 Isaac
sphere_distance.compute 同一集合同一口径 -- 不用 provider 侧带 bug-guard
过滤的 pair 表.

自检(与 S1 已交叉验证过的 Isaac 冒烟出生读数对表):
  PYTHONPATH=src .venv/bin/python -m safeduo.envs.v7_offline_geometry --selftest
期望: self_U=+27.6mm / self_F=+23.1mm / table=+1.7mm / cross=+763mm.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import yaml

from safeduo.baselines.real_geometry import (
    FR3_FLANGE_V4,
    FR3_JOINTS,
    FR3_LINKS,
    REPO_ROOT,
    ArmKinematics,
    RealGeometryProvider,
    SceneLayout,
    _precompose_hand_spheres,
    _rpy_matrix,
)
from safeduo.safety.semantics import semantics_with_safety_overrides
from safeduo.safety.sphere_specs import real_v7_arm_specs

URDF_DIR = REPO_ROOT / "assets_src/real/ur5_20260818/ur5+RH56DFX/urdf"
LAYOUT_YAML = REPO_ROOT / "assets_src/real/scene_layout_v7.yaml"
ENV_YAML = REPO_ROOT / "src/safeduo/configs/duo_env_v7.yaml"
UR_ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                 "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
UR_SIDE = {"U_L": "left", "U_R": "right"}
UR_CHAIN_LINKS = ("shoulder_link", "upper_arm_link", "forearm_link",
                  "wrist_1_link", "wrist_2_link", "wrist_3_link")


# --------------------------------------------------------------------------
# minimal URDF tree FK (S1 diag 的 UrdfFk torch 移植)
# --------------------------------------------------------------------------

class UrdfTree:
    """URDF 全树 FK(单位形, torch float64): 零位手变换提取 + 自检参照."""

    def __init__(self, urdf: Path):
        root = ET.parse(urdf).getroot()
        self.joints = []
        childs, parents = set(), set()
        for j in root.iter("joint"):
            if j.get("type") is None:      # transmission 等非运动学元素
                continue
            o = j.find("origin")
            xyz = [float(x) for x in (o.get("xyz") or "0 0 0").split()] \
                if o is not None else [0.0] * 3
            rpy = [float(x) for x in (o.get("rpy") or "0 0 0").split()] \
                if o is not None else [0.0] * 3
            ax = j.find("axis")
            axis = [float(x) for x in ax.get("xyz").split()] \
                if ax is not None else [1.0, 0.0, 0.0]
            lim = j.find("limit")
            lo = float(lim.get("lower")) if lim is not None and lim.get("lower") else 0.0
            hi = float(lim.get("upper")) if lim is not None and lim.get("upper") else 0.0
            self.joints.append({
                "name": j.get("name"), "type": j.get("type"),
                "parent": j.find("parent").get("link"),
                "child": j.find("child").get("link"),
                "R": _rpy_matrix(rpy).to(torch.float64),
                "t": torch.tensor(xyz, dtype=torch.float64),
                "axis": torch.tensor(axis, dtype=torch.float64),
                "lower": lo, "upper": hi})
            childs.add(j.find("child").get("link"))
            parents.add(j.find("parent").get("link"))
        roots = parents - childs
        assert len(roots) == 1, f"root 不唯一: {roots}"
        self.root = roots.pop()
        self.by_parent: dict = {}
        for j in self.joints:
            self.by_parent.setdefault(j["parent"], []).append(j)
        self.limits = {j["name"]: (j["lower"], j["upper"])
                       for j in self.joints if j["type"] == "revolute"}

    @staticmethod
    def _axis_rot(axis: torch.Tensor, th: float) -> torch.Tensor:
        a = axis / axis.norm()
        K = torch.tensor([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]],
                          [-a[1], a[0], 0.0]], dtype=torch.float64)
        return (torch.eye(3, dtype=torch.float64) + math.sin(th) * K
                + (1 - math.cos(th)) * (K @ K))

    def link_transforms(self, qpos: dict, base_R=None, base_t=None) -> dict:
        out = {self.root: (
            base_R if base_R is not None else torch.eye(3, dtype=torch.float64),
            base_t if base_t is not None else torch.zeros(3, dtype=torch.float64))}
        stack = [self.root]
        while stack:
            link = stack.pop()
            R, t = out[link]
            for j in self.by_parent.get(link, []):
                Rj = R @ j["R"]
                tj = R @ j["t"] + t
                if j["type"] == "revolute":
                    Rj = Rj @ self._axis_rot(j["axis"], float(qpos.get(j["name"], 0.0)))
                out[j["child"]] = (Rj, tj)
                stack.append(j["child"])
        return out

    def chain_to(self, tip: str) -> list:
        """root->tip 的 joint 记录序列(含 fixed)."""
        by_child = {j["child"]: j for j in self.joints}
        seq, cur = [], tip
        while cur != self.root:
            j = by_child[cur]
            seq.append(j)
            cur = j["parent"]
        return list(reversed(seq))


# --------------------------------------------------------------------------
# UR5 ArmKinematics(串链表由 URDF 现场提取, fixed 变换折进相邻 revolute origin)
# --------------------------------------------------------------------------

def ur5_v7_joint_table() -> tuple:
    """(joints, links): base_link->wrist_3_link, fixed 折叠后 6 个 child-z 轴关节."""
    tree = UrdfTree(URDF_DIR / "ur5_dfx_left_v7.urdf")
    seq = tree.chain_to("wrist_3_link")
    joints, links = [], []
    R_acc = torch.eye(3, dtype=torch.float64)
    t_acc = torch.zeros(3, dtype=torch.float64)
    for j in seq:
        t_acc = t_acc + R_acc @ j["t"]
        R_acc = R_acc @ j["R"]
        if j["type"] == "revolute":
            ax = j["axis"]
            assert torch.allclose(ax, torch.tensor([0.0, 0.0, 1.0],
                                                   dtype=torch.float64)), \
                f"{j['name']} 轴 {ax} 非 child-z, 需走 axes 分支"
            rpy = _mat_to_rpy(R_acc)
            joints.append((tuple(float(x) for x in t_acc), rpy))
            links.append(j["child"])
            R_acc = torch.eye(3, dtype=torch.float64)
            t_acc = torch.zeros(3, dtype=torch.float64)
    assert tuple(links) == UR_CHAIN_LINKS, links
    assert (R_acc - torch.eye(3, dtype=torch.float64)).abs().max() < 1e-9 \
        and t_acc.abs().max() < 1e-12, "wrist_3 之后不应有残余固定变换"
    return tuple(joints), tuple(links)


def _mat_to_rpy(R: torch.Tensor) -> tuple:
    """旋转矩阵 -> URDF rpy(ZYX 欧拉), 与 _rpy_matrix 互逆."""
    sy = -float(R[2, 0])
    p = math.asin(max(-1.0, min(1.0, sy)))
    if abs(abs(sy) - 1.0) < 1e-9:      # gimbal(本链不出现, 防御留门)
        r = math.atan2(float(R[0, 1]), float(R[1, 1]))
        y = 0.0
    else:
        r = math.atan2(float(R[2, 1]), float(R[2, 2]))
        y = math.atan2(float(R[1, 0]), float(R[0, 0]))
    return (r, p, y)


def ur5_v7_kin(device: "str | torch.device" = "cpu") -> ArmKinematics:
    joints, links = ur5_v7_joint_table()
    return ArmKinematics(joints, links, False, ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                         0.0, device=device)


def ur5_v7_limits() -> torch.Tensor:
    """(6,2) 弧度限位, URDF 官方值(pan/lift/wrists +-2pi, elbow +-pi)."""
    tree = UrdfTree(URDF_DIR / "ur5_dfx_left_v7.urdf")
    return torch.tensor([tree.limits[n] for n in UR_ARM_JOINTS])


# --------------------------------------------------------------------------
# v7 specs(手球预合成进 FK 帧) / 布局 / 语义 / provider
# --------------------------------------------------------------------------

def v7_specs_for_fk(repo_root=REPO_ROOT) -> dict:
    from safeduo.safety.sphere_distance import LinkSpheres

    specs = real_v7_arm_specs(repo_root)
    # F 双臂: F2 v5 手 -> fr3 法兰帧(与 make_v5_provider 同一变换代码路径;
    # 只传 F 条目, _F2_TREE 不认识 DFX 手指 link, 不能碰 U)
    _precompose_hand_spheres({"F_L": specs["F_L"], "F_R": specs["F_R"]})
    # U 双臂: DFX 手指 link 球按 URDF 零位烘进 wrist_3(=flange)帧;
    # wrist_3_link 键(手基座/掌球)本来就在腕帧, 名字在链内 -> 骑链帧, 不动
    for arm, side in UR_SIDE.items():
        tree = UrdfTree(URDF_DIR / f"ur5_dfx_{side}_v7.urdf")
        tfs = tree.link_transforms({})            # 全零位(手关节 0 = 出生默认)
        Rw, tw = tfs["wrist_3_link"]
        new_links = []
        for ls in specs[arm].links:
            hand = (ls.semantic_name or "").startswith("hand/")
            if not hand or ls.link == "wrist_3_link":
                new_links.append(ls)
                continue
            Rl, tl = tfs[ls.link]
            R_rel = Rw.T @ Rl
            t_rel = Rw.T @ (tl - tw)
            offs = []
            for o in ls.offsets:
                v = R_rel @ torch.tensor(o, dtype=torch.float64) + t_rel
                offs.append((float(v[0]), float(v[1]), float(v[2])))
            new_links.append(LinkSpheres(ls.link, offs, list(ls.radii),
                                         table_check=ls.table_check,
                                         semantic_name=ls.semantic_name))
        specs[arm].links[:] = new_links
    return specs


class SceneLayoutV7(SceneLayout):
    """scene_layout_v7.yaml 逐字: 非对称两排基座 + 0.05 厚薄板桌(Isaac 口径)."""

    def __init__(self):
        lay = yaml.safe_load(LAYOUT_YAML.read_text())
        self._base = {a: ((tuple(float(v) for v in d["base_pos"])),
                          float(d["base_yaw"]))
                      for a, d in lay["robots"]["arms"].items()}
        tb = lay["tables"]
        sx, sy, _ = (float(v) for v in tb["params"]["table_size"])
        thick = float(tb["params"]["table_top_thickness"])
        cs = []
        for key in ("table_F", "table_U"):        # 顺序 = TABLE_NAMES
            cx, cy = (float(v) for v in tb[key]["center_xy"])
            cs.append((cx, cy, float(tb[key]["top_z"]) - thick / 2))
        self.table_centers = tuple(cs)
        self.table_half_extents = ((sx / 2, sy / 2, thick / 2),) * 2
        self.table_top_z = float(tb["table_F"]["top_z"])
        ws = lay["workspace"]
        self.ws_lo = tuple(float(v) for v in ws["aabb_min"])
        self.ws_hi = tuple(float(v) for v in ws["aabb_max"])
        self.base_z = 0.80

    def base_pose(self, arm: str) -> tuple:
        return self._base[arm]


def v7_semantics():
    env = yaml.safe_load(ENV_YAML.read_text())
    return semantics_with_safety_overrides(
        str(REPO_ROOT / env["assets"]["semantics_yaml"]), env["safety"])


def v7_init_q() -> tuple:
    """duo_env_v7.yaml init_qpos -> (fr3(7,), ur(6,)) 张量."""
    iq = yaml.safe_load(ENV_YAML.read_text())["init_qpos"]
    fr3 = torch.tensor([float(iq["franka"][f"fr3_joint{i}"]) for i in range(1, 8)])
    ur = torch.tensor([float(iq["ur"][n]) for n in UR_ARM_JOINTS])
    return fr3, ur


def make_v7_provider(n_envs: int, device: "str | torch.device" = "cpu",
                     top_m: int = 64) -> RealGeometryProvider:
    fr3_q, ur_q = v7_init_q()
    kin = {"F": ArmKinematics(FR3_JOINTS, FR3_LINKS, True, FR3_FLANGE_V4,
                              0.0, device=device),
           "U": ur5_v7_kin(device=device)}
    init_q = {"F_L": tuple(fr3_q.tolist()), "F_R": tuple(fr3_q.tolist()),
              "U_L": tuple(ur_q.tolist()), "U_R": tuple(ur_q.tolist())}
    return RealGeometryProvider(
        n_envs, device=device, top_m=top_m,
        layout=SceneLayoutV7(), specs=v7_specs_for_fk(),
        kin=kin, init_q=init_q, semantics=v7_semantics())


# --------------------------------------------------------------------------
# 四桶通道 floor(生产 SphereDistanceModule pair 表口径, 违规 = margin<0)
# --------------------------------------------------------------------------

def table_margins(p: RealGeometryProvider, centers: torch.Tensor) -> torch.Tensor:
    """(N,Pt) 薄板桌 margin, 复刻 sphere_distance._table_dist_closing."""
    m = p.sph
    tc = torch.tensor(p.layout.table_centers, dtype=torch.float32)
    th = torch.tensor(p.layout.table_half_extents, dtype=torch.float32)
    sph, tab = m.pairs_table[:, 0], m.pairs_table[:, 1]
    q = (centers[:, sph] - tc[tab]).abs() - th[tab]
    outside = q.clamp_min(0.0)
    sdf = outside.norm(dim=-1) + q.amax(dim=-1).clamp_max(0.0)
    return sdf - m.radii[sph]


_STRUCT_CACHE: dict = {}


def structural_table_mask(p: RealGeometryProvider) -> torch.Tensor:
    """(Pt,) bool: U 臂 upper_arm_link #0 肩领球的对桌行(结构性悬停).

    该球(r=0.0883)球心落在 shoulder_lift 轴上(局部 (-0.001,0,0.141)),
    世界高度被 UR5 d1=0.089159 钉死在桌面上方 ~90mm -- 全 lift 行程扫描
    margin 恒 +0.9~+1.9mm, 任何出生位都抬不动(v6 JAKA link2 同类病理,
    Round 154 ③ / duo_env_v7.yaml 注 ⑤ 预告的复现场景)。
    """
    key = id(p)
    if key in _STRUCT_CACHE:
        return _STRUCT_CACHE[key]
    m = p.sph
    struct_sph = set()
    for arm in ("U_L", "U_R"):
        i = m._arm_slices[arm].start
        for ls in p.specs[arm].links:
            if ls.link == "upper_arm_link":
                struct_sph.add(i)          # 该 link 的 #0 球
                break
            i += len(ls.radii)
    mask = torch.tensor([int(s) in struct_sph for s in m.pairs_table[:, 0]],
                        dtype=torch.bool)
    # 双臂各 1 球, 每球可能对 1-2 张桌建行
    assert int(mask.sum()) in (2, 4), f"结构行数异常: {int(mask.sum())}"
    _STRUCT_CACHE[key] = mask
    return mask


def class_floors_v7(p: RealGeometryProvider, q: dict) -> dict:
    """cross/self_F/self_U/table 四桶最小 margin (N,) -- Isaac 冒烟同口径.

    附加两桶(出生位重搜口径, S4):
      table_ex     = 剔除结构性肩领悬停行后的桌通道(位形可优化部分);
      table_struct = 仅结构行(应恒 ~+1.7mm, 用于确认没被位形恶化)。
    """
    m = p.sph
    centers = p.fk_all(q)["centers"]
    r = m.radii

    def pair_min(pairs, sel=None):
        if sel is not None:
            pairs = pairs[sel]
        d = (centers[:, pairs[:, 0]] - centers[:, pairs[:, 1]]).norm(dim=-1)
        return (d - (r[pairs[:, 0]] + r[pairs[:, 1]])).amin(dim=1)

    is_f = m.self_robot == 0
    tm = table_margins(p, centers)
    struct = structural_table_mask(p)
    return {"cross": pair_min(m.pairs_cross),
            "self_F": pair_min(m.pairs_self, is_f),
            "self_U": pair_min(m.pairs_self, ~is_f),
            "table": tm.amin(dim=1),
            "table_ex": tm[:, ~struct].amin(dim=1),
            "table_struct": tm[:, struct].amin(dim=1)}


def tightest_v7(p: RealGeometryProvider, q: dict, k: int = 12) -> list:
    """全 pair(含桌)最紧 k 行: (margin, 名i, 名j/TABLE)."""
    m = p.sph
    centers = p.fk_all(q)["centers"]
    r, qn = m.radii, m.qualified_names
    rows = []
    for pairs in (m.pairs_cross, m.pairs_self):
        d = (centers[:, pairs[:, 0]] - centers[:, pairs[:, 1]]).norm(dim=-1)
        marg = d - (r[pairs[:, 0]] + r[pairs[:, 1]])
        for idx in marg[0].argsort()[:k]:
            i, j = int(pairs[idx, 0]), int(pairs[idx, 1])
            rows.append((float(marg[0, idx]), qn[i], qn[j]))
    tm = table_margins(p, centers)[0]
    for idx in tm.argsort()[:k]:
        i = int(m.pairs_table[idx, 0])
        rows.append((float(tm[idx]), qn[i], f"TABLE{int(m.pairs_table[idx, 1])}"))
    return sorted(rows)[:k]


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------

def _selftest() -> None:
    torch.manual_seed(0)
    # 1) UR5 ArmKinematics vs URDF 全树 FK: 链帧逐位对表
    p = make_v7_provider(1)
    tree = UrdfTree(URDF_DIR / "ur5_dfx_left_v7.urdf")
    base_pos, base_yaw = p.layout.base_pose("U_L")
    lim = ur5_v7_limits()
    worst = 0.0
    for trial in range(8):
        qa = (torch.rand(6) * 2 - 1) * 1.5 if trial else torch.zeros(6)
        qa = qa.clamp(lim[:, 0], lim[:, 1])
        fko = p.kin["U"].fk(qa.unsqueeze(0), base_pos, base_yaw)
        tfs = tree.link_transforms(
            {n: float(qa[i]) for i, n in enumerate(UR_ARM_JOINTS)},
            _rpy_matrix((0, 0, base_yaw)).to(torch.float64),
            torch.tensor(base_pos, dtype=torch.float64))
        for li, ln in enumerate(UR_CHAIN_LINKS):
            Rr, tr = tfs[ln]
            dev_t = float((fko["t"][0, li + 1] - tr.to(torch.float32)).abs().max())
            dev_r = float((fko["R"][0, li + 1] - Rr.to(torch.float32)).abs().max())
            worst = max(worst, dev_t, dev_r)
    print(f"UR5 chain FK parity(8 组随机位形) 最大偏差 {worst:.2e}")
    assert worst < 5e-6, "UR5 链 FK 与 URDF 不一致"

    # 2) S1 参照位形(v5 F 位 + UR retract, S4 回填前的 yaml 值)四桶
    #    vs S1 已交叉验证的 Isaac 冒烟读数 -- 固定参照, 不随 yaml 回填漂移
    fr3_q = torch.tensor([0.0592, -0.6042, -0.0997, -2.2558,
                          0.3542, 1.4988, 1.2033])
    ur_q = torch.tensor([0.0, -2.2, 1.9, -1.383, -1.57, 0.0])
    q = {"F_L": fr3_q.unsqueeze(0), "F_R": fr3_q.unsqueeze(0),
         "U_L": ur_q.unsqueeze(0), "U_R": ur_q.unsqueeze(0)}
    fl = {k: float(v[0]) for k, v in class_floors_v7(p, q).items()}
    print("静息通道(mm):", {k: round(v * 1000, 1) for k, v in fl.items()})
    ref = {"self_U": 0.0276, "self_F": 0.0231, "table": 0.0017, "cross": 0.763}
    for k, v in ref.items():
        assert abs(fl[k] - v) < 8e-4, f"{k}: {fl[k]:.4f} vs Isaac {v}"
    print("SELFTEST_PASS (与 Isaac 冒烟出生读数四桶对齐)")
    for row in tightest_v7(p, q):
        print(f"  {row[0] * 1000:+8.1f}mm  {row[1]}  <->  {row[2]}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
