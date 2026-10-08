"""组合 USD 生成器（A6-W6，B 七步换装清单第 4 步；ASSEMBLY_V3 §3）。

每次调用产出一个组合臂 USD：臂 + 法兰垫块（Ø63x17 圆柱，attach link 子碰撞体，
不新增 body/joint——与 ASSEMBLY 的 FixedJoint 方案物理等价的实现简化）+ RH56F2
手（reference 进同一 stage，删其自带 ArticulationRootAPI，FixedJoint 焊到臂
法兰 +0.017m）。fr3 侧同时删自带夹爪子树（fr3_hand/双 finger/tcp 及其关节）
—— W2.5 的 96kN 焊接伪力与 link8/tcp 非法惯量警告同根消失。

硬要求（ASSEMBLY §3）：组合后整树唯一 ArticulationRootAPI，脚本落盘前断言。
产物验证（G 门纪律：看内容不看文件大小）：落盘后重开 stage 打印 body/joint
清单 JSON（COMPOSE_REPORT 行），调用方核对 link 名与球 YAML 一致。

用法（服务器，kit 环境）：
  source ~/safeduo_setup/env.sh && cd ~/safeduo && \
  PYTHONPATH=src python -m safeduo.envs.compose_robot_usd \
    --arm fr3 --hand_usd ~/safeduo/assets_real/usd/rh56f2_left.usd \
    --out ~/safeduo/assets_real/usd/combined/fr3_f2_left.usd --headless --device cpu
  （--arm jaka 时默认 arm_usd = assets_real/usd/jaka_zu7.usd，attach=link6）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--arm", choices=["fr3", "jaka"], required=True)
parser.add_argument("--arm_usd", type=str, default="")
parser.add_argument("--hand_usd", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--flange_h", type=float, default=0.017)
parser.add_argument("--flange_r", type=float, default=0.0315)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

_FR3_GRIPPER = ("fr3_hand", "fr3_leftfinger", "fr3_rightfinger", "fr3_hand_tcp")
_ATTACH = {"fr3": "fr3_link8", "jaka": "link6"}
_DEFAULT_ARM_USD = {
    "fr3": f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaFR3/fr3.usd",
    "jaka": str(Path.home() / "safeduo/assets_real/usd/jaka_zu7.usd"),
}


def find_prim(stage, name, under="/"):
    for prim in stage.Traverse():
        if prim.GetName() == name and str(prim.GetPath()).startswith(under):
            return prim
    return None


def main() -> None:
    arm_usd = args.arm_usd or _DEFAULT_ARM_USD[args.arm]
    hand_usd = str(Path(args.hand_usd).expanduser())
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    robot = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    stage.SetDefaultPrim(robot)
    robot.GetReferences().AddReference(arm_usd)

    # ---- fr3：删自带夹爪子树 + 悬空关节 ----
    if args.arm == "fr3":
        doomed_paths = []
        for prim in stage.Traverse():
            if prim.GetName() in _FR3_GRIPPER:
                doomed_paths.append(str(prim.GetPath()))
        joints_doomed = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint):
                continue
            j = UsdPhysics.Joint(prim)
            tgts = [str(t) for rel in (j.GetBody0Rel(), j.GetBody1Rel())
                    for t in rel.GetTargets()]
            if any(any(t == d or t.startswith(d + "/") for d in doomed_paths)
                   for t in tgts):
                joints_doomed.append(str(prim.GetPath()))
        for p in set(doomed_paths + joints_doomed):
            prim = stage.GetPrimAtPath(p)
            if prim:
                prim.SetActive(False)
        print(f"COMPOSE deactivated gripper prims: {sorted(set(doomed_paths))}"
              f" joints: {sorted(set(joints_doomed))}", flush=True)

    attach = find_prim(stage, _ATTACH[args.arm])
    assert attach is not None, f"attach link {_ATTACH[args.arm]} not found"

    # ---- 法兰垫块：attach link 的子碰撞体（不新增 body）----
    spacer = UsdGeom.Cylinder.Define(
        stage, attach.GetPath().AppendChild("flange_spacer"))
    spacer.GetAxisAttr().Set(UsdGeom.Tokens.z)
    spacer.GetHeightAttr().Set(args.flange_h)
    spacer.GetRadiusAttr().Set(args.flange_r)
    spacer.GetExtentAttr().Set(
        [Gf.Vec3f(-args.flange_r, -args.flange_r, -args.flange_h / 2),
         Gf.Vec3f(args.flange_r, args.flange_r, args.flange_h / 2)])
    UsdGeom.XformCommonAPI(spacer.GetPrim()).SetTranslate(
        Gf.Vec3d(0.0, 0.0, args.flange_h / 2))
    UsdPhysics.CollisionAPI.Apply(spacer.GetPrim())

    # ---- 手 reference + 位姿（rest pose 与法兰面重合，FixedJoint 出生即满足）----
    hand_root = UsdGeom.Xform.Define(stage, "/Robot/f2_hand").GetPrim()
    hand_root.GetReferences().AddReference(hand_usd)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    attach_w = cache.GetLocalToWorldTransform(attach)
    mount_w = Gf.Matrix4d().SetTranslate(
        Gf.Vec3d(0.0, 0.0, args.flange_h)) * attach_w
    UsdGeom.Xformable(hand_root).ClearXformOpOrder()
    UsdGeom.Xformable(hand_root).AddTransformOp().Set(mount_w)

    # ---- 删手自带 articulation root（整树唯一 root 硬要求）----
    removed_api = []
    for prim in Usd.PrimRange(hand_root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            prim.RemoveAppliedSchema("PhysxArticulationAPI")
            removed_api.append(str(prim.GetPath()))
    print(f"COMPOSE removed hand articulation roots: {removed_api}", flush=True)

    # ---- 手 base body 定位 + FixedJoint 焊接 ----
    side = "left" if "left" in Path(hand_usd).stem else "right"
    base_name = f"{side}_hand_base"
    hand_base = find_prim(stage, base_name, under="/Robot/f2_hand")
    assert hand_base is not None, f"{base_name} not found in hand usd"
    joint = UsdPhysics.FixedJoint.Define(stage, "/Robot/f2_mount_joint")
    joint.GetBody0Rel().SetTargets([attach.GetPath()])
    joint.GetBody1Rel().SetTargets([hand_base.GetPath()])
    joint.GetLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, args.flange_h))
    joint.GetLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.GetLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # ---- JAKA 自碰物理配套：结构性贴邻对写 FilteredPairsAPI（A7-W7）----
    # Round-110 裁定②在 env cfg 开 enabled_self_collisions=True；A6-W6 实测
    # link3|link5、link4|link6 网格结构性贴邻（任意位形不可分离）——不过滤
    # 则自碰开启后成为永久接触力源（fr3_hand 96kN 焊接伪力的同族事故）。
    # link5|hand_base 同判：紧凑腕 + 17mm 垫块使手底座网格常压 link5，静止
    # 探针实测恒 ~6.5kN（A7 /tmp/a7_body_force_probe.py，link3 不热证明
    # FilteredPairsAPI 生效路径正确）；PhysX 只自动过滤直接成关节的 body 对，
    # 焊接链隔 2 跳（link5-j6-link6-weld-hand）不在其内。三对在语义层均已
    # 豁免（adjacency + hand_wrist），过滤零信息损失。
    filtered = []
    if args.arm == "jaka":
        pairs = [("link3", "link5"), ("link4", "link6"),
                 ("link5", base_name)]
        for a, b in pairs:
            pa, pb = find_prim(stage, a), find_prim(stage, b)
            assert pa is not None and pb is not None, f"filter pair {a}|{b} 缺 prim"
            rel = UsdPhysics.FilteredPairsAPI.Apply(pa).CreateFilteredPairsRel()
            rel.AddTarget(pb.GetPath())
            filtered.append(f"{a}|{b}")
        print(f"COMPOSE filtered_pairs: {filtered}", flush=True)

    # ---- 单 root 断言 + 落盘 ----
    roots = [str(p.GetPath()) for p in stage.Traverse()
             if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert len(roots) == 1, f"articulation root must be unique, got {roots}"
    stage.GetRootLayer().Save()

    # ---- 产物内容验证（重开 stage 盘点）----
    st2 = Usd.Stage.Open(str(out))
    bodies = [p.GetName() for p in st2.Traverse()
              if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.IsActive()]
    joints = [p.GetName() for p in st2.Traverse() if p.IsA(UsdPhysics.Joint)
              and p.IsActive() and not p.IsA(UsdPhysics.FixedJoint)]
    report = {"out": str(out), "arm": args.arm, "hand": Path(hand_usd).stem,
              "articulation_roots": roots, "n_bodies": len(bodies),
              "bodies": bodies, "n_moving_joints": len(joints),
              "moving_joints": joints, "filtered_pairs": filtered}
    print("COMPOSE_REPORT " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
    import os

    os._exit(0)
