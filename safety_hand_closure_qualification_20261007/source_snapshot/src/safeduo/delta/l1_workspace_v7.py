"""v7 reach 盒重标的工作区漫游 L1 流(S4 任务 2,2026-08-19)。

v5 版(delta/l1_workspace.py)的两处 v7 失配,本模块以纯新增方式修复
(红线:不改 v5/v6 任何行为,故不动原模块的常量与默认值):

1. reach 常量是 FR3+JAKA Zu7 标定(FULL 0.855/0.819, PRACTICAL 0.75/0.72);
   v7 U 臂换 UR5(全伸 0.850),且排距 1.10 -> 1.4964m 后工作区 AABB 本身
   就是共享带(x +-0.20),基座到带内任意点 0.55~0.95m -- v5 的 practical
   0.75/0.72 只覆盖入带可达云的 26%/33%,法兰侵入深度到不了中线。
2. EnvEEBackend 的 U 链是 UR5e 解析参数(d1=0.1625 等),v7 实机是 UR5
   (d1=0.089159);法兰差 ~7cm 级,雅可比与标定残差都被污染。

标定依据(delta/l1_reach_calib_v7.py,40 万均匀限位 FK 采样/臂型,
与 scene_layout_v7 工作区 AABB 求交,z 底抬桌面+0.05):

  practical 半径选 0.80(F 全伸 85.5% -> 93.5%;U 85.0% -> 94.1%):
    r<=0.75 时 F/U 法兰 x 侵入仅到 +0.026/-0.020(过不了中线,双排法兰
    盒无交);r<=0.80 时到 -0.014/+0.016(中线两侧各 ~15mm,加上 F2/DFX
    手沿手轴再伸 ~0.19m,手球交互覆盖整条 0.40m 带宽);再往上是全伸
    奇异区(留 50-55mm 半径余量防怼)。
  r<=0.80 入带云 1%/99% 分位盒(世界系):
    F_L x[-0.014,0.199] y[-0.791,-0.015] z[0.856,1.338](F_R y 镜像)
    U_L x[-0.199,0.018] y[-0.042,0.789] z[0.854,1.327](U_R y 镜像)
  由此取常量:X_PEN=0.015(对称化,F 让 1mm 由径向钳位兜底);
  y 用「离中线 away / 向中线 toward」半宽(F 0.30/0.47, U 0.35/0.48,
  同排双臂天然镜像);z 顶 1.33(分位盒上缘,再高 r>0.80 采不到)。

盒结构性质:四盒 x 全落在共享带内(F [-0.015,0.20] / U [-0.20,0.015]),
waypoint 追踪期 EE 恒在带内;对角双臂(F_L/U_R 同 y<0 半带,F_R/U_L 同
y>0 半带)y 区间大幅重叠 = 跨排冲突主舞台。径向钳位(继承
WorkspaceRoamMapper)保证任何盒角都被拉回 practical 球内,不产生
全伸怼墙 waypoint。

消费:
- 评测/验证:WorkspaceRoamDeltaV7(独立 l1_ws 源,envs/v7_l1_roam_smoke
  用它跑 300 步 raw 覆盖统计);
- R15 训练接线(尚未接,主线做):ConflictMixSource 需要
  env_yaml="duo_env_v7.yaml" + 本模块的 mapper 工厂
  make_roam_mapper_v7(backend);duo_env 侧 ConflictMixSource(...) 目前
  不传 env_yaml(默认 duo_env.yaml = v5 场景真值),v7 训练前必须补。
"""

from __future__ import annotations

import torch

from safeduo.baselines.real_geometry import ArmKinematics
from safeduo.delta._contract_stub import ARM_KEYS
from safeduo.delta.l1_random import L1Params
from safeduo.delta.l1_workspace import (
    Z_TABLE_MARGIN,
    WorkspaceRoamDelta,
    WorkspaceRoamMapper,
)
from safeduo.delta.l2_env_source import EnvEEBackend, RealScenePoses
from safeduo.envs.v7_offline_geometry import ur5_v7_joint_table

V7_FULL_REACH = {"F": 0.855, "U": 0.850}      # FR3 / UR5 数据手册全伸
V7_PRACTICAL_REACH = {"F": 0.80, "U": 0.80}   # 标定选型(见模块头)
V7_X_PEN = 0.015                              # 法兰过中线侵入半宽
V7_DY = {"F": (0.30, 0.47), "U": (0.35, 0.48)}  # (离中线, 向中线) y 半宽
V7_Z_HI = 1.33                                # r<=0.80 入带云 99% 上缘

# v7 漫游档位(服务器 2env 扫描 2026-08-19, artifacts/report_20260819/
# s4_l1_roam_v7):v5 野档(ou_mix 0.5/seg 0.8-2.0/ee 0.25)在 v7 长转运
# 几何下段未到带先换 waypoint,30s 带内驻留仅 ~5%;本档 30s 驻留
# 24-33%/臂、四臂同带 16%,EE 侵入到中线(x +-0.02)。
V7_ROAM_PROFILE = {"ou_mix": 0.2, "seg_dur": (2.0, 5.0), "pause_prob": 0.0,
                   "ou_sigma": 0.02, "ee_speed": 0.45}


def v7_roam_params(amp_max: float = 0.03) -> "L1Params":
    """v7 工作区漫游的 L1Params 出厂档(ee_speed 另传给 mapper)."""
    p = L1Params(amp_max=float(amp_max))
    p.ou_mix = V7_ROAM_PROFILE["ou_mix"]
    p.seg_dur = V7_ROAM_PROFILE["seg_dur"]
    p.pause_prob = V7_ROAM_PROFILE["pause_prob"]
    p.ou_sigma = V7_ROAM_PROFILE["ou_sigma"]
    return p


def roam_boxes_v7(poses: RealScenePoses) -> dict:
    """逐臂 waypoint 盒 = FK 可达域(r<=practical)∩工作区 AABB 的分位盒。

    返回 arm -> (lo(3), hi(3)) float32;数字推导见模块头的标定段。
    """
    lo, hi = poses.ws_lo, poses.ws_hi
    z_lo = max(lo[2], poses.table_top_z + Z_TABLE_MARGIN)
    z_hi = min(hi[2], V7_Z_HI)
    boxes = {}
    for arm in ARM_KEYS:
        bx, by = poses.base_pos[arm][0], poses.base_pos[arm][1]
        away, toward = V7_DY[arm[0]]
        if bx > 0:                             # F 排(基座 +x, 朝 -x 伸)
            x_lo, x_hi = max(lo[0], -V7_X_PEN), hi[0]
        else:                                  # U 排
            x_lo, x_hi = lo[0], min(hi[0], V7_X_PEN)
        if by < 0:                             # 本臂在 y<0 半带
            y_lo, y_hi = max(lo[1], by - away), min(hi[1], by + toward)
        else:
            y_lo, y_hi = max(lo[1], by - toward), min(hi[1], by + away)
        boxes[arm] = (
            torch.tensor([x_lo, y_lo, z_lo], dtype=torch.float32),
            torch.tensor([x_hi, y_hi, z_hi], dtype=torch.float32),
        )
    return boxes


class EnvEEBackendV7(EnvEEBackend):
    """EnvEEBackend 的 v7 变体:U 链换 URDF 现场提取的 UR5 运动学。

    FR3 路径与内部 yaw 自标定机制原样继承(URDF 折叠已含 base_link->
    base_link_inertia 的 pi,标定应选 yaw=0;机制保留作为漂移哨兵)。
    """

    def __init__(self, poses: RealScenePoses,
                 device: "str | torch.device" = "cpu"):
        super().__init__(poses, device=device)
        joints, links = ur5_v7_joint_table()
        self._mk_kin["U"] = lambda yaw: ArmKinematics(
            joints, links, False, ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            yaw, device=self.device)
        self.kin["U"] = self._mk_kin["U"](0.0)


def make_roam_mapper_v7(backend: EnvEEBackendV7,
                        ee_speed: float = 0.25) -> WorkspaceRoamMapper:
    """R15 训练接线用的 mapper 工厂(ConflictMixSource 共享 backend 场景)。"""
    return WorkspaceRoamMapper(backend, ee_speed=ee_speed,
                               boxes=roam_boxes_v7(backend.poses),
                               reach=V7_PRACTICAL_REACH)


class WorkspaceRoamDeltaV7(WorkspaceRoamDelta):
    """独立 v7 工作区漫游 L1 源(自持 UR5 运动学 backend + v7 盒)。"""

    def __init__(self, n_envs: int, params: "L1Params | None" = None,
                 device: "str | torch.device" = "cpu",
                 env_yaml: str = "duo_env_v7.yaml",
                 ee_speed: float = 0.25,
                 arm_keys: tuple = ARM_KEYS,
                 dof_of: "dict | None" = None):
        backend = EnvEEBackendV7(RealScenePoses(env_yaml), device=device)
        super().__init__(n_envs, params=params, device=device,
                         backend=backend,
                         mapper=make_roam_mapper_v7(backend, ee_speed),
                         arm_keys=arm_keys, dof_of=dof_of)


# --------------------------------------------------------------------------
# R25 l1_full 全域盒(2026-08-28,纯新增——上方 l1_ws 常量/盒一字不动,
# battery8 等历史评测的回归锚不受影响;消费方 delta/l1_coverage.py)
# --------------------------------------------------------------------------

# own-side 盒沿越过基座平面的深度(m):审计盲区 |x|∈[0.2,0.95],
# 0.95 ≈ base_x 0.7482 + 0.20——覆盖自家桌面上方的同排自碰/对桌流量。
V7_OWN_BACK = 0.20


def full_boxes_v7(poses: RealScenePoses, r_margin: float = 0.01) -> dict:
    """R25 l1_full 逐臂全域 waypoint 盒(own-side 半区+深互穿+全伸壳层)。

    roam_boxes_v7 把 waypoint 钉在中线共享带(x 宽 0.215 m,体积≈单臂可达
    空间 8-10%)——审计定性"原地转手腕"的根因之一。本盒改为 reach 推导,
    不再被工作区 AABB 的 x ±0.20 窄条约束:

      x  跨机侧到 base_x ∓ (r_full - r_margin)(F 深至 ≈-0.097,越过
         X_PEN=0.015 一个数量级,补"深度互穿"盲区);own 侧到
         base_x ± V7_OWN_BACK(≈±0.95,补 own-side 半区盲区);
      y  工作区 AABB 全宽 ±0.80(四臂共用 -> 同排双臂 y 区间完全重叠,
         制造同排自碰流量);
      z  桌面 +0.05 到 AABB 顶 1.40(l1_ws 的 1.33 分位盖之上再放开)。

    盒角可能超出可达球:消费方采样后按 (r_full - r_margin) 对基座径向
    钳位、方向保留——出界角点恰好被拉到全伸壳层 r∈[0.80,0.855)(又一个
    审计盲区),不是浪费样本。r_margin 默认 0.01 = 全伸奇异面内缩 1 cm,
    比 l1_ws 的 practical 0.80(内缩 5-5.5 cm)激进得多,靠 l1_full 引擎的
    seg_timeout 兜底不可达驻留。
    """
    lo, hi = poses.ws_lo, poses.ws_hi
    z_lo = max(lo[2], poses.table_top_z + Z_TABLE_MARGIN)
    boxes = {}
    for arm in ARM_KEYS:
        bx = poses.base_pos[arm][0]
        r = V7_FULL_REACH[arm[0]] - r_margin
        if bx > 0:                             # F 排(基座 +x,朝 -x 伸)
            x_lo, x_hi = bx - r, bx + V7_OWN_BACK
        else:                                  # U 排
            x_lo, x_hi = bx - V7_OWN_BACK, bx + r
        boxes[arm] = (
            torch.tensor([x_lo, lo[1], z_lo], dtype=torch.float32),
            torch.tensor([x_hi, hi[1], hi[2]], dtype=torch.float32),
        )
    return boxes
