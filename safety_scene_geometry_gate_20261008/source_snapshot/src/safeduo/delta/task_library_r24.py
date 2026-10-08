"""R24 任务库扩充（2026-08-28）：四臂全动的新任务族（刚体代理版）。

owner 诉求（MASTER_REPORT §9 R24）：轨迹太少、只有 grasp/pick_place 可看；
要多样化、要有"可能碰撞"的双臂/四臂操作、要对齐真机 demo（套枕套/抓枕头/
叠衣服），且非协同任务里另外两条臂也不能闲着。

本模块提供四个新任务族（全部四臂有活、有意设计臂间近距交互走廊；安全性
仍由逐步 sphere margin>0 硬门槛裁决 —— 走廊是"近"，不是"撞"）：

  four_lift       两台 Franka 抬一根长杆（bar_main，44 cm，横在两臂之间），
                  同时两台 JAKA 各自搬运自己的小 cube 汇入同一片中线空域
                  （交叉走廊 = U 双臂送货动线从杆的悬停区正下方/侧方穿过）
  relay_chain     四臂接力 F_L→U_R→U_L→F_R 传一只 cube（三次交接：对角、
                  JAKA 排内、对角 —— 链条顺序按 v7 几何调整过：owner 提议的
                  F_L→U_L 直传对角距离 0.87 m 超出双臂可达半径，改为相邻优
                  先的 S 形链，cube 走遍四个象限）
  dual_pick_swap  左右两对臂各自 pick_place 且落点互换（F 对在 F 桌交叉、
                  U 对在 U 桌交叉，搬运走廊必然跨越中线 —— 高冲突密度素材；
                  用 z 分层 + 时间错相把"必然交叉"控制在审计红线之上）
                  ⚠ 状态（2026-08-28 04:15）：F 对互换全通过；U 排跨线收进
                  是 DLS 贪婪下降的刀刃问题 —— 40+ 组探针（中停网格/抓取
                  候选/锁姿组合/腕部预开剂量/泊位降层/本侧梭运）证明 UR5/
                  JAKA 腕折盆地包围目标区：任一臂的位形微扰都会把另一臂的
                  下降路径挤进 self<2mm 折叠分支（例：j4 预开 -0.15 让
                  cross_a2 从 3.5mm 升到全通过，但残留分支让 pull_ul1 折；
                  反向还原又让 cross_b 折）。构建器对该族会 raise —— 属
                  预期。根治需换关节空间样条/cuRobo 规划 U 排跨线段，列为
                  R24 后续项；其余三族已过全部硬门槛可先投产。
  pillow_sheath   真机"四臂套枕套"的刚体代理：F 双臂合抬枕头代理（大扁盒）
                  悬停展示，U 双臂轮流做"套枕套"动线（从两端向中心的裹套
                  扫掠，近距但不接触）。布料版路线图见 R24 设计文档。

与 R23 底座的关系：完全复用 task_record_s9 的相位/IK/审计/npz 管线
（design_skill_v7 + compose_skill + validate_trajectory + audit_v7_channels），
只是任务规格从"单物体单主角"扩展为"多物体多主角"。多物体经由
duo_env_v7_r24_objects.yaml（新配置，不动老 duo_env_v7_objects.yaml ——
它被 17 项旧测试钉死为单 cube）。

新审计（R24 验收项）：audit_arm_participation —— 逐臂 EE 行程与 max|Δq|，
四臂都必须"真的在动"（EE 行程 >= MIN_EE_TRAVEL_M 且 Δq>0），写进构建报告
并作为硬门槛之一（不过 = 该任务构建失败）。
"""

from __future__ import annotations

import math

import numpy as np
import torch

from safeduo.delta.grasp_gen import GraspGenParams, box_antipodal_grasps
from safeduo.delta.skill_record import Phase, SkillSpec
from safeduo.safety.types import ARM_KEYS

# 延迟不了：S9Task/常量在 task_record_s9，方向是 r24 -> s9（s9 只在函数内
# 反向 import 本模块，无环）。
from safeduo.delta.task_record_s9 import (
    CUBE_NAME,
    CUBE_POS,
    CUBE_SIZE,
    GRASP_PARAMS,
    S9Task,
)

# ---------------------------------------------------------------------------
# R24 物体清单（单一来源；duo_env_v7_r24_objects.yaml 被测试钉到这些数字）
# ---------------------------------------------------------------------------

R24_ENV_YAML = "duo_env_v7_r24_objects.yaml"

# 长杆：横在两台 Franka 之间（长轴沿 y），两端留 4 cm、抓握站位 y=±0.18
# —— F 双臂 flange 间距 0.36 m，落在 dual_carry_both 审计过的 0.40 带附近
BAR_NAME, BAR_SIZE, BAR_POS = "bar_main", (0.05, 0.44, 0.05), (0.53, 0.0, 0.825)
BAR_GRIP_Y = 0.18

# F_R 的镜像 cube（dual_pick_swap 用；与 cube_main 镜像对称）
CUBE_FR_NAME, CUBE_FR_POS = "cube_f_r", (0.54, 0.22, 0.825)

# 两台 JAKA 各自的小 cube（four_lift 的交叉走廊货物 / swap 的 U 侧棋子）
CUBE_UL_NAME, CUBE_UL_POS = "cube_u_l", (-0.44, 0.42, 0.825)
CUBE_UR_NAME, CUBE_UR_POS = "cube_u_r", (-0.46, -0.20, 0.825)

# 枕头代理：大扁盒（0.10 x 0.34 x 0.12），抓握站位 y=±0.13（顶抓两端）
PILLOW_NAME, PILLOW_SIZE, PILLOW_POS = \
    "pillow_main", (0.10, 0.34, 0.12), (0.52, 0.0, 0.86)
PILLOW_GRIP_Y = 0.13

# U 臂（JAKA+DFX 手）抓取参数：TCP 在指尖捏合区 z~0.17（a22 录制端 CLI
# 默认同源）。min_flange_z=1.045 由构建探针实测反推：13 族 U_grab 用过
# 1.035，但那是 seam 远距（平面 0.33+）；R24 的 U cube 更靠基座（平面
# 0.29-0.37），近距时 DFX 腕俯仰使手垂得更深，1.036 在 IK table 地板
# (0.0010) 上被卡（实测 0.0006-0.0007）。抬到 1.045 后 TCP z=0.875，
# 距 cube 心 0.050 —— 录制端 0.12 TCP 门内，指尖仍包住 cube 上半段。
U_GRASP_PARAMS = GraspGenParams(tcp_offset=(0.0, 0.0, 0.17),
                                min_flange_z=1.045)

R24_OBJECTS = {
    CUBE_NAME: {"name": CUBE_NAME, "size": CUBE_SIZE, "init_pos": list(CUBE_POS)},
    BAR_NAME: {"name": BAR_NAME, "size": list(BAR_SIZE), "init_pos": list(BAR_POS)},
    CUBE_FR_NAME: {"name": CUBE_FR_NAME, "size": CUBE_SIZE,
                   "init_pos": list(CUBE_FR_POS)},
    CUBE_UL_NAME: {"name": CUBE_UL_NAME, "size": CUBE_SIZE,
                   "init_pos": list(CUBE_UL_POS)},
    CUBE_UR_NAME: {"name": CUBE_UR_NAME, "size": CUBE_SIZE,
                   "init_pos": list(CUBE_UR_POS)},
    PILLOW_NAME: {"name": PILLOW_NAME, "size": list(PILLOW_SIZE),
                  "init_pos": list(PILLOW_POS)},
}

R24_FAMILIES = ("four_lift", "relay_chain", "dual_pick_swap", "pillow_sheath")

# 四臂参与度硬门槛：每臂 EE 行程下限（m）。0.30 = 一次像样的预位/巡检动线
# 的量级；纯"呼吸抖动"(~0.02 m) 与完全静止都过不了 —— R24 立项验收项。
MIN_EE_TRAVEL_M = 0.30


# ---------------------------------------------------------------------------
# 抓取候选构造（在 grasp_gen 解析管线之上按物体/臂定制）
# ---------------------------------------------------------------------------


def _arm_prior(provider, arm: str) -> tuple:
    """(ref_R, base_xy)：该臂出生腕姿态 + 基座平面位置（自然度先验）。"""
    if provider is None:
        return None, None
    fko = provider.fk_all(provider.default_q())
    return (fko[arm]["R_flange"][0].tolist(),
            provider.layout.base_pose(arm)[0][:2])


def cube_cands(provider, arm: str, pos, params: GraspGenParams) -> list:
    """单个 5 cm cube 的候选（顶抓优先，按该臂自然度排序）。"""
    ref_R, base_xy = _arm_prior(provider, arm)
    return box_antipodal_grasps(pos, CUBE_SIZE, params,
                                ref_R=ref_R, base_xy=base_xy)


def grip_station_cands(provider, arm: str, station_pos, cross_size,
                       params: GraspGenParams) -> list:
    """长物体"抓握站位"的候选：把站位当成一个虚拟小盒（尺寸 = 该处横截
    面），只保留顶抓（闭合线必须横跨截面短边 —— 沿长轴闭合等于要求手指
    张开 0.34-0.44 m，物理上不存在）。"""
    ref_R, base_xy = _arm_prior(provider, arm)
    cands = box_antipodal_grasps(station_pos, cross_size, params,
                                 ref_R=ref_R, base_xy=base_xy)
    # 闭合线沿世界 x（跨过长轴 y 的短截面）的顶抓才是物理可行解
    return [c for c in cands
            if c.kind == "top" and abs(c.closing[0]) > 0.9]


# ---------------------------------------------------------------------------
# 相位构造小工具
# ---------------------------------------------------------------------------


def _mode(targets: dict, lerps: dict = ()) -> dict:
    m = {a: "hold" for a in ARM_KEYS}
    for a in targets:
        m[a] = "plan"
    for a in lerps:
        m[a] = "lerp"
    return m


def _ph(name, dur, targets=None, oris=None, lerps=None) -> Phase:
    """带 ori 的多臂相位速记。targets: arm->xyz；oris: arm->R (3x3)。"""
    targets = dict(targets or {})
    lerps = dict(lerps or {})
    return Phase(name, dur, targets=targets,
                 joint_moves=lerps,
                 mode=_mode(targets, lerps),
                 ori_targets=dict(oris or {}))


def _above(p, dz) -> tuple:
    return (p[0], p[1], p[2] + dz)


_BREATH = {1: 0.04, 3: -0.03}      # 静默臂的呼吸偏置（关节空间 lerp）


# ---------------------------------------------------------------------------
# 任务族 1: four_lift —— F 双臂抬杆 + U 双臂交叉送货
# ---------------------------------------------------------------------------


def _four_lift_task(cands: dict, u_ori: bool = True) -> S9Task:
    """cands: {"F_L","F_R","U_L","U_R"} -> GraspCandidate。

    走廊设计：F 双臂抓杆两端 -> 同步抬升 -> 平移到中线悬停带 (x 0.37) ->
    放下；U 双臂同拍抓自己的 cube -> 提起 -> 送进中线空域 (x -0.05 带，
    与悬停杆水平间隙 ~0.42 m) -> 轮流进中心（时间错相的 y 交叉）-> 放下
    回撤。四臂全程有活，交叉集中在 carry/weave 两相位。"""
    fl, fr = cands["F_L"], cands["F_R"]
    ul, ur = cands["U_L"], cands["U_R"]
    ori_f = {"F_L": fl.R_flange, "F_R": fr.R_flange}
    ori_u = {"U_L": ul.R_flange, "U_R": ur.R_flange} if u_ori else {}

    fgl, fgr = fl.flange_grasp, fr.flange_grasp
    ugl, ugr = ul.flange_grasp, ur.flange_grasp
    lift_f = 0.15                     # 杆提升高度
    lift_u = 0.11                     # cube 提升高度
    # 杆的搬运终点（seam 悬停带）：x 0.53 -> 0.37，落点仍在 F 桌半区
    bar_dx = -0.16

    # U cube 落点（送完回自己半区放下）。z 比抓取 flange 高 1.5 cm：place
    # 下探路径上 DFX 手会比抓取时垂得更深（腕位形不同），贴着抓取高度放
    # 会蹭 IK table 地板（实测 0.8 mm < 1.0 mm）；抬高后 cube 释放时离桌
    # 面 ~1 cm，自由落体收尾，视觉无碍
    place_ul = (-0.30, 0.28, ugl[2] + 0.015)
    place_ur = (-0.32, -0.28, ugr[2] + 0.015)

    phases = [
        _ph("approach", 3.0,
            targets={"F_L": fl.flange_pre, "F_R": fr.flange_pre,
                     "U_L": ul.flange_pre, "U_R": ur.flange_pre},
            oris={**ori_f, **ori_u}),
        _ph("descend", 1.5,
            targets={"F_L": fgl, "F_R": fgr, "U_L": ugl, "U_R": ugr},
            oris={**ori_f, **ori_u}),
        _ph("close", 1.2),
        _ph("lift", 1.5,
            targets={"F_L": _above(fgl, lift_f), "F_R": _above(fgr, lift_f),
                     "U_L": _above(ugl, lift_u), "U_R": _above(ugr, lift_u)},
            oris={**ori_f, **ori_u}),
        # 杆向中线平移；U 双臂把 cube 送进中线空域（x -0.10/-0.12 两条
        # x 车道，平面伸展 <=0.74 —— JAKA 高位可达域实测边界内），与杆的
        # 悬停带水平间隙 ~0.47 m —— 同一片空域、近而不撞
        _ph("carry", 3.0,
            targets={"F_L": (fgl[0] + bar_dx, fgl[1], fgl[2] + lift_f),
                     "F_R": (fgr[0] + bar_dx, fgr[1], fgr[2] + lift_f),
                     "U_L": (-0.10, 0.26, 1.12),
                     "U_R": (-0.12, -0.26, 1.12)},
            oris=dict(ori_f)),
        # 轮流进中心（sequential_center_grab 的时间切片纪律）：U_L 先探
        # 到 y+0.08（贴近 U_R 走廊的中线），F 双臂原地稳杆
        _ph("weave_l", 2.0,
            targets={"U_L": (-0.10, 0.08, 1.13)},
            lerps={"F_L": dict(_BREATH)}),
        _ph("weave_r", 2.0,
            targets={"U_L": (-0.10, 0.30, 1.12),
                     "U_R": (-0.12, -0.04, 1.13)}),
        # 杆放回桌面（x 0.37 带），U 双臂各自回撤到落点上方
        _ph("setdown", 2.0,
            targets={"F_L": (fgl[0] + bar_dx, fgl[1], fgl[2]),
                     "F_R": (fgr[0] + bar_dx, fgr[1], fgr[2]),
                     "U_L": _above(place_ul, 0.10),
                     "U_R": (-0.32, -0.28, ugr[2] + 0.10)},
            oris=dict(ori_f)),
        _ph("place_u", 1.5,
            targets={"U_L": place_ul, "U_R": place_ur},
            oris=dict(ori_u) if u_ori else None),
        _ph("release", 1.0),
        _ph("retreat", 2.5,
            targets={"F_L": (0.44, -0.30, 1.25), "F_R": (0.44, 0.30, 1.25),
                     "U_L": (-0.38, 0.42, 1.16), "U_R": (-0.40, -0.36, 1.16)}),
    ]
    # 抓握站位在杆的 ±0.18，离杆心（= 物体 root）0.18+ —— TCP 门按站位
    # 距离放宽到 0.26；cube 类保持默认（不写 gate_tcp）
    return S9Task(
        family="four_lift",
        spec=SkillSpec("s9_four_lift", phases),
        hand_events=[("close", 0.10, a, "close", 0.60) for a in ARM_KEYS] + [
            ("release", 0.30, a, "open", 0.0) for a in ARM_KEYS],
        object_events=[
            ("close", 0.90, "attach", "F_L", None, BAR_NAME, 0.26),
            ("close", 0.90, "attach", "U_L", None, CUBE_UL_NAME, None),
            ("close", 0.90, "attach", "U_R", None, CUBE_UR_NAME, None),
            ("release", 0.20, "release", None, None, BAR_NAME, None),
            ("release", 0.20, "release", None, None, CUBE_UL_NAME, None),
            ("release", 0.20, "release", None, None, CUBE_UR_NAME, None),
        ],
        success={"kind": "pick_place",
                 "place_xy": [BAR_POS[0] + bar_dx, BAR_POS[1]],
                 "radius_m": 0.10, "min_disp_m": 0.12,
                 "rest_z": BAR_POS[2]},
        objects=[BAR_NAME, CUBE_UL_NAME, CUBE_UR_NAME],
        env_yaml=R24_ENV_YAML,
    )


def _four_lift_variants(provider):
    f_l = grip_station_cands(provider, "F_L",
                             (BAR_POS[0], -BAR_GRIP_Y, BAR_POS[2]),
                             BAR_SIZE[0], GRASP_PARAMS)
    f_r = grip_station_cands(provider, "F_R",
                             (BAR_POS[0], BAR_GRIP_Y, BAR_POS[2]),
                             BAR_SIZE[0], GRASP_PARAMS)
    u_l = cube_cands(provider, "U_L", CUBE_UL_POS, U_GRASP_PARAMS)
    u_r = cube_cands(provider, "U_R", CUBE_UR_POS, U_GRASP_PARAMS)
    u_l = [c for c in u_l if c.kind == "top"]
    u_r = [c for c in u_r if c.kind == "top"]
    for k in range(min(2, len(f_l), len(f_r))):
        for j in range(min(2, len(u_l), len(u_r))):
            cands = {"F_L": f_l[k], "F_R": f_r[k], "U_L": u_l[j], "U_R": u_r[j]}
            yield (f"bar_yaw{k}/u_yaw{j}", _four_lift_task(cands))
    # 兜底：U 侧退化为位置-only 下探（ori 不锁，attach 门控仍看 TCP）
    cands = {"F_L": f_l[0], "F_R": f_r[0], "U_L": u_l[0], "U_R": u_r[0]}
    yield ("bar_yaw0/u_pos_only", _four_lift_task(cands, u_ori=False))


# ---------------------------------------------------------------------------
# 任务族 2: relay_chain —— 四臂接力 F_L -> U_R -> U_L -> F_R
# ---------------------------------------------------------------------------


def _relay_chain_task(cand, u_row_gap: float = 0.32) -> S9Task:
    """cand: cube_main 的 F_L 抓取候选。

    三个会合点（全部沿用/镜像已审计的会合几何）：
      meet1  F_L x U_R @ (y -0.40 带, flange gap 0.30) —— S5 审计的 FL_UR 车道
      meet2  U_R x U_L @ (x -0.33, y ±gap/2) —— JAKA 排内对递（新几何，
             gap 默认 0.32 与 v5 腕-腕瓶颈教训一致）
      meet3  U_L x F_R @ (y +0.38 带) —— handover_high 的对角，沿用
             "U 先泊车、F 后进近"的时序纪律（y>0 对角是脆弱侧）
    全程四臂都有戏份：不在交接的臂在做下一站预位或回撤，无静止臂。"""
    fg = cand.flange_grasp
    ori = {"F_L": cand.R_flange}
    lift_z = fg[2] + GRASP_PARAMS.lift_height
    gy = u_row_gap / 2.0

    phases = [
        # F_L 抓取序列；同拍 U_R 向 meet1 预位、U_L/F_R 向各自下一站移动
        _ph("reach", 2.5,
            targets={"F_L": cand.flange_pre, "U_R": (-0.28, -0.32, 1.14),
                     "U_L": (-0.36, 0.30, 1.14), "F_R": (0.50, 0.30, 1.25)},
            oris=dict(ori)),
        _ph("descend", 1.5, targets={"F_L": fg}, oris=dict(ori),
            lerps={"U_R": dict(_BREATH)}),
        _ph("close", 1.2, lerps={"U_L": dict(_BREATH)}),
        _ph("lift", 1.5, targets={"F_L": (fg[0], fg[1], lift_z)},
            oris=dict(ori)),
        # meet1：F_L 带 cube 到 y<0 会合带；U_R 进近（audited FL_UR 几何）
        _ph("meet1", 3.0,
            targets={"F_L": (0.15, -0.40, 1.10), "U_R": (-0.15, -0.33, 1.10)},
            oris=dict(ori),
            lerps={"F_R": dict(_BREATH)}),
        _ph("transfer1", 1.8, lerps={"U_L": dict(_BREATH)}),
        # split1：F_L 空手回撤；U_R 带 cube 驶向 JAKA 排内会合点；U_L 进近
        _ph("split1", 2.5,
            targets={"F_L": (0.44, -0.48, 1.25),
                     "U_R": (-0.33, -gy, 1.10),
                     "U_L": (-0.33, 0.30, 1.12)}),
        _ph("meet2", 2.0,
            targets={"U_L": (-0.33, gy, 1.10)},
            lerps={"F_R": dict(_BREATH)}),
        _ph("transfer2", 1.8, lerps={"F_L": dict(_BREATH)}),
        # split2：U_R 空手回撤；U_L 带 cube 驶向 y>0 会合带并泊车
        _ph("split2", 2.5,
            targets={"U_R": (-0.40, -0.30, 1.14),
                     "U_L": (-0.16, 0.35, 1.12),
                     "F_R": (0.30, 0.40, 1.20)}),
        # meet3：U_L 已泊车，F_R 单臂进近（handover_high 时序纪律）
        _ph("meet3", 2.0,
            targets={"F_R": (0.15, 0.41, 1.12)},
            lerps={"U_R": dict(_BREATH)}),
        _ph("transfer3", 1.8, lerps={"F_L": dict(_BREATH)}),
        _ph("retreat", 2.5,
            targets={"F_R": (0.44, 0.46, 1.25), "U_L": (-0.38, 0.44, 1.14),
                     "F_L": (0.46, -0.42, 1.28), "U_R": (-0.42, -0.34, 1.16)}),
        _ph("settle", 1.0),
    ]
    return S9Task(
        family="relay_chain",
        spec=SkillSpec("s9_relay_chain", phases),
        hand_events=[
            ("close", 0.10, "F_L", "close", 0.60),
            ("transfer1", 0.20, "U_R", "close", 0.60),
            ("transfer1", 1.00, "F_L", "open", 0.0),
            ("transfer2", 0.20, "U_L", "close", 0.60),
            ("transfer2", 1.00, "U_R", "open", 0.0),
            ("transfer3", 0.20, "F_R", "close", 0.60),
            ("transfer3", 1.00, "U_L", "open", 0.0),
        ],
        object_events=[
            ("close", 0.90, "attach", "F_L", None, CUBE_NAME, None),
            ("transfer1", 0.90, "attach", "U_R", (0.18, 0.6), CUBE_NAME, None),
            ("transfer2", 0.90, "attach", "U_L", (0.18, 0.6), CUBE_NAME, None),
            ("transfer3", 0.90, "attach", "F_R", (0.18, 0.6), CUBE_NAME, None),
        ],
        success={"kind": "handover", "giver": "F_L", "receiver": "F_R",
                 "follow_dist_m": 0.45, "min_travel_m": 0.10,
                 "receiver_x_sign": 1.0},
        objects=[CUBE_NAME],
        env_yaml=R24_ENV_YAML,
    )


def _relay_chain_variants(provider):
    from safeduo.delta.task_record_s9 import grasp_candidates_s9

    cands = grasp_candidates_s9(provider)
    for c in cands[:3]:
        yield (c.name, _relay_chain_task(c))
    # 兜底：JAKA 排内会合加宽到 0.36
    yield (f"{cands[0].name}/wide_meet2", _relay_chain_task(cands[0], 0.36))


# ---------------------------------------------------------------------------
# 任务族 3: dual_pick_swap —— 两对臂各自 pick&place 且落点互换
# ---------------------------------------------------------------------------

# 落点表（互换后的目标；与初始位错开 y 符号 —— 搬运走廊必然跨越 y=0 中线）。
# U 落点只过线 0.12：JAKA 载货位形（姿态锁 + 高位巡航）的 3D 可达半径
# ~0.80，过线更深（首版 ∓0.16 → 3D 0.84）会顶到伸展极限腕折
SWAP_PLACE = {
    "F_L": (0.46, 0.14),     # cube_main   (0.54,-0.22) -> y>0 侧
    "F_R": (0.46, -0.14),    # cube_f_r    (0.54, 0.22) -> y<0 侧
    # U 落点都前移到 seam 前带：贴基座排的深跨侧落点（首版 x-0.42）会把
    # JAKA 腕压进折叠分支，放完之后寸步难行（实测任何方向的小步都 self 卡）
    "U_L": (-0.34, -0.12),   # cube_u_l   (-0.44, 0.42) -> y<0 侧
    "U_R": (-0.34, 0.12),    # cube_u_r   (-0.46,-0.20) -> y>0 侧
}


def _dual_pick_swap_task(cands: dict, u_ori: bool = True) -> S9Task:
    """交叉控制：对角批次错相（F_L+U_R 先过线放下，F_R+U_L 后过线），
    搬运高度 z 分层（第一批 F 1.26/U 1.20，第二批 F 1.30/U 1.24 且此时
    第一批已下沉/回撤）——空间上四条走廊都跨中线，时间上两两错开。"""
    fl, fr = cands["F_L"], cands["F_R"]
    ul, ur = cands["U_L"], cands["U_R"]
    ori_f = {"F_L": fl.R_flange, "F_R": fr.R_flange}
    ori_u = {"U_L": ul.R_flange, "U_R": ur.R_flange} if u_ori else {}
    g = {a: cands[a].flange_grasp for a in ARM_KEYS}
    # U 落点比抓取 flange 高 1.5 cm（同 four_lift place_u 的教训：place
    # 下探腕位形不同、DFX 手垂得更深，贴抓取高度放会蹭 IK table 地板）
    pl = {a: (SWAP_PLACE[a][0], SWAP_PLACE[a][1],
              g[a][2] + (0.015 if a.startswith("U") else 0.0))
          for a in ARM_KEYS}

    phases = [
        _ph("approach", 3.0,
            targets={a: cands[a].flange_pre for a in ARM_KEYS},
            oris={**ori_f, **ori_u}),
        _ph("descend", 1.5, targets=dict(g), oris={**ori_f, **ori_u}),
        _ph("close", 1.2),
        _ph("lift", 1.5,
            targets={"F_L": _above(g["F_L"], 0.16), "F_R": _above(g["F_R"], 0.16),
                     "U_L": _above(g["U_L"], 0.12), "U_R": _above(g["U_R"], 0.12)},
            oris={**ori_f, **ori_u}),
        # 第一批过线：F_L / U_R 跨越 y=0 驶向互换落点上方（高车道）；
        # 第二批让位外扩 —— 泊车点取 13 族已审计驻停位（cross_uf 的
        # yield_FR / pre_reach 带；离自己基座平面距离太近会腕折自碰，
        # 首版 (0.60,0.40) 平面距 0.17 被 IK 守卫实测卡死）
        _ph("cross_a", 2.0,
            # U_L 此拍完全驻停（它的跨线弧线对 IK 分支极敏感，任何多臂
            # 耦合解算都可能把它带进折叠分支 —— 拆到 C 批单独走）。
            # U_R 的跨线走两跳弧线（先 seam 前伸再收进落点 —— U_L 探针
            # 配方的镜像；单跳直达的边界分支已被实测证明寸步难行）
            targets={"F_L": _above(pl["F_L"], 0.22),
                     "U_R": (-0.16, -0.04, 1.18),
                     "F_R": (0.50, 0.56, 1.30)},
            oris={"F_L": fl.R_flange, **({"U_R": ur.R_flange} if u_ori else {})},
            lerps={"U_L": dict(_BREATH)}),
        _ph("cross_a2", 1.2,
            # 镜像 U_L 已验证的 pull_ul3 中间跳：先收到 (-0.30,+0.10) 的
            # 中停点再进落点 —— 直接从 seam 前伸点单跳收进落点上方
            # (Δxy≈0.24) 实测把 UR5 腕压进 self 1.4 mm 折叠分支
            targets={"U_R": (-0.30, 0.10, 1.16)},
            oris=dict({"U_R": ur.R_flange} if u_ori else {}),
            lerps={"F_L": dict(_BREATH)}),
        _ph("cross_a3", 1.2,
            targets={"U_R": _above(pl["U_R"], 0.14)},
            oris=dict({"U_R": ur.R_flange} if u_ori else {}),
            lerps={"F_L": dict(_BREATH)}),
        _ph("place_a", 1.5,
            targets={"F_L": pl["F_L"], "U_R": pl["U_R"]},
            oris={"F_L": fl.R_flange, **({"U_R": ur.R_flange} if u_ori else {})}),
        # 松手的同时先竖直提升（v5 教训四：低位伸展姿态直接斜拉回撤会腕折
        # 自碰 —— U_R 首版实测 self 1.9 mm 被卡）；cube 在 +0.20s 就释放，
        # cosine 缓起的前 0.2s 手只抬了 ~1 cm，不影响落放
        _ph("release_a", 1.0,
            # 提升保持锁姿：位置-only 提升会让腕游走离开健康分支，下一跳
            # 立刻折（U_R 实测 self 1.2 mm）
            targets={"F_L": _above(pl["F_L"], 0.16),
                     "U_R": _above(pl["U_R"], 0.12)},
            oris={"F_L": fl.R_flange,
                  **({"U_R": ur.R_flange} if u_ori else {})}),
        # 第二批过线（反方向），第一批同拍抬升回撤 —— 双向对开走廊。
        # Franka 的大横摆全都要分跳 + 锁姿：F_R 载货 park y+0.56 → 落点
        # y-0.14（Δy0.70）与 F_L 空手 y+0.14 → 驻停 y-0.44（Δy0.58）的
        # 单跳都实测 self_F 2.6 mm 腕折（4 通道探针拆出来才看清 ——
        # 全局 self 最小值曾把锅甩给 U_L）。
        # U 排的关节空间没有"跨线后再折返回家"的余地（回程无论低位/高位/
        # 审计驻停位都实测腕折）—— 编排改为"臂也互换泊位"：U_R 放完就近
        # 驻停到 U_L 腾出的 +y 侧，U_L 之后驻停 -y 侧，谁都不折返穿越
        _ph("cross_b", 2.75,
            # U_R 泊位只到 y+0.18：更深（首版 +0.34）意味着 dy 0.78、
            # 平面 0.86 m —— 超出 UR5 可达半径，IK 只能以腕折凑合
            targets={"F_R": (0.50, 0.18, 1.28),
                     "F_L": (0.52, -0.16, 1.30), "U_R": (-0.38, 0.18, 1.16)},
            oris={"F_R": fr.R_flange, "F_L": fl.R_flange,
                  **({"U_R": ur.R_flange} if u_ori else {})}),
        _ph("cross_b2", 1.5,
            # F_L 同拍完成回家第二跳 —— 它的中停点 (0.52,-0.16) 离 F_R
            # 的落点上方只有 6 cm，不让位 F_R 就进不了场
            targets={"F_R": _above(pl["F_R"], 0.24),
                     "F_L": (0.46, -0.44, 1.28)},
            oris={"F_R": fr.R_flange},
            lerps={"U_R": dict(_BREATH)}),
        _ph("place_b", 1.5,
            targets={"F_R": pl["F_R"]},
            oris={"F_R": fr.R_flange},
            lerps={"F_L": dict(_BREATH)}),
        _ph("release_b", 1.0,
            targets={"F_R": _above(pl["F_R"], 0.16)}),
        # C 批：U_L 从 lift 位姿起单独走完跨线弧线（探针 arc_C 逐跳原数，
        # 全程锁顶抓姿态 —— 锁姿把腕钉在朝下、肘部承担运动反而不折；
        # 多臂同拍互相否决步进时 6-DOF 的 IK 分支会被挤进折叠盆地）
        _ph("pull_ul1", 1.5,
            targets={"U_L": (-0.14, 0.28, 1.16)},
            oris=dict({"U_L": ul.R_flange} if u_ori else {}),
            lerps={"F_R": dict(_BREATH)}),
        _ph("pull_ul2", 1.5,
            targets={"U_L": (-0.12, 0.04, 1.16)},
            oris=dict({"U_L": ul.R_flange} if u_ori else {}),
            lerps={"F_L": dict(_BREATH)}),
        _ph("pull_ul3", 1.5,
            targets={"U_L": (-0.30, -0.10, 1.16)},
            oris=dict({"U_L": ul.R_flange} if u_ori else {}),
            lerps={"U_R": dict(_BREATH)}),
        _ph("place_ul", 1.2,
            targets={"U_L": pl["U_L"]},
            oris=dict({"U_L": ul.R_flange} if u_ori else {})),
        _ph("release_ul", 1.0,
            targets={"U_L": _above(pl["U_L"], 0.12)}),
        # 收尾：U 双臂在互换后的半区就近settle（泊位互换 —— 与 cube 互换
        # 同构；折返穿越回原位的回程在 U 排关节空间里不存在无折叠解）
        _ph("retreat", 2.5,
            targets={"F_L": (0.46, -0.44, 1.28), "F_R": (0.46, 0.44, 1.28),
                     "U_L": (-0.42, -0.28, 1.16), "U_R": (-0.40, 0.16, 1.14)}),
    ]
    obj_of = {"F_L": CUBE_NAME, "F_R": CUBE_FR_NAME,
              "U_L": CUBE_UL_NAME, "U_R": CUBE_UR_NAME}
    return S9Task(
        family="dual_pick_swap",
        spec=SkillSpec("s9_dual_pick_swap", phases),
        hand_events=[("close", 0.10, a, "close", 0.60) for a in ARM_KEYS] + [
            ("release_a", 0.30, "F_L", "open", 0.0),
            ("release_a", 0.30, "U_R", "open", 0.0),
            ("release_b", 0.30, "F_R", "open", 0.0),
            ("release_ul", 0.30, "U_L", "open", 0.0),
        ],
        object_events=[
            ("close", 0.90, "attach", a, None, obj_of[a], None)
            for a in ARM_KEYS
        ] + [
            ("release_a", 0.20, "release", None, None, CUBE_NAME, None),
            ("release_a", 0.20, "release", None, None, CUBE_UR_NAME, None),
            ("release_b", 0.20, "release", None, None, CUBE_FR_NAME, None),
            ("release_ul", 0.20, "release", None, None, CUBE_UL_NAME, None),
        ],
        success={"kind": "pick_place", "place_xy": list(SWAP_PLACE["F_L"]),
                 "radius_m": 0.10, "min_disp_m": 0.20,
                 "rest_z": round(0.80 + CUBE_SIZE / 2, 4)},
        objects=[CUBE_NAME, CUBE_FR_NAME, CUBE_UL_NAME, CUBE_UR_NAME],
        env_yaml=R24_ENV_YAML,
    )


def _dual_pick_swap_variants(provider):
    per_arm = {
        "F_L": cube_cands(provider, "F_L", CUBE_POS, GRASP_PARAMS),
        "F_R": cube_cands(provider, "F_R", CUBE_FR_POS, GRASP_PARAMS),
        "U_L": cube_cands(provider, "U_L", CUBE_UL_POS, U_GRASP_PARAMS),
        "U_R": cube_cands(provider, "U_R", CUBE_UR_POS, U_GRASP_PARAMS),
    }
    tops = {a: [c for c in cs if c.kind == "top"] for a, cs in per_arm.items()}
    for k in range(min(2, *(len(v) for v in tops.values()))):
        yield (f"top{k}", _dual_pick_swap_task({a: tops[a][k] for a in ARM_KEYS}))
    yield ("top0/u_pos_only",
           _dual_pick_swap_task({a: tops[a][0] for a in ARM_KEYS}, u_ori=False))


# ---------------------------------------------------------------------------
# 任务族 4: pillow_sheath —— 真机"套枕套"的刚体代理
# ---------------------------------------------------------------------------


def _pillow_sheath_task(cands: dict) -> S9Task:
    """F 双臂顶抓枕头代理两端 -> 合抬到 seam 展示位 (x 0.38, z flange 1.26)
    悬停；U 双臂轮流沿枕头长轴做"裹套"扫掠（U_L 先从 +y 端扫到中线，回撤
    后 U_R 从 -y 端扫 —— 时间错相避免 JAKA 排内对撞），最后 U 回撤、F 保持
    高举收尾（grasp 判据：举起且保持）。刚体代理边界与布料版路线图见
    /tmp/r24_task_library_design.md。"""
    fl, fr = cands["F_L"], cands["F_R"]
    ori_f = {"F_L": fl.R_flange, "F_R": fr.R_flange}
    fgl, fgr = fl.flange_grasp, fr.flange_grasp
    lift = 0.20
    show_x = 0.38                     # 展示位（离 seam 近，让 U 扫掠够得着）

    phases = [
        _ph("approach", 3.0,
            targets={"F_L": fl.flange_pre, "F_R": fr.flange_pre,
                     "U_L": (-0.28, 0.32, 1.14), "U_R": (-0.30, -0.32, 1.14)},
            oris=dict(ori_f)),
        _ph("descend", 1.5, targets={"F_L": fgl, "F_R": fgr},
            oris=dict(ori_f),
            lerps={"U_L": dict(_BREATH)}),
        _ph("close", 1.2, lerps={"U_R": dict(_BREATH)}),
        _ph("lift", 2.0,
            targets={"F_L": _above(fgl, lift), "F_R": _above(fgr, lift),
                     "U_L": (-0.24, 0.30, 1.18), "U_R": (-0.26, -0.30, 1.18)},
            oris=dict(ori_f)),
        _ph("present", 2.0,
            targets={"F_L": (show_x, fgl[1], fgl[2] + lift),
                     "F_R": (show_x, fgr[1], fgr[2] + lift)},
            oris=dict(ori_f)),
        # U_L 裹套扫掠：从 +y 端斜进到中线附近（x -0.12 车道，与展示位
        # 水平间隙 0.50），模拟枕套从一端裹到中段
        _ph("sheath_l", 2.0,
            targets={"U_L": (-0.12, 0.12, 1.14)},
            lerps={"F_L": dict(_BREATH)}),
        _ph("sheath_l_back", 1.5,
            targets={"U_L": (-0.26, 0.32, 1.16)}),
        _ph("sheath_r", 2.0,
            targets={"U_R": (-0.12, -0.12, 1.14)},
            lerps={"F_R": dict(_BREATH)}),
        _ph("sheath_r_back", 1.5,
            targets={"U_R": (-0.28, -0.32, 1.16)}),
        _ph("hold_high", 1.5, lerps={"U_L": dict(_BREATH)}),
        _ph("retreat_u", 2.0,
            targets={"U_L": (-0.40, 0.42, 1.14), "U_R": (-0.42, -0.36, 1.14)}),
    ]
    return S9Task(
        family="pillow_sheath",
        spec=SkillSpec("s9_pillow_sheath", phases),
        hand_events=[
            ("close", 0.10, "F_L", "close", 0.30),
            ("close", 0.10, "F_R", "close", 0.30),
        ],
        object_events=[
            ("close", 0.90, "attach", "F_L", None, PILLOW_NAME, 0.24),
        ],
        success={"kind": "grasp", "lift_m": 0.12},
        objects=[PILLOW_NAME],
        env_yaml=R24_ENV_YAML,
    )


def _pillow_sheath_variants(provider):
    f_l = grip_station_cands(provider, "F_L",
                             (PILLOW_POS[0], -PILLOW_GRIP_Y, PILLOW_POS[2]),
                             (PILLOW_SIZE[0], 0.08, PILLOW_SIZE[2]),
                             GRASP_PARAMS)
    f_r = grip_station_cands(provider, "F_R",
                             (PILLOW_POS[0], PILLOW_GRIP_Y, PILLOW_POS[2]),
                             (PILLOW_SIZE[0], 0.08, PILLOW_SIZE[2]),
                             GRASP_PARAMS)
    for k in range(min(2, len(f_l), len(f_r))):
        yield (f"pillow_yaw{k}", _pillow_sheath_task({"F_L": f_l[k],
                                                      "F_R": f_r[k]}))


# ---------------------------------------------------------------------------
# 参与度审计（R24 验收项：四臂都得真的在动）
# ---------------------------------------------------------------------------


def audit_arm_participation(provider, traj,
                            min_travel: float = MIN_EE_TRAVEL_M) -> dict:
    """逐臂 flange 行程 + max|Δq|。行程 < min_travel 或全程 Δq=0 判闲臂。

    为什么用 EE 行程而不是关节行程：owner 的抱怨是"画面上臂不动"——
    视觉上的动 = 末端在走；关节空间的微 lerp 呼吸（~0.0005 rad/步）EE
    行程只有厘米级，正确地被这道门拦住。"""
    T = traj.n_steps
    pos = {a: [] for a in ARM_KEYS}
    for t in range(T + 1):
        q = {a: torch.tensor(traj.q[a][t: t + 1], dtype=torch.float32)
             for a in ARM_KEYS}
        fko = provider.fk_all(q)
        for a in ARM_KEYS:
            pos[a].append(fko[a]["t_flange"][0].numpy())
    out = {}
    for a in ARM_KEYS:
        p = np.asarray(pos[a])
        travel = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        dq = float(np.abs(np.diff(traj.q[a], axis=0)).max())
        out[a] = {"ee_travel_m": round(travel, 3),
                  "max_dq_rad": round(dq, 5),
                  "moving": bool(travel >= min_travel and dq > 0.0)}
    out["min_travel_m"] = min_travel
    out["all_arms_moving"] = bool(all(out[a]["moving"] for a in ARM_KEYS))
    return out


# ---------------------------------------------------------------------------
# 构建入口（task_record_s9.main 按族名分发到这里）
# ---------------------------------------------------------------------------

_VARIANTS = {
    "four_lift": _four_lift_variants,
    "relay_chain": _relay_chain_variants,
    "dual_pick_swap": _dual_pick_swap_variants,
    "pillow_sheath": _pillow_sheath_variants,
}


def build_task_r24(provider, family: str, out_dir, verbose: bool = True) -> dict:
    """R24 族构建：变体逐个试解（IK 不收敛 / 逐步审计 FAIL / 闲臂 -> 下一
    个变体），第一个全绿的胜出。与 legacy build_task 同一套审计口径 +
    participation 新门槛。"""
    import json
    from pathlib import Path

    from safeduo.delta.task_record_s9 import _compose_and_validate, resolve_events

    last_err = None
    for tag, task in _VARIANTS[family](provider):
        try:
            traj, rep = _compose_and_validate(provider, task, None)
        except RuntimeError as e:
            last_err = f"{tag}: {e}"
            print(f"R24_VARIANT_REJECTED {family} {tag}: {e}", flush=True)
            continue
        if "FAIL" in rep:
            last_err = f"{tag}: {rep['FAIL']}"
            print(f"R24_VARIANT_REJECTED {family} {tag}: {rep['FAIL']}", flush=True)
            continue
        rep["participation"] = audit_arm_participation(provider, traj)
        if not rep["participation"]["all_arms_moving"]:
            rep["FAIL"] = f"idle arm: {rep['participation']}"
            last_err = f"{tag}: {rep['FAIL']}"
            print(f"R24_VARIANT_REJECTED {family} {tag}: {rep['FAIL']}", flush=True)
            continue
        break
    else:
        raise RuntimeError(f"{family}: 所有 R24 变体都不过硬门槛，最后错误: "
                           f"{last_err}")

    phases, hand, objev, success = resolve_events(task)
    assert phases[-1]["t1"] <= traj.duration + 1e-6
    primary = task.objects[0]
    traj.meta["s9_task"] = {
        "family": task.family,
        "object": dict(R24_OBJECTS[primary]),
        "objects": [dict(R24_OBJECTS[n]) for n in task.objects],
        "env_yaml": task.env_yaml,
        "phases": phases,
        "hand_events": hand,
        "object_events": objev,
        "variant": tag,
        "success": success,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"task_{task.family}.npz"
    traj.save(fp)
    (out_dir / f"task_{task.family}_report.json").write_text(
        json.dumps(rep, indent=1, ensure_ascii=False))
    part = rep["participation"]
    print(f"{task.family}: variant={tag} steps={traj.n_steps} "
          f"dur={traj.duration:.1f}s gate={rep['hard_gate_margin_gt0']} "
          f"min_margin={rep['min_margin']} "
          f"travel={ {a: part[a]['ee_travel_m'] for a in ARM_KEYS} }")
    return rep
