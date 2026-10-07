"""S9 object-task choreography (R17 首版 2026-08-20；R23 抓取质量修复
2026-08-28): grasp / pick_place / dual-arm handover WITH a tabletop object,
on the v7 scene.

R23 修复了 owner 审片指出的"cube 是穿模蹭起来的"三根因（对应设计文档
/tmp/r23_grasp_fix_design.md 的根因→修复映射表）：
  1. position-only IK 手掌姿态不对齐物体 → 抓取类相位（reach/descend/
     close/lift/move/place）带 ori_targets 走 6 维位姿 IK（solve_waypoint
     的 R23 扩展），腕姿态由 grasp_gen 的解析 antipodal 候选给出（顶抓 +
     四侧抓，含预抓取偏移/抓取深度/主轴对齐，按自然度排序逐个试解，第一
     个过 IK+硬门槛审计的胜出）；
  2. 没有真正的 approach-grasp-lift 相位结构 → reach 到预抓取位姿（沿
     接近轴退开 ~10 cm）→ descend 沿抓取轴下探到抓取位姿 → close 闭爪
     停留（1.2 s，attach 事件在闭爪完成后才发）→ lift 竖直提升（姿态保
     持，cube 不再随腕乱转）；
  3. kinematic attach 穿模蹭起 → 录制端（a22_record_s9task）改为"闭爪
     完成 + TCP-物体距离合格才 attach"，并支持 physics_grasp 开关，见
     该文件 R23 注释。

Arm-motion pipeline = the S5 v7 skill toolchain re-used verbatim:
design_skill_v7 (margin-guarded DLS-IK from the duo_env_v7 birth pose) ->
compose_skill cosine-ease fallback -> per-step audit (hard gate margin > 0 +
V7_DESIGN_FLOORS + audit_v7_channels). cuRobo 不在本机跑（两张 5090 在训
练）：轨迹分段规划抽成 traj_planner.ArmTrajPlanner 接口，默认 IkInterp
（= 现状 cosine-ease 回退），cuRobo 存根带完整落地说明（--planner curobo
/ --segments 对接 skill_plan_server_v7 的 npz 协议）。

One deviation from make_v7_provider: the provider birth pose is overridden
with the CURRENT duo_env_v7.yaml init_qpos (S4 re-search) so the composed
q0 equals the live env reset pose -- the battery7 q0-teleport mismatch class
is designed out (pinned in tests).

npz format: SkillTrajectory (q_<arm> absolute joints @ 60 Hz) EXTENDED via
meta["s9_task"] (meta is free-form JSON, replay-compatible with every
existing consumer):

    family         grasp | pick_place | handover
    object         {name, size, init_pos, color}  (must match the
                   duo_env_v7_objects.yaml prop; pinned in tests)
    phases         [{name, t0, t1}]              caption clock
    hand_events    [{t, arm, action: close|open, frac, ramp_s}]
    object_events  [{t, action: attach|release, arm?, snap_dist?, snap_s?}]
    grasp          R23: 选中的抓取候选（名字/腕姿态/TCP 偏移/预抓取位姿）
                   + 生成参数 —— 录制端 attach 门控与复现实验都读它
    success        object_task_success.judge_from_meta params
                   (handover carries transfer_t = receiver attach time)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from safeduo.configs import load_config
from safeduo.delta.grasp_gen import GraspGenParams, box_antipodal_grasps
from safeduo.delta.skill_record import Phase, SkillSpec, compose_skill, validate_trajectory
from safeduo.delta.skill_record_v7 import (
    V7_DESIGN_FLOORS,
    audit_v7_channels,
    design_skill_v7,
)
from safeduo.delta.traj_planner import load_segments_dir, make_planner
from safeduo.safety.types import ARM_KEYS

# --- scene constants (single source; duo_env_v7_objects.yaml is pinned to
# these numbers by tests/test_s9_object_tasks.py) -------------------------
CUBE_NAME = "cube_main"
CUBE_SIZE = 0.05
CUBE_POS = (0.54, -0.22, 0.825)     # F-table pick point, planar 0.34 from F_L
PLACE_XY = (0.30, -0.45)            # place zone center (disp 0.33 m from pick)
GIVER = "F_L"
RECEIVER = "U_R"

# R23 抓取生成参数（单一来源；改这里 = 全家族生效）。tcp_offset z=0.165
# 是 F2 手指尖捏合区（掌心包覆会把指尖球压到桌面以下，球模型判负 ——
# 见 grasp_gen 模块 docstring 的几何论证）。
#
# min_flange_z 标定（远端 CPU 探针 /tmp/r23_probe_finger_span.py,
# 2026-08-28）：手竖直朝下时 F_L 参与桌面通道的最低球是四指根部
# left_*_1 r=0.045 球（球底 = flange - 0.2314；基座大球被 TABLE_SKIP
# 豁免不算）。桌顶 0.80 + 悬垂 0.2314 + 设计余量 0.012 → 1.0434。
# 深捏被碰撞球模型钳制成"cube 上半段入指间"：视觉细指尖到 flange-0.24
# （z≈0.804，低于 cube 顶 0.85）→ 画面上手指确实包住 cube，审计仍干净。
FINGER_TABLE_SPAN = 0.2314          # flange -> 手最低碰撞球底（手朝下）
TABLE_TOP_Z = 0.80
TABLE_CLEAR = 0.012                 # 高于 V7_DESIGN_FLOORS.table=0.0012 一个量级
PLACE_CLEAR_PHYS = 0.02             # R27 S2: release height above the pick pose (physics grasp)
GRASP_PARAMS = GraspGenParams(
    min_flange_z=round(TABLE_TOP_Z + FINGER_TABLE_SPAN + TABLE_CLEAR, 4))

_UR_JOINT_ORDER = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                   "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")


def v7_yaml_init_q() -> dict:
    """duo_env_v7.yaml init_qpos as the provider init_q dict (the live birth
    pose; real_geometry_v7.make_v7_provider still carries the pre-S4 one)."""
    y = load_config("duo_env_v7.yaml")
    fq = tuple(float(y["init_qpos"]["franka"][f"fr3_joint{i}"])
               for i in range(1, 8))
    uq = tuple(float(y["init_qpos"]["ur"][k]) for k in _UR_JOINT_ORDER)
    return {"F_L": fq, "F_R": fq, "U_L": uq, "U_R": uq}


def make_s9_provider(n_envs: int = 1, device: str = "cpu", spheres: str = "v7"):
    """spheres: 'v7' (production real_v7 shell) or 'r16' (S12 refined shell:
    3-ball F2 fingers -- R27 S1 uses it so the design audit stops vetoing a
    pinch depth the real fingers can reach)."""
    from safeduo.baselines.real_geometry_v7 import (make_v7_provider,
                                                    make_v7r16_provider)

    mk = make_v7r16_provider if spheres == "r16" else make_v7_provider
    provider = mk(n_envs, device=device)
    provider.init_q = v7_yaml_init_q()
    return provider


def grasp_params_r27(arm: str = GIVER, width_m: "float | None" = None,
                     grasp_depth: float = 0.025, calib="v7",
                     squeeze_extra: float = 0.06, close_ramp_s: float = 1.0,
                     approach_mode: str = "finger",
                     height_m: "float | None" = None,
                     pre_grasp_offset: "float | None" = None,
                     thumb_extra: "float | None" = None,
                     f_max: "float | None" = None):
    """R27 S1/S2: GRASP_PARAMS with the MEASURED hand frame of ``arm`` for an
    object of width ``width_m`` (default CUBE_SIZE). Defaults = the 09-05
    winner on the 5 cm cube (r16 shell): finger-direction approach (fingers
    hang beside the far face), pinch 2.5 cm below the top, squeeze 0.06 past
    contact, 1.0 s close ramp -> level face-to-face pinch, gated success. The scalar min_flange_z
    clamp is dropped -- it modelled a vertical hand; the tilted calibrated
    pinch keeps the pads lowest, and the per-step sphere audit in build_task
    remains the hard gate."""
    from dataclasses import replace

    from safeduo.delta.hand_grasp_frame import frame_for_width, load_hand_calib

    w = CUBE_SIZE if width_m is None else float(width_m)
    fkw = {} if f_max is None else {"f_max": float(f_max)}
    frame = frame_for_width(load_hand_calib(calib), arm, w,
                            squeeze_extra=float(squeeze_extra),
                            close_ramp_s=float(close_ramp_s),
                            approach_mode=approach_mode,
                            thumb_extra=thumb_extra, **fkw)
    # never pinch below the object's mid-height (pads would reach the table
    # on low objects such as the 4 cm rod)
    depth = float(grasp_depth)
    if height_m is not None:
        depth = min(depth, 0.5 * float(height_m))
    kw = {}
    if pre_grasp_offset is not None:
        kw["pre_grasp_offset"] = float(pre_grasp_offset)
    return replace(GRASP_PARAMS, hand_frame=frame, min_flange_z=None,
                   grasp_depth=depth, **kw)


def grasp_candidates_s9(provider=None, params: "GraspGenParams | None" = None) -> list:
    """cube_main 的 antipodal 抓取候选（按自然度排序，design 按序试解）。

    provider 给了就用出生位姿 FK 做参考腕姿态（少拧腕优先）+ 基座方向
    先验；不给（纯 schema 测试路径）退化为固定几何排序，无需几何栈。
    """
    ref_R = base_xy = None
    if provider is not None:
        fko = provider.fk_all(provider.default_q())
        ref_R = fko[GIVER]["R_flange"][0].tolist()
        base_xy = provider.layout.base_pose(GIVER)[0][:2]
    return box_antipodal_grasps(CUBE_POS, CUBE_SIZE, params or GRASP_PARAMS,
                                ref_R=ref_R, base_xy=base_xy)


# ---------------------------------------------------------------------------
# task specs
# ---------------------------------------------------------------------------


@dataclass
class S9Task:
    """Arm choreography + the S9 event layer.

    hand_events / object_events reference phases by name with an offset in
    seconds from the phase start; _resolve_events turns them into absolute
    trajectory times after composition.

    R24 多物体扩展：object_events 元组允许再带两个可选位
    (phase, off, action, arm|None, snap[, object_name[, gate_tcp]]) ——
    object_name 缺省 = 主物体（legacy 三任务的 cube_main 行为逐位不变）；
    gate_tcp 是录制端 attach TCP 门的逐事件覆写（长物体抓握站位离物体
    root 有设计偏距，统一门槛会误杀）。objects 为空 = legacy 单 cube。"""

    family: str
    spec: SkillSpec
    hand_events: list = field(default_factory=list)   # (phase, off, arm, action, frac)
    object_events: list = field(default_factory=list)  # (phase, off, action, arm|None, snap, ...)
    success: dict = field(default_factory=dict)
    objects: "list | None" = None       # R24: 物体名列表（[0] 为主物体）
    env_yaml: str = "duo_env_v7_objects.yaml"


_HOLD = {"F_L": "hold", "F_R": "hold", "U_L": "hold", "U_R": "hold"}


def _fl_only(mode: str = "plan") -> dict:
    return {"F_L": mode, "F_R": "hold", "U_L": "hold", "U_R": "hold"}


def _merge_side(targets: dict, mode: dict, side: "dict | None",
                phase: str) -> tuple:
    """R24 四臂全动：把该相位的闲臂副任务航点并进 targets/mode。

    副任务 = 闲臂在自己半区的巡检/预位动线（世界系 flange 目标，位置-only
    IK —— 姿态自由，与 13 族技能库同口径）。安全性不特殊处理：并进来的臂
    一样过 IK margin 守卫 + 逐步硬门槛审计。"""
    if not side or phase not in side:
        return targets, mode
    targets = dict(targets)
    mode = dict(mode)
    for arm, xyz in side[phase].items():
        assert arm not in targets, f"{phase}: side track覆写主角臂 {arm}"
        targets[arm] = tuple(xyz)
        mode[arm] = "plan"
    return targets, mode


def _pick_phases(cand, side: "dict | None" = None) -> list:
    """R23 相位化抓取前半段（三任务共用）：
    reach   到预抓取位姿（沿接近轴退开 pre_grasp_offset，腕姿态已对齐）
    descend 沿抓取轴下探到抓取位姿（姿态保持 —— 6 维 IK）
    close   闭爪停留（手指事件 +0.10s 起 0.6s ramp，attach 在闭爪完成后）
    lift    竖直提升（姿态保持，cube 不随腕乱转）
    与 R17 版的差别：targets 从裸位置变成 grasp_gen 候选的位姿链。
    R24：side 表把闲臂的巡检/预位动线并进同一批相位（四臂全动）。"""
    fg, fp = cand.flange_grasp, cand.flange_pre
    ori = {GIVER: cand.R_flange}
    lift_z = fg[2] + GRASP_PARAMS.lift_height
    rows = [
        ("reach", 2.5, {GIVER: tuple(fp)}, _fl_only()),
        ("descend", 1.5, {GIVER: tuple(fg)}, _fl_only()),
        ("close", 1.2, {}, dict(_HOLD)),
        ("lift", 1.5, {GIVER: (fg[0], fg[1], lift_z)}, _fl_only()),
    ]
    out = []
    for name, dur, tgts, mode in rows:
        tgts, mode = _merge_side(tgts, mode, side, name)
        out.append(Phase(name, dur, targets=tgts, mode=mode,
                         ori_targets=(dict(ori) if GIVER in tgts else {})))
    return out


# --- R24 闲臂副任务表（每族一张：相位名 -> {臂: flange 目标}）------------
# 走廊设计意图：闲臂动线扫向中线共享带（未来冲突真正发生的地方），与主任务
# 工作区形成"可能相交的走廊"——近而不撞（水平净距 0.4 m 量级，硬门槛兜底）。
# 数字锚定在 13 族已审计目标附近（U 低点 flange>=1.035、F 低点离自己基座
# 平面距离 >=0.33、seam 会合腕间距 >=0.30）。

_GRASP_SIDE = {
    "reach":   {"U_R": (-0.30, -0.30, 1.16), "F_R": (0.50, 0.22, 1.18),
                "U_L": (-0.30, 0.36, 1.16)},
    "descend": {"U_R": (-0.30, -0.30, 1.08), "F_R": (0.50, 0.22, 1.08),
                "U_L": (-0.30, 0.36, 1.08)},
    # 主臂闭爪停留时，三条闲臂做低位巡检扫掠（向 seam 方向推进）
    "close":   {"U_R": (-0.16, -0.28, 1.08), "F_R": (0.38, 0.20, 1.08),
                "U_L": (-0.16, 0.34, 1.08)},
    "lift":    {"U_R": (-0.30, -0.26, 1.14), "F_R": (0.48, 0.28, 1.16),
                "U_L": (-0.30, 0.38, 1.14)},
}

_PICK_PLACE_SIDE = dict(_GRASP_SIDE)

_HANDOVER_SIDE = {
    # 主对（F_L 抓取段）进行时：U_R 向 meet 预位；镜像对（F_R/U_L）向
    # y>0 会合带集结 —— 后面它们会演一场"镜像会合"（近距不接触）
    "reach":   {"U_R": (-0.28, -0.32, 1.14), "F_R": (0.46, 0.34, 1.20),
                "U_L": (-0.32, 0.34, 1.16)},
    "descend": {"F_R": (0.46, 0.34, 1.14), "U_L": (-0.30, 0.35, 1.14)},
    # U_L 先泊车到会合位（y>0 对角脆弱，沿 handover_high 时序纪律）
    "close":   {"U_L": (-0.18, 0.36, 1.14)},
    "lift":    {"F_R": (0.30, 0.39, 1.14)},
}


# 闭爪在 close+0.10s 发（0.6s ramp → +0.70s 闭合完毕），attach 在 +0.90s
# —— 时序上保证"先闭爪完成、后 attach"（录制端 R23 还会做状态门控，
# 时序只是第一道保险；test_event_clocks 钉 t_close < t_attach）
_PICK_HAND = [("close", 0.10, GIVER, "close", 0.60)]
_PICK_ATTACH = [("close", 0.90, "attach", GIVER, None)]


def task_specs_s9(cand=None, legacy_meet: bool = False) -> dict:
    """三任务编排。cand=None 用无 provider 的默认候选（顶抓，纯几何排序
    第一名）—— 保持 schema 测试可离线跑；build_task 会传入逐个候选。

    legacy_meet（仅 handover）：True 时 meet 相位回退 R17 位置-only 口径
    （S5 审计过的会合几何）—— 姿态保持版若 IK/margin 不过硬门槛就用它。

    R24 四臂全动改造：每族闲臂都有副任务动线（_*_SIDE 表 + 后段相位的
    显式闲臂目标），构建报告新增 participation 审计（四臂 EE 行程门槛）。
    """
    if cand is None:
        cand = grasp_candidates_s9(None)[0]
    fg = cand.flange_grasp
    lift_z = fg[2] + GRASP_PARAMS.lift_height
    ori = {GIVER: cand.R_flange}
    # R27 S2: a physically held cube rides ~1.5 cm up inside the hand while
    # the tilted pads converge, so placing at the pick flange height pushes
    # it into the table. Calibrated (hand_frame) candidates release from a
    # small clearance instead; legacy kinematic-attach candidates unchanged.
    place_z = fg[2] + (PLACE_CLEAR_PHYS if "hand_frame" in cand.notes else 0.0)
    # R27 S2: a calibrated candidate closes to its measured squeeze fraction
    # (contact fraction + squeeze_extra) instead of the legacy constant 0.60,
    # so the fingers stop just past contact instead of dragging the object up
    # the far face while they keep curling.
    hf = cand.notes.get("hand_frame") if isinstance(cand.notes, dict) else None
    close_frac = float(hf["f_squeeze"]) if hf else 0.60
    close_ramp = float(hf.get("close_ramp_s", 0.6)) if hf else 0.6
    pick_hand = [("close", 0.10, GIVER, "close", close_frac, close_ramp)]
    # R27 S2: with a measured hand frame the pinch point is up to ~12 cm off
    # the flange horizontally, so transport / place targets must be given for
    # the CUBE and mapped back to the flange (legacy: pinch on the flange axis,
    # flange xy == cube xy, unchanged).
    if hf:
        import numpy as _np
        _off = _np.asarray(cand.R_flange, dtype=_np.float64) @ _np.asarray(
            hf["pinch_mid"], dtype=_np.float64)

        def flange_for_cube(x, y, z):
            return (float(x - _off[0]), float(y - _off[1]), float(z - _off[2]))
    else:
        def flange_for_cube(x, y, z):
            return (float(x), float(y), float(z))
    cube_z0 = CUBE_POS[2]
    move_tgt = (flange_for_cube(PLACE_XY[0], PLACE_XY[1], cube_z0 + GRASP_PARAMS.lift_height)
                if hf else (PLACE_XY[0], PLACE_XY[1], lift_z))
    place_tgt = (flange_for_cube(PLACE_XY[0], PLACE_XY[1], cube_z0 + PLACE_CLEAR_PHYS)
                 if hf else (PLACE_XY[0], PLACE_XY[1], place_z))

    grasp = S9Task(
        family="grasp",
        spec=SkillSpec("s9_grasp", _pick_phases(cand, _GRASP_SIDE) + [
            # 主臂举着 cube 定格；三条闲臂收回巡检动线（四臂全动收尾）
            Phase("hold", 1.5,
                  targets={"U_R": (-0.36, -0.32, 1.16), "F_R": (0.52, 0.30, 1.22),
                           "U_L": (-0.36, 0.40, 1.16)},
                  mode={"F_L": "hold", "U_R": "plan", "F_R": "plan",
                        "U_L": "plan"}),
        ]),
        hand_events=list(pick_hand),
        object_events=list(_PICK_ATTACH),
        success={"kind": "grasp", "lift_m": 0.10},
    )

    pick_place = S9Task(
        family="pick_place",
        spec=SkillSpec("s9_pick_place", _pick_phases(cand, _PICK_PLACE_SIDE) + [
            # move/place 姿态保持：cube 挂在 EE 系下，腕一转 cube 就跟着
            # 空中打转（R17 视频里的穿帮点之一）。
            # 闲臂走廊：U_R 逆向巡检切进 place 区对面（水平净距 ~0.45），
            # F_R/U_L 各自向 seam 推进 —— 与主搬运动线形成对开车流
            Phase("move", 2.5,
                  targets={GIVER: move_tgt,
                           "U_R": (-0.14, -0.40, 1.10),
                           "F_R": (0.44, 0.34, 1.14),
                           "U_L": (-0.26, 0.30, 1.10)},
                  ori_targets=dict(ori),
                  mode={GIVER: "plan", "U_R": "plan", "F_R": "plan",
                        "U_L": "plan"}),
            Phase("place", 1.5,
                  # 放回抓取时的 flange 高度 → cube 底面回到桌面（attach
                  # 保持的相对位姿使 cube 高度 = flange 高度 - 定差）
                  targets={GIVER: place_tgt,
                           "U_R": (-0.14, -0.40, 1.06)},
                  ori_targets=dict(ori),
                  mode={GIVER: "plan", "U_R": "plan", "F_R": "hold",
                        "U_L": "hold"}),
            Phase("release", 1.0,
                  joint_moves={"F_R": {1: 0.04}, "U_L": {1: -0.04}},
                  mode={"F_L": "hold", "U_R": "hold", "F_R": "lerp",
                        "U_L": "lerp"}),
        ] + ([
            # R27 S2: with the sideways (finger-approach) palm the straight
            # retreat from the place pose grazes the own forearm (self_F IK
            # guard); withdraw straight up at held orientation first.
            Phase("withdraw", 1.0,
                  targets={GIVER: (place_tgt[0], place_tgt[1], place_tgt[2] + 0.12)},
                  mode={GIVER: "plan", "U_R": "hold", "F_R": "hold",
                        "U_L": "hold"},
                  ori_targets=dict(ori)),
        ] if hf else []) + [
            Phase("retreat", 2.0,
                  targets={GIVER: (0.46, -0.44, 1.25),
                           "U_R": (-0.36, -0.30, 1.14),
                           "F_R": (0.52, 0.40, 1.24),
                           "U_L": (-0.36, 0.40, 1.14)},
                  mode={GIVER: "plan", "U_R": "plan", "F_R": "plan",
                        "U_L": "plan"}),
        ]),
        hand_events=list(pick_hand) + [("release", 0.30, GIVER, "open", 0.0)],
        object_events=list(_PICK_ATTACH) + [("release", 0.20, "release", None, None)],
        success={"kind": "pick_place", "place_xy": list(PLACE_XY),
                 "radius_m": 0.10, "min_disp_m": 0.20,
                 "rest_z": round(0.80 + CUBE_SIZE / 2, 4)},
    )

    # meet numbers = the S5-audited handover_FL_UR lane (y<0, flange gap 0.31)
    # R24：F_R/U_L 镜像对在 y>0 带同步演一场"近距会合"（gap 0.35，不接触
    # 不交接）—— 画面上两场会合同时发生，四臂满负荷
    meet_kw = ({} if legacy_meet else {"ori_targets": dict(ori)})
    if hf:
        handover = _handover_phys_task(cand, hf, pick_hand, ori)
    else:
        handover = S9Task(
        family="handover",
        spec=SkillSpec("s9_handover", _pick_phases(cand, _HANDOVER_SIDE) + [
            Phase("handover-meet", 3.0,
                  targets={GIVER: (0.15, -0.40, 1.10),
                           RECEIVER: (-0.15, -0.33, 1.10)},
                  mode={GIVER: "plan", RECEIVER: "plan",
                        "F_R": "hold", "U_L": "hold"},
                  **meet_kw),
            Phase("transfer", 1.8,
                  # 主对交接静止的 1.8s：镜像对完成自己的进近（F_R 单臂
                  # 收线，U_L 已泊车 —— handover_high 的时序纪律）
                  targets={"F_R": (0.17, 0.41, 1.14)},
                  mode={GIVER: "hold", RECEIVER: "hold",
                        "F_R": "plan", "U_L": "hold"}),
            Phase("retreat", 2.5,
                  targets={GIVER: (0.44, -0.48, 1.25),
                           RECEIVER: (-0.40, -0.28, 1.14),
                           "F_R": (0.42, 0.48, 1.28),
                           "U_L": (-0.36, 0.46, 1.16)},
                  mode={GIVER: "plan", RECEIVER: "plan",
                        "F_R": "plan", "U_L": "plan"}),
            Phase("settle", 1.0, mode=dict(_HOLD)),
        ]),
        hand_events=list(pick_hand) + [
            ("transfer", 0.20, RECEIVER, "close", 0.60),
            ("transfer", 1.00, GIVER, "open", 0.0),
        ],
        # receiver attach at 0.9 s into transfer, with a 0.6 s snap-in that
        # pulls the cube from the giver grip line into the receiver hand
        object_events=list(_PICK_ATTACH) + [
            ("transfer", 0.90, "attach", RECEIVER, (0.18, 0.6)),
        ],
        success={"kind": "handover", "giver": GIVER, "receiver": RECEIVER,
                 "follow_dist_m": 0.45, "min_travel_m": 0.10},
        )
    return {t.family: t for t in (grasp, pick_place, handover)}


# R27 S2: physical two-hand transfer geometry (cube held by the giver on its
# +-Y faces from above; the receiver pinches the free TOP/BOTTOM faces with a
# horizontal approach from its own side, so neither hand enters the other's
# grip footprint). Cube centre at the meet, world frame.
HANDOVER_CUBE_MEET = (0.02, -0.36, 0.99)
HANDOVER_RECEIVER_APPROACH = (1.0, 0.0, 0.0)     # receiver moves +X into the cube
HANDOVER_RECEIVER_CLOSING = (0.0, 0.0, 1.0)      # thumb below, fingers above
HANDOVER_PRE_M = 0.12


def receiver_hand_frame(width_m: float = CUBE_SIZE, calib="v7"):
    from safeduo.delta.hand_grasp_frame import frame_for_width, load_hand_calib

    return frame_for_width(load_hand_calib(calib), RECEIVER, width_m,
                           squeeze_extra=0.06, close_ramp_s=1.0,
                           approach_mode="finger")


def _handover_phys_task(cand, hf: dict, pick_hand: list, ori: dict) -> "S9Task":
    import numpy as np

    from safeduo.delta.hand_grasp_frame import rot_from_hand_frame

    C = np.asarray(HANDOVER_CUBE_MEET, dtype=np.float64)
    R_g = np.asarray(cand.R_flange, dtype=np.float64)
    P_g = np.asarray(hf["pinch_mid"], dtype=np.float64)
    giver_meet = C - R_g @ P_g                       # giver keeps the pick wrist pose
    fr = receiver_hand_frame()
    R_r = rot_from_hand_frame(HANDOVER_RECEIVER_CLOSING, HANDOVER_RECEIVER_APPROACH, fr)
    a_r = np.asarray(HANDOVER_RECEIVER_APPROACH, dtype=np.float64)
    recv_grasp = C - R_r @ np.asarray(fr.pinch_mid, dtype=np.float64)
    recv_pre = recv_grasp - a_r * HANDOVER_PRE_M
    ori_both = {GIVER: cand.R_flange,
                RECEIVER: tuple(tuple(float(v) for v in row) for row in R_r)}
    t3 = lambda v: tuple(float(x) for x in v)
    fg = cand.flange_grasp
    return S9Task(
        family="handover",
        spec=SkillSpec("s9_handover", _pick_phases(cand, _HANDOVER_SIDE) + [
            Phase("handover-meet", 3.0,
                  targets={GIVER: t3(giver_meet), RECEIVER: t3(recv_pre)},
                  mode={GIVER: "plan", RECEIVER: "plan",
                        "F_R": "hold", "U_L": "hold"},
                  ori_targets=dict(ori_both)),
            Phase("transfer-approach", 1.5,
                  targets={RECEIVER: t3(recv_grasp)},
                  mode={GIVER: "hold", RECEIVER: "plan",
                        "F_R": "hold", "U_L": "hold"},
                  ori_targets={RECEIVER: ori_both[RECEIVER]}),
            Phase("transfer", 2.2,
                  targets={"F_R": (0.17, 0.41, 1.14)},
                  mode={GIVER: "hold", RECEIVER: "hold",
                        "F_R": "plan", "U_L": "hold"}),
            Phase("giver-withdraw", 1.2,
                  targets={GIVER: (float(giver_meet[0]), float(giver_meet[1]),
                                   float(giver_meet[2]) + 0.15)},
                  mode={GIVER: "plan", RECEIVER: "hold",
                        "F_R": "hold", "U_L": "hold"},
                  ori_targets={GIVER: cand.R_flange}),
            Phase("retreat", 2.5,
                  targets={GIVER: (0.44, -0.48, 1.25),
                           RECEIVER: (float(recv_grasp[0]) - 0.25,
                                      float(recv_grasp[1]) + 0.05,
                                      float(recv_grasp[2]) + 0.05),
                           "F_R": (0.42, 0.48, 1.28),
                           "U_L": (-0.36, 0.46, 1.16)},
                  mode={GIVER: "plan", RECEIVER: "plan",
                        "F_R": "plan", "U_L": "plan"},
                  ori_targets={RECEIVER: ori_both[RECEIVER]}),
            Phase("settle", 1.0, mode=dict(_HOLD)),
        ]),
        hand_events=list(pick_hand) + [
            ("transfer", 0.20, RECEIVER, "close", float(fr.f_squeeze), float(fr.close_ramp_s)),
            ("transfer", 1.60, GIVER, "open", 0.0, 0.6),
        ],
        # kinematic-attach events kept for the legacy recorder path; the
        # physics recorder treats them as captions only
        object_events=list(_PICK_ATTACH) + [
            ("transfer", 1.50, "attach", RECEIVER, (0.18, 0.6)),
        ],
        success={"kind": "handover", "giver": GIVER, "receiver": RECEIVER,
                 "follow_dist_m": 0.45, "min_travel_m": 0.10},
    )


# ---------------------------------------------------------------------------
# event clock resolution (pure, testable without IK)
# ---------------------------------------------------------------------------


def phase_clock(spec: SkillSpec) -> list:
    rows, t0 = [], 0.0
    for ph in spec.phases:
        rows.append({"name": ph.name, "t0": round(t0, 4),
                     "t1": round(t0 + ph.dur, 4)})
        t0 += ph.dur
    return rows


def resolve_events(task: S9Task) -> tuple:
    """(phases, hand_events, object_events, success) with absolute times."""
    phases = phase_clock(task.spec)
    start = {p["name"]: p["t0"] for p in phases}
    for name, ph in zip(start, task.spec.phases):
        assert name == ph.name
    hand = []
    for ev in task.hand_events:
        # R27 S2: optional 6th element = close/open ramp seconds (legacy
        # 5-tuples keep the historical 0.6 s ramp bit-for-bit)
        phase, off, arm, action, frac = ev[:5]
        ramp_s = float(ev[5]) if len(ev) > 5 else 0.6
        assert phase in start, f"{task.family}: unknown phase {phase}"
        row = {"t": round(start[phase] + off, 4), "arm": arm,
               "action": action, "frac": float(frac), "ramp_s": ramp_s}
        if len(ev) > 6 and ev[6] is not None:
            row["frac_thumb"] = float(ev[6])     # R27 DFX thumb-only squeeze
        hand.append(row)
    objev = []
    for ev in task.object_events:
        # R24 扩展位：ev[5]=object 名（缺省主物体）、ev[6]=attach TCP 门
        # 覆写；legacy 5 元组行为逐位不变（不写 object/gate_tcp 键）
        phase, off, action, arm, snap = ev[:5]
        obj_name = ev[5] if len(ev) > 5 else None
        gate_tcp = ev[6] if len(ev) > 6 else None
        assert phase in start, f"{task.family}: unknown phase {phase}"
        row = {"t": round(start[phase] + off, 4), "action": action}
        if arm:
            row["arm"] = arm
        if snap is not None:
            row["snap_dist"], row["snap_s"] = float(snap[0]), float(snap[1])
        if obj_name:
            row["object"] = obj_name
        if gate_tcp is not None:
            row["gate_tcp"] = float(gate_tcp)
        objev.append(row)
    success = dict(task.success)
    if success.get("kind") == "handover":
        recv = [e for e in objev
                if e["action"] == "attach" and e.get("arm") == success["receiver"]]
        assert len(recv) == 1, "handover needs exactly one receiver attach"
        success["transfer_t"] = recv[0]["t"]
    return phases, hand, objev, success


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def _compose_and_validate(provider, task: S9Task, segments) -> tuple:
    doc = design_skill_v7(provider, task.spec, verbose=True)
    traj = compose_skill(doc, segments)
    traj.meta["layout"] = "v7"
    traj.meta["birth_pose"] = "duo_env_v7_s4"
    rep = validate_trajectory(provider, traj, floors=V7_DESIGN_FLOORS)
    rep["channels4"] = audit_v7_channels(provider, traj)
    traj.meta["validation"] = rep
    return traj, rep


def build_task(provider, family: str, out_dir: Path,
               candidates: "list | None" = None,
               segments_dir: "str | None" = None,
               verbose: bool = True,
               grasp_params: "GraspGenParams | None" = None) -> dict:
    """R23：按自然度顺序逐候选试解（IK 不收敛 / 逐步审计 FAIL → 下一个），
    第一个全绿的候选胜出。handover 额外多一层回退：姿态保持的 meet 若全
    候选都不过，回 R17 位置-only meet 口径（S5 审计过的会合几何）。"""
    cands = (candidates if candidates is not None
             else grasp_candidates_s9(provider, params=grasp_params))
    attempts = [(c, False) for c in cands]
    if family == "handover":
        attempts += [(c, True) for c in cands]
    last_err = None
    for cand, legacy in attempts:
        task = task_specs_s9(cand, legacy_meet=legacy)[family]
        tag = f"{cand.name}{' legacy-meet' if legacy else ''}"
        try:
            segments = load_segments_dir(segments_dir, task.spec.name)
            traj, rep = _compose_and_validate(provider, task, segments)
        except RuntimeError as e:
            last_err = f"{tag}: {e}"
            print(f"GRASP_CAND_REJECTED {family} {tag}: {e}", flush=True)
            continue
        if "FAIL" in rep:
            last_err = f"{tag}: {rep['FAIL']}"
            print(f"GRASP_CAND_REJECTED {family} {tag}: {rep['FAIL']}", flush=True)
            continue
        # R24 参与度硬门槛：四臂 EE 行程 >= 阈值（owner: 不许有静止臂）
        from safeduo.delta.task_library_r24 import audit_arm_participation

        rep["participation"] = audit_arm_participation(provider, traj)
        if not rep["participation"]["all_arms_moving"]:
            rep["FAIL"] = f"idle arm: {rep['participation']}"
            last_err = f"{tag}: {rep['FAIL']}"
            print(f"GRASP_CAND_REJECTED {family} {tag}: {rep['FAIL']}", flush=True)
            continue
        break
    else:
        raise RuntimeError(f"{family}: 所有抓取候选都不过硬门槛，最后一个错误: "
                           f"{last_err}")

    phases, hand, objev, success = resolve_events(task)
    assert phases[-1]["t1"] <= traj.duration + 1e-6
    traj.meta["s9_task"] = {
        "family": task.family,
        "object": {"name": CUBE_NAME, "size": CUBE_SIZE,
                   "init_pos": list(CUBE_POS)},
        "env_yaml": task.env_yaml,
        "phases": phases,
        "hand_events": hand,
        "object_events": objev,
        # R23：录制端 attach 门控（TCP 偏移/腕姿态）与复现实验都读这里
        "grasp": {"arm": GIVER, "candidate": cand.meta(),
                  "params": (grasp_params or GRASP_PARAMS).meta(),
                  "legacy_meet": bool(legacy)},
        "success": success,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"task_{task.family}.npz"
    traj.save(fp)
    (out_dir / f"task_{task.family}_report.json").write_text(
        json.dumps(rep, indent=1, ensure_ascii=False))
    print(f"{task.family}: cand={cand.name} steps={traj.n_steps} "
          f"dur={traj.duration:.1f}s "
          f"gate={rep['hard_gate_margin_gt0']} "
          f"min_margin={rep['min_margin']} "
          f"channels4={ {k: (v['min_mm'] if v else None) for k, v in rep['channels4'].items()} }")
    return rep


def main(argv: "list | None" = None) -> int:
    import argparse

    from safeduo.delta.task_library_r24 import R24_FAMILIES, build_task_r24

    legacy = ["grasp", "pick_place", "handover"]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["build"])
    ap.add_argument("--task", default=None, choices=legacy + list(R24_FAMILIES),
                    help="one family; default all (legacy 3 + R24 4)")
    ap.add_argument("--out", default="artifacts/task_trajs_s9")
    ap.add_argument("--planner", default="ik", choices=["ik", "curobo"],
                    help="轨迹分段规划器：ik=DLS-IK 航点+cosine 插值（现状）；"
                         "curobo=服务器 GPU 作业存根（见 traj_planner）")
    ap.add_argument("--segments", default=None,
                    help="segments_<skill>.npz 目录（cuRobo 服务端产物）；"
                         "给了就优先用规划段，缺段回退 cosine-ease")
    ap.add_argument("--hand-calib", default=None,
                    help="R27 S1: 'v7' 或标定 JSON 路径 —— 用实测手部抓取系"
                         "（指垫捏合中点/闭合方向）生成候选，替代法兰轴 tcp_offset")
    ap.add_argument("--grasp-depth", type=float, default=0.025,
                    help="R27 S1: 捏合点低于物体顶面的深度（m），仅 --hand-calib 时生效")
    ap.add_argument("--squeeze-extra", type=float, default=0.06,
                    help="R27 S2: 夹紧比例 = 接触比例 + 此值（默认 0.15；越小四指闭合后上拖越少）")
    ap.add_argument("--close-ramp", type=float, default=1.0,
                    help="R27 S2: 闭爪 ramp 秒数（默认 0.6；chembench 慢闭合口径 ~1.0）")
    ap.add_argument("--approach", default="finger", choices=["flange_z", "finger"],
                    help="R27 S2: 接近方向定义：flange_z=法兰 +Z 去闭合分量（默认）；"
                         "finger=指根→指垫方向去闭合分量（手指竖着挂在远侧面外）")
    ap.add_argument("--spheres", default="v7", choices=["v7", "r16"],
                    help="设计期审计用的碰撞球包：v7=生产 real_v7；r16=S12 细化壳"
                         "（F2 手指 3 球），录制 yaml 需同步 assets.spheres: real_v7r16")
    ap.add_argument("--struct-exempt", action="store_true",
                    help="R27: 设计期 IK/航点守卫豁免结构性桌面行（UR5 上臂×桌，出生位即 1.6mm），"
                         "与 R29 执行层豁免同口径")
    args = ap.parse_args(argv)
    if args.struct_exempt:
        from safeduo.delta.skill_record import DESIGN_STRUCT_EXEMPT
        DESIGN_STRUCT_EXEMPT["on"] = True
        print("DESIGN_STRUCT_EXEMPT on", flush=True)

    if args.planner != "ik":
        # curobo 存根：显式失败并打印落地说明（不静默降级成 ik，避免
        # 误以为用上了规划器）；正确用法是 --segments 喂服务端产物
        make_planner(args.planner).segments_for({})

    provider = make_s9_provider(1, device="cpu", spheres=args.spheres)
    print(f"DESIGN_SPHERES {args.spheres}", flush=True)
    gparams = None
    if args.hand_calib:
        gparams = grasp_params_r27(GIVER, grasp_depth=args.grasp_depth,
                                   calib=args.hand_calib,
                                   squeeze_extra=args.squeeze_extra,
                                   close_ramp_s=args.close_ramp,
                                   approach_mode=args.approach)
        print("GRASP_HAND_FRAME " + str(gparams.hand_frame.meta()), flush=True)
    candidates = grasp_candidates_s9(provider, params=gparams)
    print("GRASP_CANDIDATES " + ", ".join(
        f"{c.name}({c.score:+.2f})" for c in candidates), flush=True)
    names = [args.task] if args.task else legacy + list(R24_FAMILIES)
    n_fail = 0
    for name in names:
        if name in R24_FAMILIES:
            rep = build_task_r24(provider, name, Path(args.out))
        else:
            rep = build_task(provider, name, Path(args.out),
                             candidates=candidates, segments_dir=args.segments,
                             grasp_params=gparams)
        if "FAIL" in rep:
            n_fail += 1
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
