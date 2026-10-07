"""R26 任务库扩充（2026-08-31）：非 cube 物体的四臂任务族（R26③ 物体多样化）。

owner 诉求（MASTER_REPORT §9 R26）：任务库只有操作 lego 小方块（cube）的
轨迹，太单调。本模块新增三个非 cube 物体任务族（全部四臂有活；安全性仍由
逐步 sphere margin>0 硬门槛 + R24 participation 审计裁决）：

  bottle_pick   "瓶子"（细高盒近似 0.04x0.04x0.17）抓放：F 双臂各自顶抓
                自己的瓶口（top-mouth，胜出变体），锁姿直立搬到各自落点
                放下；U 双臂同拍各自搬运自己的 cube（four_lift 已审计
                送货动线的裁剪版）。四臂各有一单，无静止臂。
                侧抓卡点（如实记录）：水平侧抓与 20/30/35/50 度斜侧抓共
                三轮 17 个变体全部在 approach 相位被 2mm self 地板毙
                （实测 1.0-2.0mm，见 _pitched_cands 注释与变体序列 ——
                Franka 从高位出生姿态斜探桌面高度的 DLS 路径必经腕折叠
                盆地）。变体序列保留侧抓在前作复现记录，构建时自动落到
                顶抓胜出。
                瓶高取 0.17（建议 0.16）：half.max 0.085>0.08 触发录制端
                大物体豁免，斜抓捕获的相对位姿不被隐式收拢改写（收拢会把
                瓶心拉到掌轴上 —— 瓶子会在闭爪瞬间凌空平移 5cm）。
  tray_relay    "托盘"（扁盘近似 0.17x0.17x0.03）双臂接力搬运：F_L 顶抓
                西缘 rim 站位（虚拟小盒 —— R24 bar/pillow 的站位法，指尖
                捏边），提起走 S5 审计的 FL→UR 会合车道交给 U_R，U_R 带盘
                驶回 U 半区（handover 判据）。F_R/U_L 同拍演镜像会合
                （legacy handover 的 R24 四臂化侧线，已审计）。
                盘边取 0.17（建议 0.12）：①对置双爪共抬 12cm 盘的 flange
                间距 0.12，远低于 v5 腕-腕 0.30 底线（audited 共抬带
                0.26-0.40），共抬不存在无碰撞解 —— 双臂搬运实现为接力；
                ②half.max 0.085>0.08 同样吃到大物体豁免，边缘抓不被隐式
                收拢改写。
  box_colift    "大箱"（0.15x0.20x0.10）对端共抬：F 双臂顶抓箱子 ±y 端
                顶沿站位（four_lift bar 端点抓取同款；F_L 真抓 -y 端，
                F_R 掌贴 +y 面外 5.5cm 扶托 —— 体内/贴面站位会与运动
                学箱体角力，v1/v2 重放实测 self_F -4.9/-6.8cm），同步
                抬升、向 +x 平移 0.14（朝基座、臂展收缩方向）、放下；
                U 双臂各自搬 cube 汇入中线空域
                （four_lift 交叉走廊原版航点）。端面斜侧抓（20-50 度）
                两轮全部死于 approach 腕折 self 1.7-2.0mm，降为兜底。

与 R24 的关系：完全复用 task_record_s9 的相位/IK/审计/npz 管线与
task_library_r24 的相位小工具/参与度审计/"变体逐个试解"模式；不改动 r24
文件里任何已验证任务。场景经由三份新 yaml（duo_env_v7_r26_*.yaml，extends
duo_env_v7_r18_objects.yaml —— table_objects 列表整体替换，各场景只含
本族物体）。构建入口：
    python -m safeduo.delta.task_library_r26 build [--task <family>]
产物落在 artifacts/task_trajs_r26/<family>/task_<family>.npz + report。
"""

from __future__ import annotations

import numpy as np

from safeduo.delta.grasp_gen import GraspCandidate, GraspGenParams, _mk_candidate
from safeduo.delta.skill_record import SkillSpec
from safeduo.delta.task_library_r24 import (
    _arm_prior,
    _BREATH,
    _above,
    _ph,
    CUBE_UL_NAME,
    CUBE_UL_POS,
    CUBE_UR_NAME,
    CUBE_UR_POS,
    R24_OBJECTS,
    U_GRASP_PARAMS,
    audit_arm_participation,
    cube_cands,
    grip_station_cands,
)
from safeduo.delta.task_record_s9 import (CUBE_SIZE, GRASP_PARAMS, PLACE_CLEAR_PHYS,
                                          S9Task, grasp_params_r27)

# R27 S2 (2026-09-05): physical-grasp mode for the R26 families. Off = legacy
# kinematic-attach geometry bit-for-bit. On (--hand-calib): every arm's grasp
# candidates come from the MEASURED hand frame for that object's width
# (finger-direction approach, pinch at object mid-height), the close fraction
# is the frame's squeeze fraction with a 1.0 s ramp, and objects are released
# from PLACE_CLEAR_PHYS above the pick height instead of being pressed down.
PHYS_GRASP = {"on": False, "calib": "v7", "pre": None}


def _params_for(arm: str, width_m: float, legacy: GraspGenParams,
                approach: "str | None" = None,
                height_m: "float | None" = None,
                grasp_depth: "float | None" = None,
                squeeze_extra: "float | None" = None,
                thumb_extra: "float | None" = None,
                f_max: "float | None" = None) -> GraspGenParams:
    if not PHYS_GRASP["on"]:
        return legacy
    kw = {"approach_mode": approach} if approach else {}
    if thumb_extra is not None:
        kw["thumb_extra"] = float(thumb_extra)
    if f_max is not None:
        kw["f_max"] = float(f_max)
    if PHYS_GRASP.get("pre") is not None:
        kw["pre_grasp_offset"] = PHYS_GRASP["pre"]
    if grasp_depth is not None:
        kw["grasp_depth"] = float(grasp_depth)
    if squeeze_extra is not None:
        kw["squeeze_extra"] = float(squeeze_extra)
    return grasp_params_r27(arm, width_m=width_m, calib=PHYS_GRASP["calib"],
                            height_m=height_m, **kw)


# R27: tall/thin bottle -- pinch 6 cm below the top (closer to the centre of
# mass, more body under the pads) and squeeze further (0.12 past contact) so a
# 1-2 cm gated positioning error still captures the 4 cm body
BOTTLE_PHYS_DEPTH = 0.085
BOTTLE_PHYS_SQUEEZE = 0.12
# DFX (U arms): contact for a 5 cm cube at f~0.41; +0.06 closes 4 % past
# contact and the cube slips out under 1 cm of gated positioning error --
# squeeze further (pad aperture ~3.3 cm at 0.56, fingers still clear of each
# other below 0.6)
DFX_CUBE_SQUEEZE = 0.02      # fingers park just past contact ...
DFX_CUBE_THUMB_EXTRA = 0.30  # ... and only the thumb keeps squeezing (tip pinch)


# approach modes tried in order when PHYS_GRASP is on: the finger-direction
# approach gives the level face pinch (S9 winner) but its sideways palm is not
# always IK/self-margin feasible at every station; flange-Z approach is the
# physically-proven fallback (edge pinch, still a real grasp).
PHYS_APPROACHES = ("finger", "flange_z")
# (F-arm approach, U-arm approach) combinations tried in order
PHYS_APPROACH_PAIRS = (("finger", "finger"), ("flange_z", "finger"),
                       ("finger", "flange_z"), ("flange_z", "flange_z"))


def _approach_modes():
    return PHYS_APPROACHES if PHYS_GRASP["on"] else (None,)


def _approach_pairs():
    return PHYS_APPROACH_PAIRS if PHYS_GRASP["on"] else ((None, None),)


def _close_events(cands: dict, arms, t_off: float = 0.10, phase: str = "close") -> list:
    """close events per arm: measured squeeze fraction / ramp when the arm's
    candidate carries a hand frame (plus the thumb-only squeeze fraction for
    DFX frames), legacy (0.60, 0.6 s) otherwise."""
    out = []
    for a in arms:
        hf = cands[a].notes.get("hand_frame") if a in cands else None
        if hf:
            out.append((phase, t_off, a, "close", float(hf["f_squeeze"]),
                        float(hf.get("close_ramp_s", 0.6)), hf.get("f_squeeze_thumb")))
        else:
            out.append((phase, t_off, a, "close", 0.60))
    return out


def _screen(provider, arm: str, cands: list, max_keep: int = 3) -> list:
    """R27: per-arm feasibility screen -- keep the candidates whose pre-grasp
    AND grasp flange poses are reachable by the guarded DLS-IK from the birth
    pose with only `arm` moving (the family builder then combines feasible
    candidates instead of index-pairing arms blindly). Legacy mode: identity."""
    if not PHYS_GRASP["on"]:
        return cands
    from safeduo.delta.skill_record_v7 import V7_IK_FLOORS, solve_waypoint_multiseed
    keep = []
    for c in cands:
        q = provider.default_q()
        ok = True
        seeded = []
        for tgt in (c.flange_pre, c.flange_grasp):
            q, info = solve_waypoint_multiseed(provider, q, {arm: tuple(tgt)},
                                               ori_targets={arm: c.R_flange}, floors=V7_IK_FLOORS)
            if info.get("seeded"):
                seeded.append(info["seeded"])
            if not info.get("ok"):
                ok = False
                break
        tag = ("ok" + (f" seeded={seeded}" if seeded else "")) if ok else \
            "blocked " + str(info.get("blocked") or info.get("residuals"))[:90]
        print(f"R26_SCREEN {arm} {c.name} {tag}", flush=True)
        if ok:
            keep.append(c)
            if len(keep) >= max_keep:
                break
    return keep


def _place_rel(fg, obj_pos, place_xy, legacy_dz: float) -> tuple:
    """flange target that puts the GRASPED object at place_xy: physics ->
    translate the grasp flange by (place - object) and release from
    PLACE_CLEAR_PHYS (valid for any hand frame, the pinch offset cancels);
    legacy -> historical absolute flange xy + overshoot (pinch on the axis)."""
    if PHYS_GRASP["on"]:
        return (fg[0] + (place_xy[0] - obj_pos[0]),
                fg[1] + (place_xy[1] - obj_pos[1]),
                fg[2] + PLACE_CLEAR_PHYS)
    return (place_xy[0], place_xy[1], fg[2] + legacy_dz)


def _place_z(fg, legacy_dz: float) -> float:
    """release height: physics -> pick flange height + clearance (object rides
    ~1 cm up in the hand while squeezing); legacy -> historical overshoot."""
    return fg[2] + (PLACE_CLEAR_PHYS if PHYS_GRASP["on"] else legacy_dz)
from safeduo.safety.types import ARM_KEYS

# ---------------------------------------------------------------------------
# R26 物体清单（单一来源；duo_env_v7_r26_*.yaml 与这些数字逐字段一致）
# ---------------------------------------------------------------------------

R26_ENV_BOTTLE = "duo_env_v7_r26_bottle.yaml"
R26_ENV_TRAY = "duo_env_v7_r26_tray.yaml"
R26_ENV_BIGBOX = "duo_env_v7_r26_bigbox.yaml"

# 瓶子：细高盒近似，站桌面（top z=0.97）。F_L/F_R 各一支，镜像对称
BOTTLE_SIZE = (0.04, 0.04, 0.17)
BOTTLE_L_NAME, BOTTLE_L_POS = "bottle_l", (0.50, -0.36, 0.885)
BOTTLE_R_NAME, BOTTLE_R_POS = "bottle_r", (0.50, 0.36, 0.885)
BOTTLE_NECK_Z = 0.93                 # 斜侧抓捏合点（瓶顶 0.97 下 4cm）
# 落点取与抓取点等基座臂展的弧上（F_L 基座 (0.748,-0.487)，抓/放臂展
# 均 0.28）：重放实测 sim 跟踪 droop 随臂展增大（瓶抓取点错位 6-10cm），
# 抓/放等臂展让 droop 在相对位姿捕获里对消 —— 首轮落点 (0.32,-0.46)
# 臂展 0.43，droop 增量把放置拉短成 moved 0.119<0.15 判负。位移 0.21 m
BOTTLE_PLACE_L = (0.48, -0.57)
BOTTLE_PLACE_R = (0.48, 0.57)

# 托盘：扁盘近似。西缘 rim 站位（离盘心 0.065，边内缩 0.02）
TRAY_NAME, TRAY_SIZE, TRAY_POS = \
    "tray_main", (0.17, 0.17, 0.03), (0.52, -0.40, 0.815)
TRAY_GRIP_DX = 0.065

# 大箱：F_L 抓 -y 端顶沿共抬，F_R 掌贴 +y 面外扶托，向 +x（朝 F 基座）
# 平移 0.14 后放下。方向教训（v1/v2 重放）：朝 -x 搬运时 F 臂展从 0.45
# 拉到 0.55+，droop 增量把运动学箱体推进 F_R 指隙 -> 双臂角力 self_F
# -4.9/-6.8cm、箱子被拖到离靶 0.72-0.76m；+x 方向臂展收缩，droop 对消
BOX_NAME, BOX_SIZE, BOX_POS = "box_main", (0.15, 0.20, 0.10), (0.53, 0.0, 0.85)
BOX_GRIP_Z = 0.88
BOX_DX = 0.14

# 长棒：躺平沿 y（长轴 y 与 four_lift 杆同约定，grip_station_cands 的
# "闭合线沿 x" 顶抓过滤天然适配）。躺棒不存在细高瓶的直立易倒失败模式：
# 放置=clamp 按到 rest 高度 + release settle 写平（yaw-only），跌落不翻。
# 几何整体照抄 four_lift 已验证配方：站位 y ~ ±0.19（≈杆抓 ±0.18），
# 搬运沿 -x 0.16（= bar_dx，回落到 x 0.37 悬停带高度同款 IK 已证可行；
# 首版棒心 y ∓0.36 低位下探 self 余量 0.0018<0.002 被硬门拒）
ROD_SIZE = (0.04, 0.30, 0.04)
ROD_L_NAME, ROD_L_POS = "rod_l", (0.53, -0.30, 0.82)
ROD_R_NAME, ROD_R_POS = "rod_r", (0.53, 0.30, 0.82)
ROD_GRIP_IN = 0.11                   # 抓握站位从棒心向中线内移（内端内侧 4cm）
ROD_DX = -0.16                       # 搬运位移（同 four_lift bar_dx）
# R27: 4 cm 棒顶抓的捏合深度 —— 缺省 2 cm（半高钳位）时指垫球贴桌
# （设计期 table 0.3–1 mm < 1.0 mm 地板，r27_rod_fr_diag.py）；1.2 cm 留 ~8 mm
ROD_PHYS_DEPTH = 0.012
# R27 物理抓取布局：棒外移到 y=∓0.35 并握棒心。finger 接近把法兰从捏合点
# 向中线偏 ~14 cm，内端站位（y ∓0.19）让两只 Franka 腕在中线相距 10 cm
# （self_F 2.8 mm，r27_rod_approach_diag.py）；棒心 ∓0.30 仍剩 4.8 mm；
# ∓0.35 回到结构性 23 mm（r27_rod_layout_scan.py）。握重心 = 搬运不俯仰。
# 录制 yaml duo_env_v7_r26_rod_r27r16.yaml 的 table_objects 与此一致。
ROD_PHYS_Y = 0.35
# R27 物理抓取的棒是 4×30×8 cm 的"梁"（中心 z 0.84，底面仍贴桌 0.80）：
# 4 cm 高的棒被五指从上捏合时指尖球（r 1.5 cm）必然落到桌面（sim table
# −17 mm、抓深 0.6/1.2 cm 皆然），且四指卷曲把一端压下、棒在手里俯仰 ~35°
# （r27_rod2_s0_phys_camL.mp4）。8 cm 高：指尖离桌 ~1.8 cm，捏合线在重心
# 上方 ~3 cm 成摆式稳定。
ROD_PHYS_SIZE = (0.04, 0.30, 0.08)


def _rod_layout() -> tuple:
    """(rod_l_pos, rod_r_pos, grip_offset_toward_centre, size) for the mode."""
    if PHYS_GRASP["on"]:
        z = 0.80 + 0.5 * ROD_PHYS_SIZE[2]
        return ((ROD_L_POS[0], -ROD_PHYS_Y, z), (ROD_R_POS[0], ROD_PHYS_Y, z),
                0.0, ROD_PHYS_SIZE)
    return ROD_L_POS, ROD_R_POS, ROD_GRIP_IN, ROD_SIZE
R26_ENV_ROD = "duo_env_v7_r26_rod.yaml"

R26_OBJECTS = {
    ROD_L_NAME: {"name": ROD_L_NAME, "size": list(ROD_SIZE),
                 "init_pos": list(ROD_L_POS)},
    ROD_R_NAME: {"name": ROD_R_NAME, "size": list(ROD_SIZE),
                 "init_pos": list(ROD_R_POS)},
    BOTTLE_L_NAME: {"name": BOTTLE_L_NAME, "size": list(BOTTLE_SIZE),
                    "init_pos": list(BOTTLE_L_POS)},
    BOTTLE_R_NAME: {"name": BOTTLE_R_NAME, "size": list(BOTTLE_SIZE),
                    "init_pos": list(BOTTLE_R_POS)},
    TRAY_NAME: {"name": TRAY_NAME, "size": list(TRAY_SIZE),
                "init_pos": list(TRAY_POS)},
    BOX_NAME: {"name": BOX_NAME, "size": list(BOX_SIZE),
               "init_pos": list(BOX_POS)},
    CUBE_UL_NAME: dict(R24_OBJECTS[CUBE_UL_NAME]),
    CUBE_UR_NAME: dict(R24_OBJECTS[CUBE_UR_NAME]),
}

R26_FAMILIES = ("bottle_pick", "tray_relay", "box_colift", "rod_pick")


# ---------------------------------------------------------------------------
# 抓取候选构造（几何适配：细高物斜侧抓 / 扁盘边缘抓 / 大箱端面对置斜抓）
# ---------------------------------------------------------------------------


def _pitched_cands(tcp, horiz, thetas, closings=None) -> list:
    """斜侧抓候选：approach = 水平分量 horiz·sinθ + 竖直 -cosθ（θ 从竖直
    往侧向倾），闭合线取与 horiz 垂直的水平向量（横跨截面短边）——
    细高物瓶颈 / 大箱端面上沿的"从上外侧斜下探"抓法。

    为什么不用 grasp_gen 的纯水平侧抓：腕距出生姿态的测地角 ~90°，DLS
    下降在 approach 相位就把 Franka 腕挤进 self 1.1-1.9mm（< IK 地板
    2mm）的折叠盆地 —— 首轮构建 11 个水平变体全数被毙（bottle/box 两族，
    z 0.90-0.945 扫描不敏感）。斜抓把测地角压到 θ（<=50°），留在顶抓
    已审计的腕位形邻域内。"""
    import math as _m

    n = _m.hypot(horiz[0], horiz[1])
    ux, uy = horiz[0] / n, horiz[1] / n
    if closings is None:
        closings = ((-uy, ux, 0.0), (uy, -ux, 0.0))
    params = GraspGenParams()
    out = []
    for th in thetas:
        a = (ux * _m.sin(th), uy * _m.sin(th), -_m.cos(th))
        for j, cl in enumerate(closings):
            out.append(_mk_candidate(
                f"pitch{round(_m.degrees(th))}_c{j}", "side",
                tcp, tcp, a, cl, params))
    return out


def _u_cube_tops(provider, approach: "str | None" = None) -> tuple:
    u_l = [c for c in cube_cands(provider, "U_L", CUBE_UL_POS,
                                 _params_for("U_L", CUBE_SIZE, U_GRASP_PARAMS, approach,
                                             squeeze_extra=DFX_CUBE_SQUEEZE,
                                             thumb_extra=DFX_CUBE_THUMB_EXTRA))
           if c.kind == "top"]
    u_r = [c for c in cube_cands(provider, "U_R", CUBE_UR_POS,
                                 _params_for("U_R", CUBE_SIZE, U_GRASP_PARAMS, approach,
                                             squeeze_extra=DFX_CUBE_SQUEEZE,
                                             thumb_extra=DFX_CUBE_THUMB_EXTRA))
           if c.kind == "top"]
    return u_l, u_r


# ---------------------------------------------------------------------------
# 任务族 1: bottle_pick —— F 双臂斜侧抓瓶子 + U 双臂各自搬 cube
# ---------------------------------------------------------------------------


def _bottle_pick_task(cands: dict, u_ori: bool = True) -> S9Task:
    """四臂各自 pick&place（各在自己半区，无跨线 —— R26 主题是物体几何
    多样化，不复读 dual_pick_swap 的 U 排跨线雷区）。F 全程锁姿：
    瓶子挂在 EE 系下保持直立，place 时 flange 回到抓取高度 → 瓶底触桌
    直立，release 后录制端 v3.4 settle 写合法立姿。"""
    def _slide_out(cand, pl):
        # v24: clear retreats along the reverse of the grasp approach
        # (horizontal part) -- retrace the hand's own collision-free entry
        # path. Near-top grasps (no horizontal component) keep the vertical
        # ascent.
        import math as _m
        hx, hy = -float(cand.approach[0]), -float(cand.approach[1])
        n = _m.hypot(hx, hy)
        if n < 0.30:
            return _above(pl, 0.14)
        return (pl[0] + 0.15 * hx / n, pl[1] + 0.15 * hy / n, pl[2] + 0.01)

    fl, fr = cands["F_L"], cands["F_R"]
    ul, ur = cands["U_L"], cands["U_R"]
    ori_f = {"F_L": fl.R_flange, "F_R": fr.R_flange}
    ori_u = {"U_L": ul.R_flange, "U_R": ur.R_flange} if u_ori else {}
    fgl, fgr = fl.flange_grasp, fr.flange_grasp
    ugl, ugr = ul.flange_grasp, ur.flange_grasp
    lift_f, lift_u = 0.15, 0.11
    # F 落点计划高度故意下压过冲 2cm：sim 垂直跟踪滞后 4-6cm（v5/v6
    # 实测 place 结束时瓶根仍悬 rest+4.5cm 以上），录制端 release-settle
    # 的直立写姿只在根 z < rest+2cm 时触发 —— 靠 +0.012 微抬永远够不到
    # 触发窗，瓶子半空跌落必翻（v5 err 0.177 / v6 0.164，躺姿 z 0.820
    # 两轮分毫不差=确定性翻倒）。过冲后 follow() 的 v3.3 中心 z 钳位把
    # 运动学瓶体按在合法 rest 高度，release 时 settle 必触发、写直立姿
    # 过冲幅度 -0.005（曾用 -0.02）：place 延到 2.5s 后 sim 垂直残差已
    # 到 mm 级（v9 实测 release 时 lift+0.001），settle 触发窗 rest+2cm
    # 依然稳达；-0.02 会把指尖压到离瓶口仅 ~1cm，clear 上提段 sim 的
    # ±1cm 垂直误差让指尖擦倒已立正的瓶（v7/v9 同款躺平 -0.065）
    pl_l = _place_rel(fgl, BOTTLE_L_POS, BOTTLE_PLACE_L, -0.005)
    pl_r = _place_rel(fgr, BOTTLE_R_POS, BOTTLE_PLACE_R, -0.005)
    # U 落点比抓取 flange 高 1.5cm（four_lift 教训：place 下探腕位形更深，
    # 贴抓取高度放会蹭 IK table 地板）
    place_ul = (-0.30, 0.28, _place_z(ugl, 0.015))
    place_ur = (-0.32, -0.28, _place_z(ugr, 0.015))

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
        # F 双臂高车道平移到落点上方；U 双臂把 cube 送进中线前带（four_lift
        # 审计过的 x 车道，本族不做 weave —— 各回自己半区放下）
        _ph("carry", 3.0,
            targets={"F_L": _above(pl_l, lift_f), "F_R": _above(pl_r, lift_f),
                     "U_L": (-0.10, 0.26, 1.12), "U_R": (-0.12, -0.26, 1.12)},
            oris=dict(ori_f)),
        _ph("carry_u", 1.5,
            targets={"U_L": _above(place_ul, 0.10),
                     "U_R": _above(place_ur, 0.10)},
            oris=(dict(ori_u) if u_ori else None),
            lerps={"F_L": dict(_BREATH)}),
        # 2.5s：sim 侧向跟踪误差需要 >1.5s 才收敛到 cm 级（v6 实测
        # 载运落点仍偏 ~8cm）
        _ph("place", 2.5,
            targets={"F_L": pl_l, "F_R": pl_r,
                     "U_L": place_ul, "U_R": place_ur},
            oris={**ori_f, **ori_u}),
        _ph("release", 2.5),  # v19: settle after lateral tracking converges
        # 松手后先竖直提净空再横撤（v5 教训四：低位斜抓姿态直接斜拉回撤
        # 会腕折自碰 —— tray 族首轮 carry_u 实测 self 1.8mm 被卡的同款雷）
        _ph("clear", 1.2,
            targets={"F_L": pl_l, "F_R": pl_r},
            oris=dict(ori_f)),
        _ph("retreat", 2.5,
            targets={"F_L": pl_l, "F_R": pl_r,
                     "U_L": (-0.38, 0.42, 1.16), "U_R": (-0.40, -0.36, 1.16)},
            oris=dict(ori_f)),
    ]
    # 瓶子 gate_tcp：顶抓瓶口 TCP 离瓶心竖直偏 ~0.085 + min_flange_z
    # 钳位余量（首轮重放实测 dist_tcp 0.146/0.170，0.10 的门直接超时
    # -> 'object never picked up'），门取 0.18（两瓶相距 0.72，无误附风险）
    return S9Task(
        family="bottle_pick",
        spec=SkillSpec("s9_bottle_pick", phases),
        # release 推迟到 0.60/0.75：place 下探 15cm/1.5s，sim 跟踪滞后
        # ~0.4s，v5 实测 frac 0.20 松手时法兰仍高 ~4.4cm，细高瓶半空
        # 跌落即倾倒翻滚 0.177m 脱靶；多给 ~0.7s 沉降后再松
        hand_events=(_close_events(cands, ARM_KEYS) if PHYS_GRASP["on"] else
                     [("close", 0.10, a, "close",
                       0.45 if a.startswith("F") else 0.60)
                      for a in ARM_KEYS]) + [
            ("release", 0.75, a, "open", 0.0) for a in ("U_L", "U_R")] + [
            # v12: open F fingers late in the release phase (object released at
            # frac 0.60, settled upright) instead of mid-clear -- v10 frame audit
            # showed closed fingertips sweeping the settled bottle over during
            # the clear ascent (lift -0.065 lying pose).
            ("release", 0.90, a, "open", 0.0) for a in ("F_L", "F_R")],
        object_events=[
            # gate_tcp 0.24：v5 实测 F_R 首试 dist_tcp 0.197（比 F_L 多
            # ~8cm 的 sim 侧向误差），0.18 的门 1.5s 重试窗内臂已进 lift
            # 越走越远 -> 超时空手；两瓶相距 0.72m，0.24 无误附风险
            # F 吸附推迟到 lift 0.20：指尖-瓶口间隙在 attach 捕获时冻结
            # 为常量（close 0.9 时 = 3cm 设计 - sim droop ~1.1cm ≈ 1.9cm，
            # 与放置高度无关），v7/v9/v10 三轮实锤该间隙不够 —— clear
            # 上提段 sim 误差让闭合指尖扫倒已 settle 立正的瓶（v10 帧审：
            # release 时 err 0.027 完美立正，clear 段被扫倒）。先让闭合
            # 手指随 lift 升离瓶口 ~2cm 再冻结，间隙扩到 ~3cm；代价=
            # 运载时瓶悬指尖下 ~3cm（four_lift 同款 B 级观感折衷）
            # v12: v11 recorded ATTACHED rel y-offsets of 7.8/8.7cm -- attaching
            # mid-lift froze the transient lateral tracking error into rel_p and
            # carried it to the drop point (final 0.293m). Attach back at close
            # 0.90 (bottle still standing, gate_tcp 0.24 passes per v5) with an
            # EXPLICIT snap: v3.1 recentres to the pinch midline [0,0,d] and
            # caps d at reach+s_z, so the 8cm lateral offset is zeroed while the
            # vertical barely moves (flange-to-bottle is already ~0.24). The
            # finger-tip clearance concern of v11 is moot: the bottle hangs
            # centred under the TCP, not against a fingertip.
            # snap 字段须为 (snap_dist, snap_s) 元组（R24 handover 惯例，
            # resolve_events 会解包），裸浮点会 TypeError
            ("close", 0.90, "attach", "F_L", (0.30, 0.6), BOTTLE_L_NAME, 0.24),
            ("close", 0.90, "attach", "F_R", (0.30, 0.6), BOTTLE_R_NAME, 0.24),
            ("close", 0.90, "attach", "U_L", None, CUBE_UL_NAME, None),
            ("close", 0.90, "attach", "U_R", None, CUBE_UR_NAME, None),
            ("release", 2.0, "release", None, None, BOTTLE_L_NAME, None),
            ("release", 2.0, "release", None, None, BOTTLE_R_NAME, None),
            ("release", 0.60, "release", None, None, CUBE_UL_NAME, None),
            ("release", 0.60, "release", None, None, CUBE_UR_NAME, None),
        ],
        success={"kind": "pick_place", "place_xy": list(BOTTLE_PLACE_L),
                 "radius_m": 0.10, "min_disp_m": 0.15,
                 "rest_z": round(0.80 + BOTTLE_SIZE[2] / 2, 4)},
        objects=[BOTTLE_L_NAME, BOTTLE_R_NAME, CUBE_UL_NAME, CUBE_UR_NAME],
        env_yaml=R26_ENV_BOTTLE,
    )


def _bottle_pick_variants(provider):
    if PHYS_GRASP["on"]:
        # R27: real pinch on the bottle body 2.5 cm below its top (finger
        # approach first, flange-Z fallback), U arms pick their cubes
        from safeduo.delta.grasp_gen import box_antipodal_grasps
        # per-arm: gather feasible candidates over both approach modes, then
        # combine the first feasible of each arm (finger preferred)
        per_arm = {}
        for arm, pos, size in (("F_L", BOTTLE_L_POS, BOTTLE_SIZE), ("F_R", BOTTLE_R_POS, BOTTLE_SIZE),
                               ("U_L", CUBE_UL_POS, (CUBE_SIZE,) * 3), ("U_R", CUBE_UR_POS, (CUBE_SIZE,) * 3)):
            feas = []
            for ap in PHYS_APPROACHES:
                legacy = GRASP_PARAMS if arm.startswith("F") else U_GRASP_PARAMS
                if arm.startswith("F"):
                    prm = _params_for(arm, size[0], legacy, ap, size[2],
                                      grasp_depth=BOTTLE_PHYS_DEPTH,
                                      squeeze_extra=BOTTLE_PHYS_SQUEEZE)
                else:
                    prm = _params_for(arm, size[0], legacy, ap, size[2],
                                      squeeze_extra=DFX_CUBE_SQUEEZE,
                                      thumb_extra=DFX_CUBE_THUMB_EXTRA)
                ref_R, base_xy = _arm_prior(provider, arm)
                cands = [c for c in box_antipodal_grasps(pos, size, prm, ref_R=ref_R, base_xy=base_xy)
                         if c.kind == "top"]
                feas += [(ap, c) for c in _screen(provider, arm, cands, max_keep=2)]
            per_arm[arm] = feas
            print(f"R26_SCREEN_SUMMARY bottle_pick {arm}: {[(a, c.name) for a, c in feas]}", flush=True)
        if all(per_arm[a] for a in ARM_KEYS):
            n = max(len(per_arm[a]) for a in ARM_KEYS)
            for k in range(min(n, 3)):
                pick = {a: per_arm[a][min(k, len(per_arm[a]) - 1)] for a in ARM_KEYS}
                tag = "+".join(f"{a}:{ap[0]}{c.name[-3:]}" for a, (ap, c) in pick.items())
                yield (f"phys_screened{k}/{tag}",
                       _bottle_pick_task({a: c for a, (ap, c) in pick.items()}))
        return
    u_l, u_r = _u_cube_tops(provider)
    base_u = {"U_L": u_l[0], "U_R": u_r[0]}
    hz_l = (BOTTLE_L_POS[0], BOTTLE_L_POS[1], 0.97 + 0.03)
    hz_r = (BOTTLE_R_POS[0], BOTTLE_R_POS[1], 0.97 + 0.03)
    fhl = _pitched_cands(hz_l, (1.0, 0.0), (0.0,))
    fhr = _pitched_cands(hz_r, (1.0, 0.0), (0.0,))
    yield ("v26_top_hover30mm_cx",
           _bottle_pick_task({"F_L": fhl[0], "F_R": fhr[0], **base_u}))
    neck_l = (BOTTLE_L_POS[0], BOTTLE_L_POS[1], BOTTLE_NECK_Z)
    neck_r = (BOTTLE_R_POS[0], BOTTLE_R_POS[1], BOTTLE_NECK_Z)
    # 20 / 30 度：35/50 度第二轮实测 self 1.4-2.0mm 贴 2mm 地板全毙，
    # 往竖直方向收（顶抓 0 度实测 self 15mm，裕量随倾角单调恶化）
    thetas = (0.35, 0.52)
    # 水平分量：基座→瓶方向（顺手）优先，纯 +y/-y（从外侧）次之
    for dl, dr, dtag in (((-0.89, 0.456), (-0.89, -0.456), "base"),
                         ((0.0, 1.0), (0.0, -1.0), "y")):
        fls = _pitched_cands(neck_l, dl, thetas)
        frs = _pitched_cands(neck_r, dr, thetas)
        for k in range(min(len(fls), len(frs))):
            yield (f"{dtag}/{fls[k].name}",
                   _bottle_pick_task({"F_L": fls[k], "F_R": frs[k], **base_u}))
    # 兜底 1：U 侧位置-only 下探（斜抓首选组合）
    fls = _pitched_cands(neck_l, (-0.89, 0.456), (0.35,))
    frs = _pitched_cands(neck_r, (-0.89, -0.456), (0.35,))
    yield ("base/pitch20/u_pos_only",
           _bottle_pick_task({"F_L": fls[0], "F_R": frs[0], **base_u},
                             u_ori=False))
    # 兜底 2：瓶口上方 3cm 悬停捏合（theta=0 的竖直姿态）。为什么悬空：
    # 帧审(v4 camtest)实锤 —— 贴口顶抓的指尖比瓶口低 6mm，descend 跨骑
    # 时指尖擦碰把自由态轻瓶扫倒（出生态站立、close 前已躺平）；吸附本
    # 就是运动学的（gate_tcp 门控），指尖离口 3cm 让物理上永不接触。
    # 竖直向指尖-附着物接触是角力雷（指棱柱关节吸收不了竖向载荷，wrench
    # 全传腕上），侧向夹压才是无害的 —— cube/four_lift 杆都是侧向几何
    # 30mm cx 优先：其 approach/descend 摆臂路径经 v5/v6/v7 三轮重放
    # 背书不碰瓶（构建端碰撞门不含桌面物体，摆臂是否扫倒瓶只能实测）；
    # 45mm 曾试过 —— 重解 IK 后 F_L 前臂桶身 descend 段扫倒瓶（v8 帧审）
    for dz, (cl, cr, tag) in (
            (0.03, ((1.0, 0.0), (1.0, 0.0), "cx")),
            (0.03, ((0.0, 1.0), (0.0, -1.0), "cy")),
            (0.045, ((1.0, 0.0), (1.0, 0.0), "cx")),
            (0.045, ((0.0, 1.0), (0.0, -1.0), "cy")),
    ):
        hz_l = (BOTTLE_L_POS[0], BOTTLE_L_POS[1], 0.97 + dz)
        hz_r = (BOTTLE_R_POS[0], BOTTLE_R_POS[1], 0.97 + dz)
        fhl = _pitched_cands(hz_l, cl, (0.0,))
        fhr = _pitched_cands(hz_r, cr, (0.0,))
        yield (f"top_hover{int(dz * 1000)}mm_{tag}",
               _bottle_pick_task({"F_L": fhl[0], "F_R": fhr[0], **base_u}))


# ---------------------------------------------------------------------------
# 任务族 2: tray_relay —— F_L 边缘抓托盘 → S5 审计车道交接 → U_R 带盘回半区
# ---------------------------------------------------------------------------


def _tray_relay_task(cand) -> S9Task:
    """相位/航点结构 = legacy handover（R24 四臂化，全部已审计数字），
    载荷从 cube 换成托盘边缘抓。F_R/U_L 的镜像会合侧线原样保留。"""
    fg = cand.flange_grasp
    ori = {"F_L": cand.R_flange}
    lift_z = fg[2] + GRASP_PARAMS.lift_height

    phases = [
        _ph("reach", 2.5,
            targets={"F_L": cand.flange_pre, "U_R": (-0.28, -0.32, 1.14),
                     "F_R": (0.46, 0.34, 1.20), "U_L": (-0.32, 0.34, 1.16)},
            oris=dict(ori)),
        _ph("descend", 1.5,
            targets={"F_L": fg, "F_R": (0.46, 0.34, 1.14),
                     "U_L": (-0.30, 0.35, 1.14)},
            oris=dict(ori)),
        _ph("close", 1.2, targets={"U_L": (-0.18, 0.36, 1.14)}),
        _ph("lift", 1.5,
            targets={"F_L": (fg[0], fg[1], lift_z), "F_R": (0.30, 0.39, 1.14)},
            oris=dict(ori)),
        # S5 审计的 FL_UR 会合车道（y<0 带，flange gap 0.30）
        _ph("meet", 3.0,
            targets={"F_L": (0.15, -0.40, 1.10), "U_R": (-0.15, -0.33, 1.10)},
            oris=dict(ori)),
        _ph("transfer", 1.8, targets={"F_R": (0.17, 0.41, 1.14)}),
        # 交接完成后 F_L 先竖直提净空再横撤（v5 教训四 —— 首轮 carry_u
        # 直接从会合位斜拉回家，实测 self 1.8mm 腕折被卡）
        _ph("clear", 1.2,
            targets={"F_L": (0.15, -0.40, 1.24)},
            oris=dict(ori),
            lerps={"U_L": dict(_BREATH)}),
        # U_R 带盘驶回 U 半区（handover 判据的 receiver 行程段）；F_L 空手
        # 回撤到 legacy 审计驻停位
        # F_L 回家全程锁抓取姿（第二轮实测：净空后松腕横撤仍被 DLS 转进
        # self 1.9mm 的折叠盆地 —— 锁姿让腕保持顶抓位形直到驻停）
        _ph("carry_u", 2.5,
            targets={"U_R": (-0.33, -0.26, 1.12), "F_L": (0.44, -0.48, 1.25)},
            oris=dict(ori)),
        _ph("retreat", 2.5,
            targets={"U_R": (-0.40, -0.28, 1.14), "F_R": (0.42, 0.48, 1.28),
                     "U_L": (-0.36, 0.46, 1.16)},
            lerps={"F_L": dict(_BREATH)}),
        _ph("settle", 1.0),
    ]
    # rim 站位离盘心 0.065 + min_flange_z 钳位抬链 0.063 → TCP-root 0.091，
    # 门取 0.13；接收方事件带显式 snap（0.18/0.6 take-over），TCP 门豁免
    return S9Task(
        family="tray_relay",
        spec=SkillSpec("s9_tray_relay", phases),
        hand_events=[
            ("close", 0.10, "F_L", "close", 0.60),
            ("transfer", 0.20, "U_R", "close", 0.60),
            ("transfer", 1.00, "F_L", "open", 0.0),
        ],
        object_events=[
            ("close", 0.90, "attach", "F_L", None, TRAY_NAME, 0.13),
            ("transfer", 0.90, "attach", "U_R", (0.18, 0.6), TRAY_NAME, None),
        ],
        success={"kind": "handover", "giver": "F_L", "receiver": "U_R",
                 "follow_dist_m": 0.45, "min_travel_m": 0.10},
        objects=[TRAY_NAME],
        env_yaml=R26_ENV_TRAY,
    )


# R27 physical relay (2026-09-06): the flat 17x17x3 cm tray on the table has
# no pinchable rim (the hand cannot get under it) and a rim pinch would let it
# pitch about the pads, so the physical family relays a standing 4x4x40 cm
# BATON instead -- the "long-object handover" route of R32: the giver F_L
# pinches the top (S9 cube recipe, the baton hangs stable), carries it to the
# audited FL_UR lane, the receiver U_R side-pinches the body 3 cm ABOVE the
# centre of mass (hangs stable once the giver lets go), the two hands' finger
# spheres end up ~13 cm apart (cross line 3 cm). Recording env:
# duo_env_v7_r26_tray_r27r16.yaml carries the baton instead of the tray.
BATON_NAME = "baton_main"
# Indirect relay -> the hands never hold the baton together, so it only needs
# to stand and be clawed: 20 cm (top 1.00; a 30 cm one toppled when set down
# from 3 cm, r27_baton9; the DFX top claw on a 1.20 m top is beyond the UR5
# reach at the lane)
BATON_SIZE = (0.04, 0.04, 0.20)
BATON_POS = (0.52, -0.40, 0.90)          # standing, bottom on the table, top 1.00
BATON_LIFT = 0.10
BATON_MEET_XY = (0.10, -0.40)            # FL_UR lane, 5 cm towards U for the DFX claw reach
BATON_RECV_DZ = 0.03                     # receiver pinch above the COM
# giver pinch = the validated bottle recipe (body pinch 8.5 cm below the top,
# squeeze 0.12): a 2.5-3.5 cm top pinch let the baton slide 4 cm in the carry
BATON_GIVER_DEPTH = BOTTLE_PHYS_DEPTH
BATON_GIVER_SQUEEZE = BOTTLE_PHYS_SQUEEZE
BATON_U_HOME = (-0.33, -0.26)
BATON_U_REST = (-0.40, -0.28)
# DFX side pinch: the open aperture is thumb-heavy (thumb 6.4 cm / index
# 3.3 cm from the contact pinch centre), so an approach centred on the closed
# pinch runs the index finger 1.3 cm past the baton face and knocks the
# hanging baton away (r27_baton1: receiver closed on air, d 7.5 cm). Shift the
# receiver's approach line 1.5 cm towards the index side: the index passes
# 2.8 cm clear and the thumb pushes the baton onto the index pad on closing.
BATON_RECV_SHIFT = 0.015
# r27_baton2: the carried baton hung ~3 cm beyond the receiver's open pads
# (final_x 0.179 vs lane 0.15) -> reach 3 cm deeper along the approach so the
# baton lands between the phalanges and the thumb even with that offset
BATON_RECV_DEEPER = 0.0
# Grasp drag (r27_baton4 telemetry): the F2 pads arc towards the palm while
# closing and drag the light standing baton (+4.5, +4.7) cm before the pinch
# tightens; carried to the lane the baton hangs (+6.3, +4.3) cm off the pad
# design point while the receiver reaches its own target to <1 mm. The
# receiver's target is offset by the drag (pre-compensating the giver's carry
# instead pulls the F_L hand into the receiver's pre-grasp: cross 1.9 mm).
BATON_GIVER_DRAG_XY = (0.028, 0.017)   # measured for the finger/top_yaw0 body pinch (r27_baton10)
# DFX side pinch = thumb against a finger WALL. The DFX fingers retract ~8 cm
# along their own axis while closing (index pad x 0.242 -> 0.184 in the lane
# frame, r27_baton6 calibration replay), more than the 4 cm baton is wide, so
# closing them past contact slides their pads off the +y face and the thumb
# pushes the baton out between them (baton y -0.363 -> -0.308, then dropped).
# Park the fingers at ~0.35 (pads level with the baton axis) and let only the
# thumb travel (0.85): fingers f_contact-0.10, thumb f_contact+0.40.
BATON_RECV_SQUEEZE = -0.10
BATON_RECV_THUMB_EXTRA = 0.40


def _shift_cand(c: GraspCandidate, dxyz) -> GraspCandidate:
    """Copy of a candidate translated by a world vector (pre and grasp poses)."""
    d = np.asarray(dxyz, dtype=np.float64)
    return GraspCandidate(
        name=c.name, kind=c.kind, approach=c.approach, closing=c.closing, R_flange=c.R_flange,
        tcp_grasp=tuple((np.asarray(c.tcp_grasp) + d).tolist()),
        tcp_pre=tuple((np.asarray(c.tcp_pre) + d).tolist()),
        flange_grasp=tuple((np.asarray(c.flange_grasp) + d).tolist()),
        flange_pre=tuple((np.asarray(c.flange_pre) + d).tolist()),
        score=c.score, notes=dict(c.notes))
R26_OBJECTS[BATON_NAME] = {"name": BATON_NAME, "size": list(BATON_SIZE),
                           "init_pos": list(BATON_POS)}


def _baton_meet_center() -> tuple:
    return (BATON_MEET_XY[0], BATON_MEET_XY[1], BATON_POS[2] + BATON_LIFT)


def _baton_relay_task(cands: dict) -> S9Task:
    """Indirect relay (R32 route 4, 2026-09-06): the giver F_L top-pinches the
    standing baton, carries it into the FL_UR lane and stands it back on the
    table there, withdraws; the receiver U_R then claws its top from above (the
    validated DFX cube grasp) and carries it into the U half. A simultaneous
    two-hand hold of a 4 cm baton is kinematically marginal (DFX contact
    aperture ~4.4 cm with retracting fingers; two hands need >=25 cm of
    baton between them), see r27_baton1-8 in MASTER v2.49."""
    g, r = cands["F_L"], cands["U_R"]
    ori_g = {"F_L": g.R_flange}
    ori_r = {"U_R": r.R_flange}
    fg = g.flange_grasp
    # the F2 pads drag the standing baton by BATON_GIVER_DRAG_XY while
    # closing; the receiver's claw target carries that offset (pre-compensating
    # the giver's placement pushes F_L beyond its reach at the lane)
    dxy = (BATON_MEET_XY[0] - BATON_POS[0], BATON_MEET_XY[1] - BATON_POS[1])
    f_lift = (fg[0], fg[1], fg[2] + BATON_LIFT)
    f_carry = (fg[0] + dxy[0], fg[1] + dxy[1], fg[2] + BATON_LIFT)
    # set the baton down with its base ON the table (5 mm overshoot, the pads
    # yield), not hovering: released from 3 cm a thin baton topples
    f_place = (fg[0] + dxy[0], fg[1] + dxy[1], fg[2] - 0.005)
    f_clear = (f_place[0] + 0.10, f_place[1], f_place[2] + 0.10)
    rg = r.flange_grasp
    u_lift = (rg[0], rg[1], rg[2] + BATON_LIFT)
    u_home = (rg[0] + BATON_U_HOME[0] - BATON_MEET_XY[0],
              rg[1] + BATON_U_HOME[1] - BATON_MEET_XY[1], rg[2] + BATON_LIFT)
    u_rest = (u_home[0], u_home[1], u_home[2] + 0.02)

    phases = [
        _ph("reach", 2.5,
            targets={"F_L": g.flange_pre, "U_R": (-0.28, -0.32, 1.14),
                     "F_R": (0.46, 0.34, 1.20), "U_L": (-0.32, 0.34, 1.16)},
            oris=dict(ori_g)),
        _ph("descend", 1.5,
            targets={"F_L": fg, "F_R": (0.46, 0.34, 1.14), "U_L": (-0.30, 0.35, 1.14)},
            oris=dict(ori_g)),
        _ph("close", 1.4, targets={"U_L": (-0.18, 0.36, 1.14)}),
        _ph("lift", 1.5,
            targets={"F_L": f_lift, "F_R": (0.30, 0.39, 1.14)},
            oris=dict(ori_g)),
        _ph("meet", 3.0,
            targets={"F_L": f_carry},
            oris=dict(ori_g)),
        # stand the baton on the table at the lane (2.5 s: let the pendulum
        # swing settle before release)
        _ph("place", 2.5,
            targets={"F_L": f_place},
            oris=dict(ori_g)),
        _ph("settle_hold", 1.0),
        _ph("release", 1.5),
        _ph("clear", 1.5,
            targets={"F_L": f_clear},
            oris=dict(ori_g)),
        # giver retreats home while the receiver comes over the standing baton
        _ph("handoff", 2.5,
            targets={"F_L": (0.44, -0.48, 1.25), "U_R": r.flange_pre},
            oris={**ori_g, **ori_r}),
        _ph("descend_u", 1.5,
            targets={"U_R": rg},
            oris=dict(ori_r)),
        _ph("transfer", 1.4, targets={"F_R": (0.17, 0.41, 1.14)}),
        _ph("lift_u", 1.5,
            targets={"U_R": u_lift},
            oris=dict(ori_r)),
        _ph("carry_u", 2.5,
            targets={"U_R": u_home},
            oris=dict(ori_r),
            lerps={"U_L": dict(_BREATH)}),
        _ph("retreat", 2.5,
            targets={"U_R": u_rest, "F_R": (0.42, 0.48, 1.28), "U_L": (-0.36, 0.46, 1.16)},
            oris=dict(ori_r),
            lerps={"F_L": dict(_BREATH)}),
        _ph("settle", 1.0),
    ]
    return S9Task(
        family="tray_relay",
        spec=SkillSpec("s9_tray_relay", phases),
        hand_events=_close_events({"F_L": g}, ("F_L",), t_off=0.10, phase="close")
        + [("release", 0.10, "F_L", "open", 0.0, 1.0)]
        + _close_events({"U_R": r}, ("U_R",), t_off=0.10, phase="transfer"),
        object_events=[
            ("close", 0.90, "attach", "F_L", None, BATON_NAME, 0.13),
            ("release", 0.60, "release", None, None, BATON_NAME, None),
            ("transfer", 0.90, "attach", "U_R", (0.18, 0.6), BATON_NAME, None),
        ],
        success={"kind": "handover", "giver": "F_L", "receiver": "U_R",
                 "follow_dist_m": 0.45, "min_travel_m": 0.10},
        objects=[BATON_NAME],
        env_yaml=R26_ENV_TRAY,
    )


def _baton_relay_variants(provider):
    from safeduo.delta.grasp_gen import box_antipodal_grasps
    # where the baton actually stands after the giver's drag-offset placement
    standing = (BATON_MEET_XY[0] + BATON_GIVER_DRAG_XY[0], BATON_MEET_XY[1] + BATON_GIVER_DRAG_XY[1],
                BATON_POS[2])

    def giver(ap):
        prm = _params_for("F_L", BATON_SIZE[0], GRASP_PARAMS, ap, BATON_SIZE[2],
                          grasp_depth=BATON_GIVER_DEPTH, squeeze_extra=BATON_GIVER_SQUEEZE)
        ref_R, base_xy = _arm_prior(provider, "F_L")
        return [c for c in box_antipodal_grasps(BATON_POS, BATON_SIZE, prm, ref_R=ref_R, base_xy=base_xy)
                if c.kind == "top"]

    def receiver(ap):
        # DFX top claw on the standing baton's top end (cube recipe)
        prm = _params_for("U_R", BATON_SIZE[0], U_GRASP_PARAMS, ap, BATON_SIZE[2],
                          squeeze_extra=DFX_CUBE_SQUEEZE, thumb_extra=DFX_CUBE_THUMB_EXTRA)
        ref_R, base_xy = _arm_prior(provider, "U_R")
        return [c for c in box_antipodal_grasps(standing, BATON_SIZE, prm, ref_R=ref_R, base_xy=base_xy)
                if c.kind == "top"]

    gen = {"F_L": giver, "U_R": receiver}
    for tag, cands in _screened_combos(provider, "tray_relay", gen):
        yield tag, _baton_relay_task(cands)


def _tray_relay_variants(provider):
    if PHYS_GRASP["on"]:
        yield from _baton_relay_variants(provider)
        return
    # 西缘站位（朝 seam/搬运方向）：闭合线沿 x 横跨 rim —— 与
    # grip_station_cands 的顶抓过滤天然一致
    st_w = (TRAY_POS[0] - TRAY_GRIP_DX, TRAY_POS[1], TRAY_POS[2])
    f_w = grip_station_cands(provider, "F_L", st_w,
                             (0.04, 0.04, TRAY_SIZE[2]), GRASP_PARAMS)
    for k, c in enumerate(f_w[:2]):
        yield (f"rim_w_yaw{k}", _tray_relay_task(c))
    # 兜底：东缘站位（对侧 rim，闭合线仍沿 x）
    st_e = (TRAY_POS[0] + TRAY_GRIP_DX, TRAY_POS[1], TRAY_POS[2])
    f_e = grip_station_cands(provider, "F_L", st_e,
                             (0.04, 0.04, TRAY_SIZE[2]), GRASP_PARAMS)
    for k, c in enumerate(f_e[:1]):
        yield (f"rim_e_yaw{k}", _tray_relay_task(c))


# ---------------------------------------------------------------------------
# 任务族 3: box_colift —— F 双臂对端斜侧抓共抬大箱 + U 双臂交叉送货
# ---------------------------------------------------------------------------


def _box_colift_task(cands: dict, u_ori: bool = True) -> S9Task:
    """相位结构 = four_lift（bar 换大箱、端点顶抓换端面斜侧抓），U 双臂的
    交叉走廊 / weave 时间切片纪律原样保留（全部已审计航点）。"""
    fl, fr = cands["F_L"], cands["F_R"]
    ul, ur = cands["U_L"], cands["U_R"]
    ori_f = {"F_L": fl.R_flange, "F_R": fr.R_flange}
    ori_u = {"U_L": ul.R_flange, "U_R": ur.R_flange} if u_ori else {}
    fgl, fgr = fl.flange_grasp, fr.flange_grasp
    ugl, ugr = ul.flange_grasp, ur.flange_grasp
    lift_f, lift_u = 0.12, 0.11
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
        # 箱子向 seam 平移；U 双臂送 cube 进中线空域（four_lift x 车道）
        _ph("carry", 3.0,
            targets={"F_L": (fgl[0] + BOX_DX, fgl[1], fgl[2] + lift_f),
                     "F_R": (fgr[0] + BOX_DX, fgr[1], fgr[2] + lift_f),
                     "U_L": (-0.10, 0.26, 1.12), "U_R": (-0.12, -0.26, 1.12)},
            oris=dict(ori_f)),
        # 轮流进中心（four_lift 的时间切片纪律）
        _ph("weave_l", 2.0,
            targets={"U_L": (-0.10, 0.08, 1.13)},
            lerps={"F_L": dict(_BREATH)}),
        _ph("weave_r", 2.0,
            targets={"U_L": (-0.10, 0.30, 1.12), "U_R": (-0.12, -0.04, 1.13)}),
        _ph("setdown", 2.0,
            targets={"F_L": (fgl[0] + BOX_DX, fgl[1], fgl[2]),
                     "F_R": (fgr[0] + BOX_DX, fgr[1], fgr[2]),
                     "U_L": _above(place_ul, 0.10),
                     "U_R": (-0.32, -0.28, ugr[2] + 0.10)},
            oris=dict(ori_f)),
        _ph("place_u", 1.5,
            targets={"U_L": place_ul, "U_R": place_ur},
            oris=(dict(ori_u) if u_ori else None)),
        _ph("release", 1.0),
        # 松手后 F 双臂先竖直提净空再横撤（斜抓腕位形的横撤保险）
        _ph("clear", 1.2,
            targets={"F_L": _above((fgl[0] + BOX_DX, fgl[1], fgl[2]), 0.14),
                     "F_R": _above((fgr[0] + BOX_DX, fgr[1], fgr[2]), 0.14)},
            oris=dict(ori_f)),
        _ph("retreat", 2.5,
            targets={"F_L": (0.44, -0.30, 1.25), "F_R": (0.44, 0.30, 1.25),
                     "U_L": (-0.38, 0.42, 1.16), "U_R": (-0.40, -0.36, 1.16)}),
    ]
    # 端顶沿钩抓 TCP 离箱心：y 偏 0.09 + z 偏 ~0.11（站位抬高后）→
    # ~0.15（v3 实测 0.125 + 抬高 0.025），门取 0.20（对端站位是几何
    # 设计而非姿态错误；场景内无 0.20 内的其他可误附物体）
    return S9Task(
        family="box_colift",
        spec=SkillSpec("s9_box_colift", phases),
        hand_events=[
            ("close", 0.10, "F_L", "close", 0.35),
            ("close", 0.10, "F_R", "close", 0.35),
            ("close", 0.10, "U_L", "close", 0.60),
            ("close", 0.10, "U_R", "close", 0.60),
        ] + [("release", 0.30, a, "open", 0.0) for a in ARM_KEYS],
        object_events=[
            ("close", 0.90, "attach", "F_L", None, BOX_NAME, 0.20),
            ("close", 0.90, "attach", "U_L", None, CUBE_UL_NAME, None),
            ("close", 0.90, "attach", "U_R", None, CUBE_UR_NAME, None),
            ("release", 0.20, "release", None, None, BOX_NAME, None),
            ("release", 0.20, "release", None, None, CUBE_UL_NAME, None),
            ("release", 0.20, "release", None, None, CUBE_UR_NAME, None),
        ],
        success={"kind": "pick_place",
                 "place_xy": [BOX_POS[0] + BOX_DX, BOX_POS[1]],
                 "radius_m": 0.10, "min_disp_m": 0.10,
                 "rest_z": BOX_POS[2]},
        objects=[BOX_NAME, CUBE_UL_NAME, CUBE_UR_NAME],
        env_yaml=R26_ENV_BIGBOX,
    )


# R27 physical box co-lift: the 15x20x10 cm box exceeds any pinch aperture,
# so the two F hands clamp it between their palms (palm on each +/-y end face,
# fingers horizontal along -x, thumbs up). Position-controlled arms turn the
# BOX_PALM_PRESS penetration of the palm targets into the squeeze force
# (joint kp 400 ~ 1.5 kN/m Cartesian -> ~15 N per palm, friction >> 0.4 kg).
BOX_PALM_PRESS = 0.015
# light finger curl on contact: adds pad normal force on the face and pulls
# the fingertips up off the table; thumbs stay open (they point along -x)
BOX_F_CURL = 0.25
BOX_PALM_PRE = 0.06
BOX_PHYS_LIFT = 0.10
BOX_PHYS_DX = 0.14
# palm centre 3.5 cm above the box top (box centre +0.085): with the fingers
# pointing down the knuckles sit at the top edge and the open fingertips
# (11.5 cm below the palm centre) stay ~5 mm above the table. Opposite hands
# at equal height -> pure squeeze, no tipping torque.
BOX_PALM_DZ = 0.085


def _palm_cand(arm: str, name: str, face_pt, normal_w, finger_w, pre_dist: float,
               press: float) -> GraspCandidate:
    """GraspCandidate whose flange_grasp puts the palm surface ``press`` inside
    the face at ``face_pt`` (outward ``normal_w`` is the palm-facing direction
    from the hand towards the box) and whose flange_pre sits ``pre_dist``
    outside the face."""
    from safeduo.delta.hand_grasp_frame import flange_for_palm, load_hand_calib, palm_frame
    pf = palm_frame(load_hand_calib(PHYS_GRASP["calib"]), arm)
    n = np.asarray(normal_w, dtype=np.float64)
    n = n / np.linalg.norm(n)
    fp = np.asarray(face_pt, dtype=np.float64)
    R, t_g = flange_for_palm(pf, fp + n * press, n, finger_w)
    _, t_p = flange_for_palm(pf, fp - n * pre_dist, n, finger_w)
    return GraspCandidate(
        name=name, kind="palm", approach=tuple(n.tolist()), closing=tuple(np.asarray(finger_w, float).tolist()),
        R_flange=tuple(tuple(round(float(v), 6) for v in row) for row in R),
        tcp_grasp=tuple(round(float(v), 6) for v in (fp + n * press)),
        tcp_pre=tuple(round(float(v), 6) for v in (fp - n * pre_dist)),
        flange_grasp=tuple(round(float(v), 6) for v in t_g),
        flange_pre=tuple(round(float(v), 6) for v in t_p),
        notes={"palm_frame": pf.meta(), "press_m": press})


def _box_colift_phys_task(cands: dict) -> S9Task:
    """Bimanual palm clamp: both F arms press, lift, carry +x, set down and
    unclamp together; the U arms run their own cube pick/place (rod family)."""
    fl, fr = cands["F_L"], cands["F_R"]
    ul, ur = cands["U_L"], cands["U_R"]
    ori_f = {"F_L": fl.R_flange, "F_R": fr.R_flange}
    ori_u = {"U_L": ul.R_flange, "U_R": ur.R_flange}
    fgl, fgr = fl.flange_grasp, fr.flange_grasp
    ugl, ugr = ul.flange_grasp, ur.flange_grasp
    lift_u = 0.11
    place_ul = (-0.30, 0.28, _place_z(ugl, 0.015))
    place_ur = (-0.32, -0.28, _place_z(ugr, 0.015))
    fgl_c = (fgl[0] + BOX_PHYS_DX, fgl[1], fgl[2])
    fgr_c = (fgr[0] + BOX_PHYS_DX, fgr[1], fgr[2])
    # unclamp: palms back out along their own approach (away from the box)
    out_l = (fgl_c[0], fgl_c[1] - 0.05, fgl_c[2])
    out_r = (fgr_c[0], fgr_c[1] + 0.05, fgr_c[2])

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
            targets={"F_L": _above(fgl, BOX_PHYS_LIFT), "F_R": _above(fgr, BOX_PHYS_LIFT),
                     "U_L": _above(ugl, lift_u), "U_R": _above(ugr, lift_u)},
            oris={**ori_f, **ori_u}),
        _ph("carry", 3.0,
            targets={"F_L": _above(fgl_c, BOX_PHYS_LIFT), "F_R": _above(fgr_c, BOX_PHYS_LIFT),
                     "U_L": (-0.10, 0.26, 1.12), "U_R": (-0.12, -0.26, 1.12)},
            oris=dict(ori_f)),
        _ph("carry_u", 1.5,
            targets={"U_L": _above(place_ul, 0.10), "U_R": _above(place_ur, 0.10)},
            oris=dict(ori_u), lerps={"F_L": dict(_BREATH)}),
        _ph("place", 2.5,
            targets={"F_L": fgl_c, "F_R": fgr_c, "U_L": place_ul, "U_R": place_ur},
            oris={**ori_f, **ori_u}),
        _ph("release", 1.0),
        _ph("clear", 1.5,
            targets={"F_L": out_l, "F_R": out_r},
            oris=dict(ori_f)),
        # palms-down hands: rise with the orientation locked before the free
        # retreat (an unconstrained retreat from box height folds the wrist,
        # self 1.6 mm)
        _ph("clear_up", 1.5,
            targets={"F_L": _above(out_l, 0.15), "F_R": _above(out_r, 0.15)},
            oris=dict(ori_f)),
        _ph("retreat", 2.5,
            targets={"F_L": (0.44, -0.30, 1.25), "F_R": (0.44, 0.30, 1.25),
                     "U_L": (-0.38, 0.42, 1.16), "U_R": (-0.40, -0.36, 1.16)}),
    ]
    return S9Task(
        family="box_colift",
        spec=SkillSpec("s9_box_colift", phases),
        # F hands: palms do the work, fingers curl lightly onto the face
        # (thumb kept open); U hands pinch cubes
        hand_events=_close_events(cands, ("U_L", "U_R")) + [
            ("close", 0.10, a, "close", BOX_F_CURL, 0.6, 0.0) for a in ("F_L", "F_R")] + [
            ("release", 0.75, a, "open", 0.0) for a in ARM_KEYS],
        object_events=[
            ("close", 0.90, "attach", "F_L", None, BOX_NAME, 0.20),
            ("close", 0.90, "attach", "U_L", None, CUBE_UL_NAME, None),
            ("close", 0.90, "attach", "U_R", None, CUBE_UR_NAME, None),
            ("release", 0.60, "release", None, None, BOX_NAME, None),
            ("release", 0.60, "release", None, None, CUBE_UL_NAME, None),
            ("release", 0.60, "release", None, None, CUBE_UR_NAME, None),
        ],
        success={"kind": "pick_place",
                 "place_xy": [BOX_POS[0] + BOX_PHYS_DX, BOX_POS[1]],
                 "radius_m": 0.10, "min_disp_m": 0.10,
                 "rest_z": BOX_POS[2]},
        objects=[BOX_NAME, CUBE_UL_NAME, CUBE_UR_NAME],
        env_yaml=R26_ENV_BIGBOX,
    )


def _box_colift_phys_variants(provider):
    face_l = (BOX_POS[0], BOX_POS[1] - BOX_SIZE[1] / 2, BOX_POS[2] + BOX_PALM_DZ)
    face_r = (BOX_POS[0], BOX_POS[1] + BOX_SIZE[1] / 2, BOX_POS[2] + BOX_PALM_DZ)
    # fingers straight down along the face (palm centre 3.5 cm above the box
    # top: knuckles at the top edge, the finger phalanges cover the upper
    # 8 cm of the face, fingertips ~5 mm above the table, thumbs horizontal
    # towards the table centre). Sideways palms fail: thumb-up forces the
    # fingers towards -x, whose IK folds the arm (j2 beyond -1.76), and
    # thumb-down drops the 12 cm thumb onto the table (r27_box_palm_diag.py).
    finger_opts = {"fz-": (0.0, 0.0, -1.0)}
    gen = {
        "F_L": lambda ap: [_palm_cand("F_L", f"palm_{k}", face_l, (0.0, 1.0, 0.0), fw,
                                      BOX_PALM_PRE, BOX_PALM_PRESS) for k, fw in finger_opts.items()],
        "F_R": lambda ap: [_palm_cand("F_R", f"palm_{k}", face_r, (0.0, -1.0, 0.0), fw,
                                      BOX_PALM_PRE, BOX_PALM_PRESS) for k, fw in finger_opts.items()],
        "U_L": lambda ap: _u_cube_tops(provider, ap)[0],
        "U_R": lambda ap: _u_cube_tops(provider, ap)[1],
    }
    for tag, cands in _screened_combos(provider, "box_colift", gen):
        yield tag, _box_colift_phys_task(cands)


def _box_colift_variants(provider):
    if PHYS_GRASP["on"]:
        yield from _box_colift_phys_variants(provider)
        return
    u_l, u_r = _u_cube_tops(provider)
    base_u = {"U_L": u_l[0], "U_R": u_r[0]}
    # 首选：端顶沿站位顶抓 —— four_lift bar 端点抓取的同款竖直腕位形
    # （题面钦点的参考做法；第二轮 35/50 度端面斜抓 self 1.8-2.0mm 全毙）。
    # F_L（attach 主）站位内缩 1cm 真抓箱端；F_R 站位在 +y 面外 5.5cm
    # 扶托（掌面离箱面 >=3cm —— v1 体内站位、v2 外 2.5cm 站位都在 carry
    # 段被 droop 增量吃掉间隙引发角力，见 BOX_DX 注释），间距 0.245
    # 双站位抬 5cm：指尖悬在箱顶上方 ~3cm，物理上永不触箱。帧审定案
    # （v3 指尖低箱顶 5mm -> setdown 被顶住悬空 12cm 摔箱；v4 抬 2.5cm
    # 后指尖仅高 6mm，sim 跟踪误差吃掉间隙，lift 段箱体绕指尖倾摆、
    # 规划姿态探针漂移 <=1.2 度证明不是规划的锅）。竖直向指尖-附着物
    # 接触是角力雷，见瓶族悬停注释
    st_l = (BOX_POS[0], BOX_POS[1] - (BOX_SIZE[1] / 2 - 0.01),
            BOX_POS[2] + 0.05)
    st_r = (BOX_POS[0], BOX_POS[1] + BOX_SIZE[1] / 2 + 0.055,
            BOX_POS[2] + 0.05)
    ssz = (0.04, 0.04, BOX_SIZE[2])
    fls = grip_station_cands(provider, "F_L", st_l, ssz, GRASP_PARAMS)
    frs = grip_station_cands(provider, "F_R", st_r, ssz, GRASP_PARAMS)
    for k in range(min(len(fls), len(frs), 2)):
        yield (f"endtop_yaw{k}",
               _box_colift_task({"F_L": fls[k], "F_R": frs[k], **base_u}))
    # 兜底：浅斜侧抓（20/30 度）
    thetas = (0.35, 0.52)
    tcp_l = (BOX_POS[0], BOX_POS[1] - BOX_SIZE[1] / 2 + 0.01, BOX_GRIP_Z)
    tcp_r = (BOX_POS[0], BOX_POS[1] + BOX_SIZE[1] / 2 - 0.01, BOX_GRIP_Z)
    pls = _pitched_cands(tcp_l, (0.0, 1.0), thetas)
    prs = _pitched_cands(tcp_r, (0.0, -1.0), thetas)
    for k in range(min(len(pls), len(prs))):
        yield (f"end_{pls[k].name}",
               _box_colift_task({"F_L": pls[k], "F_R": prs[k], **base_u}))
    yield ("end_pitch20/u_pos_only",
           _box_colift_task({"F_L": pls[0], "F_R": prs[0], **base_u},
                            u_ori=False))


# ---------------------------------------------------------------------------
# 任务族 4: rod_pick —— F 双臂各自搬运长棒 + U 双臂各自搬 cube
# ---------------------------------------------------------------------------


def _rod_pick_task(cands: dict, u_ori: bool = True) -> S9Task:
    """四臂各自 pick&place（同 bottle_pick 骨架）。棒中点顶抓 = four_lift
    杆端抓的同款横向跨骑几何（指尖在棒两侧下探、纯侧向夹压 —— 对运动学
    附着物无害的接触类，见瓶族悬停注释的反例说明）。放置带 -0.01 过冲：
    clamp 把棒按在 rest 高度、settle 写平，躺棒无倾倒模式。"""
    fl, fr = cands["F_L"], cands["F_R"]
    ul, ur = cands["U_L"], cands["U_R"]
    ori_f = {"F_L": fl.R_flange, "F_R": fr.R_flange}
    ori_u = {"U_L": ul.R_flange, "U_R": ur.R_flange} if u_ori else {}
    fgl, fgr = fl.flange_grasp, fr.flange_grasp
    ugl, ugr = ul.flange_grasp, ur.flange_grasp
    lift_f, lift_u = 0.12, 0.11
    pl_l = (fgl[0] + ROD_DX, fgl[1], _place_z(fgl, -0.01))
    pl_r = (fgr[0] + ROD_DX, fgr[1], _place_z(fgr, -0.01))
    place_ul = (-0.30, 0.28, _place_z(ugl, 0.015))
    place_ur = (-0.32, -0.28, _place_z(ugr, 0.015))

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
        _ph("carry", 3.0,
            targets={"F_L": _above(pl_l, lift_f + 0.01),
                     "F_R": _above(pl_r, lift_f + 0.01),
                     "U_L": (-0.10, 0.26, 1.12), "U_R": (-0.12, -0.26, 1.12)},
            oris=dict(ori_f)),
        _ph("carry_u", 1.5,
            targets={"U_L": _above(place_ul, 0.10),
                     "U_R": _above(place_ur, 0.10)},
            oris=(dict(ori_u) if u_ori else None),
            lerps={"F_L": dict(_BREATH)}),
        # 2.5s：给 sim 侧向/垂直跟踪误差收敛时间（瓶族 v6 教训）
        _ph("place", 2.5,
            targets={"F_L": pl_l, "F_R": pl_r,
                     "U_L": place_ul, "U_R": place_ur},
            oris={**ori_f, **ori_u}),
        _ph("release", 1.0),
        _ph("clear", 1.2,
            targets={"F_L": _above(pl_l, 0.14), "F_R": _above(pl_r, 0.14)},
            oris=dict(ori_f)),
        _ph("retreat", 2.5,
            targets={"F_L": (0.44, -0.30, 1.25), "F_R": (0.44, 0.30, 1.25),
                     "U_L": (-0.38, 0.42, 1.16), "U_R": (-0.40, -0.36, 1.16)}),
    ]
    # gate_tcp 0.26 = four_lift 杆同值（站位在棒内端而非棒心，TCP 到
    # 棒心距离天然 ~0.11，必须按长物体口径放宽）
    rod_l_pos, _, _, _ = _rod_layout()
    return S9Task(
        family="rod_pick",
        spec=SkillSpec("s9_rod_pick", phases),
        hand_events=_close_events(cands, ARM_KEYS) + [
            ("release", 0.75, a, "open", 0.0) for a in ARM_KEYS],
        object_events=[
            ("close", 0.90, "attach", "F_L", None, ROD_L_NAME, 0.26),
            ("close", 0.90, "attach", "F_R", None, ROD_R_NAME, 0.26),
            ("close", 0.90, "attach", "U_L", None, CUBE_UL_NAME, None),
            ("close", 0.90, "attach", "U_R", None, CUBE_UR_NAME, None),
            ("release", 0.60, "release", None, None, ROD_L_NAME, None),
            ("release", 0.60, "release", None, None, ROD_R_NAME, None),
            ("release", 0.60, "release", None, None, CUBE_UL_NAME, None),
            ("release", 0.60, "release", None, None, CUBE_UR_NAME, None),
        ],
        success={"kind": "pick_place",
                 "place_xy": [rod_l_pos[0] + ROD_DX, rod_l_pos[1]],
                 "radius_m": 0.10, "min_disp_m": 0.12,
                 "rest_z": rod_l_pos[2]},
        objects=[ROD_L_NAME, ROD_R_NAME, CUBE_UL_NAME, CUBE_UR_NAME],
        env_yaml=R26_ENV_ROD,
    )


def _screened_combos(provider, family: str, gen: dict, max_k: int = 3):
    """R27: per-arm feasibility screening over approach modes, then combine
    the k-th feasible candidate of every arm. gen: arm -> callable(approach)
    returning the arm's candidate list for that approach mode."""
    per_arm = {}
    for arm, fn in gen.items():
        feas = []
        for ap in PHYS_APPROACHES:
            feas += [(ap, c) for c in _screen(provider, arm, fn(ap), max_keep=2)]
        per_arm[arm] = feas
        print(f"R26_SCREEN_SUMMARY {family} {arm}: {[(a, c.name) for a, c in feas]}", flush=True)
    if not all(per_arm[a] for a in gen):
        return
    n = max(len(per_arm[a]) for a in gen)
    for k in range(min(n, max_k)):
        pick = {a: per_arm[a][min(k, len(per_arm[a]) - 1)] for a in gen}
        tag = "+".join(f"{a}:{ap[:2]}{c.name[-3:]}" for a, (ap, c) in pick.items())
        yield f"phys_screened{k}/{tag}", {a: c for a, (ap, c) in pick.items()}


def _rod_pick_variants(provider):
    rod_l, rod_r, grip_in, size = _rod_layout()
    st_l = (rod_l[0], rod_l[1] + grip_in, rod_l[2])
    st_r = (rod_r[0], rod_r[1] - grip_in, rod_r[2])
    if PHYS_GRASP["on"]:
        gen = {
            "F_L": lambda ap: grip_station_cands(provider, "F_L", st_l, size[0],
                                                 _params_for("F_L", size[0], GRASP_PARAMS, ap, size[2],
                                                             grasp_depth=ROD_PHYS_DEPTH)),
            "F_R": lambda ap: grip_station_cands(provider, "F_R", st_r, size[0],
                                                 _params_for("F_R", size[0], GRASP_PARAMS, ap, size[2],
                                                             grasp_depth=ROD_PHYS_DEPTH)),
            "U_L": lambda ap: _u_cube_tops(provider, ap)[0],
            "U_R": lambda ap: _u_cube_tops(provider, ap)[1],
        }
        for tag, cands in _screened_combos(provider, "rod_pick", gen):
            yield tag, _rod_pick_task(cands)
        return
    for apf, apu in _approach_pairs():
        tag_ap = f"/F{apf}-U{apu}" if apf else ""
        u_l, u_r = _u_cube_tops(provider, apu)
        base_u = {"U_L": u_l[0], "U_R": u_r[0]}
        fls = grip_station_cands(provider, "F_L", st_l, ROD_SIZE[0],
                                 _params_for("F_L", ROD_SIZE[0], GRASP_PARAMS, apf, ROD_SIZE[2]))
        frs = grip_station_cands(provider, "F_R", st_r, ROD_SIZE[0],
                                 _params_for("F_R", ROD_SIZE[0], GRASP_PARAMS, apf, ROD_SIZE[2]))
        for k in range(min(len(fls), len(frs), 2)):
            yield (f"midtop_yaw{k}{tag_ap}",
                   _rod_pick_task({"F_L": fls[k], "F_R": frs[k], **base_u}))
        yield (f"midtop_yaw0/u_pos_only{tag_ap}",
               _rod_pick_task({"F_L": fls[0], "F_R": frs[0], **base_u},
                              u_ori=False))


# ---------------------------------------------------------------------------
# 构建入口（与 build_task_r24 同一套硬门槛：IK/逐步审计/participation）
# ---------------------------------------------------------------------------

_VARIANTS = {
    "bottle_pick": _bottle_pick_variants,
    "tray_relay": _tray_relay_variants,
    "box_colift": _box_colift_variants,
    "rod_pick": _rod_pick_variants,
}


def build_task_r26(provider, family: str, out_dir, verbose: bool = True) -> dict:
    """R26 族构建：变体逐个试解（IK 不收敛 / 逐步审计 FAIL / 闲臂 -> 下一
    个变体），第一个全绿的胜出 —— 与 build_task_r24 同口径。"""
    import json
    from pathlib import Path

    from safeduo.delta.task_record_s9 import _compose_and_validate, resolve_events

    last_err = None
    for tag, task in _VARIANTS[family](provider):
        try:
            traj, rep = _compose_and_validate(provider, task, None)
        except RuntimeError as e:
            last_err = f"{tag}: {e}"
            print(f"R26_VARIANT_REJECTED {family} {tag}: {e}", flush=True)
            continue
        if "FAIL" in rep:
            last_err = f"{tag}: {rep['FAIL']}"
            print(f"R26_VARIANT_REJECTED {family} {tag}: {rep['FAIL']}",
                  flush=True)
            continue
        rep["participation"] = audit_arm_participation(provider, traj)
        if not rep["participation"]["all_arms_moving"]:
            rep["FAIL"] = f"idle arm: {rep['participation']}"
            last_err = f"{tag}: {rep['FAIL']}"
            print(f"R26_VARIANT_REJECTED {family} {tag}: {rep['FAIL']}",
                  flush=True)
            continue
        break
    else:
        raise RuntimeError(f"{family}: 所有 R26 变体都不过硬门槛，最后错误: "
                           f"{last_err}")

    phases, hand, objev, success = resolve_events(task)
    assert phases[-1]["t1"] <= traj.duration + 1e-6
    primary = task.objects[0]
    objs = {n: dict(R26_OBJECTS[n]) for n in task.objects}
    if PHYS_GRASP["on"]:
        rod_l, rod_r, _, size = _rod_layout()
        for n, p in ((ROD_L_NAME, rod_l), (ROD_R_NAME, rod_r)):
            if n in objs:
                objs[n]["init_pos"] = list(p)
                objs[n]["size"] = list(size)
    traj.meta["s9_task"] = {
        "family": task.family,
        "object": objs[primary],
        "objects": [objs[n] for n in task.objects],
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


def main(argv: "list | None" = None) -> int:
    import argparse
    import json
    import time
    from pathlib import Path

    from safeduo.delta.task_record_s9 import make_s9_provider

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["build"])
    ap.add_argument("--task", default=None, choices=list(R26_FAMILIES),
                    help="one family; default all three")
    ap.add_argument("--out", default="artifacts/task_trajs_r26")
    ap.add_argument("--hand-calib", default=None,
                    help="R27: 'v7' 或标定 JSON 路径 —— 物理抓取几何（实测抓取系/夹紧比例/放置间隙）")
    ap.add_argument("--spheres", default="v7", choices=["v7", "r16"],
                    help="设计期审计球包（r16 = 细化手指壳，物理抓取用）")
    ap.add_argument("--pre-grasp", type=float, default=None,
                    help="R27: 预抓取偏移（m），缺省用 GRASP_PARAMS 的 0.10")
    ap.add_argument("--struct-exempt", action="store_true",
                    help="R27: 设计期守卫豁免结构性桌面行（与 R29 同口径）")
    ap.add_argument("--multiseed", action="store_true",
                    help="R27: waypoint IK 被挡时用镜像臂构型重新起解（相位插值另行审计）")
    args = ap.parse_args(argv)
    if args.struct_exempt:
        from safeduo.delta.skill_record import DESIGN_STRUCT_EXEMPT
        DESIGN_STRUCT_EXEMPT["on"] = True
        print("DESIGN_STRUCT_EXEMPT on", flush=True)
    if args.multiseed:
        from safeduo.delta.skill_record_v7 import IK_MULTISEED
        IK_MULTISEED["on"] = True
        IK_MULTISEED["verbose"] = True
        print("IK_MULTISEED on", flush=True)

    if args.hand_calib:
        PHYS_GRASP["on"] = True
        PHYS_GRASP["calib"] = args.hand_calib
        PHYS_GRASP["pre"] = args.pre_grasp
        print(f"R26_PHYS_GRASP on calib={args.hand_calib} spheres={args.spheres}", flush=True)
    provider = make_s9_provider(1, device="cpu", spheres=args.spheres)
    names = [args.task] if args.task else list(R26_FAMILIES)
    out_root = Path(args.out)
    summary, n_fail = {}, 0
    for name in names:
        t0 = time.time()
        try:
            rep = build_task_r26(provider, name, out_root / name)
        except RuntimeError as e:
            print(f"R26_BUILD_FAILED {name}: {e}", flush=True)
            summary[name] = {"FAIL": str(e)}
            n_fail += 1
            continue
        summary[name] = {
            "hard_gate_margin_gt0": rep.get("hard_gate_margin_gt0"),
            "build_s": round(time.time() - t0, 1),
            "participation": rep["participation"],
        }
        if "FAIL" in rep:
            n_fail += 1
    sp = out_root / "r26_build_summary.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(sp.read_text()) if sp.exists() else {}
    old.update(summary)
    sp.write_text(json.dumps(old, indent=2, ensure_ascii=False))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
