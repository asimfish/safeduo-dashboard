"""解析式 antipodal 抓取位姿生成（R23 抓取质量修复, 2026-08-28）。

为什么存在：S9 任务演示被 owner 判定"cube 是穿模蹭起来的，不是真抓"。
三根因之一是轨迹只求位置 IK、手掌姿态与物体不匹配。本模块对 box/cuboid
物体解析生成 antipodal 抓取候选（顶抓 + 四个侧抓 × 偏航变体），每个候选
带完整的位姿链：预抓取位姿（沿接近轴退开 pre_grasp_offset）→ 抓取位姿
（含抓取深度）→ 腕姿态（闭合线对齐物体主轴）。纯几何、无 Isaac 依赖，
本地可测；IK 可达性由消费方（task_record_s9 的 design 阶段）逐候选试解
并按硬门槛 margin 审计裁决，这里只按几何先验排序（排序 = 试解顺序）。

坐标/姿态约定（与 real_geometry 的 FR3_FLANGE_V4 / v7 UR5_DFX_FLANGE 一致）：
  - flange 系 +Z = 手的延伸方向（F2/DFX 手都沿 flange +Z 悬挂 ~0.19-0.21 m）；
  - flange 系 +X ≈ 拇指对指闭合线（RH56 系手：四指列沿 Y 排开、拇指在 +X
    侧 —— 见 real_geometry._F2_TREE 的 thumb_1 x 偏移 +0.018）；
  - 世界系 R_flange 列 = [closing, binormal, approach]，approach 从手指向
    物体（顶抓 = (0,0,-1)）。
TCP（抓取参考点）= flange + R_flange @ tcp_offset。默认放在指尖捏合区
（z≈0.165）而不是掌心：5 cm cube 放在桌面（顶面 z=0.85）上，掌心包覆要求
flange 压到 ~0.97，指尖球会低于桌面 0.80 → 球模型 table 通道直接判负。
捏合 cube 上半段是这套碰撞模型下几何可行且视觉正确的唯一抓法。

cuRobo 对接：本模块只产"抓取位姿候选"，与 cuRobo plan_grasp 的输入语义
一致（grasp pose + pre-grasp offset）。后续接 cuRobo 时直接把候选喂给
plan_grasp 即可（见 traj_planner.CuRoboPlanner 的说明）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

_WORLD_UP = np.array([0.0, 0.0, 1.0])


@dataclass
class GraspGenParams:
    """抓取生成参数（全部可调；默认值按 v7 场景 5 cm cube 标定）。

    tcp_offset        flange 系下的抓取捏合点（F2 手指尖捏合区 z≈0.165；
                      掌心包覆约 0.13，但见模块 docstring —— 桌面物体禁用）
    pre_grasp_offset  预抓取位姿沿接近轴退开的距离（owner 要求 8-12 cm）
    grasp_depth       顶抓：TCP 低于物体顶面的下探深度（捏合深度）
    side_depth        侧抓：TCP 越过接近面进入物体的深度
    lift_height       抓取后竖直提升高度（供编排层引用，统一来源）
    top_yaws          顶抓的偏航变体（闭合线绕世界 Z 的角度，对齐 box 主轴）
    yaw_ref_weight    与参考腕姿态（出生位姿 FK）的测地距离扣分权重 ——
                      腕子拧得越少越"自然"，也越可能一次 IK 收敛
    top_prior         顶抓相对侧抓的先验加分（桌面物体顶抓姿态余量最大）
    approach_base_weight  侧抓：接近方向与"基座→物体"方向一致的加分
                      （臂从自己一侧探过去顺手，从远侧绕过来别扭）
    min_flange_z      场景注入的 flange 最低可行高度（= 桌顶 + 手最低碰撞
                      球悬垂 + 设计余量；由调用方用几何 provider 标定）。
                      候选低于它则整条位姿链抬升（notes 记 z_lifted_m）：
                      桌面物体的"深捏"受碰撞球模型钳制 —— 球包比视觉手指
                      粗一圈，审计过不了的深度不生成，抬到审计干净的高度，
                      视觉细指尖仍会包住物体上半段。
    """

    tcp_offset: tuple = (0.0, 0.0, 0.165)
    pre_grasp_offset: float = 0.10
    grasp_depth: float = 0.015
    side_depth: float = 0.010
    lift_height: float = 0.15
    top_yaws: tuple = (0.0, math.pi / 2, math.pi, -math.pi / 2)
    yaw_ref_weight: float = 0.3
    top_prior: float = 1.0
    approach_base_weight: float = 0.2
    min_flange_z: "float | None" = None
    # R27 S1: measured hand grasp frame (delta/hand_grasp_frame.HandGraspFrame).
    # None = legacy geometry (pinch point on the flange axis at tcp_offset,
    # thumb closing along flange +X) bit-for-bit. Set = the pinch point and
    # closing direction come from the calibration sweep, so the flange pose
    # is solved such that the REAL pad midpoint lands on the grasp point.
    hand_frame: "object | None" = None

    def meta(self) -> dict:
        out = {
            "tcp_offset": [round(float(v), 4) for v in self.tcp_offset],
            "pre_grasp_offset": self.pre_grasp_offset,
            "grasp_depth": self.grasp_depth,
            "side_depth": self.side_depth,
            "lift_height": self.lift_height,
            "yaw_ref_weight": self.yaw_ref_weight,
            "top_prior": self.top_prior,
            "min_flange_z": self.min_flange_z,
        }
        if self.hand_frame is not None:
            out["hand_frame"] = self.hand_frame.meta()
        return out


@dataclass
class GraspCandidate:
    """一个可执行的抓取候选（全部世界系；tuple 保证 JSON 可序列化）。

    approach 从手指向物体的单位向量（= 世界系下的 flange +Z 目标）；
    closing  抓取闭合线单位向量（= 世界系下的 flange +X 目标）；
    flange_* 是 IK 的直接目标（TCP 目标经 tcp_offset 反解到 flange）。
    """

    name: str
    kind: str                      # "top" | "side"
    approach: tuple
    closing: tuple
    R_flange: tuple                # 3x3 嵌套 tuple，列 = [closing, binormal, approach]
    tcp_grasp: tuple
    tcp_pre: tuple
    flange_grasp: tuple
    flange_pre: tuple
    score: float = 0.0
    notes: dict = field(default_factory=dict)

    def meta(self) -> dict:
        """写进轨迹 npz meta 的 JSON 块（录制端用它做 attach 位姿门控）。"""
        return {
            "name": self.name, "kind": self.kind,
            "approach": list(self.approach), "closing": list(self.closing),
            "R_flange": [list(r) for r in self.R_flange],
            "tcp_grasp": list(self.tcp_grasp), "tcp_pre": list(self.tcp_pre),
            "flange_grasp": list(self.flange_grasp),
            "flange_pre": list(self.flange_pre),
            "score": round(self.score, 4),
            "notes": dict(self.notes),
        }


def rot_from_closing_approach(closing, approach) -> np.ndarray:
    """由闭合线(X)与接近轴(Z)构造 flange 旋转矩阵（列向量 = 轴）。

    closing 会被投影到与 approach 垂直的平面再归一（容忍输入不严格正交）。
    """
    a = np.asarray(approach, dtype=np.float64)
    a = a / np.linalg.norm(a)
    c = np.asarray(closing, dtype=np.float64)
    c = c - a * float(c @ a)
    n = np.linalg.norm(c)
    if n < 1e-9:
        raise ValueError("closing 与 approach 共线，无法定义腕姿态")
    c = c / n
    b = np.cross(a, c)             # binormal = Z × X = Y（右手系）
    return np.stack([c, b, a], axis=1)


def _geodesic_angle(R0: np.ndarray, R1: np.ndarray) -> float:
    """两旋转的测地角 [0, pi]（腕子扭转代价的量度）。"""
    cos = (np.trace(R0.T @ R1) - 1.0) * 0.5
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def _mk_candidate(name, kind, obj_pos, tcp_grasp, approach, closing,
                  params: GraspGenParams) -> GraspCandidate:
    tcp_grasp = np.asarray(tcp_grasp, dtype=np.float64)
    notes = {}
    if params.hand_frame is not None:
        # R27 S1: wrist pose from the MEASURED hand frame -- the pad closing
        # direction maps onto `closing`, the pad approach direction onto
        # `approach`, and the flange is placed so the pad pinch midpoint (not
        # the flange axis) sits on the grasp point.
        from safeduo.delta.hand_grasp_frame import rot_from_hand_frame

        R = rot_from_hand_frame(closing, approach, params.hand_frame)
        a = np.asarray(approach, dtype=np.float64)
        a = a / np.linalg.norm(a)
        off = R @ np.asarray(params.hand_frame.pinch_mid, dtype=np.float64)
        closing_w = R @ np.asarray(params.hand_frame.closing_dir, dtype=np.float64)
        notes["hand_frame"] = params.hand_frame.meta()
    else:
        R = rot_from_closing_approach(closing, approach)
        a = R[:, 2]
        off = R @ np.asarray(params.tcp_offset, dtype=np.float64)
        closing_w = R[:, 0]
    if params.min_flange_z is not None:
        # 场景钳位：flange 不能低于审计可行高度 → 整条位姿链竖直抬升
        # （抓得浅一点，而不是生成一个必然被硬门槛毙掉的深捏）
        dz = params.min_flange_z - float(tcp_grasp[2] - off[2])
        if dz > 0.0:
            tcp_grasp = tcp_grasp + np.array([0.0, 0.0, dz])
            notes["z_lifted_m"] = round(dz, 4)
    tcp_pre = tcp_grasp - a * params.pre_grasp_offset
    return GraspCandidate(
        name=name, kind=kind,
        approach=tuple(round(float(v), 6) for v in a),
        closing=tuple(round(float(v), 6) for v in closing_w),
        R_flange=tuple(tuple(round(float(v), 6) for v in row) for row in R),
        tcp_grasp=tuple(round(float(v), 6) for v in tcp_grasp),
        tcp_pre=tuple(round(float(v), 6) for v in tcp_pre),
        flange_grasp=tuple(round(float(v), 6) for v in (tcp_grasp - off)),
        flange_pre=tuple(round(float(v), 6) for v in (tcp_pre - off)),
        notes=notes,
    )


def box_antipodal_grasps(obj_pos, obj_size,
                         params: "GraspGenParams | None" = None,
                         ref_R=None, base_xy=None) -> list:
    """对 axis-aligned box 生成 antipodal 抓取候选（顶抓×4偏航 + 侧抓×4面）。

    obj_pos   物体中心世界坐标 (x, y, z)
    obj_size  边长（标量或 [sx, sy, sz]，与 task_objects 的 size 语义一致）
    ref_R     参考腕姿态（3x3，通常传出生位姿 FK 的 R_flange）：候选按与它
              的测地距离扣分 —— 腕子少拧 = 更自然 = IK 更易收敛
    base_xy   持抓臂基座平面位置：侧抓按"从基座一侧接近"加分

    返回按 score 降序的候选列表（消费方按序试解 IK，第一个过硬门槛的胜出）。
    box 的 antipodal 性质：闭合线沿任一主轴时两接触面平行对置，天然满足
    对趾条件 —— 所以候选枚举 = 主轴方向的组合，不需要采样接触点对。
    """
    p = params or GraspGenParams()
    obj_pos = np.asarray(obj_pos, dtype=np.float64)
    size = np.asarray(obj_size, dtype=np.float64)
    if size.ndim == 0:
        size = np.full(3, float(size))
    half = size / 2.0
    top_z = obj_pos[2] + half[2]
    ref = None if ref_R is None else np.asarray(ref_R, dtype=np.float64)

    cands: list = []

    # --- 顶抓：接近轴竖直向下，闭合线绕世界 Z 转四个偏航（对齐 box 的
    # x/y 主轴；±180° 变体不冗余 —— 手是拇指/四指不对称的，两个方向腕角不同）
    tcp_top = np.array([obj_pos[0], obj_pos[1], top_z - p.grasp_depth])
    for yaw in p.top_yaws:
        closing = (math.cos(yaw), math.sin(yaw), 0.0)
        c = _mk_candidate(f"top_yaw{round(math.degrees(yaw))}", "top",
                          obj_pos, tcp_top, (0.0, 0.0, -1.0), closing, p)
        c.score = p.top_prior
        cands.append(c)

    # --- 侧抓：接近轴水平指向 ±x/±y 面，闭合线取水平切向（捏两个邻侧面）。
    # 竖直闭合（捏顶/底面）在桌面场景下指会插桌，不生成。
    for nx, ny in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        n = np.array([nx, ny, 0.0], dtype=np.float64)          # 面外法线
        approach = -n
        face_center = obj_pos + n * (half[0] if nx else half[1])
        tcp = face_center + approach * p.side_depth
        closing = np.cross(n, _WORLD_UP)                        # 水平切向
        c = _mk_candidate(f"side_{'+' if (nx + ny) > 0 else '-'}"
                          f"{'x' if nx else 'y'}", "side",
                          obj_pos, tcp, approach, closing, p)
        c.score = 0.0
        if base_xy is not None:
            # 从基座一侧伸手顺势接近（approach 与基座→物体同向）加分
            to_obj = obj_pos[:2] - np.asarray(base_xy, dtype=np.float64)
            nn = np.linalg.norm(to_obj)
            if nn > 1e-9:
                c.score += p.approach_base_weight * float(
                    approach[:2] @ (to_obj / nn))
        cands.append(c)

    if ref is not None:
        for c in cands:
            c.score -= p.yaw_ref_weight * _geodesic_angle(
                ref, np.asarray(c.R_flange)) / math.pi

    cands.sort(key=lambda c: c.score, reverse=True)
    return cands
