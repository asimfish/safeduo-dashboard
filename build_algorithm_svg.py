#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render the SafeDuo algorithm block diagram as a static SVG (no dependencies).

    python3 build_algorithm_svg.py            # writes diagrams/algorithm_v1.svg

The diagram is versioned by hand: bump VERSION and the file name when the algorithm changes
(and update algorithm.json `diagram` on the data branch). Layout is a plain box/arrow grid so the
page renders identically everywhere; text is Chinese with the key numbers of the current recipe.
"""
from pathlib import Path

VERSION = "v1 · 2026-09-08 · 配方 aF（a27 课程 @ v8 布局 + 最终执行栈 fix2_la06）"
OUT = Path(__file__).parent / "diagrams" / "algorithm_v1.svg"

W, H = 1480, 900
FONT = "-apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei','Noto Sans CJK SC',sans-serif"
COL = {
    "op": ("#fef3c7", "#d97706"),      # operators
    "obs": ("#e0f2fe", "#0284c7"),     # observation
    "pol": ("#ede9fe", "#7c3aed"),     # policy
    "exec": ("#dcfce7", "#059669"),    # execution semantics
    "env": ("#f1f5f9", "#334155"),     # env / robots
    "loss": ("#fee2e2", "#dc2626"),    # training signals
    "eval": ("#fff7ed", "#ea580c"),    # evaluation
    "plan": ("#f5f5f4", "#78716c"),    # planned
}


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class SVG:
    def __init__(self):
        self.parts = []

    def box(self, x, y, w, h, title, lines, kind, dashed=False):
        fill, stroke = COL[kind]
        dash = ' stroke-dasharray="6,4"' if dashed else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="1.6"{dash}/>')
        self.parts.append(f'<text x="{x + 12}" y="{y + 22}" font-size="14" font-weight="700" fill="#1e293b">{esc(title)}</text>')
        for i, ln in enumerate(lines):
            self.parts.append(f'<text x="{x + 12}" y="{y + 42 + i * 17}" font-size="12" fill="#334155">{esc(ln)}</text>')
        return (x, y, w, h)

    def arrow(self, x1, y1, x2, y2, label="", color="#475569", dashed=False, lx=None, ly=None):
        dash = ' stroke-dasharray="5,4"' if dashed else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.8" marker-end="url(#arr)"{dash}/>')
        if label:
            lx = (x1 + x2) / 2 if lx is None else lx
            ly = (y1 + y2) / 2 - 6 if ly is None else ly
            self.parts.append(f'<text x="{lx}" y="{ly}" font-size="11.5" fill="{color}" text-anchor="middle">{esc(label)}</text>')

    def poly(self, pts, label="", color="#475569", dashed=False, lx=None, ly=None):
        dash = ' stroke-dasharray="5,4"' if dashed else ""
        d = " ".join(f"{x},{y}" for x, y in pts)
        self.parts.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="1.8" marker-end="url(#arr)"{dash}/>')
        if label and lx is not None:
            self.parts.append(f'<text x="{lx}" y="{ly}" font-size="11.5" fill="{color}" text-anchor="middle">{esc(label)}</text>')

    def text(self, x, y, s, size=12, color="#334155", weight=400, anchor="start"):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{esc(s)}</text>')

    def band(self, x, y, w, h, label, color):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="none" stroke="{color}" stroke-width="1.2" stroke-dasharray="3,5" opacity="0.8"/>')
        self.parts.append(f'<text x="{x + 14}" y="{y + 18}" font-size="12.5" font-weight="700" fill="{color}">{esc(label)}</text>')

    def render(self):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">'
                '<defs><marker id="arr" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#475569"/></marker></defs>'
                f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
        return head + "".join(self.parts) + "</svg>"


def main():
    s = SVG()
    s.text(20, 28, "SafeDuo 算法框图 —— 学习型安全离合器（α+p）+ 解析兜底，PPO-Lagrangian（PID 乘子）训练", 17, "#0f172a", 700)
    s.text(20, 48, VERSION, 12, "#64748b")

    # ---------------- runtime loop (top band)
    s.band(14, 62, 1452, 262, "① 运行时回路（每控制步，训练与真机同一套语义）", "#0284c7")
    s.box(30, 96, 210, 200, "双人遥操 / 虚拟操作员", [
        "F 操作员 → FR3_L, FR3_R（F2 手）",
        "U 操作员 → UR5e_L, UR5e_R（DFX 手）",
        "指令 Δq_cmd：28 关节增量 / 步",
        "训练期 = 技能库回放 + 随机漫游流",
        "  幅度课程 0.03→0.10 rad，混流 0.2",
        "  受控会合：v_rel ≤ 0.15 m/s 豁免",
        "（规划）伙伴扰动：速度缩放/噪声/时序偏移",
    ], "op")
    s.box(270, 96, 250, 200, "观测构造（275 维，状态式）", [
        "四臂 q, q̇ + 指令 Δq_cmd",
        "成对特征：[距离, 接近速度, 类别 onehot]",
        "  类别 = 跨机 / 自碰 / 桌面（最差对）",
        "arm-aware 配对观测（R15）",
        "p2-obs：上一步 α / p（R18）",
        "臂几何壳：F2 三球/指 v7fix，UR 连杆球",
        "（规划）30 步历史签名 / 力矩反馈",
    ], "obs")
    s.box(550, 96, 250, 200, "策略网络（Beta actor）", [
        "输出 α₁..α₄：逐臂危险概率 ∈ (0,1)",
        "      p：刹停/保持量（联合 Beta）",
        "有界支撑 → 不卡在动作边界",
        "危险头（hazard head）共享干路",
        "训练期 α 在 0.5 二值化，",
        "  直通估计器回传梯度（消除",
        "  训练-执行失配，R18）",
    ], "pol")
    s.box(830, 96, 250, 200, "离合器（执行语义，R17/R18/R30）", [
        "α < 0.5：该臂指令逐位直通，零衰减",
        "α ≥ 0.5：仅冻结涉险臂（其余臂不动）",
        "后撤方向指令直通（retreat_release）",
        "恢复 = 操作员逐臂解锁",
        "误刹 = 无险情却冻结（软指标）",
        "cmd-stop 时延 p50 / p95",
    ], "exec")
    s.box(1110, 96, 250, 200, "解析 backstop（硬兜底）", [
        "几何距离 + 前瞻 0.06 s 速度感知阻尼带",
        "结构行豁免（UR upper_arm×桌，R29）",
        "桌面 / 自碰 / 跨机三类最小距离",
        "臂执行器保真：UR armature 0.05，",
        "  FR3/UR 重力补偿（R33 = 栈 fix2）",
        "→ 任意指令零损坏碰撞的最后一道",
    ], "exec")
    s.arrow(240, 196, 270, 196, "Δq_cmd")
    s.arrow(520, 196, 550, 196, "o_t")
    s.arrow(800, 196, 830, 196, "α, p")
    s.arrow(1080, 196, 1110, 196, "Δq_gate")

    # env box (right, below the runtime band) and feedback
    s.box(1110, 340, 250, 150, "机器人 / 仿真", [
        "Isaac Lab 4096 并行环境（A100 30109）",
        "四臂 v8 布局：两排 1.296 m 面对面",
        "任务物体：R34 工业装配族（4080 横梁、",
        "  外框夹具 v2 锥口搁座、角件、料箱）",
        "真机：FR3×2 + UR5e×2 + Inspire 五指手",
    ], "env")
    s.arrow(1235, 296, 1235, 340, "执行 Δq_exec")
    s.poly([(1110, 415), (700, 415), (700, 330), (395, 330), (395, 296)], "状态 s_{t+1}（四臂 q, q̇, 物体位姿）", lx=800, ly=408)

    # ---------------- training band
    s.band(14, 336, 1080, 370, "② 训练信号（PPO-Lagrangian，自建循环 safeduo/algo/lagrangian_ppo.py）", "#dc2626")
    s.box(30, 370, 250, 150, "代价通道（约束，不进奖励）", [
        "cross / self / table：最差对接近代价",
        "  由标准观测重建（无 env 钩子）",
        "cost-scale 4 / 17 / 19",
        "上限 0.04 / 0.05 / 0.05",
        "有效性感知 GAE（真终止 vs 超时）",
    ], "loss")
    s.box(310, 370, 250, 150, "PID-Lagrangian 乘子 λ", [
        "每迭代按 rollout 均值更新",
        "kp 0.5 / ki 0.05 / kd 0.1，积分限幅",
        "上限 λ_max 0.35 / 0 / 0.05（防绕组）",
        "变化率 ≤ 0.01 / 更新（防震荡）",
        "超限 → 抬高惩罚；回限 → 放松",
    ], "loss")
    s.box(590, 370, 250, 150, "PPO 更新", [
        "裁剪代理目标 0.2，GAE λ 0.95",
        "四个价值头：reward + 3 代价",
        "熵系数 0.006 ↔ 0.012（熵 EMA 触发",
        "  −3.5 / 释放 −3.2，代价门 0.9）",
        "48 步/环境/迭代 × 3600 迭代 ≈ 7e8 步",
    ], "loss")
    s.box(30, 546, 250, 140, "奖励（跟随操作员）", [
        "指令保真：执行 = 遥操输入的程度",
        "最小干预：不必要的刹停被扣分",
        "逐臂 α 均衡正则 w 0.01（R15：",
        "  防 F_L 成为约束汇）",
        "PBRS 项（余量 Φ 差分，R6 待消融）",
    ], "loss")
    s.box(310, 546, 250, 140, "危险 BCE（辅助监督，R18/R28）", [
        "事后标签：未来 30 步内该臂真越线？",
        "会合豁免：v_rel ≤ 0.15 不算危险",
        "尾部掩码 + 正负样本平衡，w 0.5",
        "α 死区惩罚 w 2.0（推向 0 或 1）",
        "→ α 是校准过的“危险分类器”",
    ], "loss")
    s.box(590, 546, 250, 140, "课程与随机化", [
        "指令幅度 0.03 → 0.10 rad（delta 课程 v7）",
        "技能混流 0.2 + 全域漫游流（R25）",
        "物体质量/摩擦随机（R34 目录）",
        "（规划）伙伴扰动 = SAI Stage-2",
        "（规划）基座 ±2–5 cm，灯光，桌面材质",
    ], "loss")
    s.arrow(280, 445, 310, 445, "J_c > limit?")
    s.arrow(560, 445, 590, 445, "λ·J_c")
    s.arrow(155, 546, 155, 520, "", dashed=True)
    s.arrow(435, 546, 435, 520, "", dashed=True)
    s.arrow(715, 546, 715, 520, "", dashed=True)
    s.poly([(1110, 470), (900, 470), (900, 445), (840, 445)], "越线 / 接近 / 危险标签", lx=980, ly=463)
    s.poly([(715, 370), (715, 350), (675, 350), (675, 296)], "梯度 → 策略参数", lx=560, ly=346, dashed=True)

    # ---------------- evaluation band
    s.band(14, 718, 1452, 168, "③ 评测与验收（四门口径 R28：25 环境 × 60 s，θ = 0.5 : 0.2）", "#ea580c")
    s.box(30, 750, 330, 118, "硬门（必须全 1.0）", [
        "brake：涉险时确实刹停",
        "viol-free(cross)：跨机零越线（min cross > 0）",
        "no-deadlock(hys)：不出现双方互锁",
        "→ a30b@stack_fix 0.96 未闭合 → aF 重训",
    ], "eval")
    s.box(380, 750, 330, 118, "软指标（越好越优）", [
        "误刹率 arm / any（无险情却刹）",
        "指令保真 fidelity，平均衰减",
        "任务完成度 done（hys / manual）",
        "cmd-stop 时延 p50 / p95",
    ], "eval")
    s.box(730, 750, 350, 118, "对照与消融", [
        "raw 直通 vs 门控 s0 vs 旧运动学吸附",
        "执行栈谱系：r18 → stack → fix → fix2 → la06 → v8",
        "算法淘汰赛 R3：SAC-Lag / WCSAC / CPO（待排）",
        "跨架构红线：A100 结果不与 5090 结果配对",
    ], "eval")
    s.box(1100, 750, 350, 118, "规划中（R35，参考 SAI / TRACE）", [
        "相位同步率（2–5 检查点 + 时序容差）",
        "伙伴延迟探针（一方随机停 0.5–2 s）",
        "路径签名意图特征 / 级联门控头",
        "分臂顺序微调 + 逐臂损失掩码",
    ], "plan", dashed=True)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(s.render(), encoding="utf-8")
    print("wrote", OUT, len(s.parts), "elements")


if __name__ == "__main__":
    main()
