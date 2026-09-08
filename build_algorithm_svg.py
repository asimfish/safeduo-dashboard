#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render the SafeDuo algorithm block diagram as a static SVG (no dependencies).

    python3 build_algorithm_svg.py            # writes diagrams/algorithm_v2.svg

Paper-figure style: three tinted lanes (runtime loop / training signals / evaluation), white cards
with a coloured header strip and <= 3 short body lines, orthogonal arrows with pill labels, a legend.
Bump VERSION + file name when the algorithm changes; algorithm.json `diagram` points at the file.
"""
from pathlib import Path

VERSION = "v2 · 2026-09-08 · 配方 aF（a27 课程 @ v8 布局 + 最终执行栈 fix2_la06）+ SafeTransport 增量 E0–E4"
OUT = Path(__file__).parent / "diagrams" / "algorithm_v2.svg"
W, H = 1600, 1000
FONT = "-apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei','Noto Sans CJK SC',sans-serif"


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class SVG:
    def __init__(self):
        self.p = []

    def lane(self, y, h, label, tint, color):
        self.p.append(f'<rect x="20" y="{y}" width="{W-40}" height="{h}" rx="16" fill="{tint}"/>')
        self.p.append(f'<text x="40" y="{y+26}" font-size="13" font-weight="700" fill="{color}" letter-spacing="1">{esc(label)}</text>')

    def card(self, x, y, w, h, title, lines, color, dashed=False, badge=None):
        dash = ' stroke-dasharray="7,5"' if dashed else ""
        self.p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#ffffff" stroke="{color}" stroke-width="1.4"{dash}/>')
        self.p.append(f'<path d="M{x},{y+10} a10,10 0 0 1 10,-10 h{w-20} a10,10 0 0 1 10,10 v18 h-{w} z" fill="{color}"/>')
        self.p.append(f'<text x="{x+12}" y="{y+19}" font-size="13" font-weight="700" fill="#ffffff">{esc(title)}</text>')
        if badge:
            bw = 11 * len(badge) + 14
            self.p.append(f'<rect x="{x+w-bw-8}" y="{y+6}" width="{bw}" height="16" rx="8" fill="#ffffff" opacity="0.92"/>')
            self.p.append(f'<text x="{x+w-bw/2-8}" y="{y+18}" font-size="10.5" font-weight="700" fill="{color}" text-anchor="middle">{esc(badge)}</text>')
        for i, ln in enumerate(lines):
            self.p.append(f'<text x="{x+12}" y="{y+48+i*18}" font-size="12.2" fill="#334155">{esc(ln)}</text>')

    def sub(self, x, y, w, h, title, lines, color, dashed=False):
        dash = ' stroke-dasharray="5,4"' if dashed else ""
        self.p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{color}" fill-opacity="0.08" stroke="{color}" stroke-width="1.1"{dash}/>')
        self.p.append(f'<text x="{x+10}" y="{y+17}" font-size="12.2" font-weight="700" fill="{color}">{esc(title)}</text>')
        for i, ln in enumerate(lines):
            self.p.append(f'<text x="{x+10}" y="{y+35+i*16}" font-size="11.5" fill="#334155">{esc(ln)}</text>')

    def pill(self, x, y, text, color="#475569"):
        w = 6.6 * sum(2 if ord(c) > 255 else 1 for c in text) + 16
        self.p.append(f'<rect x="{x-w/2}" y="{y-10}" width="{w}" height="19" rx="9.5" fill="#ffffff" stroke="#cbd5e1"/>')
        self.p.append(f'<text x="{x}" y="{y+4}" font-size="11" fill="{color}" text-anchor="middle">{esc(text)}</text>')

    def arrow(self, pts, color="#64748b", dashed=False, width=1.7):
        dash = ' stroke-dasharray="6,5"' if dashed else ""
        d = " ".join(f"{x},{y}" for x, y in pts)
        m = "arrd" if dashed else "arr"
        self.p.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="{width}" marker-end="url(#{m})"{dash}/>')

    def line(self, pts, color="#94a3b8", width=2.2):
        d = " ".join(f"{x},{y}" for x, y in pts)
        self.p.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round"/>')

    def text(self, x, y, s, size=12, color="#334155", weight=400, anchor="start"):
        self.p.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{esc(s)}</text>')

    def render(self):
        defs = ('<defs>'
                '<marker id="arr" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto"><path d="M0,0 L9,3.5 L0,7 z" fill="#64748b"/></marker>'
                '<marker id="arrd" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto"><path d="M0,0 L9,3.5 L0,7 z" fill="#7c3aed"/></marker>'
                '</defs>')
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">'
                f'{defs}<rect width="{W}" height="{H}" fill="#ffffff"/>' + "".join(self.p) + '</svg>')


def main():
    s = SVG()
    s.text(28, 34, "SafeDuo 算法框图", 21, "#0f172a", 800)
    s.text(215, 34, "学习型安全离合器（逐臂 α + p）+ 解析兜底 · PPO-Lagrangian 约束强化学习 · SafeTransport 容量流增量", 13.5, "#475569")
    s.text(28, 54, VERSION, 11.5, "#94a3b8")

    # ---------------- lane 1: runtime loop
    C_OP, C_OBS, C_POL, C_CL, C_BS, C_ROB = "#d97706", "#0284c7", "#7c3aed", "#059669", "#047857", "#475569"
    s.lane(66, 275, "① 运行时回路 —— 每个控制步，训练与真机同一套语义", "#f0f7ff", "#0369a1")
    y1, h1, w1 = 100, 138, 220
    xs = [40, 300, 560, 820, 1080, 1340]
    s.card(xs[0], y1, w1, h1, "双人遥操 / 虚拟操作员", ["F 操作员 → FR3_L / FR3_R", "U 操作员 → UR5e_L / UR5e_R", "训练期：技能回放 + 漫游流（参数见 ②）", "输出 Δq_cmd（28 关节增量）"], C_OP)
    s.card(xs[1], y1, w1, h1, "观测构造 · 275 维", ["四臂 q, q̇ + 指令 Δq_cmd", "成对特征：距离 / 接近速度 / 类别", "arm-aware 配对 · 上一步 α, p", "（规划）历史签名 · 力矩反馈"], C_OBS)
    s.card(xs[2], y1, w1, h1, "离合器策略 π_θ（Beta）", ["α₁..α₄：逐臂危险概率", "p：刹停 / 保持量", "有界支撑，不卡动作边界", "训练期 α 在 0.5 直通二值化"], C_POL)
    s.card(xs[3], y1, w1, h1, "离合器执行语义", ["α < 0.5：指令逐位直通，零衰减", "α ≥ 0.5：只冻结涉险臂", "后撤方向直通 · 逐臂解锁恢复", "误刹 / cmd-stop 时延 = 软指标"], C_CL)
    s.card(xs[4], y1, w1, h1, "解析 backstop · 硬兜底", ["几何距离 + 0.06 s 前瞻阻尼带", "结构行豁免 · 三类最小距离", "活腕 armature · 重力补偿", "= 执行栈 fix2_la06"], C_BS)
    s.card(xs[5], y1, w1, h1, "机器人 / 仿真", ["Isaac Lab 4096 环境（A100）", "v8 布局：两排 1.296 m 面对面", "R34 工业装配物体（横梁/夹具）", "真机：FR3×2 + UR5e×2 + 五指手"], C_ROB)
    for a, b, lab in zip(xs[:-1], xs[1:], ["Δq_cmd", "o_t", "α, p", "Δq_gate", "Δq_exec"]):
        s.arrow([(a + w1, 169), (b, 169)])
        s.pill((a + w1 + b) / 2, 158, lab)
    # state feedback (inside lane 1)
    s.arrow([(1450, 238), (1450, 292), (410, 292), (410, 238)])
    s.pill(930, 292, "状态 s_{t+1}：四臂 q, q̇ · 物体位姿 · 接触")
    # rollout bus into lane 2
    s.line([(1500, 238), (1500, 356), (60, 356)], "#94a3b8", 2.4)
    s.pill(1180, 356, "rollout 缓冲（T × N）：状态 · 动作 · 奖励 · 三条代价 · 危险标签", "#334155")

    # ---------------- lane 2: training signals
    C_COST, C_MECH, C_PPO, C_REW, C_AUX, C_CUR, C_DIAG, C_ST = "#dc2626", "#b91c1c", "#9f1239", "#ea580c", "#e11d48", "#a16207", "#6d28d9", "#0f766e"
    s.lane(346, 360, "② 训练信号 —— PPO-Lagrangian（自建循环 safeduo/algo/lagrangian_ppo.py）与 SafeTransport 增量", "#fff5f5", "#b91c1c")
    ya, ha = 386, 122
    s.card(40, ya, 300, ha, "代价通道 · 约束不进奖励", ["cross / self / table 最差对接近代价", "cost-scale 4 / 17 / 19 · 上限 0.04 / 0.05 / 0.05", "由标准观测重建 · 有效性感知 GAE"], C_COST)
    s.card(380, ya, 300, ha, "奖励 · 跟随操作员", ["指令保真：执行 = 遥操输入的程度", "最小干预：无谓刹停扣分", "逐臂 α 均衡正则 w 0.01 · PBRS 余量项"], C_REW)
    s.card(720, ya, 300, ha, "辅助监督 · 危险 BCE（R18/R28）", ["事后标签：未来 30 步内该臂真越线", "会合豁免 v_rel ≤ 0.15 · 尾部掩码 · 类平衡", "w 0.5 · α 死区惩罚 w 2.0（推向 0/1）"], C_AUX)
    s.card(1060, ya, 500, ha, "课程与随机化", ["指令幅度 0.03 → 0.10 rad · 技能混流 0.2 · 全域漫游", "物体质量 / 摩擦随机（R34 目录）", "（规划，SAI Stage-2）伙伴扰动：速度缩放 / 噪声 / 时序偏移"], C_CUR)
    for cx in (190, 530, 870, 1310):
        s.arrow([(cx, 356), (cx, ya)])
    yb, hb = 540, 150
    # constraint mechanism with two alternatives
    s.card(40, yb, 300, hb, "约束机制 · 二选一", [], C_MECH)
    s.sub(52, yb + 36, 132, 104, "PID 拉格朗日 λ", ["当前配方（aF）", "kp .5 / ki .05 / kd .1", "λ_max .35/0/.05", "变化率 ≤ .01"], C_MECH)
    s.sub(196, yb + 36, 132, 104, "SafeTransport", ["增量 E1–E4", "3 预算 → 1 容量矩阵", "γ-流 Sinkhorn F*≤S", "w_t = F*/F̂ 重加权"], C_ST)
    s.card(380, yb, 300, hb, "PPO 更新", ["裁剪 0.2 · GAE 0.95 · 四价值头", "熵系数 0.006 ↔ 0.012（熵 EMA 触发）", "48 步/环境/迭代 × 3600 迭代 ≈ 7×10⁸ 步", "目标：Σ w_t · min(r_t Â, clip(r_t) Â) − λ·Â_c"], C_PPO)
    s.card(720, yb, 300, hb, "诊断与证书（在线）", ["IS-clip 率 · 权重均值 · 容量激活比例", "转移间隙 ‖F* − F̂_θ‖₁ → 每通道认证上界", "α̂_k 动作松弛（> 3 = 格子太粗）", "λ / 代价 / 危险率轨迹 → stats.jsonl"], C_DIAG)
    s.card(1060, yb, 500, hb, "增量实验 E0–E4（30109 GPU2，a27 热启动 400 迭代，v8 + fix2_la06）", ["E0 PID 基线 ｜ E1 SafeTransport 论文默认（三通道，硬最小，c̄=max）", "E2 校准版（cross+self，c̄=mean，权重归一） ｜ E3 证书触发 Safe", "E4 Hybrid β 0.5 ｜ 同预算同环境，比：代价轨迹 · 误刹 · 四门"], C_ST, badge="进行中")
    s.arrow([(190, ya + ha), (190, yb)]); s.pill(190, (ya + ha + yb) / 2, "J_c vs 上限")
    s.arrow([(340, 615), (380, 615)])
    s.arrow([(530, ya + ha), (530, yb)]); s.pill(530, (ya + ha + yb) / 2, "Â_r")
    s.arrow([(870, ya + ha), (870, 528), (640, 528), (640, yb)]); s.pill(760, 528, "BCE / 死区项")
    s.arrow([(680, 655), (720, 655)])
    # gradient back to the policy
    s.arrow([(395, yb), (395, 330), (670, 330), (670, 238)], "#7c3aed", dashed=True); s.pill(540, 330, "梯度 → θ", "#7c3aed")

    # ---------------- lane 3: evaluation
    C_EV = "#ea580c"
    s.lane(720, 250, "③ 评测与验收 —— 四门口径 R28：25 环境 × 60 s，θ = 0.5 : 0.2；A100 与 5090 结果不配对", "#fffbeb", "#c2410c")
    yc, hc = 760, 130
    s.card(40, yc, 350, hc, "硬门 · 必须全部 1.0", ["brake：涉险时确实刹停", "viol-free(cross)：跨机零越线（min cross > 0）", "no-deadlock(hys)：不出现双方互锁", "a30b@stack_fix 0.96 未闭合 → aF 重训"], C_EV)
    s.card(420, yc, 350, hc, "软指标 · 越好越优", ["误刹率 arm / any（无险情却刹）", "指令保真 fidelity · 平均衰减", "任务完成度 done（hys / manual）", "cmd-stop 时延 p50 / p95"], C_EV)
    s.card(800, yc, 350, hc, "对照与消融", ["raw 直通 vs 门控 s0 vs 旧运动学吸附", "执行栈谱系 r18 → stack → fix → fix2 → la06 → v8", "约束机制：PID vs SafeTransport（E0–E4）", "算法淘汰赛 R3：SAC-Lag / WCSAC / CPO"], C_EV)
    s.card(1180, yc, 380, hc, "规划中 · R35（SAI / TRACE 启发）", ["相位同步率（2–5 检查点 + 时序容差）", "伙伴延迟探针（一方随机停 0.5–2 s）", "路径签名意图特征 · 级联门控头", "分排顺序微调 + 逐臂损失掩码"], "#78716c", dashed=True, badge="规划")
    s.arrow([(1450, 238), (1560, 238), (1560, 745), (1000, 745), (1000, yc)], "#64748b", dashed=True)
    s.pill(1280, 745, "训练后策略 → 四门评测（clutch_eval）")

    # legend
    lx, ly = 40, 940
    for i, (c, t) in enumerate([(C_OP, "操作员/输入"), (C_OBS, "观测"), (C_POL, "策略"), (C_CL, "执行语义 / 兜底"), (C_COST, "训练信号"), (C_ST, "SafeTransport 增量"), (C_EV, "评测"), ("#78716c", "规划中（虚线）")]):
        x = lx + i * 175
        s.p.append(f'<rect x="{x}" y="{ly-10}" width="14" height="14" rx="3" fill="{c}"/>')
        s.text(x + 20, ly + 2, t, 11.5, "#475569")
    s.text(W - 30, ly + 24, "asimfish.github.io/safeduo-dashboard · docs/ALGORITHM_DESIGN.md · 生成器 build_algorithm_svg.py", 11, "#94a3b8", anchor="end")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(s.render(), encoding="utf-8")
    print("wrote", OUT, len(s.p), "elements")


if __name__ == "__main__":
    main()
