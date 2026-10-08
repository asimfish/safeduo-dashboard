"""A4 兜底单测：约束满足 / 预算遵守 / 无活跃约束恒等 / p 分配 / 强制退开（cpu）。

C12 追加（2026-08-16）：v6 泵病理修复（Round 154 ①/A10）与 per-link table
d_min（Round 154 ②）的实装测试，见文件末两节。
"""

import pytest
import torch

from safeduo.baselines.base import ConstraintRows
from safeduo.safety.backstop import BackstopConfig, VelocityDamperBackstop
from safeduo.safety.types import ARM_KEYS, DOF_OF, DeltaCmd, zeros_delta

DT = 1 / 60


def make_rows(n=1, m=2, device="cpu"):
    """全 padding 的空行集，测试按需填。"""
    return ConstraintRows(
        d=torch.full((n, m), 1.0),
        J={"F": torch.zeros(n, m, 14), "U": torch.zeros(n, m, 12)},
        cls=torch.zeros(n, m),
        arm_mask=torch.zeros(n, m, 4, dtype=torch.bool),
        valid=torch.zeros(n, m, dtype=torch.bool),
    )


def full_alpha(n=1):
    return torch.ones(n, 4)


def cmd_const(val=0.02, n=1):
    d = zeros_delta(n)
    for k in ARM_KEYS:
        d.delta_q[k] += val
    return d


def test_identity_when_no_active_rows():
    bs = VelocityDamperBackstop()
    cmd = cmd_const(0.02)
    out, active, info = bs.project(cmd, make_rows(), full_alpha(), torch.zeros(1), DT)
    for k in ARM_KEYS:
        assert torch.allclose(out.delta_q[k], cmd.delta_q[k], atol=1e-8)
    assert not active.any()
    assert info["residual_F"].max() < 1e-8


def test_single_cross_row_exact_projection():
    # F_L joint0 正向运动闭合：J_F = [-1, 0...] -> 约束 u0 <= h
    bs = VelocityDamperBackstop(BackstopConfig(gamma=4.0, max_passes=1))
    rows = make_rows(m=1)
    rows.valid[:] = True
    rows.cls[:] = 0.0  # cross
    rows.d[:] = 0.05
    rows.J["F"][0, 0, 0] = -1.0
    rows.arm_mask[0, 0, 0] = True  # F_L
    cmd = cmd_const(0.02)
    dmin = torch.full((1, 1), 0.03)
    p = torch.zeros(1)
    out, active, info = bs.project(cmd, rows, full_alpha(), p, DT, dmin)
    # cap = 4*(0.05-0.03)*dt = 0.0013333; F 份额 0.5 -> u0 <= 0.000667
    expect = 0.5 * 4.0 * 0.02 * DT
    assert abs(out.delta_q["F_L"][0, 0].item() - expect) < 1e-7
    # 其余关节不动
    assert torch.allclose(out.delta_q["F_L"][0, 1:], cmd.delta_q["F_L"][0, 1:])
    assert torch.allclose(out.delta_q["F_R"], cmd.delta_q["F_R"])
    assert active[0, 0] and not active[0, 1:].any()


def test_priority_budget_split():
    # 对称迎面行：p=+1 -> U 让 F（U 闭合预算 0，F 拿全份）
    bs = VelocityDamperBackstop(BackstopConfig(gamma=4.0, max_passes=2))
    rows = make_rows(m=1)
    rows.valid[:] = True
    rows.cls[:] = 0.0
    rows.d[:] = 0.05
    rows.J["F"][0, 0, 0] = -1.0
    rows.J["U"][0, 0, 0] = -1.0
    rows.arm_mask[0, 0, 0] = True   # F_L
    rows.arm_mask[0, 0, 2] = True   # U_L
    cmd = cmd_const(0.02)
    dmin = torch.full((1, 1), 0.03)
    cap = 4.0 * 0.02 * DT
    out_pos, _, _ = bs.project(cmd, rows, full_alpha(), torch.ones(1), DT, dmin)
    assert abs(out_pos.delta_q["F_L"][0, 0].item() - cap) < 1e-7   # F 全预算
    assert abs(out_pos.delta_q["U_L"][0, 0].item() - 0.0) < 1e-7   # U 禁闭合
    out_neg, _, _ = bs.project(cmd, rows, full_alpha(), -torch.ones(1), DT, dmin)
    assert abs(out_neg.delta_q["F_L"][0, 0].item() - 0.0) < 1e-7
    assert abs(out_neg.delta_q["U_L"][0, 0].item() - cap) < 1e-7


def test_forced_retreat_below_dmin():
    # d < d_min：cap 为负 -> 投影后必须反向（退开）
    bs = VelocityDamperBackstop(BackstopConfig(gamma=4.0, max_passes=1))
    rows = make_rows(m=1)
    rows.valid[:] = True
    rows.cls[:] = 1.0  # self 行：整份 cap 给涉事机器人
    rows.d[:] = 0.01
    rows.J["F"][0, 0, 0] = -1.0
    rows.arm_mask[0, 0, 0] = True
    cmd = cmd_const(0.0)  # 原地不动也得退
    dmin = torch.full((1, 1), 0.03)
    out, active, _ = bs.project(cmd, rows, full_alpha(), torch.zeros(1), DT, dmin)
    cap = 4.0 * (0.01 - 0.03) * DT  # 负
    assert out.delta_q["F_L"][0, 0].item() <= cap + 1e-7  # u0 <= cap < 0
    assert active[0, 0]


def test_alpha_progress_budget():
    # 无安全行，alpha=0.5：执行沿命令方向的进展 = 0.5*||c||，且不算 backstop_active
    bs = VelocityDamperBackstop()
    cmd = cmd_const(0.02)
    alpha = torch.full((1, 4), 0.5)
    out, active, _ = bs.project(cmd, make_rows(), alpha, torch.zeros(1), DT)
    for k in ARM_KEYS:
        c = cmd.delta_q[k][0]
        u = out.delta_q[k][0]
        prog = (u @ c) / c.norm()
        assert abs(prog.item() - 0.5 * c.norm().item()) < 1e-6
    assert not active.any()  # alpha 修剪不算安全兜底


def test_multi_constraint_residual_small():
    # 随机稠密多行（最坏情形）：默认配置残差 < tol 量级；上限拉满后应逼近 0
    torch.manual_seed(0)
    n, m = 64, 8
    bs = VelocityDamperBackstop(BackstopConfig())
    rows = make_rows(n, m)
    rows.valid[:] = torch.rand(n, m) < 0.7
    rows.cls[:] = (torch.rand(n, m) < 0.5).float()  # 混 cross/self
    rows.d[:] = 0.02 + 0.06 * torch.rand(n, m)
    rows.J["F"] = torch.randn(n, m, 14) * 0.5
    rows.J["U"] = torch.randn(n, m, 12) * 0.5
    rows.arm_mask[:] = torch.rand(n, m, 4) < 0.5
    cmd = cmd_const(0.03, n)
    dmin = torch.full((n, m), 0.02)
    out, _, info = bs.project(cmd, rows, full_alpha(n), torch.zeros(n), DT, dmin)
    assert info["residual_F"].max().item() < 1e-3
    assert info["residual_U"].max().item() < 1e-3
    bs20 = VelocityDamperBackstop(BackstopConfig(max_passes=60, tol=1e-10))
    _, _, info20 = bs20.project(cmd, rows, full_alpha(n), torch.zeros(n), DT, dmin)
    assert info20["residual_F"].max().item() < 1e-5
    assert info20["residual_U"].max().item() < 1e-5


def test_most_critical_row_exact():
    # 两行冲突：最紧的行末次投影后必须严格满足
    bs = VelocityDamperBackstop(BackstopConfig(max_passes=1))
    rows = make_rows(m=2)
    rows.valid[:] = True
    rows.cls[:] = 1.0
    rows.d[0, 0], rows.d[0, 1] = 0.10, 0.03   # 行 1 更紧
    rows.J["F"][0, 0, 0] = -1.0
    rows.J["F"][0, 1, 0] = -1.0
    rows.J["F"][0, 1, 1] = -0.5
    rows.arm_mask[0, :, 0] = True
    cmd = cmd_const(0.05)
    dmin = torch.full((1, 2), 0.02)
    out, _, _ = bs.project(cmd, rows, full_alpha(), torch.zeros(1), DT, dmin)
    u = torch.cat([out.delta_q["F_L"], out.delta_q["F_R"]], dim=-1)[0]
    g, h = -rows.J["F"][0, 1], 4.0 * (0.03 - 0.02) * DT
    assert (g @ u).item() <= h + 1e-7


# ---- A6-W6：backlog_aware 储能记账（damper 审计修复，flag 门控）----

def _zero_backlog(n=1):
    return {k: torch.zeros(n, DOF_OF[k]) for k in ARM_KEYS}


def test_backlog_aware_off_is_bitwise_identical():
    # flag off（含显式传 backlog）必须与 v3 路径逐位一致——主线回归保险丝
    rows = make_rows(m=1)
    rows.valid[:] = True
    rows.d[:] = 0.05
    rows.J["F"][0, 0, 0] = -1.0
    rows.arm_mask[0, 0, 0] = True
    cmd = cmd_const(0.02)
    dmin = torch.full((1, 1), 0.03)
    bl = _zero_backlog()
    bl["F_L"][0, 0] = 0.5
    old = VelocityDamperBackstop(BackstopConfig(gamma=4.0))
    o1, _, _ = old.project(cmd_const(0.02), rows, full_alpha(), torch.zeros(1), DT, dmin)
    o2, _, _ = old.project(cmd, rows, full_alpha(), torch.zeros(1), DT, dmin, backlog=bl)
    for k in ARM_KEYS:
        assert torch.equal(o1.delta_q[k], o2.delta_q[k])


def test_backlog_aware_shrinks_budget_when_backlog_closing():
    # 闭合方向积压（J_F u0<0 闭合，积压 +0.5 rad -> G·b_eff = +box 全额吃预算）
    bs = VelocityDamperBackstop(BackstopConfig(gamma=4.0, backlog_aware=True))
    rows = make_rows(m=1)
    rows.valid[:] = True
    rows.cls[:] = 1.0  # self：整份 cap 给涉事机器人，便于闭式核对
    rows.d[:] = 0.05
    rows.J["F"][0, 0, 0] = -1.0
    rows.arm_mask[0, 0, 0] = True
    cmd = cmd_const(0.02)
    dmin = torch.full((1, 1), 0.03)
    bl = _zero_backlog()
    bl["F_L"][0, 0] = 0.5          # 米级储能，b_eff 截到 box=vmax*dt=0.025
    out, active, _ = bs.project(cmd, rows, full_alpha(), torch.zeros(1), DT, dmin,
                                backlog=bl)
    # h = cap - G·b_eff = 4*0.02*DT - 0.025 = -0.023667 < 0 -> 强制退让
    # 且被可行性下限 -0.9*box=-0.0225 clamp 住：u0 = -0.0225
    assert abs(out.delta_q["F_L"][0, 0].item() - (-0.9 * 0.025)) < 1e-6
    assert active[0, 0]
    # 反向积压（远离）不吃预算：u 不低于原路径解
    bl2 = _zero_backlog()
    bl2["F_L"][0, 0] = -0.5
    out2, _, _ = bs.project(cmd_const(0.02), rows, full_alpha(), torch.zeros(1),
                            DT, dmin, backlog=bl2)
    assert out2.delta_q["F_L"][0, 0].item() >= 4.0 * 0.02 * DT - 1e-7


def test_backlog_aware_budget_never_below_box_feasibility():
    # h 下限 -0.9*||G||_1*box：极端积压下半空间仍与速度箱相容（Dykstra 不发散）
    bs = VelocityDamperBackstop(BackstopConfig(gamma=4.0, backlog_aware=True))
    rows = make_rows(m=1)
    rows.valid[:] = True
    rows.cls[:] = 1.0
    rows.d[:] = 0.021               # 贴边 cap 微小
    rows.J["F"][0, 0, :7] = -1.0    # 7 关节全参与，||G||_1 = 7
    rows.arm_mask[0, 0, 0] = True
    dmin = torch.full((1, 1), 0.02)
    bl = _zero_backlog()
    bl["F_L"][0, :] = 10.0          # 天文积压
    out, _, info = bs.project(cmd_const(0.02), rows, full_alpha(), torch.zeros(1),
                              DT, dmin, backlog=bl)
    box = 1.5 * DT
    for k in ARM_KEYS:
        assert (out.delta_q[k].abs() <= box + 1e-8).all()
    # 残差有限（约束集与箱相容，投影收敛）
    assert torch.isfinite(info["residual_F"]).all()


# ---- C12 v6（Round 154 ①/A10 判决 B_pump 40/41）：泵病理行修复 ----
#
# 病理构造 = A10 具名肇事行的最小复刻：JAKA link2 肩球×TABLE，结构性悬停
# 17.6mm < table d_min 20mm（cap<0 物理不可满足）且 |J|≈4e-5（肩关节抬不动
# 肩部球）。v5 路径下 hs_project 的 viol/max(||g||²,eps) 除近零 -> 修正量
# ~4 rad，出口越速度箱 100+×——"兜底数值病理主动甩出"。

def _pump_rows(n=1):
    rows = make_rows(n, m=1)
    rows.valid[:] = True
    rows.cls[:] = 2.0                     # table 行：整份 cap 给涉事机器人
    rows.d[:] = 0.0176                    # 结构性悬停 17.6mm（A10 §1 机理链）
    rows.J["U"][0, 0, 0] = -4e-5          # 低行权柄 |J|≈4e-5 m/rad
    rows.arm_mask[0, 0, 2] = True         # U_L
    return rows


_PUMP_DMIN = 0.020                        # v5 table d_min（泵病理触发条件）


def test_pump_row_v5_path_reproduces_explosion():
    # 病理对照（双 flag 显式关 = v5 逐位路径）：证明新测试咬得住病灶——
    # 闭式核对 u0 = -cap/g = -1.6e-4/4e-5 = -4.0 rad（速度箱 160×）
    bs = VelocityDamperBackstop(BackstopConfig(
        gamma=4.0, max_passes=1,
        row_authority_clamp=False, exit_box_invariant=False))
    dmin = torch.full((1, 1), _PUMP_DMIN)
    out, _, _ = bs.project(cmd_const(0.0), _pump_rows(), full_alpha(),
                           torch.zeros(1), DT, dmin)
    box = 1.5 * DT
    assert out.delta_q["U_L"][0, 0].abs().item() > 10 * box
    assert abs(out.delta_q["U_L"][0, 0].item() - (-4.0)) < 1e-4


def test_pump_row_output_bounded_and_inside_box():
    # v6 主路径（默认 flag 全开）：输出有界 + 出口箱不变量成立。
    # F1 语义闭式核对：h 钳到 -0.9*||G||₁*box -> u0 = -0.9*box（单步一箱权柄，
    # A10 F1 修复变体同款 0.0225 量级，vs 病理路径的 4.0 rad）
    bs = VelocityDamperBackstop(BackstopConfig(gamma=4.0))
    dmin = torch.full((1, 1), _PUMP_DMIN)
    out, active, info = bs.project(cmd_const(0.0), _pump_rows(), full_alpha(),
                                   torch.zeros(1), DT, dmin)
    box = 1.5 * DT
    for k in ARM_KEYS:
        assert torch.isfinite(out.delta_q[k]).all()
        assert (out.delta_q[k].abs() <= box + 1e-8).all()  # 出口箱不变量
    assert abs(out.delta_q["U_L"][0, 0].item() - (-0.9 * box)) < 1e-6
    assert active[0, 2]                    # U_L 记为兜底活跃（真实退让仍在）
    assert torch.isfinite(info["residual_U"]).all()


def test_pump_row_exit_box_invariant_alone_still_bounds():
    # 第二道保险单测（F1 关、出口箱开）：任何迭代路径出口处 exec 必在速度箱
    # 内——Round 154 ①注明"不可单用"，但必须独立成立
    bs = VelocityDamperBackstop(BackstopConfig(
        gamma=4.0, max_passes=1, row_authority_clamp=False))
    dmin = torch.full((1, 1), _PUMP_DMIN)
    out, _, _ = bs.project(cmd_const(0.0), _pump_rows(), full_alpha(),
                           torch.zeros(1), DT, dmin)
    box = 1.5 * DT
    for k in ARM_KEYS:
        assert (out.delta_q[k].abs() <= box + 1e-8).all()


def test_pump_row_healthy_cross_row_unaffected():
    # A10 最小合成场景同构（1 低J桌行 + 1 健康 cross 行）：修复后健康行照常
    # 受控（约束满足、F2/F3 式误伤不发生），病理行退让被钳在一箱权柄内
    bs = VelocityDamperBackstop(BackstopConfig(gamma=4.0))
    rows = make_rows(m=2)
    rows.valid[:] = True
    rows.cls[0, 0] = 0.0                  # 健康 cross 行：U_L joint1 正向闭合
    rows.d[0, 0] = 0.05
    rows.J["U"][0, 0, 1] = -1.0
    rows.arm_mask[0, 0, 2] = True
    rows.cls[0, 1] = 2.0                  # 病理桌行（同 _pump_rows）
    rows.d[0, 1] = 0.0176
    rows.J["U"][0, 1, 0] = -4e-5
    rows.arm_mask[0, 1, 2] = True
    cmd = cmd_const(0.02)
    dmin = torch.tensor([[0.038, _PUMP_DMIN]])   # cross 行用 v6 d_min 0.038
    out, _, info = bs.project(cmd, rows, full_alpha(), torch.zeros(1), DT, dmin)
    box = 1.5 * DT
    u = out.delta_q["U_L"][0]
    cap0 = 4.0 * (0.05 - 0.038) * DT
    assert u[1].item() <= 0.5 * cap0 + 1e-6      # 健康行满足（p=0 U 侧半份额）
    assert (u.abs() <= box + 1e-8).all()         # 病理行不再把执行甩出箱
    assert torch.isfinite(info["residual_U"]).all()


# ---- C12 v6（Round 154 ②）：per-link table d_min 解析与烘焙 ----

def _sem_with_v6_overrides():
    """duo_env.__init__ 的 v6 注入同款（yaml -> sem.d_min / d_min_per_link）。"""
    from safeduo.configs import load_config, repo_root
    from safeduo.safety.semantics import ContactSemantics

    sem = ContactSemantics(
        str(repo_root() / "src/safeduo/configs/contact_semantics_v3.yaml"))
    y6 = load_config("duo_env_v6.yaml")["safety"]
    for k, v in y6["d_min_override"].items():
        sem.d_min[k] = float(v)
    for k, v in y6["d_min_per_link"].items():
        sem.d_min_per_link[str(k)] = float(v)
    if y6.get("d_warn_override") is not None:
        sem.d_warn = float(y6["d_warn_override"])
    return sem


def test_per_link_table_dmin_judge():
    sem = _sem_with_v6_overrides()
    # JAKA 肩球 link2 对两张桌都用独立 d_min（A10 肇事泵行 = U_L/U_R link2×TABLE）
    for arm in ("U_L", "U_R"):
        for tab in ("table_F", "table_U"):
            v = sem.judge(f"{arm}/link2", tab)
            assert v.keep and v.category == "table" and v.verdict == "forbid"
            assert v.d_min == pytest.approx(0.015)
    # 其余 link 对桌面维持类档 0.020；cross 行不受 per-link 影响（用 v6 0.038）
    assert sem.judge("U_L/link3", "table_U").d_min == pytest.approx(0.020)
    assert sem.judge("F_L/fr3_link4", "table_F").d_min == pytest.approx(0.020)
    assert sem.judge("U_L/link3", "F_L/fr3_link4").d_min == pytest.approx(0.038)
    # near_table 条件豁免（手-桌）语义不动：static d_min 0.020、runtime 覆写 0.005
    hv = sem.judge("F_L/hand/left_thumb_1", "table_F")
    assert hv.conditional and hv.d_min == pytest.approx(0.020)
    assert sem.near_table.d_min_override == pytest.approx(0.005)
    # d_warn 联动（Round 152 ②）：塑形带 0.08，d_soft（活跃集/tube）不动
    assert sem.d_warn == pytest.approx(0.08)
    assert sem.d_soft == pytest.approx(0.05)


def test_per_link_dmin_bakes_into_pair_table():
    # 端到端烘焙：真 v5 球分解 + v6 注入 -> pair 表里 link2×table 行 0.015、
    # 其余桌行 0.020、cross 全行 0.038（duo_env 建对同路径）
    from safeduo.configs import repo_root
    from safeduo.safety.sphere_distance import SphereDistanceModule
    from safeduo.safety.sphere_specs import real_v5_arm_specs

    sph = SphereDistanceModule(real_v5_arm_specs(repo_root()),
                               semantics=_sem_with_v6_overrides(),
                               max_active=32, d_soft=0.05, tau_ttc=0.5,
                               quota_cross=8, device="cpu")
    pc = sph._slice_cross.stop
    assert torch.allclose(sph.pair_dmin[:pc], torch.full((pc,), 0.038)), \
        "cross 全行烘焙 v6 d_min 0.038"
    qn = sph.qualified_names
    sl = sph._slice_table
    n_link2 = 0
    for k in range(sl.start, sl.stop):
        i = int(sph.pair_table[k][0])
        expect = 0.015 if qn[i].split("/", 1)[1] == "link2" else 0.020
        if expect == 0.015:
            n_link2 += 1
        assert float(sph.pair_dmin[k]) == pytest.approx(expect), qn[i]
    # jaka_zu7_v5 link2 有球、双臂×双桌都必须收到覆写
    assert n_link2 > 0 and n_link2 % 4 == 0


def test_v6_yaml_only_safety_differs_from_v5():
    # v6 配方红线：除 safety 节外与 v5 逐位一致（对照变量只有安全参数——
    # 文件头 diff 清单的机器可验证半边）
    from safeduo.configs import load_config

    y5 = load_config("duo_env_v5.yaml")
    y6 = load_config("duo_env_v6.yaml")
    assert {k: v for k, v in y6.items() if k != "safety"} == \
           {k: v for k, v in y5.items() if k != "safety"}
    s6 = y6["safety"]
    assert s6["d_min_override"] == {"self": 0.013, "cross": 0.038}  # Round 152 ②
    assert s6["d_warn_override"] == pytest.approx(0.08)             # Round 152 ②
    assert s6["d_min_per_link"] == {"link2": 0.015}   # Round 154 ②（数值暂定）
    assert s6["quota_cross"] == 8                     # Round 154：8→16 撤回
    assert s6["backstop"]["gamma"] == pytest.approx(4.0)            # 维持
    assert s6["backstop"]["row_authority_clamp"] is True            # Round 154 ①
    assert s6["backstop"]["exit_box_invariant"] is True             # Round 154 ①
    # v5 侧其余 safety 键不许漂移
    keep = {k: v for k, v in s6.items()
            if k not in ("d_min_override", "d_warn_override",
                         "d_min_per_link", "backstop")}
    keep5 = {k: v for k, v in y5["safety"].items()
             if k not in ("d_min_override", "backstop")}
    assert keep == keep5
