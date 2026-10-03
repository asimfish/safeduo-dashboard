"""L2 解析兜底训练版：批量闭式 velocity-damper 顺序投影（Dykstra 修正），A4。

约束行结构 = C 的 baselines/base.py ConstraintRows（G0 对拍就在这个接口上）：
  每条活跃球对一行 velocity-damper：J_F qd_F + J_U qd_U = d_dot >= -gamma*(d - d_min)
  离散化（对每步 delta u）：(-J_r) u_r <= h_r。
  h 分配（与 C 的 strong_cbf_qp 零外推分支逐项一致）：
    cap = gamma*(d - d_min)*dt
    跨机行 cap>=0：F 侧 h=(1+p)/2*cap，U 侧 h=(1-p)/2*cap（p=+1 -> U 让 F）
    跨机行 cap<0（已越 d_min）：两侧 h=cap（双边全额退开，不分摊）
    自碰/对桌行：整份 cap 给涉事机器人。

alpha 语义（ROUND2_DELTAS v2）：每臂进展预算约束
  <u_a, c_a/||c_a||> <= alpha_a * ||c_a||   （alpha>=1 视为不设预算）
与安全行同为投影集合成员，不是对 cmd 的预缩放。

解法：Dykstra 交替投影——逐约束闭式半空间投影 + 修正项，收敛到
min ||u - c||^2 s.t. 约束集 的精确投影点（= 部署版 slack-QP 在可行时的解），
这保证训练版与 C 的 FISTA/QP 部署版在 hard-conflict 状态上尾部一致
（safety/consistency_parity.py，指标 = P99 改写差异）。
朴素 POCS 只保可行不保最小偏离，实测 P99 差 55%，故必须带 Dykstra 修正。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from safeduo.baselines.base import ConstraintRows, stack_robot, unstack_robot
from safeduo.safety.types import ARM_KEYS, ARMS_OF_ROBOT, CLASS_CROSS, CLASS_SELF, CLASS_TABLE, DOF_OF, DeltaCmd

_ARM_IDX = {a: i for i, a in enumerate(ARM_KEYS)}


@dataclass
class BackstopConfig:
    gamma: float = 4.0        # damper 增益（1/s）：允许闭合速度 = gamma*(d-d_min)
    d_min: float = 0.03       # 缺省刹停边界；逐行覆盖优先级：dmin 参数 > rows.d_min > 此值
    vmax: float = 1.5         # 每关节速度箱约束（rad/s）
    max_passes: int = 30      # Dykstra 轮数上限（几何收敛）
    tol: float = 1e-6         # 轮间位移早停阈值（rad/步）
    eps: float = 1e-9
    # A6-W6 damper energy-store fix (audit artifacts/analysis/damper_audit):
    # the per-step increment u carries only ~8% of closing authority near
    # contact; the PD backlog (q_target - q) carries ~12x more. When enabled,
    # each row budget h is reduced by the closing authority of the backlog
    # releasable THIS step, clip(backlog, +-vmax*dt) -- the clip is what the
    # support line's full-amount external accounting missed (h<0 storms).
    # Default False: v3-era mainline behavior is bit-identical.
    backlog_aware: bool = False
    # ── v6 泵病理修正包（Round 154 ①定稿，A10 取证 32/32=B_pump）──────────
    # 病灶：物理不可满足行（结构性 cap<0 且 |J| 近零，典型 = JAKA link2 肩球
    # 悬停 17.6mm < table d_min 20mm、|J|≈4e-5）在 hs_project 里
    # viol/max(||g||²,eps) 除近零 -> 修正量 ~5 rad，出口甩到速度箱 100×，
    # 设备被自己的兜底甩飞扫穿跨机带（P(爆发|低J cap<0 在场)=97.7%）。
    # row_authority_clamp（F1 行权柄钳制）：h >= -0.9*||G_row||_1*box，把每行
    #   强制退让需求钳到"该行本步在速度箱内物理可交付量"（u=-box*sign(G)
    #   仍容许该半空间）。原只在 backlog_aware 分支，Round 154 裁定迁主路径
    #   必做治本；对可满足行是恒等式（A10 合成对拍：F1 误伤最小，
    #   最热 20 爆发步 exec_U 6.2 -> 0.053 rad）。
    # exit_box_invariant（出口箱不变量）：exec 离开求解器前 clamp 回速度箱
    #   ——契约本该如此，第二道保险，不可单用（Round 154 ①原文）。
    # 默认 True = bug 修复即契约；逐位复现 v5 病理行为需显式双关。
    row_authority_clamp: bool = True
    exit_box_invariant: bool = True
    # ── R18 件① 解析栈死区（S8 §4/§6-5，2026-08-21）────────────────────
    # engage_dist（米）：damper 行参与门——margin >= engage_dist 的行不进
    # Dykstra 投影（安全区 exact-passthrough），cap<0 的行（已越 d_min，
    # backstop 强制退开语义）无条件保留（"backstop 原样保留"，S8 网格零违规
    # 的功臣）。None = 全部活跃行照旧参与 = 现行为逐位不变（注意：现行代码
    # 并无 80mm 显式门——活跃集入口是 d_soft 50mm 或 TTC<0.5s，S8 说的
    # "80mm 介入区"是 TTC 远行 + cap 数学的经验效果；因此零漂移默认只能是
    # None 而非 0.08，硬设 0.08 门会裁掉 TTC 远行）。duo_env 从 env yaml
    # safety.damper_warn_mm（毫米）换算注入；R18 配方 = 40mm。
    engage_dist: "float | None" = None
    # R29 (2026-09-02): structural per-link rows (row d_min override != the
    # class lock-line tier) leave the damper projection set (cap>=0 side only;
    # cap<0 forced-retreat rows are always kept). Same row filter as the label
    # side (arm_hazard_gray_flags / arm_min_margin) and the R19 bypass.
    # Default False = bit-identical; caller passes project(struct_exempt=).
    exempt_structural_rows: bool = False
    # R29b: with exempt_structural_rows, a structural row re-enters the damper
    # set once its raw distance drops below this gate (metres) -- keeps a last
    # stop just before the conservative sphere touches the plane while leaving
    # the +1.7mm resting hover unthrottled. None = blanket exemption (R29).
    struct_engage_dist: "float | None" = None
    # R33 (2026-09-06): velocity-aware band. The damper limits the COMMANDED
    # closing speed to gamma*(d-d_min) but the arms track the target with a
    # PD lag (~0.15 s to shed 1 rad/s), so a head-on approach at 0.7 m/s
    # travels ~10 cm after the command stops -- more than the 40 mm engage
    # band (r16a envs 7/24: clutch locked at 36 mm, cross -5.2 mm 8 steps
    # later, headon_violation_trace.py). With lookahead_s set, every row's
    # margin is replaced by its prediction over that horizon using the actual
    # joint velocities, d_eff = d + lookahead_s * min(J qd, 0): fast closing
    # rows engage (and go into forced retreat) earlier, slow/opening rows are
    # untouched. None = bit-identical; caller passes project(qd=).
    lookahead_s: "float | None" = None
    # Optional longer lag horizon for same-robot collision rows (both F and U).
    # The pair-pressure trace showed feasible 60 ms projections followed by
    # 7--10 frames of closing motion from persistent targets. Extend only self
    # rows so cross/table keep their existing response. None preserves legacy
    # output; a shorter value never weakens the global horizon.
    self_lookahead_s: "float | None" = None
    # Extend table braking while retaining the legacy horizon for permanent
    # structural rows. Conditional near-table contact exemptions still need
    # prediction: their velocity condition can cease to hold during PD lag.
    table_lookahead_s: "float | None" = None
    # Contact permission is conditional on measured velocity. Keep damping
    # positive-cap rows before a fast command can invalidate that permission.
    retain_conditional_rows: bool = False
    # Optional stored-target risk prediction. A transient opening velocity
    # must not hide the full closing displacement still held by the PD target.
    # This complements the per-step clipped backlog debit below; it is not
    # an exact nonlinear endpoint-distance prediction.
    predict_backlog: bool = False


class VelocityDamperBackstop:
    """训练版 L2。纯 torch、无状态（p 由策略给，不在这里滞回）。"""

    def __init__(self, cfg: BackstopConfig | None = None, dof_of: dict | None = None):
        self.cfg = cfg or BackstopConfig()
        self.dof_of = dof_of or DOF_OF  # toy 对拍系统可注入 3-DoF 表

    def _alpha_rows(self, c: torch.Tensor, alpha: torch.Tensor, robot: str):
        """进展预算半空间：<u_a, dir_a> <= alpha_a*||c_a||；alpha>=1 的臂不设行。"""
        n = c.shape[0]
        G, h, rel = [], [], []
        i = 0
        for a in ARMS_OF_ROBOT[robot]:
            d = self.dof_of[a]
            ca = c[:, i:i + d]
            norm = ca.norm(dim=-1)
            a_col = alpha[:, _ARM_IDX[a]]
            use = (norm > 1e-9) & (a_col < 1.0 - 1e-6)
            dir_full = torch.zeros_like(c)
            dir_full[:, i:i + d] = ca / norm.clamp_min(1e-9).unsqueeze(-1)
            G.append(dir_full)
            h.append(a_col * norm)
            rel.append(use)
            i += d
        return torch.stack(G, dim=1), torch.stack(h, dim=1), torch.stack(rel, dim=1)

    def project(self, cmd: DeltaCmd, rows: ConstraintRows, alpha: torch.Tensor,
                p: torch.Tensor, dt: float,
                dmin: torch.Tensor | None = None,
                backlog: dict | None = None,
                bypass_arm: "torch.Tensor | None" = None,
                struct_exempt: "torch.Tensor | None" = None,
                qd: "dict | None" = None,
                contact_exempt: "torch.Tensor | None" = None,
                delta_bounds: "dict | None" = None,
                ) -> tuple[DeltaCmd, torch.Tensor, dict]:
        """cmd + 约束行 -> (执行 DeltaCmd, backstop_active (N,4), info)。

        struct_exempt: (N, M) bool, R29 structural per-link row mask -- True
        rows with cap>=0 skip the damper projection (None, or
        cfg.exempt_structural_rows=False, is the bit-identical path).

        delta_bounds: arm -> (lower, upper), target-limit displacement bounds.
        Intersect these with the speed box inside the projection, so a later
        target clamp cannot remove an opening component and reverse safety.

        dmin: (N, M) 逐行刹停边界；None 依次退 rows.d_min、cfg.d_min。
        backlog: arm -> (N, dof) 持久目标积压 q_target - q（增量前）；仅
        cfg.backlog_aware 时消费（None 或 flag off 均为 v3 逐位等价路径）。
        bypass_arm: (N, 4) bool，R19 执行层旁通（S15，2026-08-21）——True 的
        臂指令**逐位直通**（不进箱钳制、不进 Dykstra、不受 alpha 行），解析栈
        对该臂本步完全旁通；掩码由调用方（duo_env）按"离合断开（α_exec=1）
        且该臂涉及的各保留通道裕度 > 应急线（锁线+带宽，豁免行/结构 per-link
        行剔除，口径 = arm_hazard_gray_flags）"算好传入。None = 现行为逐位
        不变（默认，零漂移）。旁通臂 backstop_active 恒 False。逐臂粒度：
        同机另一臂照常投影（共享行的预算它单方面守约，旁通臂由应急带下步
        回收兜底——最后防线语义）。"""
        cfg = self.cfg
        dev = rows.d.device
        n, m = rows.d.shape
        if dmin is None:
            dmin = rows.d_min
        dm = dmin if dmin is not None else torch.full_like(rows.d, cfg.d_min)
        d_eff = rows.d
        if any(t is not None for t in (cfg.lookahead_s, cfg.self_lookahead_s,
                                       cfg.table_lookahead_s)) and qd is not None:
            # R33 velocity-aware band: predicted margin over the lag horizon
            # (closing part only -- opening motion never relaxes a row)
            ddot = torch.zeros_like(rows.d)
            for r in ("F", "U"):
                ddot = ddot + torch.einsum("nmd,nd->nm", rows.J[r], stack_robot(qd, r))
            horizon = cfg.lookahead_s or 0.0
            if cfg.self_lookahead_s is not None:
                horizon = torch.where(rows.cls == CLASS_SELF,
                                      max(horizon, cfg.self_lookahead_s), horizon)
            if cfg.table_lookahead_s is not None:
                table = rows.cls == CLASS_TABLE
                if struct_exempt is not None:
                    permanent = struct_exempt if contact_exempt is None else (
                        struct_exempt & ~contact_exempt)
                    table = table & ~permanent
                horizon = torch.where(table, max(cfg.lookahead_s or 0.0,
                                                cfg.table_lookahead_s), horizon)
            d_eff = rows.d + horizon * ddot.clamp(max=0.0)
        if cfg.predict_backlog:
            if backlog is None:
                raise ValueError('stored-target prediction requires backlog')
            stored_rate = sum(torch.einsum('nmd,nd->nm',rows.J[r],stack_robot(backlog,r))
                              for r in ('F','U'))
            d_eff = torch.minimum(d_eff,rows.d+stored_rate.clamp(max=0.0))
        cap = cfg.gamma * (d_eff - dm) * dt
        # R18 件①：安全区死区——margin >= engage_dist 的行退出投影集合
        # （逐位直通），cap<0 行（强制退开）无条件保留。None = 无门（现行为）。
        engage_gate = None
        if cfg.engage_dist is not None:
            engage_gate = (d_eff < cfg.engage_dist) | (cap < 0.0)
        if cfg.exempt_structural_rows and struct_exempt is not None:
            structural = struct_exempt.to(rows.d.device)
            if cfg.retain_conditional_rows and contact_exempt is not None:
                structural = structural & ~contact_exempt
            drop = structural & (cap >= 0.0)
            if cfg.struct_engage_dist is not None:
                drop = drop & (rows.d >= cfg.struct_engage_dist)
            keep_struct = ~drop
            engage_gate = keep_struct if engage_gate is None else (engage_gate & keep_struct)
        is_cross = rows.cls == CLASS_CROSS
        box = cfg.vmax * dt
        # 行序：margin 从松到紧（padding 记 +inf 排最前，最紧行最后投）
        d_sort = torch.where(rows.valid, d_eff, torch.full_like(rows.d, torch.inf))
        order = torch.argsort(d_sort, dim=1, descending=True)
        k_start = m - int(rows.valid.sum(dim=1).max().item()) if m else 0
        ar = torch.arange(n, device=dev)

        exec_stacked, active_arm, resid, passes_used = {}, {}, {}, {}
        for r in ("F", "U"):
            arms = ARMS_OF_ROBOT[r]
            # 与部署版 QP 同口径：先把指令 clamp 进速度箱，再投影（C 的 c_qp 语义）
            c = stack_robot(cmd.delta_q, r).clamp(-box, box)
            lower, upper = None, None
            if delta_bounds is not None:
                lower = stack_robot({a: delta_bounds[a][0] for a in ARM_KEYS}, r).clamp_min(-box)
                upper = stack_robot({a: delta_bounds[a][1] for a in ARM_KEYS}, r).clamp_max(box)
                if (lower > upper).any():
                    raise ValueError('target bounds have no feasible increment in the speed box')
            nd = c.shape[-1]
            involves = rows.arm_mask[..., [_ARM_IDX[a] for a in arms]].any(-1)
            rel = rows.valid & involves
            if engage_gate is not None:
                rel = rel & engage_gate
            G = -rows.J[r]
            bud = (1.0 + p) * 0.5 if r == "F" else (1.0 - p) * 0.5
            h_cross = torch.where(cap >= 0, bud.unsqueeze(-1) * cap, cap)
            h = torch.where(is_cross, h_cross, cap)
            if cfg.backlog_aware and backlog is not None:
                # subtract the closing authority of the backlog releasable this
                # step (A6-W6)
                b_eff = stack_robot(backlog, r).clamp(-box, box)
                h = h - torch.einsum("nmd,nd->nm", G, b_eff)
            if cfg.row_authority_clamp or (cfg.backlog_aware and backlog is not None):
                # F1 行权柄钳制（Round 154 ①/A10 §4-1）：h >= -0.9*||G_row||_1*box
                # 保证每条半空间在速度箱内可满足（u=-box*sign(G) 仍容许），
                # 物理不可满足行（cap<0 且低 |J|）不再驱动 ε 爆炸修正——
                # 泵病理的能量源在此掐断。backlog_aware 分支自 A6-W6 起就带
                # 此钳制（可行性前提），故该分支下无论 flag 均保留。
                authority = (-(G.abs().sum(-1)) * box if delta_bounds is None else
                             torch.where(G >= 0, G * lower.unsqueeze(1),
                                         G * upper.unsqueeze(1)).sum(-1))
                h = torch.maximum(h, authority * 0.9)
            aG, ah, arel = self._alpha_rows(c, alpha, r)
            n_a = aG.shape[1]
            # Dykstra：集合 = [box] + alpha 行 + 安全行（松->紧）
            u = c.clone()
            z_box = torch.zeros_like(c)
            z_a = torch.zeros(n, n_a, nd, device=dev)
            z_s = torch.zeros(n, m, nd, device=dev)

            def hs_project(y, g, hk, ok):
                viol = ((g * y).sum(-1) - hk).clamp_min(0.0) * ok
                denom = (g * g).sum(-1).clamp_min(cfg.eps)
                return y - (viol / denom).unsqueeze(-1) * g

            it = 0
            for it in range(1, cfg.max_passes + 1):
                u_prev = u
                y = u + z_box
                u = y.clamp(-box, box) if delta_bounds is None else y.maximum(lower).minimum(upper)
                z_box = y - u
                for k in range(n_a):
                    y = u + z_a[:, k]
                    u = hs_project(y, aG[:, k], ah[:, k], arel[:, k].to(u.dtype))
                    z_a[:, k] = y - u
                for k in range(k_start, m):
                    idx = order[:, k]
                    y = u + z_s[ar, idx]
                    u = hs_project(y, G[ar, idx], h[ar, idx], rel[ar, idx].to(u.dtype))
                    z_s[ar, idx] = y - u
                if (u - u_prev).abs().max().item() < cfg.tol:
                    if delta_bounds is None:
                        break
                    # With a saturated target, primal iterates can pause while
                    # Dykstra corrections still move. Require actual feasibility.
                    alpha_error = ((aG @ u.unsqueeze(-1)).squeeze(-1) - ah).clamp_min(0) * arel
                    safety_error = ((G @ u.unsqueeze(-1)).squeeze(-1) - h).clamp_min(0) * rel
                    box_error = torch.maximum((lower - u).clamp_min(0), (u - upper).clamp_min(0))
                    if max(alpha_error.max().item(), safety_error.max().item(), box_error.max().item()) < cfg.tol:
                        break
            if cfg.exit_box_invariant:
                # 出口箱不变量（Round 154 ①/A10 §4-1）：Dykstra 每轮末次投影是
                # 安全行而非箱，早停/轮数耗尽/数值病理任何迭代路径出口都可能
                # 越箱——离开求解器前强制归箱（40/41 突刺例的 100× 越箱输出
                # 正是从这里流向设备侧）。残差按归箱后的真实输出计。
                u = u.clamp(-box, box) if delta_bounds is None else u.maximum(lower).minimum(upper)
            exec_stacked[r] = u
            passes_used[r] = it
            resid[r] = (((G @ u.unsqueeze(-1)).squeeze(-1) - h).clamp_min(0.0)
                        * rel.to(u.dtype)).amax(dim=-1)
            # 参考解：只投 alpha 预算 + box（安全行修剪的归因基准）
            u_ref = c.clamp(-box, box) if delta_bounds is None else c.maximum(lower).minimum(upper)
            for k in range(n_a):
                u_ref = hs_project(u_ref, aG[:, k], ah[:, k], arel[:, k].to(u_ref.dtype))
            diff = (u - u_ref).abs()
            i = 0
            for a in arms:
                d_a = self.dof_of[a]
                active_arm[a] = diff[:, i:i + d_a].amax(dim=-1) > 1e-6
                i += d_a

        exec_ = {}
        for r in ("F", "U"):
            exec_.update(unstack_robot(exec_stacked[r], r, self.dof_of))
        active = torch.stack([active_arm[a] for a in ARM_KEYS], dim=-1)
        info = {
            "residual_F": resid["F"], "residual_U": resid["U"],
            "passes_F": passes_used["F"], "passes_U": passes_used["U"],
            "cap": cap,
        }
        if bypass_arm is not None:
            # R19 旁通：覆写发生在投影之后（对非旁通臂零扰动），旁通臂拿
            # 原始 cmd（未箱钳制——逐位直通的字面含义），active 记 False。
            for i, a in enumerate(ARM_KEYS):
                b = bypass_arm[:, i].unsqueeze(-1)
                exec_[a] = torch.where(b, cmd.delta_q[a].to(exec_[a].dtype),
                                       exec_[a])
            active = active & ~bypass_arm
            info["bypass_arm"] = bypass_arm
        return DeltaCmd(delta_q=exec_), active, info
