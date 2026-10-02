"""球体近似三类距离 + 活跃球对集合提取，v2（owner: Agent A）。

v2 变更（ROUND2_DELTAS）：
- top-K 输出 -> 活跃集合：margin < d_soft 或 TTC < tau_ttc 的球对，
  margin 升序 cap M=32 + mask（active_pairs/active_mask/active_idx）。
- 建对走 ContactSemantics（B 的 contact_semantics.yaml）：邻接豁免、
  逐对 d_min（cross/self/table 分档）、near_table 条件豁免（高度+接近速度
  双条件，运行时覆写 d_min 并抑制 VIOLATION）。无 semantics 时退回 W1
  行为（序号邻接豁免 + table_check 标志），单测两条路都盖。
- soft 类对（臂杆-操作物）W2 无操作物资产，先计数丢弃（B4 落地后接 WARN）。

不变：纯 torch、无 isaac 依赖；球心/球速由 body 位姿速度变换；
env origin 平移不变；min_margin 四桶 {cross,self_F,self_U,table}。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from safeduo.safety.semantics import ContactSemantics
from safeduo.safety.types import ARM_KEYS, CLASS_CROSS, CLASS_SELF, CLASS_TABLE, ROBOT_OF

TABLE_NAMES = ("table_F", "table_U")  # 桌索引 0/1 的语义名（对齐 scene_layout）
_DMIN_DEFAULT = {"cross": 0.03, "self": 0.02, "table": 0.02}


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """wxyz 四元数旋转向量；q:(...,4) v:(...,3)，广播安全。"""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


@dataclass
class LinkSpheres:
    """单个 link 的球组：offsets 在 link 局部系。

    link          = articulation body 名（bind_bodies 用）
    semantic_name = 语义全局名的 local 部分（默认 = link；手部用 "hand/xxx"）
    table_check   = 仅 legacy 路径用（semantics 路径由规则决定）
    """

    link: str
    offsets: list
    radii: list
    table_check: bool = True
    semantic_name: str | None = None


@dataclass
class ArmSpheres:
    links: list  # [LinkSpheres, ...]，顺序 = 运动链顺序


@dataclass
class SphereDistOut:
    active_pairs: torch.Tensor          # (N, M, 4) [dist, closing_vel, class_id, pair_id]
    active_mask: torch.Tensor           # (N, M) bool
    active_idx: torch.Tensor            # (N, M) long，静态 pair 索引（填充=-1）
    active_dmin: torch.Tensor           # (N, M) 逐行有效 d_min（含运行时豁免覆写）
    viol_exempt: torch.Tensor           # (N, M) bool，True=该行接触不算 VIOLATION
    min_margin: dict                    # {cross, self_F, self_U, table} -> (N,)
    violation: torch.Tensor             # (N,) bool：任一非豁免对 margin < 0
    dists: torch.Tensor | None = None   # (N, P) 全量（特权观测用）
    closing: torch.Tensor | None = None
    full_dmin: torch.Tensor | None = None
    full_viol_exempt: torch.Tensor | None = None


class SphereDistanceModule:
    """语义驱动的三类球对距离计算器 + 活跃集提取。

    用法：
        m = SphereDistanceModule(specs, semantics=sem, max_active=32,
                                 d_soft=0.05, tau_ttc=0.5, device="cuda")
        m.bind_bodies({arm: articulation.body_names})
        m.bind_tables(centers(2,3), half_extents(2,3))   # 顺序 = TABLE_NAMES
        out = m.compute(body_pos, body_quat, body_lin_vel, body_ang_vel, env_origins)
    """

    def __init__(self, specs: dict, semantics: ContactSemantics | None = None,
                 max_active: int = 32, d_soft: float = 0.05, tau_ttc: float = 0.5,
                 adjacent_skip: int = 2, d_min_default: dict | None = None,
                 quota_cross: int = 8, device: torch.device | str = "cpu"):
        assert set(specs) == set(ARM_KEYS), f"specs 必须覆盖 {ARM_KEYS}"
        self.specs = specs
        self.sem = semantics
        self.max_active = max_active
        self.d_soft = d_soft
        self.tau_ttc = tau_ttc
        self.quota_cross = quota_cross  # active 集为 cross 行保底的名额
        self.adjacent_skip = adjacent_skip
        self.dmin_default = dict(d_min_default or _DMIN_DEFAULT)
        self.device = torch.device(device)
        self.n_soft_dropped = 0  # soft 类对计数（B4 前不产生约束）
        # 展平球元数据
        offs, rads, arm_id, link_ord, tab_ok, qnames = [], [], [], [], [], []
        self._arm_slices: dict[str, slice] = {}
        s0 = 0
        for ai, arm in enumerate(ARM_KEYS):
            for li, ls in enumerate(specs[arm].links):
                qn = f"{arm}/{ls.semantic_name or ls.link}"
                for o, r in zip(ls.offsets, ls.radii):
                    offs.append(o)
                    rads.append(r)
                    arm_id.append(ai)
                    link_ord.append(li)
                    tab_ok.append(ls.table_check)
                    qnames.append(qn)
            self._arm_slices[arm] = slice(s0, len(rads))
            s0 = len(rads)
        self.n_spheres = len(rads)
        self.qualified_names = qnames
        self.offsets = torch.tensor(offs, dtype=torch.float32, device=self.device)
        self.radii = torch.tensor(rads, dtype=torch.float32, device=self.device)
        self.arm_id = torch.tensor(arm_id, dtype=torch.long, device=self.device)
        self.link_ord = torch.tensor(link_ord, dtype=torch.long, device=self.device)
        self.table_ok = torch.tensor(tab_ok, dtype=torch.bool, device=self.device)
        robot_id = [0 if ROBOT_OF[ARM_KEYS[a]] == "F" else 1 for a in arm_id]
        self.robot_id = torch.tensor(robot_id, dtype=torch.long, device=self.device)
        self._build_pairs()
        self._body_idx: dict[str, torch.Tensor] = {}
        self._tables: tuple[torch.Tensor, torch.Tensor] | None = None
        # geometry provider 消费的每步缓存（compute 时填）
        self.last_centers: torch.Tensor | None = None
        self.last_table_grad: torch.Tensor | None = None
        # 评测消费的每步缓存（2026-08-15 C11 豁免盲口径修复）：table 槽位的
        # 逐对裸边距 + 逐对 VIOLATION 豁免掩码。仅暴露 compute 既有中间量，
        # 官方输出（violation / min_margin）语义与数值不变。
        self.last_table_margin: torch.Tensor | None = None
        self.last_table_viol_exempt: torch.Tensor | None = None

    # ---- 建对（init 一次）----

    def _judge_link_pair(self, i: int, j: int):
        """返回 (keep, category, d_min)；无 semantics 走 legacy 规则。

        P0 前置豁免（两条路径统一，2026-08-12 监管线）：同臂内同 link（序数差 0，
        B 的管状链式球设计上就互相重叠）与邻近序数（差 < adjacent_skip）的球对
        不进危险对——否则 init 即永久假 VIOLATION 且幽灵 self 行挤爆 active 集。
        B 的 adjacency 显式表在此之上继续豁免它列出的跨序数对。
        """
        ai, aj = int(self.arm_id[i]), int(self.arm_id[j])
        if ai == aj and abs(int(self.link_ord[i]) - int(self.link_ord[j])) < self.adjacent_skip:
            return False, "", 0.0
        if self.sem is not None:
            v = self.sem.judge(self.qualified_names[i], self.qualified_names[j])
            if v.keep and v.verdict == "soft":
                self.n_soft_dropped += 1
                return False, "", 0.0
            return v.keep, v.category, v.d_min
        # legacy：跨机全保留；同机不同臂保留；同臂序号差 >= skip 保留（前置已滤）
        ri, rj = int(self.robot_id[i]), int(self.robot_id[j])
        if ri != rj:
            return True, "cross", self.dmin_default["cross"]
        cat = "self_F" if ri == 0 else "self_U"
        return True, cat, self.dmin_default["self"]

    def _judge_table_pair(self, i: int, t: int):
        """返回 (keep, d_min, conditional)。"""
        if self.sem is not None:
            v = self.sem.judge(self.qualified_names[i], TABLE_NAMES[t])
            return v.keep, (v.d_min if v.keep else 0.0), v.conditional
        if bool(self.table_ok[i]):
            return True, self.dmin_default["table"], False
        return False, 0.0, False

    def _build_pairs(self) -> None:
        cross, selfp, self_rob = [], [], []
        dmin_cross, dmin_self = [], []
        for i in range(self.n_spheres):
            for j in range(i + 1, self.n_spheres):
                keep, cat, dm = self._judge_link_pair(i, j)
                if not keep:
                    continue
                if cat == "cross":
                    cross.append((i, j))
                    dmin_cross.append(dm)
                else:
                    selfp.append((i, j))
                    self_rob.append(0 if cat == "self_F" else 1)
                    dmin_self.append(dm)
        table, dmin_tab, cond_tab = [], [], []
        for i in range(self.n_spheres):
            for t in range(len(TABLE_NAMES)):
                keep, dm, cond = self._judge_table_pair(i, t)
                if keep:
                    table.append((i, t))
                    dmin_tab.append(dm)
                    cond_tab.append(cond)

        def _t(x, dtype=torch.long):
            return torch.tensor(x if x else [], dtype=dtype, device=self.device)

        self.pairs_cross = _t(cross).reshape(-1, 2)
        self.pairs_self = _t(selfp).reshape(-1, 2)
        self.pairs_table = _t(table).reshape(-1, 2)
        self.self_robot = _t(self_rob)
        pc, ps, pt = len(self.pairs_cross), len(self.pairs_self), len(self.pairs_table)
        self.n_pairs = pc + ps + pt
        self.class_id = torch.cat([
            torch.full((pc,), CLASS_CROSS, device=self.device),
            torch.full((ps,), CLASS_SELF, device=self.device),
            torch.full((pt,), CLASS_TABLE, device=self.device),
        ])
        self.pair_id = torch.arange(self.n_pairs, dtype=torch.float32, device=self.device)
        self.pair_dmin = torch.cat([
            _t(dmin_cross, torch.float32), _t(dmin_self, torch.float32),
            _t(dmin_tab, torch.float32),
        ])
        self.pair_conditional = torch.cat([
            torch.zeros(pc + ps, dtype=torch.bool, device=self.device),
            _t(cond_tab, torch.bool),
        ])
        self.pair_table = torch.cat([self.pairs_cross, self.pairs_self, self.pairs_table], dim=0)
        # R15 ①：逐对两实体的臂索引查表 (P, 2)，ARM_KEYS 序；对桌行第二实体
        # 非臂 -> -1（arm-aware 观测里 one-hot 全零）。首列全体行都是球，直接
        # 走 arm_id；纯静态、init 一次，无消费者时零行为影响。
        arm_a = (self.arm_id[self.pair_table[:, 0]] if self.n_pairs
                 else torch.zeros(0, dtype=torch.long, device=self.device))
        arm_b = torch.cat([
            self.arm_id[self.pairs_cross[:, 1]] if pc else
            torch.zeros(0, dtype=torch.long, device=self.device),
            self.arm_id[self.pairs_self[:, 1]] if ps else
            torch.zeros(0, dtype=torch.long, device=self.device),
            torch.full((pt,), -1, dtype=torch.long, device=self.device),
        ])
        self.pair_arms = torch.stack([arm_a, arm_b], dim=-1)
        self._slice_cross = slice(0, pc)
        self._slice_self = slice(pc, pc + ps)
        self._slice_table = slice(pc + ps, self.n_pairs)

    # ---- 绑定 ----

    def bind_bodies(self, body_names: dict) -> None:
        for arm in ARM_KEYS:
            names = list(body_names[arm])
            idx = []
            for ls in self.specs[arm].links:
                assert ls.link in names, f"{arm}: link {ls.link} 不在 body_names {names}"
                idx.extend([names.index(ls.link)] * len(ls.radii))
            self._body_idx[arm] = torch.tensor(idx, dtype=torch.long, device=self.device)

    def bind_tables(self, centers: torch.Tensor, half_extents: torch.Tensor) -> None:
        """两张桌的 AABB（env 局部系）：顺序必须 = TABLE_NAMES（0=table_F,1=table_U）。"""
        self._tables = (
            centers.to(self.device, torch.float32),
            half_extents.to(self.device, torch.float32),
        )

    # ---- 运行时 ----

    def _centers_vels(self, body_pos, body_quat, body_lin_vel, body_ang_vel):
        cs, vs = [], []
        for arm in ARM_KEYS:
            bi = self._body_idx[arm]
            sl = self._arm_slices[arm]
            p = body_pos[arm][:, bi]
            q = body_quat[arm][:, bi]
            r_off = quat_rotate(q, self.offsets[sl].unsqueeze(0).expand(p.shape[0], -1, -1))
            cs.append(p + r_off)
            v = body_lin_vel[arm][:, bi] + torch.cross(body_ang_vel[arm][:, bi], r_off, dim=-1)
            vs.append(v)
        return torch.cat(cs, dim=1), torch.cat(vs, dim=1)

    @staticmethod
    def _pair_dist_closing(centers, vels, radii, pairs):
        if pairs.numel() == 0:
            n = centers.shape[0]
            e = torch.zeros(n, 0, device=centers.device)
            return e, e
        ci, cj = centers[:, pairs[:, 0]], centers[:, pairs[:, 1]]
        rel = ci - cj
        d = rel.norm(dim=-1)
        nrm = rel / d.clamp_min(1e-9).unsqueeze(-1)
        rel_v = vels[:, pairs[:, 0]] - vels[:, pairs[:, 1]]
        closing = -(nrm * rel_v).sum(dim=-1)
        margin = d - (radii[pairs[:, 0]] + radii[pairs[:, 1]])
        return margin, closing

    def _table_dist_closing(self, centers, vels, env_origins):
        tc, th = self._tables
        if self.pairs_table.numel() == 0:
            n = centers.shape[0]
            e = torch.zeros(n, 0, device=centers.device)
            return e, e, e
        tcw = env_origins.unsqueeze(1) + tc.unsqueeze(0)
        sph = self.pairs_table[:, 0]
        tab = self.pairs_table[:, 1]
        p = centers[:, sph]
        v = vels[:, sph]
        q = (p - tcw[:, tab]).abs() - th[tab].unsqueeze(0)
        outside = q.clamp_min(0.0)
        d_out = outside.norm(dim=-1)
        d_in = q.amax(dim=-1).clamp_max(0.0)
        sdf = d_out + d_in
        margin = sdf - self.radii[sph]
        sgn = torch.sign(p - tcw[:, tab])
        grad_out = sgn * outside / d_out.clamp_min(1e-9).unsqueeze(-1)
        inside_axis = torch.nn.functional.one_hot(q.argmax(dim=-1), 3).to(p.dtype)
        grad_in = sgn * inside_axis
        grad = torch.where((d_out > 0).unsqueeze(-1), grad_out, grad_in)
        self.last_table_grad = grad
        closing = -(grad * v).sum(dim=-1)
        # 球心相对桌面顶面高度（near_table 豁免条件用；高于桌面为正）
        height = p[..., 2] - (tcw[:, tab][..., 2] + th[tab][:, 2].unsqueeze(0))
        return margin, closing, height

    def compute(self, body_pos: dict, body_quat: dict, body_lin_vel: dict,
                body_ang_vel: dict, env_origins: torch.Tensor,
                need_full: bool = False) -> SphereDistOut:
        assert self._body_idx and self._tables is not None, "先 bind_bodies / bind_tables"
        centers, vels = self._centers_vels(body_pos, body_quat, body_lin_vel, body_ang_vel)
        self.last_centers = centers
        d_cross, c_cross = self._pair_dist_closing(centers, vels, self.radii, self.pairs_cross)
        d_self, c_self = self._pair_dist_closing(centers, vels, self.radii, self.pairs_self)
        d_tab, c_tab, h_tab = self._table_dist_closing(centers, vels, env_origins)
        n = centers.shape[0]
        dists = torch.cat([d_cross, d_self, d_tab], dim=1)
        closing = torch.cat([c_cross, c_self, c_tab], dim=1)
        # 逐行有效 d_min + VIOLATION 豁免（near_table 双条件，只作用于 conditional 行）
        dmin_eff = self.pair_dmin.unsqueeze(0).expand(n, -1).clone()
        viol_exempt = torch.zeros(n, self.n_pairs, dtype=torch.bool, device=self.device)
        if self.sem is not None and self.pair_conditional.any():
            ex = self.sem.near_table
            cond_cols = self.pair_conditional[self._slice_table]
            low = h_tab < ex.height_max
            slow = c_tab < ex.closing_max
            exempt = low & slow & cond_cols.unsqueeze(0)
            sl = self._slice_table
            dmin_eff[:, sl] = torch.where(exempt, torch.full_like(d_tab, ex.d_min_override),
                                          dmin_eff[:, sl])
            if not ex.violation_on_contact:
                viol_exempt[:, sl] = exempt
        # 每步缓存（C11）：endurance_eval 等评测在 rows 层做豁免感知的通道级
        # 重切用；legacy/无 conditional 路径下掩码恒 False（无豁免语义）。
        self.last_table_margin = d_tab
        self.last_table_viol_exempt = viol_exempt[:, self._slice_table]
        # 活跃集：margin < d_soft 或 0 < TTC < tau；cross 类保底 quota_cross 行
        # （防 self/table 行挤爆 M 名额造成协调器观测饥饿——正确性问题）
        ttc = torch.where(closing > 1e-6, dists / closing.clamp_min(1e-6),
                          torch.full_like(dists, torch.inf))
        eligible = (dists < self.d_soft) | ((ttc > 0) & (ttc < self.tau_ttc))
        inf_like = torch.full_like(dists, torch.inf)
        score = torch.where(eligible, dists, inf_like)
        m = min(self.max_active, self.n_pairs)
        q = min(self.quota_cross, m, max(self._slice_cross.stop, 0))
        if q > 0:
            is_cross_col = self.class_id.unsqueeze(0) == CLASS_CROSS
            score_cross = torch.where(is_cross_col, score, inf_like)
            _, idx_c = torch.topk(score_cross, q, dim=1, largest=False)
            score_rest = score.scatter(1, idx_c, torch.inf)  # 去重
            _, idx_r = torch.topk(score_rest, m - q, dim=1, largest=False)
            cand = torch.cat([idx_c, idx_r], dim=1)
            cand_v = score.gather(1, cand)
            cand_v, ord_i = torch.sort(cand_v, dim=1)        # 合并后仍按 margin 升序
            top_i = cand.gather(1, ord_i)
            top_v = cand_v
        else:
            top_v, top_i = torch.topk(score, m, dim=1, largest=False)
        mask = torch.isfinite(top_v)
        feat = torch.stack([
            dists.gather(1, top_i), closing.gather(1, top_i),
            self.class_id[top_i], self.pair_id[top_i],
        ], dim=-1)
        feat = torch.where(mask.unsqueeze(-1), feat, torch.zeros_like(feat))
        feat[..., 3] = torch.where(mask, feat[..., 3], torch.full_like(feat[..., 3], -1.0))
        active_idx = torch.where(mask, top_i, torch.full_like(top_i, -1))
        active_dmin = torch.where(mask, dmin_eff.gather(1, top_i), torch.zeros_like(top_v))
        active_exempt = torch.where(mask, viol_exempt.gather(1, top_i),
                                    torch.zeros_like(mask))
        # min_margin 四桶 + VIOLATION（非豁免对 margin<0）
        is_f = self.self_robot == 0
        inf_col = torch.full((n,), torch.inf, device=self.device)
        min_margin = {
            "cross": d_cross.amin(dim=1) if d_cross.shape[1] else inf_col,
            "self_F": d_self[:, is_f].amin(dim=1) if is_f.any() else inf_col,
            "self_U": d_self[:, ~is_f].amin(dim=1) if (~is_f).any() else inf_col,
            "table": d_tab.amin(dim=1) if d_tab.shape[1] else inf_col,
        }
        violation = ((dists < 0.0) & ~viol_exempt).any(dim=1)
        return SphereDistOut(
            active_pairs=feat, active_mask=mask, active_idx=active_idx,
            active_dmin=active_dmin, viol_exempt=active_exempt,
            min_margin=min_margin, violation=violation,
            dists=dists if need_full else None,
            closing=closing if need_full else None,
            full_dmin=dmin_eff if need_full else None,
            full_viol_exempt=viol_exempt if need_full else None,
        )

    def prioritized_out(self, out: SphereDistOut, horizon: float) -> SphereDistOut:
        """Select safety rows by predicted lock-line slack; preserve actor observations.

        Quiet small-gap rows must not hide a farther fast-closing row during
        braking lag. This remains a bounded active set, not an all-pair guarantee.
        """
        if horizon <= 0:
            raise ValueError('safety row horizon must be positive')
        if any(x is None for x in (out.dists, out.closing, out.full_dmin, out.full_viol_exempt)):
            raise ValueError('prioritized safety rows require a full distance snapshot')
        d, closing = out.dists, out.closing
        predicted = d - horizon * closing.clamp_min(0)
        ttc = torch.where(closing > 1e-6, d / closing.clamp_min(1e-6), torch.full_like(d, torch.inf))
        eligible = (predicted < self.d_soft) | ((ttc > 0) & (ttc < self.tau_ttc))
        score = (predicted - out.full_dmin).masked_fill(~eligible, torch.inf)
        m = min(self.max_active, self.n_pairs)
        q = min(self.quota_cross, m, self._slice_cross.stop)
        if q:
            cross_score = score.masked_fill(self.class_id.unsqueeze(0) != CLASS_CROSS, torch.inf)
            idx_c = torch.topk(cross_score, q, dim=1, largest=False).indices
            rest = score.scatter(1, idx_c, torch.inf)
            idx_r = torch.topk(rest, m - q, dim=1, largest=False).indices
            idx = torch.cat([idx_c, idx_r], dim=1)
            order = score.gather(1, idx).argsort(dim=1)
            idx = idx.gather(1, order)
        else:
            idx = torch.topk(score, m, dim=1, largest=False).indices
        valid = torch.isfinite(score.gather(1, idx))
        feat = torch.stack([d.gather(1, idx), closing.gather(1, idx),
                            self.class_id[idx], self.pair_id[idx]], -1)
        feat = feat.masked_fill(~valid.unsqueeze(-1), 0)
        feat[..., 3] = torch.where(valid, feat[..., 3], -1.)
        return replace(out, active_pairs=feat, active_mask=valid,
                       active_idx=torch.where(valid, idx, -1),
                       active_dmin=out.full_dmin.gather(1, idx).masked_fill(~valid, 0),
                       viol_exempt=out.full_viol_exempt.gather(1, idx) & valid)
