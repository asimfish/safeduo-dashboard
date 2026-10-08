"""Per-step margin-proximity shaping / cost signals (credit-assignment fix).

Motivation (STATUS_SERVER 08-12 03:20, A2 finding #1): duo_env's reward
carries the violation penalty only at the step the violation happens (and the
episode terminates there), so with gamma=0.99 over 600-step episodes the
terminal signal reaching early actions is ~0.99^600 ~= 0.002 -- the v1 policy
learned "stop a lot" without learning *where* stopping matters. FINAL_PROPOSAL
prescribes conflict shaping; the audited duo_env._get_rewards (W3) has track /
tube / progress / smooth / backstop-depth / terminal-violation terms but NO
margin-proximity term. This module provides it, plus the per-class cost
channels consumed by the PPO-Lagrangian port (algo/lagrangian.py).

Definition. For each constraint row (or active pair) with surface margin d
and per-row braking boundary d_min:

    x = (d_warn - d) / (d_warn - d_min)      # 0 at the warn boundary,
    cost = clamp(x, 0, clip)^2               # 1 exactly at d = d_min

The per-env scalar is the max over rows (worst pair), keeping the scale
bounded in [0, clip^2] regardless of how many pairs are active. Quadratic in
the *normalized* gap == the (d_warn - d)^2 shaping of the work order with the
per-pair d_min calibration A's semantics tables already provide.

Wiring proposal (@A2, both are one-liners inside duo_env; weights ->
configs/duo_env.yaml coordinator.reward):

  _pre_physics_step:  cache the cost while rows are in scope
      c["margin_cost"] = shaped_margin_cost(rows.d, rows.d_min, rows.valid,
                                            d_warn=self._sph.d_soft)
  _get_rewards:       r += -w_margin * c["margin_cost"]

Recommended starting weight: w_margin = 2.0 (max 2/step at the braking
boundary vs 100 terminal; at half band the term is 0.5). Registered as a
planned sweep in experiments.csv (c2_credit_*).
"""

from __future__ import annotations

import torch

from safeduo.safety.types import CLASS_CROSS, CLASS_SELF, CLASS_TABLE

# semantics thresholds (assets_src/contact_semantics.yaml); callers should
# pass the live values from ContactSemantics / env cfg where available
D_WARN_DEFAULT = 0.05
D_MIN_BY_CLASS_DEFAULT = {CLASS_CROSS: 0.03, CLASS_SELF: 0.02, CLASS_TABLE: 0.02}


def shaped_margin_cost(d: torch.Tensor, d_min: torch.Tensor,
                       valid: torch.Tensor, d_warn: float = D_WARN_DEFAULT,
                       clip: float = 2.0) -> torch.Tensor:
    """(N, R) constraint-row margins -> (N,) worst-pair proximity cost.

    d      (N, R) surface margins (m), may be negative (penetration)
    d_min  (N, R) per-row braking boundary (A's pair_dmin)
    valid  (N, R) bool, False rows ignored
    """
    band = (d_warn - d_min).clamp_min(1e-6)
    x = ((d_warn - d) / band).clamp(min=0.0, max=clip)
    cost = torch.where(valid, x * x, torch.zeros_like(x))
    if cost.shape[-1] == 0:
        return torch.zeros(cost.shape[0], device=cost.device, dtype=cost.dtype)
    return cost.amax(dim=-1)


def _pairs_cost(active_pairs: torch.Tensor, active_mask: torch.Tensor,
                d_warn: float, d_min_by_class: dict, clip: float) -> torch.Tensor:
    """(N, M, 4) active set -> (N, M) per-row cost with class-resolved d_min."""
    d = active_pairs[..., 0]
    cls = active_pairs[..., 2]
    d_min = torch.full_like(d, D_MIN_BY_CLASS_DEFAULT[CLASS_SELF])
    for c, v in d_min_by_class.items():
        d_min = torch.where(cls == c, torch.full_like(d, v), d_min)
    band = (d_warn - d_min).clamp_min(1e-6)
    x = ((d_warn - d) / band).clamp(min=0.0, max=clip)
    return torch.where(active_mask, x * x, torch.zeros_like(x))


def shaped_margin_cost_from_pairs(
        active_pairs: torch.Tensor, active_mask: torch.Tensor,
        d_warn: float = D_WARN_DEFAULT,
        d_min_by_class: "dict | None" = None,
        clip: float = 2.0) -> torch.Tensor:
    """Same cost computed from the SceneState active-pair set (N, M, 4).

    For local rollouts / BC / eval where ConstraintRows are not in scope.
    Note the active set is the M=32 capped selection -- under observation
    starvation the true worst pair is guaranteed present only because A's
    selection is margin-ascending with a cross quota.
    """
    cost = _pairs_cost(active_pairs, active_mask, d_warn,
                       d_min_by_class or D_MIN_BY_CLASS_DEFAULT, clip)
    if cost.shape[-1] == 0:
        return torch.zeros(cost.shape[0], device=cost.device, dtype=cost.dtype)
    return cost.amax(dim=-1)


def margin_cost_by_class(
        active_pairs: torch.Tensor, active_mask: torch.Tensor,
        d_warn: float = D_WARN_DEFAULT,
        d_min_by_class: "dict | None" = None,
        clip: float = 2.0) -> torch.Tensor:
    """Per-class cost channels (N, 3) ordered [cross, self, table].

    These are the PPO-Lagrangian constraint channels: episode-mean of each
    column is the observed cost the PID controller regulates against its
    limit (see algo/lagrangian.py recommended config).
    """
    cost = _pairs_cost(active_pairs, active_mask, d_warn,
                       d_min_by_class or D_MIN_BY_CLASS_DEFAULT, clip)
    cls = active_pairs[..., 2]
    out = []
    for c in (CLASS_CROSS, CLASS_SELF, CLASS_TABLE):
        sel = torch.where((cls == c) & active_mask, cost, torch.zeros_like(cost))
        out.append(sel.amax(dim=-1) if sel.shape[-1] else
                   torch.zeros(cost.shape[0], device=cost.device))
    return torch.stack(out, dim=-1)


def arm_alpha_util_gate(active_pairs: torch.Tensor, active_mask: torch.Tensor,
                        pair_arms: torch.Tensor, d_warn: float,
                        d_soft: float) -> torch.Tensor:
    """R15 ③a：逐臂"无险情"门 (N, 4) bool，True = 该臂本步可领 α→1 加分。

    险情判定（任一即封门，逐臂独立）：
    - 逼近：该臂参与的任一活跃行 margin <= d_warn（"最近对 margin > d_warn"
      的逆命题——活跃集 margin 升序保证真最近对在集内）；
    - tube：该臂参与的任一 cross 行 margin < d_soft 且 closing > 0（与
      duo_env._pre_physics_step 的 tube 判定同式，细化到臂）。
      d_soft <= d_warn 时 tube ⊂ 逼近，仍显式保留（语义独立、防阈值反转）。

    pair_arms (P, 2) long = SphereDistanceModule.pair_arms（-1 = 非臂实体）。
    padding 行 pair_id=-1 clamp 后误查行 0，由 mask 出清。无活跃行的臂 =
    无险情（门开）。纯 torch，无 isaac 依赖，本地可测。"""
    pid = active_pairs[..., 3].long().clamp_min(0)
    arms = pair_arms.to(active_pairs.device)[pid]              # (N, M, 2)
    d = active_pairs[..., 0]
    near = active_mask & (d <= d_warn)
    tube = (active_mask & (active_pairs[..., 2] == CLASS_CROSS)
            & (d < d_soft) & (active_pairs[..., 1] > 0))
    row_danger = near | tube                                   # (N, M)
    ids = torch.arange(4, device=active_pairs.device)
    involved = (arms.unsqueeze(-1) == ids).any(dim=-2)         # (N, M, 4)
    danger = (row_danger.unsqueeze(-1) & involved).any(dim=1)  # (N, 4)
    return ~danger


def alpha_util_bonus(alpha: torch.Tensor, no_danger: torch.Tensor,
                     w_alpha_util: float) -> torch.Tensor:
    """R15 ③a：α utility 显式定价 (N,) —— 无险情臂上的 -w·(α-1)² 势能罚
    （α→1 罚为 0，即"敞开走"拿满分；量纲对照 w_track=10，推荐 w=0.5，
    四臂满偏离时至多 -2/步）。险情臂零贡献——α 收缩交给约束通道定价，
    不与 λ 打架。w=0 时调用方应直接跳过（零漂移由 duo_env 侧开关保证）。"""
    return -w_alpha_util * ((alpha - 1.0).pow(2)
                            * no_danger.to(alpha.dtype)).sum(dim=-1)


def arm_hazard_gray_flags(active_pairs: torch.Tensor,
                          active_mask: torch.Tensor,
                          active_dmin: torch.Tensor,
                          viol_exempt: "torch.Tensor | None",
                          pair_arms: torch.Tensor,
                          class_dmin: torch.Tensor,
                          gray_band: torch.Tensor,
                          coop_vmax: float = 0.0,
                          coop_depth_frac: float = 0.5) -> torch.Tensor:
    """R18 件③/④：逐臂 [hazard, gray] 旗标 (N, 4, 2) bool（本步即时口径，
    hindsight 前瞻由训练器在 rollout buffer 里反向扫描完成）。

    hazard：该臂参与的任一保留行 margin < 锁线（active_dmin 逐行真值）；
    gray：非 hazard，且 margin < 锁线 + 逐类灰带宽（class 用 gray_band
    [cross, self, table] 查表）——对应执行侧滞回走廊，死区罚在此零罚。

    保留行 = 活跃 & 非豁免 & 结构行剔除。结构行 = 逐行 active_dmin 与该类
    档位 class_dmin 不符的 per-link 覆写行（v7 upper_arm_link x table 的
    0.001 档常年悬停 +1.7mm——刹车解决不了它，进标注会把 U 臂钉死在灰区/
    危险区，正是 λ_table 顶积分限幅那类结构性不可满足的标注版）。豁免行 =
    合法 near_table 停靠（低+慢），与 cost 通道同一出清逻辑（C14）。

    pair_arms (P, 2) long = SphereDistanceModule.pair_arms（-1 = 非臂）。
    padding 行 pair_id=-1 clamp 后误查行 0，由 mask 出清。纯 torch。"""
    dev = active_pairs.device
    d = active_pairs[..., 0]
    cls = active_pairs[..., 2].long().clamp(0, class_dmin.numel() - 1)
    tier = class_dmin.to(dev)[cls]
    band = gray_band.to(dev)[cls]
    keep = active_mask
    if viol_exempt is not None:
        keep = keep & ~viol_exempt
    keep = keep & ((active_dmin - tier).abs() <= 1e-9)
    row_hazard = keep & (d < active_dmin)
    # a27: controlled-rendezvous exemption. loop-4 postmortem (v2.24-v2.28):
    # cooperative hand-off approaches necessarily dip below the CROSS lock
    # line, and the immediate-caliber label (then hindsight-propagated 30
    # steps back) taught the hazard head "any rendezvous = danger" -- the
    # root cause of the handover gating failure AND the head-source
    # engage-lock loop; every eval-side static remap was proven unable to
    # fix it. Exempt CROSS rows that are (a) slow (closing < coop_vmax,
    # signed: receding rows are always slow) and (b) not deep intrusions
    # (d > coop_depth_frac * lock line). Exempted rows fall into the gray
    # corridor below. coop_vmax <= 0 keeps the old path bit-identical.
    if coop_vmax > 0.0:
        coop = ((active_pairs[..., 2] == CLASS_CROSS)
                & (active_pairs[..., 1] < coop_vmax)
                & (d > coop_depth_frac * active_dmin))
        row_hazard = row_hazard & ~coop
    row_gray = keep & ~row_hazard & (d < active_dmin + band)
    pid = active_pairs[..., 3].long().clamp_min(0)
    arms = pair_arms.to(dev)[pid]                               # (N, M, 2)
    ids = torch.arange(4, device=dev)
    involved = (arms.unsqueeze(-1) == ids).any(dim=-2)          # (N, M, 4)
    hazard = (row_hazard.unsqueeze(-1) & involved).any(dim=1)   # (N, 4)
    gray = (row_gray.unsqueeze(-1) & involved).any(dim=1)
    return torch.stack([hazard, gray], dim=-1)


def arm_class_near_flags(active_pairs: torch.Tensor,
                         active_mask: torch.Tensor,
                         active_dmin: torch.Tensor,
                         viol_exempt: "torch.Tensor | None",
                         pair_arms: torch.Tensor,
                         class_dmin: torch.Tensor,
                         gray_band: torch.Tensor,
                         cls_id: float = CLASS_CROSS) -> torch.Tensor:
    """a31b: per-arm bool (N, 4) -- the arm participates in a retained row of
    class `cls_id` that is inside lock line + that class's gray band.

    Retained-row caliber is identical to arm_hazard_gray_flags (active &
    non-exempt & structural rows removed). Used to gate the refine-stage
    backstop-intervention label: a31 (2026-09-11) labelled *any* backstop
    activity as hazard, and 100% of backstop-active steps sat at
    mm_table <= 25 mm (only 5-7% also had cross <= 38 mm), so the policy
    learned "near table = danger" and false-braked 62% of safe steps. Table /
    self projections are the analytic layer's job, not the clutch's; only
    cross-arm-adjacent interventions are clutch evidence."""
    dev = active_pairs.device
    d = active_pairs[..., 0]
    cls = active_pairs[..., 2].long().clamp(0, class_dmin.numel() - 1)
    tier = class_dmin.to(dev)[cls]
    band = gray_band.to(dev)[cls]
    keep = active_mask
    if viol_exempt is not None:
        keep = keep & ~viol_exempt
    keep = keep & ((active_dmin - tier).abs() <= 1e-9)
    row = keep & (active_pairs[..., 2] == cls_id) & (d < active_dmin + band)
    pid = active_pairs[..., 3].long().clamp_min(0)
    arms = pair_arms.to(dev)[pid]                               # (N, M, 2)
    ids = torch.arange(4, device=dev)
    involved = (arms.unsqueeze(-1) == ids).any(dim=-2)          # (N, M, 4)
    return (row.unsqueeze(-1) & involved).any(dim=1)            # (N, 4)


def arm_min_margin(active_pairs: torch.Tensor,
                   active_mask: torch.Tensor,
                   active_dmin: torch.Tensor,
                   viol_exempt: "torch.Tensor | None",
                   pair_arms: torch.Tensor,
                   class_dmin: torch.Tensor,
                   cap: float = 0.25) -> torch.Tensor:
    """a25/P2：逐臂最小锁线余量 min(d - active_dmin) (N, 4) float。

    保留行口径与 arm_hazard_gray_flags 逐行相同（活跃 & 非豁免 & 结构行
    剔除）——结构行（per-link 覆写常年悬停 +1.7mm 那类）若入选会恒定占据
    U 臂的 min，把这维特征钉死、真实威胁（5-20mm 逼近）反而被遮蔽，剔除
    理由与 bool 标注侧一致。无保留行的臂读 cap（远处安全的哨兵值）；下限
    -0.05 截断（越锁线 5cm 之外的深度不再携带新信息）。"""
    dev = active_pairs.device
    d = active_pairs[..., 0]
    cls = active_pairs[..., 2].long().clamp(0, class_dmin.numel() - 1)
    tier = class_dmin.to(dev)[cls]
    keep = active_mask
    if viol_exempt is not None:
        keep = keep & ~viol_exempt
    keep = keep & ((active_dmin - tier).abs() <= 1e-9)
    pid = active_pairs[..., 3].long().clamp_min(0)
    arms = pair_arms.to(dev)[pid]                               # (N, M, 2)
    ids = torch.arange(4, device=dev)
    involved = (arms.unsqueeze(-1) == ids).any(dim=-2)          # (N, M, 4)
    margin = (d - active_dmin).unsqueeze(-1).expand(-1, -1, 4)
    margin = torch.where(keep.unsqueeze(-1) & involved, margin,
                         torch.full_like(margin, cap))
    return margin.min(dim=1).values.clamp(-0.05, cap)


def arm_bypass_block_by_class(active_pairs: torch.Tensor,
                              active_mask: torch.Tensor,
                              active_dmin: torch.Tensor,
                              viol_exempt: "torch.Tensor | None",
                              pair_arms: torch.Tensor,
                              class_dmin: torch.Tensor,
                              band: torch.Tensor) -> torch.Tensor:
    """R19 旁通阻断的逐类归因 (N, 4, 3, 2) bool，末维 = [hazard, gray]。

    行口径与 arm_hazard_gray_flags 逐行相同（保留行 = 活跃 & 非豁免 &
    结构行剔除；hazard = margin < 锁线；gray = 锁线 <= margin < 锁线+带宽），
    只是按类（[cross, self, table]）分桶而不做类间 any——供评测侧回答
    "是谁把这臂挡在非直通区"。等价性钉：any(dim=(2,3)) ==
    arm_hazard_gray_flags(...).any(-1)（tests/test_r19_bypass.py）。"""
    dev = active_pairs.device
    d = active_pairs[..., 0]
    n_cls = class_dmin.numel()
    cls = active_pairs[..., 2].long().clamp(0, n_cls - 1)
    tier = class_dmin.to(dev)[cls]
    bandr = band.to(dev)[cls]
    keep = active_mask
    if viol_exempt is not None:
        keep = keep & ~viol_exempt
    keep = keep & ((active_dmin - tier).abs() <= 1e-9)
    row_hazard = keep & (d < active_dmin)
    row_gray = keep & ~row_hazard & (d < active_dmin + bandr)
    pid = active_pairs[..., 3].long().clamp_min(0)
    arms = pair_arms.to(dev)[pid]                               # (N, M, 2)
    ids = torch.arange(4, device=dev)
    involved = (arms.unsqueeze(-1) == ids).any(dim=-2)          # (N, M, 4)
    out = torch.zeros(d.shape[0], 4, n_cls, 2, dtype=torch.bool, device=dev)
    for c in range(n_cls):
        row_c = cls == c
        out[:, :, c, 0] = ((row_hazard & row_c).unsqueeze(-1)
                           & involved).any(dim=1)
        out[:, :, c, 1] = ((row_gray & row_c).unsqueeze(-1)
                           & involved).any(dim=1)
    return out


def margin_cost_by_class_exempt(
        active_pairs: torch.Tensor, active_mask: torch.Tensor,
        active_dmin: torch.Tensor,
        viol_exempt: "torch.Tensor | None" = None,
        d_warn: float = D_WARN_DEFAULT,
        clip: float = 2.0) -> torch.Tensor:
    """Exemption-aware per-class channels (N, 3) [cross, self, table] from the
    env-side active set with PER-ROW d_min (C14, C12_HANDOFF §2b + §3b).

    Differences vs margin_cost_by_class (the obs reconstruction):
    - active_dmin (N, M) is SphereDistOut.active_dmin -- the baked per-pair
      d_min, so per-link table overrides (v6 link2 0.015, Round 154 ②) are
      exact instead of class-flattened. This is handoff §2 option (b),
      chosen over the class-tier approximation (a) because the env already
      carries the per-row tensor and no obs-layout change is needed.
    - viol_exempt (N, M) is SphereDistOut.viol_exempt: rows under a live
      near_table exemption (low + slow, semantics.judge conditional rows)
      are DROPPED from the cost entirely -- legally resting on the own
      table must not be charged, otherwise the table channel is
      structurally unsatisfiable and lambda_table rides its integral limit
      (the V3-mystery table half, Round 151; same exemption-blind bug C11
      fixed on the eval side). Exemption is per-step: a fast approach
      breaks the (low & slow) condition and the row is charged again.
    """
    d = active_pairs[..., 0]
    cls = active_pairs[..., 2]
    keep = active_mask if viol_exempt is None else active_mask & ~viol_exempt
    band = (d_warn - active_dmin).clamp_min(1e-6)
    x = ((d_warn - d) / band).clamp(min=0.0, max=clip)
    cost = torch.where(keep, x * x, torch.zeros_like(x))
    out = []
    for c in (CLASS_CROSS, CLASS_SELF, CLASS_TABLE):
        sel = torch.where((cls == c) & keep, cost, torch.zeros_like(cost))
        out.append(sel.amax(dim=-1) if sel.shape[-1] else
                   torch.zeros(cost.shape[0], device=cost.device))
    return torch.stack(out, dim=-1)
