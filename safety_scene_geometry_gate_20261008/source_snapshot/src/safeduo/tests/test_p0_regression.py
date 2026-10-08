"""P0 回归钉死（2026-08-12 监管线）：B 真球分解下的幽灵自碰与观测饥饿。

复现事实（C 发现）：ur5e.yaml 的 wrist_2_link 两球设计上互相重叠（管状链式
分解），未豁免时 init 即全 env 永久 VIOLATION、active 集被 66 条幽灵 self
行塞满、cross 行被挤出观测。本文件用 B 的真实 YAML 做静态不变量 + 运行时
断言，防止豁免逻辑回退。
"""

import torch

from safeduo.safety.semantics import ContactSemantics
from safeduo.safety.sphere_distance import ArmSpheres, LinkSpheres, SphereDistanceModule
from safeduo.safety.sphere_specs import bundled_arm_specs
from safeduo.safety.types import ARM_KEYS, CLASS_CROSS

SEM = "assets_src/contact_semantics.yaml"


def bundled_module(**kw):
    m = SphereDistanceModule(bundled_arm_specs("."), semantics=ContactSemantics(SEM), **kw)
    return m


def test_no_same_link_or_adjacent_ordinal_self_pairs():
    # 静态不变量：self 对里不允许出现 同臂内 序数差 < 2 的球对（含同 link）
    m = bundled_module()
    ps = m.pairs_self
    assert len(ps) > 0
    ai, aj = m.arm_id[ps[:, 0]], m.arm_id[ps[:, 1]]
    same_arm = ai == aj
    ord_diff = (m.link_ord[ps[:, 0]] - m.link_ord[ps[:, 1]]).abs()
    assert not (same_arm & (ord_diff < 2)).any(), "幽灵自碰对回来了（P0 回归）"


def test_default_pose_like_fk_no_ghost_violation():
    # 用"各臂拉开 + 每 link 顺链排布"的合成 FK 模拟正常姿态：不允许出现
    # 任何 margin<0 的 self 行（B 球分解的链内重叠必须已被豁免掉）
    m = bundled_module()
    n = 2
    pos, quat, lin, ang = {}, {}, {}, {}
    arm_base = {"F_L": (0.0, 0.0, 3.0), "F_R": (0.0, 4.0, 3.0),
                "U_L": (8.0, 0.0, 3.0), "U_R": (8.0, 4.0, 3.0)}
    for arm in ARM_KEYS:
        nb = len(m.specs[arm].links)
        base = torch.tensor(arm_base[arm])
        # 顺链每 link 沿 +x 挪 0.45m（真实臂段量级：跨序数对不会人为压重叠；
        # 同 link 球的重叠与间距无关——P0 未修时此处必然爆负 margin）
        p = torch.stack([base + torch.tensor([0.45 * i, 0.0, 0.0]) for i in range(nb)])
        pos[arm] = p.unsqueeze(0).repeat(n, 1, 1).float()
        quat[arm] = torch.tensor([1.0, 0, 0, 0]).expand(n, nb, 4).contiguous()
        lin[arm] = torch.zeros(n, nb, 3)
        ang[arm] = torch.zeros(n, nb, 3)
    m.bind_bodies({arm: [ls.link for ls in m.specs[arm].links] for arm in ARM_KEYS})
    m.bind_tables(torch.tensor([[0.65, 0.0, 0.375], [-0.65, 0.0, 0.375]]),
                  torch.tensor([[0.4, 0.6, 0.375], [0.4, 0.6, 0.375]]))
    out = m.compute(pos, quat, lin, ang, torch.zeros(n, 3), need_full=True)
    sl = m._slice_self
    d_self = out.dists[:, sl]
    assert (d_self > 0).all(), f"self 行出现负 margin：min={d_self.min():.4f}（幽灵自碰）"
    assert not out.violation.any()


def test_overlapping_balls_same_link_synthetic():
    # 合成最小复现：单 link 两颗重叠球 -> 不产生 self 对
    sem = ContactSemantics(SEM)
    specs = {k: ArmSpheres(links=[LinkSpheres("wrist_2_link", [(0.0, 0.0, 0.0)], [0.05])])
             for k in ARM_KEYS}
    specs["U_L"] = ArmSpheres(links=[LinkSpheres(
        "wrist_2_link", [(0.0, 0.0, 0.0), (0.0, 0.0, 0.02)], [0.05, 0.05])])
    m = SphereDistanceModule(specs, semantics=sem)
    ps = m.pairs_self
    if len(ps):
        same = m.arm_id[ps[:, 0]] == m.arm_id[ps[:, 1]]
        assert not same.any()


def test_active_quota_keeps_cross_visible():
    # 大量更近的 self 干扰行 + 少量 cross 行：cross 必须占住保底名额
    sem = ContactSemantics(SEM)
    # F_L 用 8 个跨序数 link（序数差>=2 未被邻接表覆盖的组合会保留 self 对）
    links = [LinkSpheres(f"fr3_link{i}", [(0.0, 0.0, 0.0)], [0.03]) for i in range(8)]
    specs = {
        "F_L": ArmSpheres(links=links),
        "F_R": ArmSpheres(links=[LinkSpheres("fr3_link3", [(0.0, 0.0, 0.0)], [0.03])]),
        "U_L": ArmSpheres(links=[LinkSpheres("wrist_3_link", [(0.0, 0.0, 0.0)], [0.03])]),
        "U_R": ArmSpheres(links=[LinkSpheres("wrist_2_link", [(0.0, 0.0, 0.0)], [0.03])]),
    }
    m = SphereDistanceModule(specs, semantics=sem, max_active=8, quota_cross=4,
                             d_soft=100.0)
    m.bind_bodies({arm: [ls.link for ls in specs[arm].links] for arm in ARM_KEYS})
    m.bind_tables(torch.tensor([[50.0, 0.0, 0.375], [-50.0, 0.0, 0.375]]),
                  torch.tensor([[0.4, 0.6, 0.375], [0.4, 0.6, 0.375]]))
    n = 1
    pos, quat, lin, ang = {}, {}, {}, {}
    # F_L 的 8 球彼此挤在一起（self 行全在 ~1-2cm）；U 侧 cross 距离 ~0.5m
    fl = torch.stack([torch.tensor([0.02 * i, 0.0, 5.0]) for i in range(8)])
    pos["F_L"] = fl.unsqueeze(0).float()
    pos["F_R"] = torch.tensor([[0.05, 0.1, 5.0]]).reshape(1, 1, 3)
    pos["U_L"] = torch.tensor([[0.5, 0.0, 5.0]]).reshape(1, 1, 3)
    pos["U_R"] = torch.tensor([[0.5, 0.1, 5.0]]).reshape(1, 1, 3)
    for arm in ARM_KEYS:
        nb = pos[arm].shape[1]
        quat[arm] = torch.tensor([1.0, 0, 0, 0]).expand(n, nb, 4).contiguous()
        lin[arm] = torch.zeros(n, nb, 3)
        ang[arm] = torch.zeros(n, nb, 3)
    out = m.compute(pos, quat, lin, ang, torch.zeros(n, 3))
    cls = out.active_pairs[0, :, 2][out.active_mask[0]]
    n_cross = int((cls == CLASS_CROSS).sum())
    assert n_cross >= 4, f"cross 保底名额失效：只有 {n_cross} 行"
    # 排序仍按 margin 升序
    d = out.active_pairs[0, :, 0][out.active_mask[0]]
    assert (d[1:] >= d[:-1]).all()
