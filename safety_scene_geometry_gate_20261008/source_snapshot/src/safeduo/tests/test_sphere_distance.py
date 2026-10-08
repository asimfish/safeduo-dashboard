"""球距离 v2 单测：几何正确性（legacy 路径）+ 活跃集 + 语义建对 + 条件豁免（cpu）。"""

import math

import pytest
import torch

from safeduo.safety.semantics import ContactSemantics
from safeduo.safety.sphere_distance import (
    ArmSpheres,
    LinkSpheres,
    SphereDistanceModule,
    quat_rotate,
)
from safeduo.safety.types import ARM_KEYS, CLASS_CROSS, CLASS_SELF, CLASS_TABLE

SEM_YAML = "assets_src/contact_semantics.yaml"


def one_sphere_specs(r=0.1):
    # 每臂 1 link 1 球，offset 为 0：球心 = body 位置
    return {k: ArmSpheres(links=[LinkSpheres("b0", [(0.0, 0.0, 0.0)], [r])]) for k in ARM_KEYS}


def make_module(specs=None, max_active=16, d_soft=100.0, tau_ttc=0.5, **kw):
    # d_soft 默认拉满：所有对都活跃，等价旧 top-K 行为，方便手工断言
    m = SphereDistanceModule(specs or one_sphere_specs(), max_active=max_active,
                             d_soft=d_soft, tau_ttc=tau_ttc, **kw)
    m.bind_bodies({k: ["b0"] for k in ARM_KEYS})
    # 两张桌：中心 +/-0.65，半尺寸 (0.4, 0.6, 0.375)（桌面顶 z=0.75）
    m.bind_tables(torch.tensor([[0.65, 0.0, 0.375], [-0.65, 0.0, 0.375]]),
                  torch.tensor([[0.4, 0.6, 0.375], [0.4, 0.6, 0.375]]))
    return m


def body_tensors(pos_of: dict, vel_of: dict | None = None):
    pos, quat, lin, ang = {}, {}, {}, {}
    for k in ARM_KEYS:
        pos[k] = torch.tensor(pos_of[k]).reshape(1, 1, 3).float()
        quat[k] = torch.tensor([1.0, 0.0, 0.0, 0.0]).reshape(1, 1, 4)
        v = (vel_of or {}).get(k, (0.0, 0.0, 0.0))
        lin[k] = torch.tensor(v).reshape(1, 1, 3).float()
        ang[k] = torch.zeros(1, 1, 3)
    return pos, quat, lin, ang


HIGH = 5.0


def test_quat_rotate_90deg_z():
    q = torch.tensor([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
    v = torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(quat_rotate(q, v), torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)


def test_cross_distance_closing_and_decode():
    m = make_module()
    pos = {"F_L": (0.0, 0.0, HIGH), "F_R": (0.0, 3.0, HIGH),
           "U_L": (1.0, 0.0, HIGH), "U_R": (1.0, 3.0, HIGH)}
    vel = {"U_L": (-1.0, 0.0, 0.0)}
    out = m.compute(*body_tensors(pos, vel), env_origins=torch.zeros(1, 3), need_full=True)
    top = out.active_pairs[0]
    assert out.active_mask[0, 0]
    assert abs(top[0, 0].item() - 0.8) < 1e-5          # 球心距 1 - 两个 0.1 半径
    assert top[0, 2].item() == CLASS_CROSS
    assert abs(top[0, 1].item() - 1.0) < 1e-5          # 接近速度 +1
    pid = int(top[0, 3].item())
    assert set(m.pair_table[pid].tolist()) == {0, 2}   # F_L=球0, U_L=球2
    assert abs(out.min_margin["cross"][0].item() - 0.8) < 1e-5


def test_self_split_and_table():
    m = make_module()
    pos = {"F_L": (0.0, 0.0, HIGH), "F_R": (0.5, 0.0, HIGH),
           "U_L": (0.0, 3.0, HIGH), "U_R": (0.7, 3.0, HIGH)}
    out = m.compute(*body_tensors(pos), env_origins=torch.zeros(1, 3))
    assert abs(out.min_margin["self_F"][0].item() - 0.3) < 1e-5
    assert abs(out.min_margin["self_U"][0].item() - 0.5) < 1e-5
    assert out.active_pairs[0, 0, 2].item() == CLASS_SELF
    # 桌：球悬在 F 桌上方 1m（桌顶 0.75，半径 0.1 -> 余量 0.15）
    pos2 = {"F_L": (0.65, 0.0, 1.0), "F_R": (0.65, 3.0, HIGH),
            "U_L": (3.0, 0.0, HIGH), "U_R": (3.0, 3.0, HIGH)}
    vel2 = {"F_L": (0.0, 0.0, -2.0)}
    out2 = m.compute(*body_tensors(pos2, vel2), env_origins=torch.zeros(1, 3))
    assert abs(out2.min_margin["table"][0].item() - 0.15) < 1e-5
    rows = out2.active_pairs[0]
    tab = rows[(rows[:, 2] == CLASS_TABLE) & out2.active_mask[0]]
    assert abs(tab[0, 0].item() - 0.15) < 1e-5
    assert abs(tab[0, 1].item() - 2.0) < 1e-5


def test_active_set_gating_and_ttc():
    # d_soft=5cm：远而慢 = 不活跃；远而快逼近 = TTC 门抓进来
    m = make_module(d_soft=0.05, tau_ttc=0.5)
    pos = {"F_L": (0.0, 0.0, HIGH), "F_R": (0.0, 3.0, HIGH),
           "U_L": (1.0, 0.0, HIGH), "U_R": (1.0, 3.0, HIGH)}
    out_slow = m.compute(*body_tensors(pos), env_origins=torch.zeros(1, 3))
    assert not out_slow.active_mask[0].any()           # 全远全慢 -> 空活跃集
    assert (out_slow.active_pairs[0, :, 3] == -1.0).all()  # padding pair_id=-1
    vel = {"U_L": (-2.0, 0.0, 0.0)}                    # 0.8m / 2m/s = TTC 0.4 < 0.5
    out_fast = m.compute(*body_tensors(pos, vel), env_origins=torch.zeros(1, 3))
    assert out_fast.active_mask[0, 0]
    assert abs(out_fast.active_pairs[0, 0, 0].item() - 0.8) < 1e-5


def test_env_origin_invariance_and_sorted():
    m = make_module(max_active=4)
    pos = {"F_L": (0.0, 0.0, HIGH), "F_R": (0.4, 0.0, HIGH),
           "U_L": (1.0, 0.0, HIGH), "U_R": (2.0, 0.0, HIGH)}
    p, q, l, a = body_tensors(pos)
    p2 = {k: torch.cat([v, v + torch.tensor([10.0, 0.0, 0.0])]) for k, v in p.items()}
    q2 = {k: v.repeat(2, 1, 1) for k, v in q.items()}
    l2 = {k: v.repeat(2, 1, 1) for k, v in l.items()}
    a2 = {k: v.repeat(2, 1, 1) for k, v in a.items()}
    origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    out = m.compute(p2, q2, l2, a2, env_origins=origins)
    d = out.active_pairs[..., 0]
    mask = out.active_mask
    assert (d[:, 1:][mask[:, 1:]] >= 1e-9).all() or True  # 排序断言见下
    for i in range(d.shape[1] - 1):
        both = mask[:, i] & mask[:, i + 1]
        assert (d[both, i + 1] >= d[both, i]).all()
    assert torch.allclose(out.active_pairs[0], out.active_pairs[1], atol=1e-5)


def test_semantics_pair_building_fr3():
    # 用 B 的正式语义 + fr3 命名：邻接豁免生效、同臂跳2生成、基座只对自桌豁免
    sem = ContactSemantics(SEM_YAML)
    specs = {}
    fr3_links = [LinkSpheres(f"fr3_link{i}", [(0.0, 0.0, 0.0)], [0.05]) for i in (3, 4, 5)]
    ur_links = [LinkSpheres(n, [(0.0, 0.0, 0.0)], [0.05])
                for n in ("forearm_link", "wrist_1_link")]
    specs["F_L"] = ArmSpheres(links=fr3_links)
    specs["F_R"] = ArmSpheres(links=[LinkSpheres("fr3_link3", [(0.0, 0.0, 0.0)], [0.05])])
    specs["U_L"] = ArmSpheres(links=ur_links)
    specs["U_R"] = ArmSpheres(links=[LinkSpheres("wrist_2_link", [(0.0, 0.0, 0.0)], [0.05])])
    m = SphereDistanceModule(specs, semantics=sem)
    # F_L 内部：3-4 与 4-5 邻接豁免、3-5 非豁免（B 的 fr3 邻接表没有 3-5）
    qn = m.qualified_names
    def pair_exists(i_name, j_name):
        ii = qn.index(i_name); jj = qn.index(j_name)
        pt = m.pair_table[: len(m.pairs_cross) + len(m.pairs_self)]
        return any({int(a), int(b)} == {ii, jj} for a, b in pt)
    assert not pair_exists("F_L/fr3_link3", "F_L/fr3_link4")
    assert not pair_exists("F_L/fr3_link4", "F_L/fr3_link5")
    assert pair_exists("F_L/fr3_link3", "F_L/fr3_link5")
    # UR 同臂 forearm-wrist_1 邻接豁免
    assert not pair_exists("U_L/forearm_link", "U_L/wrist_1_link")
    # 跨机对全生成：F_L 3 球 x U 侧 3 球 + F_R 1 球 x 3 = 12
    assert len(m.pairs_cross) == 12
    # d_min 分档
    assert (m.pair_dmin[: len(m.pairs_cross)] == 0.03).all()


def test_conditional_exempt_runtime():
    # 手部球贴自桌低速 -> d_min 降 0.005 且 VIOLATION 豁免；高速拍桌不豁免
    sem = ContactSemantics(SEM_YAML)
    specs = one_sphere_specs(r=0.05)
    specs["F_L"] = ArmSpheres(links=[LinkSpheres(
        "b0", [(0.0, 0.0, 0.0)], [0.05], semantic_name="hand/index_proximal")])
    m = SphereDistanceModule(specs, semantics=sem, d_soft=100.0)
    m.bind_bodies({k: ["b0"] for k in ARM_KEYS})
    m.bind_tables(torch.tensor([[0.65, 0.0, 0.375], [-0.65, 0.0, 0.375]]),
                  torch.tensor([[0.4, 0.6, 0.375], [0.4, 0.6, 0.375]]))
    # 手球在 F 桌上方 4cm（球底距桌顶 -0.01 = 轻微接触），下压 2cm/s（低速）
    pos = {"F_L": (0.65, 0.0, 0.79), "F_R": (0.65, 1.0, HIGH),
           "U_L": (-0.65, 0.0, HIGH), "U_R": (-0.65, 1.0, HIGH)}
    vel = {"F_L": (0.0, 0.0, -0.02)}
    out = m.compute(*body_tensors(pos, vel), env_origins=torch.zeros(1, 3))
    rows = out.active_pairs[0]
    tab_rows = (rows[:, 2] == CLASS_TABLE) & out.active_mask[0]
    hand_row = tab_rows.nonzero()[0, 0]
    assert abs(out.active_dmin[0, hand_row].item() - 0.005) < 1e-9  # 豁免 d_min
    assert out.viol_exempt[0, hand_row]
    assert not out.violation[0]                        # 接触但豁免 -> 不算 VIOLATION
    # 高速下拍：不豁免，margin<0 -> VIOLATION
    vel_fast = {"F_L": (0.0, 0.0, -0.5)}
    out2 = m.compute(*body_tensors(pos, vel_fast), env_origins=torch.zeros(1, 3))
    assert not out2.viol_exempt[0].any()
    assert out2.violation[0]


def test_legacy_table_check_flag():
    specs = one_sphere_specs()
    specs["F_L"] = ArmSpheres(links=[LinkSpheres("b0", [(0.0, 0.0, 0.0)], [0.1],
                                                 table_check=False)])
    m = make_module(specs)
    assert len(m.pairs_table) == 3 * 2  # 4 球少 1，各对 2 桌


def test_compute_caches_table_slice_for_exempt_recut():
    # C11（20260815 豁免盲口径修复）：compute 每步缓存 table 槽位裸边距 +
    # VIOLATION 豁免掩码（endurance_eval 通道级重切消费）；官方输出不变，
    # 且 (margin<0 & ~exempt).any == 官方 violation（cross/self 干净场景）
    sem = ContactSemantics(SEM_YAML)
    specs = one_sphere_specs(r=0.05)
    specs["F_L"] = ArmSpheres(links=[LinkSpheres(
        "b0", [(0.0, 0.0, 0.0)], [0.05], semantic_name="hand/index_proximal")])
    m = SphereDistanceModule(specs, semantics=sem, d_soft=100.0)
    m.bind_bodies({k: ["b0"] for k in ARM_KEYS})
    m.bind_tables(torch.tensor([[0.65, 0.0, 0.375], [-0.65, 0.0, 0.375]]),
                  torch.tensor([[0.4, 0.6, 0.375], [0.4, 0.6, 0.375]]))
    pos = {"F_L": (0.65, 0.0, 0.79), "F_R": (0.65, 1.0, HIGH),
           "U_L": (-0.65, 0.0, HIGH), "U_R": (-0.65, 1.0, HIGH)}
    # 低速贴自桌：接触为豁免行 -> 通道级 flag 干净，与官方 violation 一致
    out = m.compute(*body_tensors(pos, {"F_L": (0.0, 0.0, -0.02)}),
                    env_origins=torch.zeros(1, 3))
    n_tab = len(m.pairs_table)
    assert m.last_table_margin.shape == (1, n_tab)
    assert m.last_table_viol_exempt.shape == (1, n_tab)
    neg = m.last_table_margin < 0
    assert bool((neg & m.last_table_viol_exempt).any())      # 豁免接触在场
    tv = (neg & ~m.last_table_viol_exempt).any(dim=1)
    assert not bool(tv[0]) and not bool(out.violation[0])
    # 高速下拍：掩码全 False，通道级 flag 与官方 violation 同时点亮
    out2 = m.compute(*body_tensors(pos, {"F_L": (0.0, 0.0, -0.5)}),
                     env_origins=torch.zeros(1, 3))
    neg2 = m.last_table_margin < 0
    assert not bool(m.last_table_viol_exempt.any())
    tv2 = (neg2 & ~m.last_table_viol_exempt).any(dim=1)
    assert bool(tv2[0]) and bool(out2.violation[0])
    # legacy（无 semantics）路径：掩码存在且恒 False
    ml = make_module()
    ml.compute(*body_tensors({k: (i * 2.0, 0.0, HIGH)
                              for i, k in enumerate(ARM_KEYS)}),
               env_origins=torch.zeros(1, 3))
    assert ml.last_table_viol_exempt.shape == (1, len(ml.pairs_table))
    assert not bool(ml.last_table_viol_exempt.any())
