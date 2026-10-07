"""v4 换装单测（A6-W6）：real_v3 球加载 + contact_semantics v3 + robot_keys。

钉死七步换装的三组约定（cpu，无 isaac）：
1. real_v3_arm_specs：fr3_hand 过渡球退役 / fr3_link8 法兰球在位 / JAKA 链序 /
   F2 手球带 "hand/" semantic 前缀且左右手独立；
2. 语义 v3：JAKA 命名分类、邻接豁免走 robot_keys=jaka_zu7、腕簇结构性豁免、
   F2 手腕安装豁免、near_table 条件豁免对 F2 手命名依然生效；
3. v0 文件向后兼容：无 robot_keys 节时缺省 {F: fr3, U: ur5e} 与旧行为一致。
"""

import pytest

from safeduo.configs import repo_root
from safeduo.safety.semantics import ContactSemantics
from safeduo.safety.sphere_specs import real_v3_arm_specs

SEM_V3 = str(repo_root() / "src/safeduo/configs/contact_semantics_v3.yaml")
SEM_V0 = str(repo_root() / "assets_src/contact_semantics.yaml")


@pytest.fixture(scope="module")
def specs():
    return real_v3_arm_specs(repo_root())


@pytest.fixture(scope="module")
def sem():
    return ContactSemantics(SEM_V3)


# ---- 1) real_v3 球加载 ----

def test_fr3_transition_hand_retired_flange_present(specs):
    links = [ls.link for ls in specs["F_L"].links]
    assert "fr3_hand" not in links, "fr3_hand 过渡球必须随组合 USD 退役"
    assert "fr3_link8" in links, "法兰球（ASSEMBLY_V3 3.2 节）必须合并进 F 臂"
    i8 = links.index("fr3_link8")
    ls8 = specs["F_L"].links[i8]
    assert ls8.radii == [0.038] and tuple(ls8.offsets[0]) == (0.0, 0.0, 0.0085)


def test_jaka_chain_and_hand_appended(specs):
    u = specs["U_L"].links
    arm = [ls.link for ls in u if not (ls.semantic_name or "").startswith("hand/")]
    assert arm == [f"link{i}" for i in range(1, 7)], "JAKA 链序 link1..6（base 0 球）"
    hand = [ls for ls in u if (ls.semantic_name or "").startswith("hand/")]
    assert len(hand) >= 6 and sum(len(h.radii) for h in hand) == 10, \
        "F2 手 10 球/手（B 交付预算）"


def test_hand_semantic_prefix_and_sides(specs):
    fl = {ls.semantic_name for ls in specs["F_L"].links if ls.semantic_name}
    fr = {ls.semantic_name for ls in specs["F_R"].links if ls.semantic_name}
    assert any(s.startswith("hand/left_") for s in fl), "F_L 挂左手（layout end_effector）"
    assert any(s.startswith("hand/right_") for s in fr), "F_R 挂右手"
    # 手 link（body 名含 hand_base 或指名）的 semantic 必须带 hand/ 前缀；
    # 臂 link semantic_name == link 名是 load_robot_yaml 的合法回退，不受此约束
    for arm_key in ("F_L", "F_R", "U_L", "U_R"):
        for ls in specs[arm_key].links:
            if any(k in ls.link for k in ("hand_base", "thumb", "index",
                                          "middle", "ring", "pinky")):
                assert (ls.semantic_name or "").startswith("hand/"), ls.link


# ---- 2) 语义 v3 ----

def test_v3_classify_jaka_and_f2(sem):
    assert sem.classify("U_L/link1") == "arm_base"
    assert sem.classify("U_L/base_link") == "arm_base"
    assert sem.classify("U_R/link4") == "arm_link"
    assert sem.classify("F_L/fr3_link8") == "arm_link"
    assert sem.classify("F_R/hand/right_hand_base") == "hand_palm"
    assert sem.classify("U_L/hand/left_index_1") == "hand_finger"


def test_v3_jaka_adjacency_via_robot_keys(sem):
    # 链邻接豁免（走 robot_keys: U -> jaka_zu7；semantics.py 硬编码 ur5e 的话此处必炸）
    v = sem.judge("U_L/link2", "U_L/link3")
    assert not v.keep and v.verdict == "adjacency_exempt"
    # 长肢 diff=2 对保留检查（J3/J4 大限位可折回，B patch 论证）
    v2 = sem.judge("U_L/link1", "U_L/link3")
    assert v2.keep and v2.category == "self_U" and v2.d_min == pytest.approx(0.02)
    assert sem.judge("U_L/link2", "U_L/link4").keep


def test_v3_jaka_wrist_cluster_exempt(sem):
    # A6-W6 修订：腕簇 diff=2 结构性贴邻豁免（UR5e/cuRobo 同构先例；
    # v4 冒烟出生 -2.6mm 事故的修复），腕全折风险由 hand<->arm 对兜底
    assert not sem.judge("U_L/link3", "U_L/link5").keep
    assert not sem.judge("U_R/link4", "U_R/link6").keep
    # 兜底对必须仍然活着
    assert sem.judge("U_L/link3", "U_L/link6").keep          # diff=3 保留
    assert sem.judge("U_R/link4", "U_R/hand/right_index_1").keep


def test_v3_hand_wrist_exempt(sem):
    assert not sem.judge("F_L/fr3_link8", "F_L/hand/left_hand_base").keep
    assert not sem.judge("F_L/fr3_link7", "F_L/hand/left_index_1").keep
    assert not sem.judge("U_R/link6", "U_R/hand/right_hand_base").keep
    assert not sem.judge("U_R/link5", "U_R/hand/right_thumb_2").keep
    # 非安装邻接的手-臂对保留（腕折回可真碰）
    assert sem.judge("U_R/link4", "U_R/hand/right_index_1").keep


def test_v3_cross_and_near_table(sem):
    v = sem.judge("F_L/fr3_link3", "U_R/link4")
    assert v.keep and v.category == "cross" and v.d_min == pytest.approx(0.03)
    # F2 手指对自桌：conditional_exempt（near_table 双条件运行时判）
    v2 = sem.judge("U_L/hand/left_index_1", "table_U")
    assert v2.keep and v2.conditional and v2.verdict == "conditional_exempt"
    # 对面桌不豁免
    v3 = sem.judge("U_L/hand/left_index_1", "table_F")
    assert v3.keep and not v3.conditional and v3.verdict == "forbid"


# ---- 3) v0 向后兼容 ----

def test_v0_default_robot_keys_backcompat():
    s0 = ContactSemantics(SEM_V0)
    assert s0._robot_key == {"F": "fr3", "U": "ur5e"}
    v = s0.judge("U_L/upper_arm_link", "U_L/forearm_link")
    assert not v.keep and v.verdict == "adjacency_exempt"


# ---- 4) 仪器防伪 v4 见证（sphere_audit real_v3 布局）----

def test_witness_real_v3_contains_merged_geometry():
    from safeduo.safety.sphere_audit import witness_load_real_v3
    witness, prov = witness_load_real_v3(repo_root())
    assert "fr3_link8" in witness["F_L"], "法兰球必须进见证表"
    assert prov["fr3_link8_flange"]["radius"] == 0.038
    assert any(k.startswith("hand/left_") for k in witness["F_L"])
    assert any(k.startswith("hand/right_") for k in witness["U_R"])
    assert "link6" in witness["U_L"] and "fr3_hand" not in witness["F_L"]
    assert prov["jaka_zu7"]["sha1"] and prov["f2_left"]["sha1"]


def test_provenance_real_v3_accepts_and_rejects():
    from safeduo.safety.sphere_audit import sphere_provenance

    class MockSph:
        def __init__(self, names, radii):
            self.qualified_names = names
            self.radii = radii
            self.n_spheres = len(radii)
            self.n_pairs = 0

    from safeduo.safety.sphere_specs import real_v3_arm_specs
    specs = real_v3_arm_specs(repo_root())
    names, radii = [], []
    for arm, spec in specs.items():
        for ls in spec.links:
            for r in ls.radii:
                names.append(f"{arm}/{ls.semantic_name or ls.link}")
                radii.append(float(r))
    prov = sphere_provenance(MockSph(names, radii), layout="real_v3")
    assert prov["layout"] == "real_v3" and prov["n_spheres"] == len(radii)
    # 篡改一个半径必须拒测
    bad = list(radii)
    bad[names.index("F_L/fr3_link8")] += 0.01
    import pytest as _pt
    with _pt.raises(RuntimeError, match="PARITY_SPHERE_MISMATCH"):
        sphere_provenance(MockSph(names, bad), layout="real_v3")
