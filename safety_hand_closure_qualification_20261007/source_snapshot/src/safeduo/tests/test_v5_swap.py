"""v5 换装单测（A9-W9）：real_v5 球加载（env 路径）+ d_min 重排 + 配方红线。

与 C6 的 tests/test_v5_geometry.py（provider 路径）互补，四组约定（cpu，无 isaac）：
1. sphere_specs.real_v5_arm_specs：jaka 18 球 / F2 v5 手 11 球、法兰球零改动、
   球数真值 F 31 / U 29（manifest anchors 的 27/3132 系旧账面，已 @B2 更正）；
2. A/C 双实现互拍：env 加载路径与 baselines.real_geometry.real_v5_arm_specs
   逐球位级一致（球对数一致性门的本地半边）；
3. SphereDistanceModule + contact_semantics_v3 + d_min_override（duo_env 注入
   同路径）：cross 对 3596、腕环对保留、d_min 烘焙 self/cross 0.013 / table
   0.020、near_table 条件豁免不受 override 影响；
4. duo_env_v5.yaml 配方红线：除场景层键外与 duo_env_v4.yaml 逐字节一致
   （对照变量只有场景层——C5 配方冻结）。
"""

import pytest
import torch
import yaml

from safeduo.configs import load_config, repo_root
from safeduo.safety.semantics import ContactSemantics
from safeduo.safety.sphere_distance import SphereDistanceModule
from safeduo.safety.sphere_specs import real_v3_arm_specs, real_v5_arm_specs

SEM_V3 = str(repo_root() / "src/safeduo/configs/contact_semantics_v3.yaml")


@pytest.fixture(scope="module")
def specs():
    return real_v5_arm_specs(repo_root())


def _manifest():
    return yaml.safe_load((repo_root() / "assets_src/real/v5_bundle_manifest.yaml")
                          .read_text())


def _sem_with_v5_override():
    """duo_env.__init__ 的 d_min_override 注入同款（yaml -> sem.d_min）。"""
    sem = ContactSemantics(SEM_V3)
    ov = load_config("duo_env_v5.yaml")["safety"]["d_min_override"]
    for k, v in ov.items():
        sem.d_min[k] = float(v)
    return sem


# ---- 1) real_v5 球加载（env 路径）----

def test_v5_counts_and_split(specs):
    per_arm = {a: sum(len(ls.radii) for ls in specs[a].links) for a in specs}
    assert per_arm == {"F_L": 31, "F_R": 31, "U_L": 29, "U_R": 29}
    u_arm = sum(len(ls.radii) for ls in specs["U_L"].links
                if not (ls.semantic_name or "").startswith("hand/"))
    u_hand = sum(len(ls.radii) for ls in specs["U_L"].links
                 if (ls.semantic_name or "").startswith("hand/"))
    assert (u_arm, u_hand) == (18, 11), "jaka v5 18 球 + F2 v5 手 11 球"


def test_v5_fr3_and_flange_untouched(specs):
    v3 = real_v3_arm_specs(repo_root())
    pick = lambda spec: [(ls.link, tuple(map(tuple, ls.offsets)), tuple(ls.radii))
                         for ls in spec.links
                         if not (ls.semantic_name or "").startswith("hand/")]
    assert pick(specs["F_L"]) == pick(v3["F_L"]), "F 臂+法兰球必须与 v4 逐位一致"


def test_v5_jaka_link4_wrist_resegmented(specs):
    u = {ls.link: ls for ls in specs["U_L"].links
         if not (ls.semantic_name or "").startswith("hand/")}
    assert len(u["link4"].radii) == 4, "腕环收径：link4 2->4 段"
    assert max(u["link4"].radii) < 0.07, "link4 收径后应显著小于 v4 的 0.0709"
    assert len(u["link6"].radii) == 1


def test_v5_hand_base_five_spheres(specs):
    for arm, side in (("F_L", "left"), ("F_R", "right"),
                      ("U_L", "left"), ("U_R", "right")):
        hb = [ls for ls in specs[arm].links
              if ls.link == f"{side}_hand_base"]
        assert len(hb) == 1 and len(hb[0].radii) == 5, (arm, side)
        assert hb[0].semantic_name == f"hand/{side}_hand_base"


# ---- 2) A/C 双实现互拍 ----

def test_v5_env_loader_matches_provider_loader(specs):
    from safeduo.baselines.real_geometry import real_v5_arm_specs as c_loader
    c_specs = c_loader(repo_root())
    for arm in specs:
        a = [(ls.link, ls.semantic_name, tuple(map(tuple, ls.offsets)),
              tuple(ls.radii)) for ls in specs[arm].links]
        c = [(ls.link, ls.semantic_name, tuple(map(tuple, ls.offsets)),
              tuple(ls.radii)) for ls in c_specs[arm].links]
        assert a == c, f"{arm}: env 加载与 provider 加载必须位级一致"


# ---- 3) 配对表 + d_min 烘焙（isaac 侧真值的本地镜像）----

@pytest.fixture(scope="module")
def sph_v5(specs):
    return SphereDistanceModule(specs, semantics=_sem_with_v5_override(),
                                max_active=32, d_soft=0.05, tau_ttc=0.5,
                                quota_cross=8, device="cpu")


def test_v5_pair_counts(sph_v5):
    assert int(sph_v5.pairs_cross.shape[0]) == 31 * 29 * 4 == 3596
    from safeduo.baselines.real_geometry import make_v5_provider
    p = make_v5_provider(1)
    n_cross = int((p.pair_class == 0).sum())
    n_self = int((p.pair_class == 1).sum())
    n_table = int((p.pair_class == 2).sum())
    assert int(sph_v5.pairs_cross.shape[0]) == n_cross
    assert int(sph_v5.pairs_self.shape[0]) == n_self
    assert int(sph_v5.pairs_table.shape[0]) == n_table
    assert int(sph_v5.n_pairs) == int(p.n_pairs), \
        "球对数一致性门（本地半边）：module 与 provider 配对表同构"


def test_v5_wrist_collar_pair_alive(sph_v5):
    qn = sph_v5.qualified_names
    found = False
    for k in range(int(sph_v5._slice_self.start), int(sph_v5._slice_self.stop)):
        i, j = (int(v) for v in sph_v5.pair_table[k])
        names = {qn[i].split("/", 1)[1], qn[j].split("/", 1)[1]}
        if qn[i].startswith("U") and "link4" in names \
                and any("hand_base" in n for n in names):
            found = True
            assert float(sph_v5.pair_dmin[k]) == pytest.approx(0.013), \
                "腕环对烘焙新 d_min（override 生效）"
    assert found, "腕环对（U link4|hand_base）必须保留检查（B2 checklist 第 6 步）"


def test_v5_dmin_bake_by_class(sph_v5):
    pc, ps = int(sph_v5._slice_cross.stop), int(sph_v5._slice_self.stop)
    assert torch.allclose(sph_v5.pair_dmin[:pc],
                          torch.full((pc,), 0.013)), "cross 全行 0.013"
    assert torch.allclose(sph_v5.pair_dmin[pc:ps],
                          torch.full((ps - pc,), 0.013)), "self 全行 0.013"
    tab = sph_v5.pair_dmin[ps:]
    assert torch.allclose(tab, torch.full_like(tab, 0.020)), "table 不动 0.020"


def test_v5_near_table_exemption_unaffected():
    sem = _sem_with_v5_override()
    assert sem.near_table.d_min_override == pytest.approx(0.005)
    assert sem.d_warn == pytest.approx(0.05) and sem.d_soft == pytest.approx(0.05), \
        "d_warn 维持 0.050（训练器 cost 通道硬编码 0.05，两侧一致；重排权在 C）"


# ---- 4) duo_env_v5.yaml 配方红线 ----

def test_v5_yaml_recipe_frozen_vs_v4():
    y4 = load_config("duo_env_v4.yaml")
    y5 = load_config("duo_env_v5.yaml")
    for key in ("action", "coordinator", "events", "benchmark", "init_qpos"):
        assert y5[key] == y4[key], f"{key} 节必须与 v4 逐位一致（C5 配方冻结）"
    # sim 节：除 physx 缓冲直通（凸分解 patch 需求，纯数值基建非物理行为）外一致
    sim5 = {k: v for k, v in y5["sim"].items() if k != "physx"}
    assert sim5 == y4["sim"], "sim 节除 physx 外与 v4 一致"
    assert y5["sim"]["physx"] == {"gpu_max_rigid_patch_count": 524288}
    s4 = dict(y4["safety"])
    s5 = {k: v for k, v in y5["safety"].items() if k != "d_min_override"}
    assert s5 == s4, "safety 节除 d_min_override 外与 v4 一致"
    assert y5["safety"]["d_min_override"] == {"self": 0.013, "cross": 0.013}


def test_v5_yaml_scene_layer():
    y5 = load_config("duo_env_v5.yaml")
    assert y5["assets"]["spheres"] == "real_v5"
    assert y5["scene"]["layout_yaml"].endswith("scene_layout_v5.yaml")
    for arm, usd in y5["assets"]["arm_usd_overrides"].items():
        assert usd.endswith("_v5.usd"), (arm, usd)
    lay = yaml.safe_load((repo_root() / y5["scene"]["layout_yaml"]).read_text())
    assert float(lay["tables"]["params"]["table_gap"]) == 0.0
    assert lay["tables"]["params"]["table_size"][1] == 1.50
    assert y5["assets"]["semantics_yaml"].endswith("contact_semantics_v3.yaml")


def test_v5_manifest_thresholds_match_override():
    th = _manifest()["thresholds_v5"]
    ov = load_config("duo_env_v5.yaml")["safety"]["d_min_override"]
    assert ov["self"] == pytest.approx(float(th["d_min_self"]))
    assert ov["cross"] == pytest.approx(float(th["d_min_cross"]))


# ---- 5) 仪器防伪 v5 见证 ----

def test_witness_real_v5_and_reject_tamper(specs):
    from safeduo.safety.sphere_audit import sphere_provenance, witness_load_real_v5

    witness, prov = witness_load_real_v5(repo_root())
    assert prov["layout"] == "real_v5"
    assert prov["jaka_zu7"]["path"].endswith("jaka_zu7_v5.yaml")
    assert prov["jaka_zu7"]["spec_version"] == 5
    assert prov["f2_left"]["spec_version"] == 5
    assert len(witness["U_L"]["hand/left_hand_base"]) == 5

    class MockSph:
        def __init__(self, names, radii):
            self.qualified_names = names
            self.radii = radii
            self.n_spheres = len(radii)
            self.n_pairs = 0

    names, radii = [], []
    for arm, spec in specs.items():
        for ls in spec.links:
            for r in ls.radii:
                names.append(f"{arm}/{ls.semantic_name or ls.link}")
                radii.append(float(r))
    prov2 = sphere_provenance(MockSph(names, radii), layout="real_v5")
    assert prov2["n_spheres"] == 120
    bad = list(radii)
    bad[names.index("U_L/link4")] += 0.01
    with pytest.raises(RuntimeError, match="PARITY_SPHERE_MISMATCH"):
        sphere_provenance(MockSph(names, bad), layout="real_v5")
