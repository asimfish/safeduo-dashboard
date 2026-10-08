"""球分解 spec：W1 占位球 + cuRobo YAML 加载接口（正式分解由 Agent B 交付）。

占位球只保证"每根臂身上有一串合理尺寸的球"，让距离/吞吐管线先跑通；
offsets 是按公开 DH/URDF 尺寸目测的粗略值，物理保真度以 B 的 cuRobo 分解为准。
链序 = links 列表顺序，同臂相邻豁免依赖它，别乱排。
"""

from __future__ import annotations

import yaml

from safeduo.safety.sphere_distance import ArmSpheres, LinkSpheres
from safeduo.safety.types import ARM_KEYS

# ---- 占位分解（key = 机器人变体名，config 里选）----

_PANDA = ArmSpheres(links=[
    LinkSpheres("panda_link1", [(0.0, 0.0, -0.08)], [0.08], table_check=False),
    LinkSpheres("panda_link2", [(0.0, -0.08, 0.0)], [0.08], table_check=False),
    LinkSpheres("panda_link3", [(0.0, 0.0, -0.07), (0.0, 0.0, 0.06)], [0.07, 0.07]),
    LinkSpheres("panda_link4", [(0.0, 0.0, 0.0), (-0.08, 0.08, 0.0)], [0.07, 0.06]),
    LinkSpheres("panda_link5", [(0.0, 0.0, -0.10), (0.0, 0.04, -0.22)], [0.07, 0.06]),
    LinkSpheres("panda_link6", [(0.0, 0.0, 0.0)], [0.06]),
    LinkSpheres("panda_link7", [(0.0, 0.0, 0.08)], [0.06]),
    LinkSpheres("panda_hand", [(0.0, 0.0, 0.04), (0.0, 0.0, 0.10)], [0.05, 0.05]),
])

# FR3 link 命名与 Panda 同构（fr3_link1..7 + fr3_hand），尺寸近似
_FR3 = ArmSpheres(links=[
    LinkSpheres(ls.link.replace("panda", "fr3"), ls.offsets, ls.radii, ls.table_check)
    for ls in _PANDA.links
])

_UR10 = ArmSpheres(links=[
    LinkSpheres("shoulder_link", [(0.0, 0.0, 0.0)], [0.10], table_check=False),
    LinkSpheres("upper_arm_link", [(0.0, 0.0, 0.18), (0.0, 0.0, 0.37), (0.0, 0.0, 0.55)], [0.08, 0.08, 0.08]),
    LinkSpheres("forearm_link", [(0.0, 0.0, 0.15), (0.0, 0.0, 0.32), (0.0, 0.0, 0.50)], [0.06, 0.06, 0.06]),
    LinkSpheres("wrist_1_link", [(0.0, 0.0, 0.0)], [0.05]),
    LinkSpheres("wrist_2_link", [(0.0, 0.0, 0.0)], [0.05]),
    LinkSpheres("wrist_3_link", [(0.0, 0.0, 0.02)], [0.05]),
])

# UR5e 同构、臂段更短
_UR5E = ArmSpheres(links=[
    LinkSpheres("shoulder_link", [(0.0, 0.0, 0.0)], [0.08], table_check=False),
    LinkSpheres("upper_arm_link", [(0.0, 0.0, 0.12), (0.0, 0.0, 0.26), (0.0, 0.0, 0.40)], [0.06, 0.06, 0.06]),
    LinkSpheres("forearm_link", [(0.0, 0.0, 0.10), (0.0, 0.0, 0.24), (0.0, 0.0, 0.37)], [0.05, 0.05, 0.05]),
    LinkSpheres("wrist_1_link", [(0.0, 0.0, 0.0)], [0.045]),
    LinkSpheres("wrist_2_link", [(0.0, 0.0, 0.0)], [0.045]),
    LinkSpheres("wrist_3_link", [(0.0, 0.0, 0.02)], [0.045]),
])

_VARIANTS = {"panda": _PANDA, "fr3": _FR3, "ur10": _UR10, "ur5e": _UR5E}


def placeholder_specs(variant_f: str = "panda", variant_u: str = "ur10") -> dict:
    """按 ARM_KEYS 组装四臂 spec：F 双臂用 variant_f，U 双臂用 variant_u。"""
    per_arm = {"F_L": variant_f, "F_R": variant_f, "U_L": variant_u, "U_R": variant_u}
    return {arm: _VARIANTS[per_arm[arm]] for arm in ARM_KEYS}


def from_curobo_yaml(path: str, link_order: list, table_skip: tuple = ()) -> ArmSpheres:
    """读 cuRobo 球分解 YAML -> ArmSpheres（reference/ 原始文件对拍用）。

    支持两种布局：顶层 collision_spheres: {link: [{center, radius}]}，
    或 robot_cfg.kinematics.collision_spheres。link_order 给运动链顺序。
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    spheres = data.get("collision_spheres")
    if spheres is None:
        spheres = data["robot_cfg"]["kinematics"]["collision_spheres"]
    links = []
    for name in link_order:
        if name not in spheres:
            continue
        entries = spheres[name]
        links.append(LinkSpheres(
            link=name,
            offsets=[tuple(e["center"]) for e in entries],
            radii=[float(e["radius"]) for e in entries],
            table_check=name not in table_skip,
        ))
    return ArmSpheres(links=links)


def load_robot_yaml(path: str, robot_key: str, link_order: list | None = None,
                    name_map: dict | None = None) -> ArmSpheres:
    """读 B 的正式球分解（assets_src/spheres/*.yaml，schema {robot:{link:[{center,radius}]}}）。

    link_order 给运动链顺序（scene_layout.yaml naming 节）；缺省用 YAML 键序。
    name_map：语义名 -> articulation body 名（官方 USD prim 命名不一致时用，
    LinkSpheres.link 存 body 名、semantic_name 存语义名，contact_semantics 正则匹配后者）。
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    # B 的交付件顶层是 cuRobo 兼容的 collision_spheres（见 YAML 头注释）；
    # 兼容旧 {robot: {...}} schema 以防手工文件。
    spheres = data.get(robot_key) or data["collision_spheres"]
    order = link_order or list(spheres.keys())
    links = []
    for name in order:
        if name not in spheres:
            continue
        entries = spheres[name]
        links.append(LinkSpheres(
            link=(name_map or {}).get(name, name),
            offsets=[tuple(e["center"]) for e in entries],
            radii=[float(e["radius"]) for e in entries],
            semantic_name=name,
        ))
    return ArmSpheres(links=links)


def bundled_arm_specs(repo_root, fr3_map: dict | None = None,
                      ur5e_map: dict | None = None) -> dict:
    """四臂正式 spec：F 双臂 = fr3.yaml，U 双臂 = ur5e.yaml（B 的交付件）。"""
    from pathlib import Path

    d = Path(repo_root) / "assets_src" / "spheres"
    fr3 = load_robot_yaml(str(d / "fr3.yaml"), "fr3", name_map=fr3_map)
    ur5e = load_robot_yaml(str(d / "ur5e.yaml"), "ur5e", name_map=ur5e_map)
    return {"F_L": fr3, "F_R": fr3, "U_L": ur5e, "U_R": ur5e}


# ASSEMBLY_V3 §3.2：FR3 法兰垫块球（Ø63x17 垫块外接 0.0326 + 5mm 膨胀）；
# 现役 spheres/fr3.yaml 不动，A 侧组合切换时在代码里合并（B 的指定做法）。
_FR3_FLANGE_SPHERE = ("fr3_link8", [(0.0, 0.0, 0.0085)], [0.038])


def real_v3_arm_specs(repo_root) -> dict:
    """v4 真机资产四臂 spec（scene v3：FR3+法兰+F2 / JAKA Zu7+法兰+F2）。

    - F 臂 = spheres/fr3.yaml 去 fr3_hand 过渡球（组合 USD 已删自带夹爪）
      + fr3_link8 法兰球 + real/inspire_rh56f2_{side}.yaml 手球；
    - U 臂 = real/jaka_zu7.yaml（法兰球已并入 link6）+ F2 手球；
    - 手球 semantic_name 前缀 "hand/"（contact_semantics v3 正则按
      "{arm}/hand/(right|left)_..." 匹配）；link 名 = 组合 USD body 名。
    - 左右手独立文件非镜像（B 交付）：F_L/U_L=left，F_R/U_R=right
      （scene_layout_v3 end_effector 节）。
    """
    return _real_arm_specs(repo_root, jaka_yaml="jaka_zu7.yaml",
                           hand_yaml_fmt="inspire_rh56f2_{side}.yaml")


def real_v5_arm_specs(repo_root) -> dict:
    """v5 场景包四臂 spec（B2 v5 bundle，assets_src/real/v5_bundle_manifest.yaml）。

    与 real_v3 结构完全同构，仅三件球 YAML 换 v5 修订版：
    - jaka_zu7_v5.yaml（16->18 球：link4 2->4 段腕环收径；分级去膨胀
      link1-3=+2mm / link4-6=+1mm；法兰虚拟盘方向修正 +z）；
    - inspire_rh56f2_{left,right}_v5.yaml（10->11 球：掌链 4->5；STL 顶点级
      参照；膨胀 5->1mm）——四只手全换（F/U 臂末端都是 F2）；
    - spheres/fr3.yaml 与 fr3_link8 法兰球零改动（B2 裁定 F 侧无归因病灶）。
    球数账（代码级真值，A9-W9 复核）：F 臂 = fr3 19 + 法兰 1 + 手 11 = 31；
    U 臂 = jaka 18 + 手 11 = 29。注意 bundle manifest 的 anchors_v5
    sphere_counts（27/29、cross 3132）沿用了 ASSEMBLY_V3 §4 的旧账面
    （fr3 臂记 16 球）且漏计 F 手 +1，实际 cross 对 = 31*29*4 = 3596
    （v4 真值 30*26*4 = 3120，C5-W7 grid 实测同数）。
    """
    return _real_arm_specs(repo_root, jaka_yaml="jaka_zu7_v5.yaml",
                           hand_yaml_fmt="inspire_rh56f2_{side}_v5.yaml")


def real_v7_arm_specs(repo_root) -> dict:
    """v7 场景包（R14 UR5+RH56DFX 迁移，owner 2026-08-18 资产）。

    - F 双臂 = v5 原样（fr3.yaml 去 fr3_hand + 法兰球 + F2 v5 手球）。真机 F 侧
      手型待 owner 确认（照片示裸法兰；换 DFX 归 R14 第三阶段）。
    - U 双臂 = real/ur5_v7.yaml（spheres/ur5e.yaml 去 wrist_3_link——UR5/UR5e
      连杆几何差 mm 级，首过复用，待真网格复核）+ real/rh56dfx_{side}_v7.yaml
      （generate_dfx_spheres.py 产自 owner 组合 URDF；wrist_3_link 键 = merge-
      joints 后手基座/法兰/掌并入腕 body，球归属单一化）。
    - U 球数 18+22=40（JAKA v5 为 29）：cross 对数 42*40*4=6720（v5 3596），
      GPU 距离场开销 +87%，4096 env 实测预算内（A9-W9 缓冲已抬）。
    """
    from pathlib import Path

    d = Path(repo_root) / "assets_src"
    fr3_full = load_robot_yaml(str(d / "spheres" / "fr3.yaml"), "fr3")
    arm_links = [ls for ls in fr3_full.links if ls.link != "fr3_hand"]
    arm_links.append(LinkSpheres(*_FR3_FLANGE_SPHERE))
    ur5 = load_robot_yaml(str(d / "real" / "ur5_v7.yaml"), "ur5_dfx")

    def hand_links(yaml_fmt: str, key: str, side: str) -> list:
        hand = load_robot_yaml(str(d / "real" / yaml_fmt.format(side=side)), key)
        return [LinkSpheres(ls.link, ls.offsets, ls.radii,
                            semantic_name=f"hand/{ls.link}")
                for ls in hand.links]

    f2 = lambda side: hand_links("inspire_rh56f2_{side}_v5.yaml", "rh56f2", side)
    dfx = lambda side: hand_links("rh56dfx_{side}_v7.yaml", "rh56dfx", side)
    return {
        "F_L": ArmSpheres(links=list(arm_links) + f2("left")),
        "F_R": ArmSpheres(links=list(arm_links) + f2("right")),
        "U_L": ArmSpheres(links=list(ur5.links) + dfx("left")),
        "U_R": ArmSpheres(links=list(ur5.links) + dfx("right")),
    }


def real_v7r16_arm_specs(repo_root) -> dict:
    """r16 几何壳修正版 real_v7（R16/S12 备料，S14 接线 2026-08-21）。

    与 real_v7 的唯一差异 = F 双臂球包换 assets_src/real/spheres_r16/：
    - fr3_r16.yaml：fr3_link3 顶点拟合 2球(r80/77)->3球(r72.8/59.1/57.2)、
      fr3_link6 单球(r75，不覆盖自身 mesh 缺口 +60.7mm)->4球(r55-65) 全覆盖，
      其余 link 与 spheres/fr3.yaml 逐字相同；
    - inspire_rh56f2_{side}_r16.yaml：手指 1球/指(r~45)->3球/指(r15-18，
      =S7 v7fix 数据)。
    U 双臂与法兰球与 real_v7 逐位相同。F 臂球数 31->43/侧。
    配套语义 contact_semantics_v7r16.yaml（hand_wrist.fr3 增补 fr3_link6）；
    构造与 S12 离线验证器 make_r16_provider 等价（tools/s12_r16_offline_
    validation.py），三门验证 PASS 见 artifacts/report_20260821/S12_r16_geometry.md。
    """
    from pathlib import Path

    specs = real_v7_arm_specs(repo_root)      # U 侧原样复用
    d = Path(repo_root) / "assets_src" / "real" / "spheres_r16"
    fr3_full = load_robot_yaml(str(d / "fr3_r16.yaml"), "fr3")
    arm_links = [ls for ls in fr3_full.links if ls.link != "fr3_hand"]
    arm_links.append(LinkSpheres(*_FR3_FLANGE_SPHERE))

    def hand(side: str) -> list:
        h = load_robot_yaml(str(d / f"inspire_rh56f2_{side}_r16.yaml"), "rh56f2")
        return [LinkSpheres(ls.link, ls.offsets, ls.radii,
                            semantic_name=f"hand/{ls.link}")
                for ls in h.links]

    specs["F_L"] = ArmSpheres(links=list(arm_links) + hand("left"))
    specs["F_R"] = ArmSpheres(links=list(arm_links) + hand("right"))
    return specs


def _real_arm_specs(repo_root, jaka_yaml: str, hand_yaml_fmt: str) -> dict:
    from pathlib import Path

    d = Path(repo_root) / "assets_src"
    fr3_full = load_robot_yaml(str(d / "spheres" / "fr3.yaml"), "fr3")
    arm_links = [ls for ls in fr3_full.links if ls.link != "fr3_hand"]
    arm_links.append(LinkSpheres(*_FR3_FLANGE_SPHERE))
    jaka = load_robot_yaml(str(d / "real" / jaka_yaml), "jaka_zu7")

    def hand_links(side: str) -> list:
        hand = load_robot_yaml(
            str(d / "real" / hand_yaml_fmt.format(side=side)), "rh56f2")
        return [LinkSpheres(ls.link, ls.offsets, ls.radii,
                            semantic_name=f"hand/{ls.link}")
                for ls in hand.links]

    return {
        "F_L": ArmSpheres(links=list(arm_links) + hand_links("left")),
        "F_R": ArmSpheres(links=list(arm_links) + hand_links("right")),
        "U_L": ArmSpheres(links=list(jaka.links) + hand_links("left")),
        "U_R": ArmSpheres(links=list(jaka.links) + hand_links("right")),
    }
