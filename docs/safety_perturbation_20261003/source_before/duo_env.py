"""SafeDuo 4 臂面对面场景 v2（DirectRLEnv，owner: Agent A）。

W2 变更：
- 场景几何全部以 B 的 assets_src/scene_layout.yaml 为准（F 桌 +x、基座贴桌
  后沿、F_L 在 -y；桌 = 0.05 厚桌板碰撞体，顶面 0.75）。
- 球分解默认 B 的正式版（spheres: bundled）+ contact_semantics 语义建对。
- 观测 = q/qd + 活跃球对集合（M=32 + mask，置换不变特征）。
- coordinator 模式（A5）：action = alpha(4)+p(1)，env 内部 delta 源 + L2
  兜底在环；奖励 = 跟踪 - tube - 平滑 - 兜底深度 - VIOLATION（终止）。
只在服务器上跑（Isaac）；Mac 端别 import。
"""

from __future__ import annotations

import torch
import yaml

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

from isaaclab_assets import FRANKA_PANDA_CFG, UR10_CFG

from safeduo.algo.reward_shaping import (
    alpha_util_bonus,
    arm_alpha_util_gate,
    arm_bypass_block_by_class,
    arm_hazard_gray_flags,
    arm_min_margin,
    margin_cost_by_class_exempt,
    shaped_margin_cost,
)
from safeduo.configs import load_config, repo_root
from safeduo.delta.smoke_noise import SmokeNoiseDelta
from safeduo.envs.task_objects import parse_table_objects
from safeduo.eval.target_guard_trace import (
    TargetGuardTerminationSnapshot,
    _pair_map_authority_from_sphere_distance,
    build_target_guard_step_cache,
    materialize_target_guard_termination as materialize_target_guard_termination_snapshot,
)
from safeduo.safety.backstop import BackstopConfig, VelocityDamperBackstop
from safeduo.safety.coupling import (
    coupling_from_grasps, coupling_features, group_min_alpha,
    structural_observation_mask,
)
from safeduo.safety.geometry import IsaacGeometryProvider
from safeduo.safety.semantics import semantics_with_safety_overrides
from safeduo.safety.sphere_distance import SphereDistanceModule
from safeduo.safety.sphere_specs import (
    bundled_arm_specs,
    placeholder_specs,
    real_v3_arm_specs,
    real_v5_arm_specs,
    real_v7_arm_specs,
    real_v7r16_arm_specs,
)
from safeduo.safety.target_guard import (
    TargetGuardBatchAbort,
    TargetGuardConfig,
    TargetGuardContractError,
    TargetGuardRuntime,
    TargetRebaseGuard,
)
from safeduo.safety.types import (
    ARM_KEYS,
    ARMS_OF_ROBOT,
    CLASS_CROSS,
    DOF_OF,
    PAIR_ARM_FEATURE_DIM,
    PAIR_FEATURE_DIM,
    TOTAL_DOF,
    SceneState,
    pair_obs_features,
    pair_obs_features_arm,
)

_FR3_USD = f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaFR3/fr3.usd"
_UR5E_USD = f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur5e/ur5e.usd"
_ARM_JOINT_RE = {"panda": ["panda_joint[1-7]"], "fr3": ["fr3_joint[1-7]"],
                 "ur10": [".*"], "ur5e": [".*"],
                 # JAKA Zu7（v4 换装）：只选臂 6 关节；组合 USD 里 F2 手关节名
                 # 形如 (right|left)_(thumb|...)_N_joint，不会被 fullmatch 误中
                 "jaka_zu7": ["joint[1-6]"],
                 # UR5+RH56DFX（R14 v7 换装）：标准 UR 关节名显式列出（组合 USD
                 # 里 DFX 手 12 关节名带 (left|right)_ 前缀，fullmatch 不误中）
                 "ur5_dfx": ["shoulder_pan_joint", "shoulder_lift_joint",
                             "elbow_joint", "wrist_1_joint", "wrist_2_joint",
                             "wrist_3_joint"]}
_EE_CANDIDATES = {"panda": ["panda_hand", "panda_link7"],
                  # 组合 USD（去自带夹爪）里 fr3_hand 不在 -> 退 fr3_link8 法兰
                  "fr3": ["fr3_hand", "fr3_link8", "fr3_link7"],
                  "ur10": ["ee_link", "tool0", "wrist_3_link"],
                  "ur5e": ["ee_link", "tool0", "wrist_3_link"],
                  "jaka_zu7": ["link6"],
                  # R14 v7：convert_urdf --merge-joints 把 tool0/flange 与 DFX
                  # 手基座全并入 wrist_3_link -> 腕即法兰即手底座
                  "ur5_dfx": ["wrist_3_link"]}
# ASSEMBLY §2.2 cuRobo retract 位（原 -1.9/-1.57 为 W1 手调值，W3 对齐 B 规格）
_UR_INIT_QPOS = {"shoulder_pan_joint": 0.0, "shoulder_lift_joint": -2.2,
                 "elbow_joint": 1.9, "wrist_1_joint": -1.383,
                 "wrist_2_joint": -1.57, "wrist_3_joint": 0.0}
# ASSEMBLY §2.1 FR3 预备位（朝工作区；panda 默认 j4=-2.81/j6=3.037 是蜷肘位）
_FR3_INIT_OVERRIDE = {"panda_joint4": -2.31, "panda_joint6": 1.74}
# ASSEMBLY_V3 §2.2 JAKA Zu7 预备位；A6 冒烟微调 j2 1.57->1.35（抬臂 12 度：
# B 目测位下 link2 粗球贴桌 -1.7mm 出生违规；j2=1.25 则跨机手互穿 -19mm——
# 强耦合布局的出生位形窗口很窄，改动前先跑 /tmp/a6_diag_pose2.py 同款网格）
_JAKA_INIT_QPOS = {"joint1": 0.0, "joint2": 1.35, "joint3": -1.2,
                   "joint4": 1.2, "joint5": 1.57, "joint6": 0.0}
# 组合 USD 里 F2 手 12 关节（6 主动 + 6 mimic）需要执行器覆盖（Isaac Lab 硬要求）；
# 增益 = 转换命令同值（5.0/0.5），手不进安全层动作空间、默认张开位
_F2_HAND_ACTUATOR = ImplicitActuatorCfg(
    # R14 v7：DFX 手小指名为 little（F2 为 pinky），正则并集两代手通吃
    joint_names_expr=[".*(thumb|index|middle|ring|pinky|little).*"],
    effort_limit_sim=2.0, velocity_limit_sim=6.28,
    stiffness=5.0, damping=0.5)


def _robot_cfg(variant: str, custom_usd: str, with_hands: bool = False) -> ArticulationCfg:
    if variant == "jaka_zu7":
        # v4 换装：无官方 cfg，从零构造（ASSEMBLY_V3 §2.2；USD 由
        # convert_urdf --fix-base 产出，增益烘焙在 USD、此处显式声明同值）
        cfg = ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(
                usd_path=custom_usd,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, max_depenetration_velocity=5.0),
                # Round-110 裁定②：v4 开启全 articulation 自碰物理（UR 幽灵根治，
                # A6-W6 §3：旧 ur5e 默认关 -> U 自碰在 PhysX 里穿越无力、GT 盲区）。
                # 结构性贴邻对（link3|link5、link4|link6，A6 实测网格不可分离）由
                # compose_robot_usd 写入 FilteredPairsAPI，防 fr3_hand 式焊接伪力。
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(joint_pos=dict(_JAKA_INIT_QPOS)),
            actuators={"arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]"],
                # 限位出处 jaka_zu7_clean.urdf 头注；effort 为 UR5e 同级保守估值
                effort_limit_sim={"joint[1-3]": 150.0, "joint[4-6]": 28.0},
                velocity_limit_sim=3.1416,
                stiffness=1200.0, damping=80.0)},
        )
        if with_hands:
            cfg.actuators["hand"] = _F2_HAND_ACTUATOR
        return cfg
    if variant == "ur5_dfx":
        # R14 v7 换装：owner 组合 URDF（UR5 CB 系 + RH56DFX）直转 USD。
        # 限位/effort/速度 = URDF 内官方值（150/150/150/28/28/28 Nm，3.14 rad/s）；
        # 增益 1200/80 与 JAKA/UR5e 同源（位置伺服内环刚，ASSEMBLY §2.2 惯例）。
        cfg = ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(
                usd_path=custom_usd,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, max_depenetration_velocity=5.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(joint_pos=dict(_UR_INIT_QPOS)),
            actuators={"arm": ImplicitActuatorCfg(
                joint_names_expr=list(_ARM_JOINT_RE["ur5_dfx"]),
                effort_limit_sim={"shoulder_.*|elbow_joint": 150.0,
                                  "wrist_.*": 28.0},
                velocity_limit_sim=3.1416,
                stiffness=1200.0, damping=80.0)},
        )
        if with_hands:
            cfg.actuators["hand"] = _F2_HAND_ACTUATOR
        return cfg
    if variant in ("panda", "fr3"):
        cfg = FRANKA_PANDA_CFG.copy()
        cfg.init_state.joint_pos = {**cfg.init_state.joint_pos, **_FR3_INIT_OVERRIDE}
        # ASSEMBLY §2.1 HIGH_PD 增益。FRANKA_PANDA_CFG 的 80/4 重力下垂 ~3cm：
        # B 球分解 v2 加肥后 link3|link5 settle 谷底从 +1.3cm 穿到 -3.9mm ->
        # 全模式出生违规（W3 estop 512/512 事故，STATUS_A）。手指执行器不动。
        for name, act in cfg.actuators.items():
            if "hand" not in name:
                act.stiffness = 400.0
                act.damping = 80.0
        if variant == "fr3":
            cfg.spawn.usd_path = _FR3_USD
            cfg.init_state.joint_pos = {k.replace("panda", "fr3"): v
                                        for k, v in cfg.init_state.joint_pos.items()}
            for act in cfg.actuators.values():
                act.joint_names_expr = [e.replace("panda", "fr3") for e in act.joint_names_expr]
    elif variant in ("ur10", "ur5e"):
        cfg = UR10_CFG.copy()
        cfg.init_state.joint_pos = dict(_UR_INIT_QPOS)
        # ASSEMBLY §2.2：UR e-series 位置接口内环刚（1kHz servoj），1200/80；
        # W1 借 UR10 的 800/40 是已记风险项（STATUS_A W1 §5.4），一并清账。
        for act in cfg.actuators.values():
            act.stiffness = 1200.0
            act.damping = 80.0
        if variant == "ur5e":
            cfg.spawn.usd_path = _UR5E_USD
    else:
        raise ValueError(f"未知机器人变体: {variant}")
    if custom_usd:
        cfg.spawn.usd_path = custom_usd
    if with_hands:
        # 组合 USD：自带夹爪已删（panda_hand 执行器组悬空会报错，剔除），
        # F2 手关节挂通用执行器；init_state 的 finger 初始位形键同批清除
        # （fr3_finger_joint.* 在无手指资产上 resolve 必炸——v4 冒烟首战教训）
        cfg.actuators = {name: act for name, act in cfg.actuators.items()
                         if "hand" not in name}
        cfg.actuators["f2_hand"] = _F2_HAND_ACTUATOR
        cfg.init_state.joint_pos = {k: v for k, v in cfg.init_state.joint_pos.items()
                                    if "finger" not in k}
    cfg.spawn.activate_contact_sensors = True
    return cfg


def _yaw_quat(yaw: float) -> tuple:
    import math
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def _resolve_scene_paths(scene: dict, root) -> dict:
    """R35: resolve relative usd/mdl/texture paths of the dressing block against
    the repo root (absolute / URL paths pass through)."""
    import os as _os

    def _abs(p):
        if not p or "://" in str(p) or _os.path.isabs(str(p)):
            return p
        return str(root / str(p))

    out = dict(scene)
    tm = out.get("table_material")
    if isinstance(tm, dict) and tm.get("mdl"):
        out["table_material"] = {**tm, "mdl": _abs(tm["mdl"])}
    out["static_props"] = [{**p, "usd": _abs(p["usd"])} for p in (out.get("static_props") or [])]
    bg = out.get("background")
    if isinstance(bg, dict) and bg.get("usd"):
        out["background"] = {**bg, "usd": _abs(bg["usd"])}
    if out.get("dome_texture"):
        out["dome_texture"] = _abs(out["dome_texture"])
    return out


def _r35_visual_material(spec: dict):
    """scene.table_material -> Isaac Lab visual material cfg (MDL or PreviewSurface)."""
    if spec.get("mdl"):
        ts = spec.get("texture_scale")
        return sim_utils.MdlFileCfg(mdl_path=spec["mdl"], project_uvw=bool(spec.get("project_uvw", True)),
                                    texture_scale=(tuple(float(v) for v in ts) if ts else None),
                                    albedo_brightness=(float(spec["albedo_brightness"])
                                                       if spec.get("albedo_brightness") is not None else None))
    return sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(spec.get("color", (0.5, 0.35, 0.2))),
                                       roughness=float(spec.get("roughness", 0.5)),
                                       metallic=float(spec.get("metallic", 0.0)))


def _r35_spawn_static(prim_path: str, spec: dict, collision_default: bool):
    """R35: spawn a static USD prop (fixture / dressing / background shell).
    collision=True -> static colliders (kinematic if the asset authors a rigid
    body); collision=False -> all colliders in the asset disabled."""
    import math as _math

    coll = bool(spec.get("collision", collision_default))
    yaw = _math.radians(float(spec.get("yaw", 0.0)))
    rot = (_math.cos(yaw / 2.0), 0.0, 0.0, _math.sin(yaw / 2.0))
    scale = spec.get("scale", 1.0)
    scale = tuple(scale) if isinstance(scale, (list, tuple)) else (float(scale),) * 3
    kw = {}
    if coll:
        kw["collision_props"] = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        if spec.get("kinematic", True):
            kw["rigid_props"] = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
    else:
        kw["collision_props"] = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
    cfg = sim_utils.UsdFileCfg(usd_path=spec["usd"], scale=scale, **kw)
    cfg.func(prim_path, cfg, translation=tuple(spec.get("pos", (0.0, 0.0, 0.0))), orientation=rot)


def _r35_apply_hand_torsional_patch(env_root: str, arms, spec: dict):
    """R33/R35: PhysxCollisionAPI torsionalPatchRadius on the hand colliders of
    the template env (before cloning). spec: {radius, min_radius, links_regex,
    arms}. Returns the number of collision prims patched."""
    import re as _re

    import omni.usd
    from pxr import PhysxSchema, Usd, UsdPhysics

    radius = float(spec.get("radius", 0.02))
    min_radius = float(spec.get("min_radius", 0.01))
    pat = _re.compile(spec.get("links_regex",
                               r"(?i)(finger|thumb|index|middle|ring|pinky|little|palm|hand_base|_tip|_pad)"))
    stage = omni.usd.get_context().get_stage()
    n = n_deinst = 0
    for arm in spec.get("arms", list(arms)):
        root = stage.GetPrimAtPath(f"{env_root}/{arm}")
        if not root or not root.IsValid():
            continue
        # the F2 / DFX hand `collisions` scopes are instanceable in the composed
        # robot USDs -> their meshes are read-only instance proxies; de-instance
        # the matching scopes on the template env so the API can be authored
        for prim in list(Usd.PrimRange(root)):
            if prim.IsInstance() and pat.search(str(prim.GetPath())) and prim.GetName() == "collisions":
                prim.SetInstanceable(False)
                n_deinst += 1
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if not pat.search(str(prim.GetPath())):
                continue
            api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            api.CreateTorsionalPatchRadiusAttr(radius)
            api.CreateMinTorsionalPatchRadiusAttr(min_radius)
            n += 1
    print(f"R35_HAND_TORSION de-instanced hand collision scopes: {n_deinst}", flush=True)
    print(f"R35_HAND_TORSION radius={radius} min={min_radius} collision_prims={n}", flush=True)
    if n == 0:
        sample = []
        for arm in spec.get("arms", list(arms)):
            root = stage.GetPrimAtPath(f"{env_root}/{arm}")
            if root and root.IsValid():
                sample += [str(p.GetPath()) for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.CollisionAPI)][:12]
        print(f"R35_HAND_TORSION no match; sample collision prims: {sample}", flush=True)
    return n


def _r34_fixup_mesh_object(prim_path: str, static_friction: float, dynamic_friction: float,
                           glass_tint: tuple) -> None:
    """R34 (2026-09-06): make a sim-ready mesh asset usable on the SafeDuo table.

    (1) Friction: the psilab glassware authors PhysX materials of 0.2-0.95;
        a 7 cm glass bottle at 0.2 slides out of the pinch during the carry
        (r34_reagent1). All PhysicsMaterialAPI prims under the object take the
        yaml tabletop friction (the hands' rubber pads are the real contact).
    (2) Visibility: glass bodies are OmniGlass shaders, which the headless
        real-time renderer draws fully transparent (only the cap of a reagent
        bottle was visible). Meshes bound to a glass material are rebound to
        an opaque tinted PreviewSurface; textured labels keep their material.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root or not root.IsValid():
        return
    n_mat, n_glass, n_cut = 0, 0, 0
    opaque = None
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            m = UsdPhysics.MaterialAPI(prim)
            m.CreateStaticFrictionAttr().Set(float(static_friction))
            m.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
            n_mat += 1
        if prim.IsA(UsdShade.Shader):
            # OmniPBR with an opacity map: the whole bottle body is cut out
            # (only the cap rendered in r34_reagent1/2) -> render it opaque
            sh = UsdShade.Shader(prim)
            for name in ("enable_opacity", "enable_opacity_texture"):
                inp = sh.GetInput(name)
                if inp and inp.Get():
                    inp.Set(False)
                    n_cut += 1
        if prim.IsA(UsdGeom.Mesh):
            mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if not mat:
                continue
            is_glass = False
            for sh in Usd.PrimRange(mat.GetPrim()):
                if sh.IsA(UsdShade.Shader) and UsdShade.Shader(sh).GetInput("glass_color"):
                    is_glass = True
                    break
            if not is_glass:
                continue
            if opaque is None:
                mpath = Sdf.Path(prim_path).AppendChild("Looks_R34").AppendChild("OpaqueGlass")
                opaque = UsdShade.Material.Define(stage, mpath)
                shader = UsdShade.Shader.Define(stage, mpath.AppendChild("Shader"))
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*glass_tint))
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.25)
                shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
                opaque.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI(prim).Bind(opaque, UsdShade.Tokens.strongerThanDescendants)
            n_glass += 1
    print(f"R34_MESH_OBJECT {prim_path}: friction {static_friction}/{dynamic_friction} on {n_mat} materials, "
          f"{n_glass} glass meshes rebound, {n_cut} opacity inputs disabled", flush=True)


@configclass
class DuoEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 10.0
    action_space = TOTAL_DOF
    observation_space = 1
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=2)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=4.0,
                                                     replicate_physics=True)
    robot_cfgs: dict = {}
    variant_of: dict = {}
    table_board: dict = {}        # {centers: ((x,y,z)*2), half: (hx,hy,hz)} 桌板碰撞体
    base_poses: dict = {}         # arm -> (pos(3), rot_wxyz(4))
    delta_clip: float = 0.08
    safety_cfg: dict = {}
    spheres_mode: str = "bundled"
    semantics_yaml: str = ""
    coordinator: dict = {}
    events_cfg: dict = {}
    enable_contact_gt: bool = False
    enable_viz_camera: bool = False
    # R17 S9: optional tabletop task objects (assets.table_objects, see
    # envs/task_objects.py schema). Default [] = scene graph bit-for-bit
    # unchanged; props are NOT in the sphere safety model / observation.
    table_objects: list = []
    # R27 S3: optional solver iteration overrides for the task objects
    # (assets.object_solver: {position_iterations, velocity_iterations});
    # {} = historical RigidBodyPropertiesCfg bit-for-bit.
    object_solver: dict = {}
    # R35 scene dressing, yaml top-level block `dressing:` (default {} =
    # scene graph bit-for-bit; `scene` itself is Isaac Lab's InteractiveSceneCfg):
    #   dressing.table_material: {mdl: <path>, texture_scale: [sx, sy]} or
    #                            {color: [r,g,b], roughness, metallic}
    #   dressing.static_props: [{name, usd, pos, yaw, scale, collision: bool,
    #                            kinematic: bool}]  -- fixtures / dressing
    #   dressing.background: {usd, pos, yaw, scale}  -- environment shell
    #   dressing.ground_visible: bool (hide the grid ground under a background)
    #   dressing.dome_intensity: float, dressing.dome_texture: <hdr path>
    scene_dressing: dict = {}
    # R33/R35 contact fidelity: torsional patch radius on the hand colliders
    # (assets.hand_torsional_patch: {radius, min_radius, links_regex}); PhysX
    # has no torsional friction by default, so a two-pad pinch cannot resist
    # rotation about the pinch axis (profile_rail pitch, r34_rail3/4/5).
    hand_torsional_patch: dict = {}


def make_duo_env_cfg(num_envs: int | None = None, device: str = "cuda:0",
                     yaml_name: str = "duo_env.yaml",
                     franka_variant: str | None = None,
                     ur_variant: str | None = None,
                     coordinator: bool | None = None,
                     arm_aware_obs: bool | None = None,
                     p2_obs: bool | None = None) -> DuoEnvCfg:
    y = load_config(yaml_name)
    if franka_variant:
        y["assets"]["franka"]["variant"] = franka_variant
    if ur_variant:
        y["assets"]["ur"]["variant"] = ur_variant
    if coordinator is not None:
        y["coordinator"]["enabled"] = bool(coordinator)
    # R15 ①：arm-aware pair 观测开关（None = 沿用 yaml，历史 yaml 无此键 ->
    # 关，obs 维度逐位不变；开启改变 obs_dim，旧 checkpoint 不兼容——只随
    # v7 重训上线，勿在 resume 旧 run 时开）
    if arm_aware_obs is not None:
        y["coordinator"]["arm_aware_obs"] = bool(arm_aware_obs)
    # a25/P2：策略观测补全开关（None = 沿用 yaml，历史 yaml 无此键 -> 关，
    # obs 维度逐位不变；开启 +16 维改变 obs_dim，旧 checkpoint 不兼容——
    # 只随 a25 重训上线）。需要 r18_hazard_flags 供 hazard/gray 缓存，
    # DuoEnv.__init__ 有断言把关。
    if p2_obs is not None:
        y["coordinator"]["p2_obs"] = bool(p2_obs)
        if p2_obs:
            # train_lagrangian --p2-obs 同时点亮两开关（其 284-288 行的耦合）；
            # eval 侧经此入口建环境时补齐同一联动，否则 __init__ 断言拦下。
            y["coordinator"]["r18_hazard_flags"] = True
    root = repo_root()
    with open(root / y["scene"]["layout_yaml"]) as f:
        layout = yaml.safe_load(f)

    cfg = DuoEnvCfg()
    cfg.decimation = int(y["sim"]["decimation"])
    cfg.episode_length_s = float(y["sim"]["episode_length_s"])
    sim_kwargs = {}
    # PhysX 缓冲直通（A9-W9）：v5 凸分解使 contact patch 需求超默认
    # 163840（4096 env 实测要 >=249856，溢出 = PhysX 静默丢接触）。yaml 无
    # physx 节时不传参 -> SimulationCfg 全默认，v3/v4 行为逐位不变。
    physx_over = y["sim"].get("physx") or {}
    if physx_over:
        from isaaclab.sim import PhysxCfg

        px = PhysxCfg()
        for k, v in physx_over.items():
            assert hasattr(px, k), f"未知 PhysxCfg 字段: {k}"
            setattr(px, k, type(getattr(px, k))(v))
        sim_kwargs["physx"] = px
    cfg.sim = SimulationCfg(dt=float(y["sim"]["dt"]), render_interval=cfg.decimation,
                            device=device, **sim_kwargs)
    cfg.scene = InteractiveSceneCfg(num_envs=int(num_envs or y["scene"]["num_envs"]),
                                    env_spacing=float(y["scene"]["env_spacing"]),
                                    replicate_physics=True)
    # 桌板：以 scene_layout 为准（0.05 厚板，顶面 top_z；腿省略——臂够不到）
    tb = layout["tables"]
    sx, sy, _ = tb["params"]["table_size"]
    thick = float(tb["params"]["table_top_thickness"])
    centers = []
    for key in ("table_F", "table_U"):  # 顺序 = sphere_distance.TABLE_NAMES
        cx, cy = tb[key]["center_xy"]
        top = float(tb[key]["top_z"])
        centers.append((float(cx), float(cy), top - thick / 2))
    cfg.table_board = {"centers": tuple(centers), "half": (sx / 2, sy / 2, thick / 2)}
    # 基座位姿：scene_layout robots.arms；|x| 按 Round-12 仲裁值覆盖
    base_x = y["scene"].get("base_x_abs")
    cfg.base_poses = {}
    for arm in ARM_KEYS:
        a = layout["robots"]["arms"][arm]
        pos = [float(v) for v in a["base_pos"]]
        if base_x is not None:
            pos[0] = float(base_x) if pos[0] > 0 else -float(base_x)
        cfg.base_poses[arm] = (tuple(pos), _yaw_quat(float(a["base_yaw"])))
    vf, vu = y["assets"]["franka"]["variant"], y["assets"]["ur"]["variant"]
    cfg.variant_of = {"F_L": vf, "F_R": vf, "U_L": vu, "U_R": vu}
    cfg.robot_cfgs = {}
    # v4 换装：左右手 F2 不同 -> 每臂 USD 可独立覆盖（arm_usd_overrides 优先于
    # per-robot custom_usd）；hands=true 时挂 F2 手执行器组。相对路径按 repo 根解析。
    arm_usd = y["assets"].get("arm_usd_overrides") or {}

    def _resolve(p: str) -> str:
        if p and not p.startswith("/") and "://" not in p and "{" not in p:
            return str(root / p)
        return p

    # 出生位 yaml 覆盖（A7-W7 出生位重工程，Round-114 裁定②）：键 = 最终关节名
    # （fr3_joint1..7 / joint1..6），在变体改名之后应用；v3 及更早 yaml 无此节
    # -> 走 _FR3_INIT_OVERRIDE/_JAKA_INIT_QPOS 代码默认，历史行为逐位不变。
    init_qpos = y.get("init_qpos") or {}
    for arm in ARM_KEYS:
        side = y["assets"]["franka"] if arm[0] == "F" else y["assets"]["ur"]
        usd = _resolve(arm_usd.get(arm) or side["custom_usd"])
        base = _robot_cfg(cfg.variant_of[arm], usd,
                          with_hands=bool(side.get("hands")))
        # R27 S3 (2026-09-05): physical-grasp recording envs may raise the
        # hand actuator gains / solver iterations (assets.hand_actuator,
        # assets.solver). Absent keys -> historical values bit-for-bit; the
        # shared _F2_HAND_ACTUATOR instance is never mutated in place.
        ha = y["assets"].get("hand_actuator") or {}
        if ha:
            for name, act in list(base.actuators.items()):
                if "hand" in name:
                    ov = {k: float(ha[k]) for k in ("stiffness", "damping",
                                                     "effort_limit_sim",
                                                     "velocity_limit_sim")
                          if k in ha}
                    base.actuators[name] = act.replace(**ov)
        # R27 (2026-09-05): per-side arm joint armature (assets.arm_armature:
        # {franka: x, ur: y}). The UR5-DFX wrist_3 link has ~3e-4 kg m^2 about
        # its axis; with kp 1200 the implicit drive is numerically unstable at
        # 120 Hz (joint ran at the velocity limit into its stop in every v7 run
        # since R14 -> U hand roll uncontrolled). armature 0.05 makes it hold
        # and track cleanly (wrist3_armature_probe). Absent -> bit-identical.
        aa = y["assets"].get("arm_armature") or {}
        side_key = "franka" if arm[0] == "F" else "ur"
        if aa.get(side_key) is not None:
            for name, act in list(base.actuators.items()):
                if "hand" not in name:
                    base.actuators[name] = act.replace(armature=float(aa[side_key]))
        # R27 (2026-09-05): per-side gravity compensation (assets.arm_gravity_comp:
        # {franka: true}). The FR3 at kp 400 sags 4-7 deg at the shoulder in
        # stretched poses (tau2 ~ -25 Nm = gravity), i.e. ~5 cm at the hand
        # (rod_pick place error, fl_track_probe.py); the real FR3 compensates
        # gravity in its 1 kHz loop. Robot links only -- objects keep gravity.
        gc = y["assets"].get("arm_gravity_comp") or {}
        if gc.get(side_key):
            base.spawn.rigid_props = base.spawn.rigid_props.replace(disable_gravity=True)
        sv = y["assets"].get("solver") or {}
        if sv:
            ap = base.spawn.articulation_props
            if ap is None:
                ap = sim_utils.ArticulationRootPropertiesCfg()
            ov = {}
            if "position_iterations" in sv:
                ov["solver_position_iteration_count"] = int(sv["position_iterations"])
            if "velocity_iterations" in sv:
                ov["solver_velocity_iteration_count"] = int(sv["velocity_iterations"])
            base.spawn.articulation_props = ap.replace(**ov)
        iq = init_qpos.get("franka" if arm[0] == "F" else "ur")
        if iq:
            base.init_state.joint_pos = {**base.init_state.joint_pos,
                                         **{k: float(v) for k, v in iq.items()}}
        pos, rot = cfg.base_poses[arm]
        cfg.robot_cfgs[arm] = base.replace(
            prim_path=f"/World/envs/env_.*/{arm}",
            init_state=base.init_state.replace(pos=pos, rot=rot))
    cfg.delta_clip = float(y["action"]["delta_clip"])
    cfg.safety_cfg = dict(y["safety"])
    cfg.spheres_mode = y["assets"]["spheres"]
    cfg.semantics_yaml = str(root / y["assets"]["semantics_yaml"])
    cfg.coordinator = dict(y["coordinator"])
    cfg.events_cfg = dict(y["events"])
    cfg.table_objects = parse_table_objects(y)
    cfg.object_solver = dict(y["assets"].get("object_solver") or {})
    cfg.scene_dressing = _resolve_scene_paths(dict(y.get("dressing") or {}), root)
    cfg.hand_torsional_patch = dict(y["assets"].get("hand_torsional_patch") or {})
    cfg.enable_contact_gt = bool(y["safety"]["contact_gt"])
    m = int(y["safety"]["max_active"])
    # v3 P0：pair 行 = [dist, closing_vel, class onehot(3)]（5 维，去 raw
    # pair_id——出生饱和根因），M=32 时 obs 243 -> 275；BC 侧同步在
    # warmstart_oracle.obs_features_duo_env（同一 pair_obs_features 出处）。
    # R15 ①：arm_aware_obs 开启时每行追加两臂 one-hot（13 维/行，M=32 时
    # obs 275 -> 531）；训练器侧 ObsLayout(pair_dim=13) 同步（单一开关双写，
    # train_lagrangian --arm-aware-obs 一次点亮两侧）
    pair_dim = (PAIR_ARM_FEATURE_DIM if y["coordinator"].get("arm_aware_obs")
                else PAIR_FEATURE_DIM)
    obs_dim = 2 * TOTAL_DOF + m * pair_dim + m
    if y["coordinator"]["enabled"]:
        cfg.action_space = 5                      # alpha(4)+p(1)
        obs_dim += TOTAL_DOF + 5                  # 当前 delta cmd + 上一步 alpha/p
        if y["coordinator"].get("p2_obs"):
            # a25/P2：逐臂 [hazard, gray] 旗标(8) + min-margin(4) + 目标
            # 积压 backlog 范数(4)。原 531 维里没有任何"离锁线多近"的显式
            # 信号（pair 行有 dist 但没减 d_min，且被 32 行淹没），安全头
            # 一直在盲猜——这 16 维就是 R22-F3 的修复
            obs_dim += 16
        if y["coordinator"].get("coupling_obs"):
            obs_dim += 6
        cfg.state_space = obs_dim + 13            # critic 特权：margins/激活/残差等
    else:
        cfg.action_space = TOTAL_DOF
    cfg.observation_space = obs_dim
    return cfg


class DuoEnv(DirectRLEnv):
    """4 臂 26 DoF。raw 模式 = 关节 delta 直通；coordinator 模式 = (alpha,p)。"""

    cfg: DuoEnvCfg

    def __init__(self, cfg: DuoEnvCfg, render_mode: str | None = None, **kwargs):
        # DirectRLEnv may reset during base construction.  The production guard
        # is opt-in and is constructed only after all safety tensors exist.
        self._target_guard_enabled = False
        super().__init__(cfg, render_mode, **kwargs)
        self._joint_idx: dict[str, torch.Tensor] = {}
        self._ee_idx: dict[str, int] = {}
        for arm in ARM_KEYS:
            art = self._arms[arm]
            ids, _ = art.find_joints(_ARM_JOINT_RE[self.cfg.variant_of[arm]])
            assert len(ids) == DOF_OF[arm], f"{arm}: 找到 {len(ids)} 关节，期望 {DOF_OF[arm]}"
            self._joint_idx[arm] = torch.tensor(ids, dtype=torch.long, device=self.device)
            for cand in _EE_CANDIDATES[self.cfg.variant_of[arm]]:
                if cand in art.body_names:
                    self._ee_idx[arm] = art.body_names.index(cand)
                    break
            assert arm in self._ee_idx, f"{arm}: EE body 未找到于 {art.body_names}"
        # 语义 + 球分解 + 距离模块
        sc = self.cfg.safety_cfg
        self._safety_row_horizon = sc.get('row_priority_lookahead_s')
        self._project_target_limits = bool(sc.get('project_target_limits', False))
        if self._safety_row_horizon is not None and self._safety_row_horizon <= 0:
            raise ValueError('safety.row_priority_lookahead_s must be positive')
        # safety 节覆写序列抽为公共函数（C14）：d_min_override（A6-W6 旋钮）→
        # d_min_per_link（v6 Round 154 ②，建对前注入随 pair 表烘焙）→
        # d_warn_override（v6 Round 152 ②，只动塑形带不动 d_soft）。行为与
        # 旧 inline 块逐位一致；train_lagrangian 消费同一函数取训练器侧
        # d_warn/d_min（C12_HANDOFF §1/§2 的"单源读取"），两侧语义不再分叉。
        sem = semantics_with_safety_overrides(self.cfg.semantics_yaml, sc)
        if self.cfg.spheres_mode == "bundled":
            specs = bundled_arm_specs(repo_root())
        elif self.cfg.spheres_mode == "real_v3":
            specs = real_v3_arm_specs(repo_root())
        elif self.cfg.spheres_mode == "real_v5":
            # v5 场景包（A9-W9）：jaka_zu7_v5 + F2 v5 手球（fr3 零改动）
            specs = real_v5_arm_specs(repo_root())
        elif self.cfg.spheres_mode == "real_v7":
            # v7 场景包（R14）：U 臂 UR5+DFX 手球；F 臂沿用 v5
            specs = real_v7_arm_specs(repo_root())
        elif self.cfg.spheres_mode == "real_v7r16":
            # r16 几何壳修正（R16/S12）：F 臂 link3/link6/手指球重拟合，
            # U 侧与 real_v7 相同；配 contact_semantics_v7r16.yaml
            specs = real_v7r16_arm_specs(repo_root())
        else:
            specs = placeholder_specs(self.cfg.variant_of["F_L"], self.cfg.variant_of["U_L"])
            sem = None  # 占位球 link 名不走语义正则，退 legacy
        self._sph = SphereDistanceModule(
            specs, semantics=sem, max_active=int(sc["max_active"]),
            d_soft=(sem.d_soft if sem else 0.05), tau_ttc=float(sc["tau_ttc"]),
            quota_cross=int(sc.get("quota_cross", 8)),
            device=self.device)
        self._sph.bind_bodies({arm: self._arms[arm].body_names for arm in ARM_KEYS})
        # 塑形带上界：历史上与 d_soft 同值 0.05（v4/v5 行为不变）；v6 起由
        # semantics d_warn（含 d_warn_override）承载，legacy 无语义路径退 d_soft
        self._d_warn = float(sem.d_warn) if sem is not None else float(self._sph.d_soft)
        tb = self.cfg.table_board
        self._sph.bind_tables(torch.tensor(tb["centers"]),
                              torch.tensor(tb["half"]).repeat(2, 1))
        # T_FU（W2 静态：F_L 基座系下 U_L 基座位姿；随机化 M1 接）
        f_pos, f_rot = (torch.tensor(v, device=self.device, dtype=torch.float32)
                        for v in self.cfg.base_poses["F_L"])
        u_pos, u_rot = (torch.tensor(v, device=self.device, dtype=torch.float32)
                        for v in self.cfg.base_poses["U_L"])
        qf_inv = quat_inv(f_rot.unsqueeze(0))
        rel_p = quat_apply(qf_inv, (u_pos - f_pos).unsqueeze(0))
        rel_q = quat_mul(qf_inv, u_rot.unsqueeze(0))
        self._t_fu = torch.cat([rel_p, rel_q], dim=-1).expand(self.num_envs, 7).contiguous()
        # coordinator 组件
        self._coord = bool(self.cfg.coordinator.get("enabled"))
        target_guard_cfg = sc.get("target_guard", {})
        if not isinstance(target_guard_cfg, dict):
            raise ValueError("safety.target_guard must be a mapping")
        unknown_target_guard = set(target_guard_cfg) - {
            "enabled",
            "gamma",
            "d_min",
            "vmax",
            "residual_tol",
            "closing_tol",
            "clear_margin",
        }
        if unknown_target_guard:
            raise ValueError(
                f"unknown safety.target_guard keys: {sorted(unknown_target_guard)}"
            )
        target_guard_enabled = target_guard_cfg.get("enabled", False)
        if type(target_guard_enabled) is not bool:
            raise ValueError("safety.target_guard.enabled must be a bool")
        self._target_guard_enabled = target_guard_enabled
        if self._target_guard_enabled and not self._coord:
            raise ValueError("safety.target_guard requires coordinator.enabled=true")
        # R15 ①：arm-aware pair 观测（默认关 = pair_obs_features 原路径逐位不变）
        self._arm_aware_obs = bool(self.cfg.coordinator.get("arm_aware_obs", False))
        # R15 ③a：α utility 显式定价权重（默认 0 = 奖励组装逐位不变）
        self._w_alpha_util = float(
            (self.cfg.coordinator.get("reward") or {}).get("w_alpha_util", 0.0))
        # R18 件③：逐臂 hazard/gray 旗标缓存开关（train_lagrangian 在
        # w_hazard_bce / w_alpha_deadzone > 0 时置位；默认关 = 不算不缓存，
        # 热路径零开销零漂移）。锁线 = semantics 逐类 d_min；灰带宽默认
        # cross/self/table = 10/7/10mm（锁线上方的滞回走廊，对应执行侧
        # θ_hi/θ_lo 缓冲），可由 coordinator.hazard_gray_band 覆盖。
        self._r18_hazard = bool(self.cfg.coordinator.get("r18_hazard_flags",
                                                         False))
        if self._r18_hazard:
            assert sem is not None, "r18_hazard_flags 需要 semantics yaml（锁线来源）"
            self._r18_class_dmin = torch.tensor(
                [float(sem.d_min["cross"]), float(sem.d_min["self"]),
                 float(sem.d_min["table"])], device=self.device)
            gb = self.cfg.coordinator.get("hazard_gray_band") or {}
            self._r18_gray_band = torch.tensor(
                [float(gb.get("cross", 0.010)), float(gb.get("self", 0.007)),
                 float(gb.get("table", 0.010))], device=self.device)
            # a27: controlled-rendezvous label exemption speed gate
            # (train_lagrangian --hazard-coop-vmax; 0 = off, bit-identical)
            self._r18_coop_vmax = float(
                self.cfg.coordinator.get("hazard_coop_vmax", 0.0) or 0.0)
        # a25/P2 观测补全：依赖 r18 旗标缓存（同一保留行口径），单独开会
        # 拿到未定义特征——直接拒绝而不是静默喂零
        self._p2_obs = bool(self.cfg.coordinator.get("p2_obs", False))
        self._coupling_obs = bool(self.cfg.coordinator.get("coupling_obs", False))
        self._coupling_enabled = bool(self.cfg.coordinator.get("coupling_enabled", False))
        self._coupling_graph = torch.zeros(self.num_envs, 4, 4,
                                          dtype=torch.bool, device=self.device)
        self._mask_structural_obs = bool(self.cfg.coordinator.get("mask_structural_obs", False))
        if self._p2_obs:
            assert self._r18_hazard, \
                "p2_obs 需要 r18_hazard_flags（train_lagrangian --p2-obs 会同时点亮两者）"
        self._targets = {arm: self._arms[arm].data.joint_pos[:, self._joint_idx[arm]].clone()
                         for arm in ARM_KEYS}
        self._last_out = None
        if self._coord:
            self._provider = IsaacGeometryProvider(self._arms, self._joint_idx,
                                                   self._sph, self.device,
                                                   jacobian_reference=sc.get('jacobian_reference', 'link'))
            bs = sc["backstop"]
            # R18 件①（S8 §6-5）：safety.damper_warn_mm（毫米）= damper 行
            # 参与门；键缺省 -> None = 现行为逐位不变（零漂移）。cap<0 的
            # backstop 强制退开行无条件保留（backstop.py 注释）。
            dw_mm = sc.get("damper_warn_mm")
            self._backstop = VelocityDamperBackstop(BackstopConfig(
                gamma=float(bs["gamma"]), vmax=float(bs["vmax"]),
                max_passes=int(bs["max_passes"]), tol=float(bs["tol"]),
                backlog_aware=bool(bs.get("backlog_aware", False)),
                # v6 泵病理修正包（Round 154 ①）：缺省即开（bug 修复即契约），
                # yaml 可显式关以逐位复现 v5 病理行为（审计/回放用）
                row_authority_clamp=bool(bs.get("row_authority_clamp", True)),
                exit_box_invariant=bool(bs.get("exit_box_invariant", True)),
                engage_dist=(float(dw_mm) / 1000.0
                             if dw_mm is not None else None),
                exempt_structural_rows=bool(bs.get("exempt_structural_rows",
                                                   False)),
                # R33: velocity-aware band (seconds of PD lag to look ahead);
                # None = bit-identical command damper
                lookahead_s=(float(bs["lookahead_s"])
                             if bs.get("lookahead_s") is not None else None),
                self_lookahead_s=(float(bs["self_lookahead_s"])
                                  if bs.get("self_lookahead_s") is not None else None),
                table_lookahead_s=(float(bs["table_lookahead_s"])
                                   if bs.get("table_lookahead_s") is not None else None),
                retain_conditional_rows=bool(bs.get('retain_conditional_rows', False)),
                struct_engage_dist=(float(bs["struct_engage_mm"]) / 1000.0
                                    if bs.get("struct_engage_mm") is not None
                                    else None)))
            # R29: the structural-row mask needs the per-class lock-line
            # tiers (semantics d_min); key absent -> not computed (zero drift)
            self._bs_struct_class_dmin = None
            if bool(bs.get("exempt_structural_rows", False)):
                _sem = self._sph.sem
                assert _sem is not None, \
                    "backstop.exempt_structural_rows needs a semantics yaml"
                self._bs_struct_class_dmin = torch.tensor(
                    [float(_sem.d_min["cross"]), float(_sem.d_min["self"]),
                     float(_sem.d_min["table"])], device=self.device)
            # R19 执行层旁通（S15，2026-08-21）：safety.bypass_emergency_mm
            # （毫米，标量）= 应急带宽——离合断开（α_exec=1）且该臂涉及的各
            # 保留行裕度 >= 锁线+带宽 时指令逐位直通（解析栈完全旁通，含箱
            # 钳制）；任一保留行进应急带或该臂被离合冻结时解析栈全量介入
            # （最后防线）。保留行口径 = arm_hazard_gray_flags（豁免行 +
            # 结构 per-link 行剔除，与 R18 hindsight 标注同源——S4 upper_arm
            # 恒 +1.7mm 悬停行不得把 U 臂钉死在非直通区）。键缺省 -> None =
            # 现行为逐位不变（零漂移）；评测 CLI 经 set_r19_bypass_mm 覆写。
            self._r19_bypass_band = None
            # R30 (2026-09-03): retreat pass-through -- a clutch-locked arm whose
            # operator command is non-closing on every retained row within the
            # d_warn band (or has no row in band) is released for this step.
            # Lit only by eval/recorder via coordinator.retreat_passthrough;
            # default off = bit-identical.
            self._retreat_pt = bool(self.cfg.coordinator.get(
                "retreat_passthrough", False))
            # R30b: dwell gate (steps of continuous lock before release may
            # apply); default 10 = 0.17 s at 60 Hz (four-gate 2026-09-03: dwell 10 dominates 30). Counter lives per arm.
            self._retreat_dwell = int(self.cfg.coordinator.get(
                "retreat_dwell_steps", 10))
            # Optional narrow backlog mitigation: when a safety layer actually
            # changes an arm command, rebase its persistent PD target at the
            # current joint state.  Legacy accumulation remains the default;
            # evaluation profiles can opt in explicitly.
            self._target_rebase_on_safety = bool(
                self.cfg.coordinator.get("target_rebase_on_safety", False))
            self._lock_steps = None
            bp_mm = sc.get("bypass_emergency_mm")
            if bp_mm is not None:
                self.set_r19_bypass_mm(float(bp_mm))
            # 真人手套量级限速：默认 0.015 rad/步 @60Hz ~= 0.9 rad/s 关节速度；
            # L1 默认 0.06 会到 ~3.6 rad/s，一步吞 2-3cm 边距，物理上刹不住。
            amp = float(self.cfg.coordinator.get("delta_amp_max", 0.015))
            src_kind = self.cfg.coordinator.get("delta_source")
            if src_kind == "l1":
                from safeduo.delta.l1_random import L1Params, L1RandomDelta
                self._delta_src = L1RandomDelta(
                    self.num_envs, params=L1Params(amp_max=amp), device=self.device)
            elif src_kind == "l2_mix":
                # 冲突富集课程流（C2-W3）：L1 + 8 类 L2 场景逐 env 混合，
                # 比例/权重在 configs/delta_curriculum.yaml；tube_fraction≈0
                # 的 v1 缺陷靠它修（STATUS_SERVER 08-12 03:20 发现二）。
                # T2/T3 扩展（2026-08-13）：coordinator.curriculum_yaml 可指
                # 其他课程配方（如 delta_curriculum_t3.yaml：amp 三档课程 +
                # 工作区全域 l1 + 技能族混流）；缺省文件名 = 原行为。
                from safeduo.delta.l2_env_source import (
                    ConflictMixSource,
                    load_curriculum_cfg,
                )
                # R15 v7 接线（2026-08-20）：coordinator.delta_env_yaml 让
                # ConflictMixSource 的 FK backend/漫游盒读到 v7 场景真值；
                # 缺省 duo_env.yaml = v5 原行为（geometry 键在课程 yaml 里）。
                self._delta_src = ConflictMixSource(
                    self.num_envs,
                    load_curriculum_cfg(str(self.cfg.coordinator.get(
                        "curriculum_yaml", "delta_curriculum.yaml"))),
                    device=self.device, amp_max=amp,
                    env_yaml=str(self.cfg.coordinator.get(
                        "delta_env_yaml", "duo_env.yaml")))
            elif src_kind == "skill_replay":
                # 技能轨迹回放源（spec S1-T1/T2，STATUS_T 预留接线点）：
                # coordinator.skill_dir 指 npz 库目录，skill_tier 0-3 噪声档，
                # skill_loop 播完循环。duo_env 每控制步恰好一次 sample()，
                # auto_advance=True 契约成立（时钟陷阱见 skill_replay 模块头）。
                from safeduo.delta.skill_replay import (
                    SkillNoiseParams,
                    SkillReplayDelta,
                    load_library,
                )
                self._delta_src = SkillReplayDelta(
                    self.num_envs,
                    load_library(str(self.cfg.coordinator["skill_dir"])),
                    params=SkillNoiseParams.tier(
                        int(self.cfg.coordinator.get("skill_tier", 1))),
                    device=self.device,
                    loop=bool(self.cfg.coordinator.get("skill_loop", True)))
            elif src_kind == "l1_full":
                # R25 全域覆盖随机流(卦限 LRU+6D 姿态+变速;delta/l1_coverage.py)
                from safeduo.delta.l1_coverage import make_l1_full_v7
                self._delta_src = make_l1_full_v7(
                    self.num_envs, amp_max=amp, device=self.device, cfg=self.cfg.coordinator)
            else:
                self._delta_src = SmokeNoiseDelta(self.num_envs, device=self.device)
            if self.cfg.coordinator.get("a31_replay_manifest"):
                from safeduo.delta.a31_replay_mix import A31ReplayMix
                self._delta_src = A31ReplayMix(self._delta_src, self.num_envs,
                    str(repo_root() / self.cfg.coordinator["a31_replay_manifest"]), self.device)
            self._q_soft_limits = {
                arm: self._arms[arm].data.soft_joint_pos_limits[:, self._joint_idx[arm], :]
                for arm in ARM_KEYS
            }
            if self._target_guard_enabled:
                if self._backstop.cfg.backlog_aware:
                    raise ValueError(
                        "safety.target_guard requires backstop.backlog_aware=false"
                    )
                self._target_guard_pair_map_authority = (
                    _pair_map_authority_from_sphere_distance(self._sph)
                )
                self._target_guard = TargetGuardRuntime(
                    TargetRebaseGuard(
                        TargetGuardConfig(
                            enabled=True,
                            gamma=float(target_guard_cfg.get("gamma", bs["gamma"])),
                            d_min=float(target_guard_cfg.get("d_min", 0.03)),
                            vmax=float(target_guard_cfg.get("vmax", bs["vmax"])),
                            residual_tol=float(
                                target_guard_cfg.get("residual_tol", 5e-4)
                            ),
                            closing_tol=float(
                                target_guard_cfg.get("closing_tol", 1e-6)
                            ),
                            clear_margin=float(
                                target_guard_cfg.get("clear_margin", 5e-3)
                            ),
                        )
                    )
                )
            self._gen = torch.Generator(device=self.device)
            self._gen.manual_seed(int(self.cfg.seed) if self.cfg.seed else 0)
            self._alpha_prev = torch.ones(self.num_envs, 4, device=self.device)
            self._p_prev = torch.zeros(self.num_envs, device=self.device)
            # p 翻转罚状态：上一次越出死区（±0.1）的符号，0=尚未出过死区
            self._p_sign_prev = torch.zeros(self.num_envs, device=self.device)
            self._step_cache: dict = {}
            self._pending_cmd = None  # 每步只采样一次：观测里给策略看的 = 实际执行的

    # ---- 场景 ----

    def _setup_scene(self):
        self._arms = {arm: Articulation(cfg) for arm, cfg in self.cfg.robot_cfgs.items()}
        tb = self.cfg.table_board
        half = tb["half"]
        _scene = getattr(self.cfg, "scene_dressing", None) or {}
        _tm = _scene.get("table_material")
        table_cfg = sim_utils.CuboidCfg(
            size=(half[0] * 2, half[1] * 2, half[2] * 2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=(_r35_visual_material(_tm) if _tm
                             else sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.35, 0.2))),
            # dressing.table_visible=false: the slab stays the collider / safety
            # plane while a workbench prop provides the visual (R35)
            visible=bool(_scene.get("table_visible", True)),
        )
        for name, center in zip(("TableF", "TableU"), tb["centers"]):
            table_cfg.func(f"/World/envs/env_.*/{name}", table_cfg, translation=center)
        # R35 scene dressing: static fixtures / props (optional colliders) and
        # an environment shell; task-layer visuals only, outside the sphere
        # safety model and the policy observation like the table objects.
        for _p in _scene.get("static_props") or []:
            _r35_spawn_static(f"/World/envs/env_.*/Prop_{_p['name']}", _p, collision_default=True)
        if _scene.get("background"):
            _r35_spawn_static("/World/envs/env_.*/Background", _scene["background"], collision_default=False)
        if getattr(self.cfg, "hand_torsional_patch", None):
            _r35_apply_hand_torsional_patch("/World/envs/env_0", self._arms.keys(), self.cfg.hand_torsional_patch)
        # R17 S9: optional tabletop task objects (default empty = no-op).
        # Demo props for the recognizable basic-task videos: rigid dynamic
        # cubes with high-friction material; deliberately OUTSIDE the sphere
        # safety model and the policy observation (task layer only).
        self._objects = {}
        if self.cfg.table_objects:
            from isaaclab.assets import RigidObject, RigidObjectCfg

            _osv = getattr(self.cfg, "object_solver", None) or {}
            _obj_solver_kw = {}
            if "position_iterations" in _osv:
                _obj_solver_kw["solver_position_iteration_count"] = int(
                    _osv["position_iterations"])
            if "velocity_iterations" in _osv:
                _obj_solver_kw["solver_velocity_iteration_count"] = int(
                    _osv["velocity_iterations"])
            import math as _math

            for spec in self.cfg.table_objects:
                _yaw = _math.radians(float(spec.get("yaw", 0.0)))
                _rot = (_math.cos(_yaw / 2.0), 0.0, 0.0, _math.sin(_yaw / 2.0))
                if spec.get("rpy_deg"):
                    # R35: Y-up assets (YCB cans) need a roll/pitch to stand
                    _r, _p, _y = (_math.radians(float(v)) for v in spec["rpy_deg"])
                    _cr, _sr = _math.cos(_r / 2), _math.sin(_r / 2)
                    _cp, _sp = _math.cos(_p / 2), _math.sin(_p / 2)
                    _cy, _sy = _math.cos(_y / 2), _math.sin(_y / 2)
                    _rot = (_cr * _cp * _cy + _sr * _sp * _sy, _sr * _cp * _cy - _cr * _sp * _sy,
                            _cr * _sp * _cy + _sr * _cp * _sy, _cr * _cp * _sy - _sr * _sp * _cy)
                if spec.get("usd"):
                    # R34 (2026-09-06): sim-ready mesh asset (lab glassware,
                    # bins, ...). The asset authors its own rigid-body / mass /
                    # mesh-collision APIs; we only add the solver knobs, an
                    # optional mass override and the tabletop friction.
                    _mass_kw = ({"mass_props": sim_utils.MassPropertiesCfg(mass=spec["mass"])}
                                if spec.get("mass") is not None else {})
                    spawn = sim_utils.UsdFileCfg(
                        usd_path=spec["usd"],
                        scale=tuple(spec.get("scale", (1.0, 1.0, 1.0))),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            disable_gravity=False,
                            max_depenetration_velocity=5.0,
                            **_obj_solver_kw),
                        # the asset authors its own PhysicsMaterialAPI; UsdFileCfg
                        # has no physics_material slot in this Isaac Lab
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        **_mass_kw,
                    )
                else:
                    spawn = sim_utils.CuboidCfg(
                        size=tuple(spec["size"]),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            disable_gravity=False,
                            max_depenetration_velocity=5.0,
                            **_obj_solver_kw),
                        mass_props=sim_utils.MassPropertiesCfg(
                            mass=spec["mass"]),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        physics_material=sim_utils.RigidBodyMaterialCfg(
                            static_friction=spec["static_friction"],
                            dynamic_friction=spec["dynamic_friction"]),
                        # emissive keeps the small prop clearly visible in the
                        # flat demo lighting (video readability, not physics)
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=tuple(spec["color"]),
                            emissive_color=tuple(0.45 * c for c in spec["color"])),
                    )
                obj_cfg = RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/Obj_{spec['name']}",
                    spawn=spawn,
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=tuple(spec["pos"]), rot=_rot),
                )
                self._objects[spec["name"]] = RigidObject(obj_cfg)
                if spec.get("usd"):
                    # template prim (env_0) is edited before cloning so every
                    # env inherits the tabletop friction and opaque glass
                    _r34_fixup_mesh_object(f"/World/envs/env_0/Obj_{spec['name']}",
                                           spec["static_friction"], spec["dynamic_friction"],
                                           tuple(spec.get("color") or (0.55, 0.35, 0.2)))
        ground = sim_utils.GroundPlaneCfg(visible=bool(_scene.get("ground_visible", True)))
        ground.func("/World/ground", ground)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        for arm, art in self._arms.items():
            self.scene.articulations[arm] = art
        for name, obj in self._objects.items():
            self.scene.rigid_objects[name] = obj
        self._contact = {}
        if self.cfg.enable_contact_gt:
            from isaaclab.sensors import ContactSensor, ContactSensorCfg

            for arm in ARM_KEYS:
                sensor = ContactSensor(ContactSensorCfg(prim_path=f"/World/envs/env_.*/{arm}/.*"))
                self.scene.sensors[f"contact_{arm}"] = sensor
                self._contact[arm] = sensor
        # viz 相机必须在 sim play 前建好，否则错过 sensor 初始化回调
        # （_ALL_INDICES 缺失，set_world_poses/data 都会炸）——record_video 用
        self._viz_cam = None
        if getattr(self.cfg, "enable_viz_camera", False):
            from isaaclab.sensors import Camera, CameraCfg

            cam_cfg = CameraCfg(
                prim_path="/World/viz_cam",
                width=1280, height=720,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=18.0,
                                                 clipping_range=(0.1, 60.0)),
            )
            self._viz_cam = Camera(cam_cfg)
            self.scene.sensors["viz_cam"] = self._viz_cam
        light = sim_utils.DomeLightCfg(intensity=float(_scene.get("dome_intensity", 2500.0)),
                                       texture_file=_scene.get("dome_texture"))
        light.func("/World/Light", light)

    # ---- 距离/状态 ----

    def _body_data(self, arm: str):
        d = self._arms[arm].data
        return (d.body_link_pos_w, d.body_link_quat_w,
                d.body_link_lin_vel_w, d.body_link_ang_vel_w)

    def compute_dist(self, need_full: bool = False):
        pos, quat, lin, ang = {}, {}, {}, {}
        for arm in ARM_KEYS:
            pos[arm], quat[arm], lin[arm], ang[arm] = self._body_data(arm)
        self._body_pos_cache = pos
        self._last_out = self._sph.compute(pos, quat, lin, ang, self.scene.env_origins,
                                           need_full=need_full or self._safety_row_horizon is not None)
        return self._last_out

    def safety_dist_out(self):
        if self._safety_row_horizon is None:
            return self._last_out
        return self._sph.prioritized_out(self._last_out, self._safety_row_horizon)

    def scene_state(self) -> SceneState:
        q, qd, ee_p, ee_q = {}, {}, {}, {}
        for arm in ARM_KEYS:
            art, jid = self._arms[arm], self._joint_idx[arm]
            q[arm] = art.data.joint_pos[:, jid]
            qd[arm] = art.data.joint_vel[:, jid]
            ee_p[arm] = art.data.body_link_pos_w[:, self._ee_idx[arm]] - self.scene.env_origins
            ee_q[arm] = art.data.body_link_quat_w[:, self._ee_idx[arm]]
        out = self._last_out or self.compute_dist()
        return SceneState(q=q, qd=qd, ee_pos=ee_p, ee_quat=ee_q,
                          active_pairs=out.active_pairs, active_mask=out.active_mask,
                          T_FU=self._t_fu, dt=self.cfg.sim.dt * self.cfg.decimation)

    # ---- 动作 ----

    def set_r19_bypass_mm(self, mm: "float | None") -> None:
        """R19 旁通应急带宽运行时开关（评测 CLI 用）；None = 关 = 零漂移。"""
        if mm is None:
            self._r19_bypass_band = None
            return
        sem = self._sph.sem
        assert sem is not None, "r19 bypass 需要 semantics yaml（锁线来源）"
        self._r19_class_dmin = torch.tensor(
            [float(sem.d_min["cross"]), float(sem.d_min["self"]),
             float(sem.d_min["table"])], device=self.device)
        self._r19_bypass_band = torch.full((3,), float(mm) / 1000.0,
                                           device=self.device)

    def _retreat_passthrough(self, alpha, cmd, rows, out):
        """R30: delegate to safety.retreat.retreat_release (pure torch, pinned)."""
        from safeduo.safety.retreat import retreat_release
        exempt = getattr(out, "viol_exempt", None) if out is not None else None
        locked = alpha < 1e-6
        if self._lock_steps is None or self._lock_steps.shape != alpha.shape:
            self._lock_steps = torch.zeros_like(alpha, dtype=torch.long)
        # count consecutive locked steps on the INCOMING alpha (a release step
        # does not reset the counter -- the driver still holds the lock)
        self._lock_steps = torch.where(locked, self._lock_steps + 1,
                                       torch.zeros_like(self._lock_steps))
        alpha, self._retreat_released = retreat_release(
            alpha, cmd, rows, exempt, self._d_warn,
            lock_steps=self._lock_steps, dwell=self._retreat_dwell)
        return alpha

    def set_grasp_state(self, object_ids, closed, near):
        """Task-layer estimate for the CURRENT observation/action, no attach inference."""
        graph = coupling_from_grasps(object_ids.to(self.device),
                                    closed.to(self.device), near.to(self.device))
        if graph.shape != self._coupling_graph.shape:
            raise ValueError("grasp state batch differs from environment")
        self._coupling_graph.copy_(graph)

    def _pre_physics_step(self, actions: torch.Tensor):
        if not self._coord:
            d = actions.clamp(-1.0, 1.0) * self.cfg.delta_clip
            i = 0
            for arm in ARM_KEYS:
                jid = self._joint_idx[arm]
                q = self._arms[arm].data.joint_pos[:, jid]
                self._targets[arm] = q + d[:, i:i + DOF_OF[arm]]
                i += DOF_OF[arm]
            return
        # coordinator：alpha/p -> delta 源 -> L2 兜底 -> 关节目标
        alpha = (actions[:, :4].clamp(-1.0, 1.0) + 1.0) * 0.5
        p = actions[:, 4].clamp(-1.0, 1.0)
        state = self.scene_state()
        cmd = self._pending_cmd if self._pending_cmd is not None \
            else self._delta_src.sample(state)
        out = self.safety_dist_out()
        rows = self._provider.rows_from(out, self._body_pos_cache)
        backlog = None
        if self._backstop.cfg.backlog_aware:
            backlog = {arm: self._targets[arm] - state.q[arm] for arm in ARM_KEYS}
        # R19 旁通掩码：α_exec=1（离合断开）且该臂无保留行进应急带
        # （hazard|gray，gray 带宽 = 应急带宽）-> 本步该臂逐位直通。
        # 逐类分桶版（与 arm_hazard_gray_flags 等价，有测试钉），
        # blocked_cls 进 step_cache 供评测侧归因"谁挡了直通"。
        bypass_arm = None
        blocked_cls = None
        if self._r19_bypass_band is not None:
            blocked_cls = arm_bypass_block_by_class(
                out.active_pairs, out.active_mask, out.active_dmin,
                out.viol_exempt, self._sph.pair_arms,
                self._r19_class_dmin, self._r19_bypass_band)
            bypass_arm = (alpha >= 1.0 - 1e-6) & ~blocked_cls.any(dim=(2, 3))
        struct_exempt = None
        if self._bs_struct_class_dmin is not None:
            _cls = out.active_pairs[..., 2].long().clamp(0, 2)
            struct_exempt = (out.active_mask
                             & ((out.active_dmin
                                 - self._bs_struct_class_dmin[_cls]).abs() > 1e-9))
        if self._retreat_pt:
            alpha = self._retreat_passthrough(alpha, cmd, rows, out)
        # After retreat release so one arm cannot independently reopen a group.
        if self._coupling_enabled:
            alpha = group_min_alpha(alpha, self._coupling_graph)
            if bypass_arm is not None:
                bypass_arm = bypass_arm & (alpha >= 1.0 - 1e-6)
        exec_cmd, bs_active, info = self._backstop.project(
            cmd, rows, alpha, p, state.dt, dmin=out.active_dmin, backlog=backlog,
            bypass_arm=bypass_arm, struct_exempt=struct_exempt,
            qd=(state.qd if (self._backstop.cfg.lookahead_s is not None
                             or self._backstop.cfg.self_lookahead_s is not None
                             or self._backstop.cfg.table_lookahead_s is not None) else None),
            contact_exempt=out.viol_exempt,
            delta_bounds=({a: (self._q_soft_limits[a][..., 0] - self._targets[a],
                               self._q_soft_limits[a][..., 1] - self._targets[a]) for a in ARM_KEYS}
                          if self._project_target_limits else None))
        # 持久目标积分：alpha=0 / 兜底清零时目标冻结，PD 收敛到定点才真停得住；
        # 按当前 q 重定基会追着惯性走，实测滑穿 d_min（见 STATUS_SERVER 复盘）。
        if not self._target_guard_enabled:
            for arm in ARM_KEYS:
                lim = self._q_soft_limits[arm]
                self._targets[arm] = (self._targets[arm] + exec_cmd.delta_q[arm]).clamp(
                    lim[..., 0], lim[..., 1])
            if self._target_rebase_on_safety:
                for arm_i, arm in enumerate(ARM_KEYS):
                    lim = self._q_soft_limits[arm]
                    rebase = (bs_active[:, arm_i]
                              | (alpha[:, arm_i] <= 1e-6)).unsqueeze(-1)
                    safe_target = state.q[arm] + exec_cmd.delta_q[arm]
                    self._targets[arm] = torch.where(
                        rebase, safe_target, self._targets[arm]).clamp(
                            lim[..., 0], lim[..., 1])
        else:
            target_before = dict(self._targets)
            try:
                guard_output = self._target_guard.apply(
                    q=state.q,
                    qd=state.qd,
                    target=target_before,
                    exec_delta=exec_cmd.delta_q,
                    rows=rows,
                    dt=state.dt,
                    pair_ids=out.active_idx,
                    exempt=out.viol_exempt,
                    soft_limits=self._q_soft_limits,
                    priority_p=p,
                )
            except TargetGuardContractError:
                # A programming-contract breach has no numeric emergency
                # decision to trace.  Runtime poison makes it process-fatal;
                # preserve targets/cache and leave before any simulator write.
                raise
            except TargetGuardBatchAbort as abort:
                self._step_cache = {
                    "alpha": alpha,
                    "p": p,
                    "cmd": cmd,
                    "exec": exec_cmd,
                    "bs_active": bs_active,
                    "violation": out.violation,
                    "target_guard": build_target_guard_step_cache(
                        decision=abort.emergency,
                        rows=rows,
                        pair_ids=out.active_idx,
                        exempt=out.viol_exempt,
                        closing=out.active_pairs[..., 1],
                        q=state.q,
                        qd=state.qd,
                        target_before=target_before,
                        exec_delta=exec_cmd.delta_q,
                        soft_limits=self._q_soft_limits,
                        dt=state.dt,
                        priority_p=p,
                        pair_map_authority=self._target_guard_pair_map_authority,
                    ),
                }
                raise
            target_guard_cache = build_target_guard_step_cache(
                decision=guard_output,
                rows=rows,
                pair_ids=out.active_idx,
                exempt=out.viol_exempt,
                closing=out.active_pairs[..., 1],
                q=state.q,
                qd=state.qd,
                target_before=target_before,
                exec_delta=exec_cmd.delta_q,
                soft_limits=self._q_soft_limits,
                dt=state.dt,
                priority_p=p,
                pair_map_authority=self._target_guard_pair_map_authority,
            )
            for arm in ARM_KEYS:
                self._targets[arm] = guard_output.next_target[arm]
        # 冲突管：任一跨机活跃行 margin<d_soft 且 closing>0
        ap, mask = out.active_pairs, out.active_mask
        tube = ((ap[..., 2] == CLASS_CROSS) & mask
                & (ap[..., 0] < self._sph.d_soft) & (ap[..., 1] > 0)).any(dim=-1)
        # R15 ③a：逐臂无险情门（与 tube/margin_cost 同一 pre-step 状态口径，
        # 即"动作决策时刻"的险情；开关关时不算不缓存，热路径零开销）
        alpha_util_gate = (arm_alpha_util_gate(ap, mask, self._sph.pair_arms,
                                               self._d_warn, self._sph.d_soft)
                           if self._w_alpha_util > 0.0 else None)
        self._step_cache = {
            "alpha": alpha, "p": p, "cmd": cmd, "exec": exec_cmd,
            "bs_active": bs_active, "violation": out.violation, "tube": tube,
            # Evaluation-only causal traces consume these already-computed
            # safety-row maxima.  Keeping the two (N,) tensors in the cache
            # adds no solver work and lets a recorded transition distinguish
            # a projection residual from downstream target/physics motion.
            "backstop_residual_F": info["residual_F"],
            "backstop_residual_U": info["residual_U"],
            "rows_d_min": rows.d.amin(dim=-1) if rows.d.shape[1] else None,
            # margin 逼近塑形（C2-W3 信用分配修复）：最坏行的归一化逼近度平方，
            # 幅值 [0, 4]；rows 只在此处在辖域内，先算好缓存给 _get_rewards。
            "margin_cost": shaped_margin_cost(rows.d, rows.d_min, rows.valid,
                                              d_warn=self._d_warn),
        }
        if self._target_guard_enabled:
            self._step_cache["target_guard"] = target_guard_cache
        if alpha_util_gate is not None:
            self._step_cache["alpha_util_gate"] = alpha_util_gate
        if bypass_arm is not None:
            self._step_cache["bypass_arm"] = bypass_arm
            self._step_cache["bypass_blocked_cls"] = blocked_cls

    def materialize_target_guard_termination(
        self,
        error: TargetGuardBatchAbort | TargetGuardContractError,
        *,
        attempted_transition_index: int,
    ) -> TargetGuardTerminationSnapshot:
        """Close the poisoned guard state before any target or physics write."""

        if not self._target_guard_enabled:
            raise RuntimeError("target-guard termination requires an enabled guard")
        if error is not self._target_guard.poison_error:
            raise ValueError("termination error differs from the runtime poison")

        emergency_cache = None
        pair_map_authority = None
        if type(error) is TargetGuardBatchAbort:
            emergency_cache = self._step_cache.get("target_guard")
            if not isinstance(emergency_cache, dict):
                raise ValueError("numeric guard abort is missing its exact step cache")
            pair_map_authority = self._target_guard_pair_map_authority
        elif type(error) is not TargetGuardContractError:
            raise TypeError("unsupported target-guard termination error")

        return materialize_target_guard_termination_snapshot(
            error,
            emergency_cache=emergency_cache,
            expected_pair_map_authority=pair_map_authority,
            attempted_transition_index=attempted_transition_index,
            pre_state_index=attempted_transition_index,
            committed_transitions=attempted_transition_index,
            sim_step_counter_before=int(self._sim_step_counter),
            episode_length_before=self.episode_length_buf,
        )

    def _apply_action(self):
        for arm in ARM_KEYS:
            self._arms[arm].set_joint_position_target(
                self._targets[arm], joint_ids=self._joint_idx[arm].tolist())

    # ---- 观测/奖励/终止 ----

    def _get_observations(self) -> dict:
        out = self.compute_dist()
        qs, qds = [], []
        for arm in ARM_KEYS:
            jid = self._joint_idx[arm]
            qs.append(self._arms[arm].data.joint_pos[:, jid])
            qds.append(self._arms[arm].data.joint_vel[:, jid])
        obs_mask = out.active_mask
        if self._mask_structural_obs:
            if self._bs_struct_class_dmin is None:
                raise ValueError("mask_structural_obs requires R29 structural exemption")
            obs_mask = structural_observation_mask(
                out.active_pairs, out.active_mask, out.active_dmin,
                self._bs_struct_class_dmin,
                lookahead_s=self._backstop.cfg.lookahead_s or 0.0,
                struct_engage_dist=self._backstop.cfg.struct_engage_dist)
        pair_feats = (pair_obs_features_arm(out.active_pairs, obs_mask,
                                            self._sph.pair_arms)
                      if self._arm_aware_obs else
                      pair_obs_features(out.active_pairs, obs_mask))
        parts = qs + qds + [pair_feats, obs_mask.float()]
        if self._coord:
            state = self.scene_state()
            self._pending_cmd = self._delta_src.sample(state)  # 下一步执行的同一 delta
            if hasattr(self._delta_src, "last_graph"):
                self._coupling_graph.copy_(self._delta_src.last_graph)
            parts += [self._pending_cmd.stacked(), self._alpha_prev,
                      self._p_prev.unsqueeze(-1)]
            # 豁免感知逐类 cost（C14，C12_HANDOFF §3 方案 b：env 侧算好放
            # 缓存、不动 obs 布局）：合法 near_table 豁免行剔除（λ_table 顶
            # 积分限幅的根源修复，Round 151），d_min 用 active_dmin 逐行真值
            # （per-link 覆写精确，§2 方案 b）。挂在 _get_observations 是因为
            # 此处 out = 本步后状态——与训练器"动作 t 的 cost 记 obs_{t+1}"
            # 口径对齐；train_lagrangian._EnvAdapter.cost_channels 消费。
            self._step_cache["cost_by_class"] = margin_cost_by_class_exempt(
                out.active_pairs, out.active_mask, out.active_dmin,
                viol_exempt=out.viol_exempt, d_warn=self._d_warn)
            # R18 件③：逐臂 hazard/gray 旗标（同一本步后状态口径，训练器
            # 在 buffer 里做 hindsight 前瞻；开关关时不算不缓存）
            if self._r18_hazard:
                self._step_cache["hazard_gray"] = arm_hazard_gray_flags(
                    out.active_pairs, out.active_mask, out.active_dmin,
                    out.viol_exempt, self._sph.pair_arms,
                    self._r18_class_dmin, self._r18_gray_band,
                    coop_vmax=self._r18_coop_vmax)
            if self._p2_obs:
                # a25/P2 追加 16 维（顺序钉死：flags 8 -> margin 4 ->
                # backlog 4，ObsLayout.extra_dim 只记总宽）。margin x4
                # 把 [-0.05, 0.25]m 拉到 [-0.2, 1]（与旗标同量级）；
                # backlog = 关节目标积压范数（rad），钳 2 防炸尺度
                hg = self._step_cache["hazard_gray"]
                mm_arm = arm_min_margin(
                    out.active_pairs, out.active_mask, out.active_dmin,
                    out.viol_exempt, self._sph.pair_arms,
                    self._r18_class_dmin)
                backlog = torch.stack(
                    [(self._targets[a] - state.q[a]).norm(dim=-1)
                     for a in ARM_KEYS], dim=-1)
                parts += [hg.reshape(self.num_envs, 8).float(),
                          mm_arm * 4.0, backlog.clamp(0.0, 2.0)]
            if self._coupling_obs:
                parts.append(coupling_features(self._coupling_graph))
        obs = torch.cat(parts, dim=-1)
        result = {"policy": obs}
        if self._coord and self.cfg.state_space:
            mm = out.min_margin
            c = self._step_cache
            priv = torch.cat([
                torch.stack([mm["cross"], mm["self_F"], mm["self_U"], mm["table"]],
                            dim=-1).clamp(-1, 10),
                out.active_mask.float().sum(-1, keepdim=True),
                c.get("bs_active", torch.zeros(self.num_envs, 4, device=self.device)).float(),
                out.violation.float().unsqueeze(-1),
                c.get("tube", torch.zeros(self.num_envs, device=self.device,
                                          dtype=torch.bool)).float().unsqueeze(-1),
                self._alpha_prev.mean(-1, keepdim=True),
                self._p_prev.unsqueeze(-1),
            ], dim=-1)
            result["critic"] = torch.cat([obs, priv], dim=-1)
        return result

    def _get_rewards(self) -> torch.Tensor:
        if not self._coord:
            return torch.zeros(self.num_envs, device=self.device)
        w = self.cfg.coordinator["reward"]
        c = self._step_cache
        cmd_s, exec_s = c["cmd"].stacked(), c["exec"].stacked()
        track = -float(w["w_track"]) * (exec_s - cmd_s).pow(2).sum(-1)
        # 兜底深度 = 沿命令方向的进展损失（惩罚改写深度，不惩罚激活）
        dirn = cmd_s / cmd_s.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        prog_loss = ((cmd_s - exec_s) * dirn).sum(-1).clamp_min(0.0)
        backstop_pen = -float(w["w_backstop"]) * prog_loss
        tube = c["tube"].float()
        tube_pen = -float(w["w_tube"]) * tube
        progress = float(w["w_progress"]) * tube * (exec_s * dirn).sum(-1)
        # v3（v3_recipe_proposal §2.2）：α 平滑项保留；p 撤出平滑罚——
        # w_smooth·(Δp)² 对 p 是"角落锁"（clamp 角恒零罚、离开先交钱，
        # r2 饱和被它加固），改为 tube 门控死区翻转罚，口径与
        # eval/metrics._priority_flips 一致（死区 ±0.1，越死区符号翻转记一次）；
        # 无冲突步 p 自由旋转不受罚。w_flip 消融臂 {0, 0.5, 2} 见 yaml。
        smooth = -float(w["w_smooth"]) * (c["alpha"] - self._alpha_prev).pow(2).sum(-1)
        sgn = torch.where(c["p"] > 0.1, 1.0, torch.where(c["p"] < -0.1, -1.0, 0.0))
        flip = (sgn != 0) & (self._p_sign_prev != 0) & (sgn != self._p_sign_prev)
        flip_pen = -float(w.get("w_flip", 0.0)) * tube * flip.float()
        self._p_sign_prev = torch.where(sgn != 0, sgn, self._p_sign_prev)
        viol = -float(w["w_violation"]) * c["violation"].float()
        # 逐步 margin 逼近惩罚：把违规信号即时化（0.99^600≈0.002 的终端信号
        # 到不了早期动作——v1 学会"多停"却不知道该在哪停的根因之一）。
        margin_pen = -float(w.get("w_margin", 0.0)) * c["margin_cost"]
        # R15 ③a：α utility 显式定价——无险情臂 -w·(α-1)² 势能罚（v6.2 尸检：
        # cross 贴限定价长期抽 α 税、挑战跑证明 0.755 能力被压回 0.576——
        # "敞开走"缺一个与 λ 对坐的显式收益方）。险情臂零贡献，门在
        # _pre_physics_step 与 tube 同状态算好；w=0 时整支路不存在（零漂移）。
        alpha_util = (alpha_util_bonus(c["alpha"], c["alpha_util_gate"],
                                       self._w_alpha_util)
                      if self._w_alpha_util > 0.0 else None)
        self._alpha_prev = c["alpha"]
        self._p_prev = c["p"]
        # 奖励八分项缓存（C14，C13 遗留：MASTER_REPORT §5 task 面板缺口）：
        # 加权后各分项的 env 均值，0 维张量（不 .item()，热路径零 GPU 同步）。
        # 只进 _step_cache 供 _EnvAdapter.step_telemetry 透出到 wandb
        # task/reward_{term} 八键；EventWriter/stats.jsonl 不读此键，零漂移。
        self._step_cache["reward_terms"] = {
            "track": track.mean().detach(),
            "backstop": backstop_pen.mean().detach(),
            "tube": tube_pen.mean().detach(),
            "progress": progress.mean().detach(),
            "smooth": smooth.mean().detach(),
            "flip": flip_pen.mean().detach(),
            "violation": viol.mean().detach(),
            "margin": margin_pen.mean().detach(),
        }
        total = (track + backstop_pen + tube_pen + progress + smooth + viol
                 + margin_pen + flip_pen)
        if alpha_util is not None:
            self._step_cache["reward_terms"]["alpha_util"] = \
                alpha_util.mean().detach()
            total = total + alpha_util
        return total

    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self._coord and self._step_cache \
                and self.cfg.coordinator.get("terminate_on_violation", True):
            # terminate_on_violation=False 仅供测量协议用（parity v3.1 去删失：
            # 违规瞬间重置会把"球报警->力发展"的因果链掐断）；训练勿动默认值
            terminated = self._step_cache["violation"].clone()
        else:
            terminated = torch.zeros_like(time_out)
        return terminated, time_out

    def _reset_idx(self, env_ids):
        if self._target_guard_enabled:
            self._target_guard.reset(env_ids)
        super()._reset_idx(env_ids)
        for arm in ARM_KEYS:
            art = self._arms[arm]
            root = art.data.default_root_state[env_ids].clone()
            root[:, :3] += self.scene.env_origins[env_ids]
            art.write_root_pose_to_sim(root[:, :7], env_ids)
            art.write_root_velocity_to_sim(root[:, 7:], env_ids)
            art.write_joint_state_to_sim(art.data.default_joint_pos[env_ids],
                                         art.data.default_joint_vel[env_ids], None, env_ids)
            self._targets[arm][env_ids] = art.data.default_joint_pos[
                env_ids][:, self._joint_idx[arm]]
        for obj in getattr(self, "_objects", {}).values():
            root = obj.data.default_root_state[env_ids].clone()
            root[:, :3] += self.scene.env_origins[env_ids]
            obj.write_root_pose_to_sim(root[:, :7], env_ids)
            obj.write_root_velocity_to_sim(root[:, 7:], env_ids)
        if self._coord:
            self._delta_src.reset(env_ids, self._gen)
            self._coupling_graph[env_ids] = False
            if hasattr(self._delta_src, "initial_positions"):
                ids, positions = self._delta_src.initial_positions(env_ids)
                if ids.numel():
                    for arm in ARM_KEYS:
                        art = self._arms[arm]
                        jp = art.data.joint_pos[ids].clone()
                        jp[:, self._joint_idx[arm]] = positions[arm]
                        art.write_joint_state_to_sim(jp, torch.zeros_like(jp), None, ids)
                        self._targets[arm][ids] = positions[arm]
            self._alpha_prev[env_ids] = 1.0
            self._p_prev[env_ids] = 0.0
            self._p_sign_prev[env_ids] = 0.0
