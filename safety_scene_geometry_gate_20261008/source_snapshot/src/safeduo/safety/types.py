"""SafeDuo 核心接口契约 v2（owner: Agent A；C 的 delta/baselines/eval 依赖此文件）。

v2 变更（2026-08-11 ROUND2_DELTAS + 监管线仲裁，改动必须在 STATUS_A.md 里 @C）：
- top-K=16 观测 -> 活跃球对集合：active_pairs (N, M, 4) + active_mask (N, M)。
  选入规则：margin < d_soft 或 TTC = margin/closing_vel < tau_ttc（closing>0），
  按 margin 升序取前 M=32；不足 M 的行零填充、pair_id=-1、mask=False。
  下游用置换不变 pooling 消费，杜绝 top-K 排名截断引起的观测跳变。
- alpha 语义：每臂进展预算（L2 必须遵守），不是对 raw delta 的预缩放。
  L2 在 [硬安全约束 + 进展预算] 内做最小改写：执行量沿命令方向的进展
  <exec_a, cmd_a>/||cmd_a|| 不得超过 alpha_a * ||cmd_a||（训练版关节空间口径；
  部署版可等价换算到笛卡尔 twist 上限）。真实现：safety/backstop.py。

不变约定：
- 臂序 ARM_KEYS=("F_L","F_R","U_L","U_R")，DoF=(7,7,6,6)，总 26。
- 批量张量第一维 = env 数 N；四元数 wxyz；本文件纯 torch 无 isaac 依赖。
- active_pairs 末维 = [dist, closing_vel, class_id, pair_id]：
    dist        球面间距（米，已减半径，可为负=穿透）
    closing_vel 接近速度（米/秒，>0 = 正在靠近）
    class_id    0=跨机(F<->U) / 1=自碰(同机器人内) / 2=对桌（3 类，仲裁定稿）
    pair_id     球对静态全局索引（编码见下）
- min_margin 四桶 {cross, self_F, self_U, table} 保留不变（SphereDistOut）。

pair_id 编码（A 定，2026-08-11）：
- 构建 SphereDistanceModule 时静态枚举：先跨机对、再自碰对、最后对桌对，
  各块内按 (sphere_i, sphere_j) 升序；pair_id = 该枚举的行号（float 存储）。
- 解码：module.pair_table[pid] -> (sphere_i, sphere_j)（对桌行第二列 = 桌索引
  0=table_F 1=table_U）；球元数据 module.arm_id/link_ord/qualified_names 反查
  臂名与 link。pair_id 依赖球分解版本，跨 run 对齐需带 spec 版本号（日志记）。
- 填充行 pair_id = -1。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch

# ---- 臂与自由度 ----

ARM_KEYS: tuple[str, ...] = ("F_L", "F_R", "U_L", "U_R")
ARM_DOF: tuple[int, ...] = (7, 7, 6, 6)
DOF_OF: dict[str, int] = dict(zip(ARM_KEYS, ARM_DOF))
TOTAL_DOF: int = sum(ARM_DOF)  # 26
ROBOT_OF: dict[str, str] = {"F_L": "F", "F_R": "F", "U_L": "U", "U_R": "U"}
ARMS_OF_ROBOT: dict[str, tuple[str, str]] = {"F": ("F_L", "F_R"), "U": ("U_L", "U_R")}

# ---- active_pairs 的 class_id 编码（3 类，监管线仲裁定稿）----

CLASS_CROSS: float = 0.0  # 跨机器人 F<->U
CLASS_SELF: float = 1.0   # 同机器人内（含双臂互碰与单臂自碰）
CLASS_TABLE: float = 2.0  # 对桌面

# 默认活跃集上限（ROUND2_DELTAS 定 M=32；env cfg 可覆盖）
MAX_ACTIVE_PAIRS: int = 32

# ---- v3 P0：观测端 pair 特征化（2026-08-12，v3_recipe_proposal §1a）----
# active_pairs 原样直灌观测曾把 raw pair_id（absmax ~2569，比物理特征大
# 3 个量级的名义标识列）喂进网络 -> 随机初始化的 p 头与 3/4 α 头在 iter0
# 即钉死 clamp 角（r2 尸检 artifacts/analysis/p_head_diagnosis/REPORT.md）。
# 观测里每行改为 [dist, closing_vel, class one-hot(3)]（5 维/行）；pair_id
# 保留在 SceneState/SphereDistOut 供 backstop 行构造与诊断反查，不进网络。
# duo_env._get_observations 与 algo/warmstart_oracle.obs_features_duo_env
# （BC 数据集）共用本函数，杜绝两侧布局漂移。
PAIR_FEATURE_DIM: int = 5

# ---- R15 ①：arm-aware pair 观测（2026-08-19，A1/A2 补诊定案）----
# v6.2 F_L 成为"约束汇"（corr(λ_cross, α_F_L)=-0.661、首败 76%）的结构性
# 偏置之一是 pair 行不带臂身份——策略无法把 cross 定价折到具体臂上。开关
# 开启时每行在 [dist, closing_vel, onehot3] 后追加两臂 4+4 one-hot（13 维/
# 行）。选 one-hot 而非 2×2 二进制：对桌行第二实体用全零表示"非臂"，不与
# F_L 的编码（二进制 00）混叠；改动面与二进制方案相同（同一函数+一个常量）。
# 臂身份查表 = SphereDistanceModule.pair_arms（(P,2) long，-1=桌/无），
# pair_id 列即静态行号。默认关（PAIR_FEATURE_DIM 路径逐位不变）。
PAIR_ARM_FEATURE_DIM: int = PAIR_FEATURE_DIM + 2 * len(ARM_KEYS)  # 13


def pair_obs_features(active_pairs: torch.Tensor,
                      active_mask: torch.Tensor) -> torch.Tensor:
    """(N, M, 4) [dist, closing_vel, class_id, pair_id] -> (N, M*5) 展平的
    [dist, closing_vel, onehot3(class)]。padding 行整行清零：约定里 padding
    的 pair_id=-1 / class_id=0，不乘 mask 会把 padding 行点亮成 cross 指示。"""
    cls = active_pairs[..., 2].long().clamp(0, 2)
    onehot = torch.nn.functional.one_hot(cls, num_classes=3).to(active_pairs.dtype)
    feat = torch.cat([active_pairs[..., :2], onehot], dim=-1)
    return (feat * active_mask.unsqueeze(-1).to(active_pairs.dtype)).flatten(1)


def pair_obs_features_arm(active_pairs: torch.Tensor,
                          active_mask: torch.Tensor,
                          pair_arms: torch.Tensor) -> torch.Tensor:
    """R15 ① arm-aware 版：(N, M, 4) -> (N, M*13) 展平的
    [dist, closing_vel, onehot3(class), onehot4(arm_a), onehot4(arm_b)]。

    pair_arms (P, 2) long = 静态 pair 表逐行两实体的臂索引（ARM_KEYS 序，
    -1 = 非臂实体如桌面 -> one-hot 全零）。行内查表键 = pair_id 列（静态
    行号）；padding 行 pair_id=-1，clamp 后会误查行 0，但与基础特征同样被
    mask 整行清零，不泄漏。前 5 维布局与 pair_obs_features 逐位一致，下游
    cost 重建（lagrangian_ppo.cost_channels_from_obs 的 [2:5] 类切片）免改。"""
    pid = active_pairs[..., 3].long().clamp_min(0)
    arms = pair_arms.to(active_pairs.device)[pid]            # (N, M, 2)
    arm_oh = (torch.nn.functional.one_hot(arms.clamp_min(0), len(ARM_KEYS))
              * (arms >= 0).unsqueeze(-1)).to(active_pairs.dtype)
    cls = active_pairs[..., 2].long().clamp(0, 2)
    onehot = torch.nn.functional.one_hot(cls, num_classes=3).to(active_pairs.dtype)
    feat = torch.cat([active_pairs[..., :2], onehot,
                      arm_oh.flatten(-2)], dim=-1)
    return (feat * active_mask.unsqueeze(-1).to(active_pairs.dtype)).flatten(1)


@dataclass
class SceneState:
    """安全层看到的场景快照（env 每 step 构造一次，全 torch 张量）。"""

    q: dict[str, torch.Tensor] = field(default_factory=dict)        # arm -> (N, dof)
    qd: dict[str, torch.Tensor] = field(default_factory=dict)       # arm -> (N, dof)
    ee_pos: dict[str, torch.Tensor] = field(default_factory=dict)   # arm -> (N, 3)
    ee_quat: dict[str, torch.Tensor] = field(default_factory=dict)  # arm -> (N, 4) wxyz
    # (N, M, 4)：[dist, closing_vel, class_id, pair_id]，margin 升序，零填充
    active_pairs: torch.Tensor | None = None
    # (N, M) bool：True = 该行是真实活跃球对
    active_mask: torch.Tensor | None = None
    # (N, 7)：机器人 U 基座在机器人 F 基座系下的位姿 pos+quat(wxyz)
    T_FU: torch.Tensor | None = None
    dt: float = 0.02

    @property
    def n_envs(self) -> int:
        return next(iter(self.q.values())).shape[0]

    @property
    def device(self) -> torch.device:
        return next(iter(self.q.values())).device


@dataclass
class DeltaCmd:
    """一步关节增量指令（操作员意图流的最小表示）。"""

    delta_q: dict[str, torch.Tensor] = field(default_factory=dict)  # arm -> (N, dof)

    def clone(self) -> "DeltaCmd":
        return DeltaCmd(delta_q={k: v.clone() for k, v in self.delta_q.items()})

    def stacked(self) -> torch.Tensor:
        # (N, 26) 按 ARM_KEYS 顺序拼接，供网络/日志用
        return torch.cat([self.delta_q[k] for k in ARM_KEYS], dim=-1)


def zeros_delta(n_envs: int, device: torch.device | str = "cpu") -> DeltaCmd:
    return DeltaCmd(
        delta_q={k: torch.zeros(n_envs, DOF_OF[k], device=device) for k in ARM_KEYS}
    )


def split_stacked(x: torch.Tensor) -> DeltaCmd:
    # (N, 26) -> DeltaCmd，stacked() 的逆
    out, i = {}, 0
    for k in ARM_KEYS:
        out[k] = x[:, i : i + DOF_OF[k]]
        i += DOF_OF[k]
    return DeltaCmd(delta_q=out)


class DeltaSource(ABC):
    """意图流生成器接口。A 只定义接口与冒烟实现；L1-L4 真实现归 Agent C。"""

    @abstractmethod
    def reset(self, env_ids: torch.Tensor, generator: torch.Generator | None = None) -> None:
        """重置指定 env 的内部状态；generator 传入则后续采样用它（保证可复现）。"""

    @abstractmethod
    def sample(self, state: SceneState) -> DeltaCmd:
        """按当前场景产出本步 delta（形状约束见 DeltaCmd）。"""


def backstop_project(
    delta: DeltaCmd,
    state: SceneState,
    alpha: torch.Tensor,
    p: torch.Tensor,
) -> tuple[DeltaCmd, torch.Tensor]:
    """L2 兜底接口签名（形状契约桩——真实现在 safety/backstop.py）。

    参数:
        delta: 原始意图 delta。
        alpha: (N, 4) 每臂进展预算 in [0,1]（v2 语义：L2 内部约束，非预缩放）。
        p:     (N,) 让行标量 in [-1,1]，+1=U 让 F；跨机行闭合速度预算按
               (1+p)/2 : (1-p)/2 分配给 F/U 两侧。
    返回:
        (执行 DeltaCmd, backstop_active (N,4) bool)。

    真实现 = safety/backstop.py::VelocityDamperBackstop（批量闭式顺序投影，
    需要 GeometryProvider 约束行——SceneState 本身不含梯度，故本函数无法做
    真投影）。此桩只保证形状/直通，别在训练管线里调它。
    """
    n = state.n_envs
    active = torch.zeros(n, len(ARM_KEYS), dtype=torch.bool, device=state.device)
    return delta.clone(), active
