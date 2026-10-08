"""R25 l1_full 全域覆盖随机流:卦限 LRU 采样 + 6D 姿态意图 + 变速段。

为什么要有这条流(owner 2026-08-27 审计,主报告 §9 R25):l1_ws 被定性为
"原地转手腕",三重根因——
  1. mapper 只有 3 维位置雅可比,姿态从未被指令,OU 噪声在位置零空间自由
     积分(v7 档 60s 腕漂 σ≈±49°,漂移不是意图);
  2. v7 漫游盒是中线窄条(x 宽 0.215 m,体积 ≈ 单臂可达空间 8-10%),四臂
     几乎不离开 home 附近,安全模型只见过极小状态空间;
  3. 段长 2-5 s 远大于带内转移时间 ~1.1 s,每段 60-70% 时间在 waypoint 上
     悬停。
盲区清单:own-side 半区(|x|∈[0.2,0.95])、姿态大变化、全伸壳层
r∈[0.80,0.855]、深度互穿(X_PEN 仅 ±0.015)、同排自碰流量、速度谱单点
(恒 0.45 m/s)。

方案(与主报告 R25 定案一一对应,l1_ws 一字不动 = 历史评测回归锚):
  * 全域盒分层采样:l1_workspace_v7.full_boxes_v7(reach 推导的大盒)按
    2x2x2 卦限做 LRU 轮访(最久未访优先,平局随机)保证系统性扫全域;
    p_band=0.3 回访 l1_ws 冲突带盒(跨机流量密度保底),p_deep=0.1 采
    深互穿板(x 越过 X_PEN 直至对侧全伸)。
  * 到达触发换段:|FK 法兰 - waypoint| < arrive_eps(0.03 m)即进驻留
    U(0.3,1) s,驻留完换点——不再定时悬停;不可达/慢档长距由
    seg_timeout U(5,9) s 兜底(95 分位转移 ~1.2 m / 0.15 m/s ≈ 8 s)。
  * 6D 姿态意图:每段采相对旋转(轴 ~ S² 均匀,角 ~ U(30°,120°)),目标
    R_tgt = R_delta @ R_now;EEBackend6DV7 把角速度雅可比(= FK 已有的
    各关节世界轴 z_j,零额外 FK 成本)拼进 6 行加权阻尼伪逆,姿态行权重
    w_ori=0.4——姿态从"零空间漂移"变成"被指令的意图"。
  * 变速:每段 EE 线速度三档 {0.15,0.45,0.80} m/s @ {0.3,0.5,0.2}
    (0.45 = l1_ws 恒速档),角速度帽随档位等比。
  * 不做避碰:与 l1_ws 同哲学,随机流故意产生碰撞风险,危险样本是安全层
    的训练素材;安全兜底仍由 env 侧软限位钳制/backstop 负责。

waypoint 载荷 13 维 = [pos(3), R_tgt 展平(9), 速度档(1)]:速度/姿态目标随
waypoint 一起存,引擎 torch.where(stale,new,old) 的换段逻辑天然保证"段内
恒速恒姿态目标、换段整体重抽",reset 的 NaN 标脏也整行生效——L1RandomDelta
引擎零改动。

位置误差/到达判定用本模块 FK 后端的法兰位置(p_fl),与采样目标同一坐标
链,整条回路零系统差(l1_ws 用 state.ee_pos 伺服,报告 body 与解析法兰的
安装偏置会吃掉 arrive_eps 量级的精度;calib_report 仍照常记录漂移哨兵)。

消费(全部并列新增,不改任何现有流的行为):
  * 评测:endurance_eval --flow l1_full --geometry v7;
  * 训练直连:duo_env coordinator.delta_source: l1_full;
  * 训练课程:configs/delta_curriculum_v7_cov.yaml(ConflictMixSource 的
    l1_full 族,l1 0.4 拆 l1_ws 0.15 + l1_full 0.25)。

验收口径(主报告 R25,GPU 空闲后跑):60 s 单臂卦限 8/8、EE x 极差
≥0.5 m、y 极差 ≥0.9 m、姿态指令/漂移 >3:1、带内驻留 ≥15%、
(课程侧)tube_fraction 相对 v7 配方降幅 <30%。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch

from safeduo.delta._contract_stub import ARM_KEYS, SceneState
from safeduo.delta.l1_random import IntentMapper, L1Params, L1RandomDelta
from safeduo.delta.l1_workspace_v7 import (
    V7_FULL_REACH,
    V7_X_PEN,
    EnvEEBackendV7,
    full_boxes_v7,
    roam_boxes_v7,
)
from safeduo.delta.l2_env_source import RealScenePoses


# --------------------------------------------------------------------------
# 参数(默认值即 R25 设计值;yaml/coordinator 可逐字段覆盖)
# --------------------------------------------------------------------------

@dataclass
class L1FullParams(L1Params):
    """l1_full 参数。父类的 seg_dur/pause_prob/pause_dur 不再被消费
    (段机换成到达触发);amp_max/ou_* /bandwidth_hz/smooth_hz 语义不变。"""

    # ---- 段机(到达触发) ----
    arrive_eps: float = 0.03        # 位置到达阈值(m,FK 法兰系,自洽零漂)
    dwell: tuple = (0.3, 1.0)       # 到达后驻留 U(s)——替代定时段的被动悬停
    seg_timeout: tuple = (4.0, 7.0) # 未到达强制换点(不可达/伪逆局部困住的
                                    # 活锁保险;开环冒烟实测 FR3 肘伸直折叠
                                    # 附近会困住,上限压到 7s 限定垃圾时长)
    # ---- waypoint 三路混合 ----
    p_band: float = 0.3             # 回访 l1_ws 冲突带盒的概率
    p_deep: float = 0.1             # 深互穿板概率(x 越过 X_PEN 直至全伸)
    pair_period: int = 4            # 每 4 个段周期触发一次成对交集路线
    deep_dy: float = 0.25           # 深穿板 y 半宽(贴基座 y,方向近 -x 保深度)
    deep_z: tuple = (0.85, 1.15)    # 深穿板 z 带(近桌,对手手部工作高度)
    # ---- 6D 姿态意图 ----
    ori_angle: tuple = (math.pi / 6.0, 2.0 * math.pi / 3.0)  # 相对转角 U(30°,120°)
    w_ori: float = 0.4              # 6 行加权最小二乘的姿态行权重
    omega_base: float = 1.0         # speed_ref 档的角速度帽(rad/s)
    speed_ref: float = 0.45         # 角速度帽的归一锚(= l1_ws 恒速档)
    # ---- 零空间姿态居中(开环冒烟发现的修复) ----
    # 7-DoF F 臂的 6D 任务留 1 维肘部 swivel 零空间;深穿/全伸目标会把肘
    # 推进 q4≈-0.07 的伸直折叠,阻尼伪逆在折叠面局部困住、法兰甩到基座
    # 后方(冒烟 F_R 10% 步出盒)。以 k_null*(出生位形-q) 在零空间回拉
    # 肘 swivel,不触碰 6D 任务行;6-DoF U 臂零空间≈空,天然无效。
    # 增益扫描(开环 60s x4env):0.4 -> F_R 6.5% 出盒,0.8 -> 1.1%,
    # 1.2 -> 全臂 <=0.03%,卦限/带内驻留不受影响,故默认 1.2。
    k_null: float = 1.2             # 零空间回拉增益(rad/s;0 = 关)
    # ---- 变速 ----
    speed_tiers: tuple = (0.15, 0.45, 0.80)  # EE 线速度三档(m/s)
    speed_probs: tuple = (0.3, 0.5, 0.2)     # 三档概率
    # ---- 几何 ----
    r_margin: float = 0.01          # 全伸内缩(m):0.855/0.850 -> 0.845/0.840
    # 父类字段改默认:姿态已被直接指令,OU 只留少量抖动质感(审计根因①
    # 的"OU 在零空间积分"不再是姿态变化的唯一来源)
    ou_mix: float = 0.15
    ou_sigma: float = 0.02


# --------------------------------------------------------------------------
# 零空间姿态参考(v7 出生位形)
# --------------------------------------------------------------------------

_FR3_JOINT_NAMES = tuple(f"fr3_joint{i}" for i in range(1, 8))
_UR_JOINT_NAMES = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                   "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")


def _posture_ref(env_yaml: str) -> "dict | None":
    """env yaml 的 init_qpos(S4 出生位重搜产物)-> 逐臂零空间参考位形。

    出生位形本身就是"肘部折好、四臂无碰"的搜索产物,用它当 swivel 回拉
    锚点即可;yaml 无 init_qpos(非 v7 配置)时返回 None = 偏置关闭。
    """
    from safeduo.configs import load_config

    iq = (load_config(env_yaml) or {}).get("init_qpos") or {}
    if "franka" not in iq or "ur" not in iq:
        return None
    f = torch.tensor([float(iq["franka"][n]) for n in _FR3_JOINT_NAMES])
    u = torch.tensor([float(iq["ur"][n]) for n in _UR_JOINT_NAMES])
    return {"F_L": f, "F_R": f.clone(), "U_L": u, "U_R": u.clone()}


# --------------------------------------------------------------------------
# SO(3) 小工具(纯 torch,批量)
# --------------------------------------------------------------------------

def _rodrigues(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """轴角 -> 旋转矩阵。axis (N,3) 需单位化,angle (N,) 弧度。"""
    n = axis.shape[0]
    K = torch.zeros(n, 3, 3, dtype=axis.dtype, device=axis.device)
    ax, ay, az = axis[:, 0], axis[:, 1], axis[:, 2]
    K[:, 0, 1], K[:, 0, 2] = -az, ay
    K[:, 1, 0], K[:, 1, 2] = az, -ax
    K[:, 2, 0], K[:, 2, 1] = -ay, ax
    s = torch.sin(angle).view(-1, 1, 1)
    c = torch.cos(angle).view(-1, 1, 1)
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device).expand_as(K)
    return eye + s * K + (1.0 - c) * (K @ K)


def _so3_log(R: torch.Tensor) -> torch.Tensor:
    """旋转矩阵 -> 世界系旋转向量。θ→π 时 vee(R-R^T)→0,输出退化为 0
    向量:意图流可容忍(该段最终由 seg_timeout 兜底),不做完整 π 分支。"""
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    ang = torch.acos(cos)
    vee = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                       R[..., 0, 2] - R[..., 2, 0],
                       R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    return vee * (ang / (2.0 * torch.sin(ang))).unsqueeze(-1)


# --------------------------------------------------------------------------
# FK 后端:v7 链 + 旋转雅可比缓存
# --------------------------------------------------------------------------

class EEBackend6DV7(EnvEEBackendV7):
    """EnvEEBackendV7 + 角速度雅可比/法兰姿态缓存(审计根因①的修复前提)。

    零额外 FK 成本:ArmKinematics.fk 已返回各关节世界系轴 z_j(fko["z"]),
    转动链的角速度雅可比列恰好就是 z_j。缓存同时保留父类 3 行契约
    (J/JJt_inv,阻尼 1e-2 逐位一致),父类消费者行为不变。
    """

    def __init__(self, poses: RealScenePoses,
                 device: "str | torch.device" = "cpu",
                 w_ori: float = 0.4, damping: float = 1e-2):
        super().__init__(poses, device=device)
        self.w_ori = float(w_ori)
        self.damping = float(damping)

    def refresh(self, state: SceneState, token: int) -> None:
        if token == self._cache_token:
            return
        if not self.calibrated:
            self.calibrate(state)
        for arm in ARM_KEYS:
            kin = self.kin[arm[0]]
            pos, yaw = self.poses.base_pos[arm], self.poses.base_yaw[arm]
            fko = kin.fk(state.q[arm], pos, yaw)
            p = fko["t_flange"]
            fidx = torch.tensor([kin.dof], dtype=torch.long, device=p.device)
            Jp = kin.point_jacobian(fko, p.unsqueeze(1), fidx)[:, 0]  # (N,3,dof)
            Jw = fko["z"].transpose(1, 2)                             # (N,3,dof)
            # 加权最小二乘的一致写法:J' = [Jp; w*Jw],rhs 侧同乘 w(见 map)
            J6 = torch.cat([Jp, self.w_ori * Jw], dim=1)
            JJt = Jp @ Jp.transpose(-1, -2)
            eye3 = torch.eye(3, device=p.device, dtype=p.dtype).expand_as(JJt)
            J6Jt = J6 @ J6.transpose(-1, -2)
            eye6 = torch.eye(6, device=p.device, dtype=p.dtype).expand_as(J6Jt)
            self._cache[arm] = {
                "J": Jp, "JJt_inv": torch.linalg.inv(JJt + 1e-2 * eye3),
                "J6": J6,
                "J6Jt_inv": torch.linalg.inv(J6Jt + self.damping * eye6),
                "R_fl": fko["R_flange"], "p_fl": p,
            }
        self._cache_token = token


# --------------------------------------------------------------------------
# mapper:全域 6D waypoint
# --------------------------------------------------------------------------

class FullCoverageMapper(IntentMapper):
    """waypoint = [pos(3), R_tgt 展平(9), 速度档(1)] 共 13 维的 6D mapper。

    LRU 卦限账本按 (env, arm) 记"上次抽中该卦限的段序号",argmin = 最久
    未访;只有真正换段且走覆盖路线(非 band/deep)的 env 才入账,band/deep
    回访不清 LRU 欠账——60 s 内 8/8 卦限的验收由此保证。
    """

    WP_DIM = 13

    def __init__(self, backend: EEBackend6DV7, params: L1FullParams,
                 n_envs: int, device: "str | torch.device" = "cpu",
                 q_ref: "dict | None" = None):
        self.backend = backend
        self.p = params
        self.q_ref = q_ref          # 零空间回拉锚(None = 偏置关闭)
        self.device = torch.device(device)
        poses = backend.poses
        self.boxes_full = full_boxes_v7(poses, r_margin=params.r_margin)
        self.boxes_band = roam_boxes_v7(poses)   # l1_ws 冲突带盒,原样回访
        self.boxes_deep = self._deep_slabs(poses)
        # A conservative common target slab.  It is deliberately smaller than
        # the task AABB: both selected arms are asked to enter the same narrow
        # centre region, while radial reach clamping remains the final guard.
        self.box_pair = (
            torch.tensor([-0.06, -0.22, 0.86], dtype=torch.float32),
            torch.tensor([0.06, 0.22, 1.20], dtype=torch.float32),
        )
        self._env_index = torch.arange(n_envs, device=self.device)
        # Cross-row pairs share a real centre workspace.  Assigning one pair
        # per environment at reset makes the pair route coordinated even
        # when the four arms renew waypoints on different wall-clock steps.
        self._cross_pairs = ((0, 2), (0, 3), (1, 2), (1, 3))
        self._pair_route = self._env_index % len(self._cross_pairs)
        self._renew_no = {a: torch.zeros(n_envs, dtype=torch.long,
                                         device=self.device)
                          for a in ARM_KEYS}
        self._base = {a: torch.tensor(poses.base_pos[a], dtype=torch.float32)
                      for a in ARM_KEYS}
        self._r_max = {a: float(V7_FULL_REACH[a[0]]) - params.r_margin
                       for a in ARM_KEYS}
        self._lru = {a: torch.zeros(n_envs, 8, device=self.device)
                     for a in ARM_KEYS}
        self._seg_no = {a: 0.0 for a in ARM_KEYS}
        self._tiers = torch.tensor(params.speed_tiers, dtype=torch.float32,
                                   device=self.device)
        self._tier_w = torch.tensor(params.speed_probs, dtype=torch.float32,
                                    device=self.device)

    def _deep_slabs(self, poses: RealScenePoses) -> dict:
        """深互穿板:x 从 X_PEN 线一直到对侧全伸;y 贴基座 ±deep_dy、z 压
        在近桌带——方向近 -x,径向钳位后深度得以保留(斜向大偏置会被钳
        位拉回浅区,故不取全盒 y/z)。"""
        p = self.p
        boxes = {}
        for arm in ARM_KEYS:
            bx, by = poses.base_pos[arm][0], poses.base_pos[arm][1]
            r = V7_FULL_REACH[arm[0]] - p.r_margin
            if bx > 0:                         # F 排:深穿 = x 负向
                x_lo, x_hi = bx - r, -V7_X_PEN
            else:                              # U 排:深穿 = x 正向
                x_lo, x_hi = V7_X_PEN, bx + r
            boxes[arm] = (
                torch.tensor([x_lo, by - p.deep_dy, p.deep_z[0]],
                             dtype=torch.float32),
                torch.tensor([x_hi, by + p.deep_dy, p.deep_z[1]],
                             dtype=torch.float32),
            )
        return boxes

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        """episode 重置:该 env 的卦限账本清零,新一轮轮访顺序由平局噪声
        重新随机化。"""
        ids = env_ids.to(self.device)
        for a in ARM_KEYS:
            self._lru[a][ids] = 0.0
            self._renew_no[a][ids] = 0

    def sample_waypoint(self, arm, state, generator, renew_mask=None):
        p = self.p
        n = state.ee_pos[arm].shape[0]
        dev = state.ee_pos[arm].device
        mask = (torch.ones(n, dtype=torch.bool, device=dev)
                if renew_mask is None else renew_mask)
        lo_f, hi_f = (t.to(dev) for t in self.boxes_full[arm])
        lo_b, hi_b = (t.to(dev) for t in self.boxes_band[arm])
        lo_d, hi_d = (t.to(dev) for t in self.boxes_deep[arm])
        # 1) 覆盖路线:LRU 卦限(平局加噪随机化)内均匀采样
        mid = 0.5 * (lo_f + hi_f)
        tie = torch.rand(n, 8, device=dev, generator=generator)
        oct_idx = (self._lru[arm].to(dev) + tie).argmin(dim=-1)          # (n,)
        bits = ((oct_idx.unsqueeze(-1) >> torch.arange(3, device=dev)) & 1
                ).to(lo_f.dtype)                                         # (n,3)
        half = mid - lo_f
        o_lo = lo_f + bits * half
        u = torch.rand(n, 3, device=dev, generator=generator)
        wp_cov = o_lo + u * half
        # 2) band/deep 路线:各自盒内均匀
        ub = torch.rand(n, 3, device=dev, generator=generator)
        wp_band = lo_b + ub * (hi_b - lo_b)
        ud = torch.rand(n, 3, device=dev, generator=generator)
        wp_deep = lo_d + ud * (hi_d - lo_d)
        lp, hp = (t.to(dev) for t in self.box_pair)
        up = torch.rand(n, 3, device=dev, generator=generator)
        wp_pair = lp + up * (hp - lp)
        route = torch.rand(n, device=dev, generator=generator)
        # Pair route is coordinated without changing the DeltaSource contract:
        # the selected pair is fixed per environment and both members take
        # the pair route on their first renewal and then every pair_period-th
        # renewal.  This remains synchronized despite per-arm renew timing.
        arm_idx = ARM_KEYS.index(arm)
        pair_ids = torch.tensor(
            [k for k, pair in enumerate(self._cross_pairs) if arm_idx in pair],
            device=dev, dtype=torch.long)
        slot = self._pair_route.to(dev)
        renew_no = self._renew_no[arm].to(dev)
        take_pair = mask & (renew_no % max(1, int(p.pair_period)) == 0)
        take_pair &= torch.isin(slot, pair_ids)
        take_band = (~take_pair) & (route < p.p_band)
        take_deep = (~take_pair) & (~take_band) & (route < p.p_band + p.p_deep)
        pos = torch.where(take_pair.unsqueeze(-1), wp_pair,
                          torch.where(take_band.unsqueeze(-1), wp_band,
                                      torch.where(take_deep.unsqueeze(-1), wp_deep, wp_cov)))
        # 径向钳位到 (r_full - r_margin) 球,方向保留:盒外角点落全伸壳层
        base = self._base[arm].to(dev)
        v = pos - base
        r = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        pos = base + v * torch.clamp(self._r_max[arm] / r, max=1.0)
        # 3) 姿态意图:R_tgt = Rot(轴 ~ S² 均匀, 角 ~ U(ori_angle)) @ R_now
        R_now = self.backend.jac(arm)["R_fl"]
        axis = torch.randn(n, 3, device=dev, generator=generator)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        ang = p.ori_angle[0] + torch.rand(n, device=dev, generator=generator) \
            * (p.ori_angle[1] - p.ori_angle[0])
        R_tgt = _rodrigues(axis, ang) @ R_now
        # 4) 速度档
        tier_idx = torch.multinomial(
            self._tier_w.to(dev).expand(n, -1), 1, replacement=True,
            generator=generator).squeeze(-1)
        tier = self._tiers.to(dev)[tier_idx]
        # 5) 卦限账本:仅"真正换段且走覆盖路线"的 env 入账
        booked = mask & ~take_band & ~take_deep & ~take_pair
        self._renew_no[arm] += mask.to(self._renew_no[arm].dtype)
        self._lru[arm][booked, oct_idx[booked]] = self._seg_no[arm]
        return torch.cat([pos, R_tgt.reshape(n, 9), tier.unsqueeze(-1)],
                         dim=-1)

    def map(self, arm, state, tgt, dt, speed=None):
        p = self.p
        c = self.backend.jac(arm)
        pos_t = tgt[:, 0:3]
        R_t = tgt[:, 3:12].reshape(-1, 3, 3)
        tier = tgt[:, 12:13]                                   # (N,1) m/s
        # 位置误差用与采样/到达同源的 FK 法兰(p_fl),整条回路零系统差
        err = pos_t - c["p_fl"]
        dist = err.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        v_ee = err / dist * torch.minimum(dist / dt, tier)
        # 姿态误差:世界系 log(R_tgt R_now^T);角速度帽随速度档等比
        w_vec = _so3_log(R_t @ c["R_fl"].transpose(-1, -2))
        wn = w_vec.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        w_cap = p.omega_base * tier / p.speed_ref
        w_ee = w_vec / wn * torch.minimum(wn / dt, w_cap)
        # 加权阻尼伪逆:J'=[Jp; w*Jw] 已在 backend 缓存,rhs 同乘 w 保持
        # min ||W(J qd - u)||² 的一致加权
        rhs = torch.cat([v_ee, p.w_ori * w_ee], dim=-1).unsqueeze(-1)  # (N,6,1)
        qd = (c["J6"].transpose(-1, -2) @ (c["J6Jt_inv"] @ rhs)).squeeze(-1)
        if p.k_null > 0.0 and self.q_ref is not None:
            # 零空间姿态居中:ns 先投影掉 6D 任务分量再叠加——只动肘部
            # swivel 之类的冗余方向,位置/姿态伺服不受影响(见参数注释)
            ns = p.k_null * (self.q_ref[arm].to(tgt.device) - state.q[arm])
            task = (c["J6"].transpose(-1, -2)
                    @ (c["J6Jt_inv"] @ (c["J6"] @ ns.unsqueeze(-1))))
            qd = qd + (ns - task.squeeze(-1))
        return qd * dt


# --------------------------------------------------------------------------
# 源:到达触发段机
# --------------------------------------------------------------------------

class L1FullDelta(L1RandomDelta):
    """R25 l1_full 源:到达触发换段 + 全域 6D mapper + 自持 v7 FK 后端。

    对外契约与 l1_ws(WorkspaceRoamDeltaV7)完全一致:reset(env_ids,
    generator) / sample(state) -> DeltaCmd,无限时域,随机数全走 generator。
    差异只在段机(定时 -> 到达+驻留+超时)与 waypoint 语义(3 维位置 ->
    13 维位姿+速度)。父类的 OU/混合/amp 钳制/EMA 路径原样复用。
    """

    def __init__(self, n_envs: int, params: "L1FullParams | None" = None,
                 device: "str | torch.device" = "cpu",
                 env_yaml: str = "duo_env_v7.yaml",
                 backend: "EEBackend6DV7 | None" = None,
                 mapper: "FullCoverageMapper | None" = None,
                 arm_keys: tuple = ARM_KEYS,
                 dof_of: "dict | None" = None):
        p = params or L1FullParams()
        if mapper is None:
            backend = backend or EEBackend6DV7(
                RealScenePoses(env_yaml), device=device, w_ori=p.w_ori)
            mapper = FullCoverageMapper(backend, p, n_envs, device=device,
                                        q_ref=_posture_ref(env_yaml))
        self._backend = backend
        self._token = 0
        super().__init__(n_envs, params=p, mapper=mapper, device=device,
                         arm_keys=arm_keys, dof_of=dof_of)

    def reset(self, env_ids: torch.Tensor,
              generator: "torch.Generator | None" = None) -> None:
        super().reset(env_ids, generator)
        if isinstance(self.mapper, FullCoverageMapper):
            self.mapper.reset_envs(env_ids)

    def _advance_segments(self, arm: str, state: SceneState, dt: float) -> None:
        """到达触发段机(整体替换父类的定时段机)。

        状态转移(_paused 复用为"驻留中"):
          MOVE --(|p_fl-wp|<arrive_eps)--> DWELL(t=U(dwell))
          MOVE --(t 耗尽,未到达)------> MOVE(换新点,t=U(seg_timeout))
          DWELL --(t 耗尽)------------> MOVE(换新点)
          reset 标脏(NaN)-------------> MOVE(换新点)
        """
        p = self.p
        t = self._t_left[arm] - dt
        wp = self._wp[arm]
        pos_now = (self._backend.jac(arm)["p_fl"] if self._backend is not None
                   else state.ee_pos[arm])
        n, dev = pos_now.shape[0], pos_now.device
        if wp is None:
            stale = torch.ones(n, dtype=torch.bool, device=dev)
            pos_err = torch.full((n,), float("inf"), device=dev)
        else:
            stale = torch.isnan(wp).any(dim=-1)
            pos_err = (pos_now - wp[:, :3]).norm(dim=-1)
            pos_err = torch.where(stale, torch.full_like(pos_err, float("inf")),
                                  pos_err)
        paused = self._paused[arm]
        arrived = (~paused) & (~stale) & (pos_err < p.arrive_eps)
        expired = t <= 0.0
        renew = stale | (paused & expired) | ((~paused) & expired & (~arrived))
        new_paused = (paused | arrived) & (~renew)
        dur_dwell = p.dwell[0] + self._rand(self.n) * (p.dwell[1] - p.dwell[0])
        dur_move = (p.seg_timeout[0]
                    + self._rand(self.n) * (p.seg_timeout[1] - p.seg_timeout[0]))
        t = torch.where(arrived & (~renew), dur_dwell, t)
        t = torch.where(renew, dur_move, t)
        if bool(renew.any()):
            wp_new = self.mapper.sample_waypoint(arm, state, self.gen,
                                                 renew_mask=renew)
            wp = wp_new if wp is None else torch.where(renew.unsqueeze(-1),
                                                       wp_new, wp)
        self._wp[arm] = wp
        self._paused[arm] = new_paused
        self._t_left[arm] = t

    def sample(self, state: SceneState):
        self._token += 1
        if self._backend is not None:
            self._backend.refresh(state, self._token)
        return super().sample(state)


# --------------------------------------------------------------------------
# 构造入口(duo_env / endurance_eval / ConflictMixSource 共用)
# --------------------------------------------------------------------------

def make_l1_full_v7(n_envs: int, amp_max: float = 0.015,
                    device: "str | torch.device" = "cpu",
                    cfg: "dict | None" = None,
                    env_yaml: "str | None" = None) -> L1FullDelta:
    """按配置构造 l1_full 源。

    cfg 兼容两种来源(键都可缺省):
      * duo_env 的 coordinator 段整体:读 delta_env_yaml 与 l1_full 子 dict
        (子 dict = L1FullParams 字段覆盖);
      * 课程/评测侧手工 dict:顶层即 L1FullParams 字段覆盖。
    显式 env_yaml 参数优先于 cfg 里的 delta_env_yaml;amp_max 恒以形参为准
    (评测按窗口幅度扫,课程按 amp_stages 事后改 p.amp_max)。
    """
    cfg = dict(cfg or {})
    names = {f.name for f in fields(L1FullParams)}
    sub = cfg.get("l1_full")
    over = (dict(sub) if isinstance(sub, dict)
            else {k: v for k, v in cfg.items() if k in names})
    over.pop("weight", None)          # 课程族权重键,不是参数
    over = {k: (tuple(v) if isinstance(v, list) else v)
            for k, v in over.items() if k in names}
    over["amp_max"] = float(amp_max)
    yaml_name = env_yaml or str(cfg.get("delta_env_yaml", "duo_env_v7.yaml"))
    return L1FullDelta(n_envs, params=L1FullParams(**over), device=device,
                       env_yaml=yaml_name)
