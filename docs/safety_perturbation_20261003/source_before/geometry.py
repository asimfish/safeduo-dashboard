"""Isaac 几何行 provider：活跃球对 -> ConstraintRows（J 来自 PhysX 雅可比）。

行约定 = baselines/base.py（d_dot = J_F qd_F + J_U qd_U，J 单位 m/rad）：
  球对行：margin = ||c_i - c_j|| - r_i - r_j
    dmargin/dq_arm(i) = n_hat^T J_ci，dmargin/dq_arm(j) = -n_hat^T J_cj
  对桌行：margin = boxSDF(c_i) - r_i，dmargin/dq = grad_sdf^T J_ci
  球心雅可比：J_c = J_lin + J_ang x r_off。
  PhysX 的 J_lin 在部件质心处，故 r_off = 球心 - body COM，均为世界系。

PhysX 雅可比（fixed base）形状 (N, n_links-1, 6, n_dof)，body b 的行 = b-1；
b=0（固定基座 link，如 fr3_link0）无行，其 J 恒零（正确：基座不动）。
"""

from __future__ import annotations

import torch

from safeduo.baselines.base import ConstraintRows
from safeduo.safety.sphere_distance import SphereDistanceModule, SphereDistOut
from safeduo.safety.types import ARM_KEYS, ROBOT_OF

_ROBOT_ARMS = {"F": ("F_L", "F_R"), "U": ("U_L", "U_R")}


class IsaacGeometryProvider:
    """绑定 env 的四个 articulation + 球模块，按活跃集产出约束行。"""

    def __init__(self, arms: dict, joint_idx: dict, module: SphereDistanceModule,
                 device: torch.device, jacobian_reference: str = 'com'):
        if jacobian_reference not in ('com', 'link'):
            raise ValueError('jacobian_reference must be com or legacy link')
        self.jacobian_reference = jacobian_reference
        self.arms = arms                    # arm -> Articulation
        self.joint_idx = joint_idx          # arm -> (dof,) long 关节列索引
        self.m = module
        self.device = device
        # 每球所属 arm 的局部信息：body 索引（articulation 内）、arm 内列偏移
        self.sphere_body = torch.zeros(module.n_spheres, dtype=torch.long, device=device)
        for arm in ARM_KEYS:
            sl = module._arm_slices[arm]
            self.sphere_body[sl] = module._body_idx[arm]
        self.sphere_arm = module.arm_id      # (S,) 0..3
        # robot 内堆叠列偏移：F=[F_L|F_R]，U=[U_L|U_R]
        self.col_off = {}
        for r, (a1, a2) in _ROBOT_ARMS.items():
            self.col_off[a1] = 0
            self.col_off[a2] = len(self.joint_idx[a1])
        self.dof_r = {r: len(self.joint_idx[a1]) + len(self.joint_idx[a2])
                      for r, (a1, a2) in _ROBOT_ARMS.items()}

    def _sphere_jacobian(self, arm: str, body_rows: torch.Tensor,
                         r_off: torch.Tensor) -> torch.Tensor:
        """body_rows:(N,K) 该臂 body 索引，r_off:(N,K,3) -> J_c (N,K,3,dof_arm)。"""
        art = self.arms[arm]
        jac = art.root_physx_view.get_jacobians()          # (N, L-1, 6, D)
        cols = self.joint_idx[arm]
        jac = jac[..., cols]                                # (N, L-1, 6, dof)
        n, k = body_rows.shape
        ar = torch.arange(n, device=self.device).unsqueeze(-1).expand(n, k)
        row = (body_rows - 1).clamp(0, jac.shape[1] - 1)    # b=0 无行，取 0 后置零
        j = jac[ar, row]                                    # (N,K,6,dof)
        j_lin, j_ang = j[:, :, :3], j[:, :, 3:]
        j_c = j_lin + torch.cross(j_ang, r_off.unsqueeze(-1).expand_as(j_ang), dim=2)
        return torch.where((body_rows > 0)[..., None, None], j_c, torch.zeros_like(j_c))

    def rows_from(self, out: SphereDistOut, body_pos: dict) -> ConstraintRows:
        """out = SphereDistanceModule.compute(...)（同一步、need cache centers）。"""
        m = self.m
        assert m.last_centers is not None, "compute() 先行"
        idx = out.active_idx.clamp_min(0)                   # (N,M)
        n, M = idx.shape
        valid = out.active_mask
        pt = m.pair_table[idx]                              # (N,M,2)
        is_table = idx >= m._slice_table.start
        i_sph = pt[..., 0]
        j_sph = torch.where(is_table, torch.zeros_like(pt[..., 1]), pt[..., 1])
        centers = m.last_centers                            # (N,S,3)
        ci = centers.gather(1, i_sph.unsqueeze(-1).expand(n, M, 3))
        cj = centers.gather(1, j_sph.unsqueeze(-1).expand(n, M, 3))
        nrm = ci - cj
        nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        # 对桌行方向 = 缓存的 SDF 梯度
        tab_rel = (idx - m._slice_table.start).clamp_min(0)
        grad = m.last_table_grad
        if grad is not None and grad.shape[1] > 0:
            g_tab = grad.gather(1, tab_rel.unsqueeze(-1).expand(n, M, 3))
            nrm = torch.where(is_table.unsqueeze(-1), g_tab, nrm)
        J = {r: torch.zeros(n, M, self.dof_r[r], device=self.device) for r in ("F", "U")}
        arm_mask = torch.zeros(n, M, 4, dtype=torch.bool, device=self.device)
        for ai, arm in enumerate(ARM_KEYS):
            r = ROBOT_OF[arm]
            off = self.col_off[arm]
            dof = len(self.joint_idx[arm])
            # `link` is retained solely to reproduce frozen legacy campaigns.
            bp = (self.arms[arm].data.body_com_pos_w if self.jacobian_reference == 'com'
                  else body_pos[arm])                       # (N,B,3)
            for side, sph, sign in (("i", i_sph, 1.0), ("j", j_sph, -1.0)):
                on_arm = (self.sphere_arm[sph] == ai) & valid
                if side == "j":
                    on_arm = on_arm & ~is_table             # 桌行无 j 侧
                if not on_arm.any():
                    continue
                # 非本臂行的球索引指向别的 articulation，gather 前必须清零
                # （数值随后被 on_arm 掩码清掉，不影响结果，只防越界 assert）。
                body = torch.where(on_arm, self.sphere_body[sph],
                                   torch.zeros_like(sph))   # (N,M)
                r_off = ci if side == "i" else cj
                bpos = bp.gather(1, body.unsqueeze(-1).expand(n, M, 3))
                jc = self._sphere_jacobian(arm, body, r_off - bpos)  # (N,M,3,dof)
                contrib = sign * torch.einsum("nmc,nmcd->nmd", nrm, jc)
                J[r][..., off:off + dof] += torch.where(
                    on_arm.unsqueeze(-1), contrib, torch.zeros_like(contrib))
                arm_mask[..., ai] |= on_arm
        return ConstraintRows(
            d=out.active_pairs[..., 0],
            J=J,
            cls=out.active_pairs[..., 2],
            arm_mask=arm_mask,
            valid=valid,
            d_min=out.active_dmin,
        )
