"""S4 任务 2:v7 reach 盒重标的 FK 采样标定(本地 CPU,一次性证据脚本)。

方法(任务书口径「每臂 FK 采样可达域与场景 AABB 求交」):
- 每臂型均匀采 N 组限位内关节位形(FR3 用 birth_pose_search_v7.FR3_LIMITS,
  UR5 用 URDF 限位 clamp 到 +-pi),批量 FK 得法兰世界位置(与 l1 漫游的
  waypoint 同一帧:EnvEEBackend 的 flange 口径);
- 与 scene_layout_v7 工作区 AABB(x +-0.20 / y +-0.80 / z 0.80~1.40,
  z 底按 workspace_spec 惯例抬到桌面 +0.05)求交;
- 输出逐臂:可达点入带占比、入带云 1%/99% 分位盒、按 practical 半径
  过滤后的 x 侵入深度 -- 这些数字回填 l1_workspace_v7 的常量并写进
  artifacts/report_20260819/S4_v7_env_prep.md。

运行:PYTHONPATH=src python -m safeduo.delta.l1_reach_calib_v7
"""

from __future__ import annotations

import json

import torch

from safeduo.baselines.real_geometry import (
    FR3_FLANGE,
    FR3_JOINTS,
    FR3_LINKS,
    ArmKinematics,
)
from safeduo.delta._contract_stub import ARM_KEYS
from safeduo.delta.l2_env_source import RealScenePoses
from safeduo.envs.birth_pose_search_v7 import FR3_LIMITS, UR_LIMITS
from safeduo.envs.v7_offline_geometry import ur5_v7_kin

N = 400_000
CHUNK = 50_000
Z_TABLE_MARGIN = 0.05
RADII = (0.70, 0.75, 0.78, 0.80, 0.855)


def flange_cloud(kin, limits, base_pos, base_yaw, seed) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    lo, hi = limits[:, 0], limits[:, 1]
    q = lo + torch.rand(N, limits.shape[0], generator=g) * (hi - lo)
    outs = []
    for i in range(0, N, CHUNK):
        outs.append(kin.fk(q[i:i + CHUNK], base_pos, base_yaw)["t_flange"])
    return torch.cat(outs)


def main() -> None:
    poses = RealScenePoses("duo_env_v7.yaml")
    lo = torch.tensor(poses.ws_lo)
    hi = torch.tensor(poses.ws_hi)
    lo[2] = max(float(lo[2]), poses.table_top_z + Z_TABLE_MARGIN)
    kin = {"F": ArmKinematics(FR3_JOINTS, FR3_LINKS, True, FR3_FLANGE, 0.0),
           "U": ur5_v7_kin()}
    lim = {"F": FR3_LIMITS, "U": UR_LIMITS}
    rep = {"ws_aabb_lifted": [[round(float(v), 3) for v in lo],
                              [round(float(v), 3) for v in hi]]}
    for arm in ARM_KEYS:
        base = torch.tensor(poses.base_pos[arm])
        pts = flange_cloud(kin[arm[0]], lim[arm[0]], poses.base_pos[arm],
                           poses.base_yaw[arm], seed=hash(arm) % 2**31)
        inb = ((pts >= lo) & (pts <= hi)).all(dim=-1)
        cloud = pts[inb]
        r = (cloud - base).norm(dim=-1)
        q01 = torch.quantile(cloud, 0.01, dim=0)
        q99 = torch.quantile(cloud, 0.99, dim=0)
        entry = {
            "frac_in_aabb": round(float(inb.float().mean()), 4),
            "n_in": int(inb.sum()),
            "band_cloud_q01": [round(float(v), 3) for v in q01],
            "band_cloud_q99": [round(float(v), 3) for v in q99],
            "r_in_band_q05_q50_q95": [round(float(torch.quantile(r, qq)), 3)
                                      for qq in (0.05, 0.5, 0.95)],
        }
        # practical 半径筛后的 x 侵入深度(F 取 1% 分位最小 x, U 取 99% 最大 x)
        for rad in RADII:
            sel = cloud[r <= rad]
            if sel.numel() == 0:
                entry[f"x_pen_r{rad}"] = None
                continue
            xq = torch.quantile(sel[:, 0], 0.01 if arm[0] == "F" else 0.99)
            entry[f"x_pen_r{rad}"] = round(float(xq), 3)
            entry[f"frac_within_r{rad}"] = round(
                float((r <= rad).float().mean()), 3)
        rep[arm] = entry
    print("REACH_CALIB_V7 " + json.dumps(rep))


if __name__ == "__main__":
    main()
