"""球距离 vs PhysX ContactSensor 对拍脚本骨架（A3 正式交付 10^5 构型报告，W1 先立结构）。

思路：小规模 env 里灌随机动作让四臂乱动，逐步对比
  预测碰撞  = 球距离模块 min_margin < 0
  真实碰撞  = 任一 body 净接触力 > force_eps（对拍口径：机器人-机器人/机器人-桌）
统计漏检（gt 碰但预测没碰）与误报（预测碰但 gt 没碰）。
W1 已知粗糙点：net force 无法区分碰撞对象（含自碰/桌），归因细化等 B 的正式球分解后做。

用法：python -m safeduo.safety.contact_parity --num_envs 64 --steps 500 --headless
"""

from __future__ import annotations

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--force_eps", type=float, default=1.0, help="判定接触的净力阈值（N）")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from safeduo.envs.duo_env import DuoEnv, make_duo_env_cfg  # noqa: E402
from safeduo.safety.types import ARM_KEYS  # noqa: E402


def main():
    cfg = make_duo_env_cfg(num_envs=args.num_envs)
    cfg.enable_contact_gt = True
    env = DuoEnv(cfg)
    env.reset()
    n = args.num_envs
    stats = {"steps": 0, "gt_hit": 0, "pred_hit": 0, "miss": 0, "false_pos": 0}
    for _ in range(args.steps):
        act = torch.rand(n, cfg.action_space, device=env.device) * 2 - 1
        env.step(act)
        out = env._last_dist_out
        pred = torch.stack(list(out.min_margin.values()), dim=-1).amin(dim=-1) < 0.0  # (N,)
        forces = []
        for arm in ARM_KEYS:
            f = env._contact[arm].data.net_forces_w  # (N, B, 3)
            forces.append(f.norm(dim=-1).amax(dim=-1))
        gt = torch.stack(forces, dim=-1).amax(dim=-1) > args.force_eps  # (N,)
        stats["steps"] += n
        stats["gt_hit"] += int(gt.sum())
        stats["pred_hit"] += int(pred.sum())
        stats["miss"] += int((gt & ~pred).sum())
        stats["false_pos"] += int((pred & ~gt).sum())
    stats["miss_rate"] = stats["miss"] / max(stats["gt_hit"], 1)
    stats["false_rate"] = stats["false_pos"] / max(stats["pred_hit"], 1)
    print("PARITY " + json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
    app.close()
