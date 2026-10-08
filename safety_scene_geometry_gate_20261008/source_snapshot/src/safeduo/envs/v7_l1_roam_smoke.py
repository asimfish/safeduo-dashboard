"""S4 任务 2 验收跑:v7 新出生位 + 新 reach 盒下的 l1 漫游 raw 覆盖验证。

- delta 源 = WorkspaceRoamDeltaV7(v7 UR5 运动学 backend + FK 标定 reach 盒,
  delta/l1_workspace_v7);endurance l1 野档参数(不停歇/短段/OU 抖动);
- raw 口径 = 安全投影去除(endurance_eval._RawShim 同款:cmd 原样通过,
  active 行全 False),看纯随机流本身的几何行为;
- 出生读数 = reset 后、step 前 compute_dist(与 birth_pose_search_v7
  名义读数同轴),验收门:出生违规率 0;
- 覆盖统计 = 逐步记录四臂 EE(env 局部系),报告逐臂/四臂同时进入共享带
  (|x|<0.20)的步占比 + x-y 占据热区图(png + npz 落 --out)。

用法(服务器):
  source ~/safeduo_setup/env.sh && cd ~/safeduo && \
  PYTHONPATH=src python -m safeduo.envs.v7_l1_roam_smoke \
      --num_envs 2 --steps 300 --headless \
      --out artifacts/report_20260819/s4_l1_roam_v7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--seed", type=int, default=20260819)
parser.add_argument("--amp", type=float, default=0.03,
                    help="l1 段幅(endurance regular 档)")
parser.add_argument("--ou_mix", type=float, default=None,
                    help="OU 权重覆盖(默认 L1Params 0.5;调低 = 更目标导向)")
parser.add_argument("--ee_speed", type=float, default=0.25,
                    help="waypoint 追踪 EE 速度帽 m/s")
parser.add_argument("--seg_dur", type=float, nargs=2, default=(0.8, 2.0),
                    help="段时长范围 s(v7 转运距离长, 建议 2-5)")
parser.add_argument("--tag", type=str, default="",
                    help="附加进 stats.json 的档位标签")
parser.add_argument("--env_yaml", type=str, default="duo_env_v7.yaml")
parser.add_argument("--out", type=str,
                    default="artifacts/report_20260819/s4_l1_roam_v7")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from safeduo.delta._contract_stub import ARM_KEYS  # noqa: E402
from safeduo.delta.l1_random import L1Params  # noqa: E402
from safeduo.delta.l1_workspace_v7 import WorkspaceRoamDeltaV7  # noqa: E402
from safeduo.envs.duo_env import DuoEnv, make_duo_env_cfg  # noqa: E402

BAND_X = 0.20


class _RawShim:
    """endurance_eval._RawShim 同款:无安全投影 passthrough."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def project(self, cmd, rows, alpha, p, dt, dmin=None, backlog=None, **kw):
        act = torch.zeros(alpha.shape[0], len(ARM_KEYS),
                          dtype=torch.bool, device=alpha.device)
        return cmd, act, {}


@torch.no_grad()
def main() -> None:
    cfg = make_duo_env_cfg(num_envs=args.num_envs, yaml_name=args.env_yaml,
                           coordinator=True)
    cfg.seed = args.seed
    cfg.coordinator["delta_source"] = "l1"      # 占位, 构造后换 v7 源
    env = DuoEnv(cfg)
    n = env.num_envs
    env._backstop = _RawShim(env._backstop)
    p = L1Params(amp_max=float(args.amp))
    p.pause_prob = 0.0
    p.seg_dur = (float(args.seg_dur[0]), float(args.seg_dur[1]))
    p.ou_sigma = 0.02
    if args.ou_mix is not None:
        p.ou_mix = float(args.ou_mix)
    env._delta_src = WorkspaceRoamDeltaV7(n, params=p, device=env.device,
                                          ee_speed=float(args.ee_speed))
    env._pending_cmd = None

    env.reset(seed=args.seed)
    out0 = env.compute_dist()
    report: dict = {
        "num_envs": n, "steps": args.steps, "amp": args.amp,
        "seed": args.seed, "ou_mix": args.ou_mix, "ee_speed": args.ee_speed,
        "seg_dur": list(args.seg_dur), "tag": args.tag,
        "birth_min_margin": {k: round(float(v.min()), 4)
                             for k, v in out0.min_margin.items()},
        "birth_violation_frac": round(float(out0.violation.float().mean()), 5),
    }

    action = torch.zeros(n, 5, device=env.device)
    action[:, :4] = 1.0
    traj = {a: [] for a in ARM_KEYS}
    viol_steps = 0
    for _ in range(args.steps):
        env.step(action)
        st = env.scene_state()
        for a in ARM_KEYS:
            traj[a].append(st.ee_pos[a][:, :3].cpu().clone())
        viol_steps += int(env._last_out.violation.sum())

    xyz = {a: torch.stack(traj[a]) for a in ARM_KEYS}       # (T, N, 3)
    band = {a: (xyz[a][:, :, 0].abs() < BAND_X) for a in ARM_KEYS}
    report["band_frac_per_arm"] = {
        a: round(float(band[a].float().mean()), 4) for a in ARM_KEYS}
    all4 = torch.stack([band[a] for a in ARM_KEYS]).all(dim=0)
    report["band_frac_all4"] = round(float(all4.float().mean()), 4)
    report["viol_env_steps_frac"] = round(viol_steps / (args.steps * n), 5)
    report["ee_x_range"] = {
        a: [round(float(xyz[a][:, :, 0].min()), 3),
            round(float(xyz[a][:, :, 0].max()), 3)] for a in ARM_KEYS}
    report["calib"] = {k: round(v, 4) if isinstance(v, float) else v
                       for k, v in env._delta_src._backend.calib_report.items()}

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.savez(out_dir / "ee_traj.npz",
             **{a: xyz[a].numpy() for a in ARM_KEYS})
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    for ax, a in zip(axes.flat, ARM_KEYS):
        pts = xyz[a].reshape(-1, 3)
        h = ax.hist2d(pts[:, 0].numpy(), pts[:, 1].numpy(),
                      bins=(45, 34), range=[[-0.45, 0.45], [-0.85, 0.85]],
                      cmap="hot")
        ax.axvline(-BAND_X, color="cyan", lw=1)
        ax.axvline(BAND_X, color="cyan", lw=1)
        ax.set_title(f"{a}  band {report['band_frac_per_arm'][a]:.1%}")
        fig.colorbar(h[3], ax=ax)
    fig.suptitle(f"v7 l1 roam raw EE occupancy ({n} env x {args.steps} steps, "
                 f"amp {args.amp}) -- band |x|<{BAND_X}")
    fig.savefig(out_dir / "ee_heatmap.png", dpi=110, bbox_inches="tight")
    (out_dir / "stats.json").write_text(json.dumps(report, indent=1))

    print("V7_L1_ROAM " + json.dumps(report), flush=True)
    ok = report["birth_violation_frac"] == 0.0
    print("V7_L1_ROAM_" + ("PASS" if ok else "FAIL"), flush=True)


if __name__ == "__main__":
    main()
    import os

    os._exit(0)
