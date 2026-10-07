"""v4 场景冒烟（A6-W6，B 七步换装清单第 6 步）：真布局+真资产 duo_env 首次点火。

验证面（一次运行全出，G 门口径=看内容）：
1. 四臂组合 USD 装载 + 执行器全覆盖（Isaac 硬要求，缺了 init 即炸）；
2. 26 关节契约（fr3_joint[1-7]x2 + joint[1-6]x2，手关节不入动作空间）；
3. obs 契约 275 维 + pairs 块量级（A5 的 P0 口径沿用）；
4. 出生违规率（v3 时代 W3 事故口径：settle 后不得全场违规）+ 四桶最小边距；
5. 球绑定计数（F 臂 = fr3 19+法兰 1+手 10；U 臂 = jaka 16+手 10）；
6. 吞吐计时（默认 64 env 对照 W1 的 1128 steps/s 量级；--num_envs 4096 可做
   G0' 快速复验，门 >= 1e4 control steps/s）。
delta 源强制 l1：l2_mix 场景库工作点等 C 线在 v4 布局重扫（STATUS_B @C 预告），
冒烟不消费它。

用法（服务器）：
  source ~/safeduo_setup/env.sh && cd ~/safeduo && \
  PYTHONPATH=src python -m safeduo.envs.v4_scene_smoke --num_envs 64 --headless
"""

from __future__ import annotations

import argparse
import json
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--timed_steps", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--env_yaml", type=str, default="duo_env_v4.yaml",
                    help="env profile（duo_env_v5.yaml = v5 bundle 场景，A9-W9）")
parser.add_argument("--marker", type=str, default="",
                    help="success marker file written by THIS script on PASS "
                         "(kit overwrites the process exit code -- Round-76 "
                         "lesson: never trust '&& touch', use verdict files)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from safeduo.envs.duo_env import DuoEnv, make_duo_env_cfg  # noqa: E402
from safeduo.safety.types import ARM_KEYS, DOF_OF  # noqa: E402


@torch.no_grad()
def main() -> None:
    cfg = make_duo_env_cfg(num_envs=args.num_envs, yaml_name=args.env_yaml,
                           coordinator=True)
    cfg.seed = args.seed
    cfg.coordinator["delta_source"] = "l1"
    env = DuoEnv(cfg)
    n = env.num_envs
    report: dict = {"num_envs": n, "env_yaml": args.env_yaml}

    # 2) 关节契约
    for arm in ARM_KEYS:
        got = len(env._joint_idx[arm])
        assert got == DOF_OF[arm], f"{arm}: {got} != {DOF_OF[arm]}"
    report["arm_dof"] = {a: int(len(env._joint_idx[a])) for a in ARM_KEYS}
    report["ee_body"] = {a: env._arms[a].body_names[env._ee_idx[a]]
                         for a in ARM_KEYS}
    # 5) 球绑定计数（逐臂）
    report["spheres_per_arm"] = {
        a: int(env._sph._arm_slices[a].stop - env._sph._arm_slices[a].start)
        for a in ARM_KEYS}
    report["n_spheres_total"] = int(env._sph.n_spheres)
    report["n_pairs"] = int(env._sph.n_pairs)

    obs, _ = env.reset(seed=args.seed)
    x = obs["policy"]
    report["obs_dim"] = int(x.shape[1])
    assert x.shape[1] == 275, f"obs dim {x.shape[1]} != 275"

    # 4a) 严格出生读数（A9-W9 门③口径）：reset 后、任何 step 前的名义边距
    # （settle 前无 PD 下垂/无 delta 流，与 birth_pose_search 名义读数同轴）
    out0 = env.compute_dist()
    report["birth_min_margin_min"] = {
        k: round(float(v.min()), 4) for k, v in out0.min_margin.items()}
    report["birth_min_margin_p05"] = {
        k: round(float(v.quantile(0.05)), 4) for k, v in out0.min_margin.items()}
    report["birth_violation_frac"] = round(
        float(out0.violation.float().mean()), 5)

    # 4) 出生 settle：passthrough 空指令 30 步看边距与违规
    action = torch.zeros(n, 5, device=env.device)
    action[:, :4] = 1.0
    viol_steps = 0
    absmax_pairs = 0.0
    for t in range(args.steps):
        obs, _, _, _, _ = env.step(action)
        viol_steps += int(env._last_out.violation.sum())
        absmax_pairs = max(absmax_pairs,
                           float(obs["policy"][:, 52:212].abs().max()))
        if t == 29:
            mm = env._last_out.min_margin
            report["min_margin_p05_at_settle"] = {
                k: round(float(v.quantile(0.05)), 4) for k, v in mm.items()}
            report["violation_frac_at_settle"] = round(
                float(env._last_out.violation.float().mean()), 4)
    report["viol_env_steps_frac"] = round(viol_steps / (args.steps * n), 5)
    report["absmax_pairs_block"] = round(absmax_pairs, 3)
    print("V4_SMOKE_SETTLE " + json.dumps(report), flush=True)

    # 6) 吞吐（control steps/s = env-steps x num_envs / wall）
    is_cuda = str(env.device).startswith("cuda")
    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.timed_steps):
        obs, _, _, _, _ = env.step(action)
    if is_cuda:
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    report["control_steps_per_s"] = round(args.timed_steps * n / wall, 1)
    report["wall_per_env_step_ms"] = round(wall / args.timed_steps * 1000, 2)

    print("V4_SMOKE " + json.dumps(report), flush=True)
    ok = (report["violation_frac_at_settle"] < 0.05
          and report["absmax_pairs_block"] <= 50)
    print("V4_SMOKE_" + ("PASS" if ok else "FAIL"), flush=True)
    if ok and args.marker:
        from pathlib import Path

        Path(args.marker).expanduser().write_text(json.dumps(report))


if __name__ == "__main__":
    main()
    import os

    os._exit(0)
