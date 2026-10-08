"""吞吐基准：单个 env 数一次进程（Isaac 同进程重建 env 不稳，扫点由外层 shell 循环）。

用法（服务器）：
  python -m safeduo.train.benchmark --num_envs 4096 --headless [--pipeline]
输出一行 JSON（BENCH 前缀）并落盘 ~/safeduo/artifacts/bench/。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--pipeline", action="store_true",
                    help="附带 SmokeNoiseDelta + backstop 直通 + SceneState 构造的整链路计时")
parser.add_argument("--franka", type=str, default=None, help="覆盖 franka 变体（panda|fr3）")
parser.add_argument("--ur", type=str, default=None, help="覆盖 ur 变体（ur10|ur5e）")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from safeduo.configs import load_config  # noqa: E402
from safeduo.delta.smoke_noise import SmokeNoiseDelta  # noqa: E402
from safeduo.envs.duo_env import DuoEnv, make_duo_env_cfg  # noqa: E402
from safeduo.safety.types import backstop_project  # noqa: E402


def run(n_envs: int, warmup: int, steps: int, pipeline: bool) -> dict:
    cfg = make_duo_env_cfg(num_envs=n_envs, franka_variant=args.franka, ur_variant=args.ur)
    env = DuoEnv(cfg)
    env.reset()
    n_arms = 4
    delta_src = SmokeNoiseDelta(n_envs, device=env.device) if pipeline else None
    if delta_src:
        delta_src.reset(torch.arange(n_envs, device=env.device),
                        torch.Generator(device=env.device).manual_seed(0))

    def one_step():
        if delta_src is None:
            act = torch.rand(n_envs, cfg.action_space, device=env.device) * 2 - 1
        else:
            state = env.scene_state()
            alpha = torch.ones(n_envs, n_arms, device=env.device)
            p = torch.zeros(n_envs, device=env.device)
            cmd, _ = backstop_project(delta_src.sample(state), state, alpha, p)
            act = cmd.stacked() / cfg.delta_clip
        env.step(act)

    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        one_step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    sps = n_envs * steps / dt
    return {
        "num_envs": n_envs,
        "mode": "pipeline" if pipeline else "raw",
        "franka": cfg.variant_of["F_L"],
        "ur": cfg.variant_of["U_L"],
        "control_steps_per_s": round(sps),
        "physics_steps_per_s": round(sps * cfg.decimation),
        "wall_s": round(dt, 2),
        "gpu": torch.cuda.get_device_name(0),
    }


def main():
    bench = load_config()["benchmark"]
    result = run(args.num_envs, int(bench["warmup_steps"]), int(bench["timed_steps"]),
                 args.pipeline)
    # kit 退出会 os._exit 吞掉缓冲，必须 flush + 落盘双保险
    print("BENCH " + json.dumps(result), flush=True)
    out_dir = Path.home() / "safeduo" / "artifacts" / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{result['mode']}_{result['franka']}_{result['ur']}_{result['num_envs']}"
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
    app.close()
