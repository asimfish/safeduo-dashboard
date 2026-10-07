"""A5 训练入口：RSL-RL PPO + 非对称 actor-critic + 事件落盘 + 稳定性护栏。

用法（服务器）：
  python -m safeduo.train.train_ppo --num_envs 4096 --max_iterations 2000 \
      --headless [--resume <ckpt.pt>] [--seed 42] [--logger tensorboard]
护栏：kit 退出吞缓冲 -> 关键行 flush；checkpoint 周期落盘可 resume；
      seed 固定；日志 ~/safeduo/artifacts/runs/<name>/。
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", type=str, default="")
parser.add_argument("--run_name", type=str, default="")
parser.add_argument("--logger", type=str, default="tensorboard")
parser.add_argument("--save_interval", type=int, default=200)
parser.add_argument("--gamma", type=float, default=0.99,
                    help="PPO 折扣；v2 信用分配配方=0.995（0.99^600≈0.002 信号到不了早期动作）")
parser.add_argument("--bc_init", type=str, default="",
                    help="BC warm-start checkpoint（bc_warmstart_run 的 model.pt）初始化 actor")
parser.add_argument("--bc_p_neutral", action="store_true",
                    help="只移植 trunk+alpha 头，p 行保留 PPO 初始化（BC p 头退化时用）")
parser.add_argument("--bc_protect_iters", type=int, default=0,
                    help="critic 预热冻结 actor 的 update 数（C4 W6-2b，"
                         "algo/warmstart_protection；0=位元级 no-op 即消融臂，"
                         "BC-init 推荐 10）")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
)

from safeduo.envs.duo_env import DuoEnv, make_duo_env_cfg  # noqa: E402
from safeduo.safety.types import ARM_KEYS  # noqa: E402
from safeduo.train.events import EventWriter  # noqa: E402


def make_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        seed=args.seed,
        num_steps_per_env=24,
        max_iterations=args.max_iterations,
        save_interval=args.save_interval,
        experiment_name="safeduo_a5",
        run_name=args.run_name,
        logger=args.logger,
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=0.5,
            # [256,256] = C2 CoordinatorMLP 主干同构（BC warm-start 权重直灌，
            # train/bc_init.py），不用 BC 时也统一此结构保配方单一
            actor_hidden_dims=[256, 256],
            critic_hidden_dims=[256, 256],
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
            entropy_coef=0.003, num_learning_epochs=5, num_mini_batches=4,
            learning_rate=5.0e-4, schedule="adaptive", gamma=args.gamma, lam=0.95,
            desired_kl=0.01, max_grad_norm=1.0,
        ),
    )


def main():
    run = args.run_name or datetime.now().strftime("a5_%m%d_%H%M")
    log_dir = Path.home() / "safeduo" / "artifacts" / "runs" / run
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_duo_env_cfg(num_envs=args.num_envs, coordinator=True)
    cfg.seed = args.seed
    env = DuoEnv(cfg)
    # 事件写盘：挂进 env（coordinator 模式 _get_rewards 后由我们钩子取 cache）
    writer = EventWriter(str(log_dir / "events"),
                         n_log_envs=int(cfg.events_cfg["n_log_envs"]),
                         flush_every=int(cfg.events_cfg["flush_every"]))
    orig_rewards = env._get_rewards
    tick = {"t": 0}
    agg = {"tube": 0.0, "viol": 0.0, "n": 0}
    steps_per_iter = 24  # = runner_cfg.num_steps_per_env

    def rewards_with_events():
        r = orig_rewards()
        c = env._step_cache
        if c:
            out = env._last_out
            # C2 W4-1 schema 建议全采纳：mm_self（违规归因不再靠排除法）、
            # ep_step（回合边界重建）、逐臂 backstop、逐臂 cmd/exec 范数
            # （tracking/progress/stall 保真类指标可从训练流直接算）。
            writer.add(tick["t"], {
                "alpha": c["alpha"], "p": c["p"],
                "violation": c["violation"].float(), "tube": c["tube"].float(),
                "backstop_any": c["bs_active"].any(-1).float(),
                "backstop": c["bs_active"].float(),
                "mm_cross": out.min_margin["cross"].clamp(-1, 10),
                "mm_table": out.min_margin["table"].clamp(-1, 10),
                "mm_self_f": out.min_margin["self_F"].clamp(-1, 10),
                "mm_self_u": out.min_margin["self_U"].clamp(-1, 10),
                "ep_step": env.episode_length_buf.float(),
                "cmd_norm": torch.stack(
                    [c["cmd"].delta_q[a].norm(dim=-1) for a in ARM_KEYS], -1),
                "exec_norm": torch.stack(
                    [c["exec"].delta_q[a].norm(dim=-1) for a in ARM_KEYS], -1),
                "reward": r,
            })
            # l2_mix 接线验证/监控：tube_fraction 必须 > 0（v1 核心缺陷 =
            # 冲突从不发生）；MIX_FRACTIONS/UR_CALIB 开训打一次（C2 要求）
            agg["tube"] += c["tube"].float().mean().item()
            agg["viol"] += c["violation"].float().mean().item()
            agg["n"] += 1
            src = getattr(env, "_delta_src", None)
            if tick["t"] == 0 and src is not None:
                if hasattr(src, "mix_fractions"):
                    print("MIX_FRACTIONS " + json.dumps(src.mix_fractions), flush=True)
                be = getattr(src, "backend", None)
                if be is not None and getattr(be, "calib_report", None):
                    print("UR_CALIB " + json.dumps(be.calib_report), flush=True)
            tick["t"] += 1
            if tick["t"] % 600 == 0:
                print(f"TUBE_STATS t={tick['t']} tube_frac={agg['tube']/agg['n']:.4f} "
                      f"viol_frac={agg['viol']/agg['n']:.4f}", flush=True)
                agg.update(tube=0.0, viol=0.0, n=0)
            # 课程推进（stages 未启用时为平权 no-op；C2 的 set_progress 契约）
            if hasattr(src, "set_progress") and tick["t"] % steps_per_iter == 0:
                src.set_progress(tick["t"] / steps_per_iter / max(args.max_iterations, 1))
        return r

    env._get_rewards = rewards_with_events
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner_cfg = make_runner_cfg()
    runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), str(log_dir), args.device)
    if args.resume:
        runner.load(args.resume)
        print(f"RESUMED from {args.resume}", flush=True)
    elif args.bc_init:
        from safeduo.train.bc_init import load_bc_actor_init

        policy = getattr(runner.alg, "policy", None)
        if policy is None:
            policy = runner.alg.actor_critic  # rsl_rl 旧版属性名
        meta = load_bc_actor_init(policy.actor, args.bc_init,
                                  p_neutral=args.bc_p_neutral)
        print("BC_INIT " + json.dumps(meta), flush=True)
    # 早期保护（C4 W6-2b 三行接线）：前 N 个 update 冻结 actor+std，critic
    # 在固定策略下先学 V；释放时 lr 复位（adaptive-KL 在冻结期会吹大 lr）。
    # n=0 是位元级 no-op（消融臂）。必须在 bc_init 移植之后 attach。
    from safeduo.algo.warmstart_protection import attach_warmstart_protection

    protect_meta = attach_warmstart_protection(runner.alg,
                                               n_iters=args.bc_protect_iters)
    print("BC_PROTECT " + json.dumps(protect_meta, default=str), flush=True)
    t0 = time.time()
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    writer.flush()
    stats = {"run": run, "iterations": args.max_iterations, "seed": args.seed,
             "wall_s": round(time.time() - t0, 1), "num_envs": args.num_envs,
             "gamma": args.gamma, "bc_init": args.bc_init or None,
             "bc_protect_iters": args.bc_protect_iters}
    (log_dir / "done.json").write_text(json.dumps(stats))
    print("TRAIN_DONE " + json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
    app.close()
