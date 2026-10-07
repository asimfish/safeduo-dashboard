"""R3 batch-1 entry: SAC-Lagrangian on the live DuoEnv (or local smoke).

C15-owned, zero-touch discipline: mirrors train_lagrangian.py's proven
wiring (AppLauncher-first ordering, env adapter, cost_sync single-source,
WandbLogger, done-marker from python's success path) WITHOUT modifying it --
the adapter is copied with attribution. Differences from the PPO entry,
both sanctioned by the R3 task card:
  - EventWriter shards are omitted (the freeze-sentinel chain belongs to
    the PPO mainline; the race arm's health signals ride stats.jsonl +
    wandb, which carry the same freeze-stat fields).
  - --smoke runs locally on KinematicDuoEnv with NO isaac import at all
    (isaaclab import is guarded), so this module also imports cleanly in
    the local .venv for tests.

Local full-chain smoke (CPU):
  PYTHONPATH=src .venv/bin/python -m safeduo.algo.train_sac_lagrangian --smoke
Server (bjxy_5090, canonical env; launch discipline in STATUS_C @A7):
  PYTHONPATH=src python -m safeduo.algo.train_sac_lagrangian --headless \
      --num_envs 4096 --max_iterations 2000 --seed 42 \
      --run_name c15_sac_race [--controller pid|dual|off] [--device cuda:0]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

try:
    from isaaclab.app import AppLauncher
except ImportError:  # local .venv: smoke path only
    AppLauncher = None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true",
                   help="local kinematic full-chain smoke (CPU, no isaac)")
    p.add_argument("--n-envs", type=int, default=16,
                   help="smoke-mode env count")
    p.add_argument("--iterations", type=int, default=25,
                   help="smoke-mode iterations")
    p.add_argument("--out", default="artifacts/analysis/c15_sac_lagrangian_smoke",
                   help="smoke-mode summary dir")
    # server args, mirroring train_lagrangian.py
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--max_iterations", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--env-yaml", type=str, default="duo_env_v4.yaml")
    p.add_argument("--controller", choices=["pid", "dual", "off"],
                   default="pid")
    p.add_argument("--cost_limit", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--save_interval", type=int, default=50)
    p.add_argument("--resume", type=str, default="")
    # SAC-specific knobs (defaults argued in sac_lagrangian.py)
    p.add_argument("--replay_capacity", type=int, default=524_288)
    p.add_argument("--learning_starts", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--updates_per_iteration", type=int, default=32)
    if AppLauncher is not None:
        AppLauncher.add_app_launcher_args(p)
    return p


def run_server(args) -> None:
    """Live DuoEnv training. AppLauncher must exist before any isaac-side
    import, hence everything below is lazy (train_lagrangian.py ordering)."""
    if AppLauncher is None:
        raise SystemExit("isaaclab not installed -- server mode needs the "
                         "canonical env; locally use --smoke")
    app = AppLauncher(args).app

    import torch  # noqa: F401  (device strings below)

    from safeduo.algo.sac_lagrangian import (
        SACLagrangian,
        SACLagrangianConfig,
        cost_sync_from_env_cfg,
    )
    from safeduo.envs.duo_env import DuoEnv, make_duo_env_cfg
    from safeduo.train.wandb_logger import WandbLogger

    class _EnvAdapter:
        """Copy of train_lagrangian._EnvAdapter (zero-touch discipline):
        pins num_envs/device, exposes the step_telemetry + cost_channels
        hooks the loop consumes. See the original for the hook contracts."""

        def __init__(self, env: DuoEnv):
            self.env = env
            self.num_envs = env.num_envs
            self.device = env.device

        def reset(self):
            return self.env.reset()

        def step(self, action):
            return self.env.step(action.to(self.env.device))

        def step_telemetry(self) -> dict:
            c = getattr(self.env, "_step_cache", None)
            out = getattr(self.env, "_last_out", None)
            if not c or out is None:
                return {}
            frame = {
                "violation": c["violation"].detach().clone(),
                "tube": c["tube"].detach().clone(),
                "bs_active": c["bs_active"].detach().float().clone(),
            }
            mm = out.min_margin
            for key, cls in (("mm_cross", "cross"), ("mm_table", "table"),
                             ("mm_self_f", "self_F"), ("mm_self_u", "self_U")):
                frame[key] = mm[cls].detach().clamp(-1, 10).clone()
            return frame

        def cost_channels(self):
            c = getattr(self.env, "_step_cache", None)
            if not c or "cost_by_class" not in c:
                return None
            return c["cost_by_class"].detach()

    run = args.run_name or datetime.now().strftime("c15_sac_%m%d_%H%M")
    log_dir = Path.home() / "safeduo" / "artifacts" / "runs" / run
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_duo_env_cfg(num_envs=args.num_envs, coordinator=True,
                           yaml_name=args.env_yaml)
    cfg.seed = args.seed
    # Lagrangian arm: margin cost rides the constraint channels, NOT the
    # reward (same double-charging guard as the PPO entry)
    if args.controller != "off":
        cfg.coordinator["reward"]["w_margin"] = 0.0
    env = DuoEnv(cfg)

    sync = cost_sync_from_env_cfg(cfg.safety_cfg, cfg.semantics_yaml)
    loop_cfg = SACLagrangianConfig(
        gamma=args.gamma, controller=args.controller,
        cost_limits=(args.cost_limit,) * 3,
        save_interval=args.save_interval,
        replay_capacity=args.replay_capacity,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        updates_per_iteration=args.updates_per_iteration,
        d_warn=sync["d_warn"], d_min_by_class=sync["d_min_by_class"])
    wandb_logger = WandbLogger(
        run, log_dir=log_dir,
        config={"run": run, "algo": "sac_lagrangian",
                "num_envs": args.num_envs,
                "max_iterations": args.max_iterations, "seed": args.seed,
                "controller": args.controller, "gamma": args.gamma,
                "cost_limit": args.cost_limit, "env_yaml": args.env_yaml,
                "replay_capacity": args.replay_capacity,
                "batch_size": args.batch_size,
                "updates_per_iteration": args.updates_per_iteration,
                "save_interval": args.save_interval})
    trainer = SACLagrangian(_EnvAdapter(env), loop_cfg, log_dir=log_dir,
                            device=str(env.device), seed=args.seed,
                            wandb_logger=wandb_logger)
    if args.resume:
        trainer.load(args.resume)
        print(f"RESUMED from {args.resume} at iter {trainer.iteration}",
              flush=True)
    t0 = time.time()
    trainer.learn(args.max_iterations)
    wandb_logger.finish()
    stats = {"run": run, "algo": "sac_lagrangian",
             "iterations": args.max_iterations, "seed": args.seed,
             "controller": args.controller, "num_envs": args.num_envs,
             "gamma": args.gamma, "env_yaml": args.env_yaml,
             "wall_s": round(time.time() - t0, 1)}
    (log_dir / "done.json").write_text(json.dumps(stats))
    print("TRAIN_DONE " + json.dumps(stats), flush=True)
    app.close()


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke:
        from safeduo.algo.sac_lagrangian import print_smoke_summary, run_smoke

        summary = run_smoke(args.n_envs, args.iterations, args.seed, args.out)
        print_smoke_summary(summary)
        raise SystemExit(0 if summary["pass"] else 1)
    run_server(args)


if __name__ == "__main__":
    main()
