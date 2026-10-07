"""v4 recipe server entry: live DuoEnv + the self-built PPO-Lagrangian loop.

C-line owned (algo/): train/train_ppo.py stays untouched -- this entry
mirrors its proven wiring (AppLauncher first, EventWriter hook on
_get_rewards, l2_mix logging, done-marker written from python) but swaps the
learning stack for algo/lagrangian_ppo.LagrangianPPO per the Round-110
supervision ruling (PID-Lagrangian + Beta actor + validity-aware GAE +
peak-harvest checkpoints).

Usage (bjxy_5090, canonical env; launch discipline in STATUS_C @A7):
  PYTHONPATH=src python -m safeduo.algo.train_lagrangian --headless \
      --num_envs 4096 --max_iterations 2000 --seed 42 \
      --run_name c5_v4_main [--controller pid|dual|off] [--device cuda:0]
Pair it with the freeze sentinel sidecar (separate OS process):
  PYTHONPATH=src python -m safeduo.algo.freeze_sentinel \
      --events-dir ~/safeduo/artifacts/runs/<run>/events --watch

Notes:
- env profile: duo_env_v4.yaml is selected by make_duo_env_cfg's yaml_name
  (v4 asset swap is A-line's call; --env-yaml duo_env.yaml runs the v3-era
  scene for A/B).
- events shards carry alpha/p/reward etc. (train/events.py schema) so the
  sentinel, run-events gates and p_head_diagnosis all work unchanged;
  stats.jsonl additionally carries lambda/cost channels per iteration.
- the freeze-stats + peak-harvest live in the loop itself (stats.jsonl,
  model_<iter>.pt every save_interval); block1_harness --run-dir picks the
  peak checkpoint by default.
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
parser.add_argument("--run_name", type=str, default="")
parser.add_argument("--env-yaml", type=str, default="duo_env_v4.yaml",
                    help="duo_env profile (duo_env_v4.yaml = real assets; "
                         "duo_env.yaml = v3-era scene)")
parser.add_argument("--curriculum-yaml", type=str, default="",
                    help="delta curriculum recipe forwarded to "
                         "cfg.coordinator['curriculum_yaml'] (consumed by "
                         "duo_env's l2_mix branch only). Empty = keep the "
                         "env profile's current behavior (zero drift); "
                         "delta_curriculum_t3.yaml = R2 amp 0.03->0.10 "
                         "stages + skill mix + workspace roam.")
parser.add_argument("--controller", choices=["pid", "dual", "off"],
                    default="pid",
                    help="constraint controller arm (off = fixed-penalty "
                         "ablation: margin stays in the reward as w_margin)")
parser.add_argument("--cost_limit", type=float, default=0.05)
parser.add_argument("--cost-limits", type=str, default="",
                    help="R13: per-channel limits 'cross,self,table' (e.g. "
                         "'0.08,0.05,0.05'); overrides --cost_limit. empty = "
                         "scalar --cost_limit on all channels (zero drift)")
parser.add_argument("--cost-scale", type=str, default="",
                    help="R10: per-channel cost divisors 'cross,self,table' "
                         "(e.g. '1,17,19'); empty = (1,1,1) zero-drift")
parser.add_argument("--lambda-max", type=str, default="",
                    help="R13: per-channel PID multiplier caps with "
                         "anti-windup, 'cross,self,table'; entries <= 0 mean "
                         "no cap (e.g. '0.35,0,0'). empty = uncapped")
parser.add_argument("--entropy-coef", type=float, default=0.003,
                    help="R13: base entropy coefficient (v6.2_core: 0.006)")
parser.add_argument("--entropy-coef-hi", type=float, default=0.0,
                    help="R13: adaptive entropy bump; > 0 enables the EMA20 "
                         "latch (<= -4.8 raises to this value, >= -4.5 "
                         "releases). v6.2_core: 0.012. 0 = disabled")
parser.add_argument("--gamma", type=float, default=0.995)
parser.add_argument("--save_interval", type=int, default=50)
parser.add_argument("--resume", type=str, default="")
# ---- R15 (v7 utility-recovery recipe) -- every default is a no-op ----
parser.add_argument("--arm-aware-obs", action="store_true",
                    help="R15 ①: pair rows carry two-arm one-hot identity "
                         "(obs 275 -> 531 @ M=32; env + trainer switched "
                         "together). Fresh runs only -- old checkpoints "
                         "have the 275-dim obs and will not load")
parser.add_argument("--w-armbal", type=float, default=0.0,
                    help="R15 ①: per-arm alpha balance regularizer weight "
                         "w*(max_arm mean-alpha - min_arm mean-alpha) on the "
                         "loss (recommended 0.01). 0 = off")
parser.add_argument("--ent-trigger", type=float, default=-4.8,
                    help="R15 ②: entropy-latch engage threshold on the "
                         "EMA20 (was hardcoded -4.8 -- v6.2 never fired, "
                         "EMA floor -4.274; recommended -3.5)")
parser.add_argument("--ent-release", type=float, default=-4.5,
                    help="R15 ②: entropy-latch release threshold (must sit "
                         "ABOVE --ent-trigger for real hysteresis; with "
                         "trigger -3.5 use e.g. -3.2)")
parser.add_argument("--ent-cost-gate", type=float, default=float("inf"),
                    help="R15 ②: latch engages only while cost_cross < "
                         "gate * limit_cross (recommended 0.9). inf = no "
                         "gate = pre-R15 trigger (NOTE 1.0 is not a no-op: "
                         "v6.2 sat at/above the limit 46.8%% of iterations)")
parser.add_argument("--w-alpha-util", type=float, default=0.0,
                    help="R15 ③a: alpha->1 utility pricing on danger-free "
                         "arms, reward += -w*(alpha-1)^2 per gated arm "
                         "(recommended 0.5 vs w_track=10). 0 = off")
parser.add_argument("--lambda-rate-max", type=float, default=float("inf"),
                    help="R15 ③b: PID multiplier slew limit per update, "
                         "|d lambda| <= rate (recommended 0.01); applied "
                         "before the --lambda-max cap. inf = off")
# ---- R18 (clutch-semantics retraining, S8 recipe) -- every default is a
# no-op; the a22 recipe turns all three on together ----
parser.add_argument("--clutch-train", action="store_true",
                    help="R18 (2): binarize rollout alpha at 0.5 (straight-"
                         "through) before the env executes it -- train-exec "
                         "match with the eval-side clutch. Stored samples "
                         "stay continuous (PPO replay unchanged)")
parser.add_argument("--w-hazard-bce", type=float, default=0.0,
                    help="R18 (3): auxiliary BCE weight on mean alpha vs "
                         "hindsight hazard labels (margin below the lock "
                         "line within --hazard-horizon steps). recommended "
                         "0.5. 0 = off")
parser.add_argument("--w-alpha-deadzone", type=float, default=0.0,
                    help="R18 (4): bidirectional dead-zone pricing on the "
                         "EXECUTED alpha (hazard: a^2, safe: (1-a)^2, gray "
                         "band: 0). replaces --w-alpha-util (mutually "
                         "exclusive). 0 = off")
parser.add_argument("--hazard-horizon", type=int, default=30,
                    help="R18 (3): hindsight lookahead in control steps "
                         "(30 = 0.5 s at 60 Hz). NOTE with rollout T=24 "
                         "every window is buffer-truncated; see "
                         "--hazard-tail-mask (S14)")
parser.add_argument("--hazard-tail-mask", action="store_true",
                    help="S14 (R18 hindsight-window fix): drop ambiguous "
                         "buffer-tail labels from the BCE (masked mean) "
                         "and zero-price them in the dead-zone. Off = "
                         "S10/a22 behavior. Recommended for the next "
                         "retrain together with --hazard-horizon 12 "
                         "(0.2 s; T=24 keeps 12 full-window steps) or a "
                         "longer rollout (num_steps_per_env 48 + 30)")
parser.add_argument("--num-steps-per-env", type=int, default=24,
                    help="R22: PPO rollout length T per env (24 = A5 runner "
                         "cfg). 48 keeps a --hazard-horizon 30 window fully "
                         "inside the buffer, so tail-mask labels stay dense "
                         "instead of collapsing to the episode tail")
parser.add_argument("--hazard-coop-vmax", type=float, default=0.0,
                    help="a27: exempt controlled-rendezvous CROSS rows "
                         "(closing < this speed, not deep intrusions) from "
                         "the hazard label; 0 = off (bit-identical)")
parser.add_argument("--hazard-bce-balance", action="store_true",
                    help="R22: class-balance the R18 hazard BCE so hazard "
                         "and safe labels each carry half the loss -- with "
                         "the ~9:1 safe-heavy mix a plain mean lets the "
                         "majority class pin the alpha head to 1")
parser.add_argument("--hazard-head", action="store_true",
                    help="a25/P3: train the hazard BCE on a dedicated "
                         "per-arm logit head (shared backbone) instead of "
                         "dragging the Beta mean -- frees alpha for the RL "
                         "signal so entropy can recover. FRESH RUNS ONLY "
                         "(actor state_dict gains keys; old checkpoints "
                         "will not load with this on)")
parser.add_argument("--clutch-tau-start", type=float, default=0.0,
                    help="a25/P3 soft clutch: executed alpha becomes "
                         "sigmoid((alpha-thr)/tau) during rollouts, tau "
                         "annealing geometrically from this value to "
                         "--clutch-tau-end over --clutch-tau-anneal-iters. "
                         "0.0 = hard binarize (a22/a23/a24 behavior). The "
                         "hard step makes reward flat in alpha on both "
                         "sides of thr; the ramp restores the gradient")
parser.add_argument("--clutch-tau-end", type=float, default=0.02)
parser.add_argument("--clutch-tau-anneal-iters", type=int, default=600)
parser.add_argument("--p2-obs", action="store_true",
                    help="a25/P2: append the 16-dim safety tail to the "
                         "policy obs (per-arm hazard/gray flags 8 + min "
                         "lock-line margin 4 + target backlog 4). The 531-d "
                         "obs has no explicit distance-to-lock-line signal; "
                         "the safety head was guessing. FRESH RUNS ONLY "
                         "(obs dim changes, old checkpoints incompatible)")
parser.add_argument("--a31-stage", choices=["off", "f", "u", "refine"], default="off")
parser.add_argument("--partner-checkpoint", default="")
parser.add_argument("--a31-replay-manifest", default="")
parser.add_argument("--w-coupling-sync", type=float, default=0.0)
parser.add_argument("--intervention-labels", action="store_true")
parser.add_argument("--intervention-label-class", choices=["any", "cross"], default="any",
                    help="a31b: 'any' (a31 2026-09-11 behaviour) labels every backstop-active "
                         "arm-step as hazard; 'cross' keeps only arm-steps whose retained CROSS "
                         "row is inside lock line + gray band (table/self projections excluded)")
parser.add_argument("--max-wall-seconds", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from safeduo.algo.lagrangian_ppo import (  # noqa: E402
    LagrangianPPO,
    LagrangianPPOConfig,
    cost_sync_from_env_cfg,
)
from safeduo.algo.reward_shaping import arm_class_near_flags  # noqa: E402
from safeduo.envs.duo_env import DuoEnv, make_duo_env_cfg  # noqa: E402
from safeduo.safety.types import ARM_KEYS, CLASS_CROSS  # noqa: E402
from safeduo.train.events import EventWriter  # noqa: E402
from safeduo.train.wandb_logger import WandbLogger  # noqa: E402


class _EnvAdapter:
    """DuoEnv already speaks the loop's step contract; this only pins
    num_envs/device attributes and keeps reset() signature stable."""

    def __init__(self, env: DuoEnv, stage="off", partner=None):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.stage = stage
        self.partner = partner
        self.obs = None
        if stage == "u" and partner is None:
            raise ValueError("a31 U stage requires frozen F partner")

    def reset(self):
        out = self.env.reset()
        self.obs = out[0]["policy"]
        return out

    def step(self, action: torch.Tensor):
        action = action.to(self.env.device)
        if self.stage == "f":
            action = action.clone()
            action[:, 2:4] = 1.0  # scripted U commands still pass through L2
        elif self.stage == "u":
            with torch.no_grad():
                partner_action = self.partner(self.obs)
            action = action.clone()
            # Deploy frozen F partner with the training soft-clutch transform.
            partner_alpha = (partner_action[:, :2] + 1.0) * .5
            action[:, :2] = torch.sigmoid((partner_alpha - .5) / .05) * 2 - 1
        out = self.env.step(action)
        self.obs = out[0]["policy"]
        return out

    def executed_alpha(self):
        return self.env._a31_transition_alpha

    def intervention_flags(self):
        return self.env._a31_transition_intervention

    def step_telemetry(self) -> dict:
        """Per-step diagnostics for the wandb safety/geometry panels (R1).

        Read-only view over duo_env's existing _step_cache/_last_out --
        the exact tensors the EventWriter hook already consumes -- so the
        numbers match the event shards without any recomputation. Called by
        LagrangianPPO.collect() once per env step, only when a wandb logger
        is attached. Clamp mirrors the events hook (-1, 10)."""
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
        # reward breakdown (C14): 0-dim env-means cached by _get_rewards;
        # keys become the wandb task/reward_{term} panel (REWARD_TERMS)
        for name, v in (c.get("reward_terms") or {}).items():
            frame[f"rew_{name}"] = v.detach().clone()
        return frame

    def hazard_flags(self) -> "torch.Tensor | None":
        """R18 (3): per-arm [hazard, gray] flags (N, 4, 2) bool cached by
        duo_env._get_observations on the POST-step state (same convention as
        cost_channels) when coordinator.r18_hazard_flags is on. None sends
        the trainer to a loud RuntimeError -- silent label dropout would
        train the BCE against garbage."""
        c = getattr(self.env, "_step_cache", None)
        if not c or "hazard_gray" not in c:
            return None
        return c["hazard_gray"].detach()

    def cost_channels(self) -> "torch.Tensor | None":
        """Exemption-aware per-class cost for the constraint channels (C14,
        C12_HANDOFF §3 option b). duo_env._get_observations caches it on the
        POST-step state (same state the trainer's obs_{t+1} reconstruction
        reads), with per-row d_min (active_dmin, per-link exact) and legal
        near_table-exempt rows dropped. None (cache not filled yet) sends
        the trainer to its obs fallback."""
        c = getattr(self.env, "_step_cache", None)
        if not c or "cost_by_class" not in c:
            return None
        return c["cost_by_class"].detach()


def main() -> None:
    run = args.run_name or datetime.now().strftime("c5_lag_%m%d_%H%M")
    log_dir = Path.home() / "safeduo" / "artifacts" / "runs" / run
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_duo_env_cfg(num_envs=args.num_envs, device=args.device, coordinator=True,
                           yaml_name=args.env_yaml,
                           # R15 ①: None keeps the yaml/profile value (off in
                           # every existing profile -> obs layout unchanged)
                           arm_aware_obs=True if args.arm_aware_obs else None,
                           # a25/P2: None = yaml/profile value (absent -> off)
                           p2_obs=True if args.p2_obs else None)
    cfg.seed = args.seed
    if args.a31_stage != "off" and not cfg.coordinator.get("coupling_obs"):
        raise ValueError("a31 stages require duo_env_a31.yaml")
    if args.a31_replay_manifest:
        cfg.coordinator["a31_replay_manifest"] = args.a31_replay_manifest
    if args.intervention_label_class == "cross" and not args.intervention_labels:
        raise ValueError("--intervention-label-class cross requires --intervention-labels")
    # Record the resolved environment, not just the inheritance filename.
    from safeduo.configs import load_config
    resolved = load_config(args.env_yaml)
    (log_dir / "arguments.json").write_text(json.dumps(vars(args), default=str, indent=2))
    # R15 ③a: alpha->1 utility pricing rides the env reward assembly (same
    # place as the other eight shaped terms and the only side that has the
    # pair->arm lookup for the per-arm danger gate). 0.0 = key absent = the
    # reward sum is bit-identical.
    # R18 (4) vs R15 (3)a: two alpha-pricing paths must not stack -- the
    # dead-zone penalty REPLACES alpha_util (old switch code stays for old
    # recipes, but one run picks one)
    assert not (args.w_alpha_util > 0.0 and args.w_alpha_deadzone > 0.0), \
        "--w-alpha-util and --w-alpha-deadzone are mutually exclusive (R18)"
    if args.w_alpha_util > 0.0:
        cfg.coordinator["reward"]["w_alpha_util"] = args.w_alpha_util
    # R18 (3): the hindsight labels need the env-side hazard/gray cache;
    # switch both sides together (trainer raises if the hook serves None)
    if args.w_hazard_bce > 0.0 or args.w_alpha_deadzone > 0.0:
        cfg.coordinator["r18_hazard_flags"] = True
    # a27: rendezvous exemption rides the same env-side label cache
    if args.hazard_coop_vmax > 0.0:
        cfg.coordinator["hazard_coop_vmax"] = args.hazard_coop_vmax
    # a25/P2: the 16-dim obs tail reads the same cache -- light it even if
    # BCE/deadzone are off (duo_env asserts the dependency)
    if args.p2_obs:
        cfg.coordinator["r18_hazard_flags"] = True
    # R2 trainer half (C13): forward the curriculum recipe. duo_env reads
    # coordinator["curriculum_yaml"] only in its delta_source == "l2_mix"
    # branch (duo_env.py:374-385); empty flag leaves the profile's own
    # default (delta_curriculum.yaml) in place -- zero behavior drift.
    if args.curriculum_yaml:
        cfg.coordinator["curriculum_yaml"] = args.curriculum_yaml
    # Lagrangian arm: margin cost rides the constraint channels, NOT the
    # reward (double-charging would re-create the v3 landscape disease the
    # recipe exists to fix). The 'off' ablation keeps the v3-style reward.
    if args.controller != "off":
        cfg.coordinator["reward"]["w_margin"] = 0.0
    resolved["coordinator"] = cfg.coordinator
    resolved["runtime"] = {"seed": cfg.seed, "num_envs": args.num_envs,
                           "observation_space": cfg.observation_space,
                           "device": str(cfg.sim.device)}
    (log_dir / "resolved_environment.json").write_text(json.dumps(resolved, indent=2))
    env = DuoEnv(cfg)

    # ---- event shards for the sentinel/gates (train_ppo's proven hook) ----
    writer = EventWriter(str(log_dir / "events"),
                         n_log_envs=int(cfg.events_cfg["n_log_envs"]),
                         flush_every=int(cfg.events_cfg["flush_every"]))
    orig_rewards = env._get_rewards
    tick = {"t": 0}
    if args.intervention_label_class == "cross" and not getattr(env, "_r18_hazard", False):
        raise ValueError("--intervention-label-class cross needs R18 hazard flags "
                         "(class d_min / gray band tables) on the env")

    def rewards_with_events():
        r = orig_rewards()
        c = env._step_cache
        if c:
            # Capture before DirectRLEnv autoreset can replace post-step state.
            env._a31_transition_alpha = c["alpha"].detach().clone()
            bs_now = c["bs_active"].detach().bool().clone()
            out = env._last_out
            if args.intervention_label_class == "cross":
                # a31b: only backstop activity adjacent to a retained CROSS row
                # counts as clutch evidence (a31 labelled table-class projections
                # as hazard -> 62% false brake; see arm_class_near_flags).
                bs_now = bs_now & arm_class_near_flags(
                    out.active_pairs, out.active_mask, out.active_dmin,
                    out.viol_exempt, env._sph.pair_arms,
                    env._r18_class_dmin, env._r18_gray_band, CLASS_CROSS)
            env._a31_transition_intervention = bs_now
            # full 28-column schema, mirroring train_ppo's hook byte-for-byte
            # (A7-W7: the sentinel/run-events gates only need alpha/p/reward,
            # but train_events_analysis + attribution need mm_table, per-arm
            # backstop and cmd/exec norms -- C5 handed the isaac wiring over
            # untested, the micro-smoke gate is "28 columns present")
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
            src = getattr(env, "_delta_src", None)
            if tick["t"] == 0 and src is not None \
                    and hasattr(src, "mix_fractions"):
                print("MIX_FRACTIONS " + json.dumps(src.mix_fractions),
                      flush=True)
            tick["t"] += 1
        return r

    env._get_rewards = rewards_with_events

    # d_warn / per-class d_min sync (C14, C12_HANDOFF §1/§2): single-source
    # read of the SAME safety section + semantics yaml duo_env just consumed,
    # so the trainer's obs-fallback cost channel and the env shaping band can
    # never diverge again (the duo_env_v5.yaml L78-81 fork). v4 profile
    # resolves to the old hardcoded numbers (0.05 / 0.03 / 0.02 / 0.02) --
    # zero drift; v6 resolves to {0.08, cross 0.038, self 0.013, table 0.020}.
    sync = cost_sync_from_env_cfg(cfg.safety_cfg, cfg.semantics_yaml)
    cost_scale = (tuple(float(x) for x in args.cost_scale.split(","))
                  if args.cost_scale else (1.0, 1.0, 1.0))
    assert len(cost_scale) == 3 and all(s > 0 for s in cost_scale), \
        f"--cost-scale needs 3 positive values, got {args.cost_scale!r}"
    cost_limits = (tuple(float(x) for x in args.cost_limits.split(","))
                   if args.cost_limits else (args.cost_limit,) * 3)
    assert len(cost_limits) == 3 and all(v > 0 for v in cost_limits), \
        f"--cost-limits needs 3 positive values, got {args.cost_limits!r}"
    lambda_max = (tuple(float(x) for x in args.lambda_max.split(","))
                  if args.lambda_max else None)
    assert lambda_max is None or len(lambda_max) == 3, \
        f"--lambda-max needs 3 values, got {args.lambda_max!r}"
    loop_cfg = LagrangianPPOConfig(
        gamma=args.gamma, controller=args.controller,
        cost_limits=cost_limits,
        cost_scale=cost_scale,
        lambda_max=lambda_max,
        entropy_coef=args.entropy_coef,
        entropy_coef_hi=args.entropy_coef_hi,
        entropy_ema_lo=args.ent_trigger,
        entropy_ema_hi=args.ent_release,
        ent_cost_gate=args.ent_cost_gate,
        arm_aware_obs=bool(cfg.coordinator.get("arm_aware_obs")),
        w_armbal=args.w_armbal,
        lambda_rate_max=args.lambda_rate_max,
        clutch_train=args.clutch_train,
        w_hazard_bce=args.w_hazard_bce,
        w_alpha_deadzone=args.w_alpha_deadzone,
        hazard_horizon=args.hazard_horizon,
        hazard_tail_mask=args.hazard_tail_mask,
        num_steps_per_env=args.num_steps_per_env,
        hazard_bce_balance=args.hazard_bce_balance,
        hazard_head=args.hazard_head,
        clutch_tau_start=args.clutch_tau_start,
        clutch_tau_end=args.clutch_tau_end,
        clutch_tau_anneal_iters=args.clutch_tau_anneal_iters,
        p2_obs=bool(cfg.coordinator.get("p2_obs")),
        coupling_obs=bool(cfg.coordinator.get("coupling_obs")),
        w_coupling_sync=args.w_coupling_sync,
        intervention_labels=args.intervention_labels,
        save_interval=args.save_interval,
        d_warn=sync["d_warn"], d_min_by_class=sync["d_min_by_class"])
    # R1 wandb telemetry: offline by default (WANDB_MODE overrides), lands
    # in <log_dir>/wandb for a later `wandb sync`; degrades to a no-op if
    # wandb is missing/broken so training never depends on it.
    wandb_logger = WandbLogger(
        run, log_dir=log_dir,
        config={"run": run, "num_envs": args.num_envs,
                "max_iterations": args.max_iterations, "seed": args.seed,
                "controller": args.controller, "gamma": args.gamma,
                "cost_limit": args.cost_limit, "cost_scale": cost_scale,
                "cost_limits": cost_limits, "lambda_max": lambda_max,
                "entropy_coef": args.entropy_coef,
                "entropy_coef_hi": args.entropy_coef_hi,
                "ent_trigger": args.ent_trigger,
                "ent_release": args.ent_release,
                "ent_cost_gate": args.ent_cost_gate,
                "arm_aware_obs": args.arm_aware_obs,
                "w_armbal": args.w_armbal,
                "w_alpha_util": args.w_alpha_util,
                "lambda_rate_max": args.lambda_rate_max,
                "clutch_train": args.clutch_train,
                "w_hazard_bce": args.w_hazard_bce,
                "hazard_coop_vmax": args.hazard_coop_vmax,
                "w_alpha_deadzone": args.w_alpha_deadzone,
                "hazard_horizon": args.hazard_horizon,
                "hazard_tail_mask": args.hazard_tail_mask,
                "num_steps_per_env": args.num_steps_per_env,
                "hazard_bce_balance": args.hazard_bce_balance,
                "hazard_head": args.hazard_head,
                "clutch_tau_start": args.clutch_tau_start,
                "clutch_tau_end": args.clutch_tau_end,
                "clutch_tau_anneal_iters": args.clutch_tau_anneal_iters,
                "p2_obs": args.p2_obs,
                "env_yaml": args.env_yaml,
                "curriculum_yaml": args.curriculum_yaml,
                "save_interval": args.save_interval})
    partner = None
    if args.partner_checkpoint:
        from safeduo.eval.block1_harness import ActorMLP
        partner = ActorMLP.from_checkpoint(args.partner_checkpoint).to(env.device).eval()
        for parameter in partner.parameters():
            parameter.requires_grad_(False)
    trainer = LagrangianPPO(_EnvAdapter(env, args.a31_stage, partner), loop_cfg, log_dir=log_dir,
                            device=str(env.device), seed=args.seed,
                            wandb_logger=wandb_logger)
    if args.resume:
        trainer.load(args.resume)
        print(f"RESUMED from {args.resume} at iter {trainer.iteration}",
              flush=True)
    t0 = time.time()
    history = trainer.learn(args.max_iterations, max_wall_seconds=args.max_wall_seconds)
    writer.flush()
    wandb_logger.finish()
    stats = {"run": run, "iterations": len(history), "requested_iterations": args.max_iterations, "seed": args.seed,
             "controller": args.controller, "num_envs": args.num_envs,
             "gamma": args.gamma, "env_yaml": args.env_yaml,
             "wall_s": round(time.time() - t0, 1)}
    # done marker written from python's success path (A6 lesson: `&& touch`
    # fires on kit crashes too)
    env.close()
    (log_dir / "done.json").write_text(json.dumps(stats))
    print("TRAIN_DONE " + json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
    app.close()
