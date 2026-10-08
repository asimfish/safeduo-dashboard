"""BC-init no-freeze smoke check (acceptance gate #3, handed to A3).

Three independently usable checks, all CLI-wired:

  checkpoint check   loads the v2 checkpoint, verifies the recorded holdout
                     acceptance gates and that the distilled p row is real
                     (r1 lesson: a zero/degenerate p row or saturated head
                     must fail here, before any GPU time is spent)
  run-events check   reads the FIRST event shard(s) of a launched PPO run
                     (train/events.py schema) and applies the no-freeze
                     gates on the earliest `window` steps:
                       alpha mean in [0.3, 0.9]
                       every arm's mean alpha > 0.05   (r1: three arms at 0.02)
                       alpha std > 0.02                (not collapsed)
                       p not pinned (modal-extreme fraction < 0.95) and
                       two-sided (>=1% of steps beyond +-0.05 on each side;
                       init_noise_std=0.5 guarantees this for a healthy init)
  live-rollout check (W6-2a; C5-W7 re-aimed at the LAUNCH OBJECT) drives the
                     checkpoint CLOSED-LOOP and compares the live alpha/p
                     behavior against the checkpoint's own recorded open-loop
                     holdout stats. Default target since W7 is the COMPOSITE
                     = transplant actor + exploration noise (what PPO
                     executes at iter 0); c4_bc_protect_ctrl proved the W6
                     compat-net gate validates the wrong object (compat 0.81
                     bimodal / p_pin 0.24 vs run-measured composite alpha
                     [0.14, 1.00, 0.14, 0.13] / p_pin 0.94). The A4 signature
                     this catches: holdout alpha 0.585 / p ~0 but live
                     closed-loop alpha=0.998 / p=-0.98 from the very first
                     step -- every holdout gate green while the net
                     full-throttles the real env, which then feeds the PPO
                     freeze attractor. Gates:
                       |mean(alpha_live) - holdout alpha_pred_mean| <= 0.25
                       frac(alpha_live > 0.95) <= 0.5   (full-throttle lock)
                       frac(alpha_live < 0.05) <= 0.5   (freeze lock)
                       frac(|p_live| > 0.9)    <= 0.5   (p pinned)
                     Backends: "kinematic" (local: real-geometry alpha-scaled
                     closed loop, no Isaac) and "isaac" (server: live duo_env,
                     the faithful pre-launch gate -- A4's triage procedure
                     productized; run it before every GO).

Usage (server, after the first shard lands ~4 min into training):
  PYTHONPATH=src python -m safeduo.algo.bc_smoke_check \
      --run-events ~/safeduo/artifacts/runs/<run>/events [--window 512]
  PYTHONPATH=src python -m safeduo.algo.bc_smoke_check --ckpt <model.pt>
  # pre-launch closed-loop gate (server, ~1 min at 64 env x 100 steps CPU):
  PYTHONPATH=src python -m safeduo.algo.bc_smoke_check \
      --live-rollout <model.pt> --live-backend isaac
Exit code 0 = PASS, 1 = FAIL (wire into the launch watchdog directly) --
EXCEPT under the isaac backend, where kit teardown overwrites the SystemExit
code (measured 08-12: FAIL verdict, exit 0). The isaac path therefore also
writes live_gate_verdict.json next to the checkpoint; judge THAT (the
launcher does).

Measured true positive (08-12, W6): server_v2_20260812_002737/model.pt --
the exact artifact whose transplant froze the A4 launch -- fails the isaac
live gate with alpha_sat_hi 0.698 / p_pinned 0.923 while all its holdout
gates are green. Verdict file kept next to the checkpoint.

The r1 failure signature (alpha per-arm [0.62, 0.02, 0.02, 0.02], p == +1.000
at every step) is pinned as a mandatory FAIL in tests/test_bc_smoke_check.py;
the A4 live-saturation signature likewise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ALPHA_MEAN_BAND = (0.3, 0.9)
ALPHA_ARM_MIN = 0.05
ALPHA_STD_MIN = 0.02
P_PIN_FRAC_MAX = 0.95
P_TWO_SIDED_MIN = 0.01

# live closed-loop gate thresholds (A4 failure: shift 0.41 / sat-hi 1.0 /
# p-pin 1.0; healthy local_smoke holdout: alpha mean 0.43 std 0.25 -> both
# saturation fractions are small for any non-degenerate net)
LIVE_ALPHA_SHIFT_MAX = 0.25
LIVE_ALPHA_SAT_HI_MAX = 0.5
LIVE_ALPHA_SAT_LO_MAX = 0.5
LIVE_P_PIN_MAX = 0.5


def check_checkpoint(ckpt_path: "str | Path") -> dict:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    out = {"ckpt": str(ckpt_path), "format": ck.get("format", "bc_v1?")}
    checks = {}
    checks["is_v2_format"] = ck.get("format") == "bc_v2"
    acc = ck.get("acceptance", {})
    checks["alpha_gate_recorded_pass"] = bool(acc.get("alpha_gate_pass"))
    checks["p_dir_gate_recorded_pass"] = bool(acc.get("p_dir_gate_pass"))
    sd = ck.get("state_dict", {})
    p_w = sd.get("head_p.weight")
    checks["p_row_nonzero"] = bool(p_w is not None
                                   and p_w.abs().max().item() > 1e-6)
    fit = ck.get("distill_fit", {})
    checks["distill_alpha_mae_ok"] = bool(
        fit.get("alpha_distill_mae", 1.0) < 0.05)
    checks["distill_p_dir_agree_ok"] = bool(
        fit.get("p_dir_agree", 0.0) > 0.95)
    out["checks"] = checks
    out["acceptance"] = acc
    out["distill_fit"] = fit
    out["pass"] = all(checks.values())
    return out


def check_run_events(events_dir: "str | Path", window: int = 512) -> dict:
    """Gates on the earliest `window` steps of a (possibly live) run.

    Deliberately pandas-free: the server canonical env ships pyarrow (the
    event WRITER uses it) but not pandas -- found 08-12 when this script
    first ran there. torch + pyarrow only."""
    import pyarrow.parquet as pq

    files = sorted(Path(events_dir).glob("events_*.parquet"))
    if not files:
        return {"pass": False, "error": f"no event shards in {events_dir} "
                "(first flush lands after 512 env steps ~ iter 21)"}
    cols = ["t"] + [f"alpha_{i}" for i in range(4)] + ["p"]
    chunks = []
    got = 0
    for f in files:
        tab = pq.read_table(f, columns=cols)
        chunks.append(tab)
        got = int(max(tab.column("t").to_pylist())) + 1
        if got >= window:
            break
    import pyarrow as pa

    tab = pa.concat_tables(chunks)
    t_col = torch.tensor(tab.column("t").to_numpy(zero_copy_only=False),
                         dtype=torch.long)
    keep = t_col < window
    if not bool(keep.any()):
        # C6-W8 (A8 finding): the watchdog's rolling archiver prunes early
        # shards locally (keeps only the newest N), so at final-eval time the
        # earliest available t may already exceed the gate window -- the old
        # code crashed on p.max() over an empty selection. The gate itself is
        # fine; it just needs the full archive.
        t_lo = int(t_col.min().item()) if t_col.numel() else 0
        return {
            "pass": False, "events_dir": str(events_dir), "steps_checked": 0,
            "error": (
                f"no rows with t < window={window} (earliest available "
                f"t={t_lo}): early shards were pruned by the rolling "
                "archiver -- point --run-events at the full archive "
                "(NAS safeduo_archive/runs/<run>_events/)"),
        }
    alpha = torch.stack([
        torch.tensor(tab.column(f"alpha_{i}").to_numpy(zero_copy_only=False),
                     dtype=torch.float32)[keep] for i in range(4)], dim=-1)
    p = torch.tensor(tab.column("p").to_numpy(zero_copy_only=False),
                     dtype=torch.float32)[keep]
    steps_checked = int(t_col[keep].max().item()) + 1 if keep.any() else 0

    arm_means = alpha.mean(dim=0)
    a_mean = float(alpha.mean().item())
    p_hi = float((p > 0.05).float().mean().item())
    p_lo = float((p < -0.05).float().mean().item())
    pin = max(float((p >= p.max() - 1e-6).float().mean().item()),
              float((p <= p.min() + 1e-6).float().mean().item()))
    checks = {
        "alpha_mean_in_band": ALPHA_MEAN_BAND[0] <= a_mean <= ALPHA_MEAN_BAND[1],
        "no_arm_frozen": bool((arm_means > ALPHA_ARM_MIN).all().item()),
        "alpha_not_collapsed": float(alpha.std().item()) > ALPHA_STD_MIN,
        "p_not_pinned": pin < P_PIN_FRAC_MAX,
        "p_two_sided": p_hi >= P_TWO_SIDED_MIN and p_lo >= P_TWO_SIDED_MIN,
    }
    return {
        "events_dir": str(events_dir), "steps_checked": steps_checked,
        "alpha_mean": round(a_mean, 4),
        "alpha_arm_means": [round(float(v), 4) for v in arm_means],
        "alpha_std": round(float(alpha.std().item()), 4),
        "p_frac_pos": round(p_hi, 4), "p_frac_neg": round(p_lo, 4),
        "p_pinned_frac": round(pin, 4),
        "checks": checks,
        "pass": all(checks.values()),
    }


# ------------------------------------------------- live closed-loop gate (W6)

def load_compat_net(ckpt_path: "str | Path"):
    """v2 checkpoint "state_dict" (normalizer folded, raw-obs) ->
    CoordinatorMLP ready for closed-loop driving."""
    from safeduo.algo.warmstart_oracle import CoordinatorMLP

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = CoordinatorMLP(int(ck["obs_dim"]), int(ck["hidden"]))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck


def load_transplant_actor(ckpt_path: "str | Path", p_neutral: bool = False):
    """The LAUNCH OBJECT (C5-W7 instrument fix, c4_bc_protect_ctrl root
    cause): PPO does not execute the BC CoordinatorMLP -- it executes the
    rsl_rl actor AFTER train/bc_init.load_bc_actor_init's transplant (alpha
    rows x0.5 + env clamp mapping) with exploration noise on top. The A5/A6
    failure: the compat net gated 0.81 bimodal / p_pin 0.24 while the real
    composite ran alpha [0.14, 1.00, 0.14, 0.13] / p_pin 0.94 in the freeze
    window. This builds the same actor MLP train_ppo constructs
    (actor_hidden_dims=[H,H], elu) and applies the same transplant code."""
    from safeduo.train.bc_init import load_bc_actor_init

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hidden, obs_dim = int(ck["hidden"]), int(ck["obs_dim"])
    actor = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, hidden), torch.nn.ELU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ELU(),
        torch.nn.Linear(hidden, 5))
    meta = load_bc_actor_init(actor, str(ckpt_path), p_neutral=p_neutral)
    actor.eval()
    return actor, ck, meta


def composite_action(actor, obs: torch.Tensor, noise_std: float,
                     gen: "torch.Generator | None" = None) -> torch.Tensor:
    """One PPO-iter-0 action: raw actor mean + N(0, sigma) exploration noise,
    clipped to +-1 (RslRlVecEnvWrapper clip_actions=1.0). duo_env re-derives
    alpha=(a+1)/2, p=a4 from this."""
    raw = actor(obs.float())
    noise = torch.randn(raw.shape, generator=gen) * noise_std
    return (raw + noise).clamp(-1.0, 1.0)


def closedloop_gate(alpha_live: torch.Tensor, p_live: torch.Tensor,
                    acceptance: dict) -> dict:
    """Pure gate: live closed-loop predictions vs recorded open-loop holdout
    stats. alpha_live (S, 4) or (S,), p_live (S,)."""
    a = alpha_live.flatten().float()
    p = p_live.flatten().float()
    hold_mean = acceptance.get("alpha_pred_mean")
    if hold_mean is None:
        return {"pass": False,
                "error": "checkpoint acceptance lacks alpha_pred_mean "
                         "(pre-v2 checkpoint?) -- live gate needs the "
                         "holdout reference"}
    shift = abs(float(a.mean().item()) - float(hold_mean))
    sat_hi = float((a > 0.95).float().mean().item())
    sat_lo = float((a < 0.05).float().mean().item())
    p_pin = float((p.abs() > 0.9).float().mean().item())
    checks = {
        "alpha_mean_shift_ok": shift <= LIVE_ALPHA_SHIFT_MAX,
        "alpha_not_saturated_high": sat_hi <= LIVE_ALPHA_SAT_HI_MAX,
        "alpha_not_saturated_low": sat_lo <= LIVE_ALPHA_SAT_LO_MAX,
        "p_not_pinned_live": p_pin <= LIVE_P_PIN_MAX,
    }
    return {
        "alpha_live_mean": round(float(a.mean().item()), 4),
        "alpha_holdout_mean": round(float(hold_mean), 4),
        "alpha_mean_shift": round(shift, 4),
        "alpha_sat_hi_frac": round(sat_hi, 4),
        "alpha_sat_lo_frac": round(sat_lo, 4),
        "p_pinned_frac": round(p_pin, 4),
        "n_samples": int(a.numel()),
        "checks": checks,
        "pass": all(checks.values()),
    }


def collect_closedloop_kinematic(net, n_envs: int = 16, steps: int = 120,
                                 seed: int = 0, device: str = "cpu") -> dict:
    """Local closed-loop collector: real-geometry conflict traffic, execution
    = the net's own alpha-scaled command (its first-order budget effect), obs
    fed back with the net's own previous (alpha, p) -- deployment semantics.
    No Isaac; catches self-induced distribution drift. The FAITHFUL gate for
    duo_env obs quirks is the isaac backend."""
    from safeduo.algo.bc_warmstart_run import ASSEMBLY_PREP_Q
    from safeduo.algo.warmstart_oracle import obs_features_duo_env
    from safeduo.baselines.real_geometry import RealGeometryProvider, SceneLayout
    from safeduo.delta._contract_stub import ARM_KEYS
    from safeduo.delta.l1_random import JacobianMapper
    from safeduo.delta.l2_env_source import ConflictMixSource, RealScenePoses

    dt = 1.0 / 60.0
    layout = SceneLayout(base_x=0.70)
    provider = RealGeometryProvider(n_envs, device=device, layout=layout)
    poses = RealScenePoses()
    f_sign = 1.0 if layout.base_pose("F_L")[0][0] > 0 else -1.0
    ws = poses.workspace_spec(f_sign=f_sign)
    mapper = JacobianMapper(provider.ee_jacobian, poses.ws_lo, poses.ws_hi)
    cfg = {"mix": {"l1": 0.1, "l2": 0.9}, "n_variants": 8, "split": "train",
           "scenarios": {"head_on_crossing": 2.0, "center_grab": 2.0,
                         "chase": 2.0, "handover_approach": 1.0}}
    source = ConflictMixSource(n_envs, cfg, device=device, mapper=mapper,
                               ws=ws, amp_max=0.015)
    g = torch.Generator().manual_seed(seed)
    source.reset(torch.arange(n_envs), g)
    q = {a: torch.as_tensor(v, dtype=torch.float32).expand(n_envs, -1).clone()
         for a, v in ASSEMBLY_PREP_Q.items()}
    qd = {a: torch.zeros_like(q[a]) for a in ARM_KEYS}
    alpha_prev = torch.ones(n_envs, 4)
    p_prev = torch.zeros(n_envs)
    A, P = [], []
    with torch.no_grad():
        for _ in range(steps):
            state = provider.scene_state(q, qd, dt=dt)
            cmd = source.sample(state)
            obs = obs_features_duo_env(state, cmd, alpha_prev, p_prev)
            alpha, p = net(obs.float())
            A.append(alpha)
            P.append(p)
            exec_ = {a: alpha[:, i:i + 1] * cmd.delta_q[a]
                     for i, a in enumerate(ARM_KEYS)}
            q = {a: q[a] + exec_[a] for a in ARM_KEYS}
            qd = {a: exec_[a] / dt for a in ARM_KEYS}
            alpha_prev, p_prev = alpha, p
    return {"alpha": torch.cat(A), "p": torch.cat(P)}


def collect_closedloop_isaac(net, n_envs: int = 64, steps: int = 100,
                             device: str = "cpu") -> dict:
    """Server closed-loop collector: the live duo_env (A4's triage procedure).
    Caller must have launched the kit app (main() does)."""
    from safeduo.envs.duo_env import DuoEnv, make_duo_env_cfg
    from safeduo.eval.block1_harness import filter_output_to_action

    cfg = make_duo_env_cfg(num_envs=n_envs, device=device, coordinator=True)
    env = DuoEnv(cfg)
    obs, _ = env.reset()
    A, P = [], []
    with torch.no_grad():
        for _ in range(steps):
            alpha, p = net(obs["policy"].float().cpu())
            A.append(alpha)
            P.append(p)
            action = filter_output_to_action(alpha, p).to(env.device)
            obs, _, _, _, _ = env.step(action)
    return {"alpha": torch.cat(A), "p": torch.cat(P)}


def collect_composite_kinematic(actor, noise_std: float = 0.5,
                                n_envs: int = 16, steps: int = 120,
                                seed: int = 0, device: str = "cpu") -> dict:
    """Composite (transplant actor + exploration noise) closed loop on the
    local kinematic env -- the launch object driven the way PPO drives it.
    Records the EXECUTED alpha/p (post noise + clamp + env mapping)."""
    from safeduo.algo.kinematic_env import KinematicDuoEnv, KinematicEnvConfig

    env = KinematicDuoEnv(n_envs, cfg=KinematicEnvConfig(episode_length=steps),
                          device=device, seed=seed)
    gen = torch.Generator().manual_seed(seed + 1)
    obs_d, _ = env.reset()
    A, P = [], []
    with torch.no_grad():
        for _ in range(steps):
            a = composite_action(actor, obs_d["policy"], noise_std, gen)
            A.append((a[:, :4] + 1.0) * 0.5)
            P.append(a[:, 4])
            obs_d, _r, _t, _tr, _e = env.step(a)
    return {"alpha": torch.cat(A), "p": torch.cat(P)}


def collect_composite_isaac(actor, noise_std: float = 0.5,
                            n_envs: int = 64, steps: int = 100,
                            device: str = "cpu") -> dict:
    """Composite closed loop on the live duo_env (server pre-launch gate).
    Caller must have launched the kit app (main() does)."""
    from safeduo.envs.duo_env import DuoEnv, make_duo_env_cfg

    cfg = make_duo_env_cfg(num_envs=n_envs, device=device, coordinator=True)
    env = DuoEnv(cfg)
    obs, _ = env.reset()
    gen = torch.Generator().manual_seed(1)
    A, P = [], []
    with torch.no_grad():
        for _ in range(steps):
            a = composite_action(actor, obs["policy"].float().cpu(),
                                 noise_std, gen)
            A.append((a[:, :4] + 1.0) * 0.5)
            P.append(a[:, 4])
            obs, _r, _t, _tr, _e = env.step(a.to(env.device))
    return {"alpha": torch.cat(A), "p": torch.cat(P)}


def check_live_closedloop(ckpt_path: "str | Path", backend: str = "kinematic",
                          n_envs: int = 16, steps: int = 120, seed: int = 0,
                          device: str = "cpu", target: str = "composite",
                          noise_std: float = 0.5,
                          p_neutral: bool = False) -> dict:
    """target='composite' (C5-W7 default) gates the transplant actor +
    exploration noise -- the object PPO actually executes; 'compat' keeps
    the W6 behavior (the BC CoordinatorMLP itself) for comparison/forensics."""
    if target == "composite":
        actor, ck, meta = load_transplant_actor(ckpt_path, p_neutral=p_neutral)
        if backend == "kinematic":
            roll = collect_composite_kinematic(actor, noise_std, n_envs=n_envs,
                                               steps=steps, seed=seed,
                                               device=device)
        elif backend == "isaac":
            roll = collect_composite_isaac(actor, noise_std, n_envs=n_envs,
                                           steps=steps, device=device)
        else:
            raise ValueError(f"unknown live backend {backend}")
    elif target == "compat":
        net, ck = load_compat_net(ckpt_path)
        if backend == "kinematic":
            roll = collect_closedloop_kinematic(net, n_envs=n_envs, steps=steps,
                                                seed=seed, device=device)
        elif backend == "isaac":
            roll = collect_closedloop_isaac(net, n_envs=n_envs, steps=steps,
                                            device=device)
        else:
            raise ValueError(f"unknown live backend {backend}")
    else:
        raise ValueError(f"unknown live target {target}")
    out = closedloop_gate(roll["alpha"], roll["p"], ck.get("acceptance", {}))
    out.update({"ckpt": str(ckpt_path), "backend": backend, "target": target,
                "noise_std": (noise_std if target == "composite" else None),
                "n_envs": n_envs, "steps": steps})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--run-events", default="")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--live-rollout", default="",
                    help="checkpoint for the closed-loop behavior gate")
    ap.add_argument("--live-backend", choices=["kinematic", "isaac"],
                    default="kinematic")
    ap.add_argument("--live-target", choices=["composite", "compat"],
                    default="composite",
                    help="composite = transplant actor + exploration noise "
                         "(the launch object, C5-W7 default); compat = the "
                         "BC net itself (W6 behavior, forensics only)")
    ap.add_argument("--live-noise-std", type=float, default=0.5,
                    help="exploration noise of the composite (train_ppo "
                         "init_noise_std)")
    ap.add_argument("--live-p-neutral", action="store_true",
                    help="mirror a --bc_p_neutral launch in the composite")
    ap.add_argument("--live-envs", type=int, default=0,
                    help="0 = backend default (kinematic 16, isaac 64)")
    ap.add_argument("--live-steps", type=int, default=0,
                    help="0 = backend default (kinematic 120, isaac 100)")
    ap.add_argument("--live-device", default="cpu")
    args = ap.parse_args()
    if not args.ckpt and not args.run_events and not args.live_rollout:
        ap.error("give --ckpt, --run-events and/or --live-rollout")
    ok = True
    # flush every verdict print: kit teardown swallows unflushed stdout
    # (the A4 lesson train_ppo already guards against), and the isaac
    # backend's verdict additionally lands in a JSON file for the watchdog
    if args.ckpt:
        r = check_checkpoint(args.ckpt)
        print(json.dumps(r, indent=2, default=str), flush=True)
        ok &= r["pass"]
    if args.run_events:
        r = check_run_events(args.run_events, window=args.window)
        print(json.dumps(r, indent=2, default=str), flush=True)
        ok &= r["pass"]
    if args.live_rollout:
        defaults = {"kinematic": (16, 120), "isaac": (64, 100)}
        d_env, d_steps = defaults[args.live_backend]
        if args.live_backend == "isaac":
            from isaaclab.app import AppLauncher

            AppLauncher(headless=True)      # kit must live for the rollout
        r = check_live_closedloop(
            args.live_rollout, backend=args.live_backend,
            n_envs=args.live_envs or d_env, steps=args.live_steps or d_steps,
            device=args.live_device, target=args.live_target,
            noise_std=args.live_noise_std, p_neutral=args.live_p_neutral)
        print(json.dumps(r, indent=2, default=str), flush=True)
        verdict = Path(args.live_rollout).with_name("live_gate_verdict.json")
        verdict.write_text(json.dumps(r, indent=2, default=str))
        print(f"[live-gate] verdict -> {verdict}", flush=True)
        ok &= r["pass"]
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
