"""p-head pin diagnosis (W6-3): why the r2 priority head sits at +1.000.

Three independent evidence lines, one CLI:

  events     executed-p statistics per training shard (train/events.py
             28-col schema): mean/std, pinned fractions, two-sidedness,
             tube overlap -- WHEN the head locked and whether it ever moved.
  ckpts      pre-clamp head outputs across model_*.pt on a fixed local obs
             battery (real geometry, oracle-filtered rollout): the executed
             p is clip(mean + noise, -1, 1) (RslRlVecEnvWrapper
             clip_actions=1.0), so once |mean| - 2*sigma > 1 every sample
             clips to the same corner -> events show std == 0 while the
             Gaussian itself may still be drifting. The battery separates
             "head output drifted past the clamp" from "head output pinned
             by data".
  oracle     demand density: how often the r2 traffic mix even ASKS for a
             nonzero priority. The fixed-rule oracle (strengthened CBF-QP
             core, the same label source as BC) is run locally on the
             VERBATIM r2 curriculum (configs/delta_curriculum.yaml) and on a
             conflict-weighted contrast mix, reporting frac(|p*| > 0.5)
             overall and per scenario family. This quantifies hypothesis 1
             (symmetric deadlocks too rare) and the alpha-domination ratio
             (p's reward exposure vs alpha's every-step exposure).

Usage (local, after rsyncing the run dir):
  PYTHONPATH=src .venv/bin/python -m safeduo.eval.p_head_diagnosis \
      --events-dir artifacts/analysis/raw/a5_v2_r2/events \
      --ckpt-dir artifacts/analysis/raw/a5_v2_r2 \
      --out artifacts/analysis/p_head_diagnosis
Any of the three parts can be skipped by omitting its flag; plots land in
--out (REMEMBER the A4 lesson: open the PNG and look at it).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

P_PIN_EPS = 1e-3         # executed |p| within this of +-1 counts as pinned
ORACLE_ENGAGE = 0.5      # |p*| above this = the oracle demands a yield sign


# ------------------------------------------------------------------ events

def shard_p_stats(events_dir: "str | Path", max_shards: int = 0) -> list:
    """Per-shard executed-p statistics (pyarrow-only, server-env safe)."""
    import pyarrow.parquet as pq

    files = sorted(Path(events_dir).glob("events_*.parquet"))
    if max_shards:
        files = files[:max_shards]
    if not files:
        raise FileNotFoundError(f"no event shards in {events_dir}")
    cols = ["t", "p", "tube"] + [f"alpha_{i}" for i in range(4)]
    out = []
    for f in files:
        tab = pq.read_table(f, columns=cols)

        def col(name):
            return torch.tensor(tab.column(name).to_numpy(
                zero_copy_only=False), dtype=torch.float32)

        p, tube = col("p"), col("tube") > 0.5
        alpha = torch.stack([col(f"alpha_{i}") for i in range(4)], dim=-1)
        t = col("t")
        rec = {
            "shard": f.name,
            "t_min": int(t.min().item()), "t_max": int(t.max().item()),
            "p_mean": float(p.mean().item()),
            "p_std": float(p.std().item()),
            "p_pin_hi": float((p >= 1.0 - P_PIN_EPS).float().mean().item()),
            "p_pin_lo": float((p <= -1.0 + P_PIN_EPS).float().mean().item()),
            "p_neg_frac": float((p < -0.05).float().mean().item()),
            "p_mid_frac": float((p.abs() < 0.9).float().mean().item()),
            "tube_frac": float(tube.float().mean().item()),
            "alpha_mean": float(alpha.mean().item()),
        }
        if tube.any():
            rec["p_mean_tube"] = float(p[tube].mean().item())
            rec["p_std_tube"] = float(p[tube].std().item())
            rec["alpha_arm_mean_tube"] = [
                round(float(v), 4) for v in alpha[tube].mean(dim=0)]
        out.append(rec)
    return out


# -------------------------------------------------------------- checkpoints

def _ckpt_iter(path: Path) -> int:
    m = re.search(r"model_(\d+)\.pt$", path.name)
    return int(m.group(1)) if m else -1


def checkpoint_p_battery(ckpt_paths: list, obs: torch.Tensor) -> list:
    """Pre-clamp actor outputs on a fixed obs battery, per checkpoint.

    Executed p = clip(mean + sigma * eps, -1, 1): the head is behaviorally
    saturated once mean - 2*sigma > 1 (virtually every sample clips to +1,
    the corner where the smoothness penalty is also zero). p_escape_frac
    measures the exploration mass that still lands below the clamp."""
    from safeduo.eval.block1_harness import ActorMLP

    out = []
    for path in sorted(ckpt_paths, key=_ckpt_iter):
        net = ActorMLP.from_checkpoint(path)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict", ck)
        sigma = sd.get("std")
        sigma_p = float(sigma[4].item()) if sigma is not None else float("nan")
        with torch.no_grad():
            a = net(obs.float())
        mu_p = a[:, 4]
        mu_alpha = a[:, :4]
        # P(clip(mu + sigma*eps) < 1) under eps ~ N(0,1), averaged over battery
        z = (1.0 - mu_p) / max(sigma_p, 1e-8)
        escape = 0.5 * (1.0 + torch.erf(z / 2.0 ** 0.5))
        out.append({
            "ckpt": Path(path).name, "iter": _ckpt_iter(Path(path)),
            "p_preclamp_mean": float(mu_p.mean().item()),
            "p_preclamp_std": float(mu_p.std().item()),
            "p_frac_mean_beyond_clamp": float((mu_p > 1.0).float().mean().item()),
            "p_sigma": sigma_p,
            "p_escape_frac": float(escape.mean().item()),
            "alpha_preclamp_mean": float(mu_alpha.mean().item()),
        })
    return out


def build_obs_battery(n_envs: int = 32, steps: int = 120, seed: int = 0,
                      curriculum: "str | None" = "delta_curriculum.yaml",
                      device: str = "cpu") -> torch.Tensor:
    """Fixed local obs battery: oracle-filtered rollout on the r2 curriculum
    (duo_env obs layout, 243-dim), the same machinery as the BC dataset."""
    from safeduo.algo.bc_warmstart_run import ASSEMBLY_PREP_Q, PRESETS
    from safeduo.algo.warmstart_oracle import BCDatasetConfig, build_bc_dataset
    from safeduo.baselines.real_geometry import RealGeometryProvider, SceneLayout
    from safeduo.delta.l1_random import JacobianMapper
    from safeduo.delta.l2_env_source import (
        ConflictMixSource,
        RealScenePoses,
        load_curriculum_cfg,
    )

    layout = SceneLayout(base_x=0.70)
    provider = RealGeometryProvider(n_envs, device=device, layout=layout)
    poses = RealScenePoses()
    f_sign = 1.0 if layout.base_pose("F_L")[0][0] > 0 else -1.0
    ws = poses.workspace_spec(f_sign=f_sign)
    mapper = JacobianMapper(provider.ee_jacobian, poses.ws_lo, poses.ws_hi)
    if curriculum:
        cfg = load_curriculum_cfg(curriculum)
    else:   # conflict-weighted contrast (the local_smoke recipe)
        p = PRESETS["local_smoke"]
        cfg = {"mix": p["mix"], "scenarios": p["scenarios"],
               "n_variants": p["n_variants"], "split": "train"}
    source = ConflictMixSource(n_envs, cfg, device=device, mapper=mapper,
                               ws=ws, amp_max=0.015)
    ds = build_bc_dataset(source, provider, n_envs,
                          BCDatasetConfig(steps=steps, dt=1.0 / 60.0,
                                          seed=seed, init_q=ASSEMBLY_PREP_Q,
                                          init_jitter=0.0),
                          obs_layout="duo_env")
    return ds["obs"]


OBS_BLOCKS = (("q", 0, 26), ("qd", 26, 52), ("active_pairs", 52, 180),
              ("active_mask", 180, 212), ("delta_cmd", 212, 238),
              ("alpha_prev", 238, 242), ("p_prev", 242, 243))
PAIR_SLOTS = ("dist", "closing_vel", "class_id", "pair_id")


def obs_column_audit(obs: torch.Tensor) -> dict:
    """Magnitude audit of the duo_env 243-dim observation blocks. The culprit
    this surfaced (2026-08-12): active_pairs carries the raw pair_id -- a
    NOMINAL identifier with absmax ~2569 / absmean ~215, three orders of
    magnitude above every physical feature -- so randomly initialized heads
    are born at |pre-clamp| ~ 50."""
    out = {}
    for name, lo, hi in OBS_BLOCKS:
        b = obs[:, lo:hi]
        out[name] = {"absmax": float(b.abs().max().item()),
                     "absmean": float(b.abs().mean().item())}
    ap = obs[:, 52:180].view(obs.shape[0], 32, 4)
    for j, nm in enumerate(PAIR_SLOTS):
        out[f"pairs_{nm}"] = {
            "absmax": float(ap[..., j].abs().max().item()),
            "absmean": float(ap[..., j].abs().mean().item())}
    return out


# ------------------------------------------------------------ oracle demand

def oracle_p_demand(n_envs: int = 64, steps: int = 240, seed: int = 0,
                    curriculum: "str | None" = "delta_curriculum.yaml",
                    scenarios: "dict | None" = None,
                    mix: "dict | None" = None,
                    device: str = "cpu") -> dict:
    """Fixed-rule oracle on a traffic mix -> how often |p*| engages, overall
    and per family (assignment is constant per env in a no-reset rollout, so
    the per-env cut IS the per-family cut)."""
    from safeduo.algo.bc_warmstart_run import ASSEMBLY_PREP_Q
    from safeduo.algo.warmstart_oracle import BCDatasetConfig, WarmStartOracle
    from safeduo.baselines.real_geometry import RealGeometryProvider, SceneLayout
    from safeduo.delta._contract_stub import ARM_KEYS
    from safeduo.delta.l1_random import JacobianMapper
    from safeduo.delta.l2_env_source import (
        ConflictMixSource,
        RealScenePoses,
        load_curriculum_cfg,
    )

    layout = SceneLayout(base_x=0.70)
    provider = RealGeometryProvider(n_envs, device=device, layout=layout)
    poses = RealScenePoses()
    f_sign = 1.0 if layout.base_pose("F_L")[0][0] > 0 else -1.0
    ws = poses.workspace_spec(f_sign=f_sign)
    mapper = JacobianMapper(provider.ee_jacobian, poses.ws_lo, poses.ws_hi)
    if curriculum:
        cfg = load_curriculum_cfg(curriculum)
    else:
        cfg = {"mix": mix or {"l1": 0.0, "l2": 1.0}, "n_variants": 8,
               "split": "train"}
        if scenarios:
            cfg["scenarios"] = scenarios
    source = ConflictMixSource(n_envs, cfg, device=device, mapper=mapper,
                               ws=ws, amp_max=0.015)
    dcfg = BCDatasetConfig(steps=steps, dt=1.0 / 60.0, seed=seed,
                           init_q=ASSEMBLY_PREP_Q, init_jitter=0.0)
    g = torch.Generator().manual_seed(dcfg.seed)
    oracle = WarmStartOracle(n_envs, provider)
    oracle.reset(torch.arange(n_envs))
    source.reset(torch.arange(n_envs), g)
    q = {a: torch.as_tensor(v, dtype=torch.float32).expand(n_envs, -1).clone()
         for a, v in ASSEMBLY_PREP_Q.items()}
    qd = {a: torch.zeros_like(q[a]) for a in ARM_KEYS}
    fam = [source.names[i] for i in source.assignment.tolist()]
    engaged = torch.zeros(steps, n_envs, dtype=torch.bool)
    gated = torch.zeros(steps, n_envs, dtype=torch.bool)
    for t in range(steps):
        state = provider.scene_state(q, qd, dt=dcfg.dt)
        cmd = source.sample(state)
        alpha_star, p_star, exec_ = oracle.labels_and_exec(state, cmd)
        engaged[t] = p_star.abs() > ORACLE_ENGAGE
        gated[t] = (alpha_star < 0.9).any(dim=-1)
        q = {a: q[a] + exec_[a] for a in ARM_KEYS}
        qd = {a: exec_[a] / dcfg.dt for a in ARM_KEYS}
    fams = sorted(set(fam))
    per_family = {}
    for f in fams:
        idx = torch.tensor([i for i, x in enumerate(fam) if x == f])
        per_family[f] = {
            "n_envs": int(idx.numel()),
            "p_engaged_frac": float(engaged[:, idx].float().mean().item()),
            "alpha_gated_frac": float(gated[:, idx].float().mean().item()),
        }
    p_frac = float(engaged.float().mean().item())
    a_frac = float(gated.float().mean().item())
    return {
        "n_envs": n_envs, "steps": steps,
        "curriculum": curriculum or "custom",
        "mix_fractions": source.mix_fractions,
        "p_engaged_frac": p_frac,
        "alpha_gated_frac": a_frac,
        # exposure ratio: how many alpha-shaped steps per p-shaped step the
        # PPO gradient sees (the alpha-domination number, hypothesis 2)
        "alpha_to_p_exposure": (a_frac / p_frac) if p_frac > 0 else float("inf"),
        "per_family": per_family,
    }


# ------------------------------------------------------------------- report

def plot_diagnosis(shards: list, ckpts: list, out_dir: Path) -> "Path | None":
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    if shards:
        t = [s["t_max"] for s in shards]
        ax = axes[0, 0]
        ax.plot(t, [s["p_mean"] for s in shards], label="executed p mean")
        ax.fill_between(t,
                        [s["p_mean"] - s["p_std"] for s in shards],
                        [s["p_mean"] + s["p_std"] for s in shards], alpha=0.3)
        ax.axhline(1.0, color="r", ls="--", lw=0.8)
        ax.set_title("executed p over training (events)")
        ax.set_xlabel("env step")
        ax.legend()
        ax = axes[0, 1]
        ax.plot(t, [s["p_pin_hi"] for s in shards], label="frac p = +1")
        ax.plot(t, [s["p_neg_frac"] for s in shards], label="frac p < -0.05")
        ax.plot(t, [s["tube_frac"] for s in shards], label="tube frac")
        ax.set_title("pin fraction / two-sidedness / tube")
        ax.set_xlabel("env step")
        ax.legend()
    if ckpts:
        it = [c["iter"] for c in ckpts]
        ax = axes[1, 0]
        mu = [c["p_preclamp_mean"] for c in ckpts]
        sg = [c["p_sigma"] for c in ckpts]
        ax.plot(it, mu, marker="o", label="pre-clamp p mean (battery)")
        ax.fill_between(it, [m - 2 * s for m, s in zip(mu, sg)],
                        [m + 2 * s for m, s in zip(mu, sg)], alpha=0.25,
                        label="+-2 sigma (exploration)")
        ax.axhline(1.0, color="r", ls="--", lw=0.8, label="clamp")
        ax.axhline(-1.0, color="r", ls="--", lw=0.8)
        ax.set_title("actor p head vs clamp (checkpoints)")
        ax.set_xlabel("iteration")
        ax.legend()
        ax = axes[1, 1]
        ax.plot(it, [c["p_escape_frac"] for c in ckpts], marker="o",
                label="P(sample < +1)")
        ax.plot(it, [c["p_sigma"] for c in ckpts], marker="s",
                label="sigma_p")
        ax.set_title("exploration escape mass below the clamp")
        ax.set_xlabel("iteration")
        ax.legend()
    fig.tight_layout()
    path = out_dir / "p_head_diagnosis.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events-dir", default="")
    ap.add_argument("--ckpt-dir", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--battery-envs", type=int, default=32)
    ap.add_argument("--battery-steps", type=int, default=120)
    ap.add_argument("--demand-envs", type=int, default=64)
    ap.add_argument("--demand-steps", type=int, default=240)
    ap.add_argument("--skip-oracle", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    if args.events_dir:
        print("[p-diag] events pass...", flush=True)
        report["shards"] = shard_p_stats(args.events_dir)
    if args.ckpt_dir:
        print("[p-diag] building obs battery (r2 curriculum)...", flush=True)
        obs = build_obs_battery(args.battery_envs, args.battery_steps)
        report["obs_audit"] = obs_column_audit(obs)
        ckpts = sorted(Path(args.ckpt_dir).glob("model_*.pt"))
        print(f"[p-diag] probing {len(ckpts)} checkpoints...", flush=True)
        report["checkpoints"] = checkpoint_p_battery(ckpts, obs)
    if not args.skip_oracle:
        print("[p-diag] oracle demand: r2 curriculum...", flush=True)
        report["oracle_demand_r2_mix"] = oracle_p_demand(
            args.demand_envs, args.demand_steps)
        print("[p-diag] oracle demand: conflict-weighted contrast...",
              flush=True)
        report["oracle_demand_conflict_weighted"] = oracle_p_demand(
            args.demand_envs, args.demand_steps, curriculum=None,
            mix={"l1": 0.1, "l2": 0.9},
            scenarios={"head_on_crossing": 2.0, "center_grab": 2.0,
                       "chase": 2.0, "handover_approach": 1.0,
                       "sweep_across": 1.0, "table_slam": 1.0})
    (out_dir / "p_head_diagnosis.json").write_text(
        json.dumps(report, indent=2))
    fig = plot_diagnosis(report.get("shards", []),
                         report.get("checkpoints", []), out_dir)
    print(f"[p-diag] wrote {out_dir}/p_head_diagnosis.json"
          + (f" and {fig}" if fig else " (matplotlib unavailable, no plot)"),
          flush=True)


if __name__ == "__main__":
    main()
