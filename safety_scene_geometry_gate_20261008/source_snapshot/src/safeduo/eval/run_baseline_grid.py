"""One-shot baseline tuning grid (experiments.csv grid_* planned rows).

Enumerates the registered 8+8+8+4 configs x 3 seeds, rolls each on the L2
conflict battery (ConflictMixSource, l2-only, TRAIN split -- eval split stays
untouched for the C1 tables), computes the full ACCEPTANCE metrics via the
eval runner, and writes:

  artifacts/grid/<family>/results.csv          one row per (config, seed)
  artifacts/grid/<family>/agg.csv              per-config mean over seeds
  artifacts/grid/<family>/<config>_s<seed>.json  full metric dicts

GPU etiquette (Round-25.5 scheduling): the stack is pure torch and runs fine
on CPU; on bjxy_5090 run it only in a v2-training lull, check nvidia-smi
first, and prefer --device cpu unless a GPU is actually idle.

Usage (server, canonical env):
  source ~/safeduo_setup/env.sh && cd ~/safeduo && \
    PYTHONPATH=src python -m safeduo.eval.run_baseline_grid \
      --families estop speed cbf cbf_fine --n-envs 32 --steps 600 \
      --out artifacts/grid [--register]
  # --register appends done-rows to artifacts/experiments.csv (off by default)
v4 rescan (C5-W7, strong-coupling real layout; CPU fine):
    ... --scene v4 --families estop speed cbf cbf_fine
  # writes artifacts/grid_v4/ by default; init_jitter forced to 0 (razor-thin
  # v4 birth window, see SCENES note below)
v5 rescan (C6-W8, B2 bundle: de-inflated spheres / gap=0 / manifest d_min):
    ... --scene v5 --families estop estop_lo estop_v5lo speed speed_lo \
        speed_v5lo cbf cbf_fine        # -> artifacts/grid_v5/
    ... --scene v5eqd --families cbf   # d_min sensitivity arm -> grid_v5eqd/

Local smoke: --n-envs 4 --steps 60 --seeds 1 --families estop
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import torch

from safeduo.baselines.estop import EStopConfig, EStopFilter
from safeduo.baselines.speed_scaling import SpeedScalingConfig, SpeedScalingFilter
from safeduo.baselines.strong_cbf_qp import StrongCBFQPConfig, StrongCBFQPFilter
from safeduo.eval.runner import EvalConfig, run_eval
from safeduo.eval.metrics import compute_all_metrics

REPO_ROOT = Path(__file__).resolve().parents[3]

# summary columns pulled into results.csv (full dicts land in the JSONs)
KEY_METRICS = (
    "collision_episodes", "episodes", "collision_rate_ub95",
    "intervention_rate", "tracking_rmse", "time_to_clear_mean",
    "time_to_clear_p95", "oscillations_per_episode",
    "projected_progress_conflict", "backstop_triggers_per_episode",
    "stall_events",
)


def grid_configs(family: str, cbf_best_gamma: float = 4.0) -> list:
    """(config_id, ctor_kwargs) pairs; counts pinned by tests to the
    experiments.csv registration (8/8/8/4 + lo-extensions 6/4).

    The *_lo extensions exist because of a measured geometry fact (C3-W5
    dry-run, STATUS_C): the FR3 rest pose's tightest self pair sits at margin
    0.0322 m under B's deliberately fat spheres, so every registered estop
    config (d_stop >= 0.04 > 0.0322) latches the F arms at birth and every
    speed config crawls them; hysteresis release additionally requires
    d_resume < 0.0322 (a stopped arm freezes its own self margins -- E-stop
    blocks retreat too, so a latch on a self row can never self-release).
    The lo ranges probe the only releasable regime; the structural
    freeze-or-late dilemma itself is a finding for baseline_matrix.md.
    """
    if family == "estop":
        return [(f"estop_ds{ds:g}_dr{ds + off:g}",
                 {"d_stop": ds, "d_resume": ds + off})
                for ds, off in itertools.product((0.04, 0.05, 0.06, 0.08),
                                                 (0.02, 0.03))]
    if family == "estop_lo":
        return [(f"estop_ds{ds:g}_dr{ds + off:g}",
                 {"d_stop": ds, "d_resume": ds + off})
                for ds, off in itertools.product((0.01, 0.015, 0.02),
                                                 (0.005, 0.01))]
    if family == "estop_v4lo":
        # v4 extension of the C3 lo-precedent: the v4 prep pose's tightest
        # (self, hand<->forearm) margin is 5.4 mm, so every d_stop >= 0.01
        # latches all four arms at birth and a latched arm freezes its own
        # self margin -> permanent. Only d_stop < 5.4 mm can even start.
        return [(f"estop_ds{ds:g}_dr{ds + off:g}",
                 {"d_stop": ds, "d_resume": ds + off})
                for ds, off in itertools.product((0.002, 0.003, 0.005),
                                                 (0.002, 0.005))]
    if family == "estop_v5lo":
        # v5 working-point band (C6-W8): de-inflation lifts the birth floor
        # to ~19 mm (table, U link2 end sphere; self_F 23.1 / wrist ring
        # 25.3-26.6 mm), so d_stop < 18 mm starts unlatched -- the first
        # scene generation where a sub-floor estop threshold is ABOVE the
        # braking caliber (d_min_v5 13 mm) instead of below it. Probes
        # {8, 10} mm (late regime, < d_min) and {13, 15} mm (at/above).
        return [(f"estop_ds{ds:g}_dr{ds + off:g}",
                 {"d_stop": ds, "d_resume": ds + off})
                for ds, off in itertools.product((0.008, 0.010, 0.013, 0.015),
                                                 (0.002, 0.004))]
    if family == "speed":
        return [(f"speed_dslow{dsl:g}_floor{fl:g}",
                 {"d_slow": dsl, "alpha_floor": fl, "ema_hz": 2.0})
                for dsl, fl in itertools.product((0.10, 0.15, 0.20, 0.25),
                                                 (0.0, 0.05))]
    if family == "speed_lo":
        return [(f"speed_dslow{dsl:g}_floor{fl:g}",
                 {"d_slow": dsl, "alpha_floor": fl, "ema_hz": 2.0})
                for dsl, fl in itertools.product((0.05, 0.075),
                                                 (0.0, 0.05))]
    if family == "speed_v4lo":
        # v4 extension (same rationale as estop_v4lo): d_slow above the
        # 5.4 mm birth floor crawls every arm from step 0
        return [(f"speed_dslow{dsl:g}_floor{fl:g}",
                 {"d_slow": dsl, "alpha_floor": fl, "ema_hz": 2.0})
                for dsl, fl in itertools.product((0.01, 0.02, 0.03),
                                                 (0.0, 0.05))]
    if family == "speed_v5lo":
        # v5 working-point band (C6-W8): alpha floor pinned to the v5
        # braking caliber (cfg d_min 0.013, family default was 0.03) and
        # d_slow around the ~19-26 mm birth floors so slowdown engages
        # after birth instead of crawling from step 0.
        return [(f"speed_dm13_dslow{dsl:g}_floor{fl:g}",
                 {"d_min": 0.013, "d_slow": dsl, "alpha_floor": fl,
                  "ema_hz": 2.0})
                for dsl, fl in itertools.product((0.03, 0.05, 0.08),
                                                 (0.0, 0.05))]
    if family == "cbf":
        return [(f"cbf_g{g:g}_{ex}", {"gamma": g, "extrap": ex})
                for g, ex in itertools.product((2.0, 4.0, 6.0, 8.0),
                                               ("linear", "zero"))]
    if family == "cbf_fine":
        return [(f"cbffine_db{db:g}_rate{rt:g}",
                 {"gamma": cbf_best_gamma, "prio_deadband": db,
                  "prio_rate": rt})
                for db, rt in itertools.product((1e-4, 2e-4), (0.1, 0.2))]
    raise ValueError(f"unknown family {family}")


def make_filter(family: str, kwargs: dict, n_envs: int, provider,
                device: str = "cpu"):
    if family in ("estop", "estop_lo", "estop_v4lo", "estop_v5lo"):
        return EStopFilter(n_envs, provider, EStopConfig(**kwargs),
                           device=device)
    if family in ("speed", "speed_lo", "speed_v4lo", "speed_v5lo"):
        return SpeedScalingFilter(n_envs, provider,
                                  SpeedScalingConfig(**kwargs), device=device)
    if family in ("cbf", "cbf_fine"):
        return StrongCBFQPFilter(n_envs, provider,
                                 StrongCBFQPConfig(**kwargs), device=device)
    raise ValueError(family)


def default_provider_fn(n_envs: int, device: str):
    """Real 26-DoF geometry at the arbitrated +-0.70 layout."""
    from safeduo.baselines.real_geometry import RealGeometryProvider, SceneLayout

    return RealGeometryProvider(n_envs, device=device,
                                layout=SceneLayout(base_x=0.70))


def v4_provider_fn(n_envs: int, device: str):
    """v4 real scene: JAKA Zu7 + F2 hands + scene_layout_v3 (C5-W7)."""
    from safeduo.baselines.real_geometry import make_v4_provider

    return make_v4_provider(n_envs, device=device)


def v5_provider_fn(n_envs: int, device: str):
    """v5 bundle scene (C6-W8): de-inflated spheres + gap=0 layout + v4.1
    births + manifest d_min caliber (self/cross 0.013 / table 0.020)."""
    from safeduo.baselines.real_geometry import make_v5_provider

    return make_v5_provider(n_envs, device=device)


def v5eqd_provider_fn(n_envs: int, device: str):
    """v5 sensitivity arm: same scene, cbf braking boundary at B2's
    same-real-buffer equivalence values (self 0.028 / cross 0.038) --
    isolates how much of the cbf intervention-tax change is the d_min
    caliber vs the geometry itself. Only the cbf family consumes rows.d_min,
    so scanning estop/speed under this scene is redundant."""
    from safeduo.baselines.real_geometry import make_v5_provider

    return make_v5_provider(n_envs, device=device, dmin_caliber="equivalence")


def _source_fn(env_yaml: str):
    """L2-only conflict battery, TRAIN split (never the eval split here)."""
    def fn(n_envs: int, provider, device: str):
        from safeduo.delta.l1_random import JacobianMapper
        from safeduo.delta.l2_env_source import ConflictMixSource, RealScenePoses

        poses = RealScenePoses(env_yaml)
        f_sign = 1.0 if provider.layout.base_pose("F_L")[0][0] > 0 else -1.0
        ws = poses.workspace_spec(f_sign=f_sign)
        mapper = JacobianMapper(provider.ee_jacobian, poses.ws_lo, poses.ws_hi)
        cfg = {"mix": {"l1": 0.0, "l2": 1.0}, "split": "train", "n_variants": 100}
        return ConflictMixSource(n_envs, cfg, device=device, mapper=mapper,
                                 ws=ws, amp_max=0.015)
    return fn


default_source_fn = _source_fn("duo_env.yaml")

# scene -> (provider_fn, source_fn, init_jitter, default out subdir).
# v4 init_jitter MUST be 0: the strong-coupling prep pose sits 5.4 mm from
# the nearest sphere boundary and duo_env resets to the EXACT pose; measured
# birth-violation rate under uniform joint jitter (C5-W7, 256 envs):
# 0.01 rad -> 16%, 0.02 -> 44%, 0.05 -> 57%. Episode diversity in v4 comes
# from scenario variants + delta traffic, not the birth pose.
# v5 keeps init_jitter=0 for protocol continuity (duo_env resets exact; the
# v5 floor is ~19 mm so small jitter would be survivable, but the eval axis
# must not change two things at once) and reuses the v4 env yaml for the
# conflict source: scene_layout_v5 keeps workspace/base anchors/table top
# unchanged, so RealScenePoses reads identical waypoint distributions --
# geometry differences enter exclusively through the provider.
SCENES = {
    "v0": (default_provider_fn, default_source_fn, 0.05, "grid"),
    "v4": (v4_provider_fn, _source_fn("duo_env_v4.yaml"), 0.0, "grid_v4"),
    "v5": (v5_provider_fn, _source_fn("duo_env_v4.yaml"), 0.0, "grid_v5"),
    "v5eqd": (v5eqd_provider_fn, _source_fn("duo_env_v4.yaml"), 0.0,
              "grid_v5eqd"),
}


def run_grid(families: list, n_envs: int = 32, steps: int = 600,
             seeds: tuple = (0, 1, 2), device: str = "cpu",
             out_root: "Path | str | None" = None,
             provider_fn=None,
             source_fn=None,
             register: bool = False,
             cbf_best_gamma: float = 4.0,
             scene: str = "v0") -> "object":
    import pandas as pd

    scene_provider, scene_source, init_jitter, subdir = SCENES[scene]
    provider_fn = provider_fn or scene_provider
    source_fn = source_fn or scene_source
    out_root = Path(out_root or (REPO_ROOT / "artifacts" / subdir))
    all_rows = []
    for family in families:
        fam_dir = out_root / family
        fam_dir.mkdir(parents=True, exist_ok=True)
        for cfg_id, kwargs in grid_configs(family, cbf_best_gamma):
            for seed in seeds:
                t0 = time.time()
                provider = provider_fn(n_envs, device)
                source = source_fn(n_envs, provider, device)
                filt = make_filter(family, kwargs, n_envs, provider, device)
                batch, _ = run_eval(
                    source, filt, provider, n_envs,
                    EvalConfig(steps=steps, dt=1.0 / 60.0, seed=seed,
                               init_jitter=init_jitter))
                met = compute_all_metrics(batch)
                row = {"family": family, "config": cfg_id, "seed": seed,
                       "n_envs": n_envs, "steps": steps,
                       "wall_s": round(time.time() - t0, 1),
                       **{k: met[k] for k in KEY_METRICS}}
                all_rows.append(row)
                (fam_dir / f"{cfg_id}_s{seed}.json").write_text(
                    json.dumps({**row, "metrics_full": met}, indent=2))
                print(f"[grid] {cfg_id} s{seed}: "
                      f"coll={met['collision_episodes']}/{met['episodes']} "
                      f"interv={met['intervention_rate']:.3f} "
                      f"ttc={met['time_to_clear_mean']:.2f}s "
                      f"({row['wall_s']}s)", flush=True)
        fam_rows = [r for r in all_rows if r["family"] == family]
        df = pd.DataFrame(fam_rows)
        df.to_csv(fam_dir / "results.csv", index=False)
        num_cols = [c for c in df.columns
                    if c not in ("family", "config", "seed")]
        df.groupby("config")[num_cols].mean().to_csv(fam_dir / "agg.csv")
    result = pd.DataFrame(all_rows)
    if register and len(result):
        _register_rows(result, out_root, scene)
    return result


def _register_rows(result, out_root: Path, scene: str = "v0") -> None:
    """Append done-rows to experiments.csv (called only with --register)."""
    csv = REPO_ROOT / "artifacts" / "experiments.csv"
    stamp = time.strftime("%Y%m%d", time.gmtime())
    tag = "" if scene == "v0" else f"{scene}_"
    battery = "L2-train" if scene == "v0" else f"L2-train-{scene}"
    lines = []
    for family, sub in result.groupby("family"):
        n_cfg = sub["config"].nunique()
        n_seed = sub["seed"].nunique()
        best = sub.groupby("config")["intervention_rate"].mean().idxmin()
        lines.append(
            f"grid_{tag}{family}_run_{stamp},B1,{family},{battery},{n_seed},"
            f"local-exec,done,{out_root / family}/,"
            f"executed {n_cfg} configs x {n_seed} seeds; "
            f"best-by-intervention={best}; see agg.csv")
    with open(csv, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", nargs="+",
                    default=["estop", "speed", "cbf", "cbf_fine"],
                    choices=["estop", "estop_lo", "estop_v4lo", "estop_v5lo",
                             "speed", "speed_lo", "speed_v4lo", "speed_v5lo",
                             "cbf", "cbf_fine"])
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--cbf-best-gamma", type=float, default=4.0,
                    help="gamma used by the cbf_fine family (set after "
                         "inspecting the cbf family agg)")
    ap.add_argument("--scene", choices=list(SCENES), default="v0",
                    help="v0 = +-0.70 UR5e legacy; v4 = real strong-coupling "
                         "layout (JAKA+F2, init_jitter forced 0)")
    args = ap.parse_args()
    run_grid(args.families, n_envs=args.n_envs, steps=args.steps,
             seeds=tuple(range(args.seeds)), device=args.device,
             out_root=args.out or None, register=args.register,
             cbf_best_gamma=args.cbf_best_gamma, scene=args.scene)


if __name__ == "__main__":
    main()
