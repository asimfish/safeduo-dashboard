"""Warm-start BC on L2 conflict rollouts — v2 pipeline (W5 fix work order).

v1 history: two server runs archived under artifacts/warmstart/ failed A3's
transplant (r1 froze three arms at iter 70; the pose-aligned regen scored
alpha MSE 0.299 vs 0.219 mean baseline). Root causes and the v2 fixes:

  spawn mismatch   -> dataset now spawns at the ASSEMBLY §2 prep poses
                      (identical to duo_env after A3's W3 fix) with
                      init_jitter=0.0 (duo_env reset has no joint noise)
  raw 243-dim obs  -> ObsNormalizer, folded into the exported first layer
  tanh/sigmoid+MSE -> Beta-NLL heads (algo/bc_pretrain_v2), conflict-weighted
                      p dimension
  p* only {0,+1}   -> oracle tiebreak randomization: half the envs label with
                      prio_tiebreak=+1, half with -1 (the tiebreak is an
                      arbitrary convention; flipping it yields symmetric-
                      deadlock labels in BOTH directions without touching
                      the obs — cleaner than mirror augmentation)
  traffic mix      -> server preset uses configs/delta_curriculum.yaml
                      verbatim = the a5_v2_r2 training mix (l1:l2 0.5:0.5)

Checkpoint v2 keeps the train/bc_init.py contract: "state_dict" holds the
distilled CoordinatorMLP view (normalizer folded, REAL p row), so A3 loads it
unchanged and --bc_p_neutral should not be needed. Acceptance gates are
computed on a holdout and stored in the checkpoint + summary.json; run
algo/bc_smoke_check.py after the PPO launch for the no-freeze gate.

Usage:
  PYTHONPATH=src .venv/bin/python -m safeduo.algo.bc_warmstart_run --preset local_smoke
  # server (canonical env; CPU is fine, check nvidia-smi before using a GPU):
  source ~/safeduo_setup/env.sh && cd ~/safeduo && \
    PYTHONPATH=src python -m safeduo.algo.bc_warmstart_run --preset server
  # pipeline ablations on the same data: --pipeline v2 | v2_nonorm | v1_legacy
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from safeduo.algo.bc_pretrain_v2 import (
    BCV2Config,
    bc_pretrain_v2,
    evaluate_bc,
    save_checkpoint_v2,
)
from safeduo.algo.warmstart_oracle import BCDatasetConfig, build_bc_dataset
from safeduo.baselines.real_geometry import RealGeometryProvider, SceneLayout
from safeduo.baselines.strong_cbf_qp import StrongCBFQPConfig
from safeduo.delta.l1_random import JacobianMapper
from safeduo.delta.l2_env_source import ConflictMixSource, RealScenePoses

REPO_ROOT = Path(__file__).resolve().parents[3]

# ASSEMBLY §2 spawn poses == duo_env W3 init (A3 fix): FR3 prep position
# (j4=-2.31/j6=1.74 toward the workspace) + UR5e cuRobo retract
# (lift=-2.2/wrist1=-1.383). real_geometry.FR3_INIT_Q/UR5E_INIT_Q constants
# are still the stale W2 values (1.3 rad away on j6) — kept there because
# parity baselines reference them; BC data must use the env truth below.
ASSEMBLY_PREP_Q = {
    "F_L": (0.0, -0.569, 0.0, -2.31, 0.0, 1.74, 0.741),
    "F_R": (0.0, -0.569, 0.0, -2.31, 0.0, 1.74, 0.741),
    "U_L": (0.0, -2.2, 1.9, -1.383, -1.57, 0.0),
    "U_R": (0.0, -2.2, 1.9, -1.383, -1.57, 0.0),
}

PRESETS = {
    # tiny end-to-end check (used by tests; ~seconds)
    "tiny": dict(n_envs=4, steps=60, epochs=3, batch_size=128, hidden=32,
                 n_variants=4, curriculum=None,
                 mix={"l1": 0.0, "l2": 1.0},
                 scenarios={"head_on_crossing": 1.0, "chase": 1.0}),
    # local Mac CPU validation run (~3-5 min), conflict-weighted so BOTH
    # label kinds appear (W3 findings: conflicts need >=4 s; family lottery
    # needs enough envs -- 24 = two tiebreak halves of 12). This is the
    # pipeline-acceptance preset.
    "local_smoke": dict(n_envs=24, steps=300, epochs=25, batch_size=1024,
                        hidden=128, n_variants=16, curriculum=None,
                        mix={"l1": 0.1, "l2": 0.9},
                        scenarios={"head_on_crossing": 2.0, "center_grab": 2.0,
                                   "chase": 2.0, "handover_approach": 1.0,
                                   "sweep_across": 1.0, "table_slam": 1.0}),
    # server dataset for the real warm-start checkpoint: EXACT a5_v2_r2
    # training mix (configs/delta_curriculum.yaml, l1:l2 0.5:0.5)
    "server": dict(n_envs=128, steps=400, epochs=40, batch_size=2048,
                   hidden=256, n_variants=100, curriculum="delta_curriculum.yaml",
                   mix=None, scenarios=None),
}


def _build_half(n_envs: int, steps: int, seed: int, tiebreak: float,
                preset: dict, device: str) -> dict:
    """One dataset half with a fixed oracle tiebreak sign."""
    layout = SceneLayout(base_x=0.70)
    provider = RealGeometryProvider(n_envs, device=device, layout=layout)
    poses = RealScenePoses()
    f_sign = 1.0 if layout.base_pose("F_L")[0][0] > 0 else -1.0
    ws = poses.workspace_spec(f_sign=f_sign)
    mapper = JacobianMapper(provider.ee_jacobian, poses.ws_lo, poses.ws_hi)
    if preset["curriculum"]:
        from safeduo.delta.l2_env_source import load_curriculum_cfg

        cfg = load_curriculum_cfg(preset["curriculum"])
    else:
        cfg = {"mix": preset["mix"], "n_variants": preset["n_variants"],
               "split": "train"}
        if preset["scenarios"]:
            cfg["scenarios"] = preset["scenarios"]
    source = ConflictMixSource(n_envs, cfg, device=device, mapper=mapper,
                               ws=ws, amp_max=0.015)
    return build_bc_dataset(
        source, provider, n_envs,
        BCDatasetConfig(steps=steps, dt=1.0 / 60.0, seed=seed,
                        init_q=ASSEMBLY_PREP_Q, init_jitter=0.0),
        oracle_cfg=StrongCBFQPConfig(prio_tiebreak=tiebreak),
        obs_layout="duo_env")


def build_dataset_v2(preset: dict, seed: int, device: str) -> dict:
    """Two tiebreak halves -> concatenated dataset with two-sided p* labels."""
    half = max(1, preset["n_envs"] // 2)
    ds_pos = _build_half(half, preset["steps"], seed, +1.0, preset, device)
    ds_neg = _build_half(half, preset["steps"], seed + 1, -1.0, preset, device)
    return {k: torch.cat([ds_pos[k], ds_neg[k]]) for k in ds_pos}


def run(preset: str, device: str = "cpu", seed: int = 0,
        out_root: "Path | None" = None, pipeline: str = "v2") -> dict:
    p = PRESETS[preset]
    t0 = time.time()
    ds = build_dataset_v2(p, seed, device)
    t_ds = time.time() - t0
    a, pp = ds["alpha_star"], ds["p_star"]
    label_stats = {
        "n_samples": int(ds["obs"].shape[0]), "obs_dim": int(ds["obs"].shape[1]),
        "dataset_seconds": round(t_ds, 1),
        "alpha_gated_frac": round((a < 0.9).float().mean().item(), 4),
        "p_pos_frac": round((pp > 0.5).float().mean().item(), 4),
        "p_neg_frac": round((pp < -0.5).float().mean().item(), 4),
        # measured 08-12 (local_smoke): real-geometry conflicts resolve via
        # the aggressor rule far more often than the symmetric tiebreak, and
        # small rollouts can label only one direction. One-sided data still
        # beats a zeroed p row, but check this flag before trusting the p
        # transplant on a small dataset.
        "p_two_sided_labels": bool((pp > 0.5).any() and (pp < -0.5).any()),
    }
    out_dir = (out_root or (REPO_ROOT / "artifacts" / "warmstart")) / \
        f"{preset}_{pipeline}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if pipeline in ("v2", "v2_nonorm"):
        cfg = BCV2Config(epochs=p["epochs"], batch_size=p["batch_size"],
                         hidden=p["hidden"], seed=seed,
                         normalize=(pipeline == "v2"))
        model, normalizer, report = bc_pretrain_v2(ds, cfg, device=device)
        save_checkpoint_v2(out_dir / "model.pt", model, normalizer,
                           ds["obs"][:4096].float(), report)
        summary = {**label_stats, "pipeline": pipeline, "preset": preset,
                   "seed": seed, "losses_first_last": [round(report["losses"][0], 4),
                                                       round(report["losses"][-1], 4)],
                   **{k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in report["holdout"].items()}}
    elif pipeline == "v1_legacy":   # ablation baseline: the broken W2 recipe
        from safeduo.algo.warmstart_oracle import bc_pretrain

        model, losses = bc_pretrain(ds, epochs=p["epochs"],
                                    batch_size=p["batch_size"],
                                    hidden=p["hidden"], seed=seed,
                                    device=device)
        with torch.no_grad():
            a_hat, p_hat = model(ds["obs"].float().to(device))
        mse_a = (a_hat.cpu() - a).pow(2).mean().item()
        base_a = (a.mean(0, keepdim=True) - a).pow(2).mean().item()
        conflict = pp.abs() > 0.5
        dir_acc = (torch.sign(p_hat.cpu()[conflict])
                   == torch.sign(pp[conflict])).float().mean().item() \
            if conflict.any() else float("nan")
        summary = {**label_stats, "pipeline": pipeline, "preset": preset,
                   "seed": seed,
                   "losses_first_last": [round(losses[0], 4), round(losses[-1], 4)],
                   "mse_alpha": round(mse_a, 4),
                   "mse_alpha_baseline": round(base_a, 4),
                   "alpha_gate_pass": bool(mse_a < 0.75 * base_a),
                   "p_dir_acc_conflict": round(dir_acc, 4),
                   "p_sat_frac": round((p_hat.abs() > 0.99).float()
                                       .mean().item(), 4)}
        torch.save({"state_dict": model.state_dict(), "format": "bc_v1",
                    "obs_dim": ds["obs"].shape[1], "hidden": p["hidden"],
                    "obs_layout": "duo_env"}, out_dir / "model.pt")
    else:
        raise ValueError(f"unknown pipeline {pipeline}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    summary["out_dir"] = str(out_dir)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="local_smoke")
    ap.add_argument("--pipeline", choices=["v2", "v2_nonorm", "v1_legacy"],
                    default="v2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    summary = run(args.preset, device=args.device, seed=args.seed,
                  pipeline=args.pipeline)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
