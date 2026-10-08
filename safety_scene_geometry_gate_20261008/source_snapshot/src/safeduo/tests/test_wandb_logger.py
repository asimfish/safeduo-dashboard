"""wandb_logger unit + learn-loop integration tests (R1, C13).

Run locally (no Isaac needed):
  PYTHONPATH=src .venv/bin/pytest src/safeduo/tests/test_wandb_logger.py -q

Covers the two hard requirements of the R1 wiring:
- offline mode actually lands a syncable run under <log_dir>/wandb;
- every degradation path (wandb missing / init failure / log failure) is a
  silent no-op -- the training process must never crash on telemetry.
Plus the zero-drift discipline: stats.jsonl written by learn() carries no
wandb-only keys even when a logger is attached.
"""

from __future__ import annotations

import json
import sys

import pytest

from safeduo.train.wandb_logger import WandbLogger


def test_offline_run_lands_in_log_dir(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    monkeypatch.delenv("WANDB_MODE", raising=False)  # exercise the default
    logger = WandbLogger("c13_test_offline", config={"seed": 0},
                         log_dir=tmp_path)
    assert logger.active
    for it in range(5):
        logger.log(it, {"task/reward_mean": -1.0 + 0.1 * it,
                        "safety/lambda_cross": 0.01 * it,
                        "sys/amp_max": 0.03})
    logger.log_checkpoint(4, tmp_path / "model_4.pt", score=-0.6)
    logger.finish()
    wandb_dir = tmp_path / "wandb"
    assert wandb_dir.is_dir(), "offline run must land under <log_dir>/wandb"
    runs = [p for p in wandb_dir.iterdir() if "offline-run" in p.name]
    assert runs, f"no offline-run-* dir in {list(wandb_dir.iterdir())}"


def test_noop_when_wandb_not_installed(tmp_path, monkeypatch, capsys):
    # sys.modules[name] = None makes `import wandb` raise ImportError --
    # simulates a server env where wandb was never pip-installed
    monkeypatch.setitem(sys.modules, "wandb", None)
    logger = WandbLogger("c13_test_missing", log_dir=tmp_path)
    assert not logger.active
    logger.log(0, {"task/reward_mean": 0.0})
    logger.log_checkpoint(0, tmp_path / "model_0.pt", score=0.0)
    logger.finish()  # none of the above may raise
    out = capsys.readouterr().out
    assert "WARNING" in out and out.count("WARNING") == 1


def test_noop_when_init_fails(tmp_path, monkeypatch):
    wandb = pytest.importorskip("wandb")

    def boom(**_kwargs):
        raise RuntimeError("simulated init failure")

    monkeypatch.setattr(wandb, "init", boom)
    logger = WandbLogger("c13_test_initfail", log_dir=tmp_path)
    assert not logger.active
    logger.log(0, {"x": 1.0})
    logger.finish()


def test_log_failure_degrades_instead_of_raising(capsys):
    class _BrokenRun:
        def log(self, *_a, **_k):
            raise OSError("disk full")

    logger = WandbLogger.__new__(WandbLogger)
    logger._run = _BrokenRun()
    logger._warned = False
    logger.log(0, {"x": 1.0})  # must not raise
    assert not logger.active
    assert "WARNING" in capsys.readouterr().out


def test_learn_loop_offline_integration(tmp_path, monkeypatch):
    """Two tiny iterations on the kinematic env with a live offline logger:
    wandb dir appears, checkpoints logged, stats.jsonl stays drift-free."""
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    from safeduo.algo.kinematic_env import KinematicDuoEnv, KinematicEnvConfig
    from safeduo.algo.lagrangian_ppo import LagrangianPPO, LagrangianPPOConfig

    env = KinematicDuoEnv(4, cfg=KinematicEnvConfig(episode_length=30), seed=0)
    run_dir = tmp_path / "run"
    logger = WandbLogger("c13_test_learn", config={"n_envs": 4},
                         log_dir=tmp_path)
    trainer = LagrangianPPO(
        env,
        LagrangianPPOConfig(num_steps_per_env=8, num_learning_epochs=2,
                            num_mini_batches=2, save_interval=1),
        log_dir=run_dir, seed=0, wandb_logger=logger)
    history = trainer.learn(2, quiet=True)
    logger.finish()

    assert len(history) == 2
    assert (tmp_path / "wandb").is_dir()
    # zero-drift discipline: the sidecar files learn() always wrote are
    # unchanged, and the wandb-only health keys never leak into stats.jsonl
    lines = (run_dir / "stats.jsonl").read_text().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert "approx_kl" not in rec and "clip_frac" not in rec
    assert {"iter", "reward_mean", "multipliers", "cost_means",
            "alpha_mean", "wall_s"} <= set(rec)
    assert (run_dir / "model_last.pt").exists()
