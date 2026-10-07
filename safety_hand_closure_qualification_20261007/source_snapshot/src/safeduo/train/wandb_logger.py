"""Thin, crash-proof wandb wrapper for the SafeDuo trainers (R1, C13).

Design constraints (MASTER_REPORT §5 / owner P0):
- offline-first: WANDB_MODE defaults to "offline" so every run lands under
  <log_dir>/wandb/offline-run-* and is shipped later with `wandb sync`
  (server network flakiness must never block training). Exporting
  WANDB_MODE=online/disabled before launch overrides the default.
- never crashes training: wandb not installed, init failure, or a failure
  inside any log call all degrade to a permanent no-op with ONE printed
  warning. The trainer keeps running either way.
- parallel to EventWriter/stats.jsonl, not a replacement: the freeze
  sentinel and run-events gates keep consuming the binary shards untouched.

Usage:
    logger = WandbLogger(run_name, config=vars(args), log_dir=log_dir)
    logger.log(iteration, {"task/reward_mean": r, ...})
    logger.log_checkpoint(iteration, path, score=r)
    logger.finish()
"""

from __future__ import annotations

import os
from pathlib import Path


class WandbLogger:
    """No-op-degradable wandb run handle (see module docstring)."""

    def __init__(self, run_name: str, config: "dict | None" = None,
                 log_dir: "str | Path | None" = None,
                 project: str = "safeduo", mode: "str | None" = None):
        self._run = None
        self._warned = False
        mode = mode or os.environ.get("WANDB_MODE", "offline")
        try:
            import wandb

            os.environ["WANDB_MODE"] = mode
            # keep the trainer's stdout clean (LAG_PPO lines are parsed by
            # the ops tooling); override by exporting WANDB_SILENT=false
            os.environ.setdefault("WANDB_SILENT", "true")
            kwargs = dict(project=project, name=run_name,
                          config=dict(config or {}), mode=mode)
            if log_dir is not None:
                Path(log_dir).mkdir(parents=True, exist_ok=True)
                kwargs["dir"] = str(log_dir)
            self._run = wandb.init(**kwargs)
        except Exception as exc:  # includes ImportError: telemetry is optional
            self._disable(f"init failed ({type(exc).__name__}: {exc})")

    # ---- state ----------------------------------------------------------
    @property
    def active(self) -> bool:
        return self._run is not None

    def _disable(self, why: str) -> None:
        self._run = None
        if not self._warned:
            print(f"[wandb_logger] WARNING: {why}; telemetry degraded to "
                  "no-op, training continues", flush=True)
            self._warned = True

    # ---- logging ----------------------------------------------------------
    def log(self, step: int, metrics: dict) -> None:
        """Log one iteration's metric dict at the given step. Never raises."""
        if self._run is None:
            return
        try:
            self._run.log(dict(metrics), step=int(step))
        except Exception as exc:
            self._disable(f"log failed ({type(exc).__name__}: {exc})")

    def log_checkpoint(self, iteration: int, path: "str | Path",
                       score: "float | None" = None) -> None:
        """Record checkpoint metadata (artifacts panel). Never raises."""
        if self._run is None:
            return
        try:
            m = {"artifacts/checkpoint_iter": int(iteration)}
            if score is not None:
                m["artifacts/checkpoint_score"] = float(score)
            self._run.log(m, step=int(iteration))
            self._run.summary["last_checkpoint"] = str(path)
        except Exception as exc:
            self._disable(f"log_checkpoint failed ({type(exc).__name__}: {exc})")

    def finish(self) -> None:
        if self._run is None:
            return
        try:
            self._run.finish()
        except Exception as exc:
            self._disable(f"finish failed ({type(exc).__name__}: {exc})")
        self._run = None
