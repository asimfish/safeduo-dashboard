"""Content-addressed D-032 replay source for paired raw/S0 evaluation.

The source is deliberately state-independent and never advances its own clock.
The runner sets an explicit transition index before each control step, which
prevents raw and S0 command streams from diverging after their simulated states
do.  Arm commands are the recorded command plus the already-frozen perturbation;
no clipping, resampling, or regeneration occurs here.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from safeduo.delta._contract_stub import ARM_KEYS, DOF_OF, DeltaCmd, DeltaSource
from safeduo.eval.trajectory_task_bundle import load_episode_bundle


class ReplayContractError(ValueError):
    """The caller violated the explicit-step paired replay contract."""


@dataclass(frozen=True)
class HandTarget:
    """One named hand's absolute target for a control transition."""

    arm: str
    joint_names: tuple[str, ...]
    q: torch.Tensor
    valid: torch.Tensor


def _execution_trace_digest(commands: dict[str, np.ndarray]) -> str:
    """Hash exact little-endian float32 bytes in canonical arm order."""
    digest = hashlib.sha256()
    digest.update(b"d032-execution-trace-f32le-v1\0")
    for arm in ARM_KEYS:
        encoded_arm = arm.encode("ascii")
        array = np.ascontiguousarray(commands[arm], dtype="<f4")
        digest.update(struct.pack("<I", len(encoded_arm)))
        digest.update(encoded_arm)
        digest.update(struct.pack("<II", *array.shape))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


class TrajectoryTaskReplayDelta(DeltaSource):
    """Explicit-step, N=1 replay source backed by an immutable episode bundle."""

    def __init__(
        self,
        bundle_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        n_envs: int = 1,
    ) -> None:
        if n_envs != 1:
            raise ReplayContractError("D-032 recorded paired replay is N=1 only")
        loaded = load_episode_bundle(bundle_path)
        self.manifest: dict[str, Any] = loaded["manifest"]
        self.exogenous: dict[str, Any] = loaded["exogenous"]
        self.device = torch.device(device)
        self.n_envs = n_envs
        self.steps = int(self.manifest["steps"])
        self.exogenous_digest = str(self.manifest["exogenous_digest"])
        self._step = 0

        raw = self.exogenous["raw_delta"]
        perturb = self.exogenous["perturbation"]["delta"]
        self._commands = {
            arm: np.asarray(raw[arm] + perturb[arm], dtype=np.float64)
            for arm in ARM_KEYS
        }
        self.execution_trace_sha256 = _execution_trace_digest(self._commands)
        self._hand_targets = self._build_hand_targets()

    def _build_hand_targets(self) -> dict[str, dict[str, Any]]:
        targets: dict[str, dict[str, Any]] = {}
        for name, hand in self.exogenous["hands"].items():
            if hand["kind"] == "q":
                # Transition t commands the recorded post-state hand target.
                q = np.asarray(hand["values"][1:], dtype=np.float64)
                valid = np.asarray(hand["valid"][1:], dtype=np.bool_)
            else:
                q = np.asarray(hand["initial"], dtype=np.float64)[None, :] + np.cumsum(
                    np.asarray(hand["values"], dtype=np.float64), axis=0
                )
                valid = np.asarray(hand["valid"], dtype=np.bool_)
            if q.shape[0] != self.steps:
                raise ReplayContractError(
                    f"hand {name} target rows {q.shape[0]} != steps {self.steps}"
                )
            targets[name] = {
                "arm": str(hand["arm"]),
                "joint_names": tuple(str(value) for value in hand["joint_names"]),
                "q": q,
                "valid": valid,
            }
        return targets

    def _checked_step(self, step: int) -> int:
        if isinstance(step, bool) or not isinstance(step, int):
            raise ReplayContractError("replay step must be an integer")
        if not 0 <= step < self.steps:
            raise ReplayContractError(
                f"replay step {step} is outside [0, {self.steps})"
            )
        return step

    def set_step(self, step: int) -> None:
        """Select the next transition without sampling or advancing it."""
        self._step = self._checked_step(step)

    @property
    def step(self) -> int:
        return self._step

    def command_at(self, step: int) -> DeltaCmd:
        """Return a fresh tensor command for exactly one saved transition."""
        index = self._checked_step(step)
        return DeltaCmd(
            delta_q={
                arm: torch.as_tensor(
                    self._commands[arm][index], dtype=torch.float32, device=self.device
                )
                .reshape(1, DOF_OF[arm])
                .clone()
                for arm in ARM_KEYS
            }
        )

    def sample(self, state) -> DeltaCmd:  # type: ignore[no-untyped-def]
        """Return the selected command; state is intentionally not consulted."""
        if getattr(state, "n_envs", self.n_envs) != self.n_envs:
            raise ReplayContractError(
                "scene state environment count differs from replay"
            )
        return self.command_at(self._step)

    def hand_targets_at(self, step: int) -> dict[str, HandTarget]:
        """Return fresh absolute named-joint hand targets for one transition."""
        index = self._checked_step(step)
        return {
            name: HandTarget(
                arm=value["arm"],
                joint_names=value["joint_names"],
                q=torch.as_tensor(
                    value["q"][index], dtype=torch.float32, device=self.device
                ).clone(),
                valid=torch.as_tensor(
                    value["valid"][index], dtype=torch.bool, device=self.device
                ).clone(),
            )
            for name, value in self._hand_targets.items()
        }

    def q_reference_at(self, state_index: int) -> dict[str, torch.Tensor]:
        """Return the saved T+1 arm state used for replay diagnostics."""
        if isinstance(state_index, bool) or not isinstance(state_index, int):
            raise ReplayContractError("state index must be an integer")
        if not 0 <= state_index <= self.steps:
            raise ReplayContractError(
                f"state index {state_index} is outside [0, {self.steps}]"
            )
        return {
            arm: torch.as_tensor(
                self.exogenous["q_ref"][arm][state_index],
                dtype=torch.float32,
                device=self.device,
            )
            .reshape(1, DOF_OF[arm])
            .clone()
            for arm in ARM_KEYS
        }

    def reset(
        self,
        env_ids: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> None:
        """Reset only the explicit cursor; randomness is forbidden and ignored."""
        del generator
        ids = torch.as_tensor(env_ids).detach().cpu().reshape(-1).tolist()
        if ids != [0]:
            raise ReplayContractError("D-032 N=1 reset must target exactly env 0")
        self._step = 0


__all__ = [
    "HandTarget",
    "ReplayContractError",
    "TrajectoryTaskReplayDelta",
]
