"""SkillReplayDelta: recorded skill-trajectory replay + T2 noise (spec S1 T1/T2).

Owner spec (TELEOP_SAFETY_DATA_EVAL_SPEC_20260813 S1): T1 = common four-arm
cooperation skills (handover / dual carry / cross pick-place ...) planned with
the bundled cuRobo, recorded as delta sequences and replayed as a DeltaSource;
T2 = the same trajectories with online perturbations (joint noise, time
jitter, amplitude scaling, start/end randomization), noise tiers feeding the
curriculum.

Design decisions (T1-line, 2026-08-13; rationale in STATUS_T.md):

- **Storage is ABSOLUTE joint positions** on a uniform dt grid (npz per
  skill), not deltas. Deltas are derived at replay time as
  ``qref(t + dt) - qref(t)`` with linear interpolation. Absolute storage keeps
  time-warp noise well-defined (a per-step-delta file cannot be resampled
  without drift) and start-phase randomization exact.

- **The clock trap** (STATUS_SERVER 05:55): stateful delta sources must NOT
  assume sample() is called exactly once per control step -- demo/recording
  loops call it at other cadences and a self-advancing phase then runs fast,
  integrates against itself and the arms freeze. Therefore the playback phase
  here is advanced in exactly one of two mutually exclusive ways:

    * ``auto_advance=True`` (default): phase += rate * state.dt inside
      sample(). Correct for duo_env, whose coordinator samples once per
      control step (_get_observations -> _pending_cmd).
    * ``auto_advance=False``: sample() NEVER moves the phase; the driving
      loop owns it via ``set_time(t)``. Use this in any script that may call
      sample() more or less than once per step (record_s0_demo-style loops).

- Interface parity with L1RandomDelta: ``__init__(n_envs, params, device,
  arm_keys, dof_of)``, ``reset(env_ids, generator)`` reproducible through a
  torch.Generator, ``sample(state) -> DeltaCmd`` with (N, dof) per arm, output
  clamped to ``amp_max``. Toy tests override ``dof_of`` / trajectories.

T2 noise model (all randomness through the reset() generator; per-env draws):

    joint_noise   additive white noise on the emitted delta, rad/step std
    time_jitter   OU wobble on the playback rate (fractional, bandwidth-limited)
    speed_range   per-env constant playback rate sampled at reset
    amp_range     per-env delta amplitude scale sampled at reset
    start_frac    playback starts at phase u*start_frac*T, u~U[0,1)
    end_frac      playback ends at phase (1-u*end_frac)*T, then holds
    tier(k)       canonical presets 0..3 (0 = exact replay)

After a trajectory ends (phase >= end time) the source emits zeros (hold) --
mirroring an operator who finished the skill and rests. ``loop=True`` restarts
from the (randomized) start phase instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from safeduo.delta._contract_stub import (
    ARM_KEYS,
    DOF_OF,
    DeltaCmd,
    DeltaSource,
    SceneState,
)

FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# trajectory container + npz I/O
# ---------------------------------------------------------------------------


@dataclass
class SkillTrajectory:
    """One recorded skill: per-arm absolute joint positions on a uniform grid.

    q[arm]: (T+1, dof) float32; all arms share the same grid length and dt.
    meta: free-form provenance (skill name, layout/birth-pose version, planner,
    margin audit numbers ...) -- written into the npz, round-trips on load.
    """

    q: dict
    dt: float
    meta: dict = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        """Number of delta steps (grid points - 1)."""
        return next(iter(self.q.values())).shape[0] - 1

    @property
    def duration(self) -> float:
        return self.n_steps * self.dt

    def validate(self, arm_keys: tuple = ARM_KEYS, dof_of: "dict | None" = None) -> None:
        dof_of = dof_of or DOF_OF
        lens = {a: self.q[a].shape[0] for a in arm_keys}
        if len(set(lens.values())) != 1:
            raise ValueError(f"per-arm grid lengths differ: {lens}")
        if self.n_steps < 1:
            raise ValueError("trajectory needs at least 2 grid points")
        for a in arm_keys:
            if self.q[a].shape[1] != dof_of[a]:
                raise ValueError(
                    f"{a}: dof {self.q[a].shape[1]} != contract {dof_of[a]}")

    def save(self, path: "str | Path") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrs = {f"q_{a}": np.asarray(v, dtype=np.float32) for a, v in self.q.items()}
        np.savez_compressed(
            path, dt=np.float64(self.dt),
            meta=json.dumps({"format_version": FORMAT_VERSION, **self.meta},
                            ensure_ascii=False),
            **arrs)

    @classmethod
    def load(cls, path: "str | Path") -> "SkillTrajectory":
        z = np.load(path, allow_pickle=False)
        q = {k[2:]: z[k] for k in z.files if k.startswith("q_")}
        meta = json.loads(str(z["meta"]))
        return cls(q=q, dt=float(z["dt"]), meta=meta)


# ---------------------------------------------------------------------------
# T2 noise parameters
# ---------------------------------------------------------------------------


@dataclass
class SkillNoiseParams:
    """T2 perturbation knobs; tier() gives the canonical curriculum presets."""

    joint_noise: float = 0.0        # additive delta noise std (rad/step)
    time_jitter: float = 0.0        # OU rate-wobble stationary std (fraction of rate)
    jitter_bw_hz: float = 0.5       # OU bandwidth for the rate wobble
    speed_range: tuple = (1.0, 1.0)  # per-env playback rate at reset
    amp_range: tuple = (1.0, 1.0)    # per-env amplitude scale at reset
    start_frac: float = 0.0        # start phase in [0, start_frac]*T
    end_frac: float = 0.0          # end phase in [(1-end_frac), 1]*T
    amp_max: float = 0.06          # hard per-step cap (contract scale, rad)

    TIERS = {
        # 0: exact replay (T1 layer verbatim)
        0: {},
        # 1: mild -- glove-tremor scale noise, +-10% tempo, small start shift
        1: {"joint_noise": 0.002, "time_jitter": 0.05,
            "speed_range": (0.9, 1.1), "amp_range": (0.95, 1.05),
            "start_frac": 0.10, "end_frac": 0.05},
        # 2: medium -- visible sloppiness, +-25% tempo, amplitude drift
        2: {"joint_noise": 0.005, "time_jitter": 0.10,
            "speed_range": (0.75, 1.25), "amp_range": (0.85, 1.20),
            "start_frac": 0.25, "end_frac": 0.10},
        # 3: aggressive -- adversarial-adjacent (spec: 0.015 rad/step is the
        #    full glove scale; tier-3 noise alone reaches that on top of the
        #    skill motion), double-speed playback allowed, deep crops
        3: {"joint_noise": 0.015, "time_jitter": 0.20,
            "speed_range": (0.5, 2.0), "amp_range": (0.7, 1.5),
            "start_frac": 0.50, "end_frac": 0.25},
    }

    @classmethod
    def tier(cls, level: int) -> "SkillNoiseParams":
        if level not in cls.TIERS:
            raise ValueError(f"unknown noise tier {level}; have {sorted(cls.TIERS)}")
        return cls(**cls.TIERS[level])


# ---------------------------------------------------------------------------
# the delta source
# ---------------------------------------------------------------------------


class SkillReplayDelta(DeltaSource):
    """Replays a library of SkillTrajectory files as a DeltaSource.

    Each env plays one trajectory from the library (chosen at reset through
    the generator), with per-env T2 noise state. See module docstring for the
    phase-advance contract (auto_advance vs set_time).
    """

    def __init__(self, n_envs: int, trajectories: "list[SkillTrajectory]",
                 params: "SkillNoiseParams | None" = None,
                 device: "str | torch.device" = "cpu",
                 arm_keys: tuple = ARM_KEYS,
                 dof_of: "dict | None" = None,
                 auto_advance: bool = True,
                 loop: bool = False):
        if not trajectories:
            raise ValueError("empty trajectory library")
        self.n = n_envs
        self.p = params or SkillNoiseParams()
        self.device = torch.device(device)
        self.arms = arm_keys
        self.dof_of = dof_of or DOF_OF
        self.auto_advance = auto_advance
        self.loop = loop
        self.gen: "torch.Generator | None" = None

        for tr in trajectories:
            tr.validate(arm_keys, self.dof_of)
        self._lib = trajectories
        # family protocol (duck-typed, mirrors ConflictMixSource): .names +
        # .assignment let endurance_eval tag every run with its skill family,
        # which is what the spec-2A skill-bucket matrix groups by
        self.names = [str(tr.meta.get("skill", f"skill{k}"))
                      for k, tr in enumerate(trajectories)]
        # library as padded tensors: (K, Tmax+1, dof) per arm + per-traj length
        self._n_steps = torch.tensor([tr.n_steps for tr in trajectories],
                                     dtype=torch.long, device=self.device)
        self._dt_traj = torch.tensor([tr.dt for tr in trajectories],
                                     dtype=torch.float32, device=self.device)
        tmax = int(self._n_steps.max().item())
        self._qlib = {}
        for a in arm_keys:
            dof = self.dof_of[a]
            buf = torch.zeros(len(trajectories), tmax + 1, dof, device=self.device)
            for k, tr in enumerate(trajectories):
                qa = torch.as_tensor(np.asarray(tr.q[a]), dtype=torch.float32,
                                     device=self.device)
                buf[k, : qa.shape[0]] = qa
                buf[k, qa.shape[0]:] = qa[-1]  # pad by holding the last pose
            self._qlib[a] = buf

        # per-env playback state
        self._traj = torch.zeros(n_envs, dtype=torch.long, device=self.device)
        self._tau = torch.zeros(n_envs, device=self.device)      # playback phase (s)
        self._t_end = torch.full((n_envs,), float("inf"), device=self.device)
        self._t_start = torch.zeros(n_envs, device=self.device)
        self._speed = torch.ones(n_envs, device=self.device)
        self._amp = torch.ones(n_envs, device=self.device)
        self._jit = torch.zeros(n_envs, device=self.device)      # OU rate wobble

    # ---- phase control -----------------------------------------------------

    def set_time(self, t: "float | torch.Tensor",
                 env_ids: "torch.Tensor | None" = None) -> None:
        """Authoritative phase override; THE fix for the sample()-cadence trap.

        ``t`` is the driving loop's episode clock in seconds (scalar, (N,), or
        (len(env_ids),)). Effective phase = t_start + t * speed, so start/speed
        randomization from reset() still applies under an external clock. The
        OU time_jitter wobble is an auto_advance-mode feature and does not
        bend an externally set clock; external-clock loops that want noise
        tiers should still expect joint_noise/amp to apply (each sample() call
        draws noise -- call sample() exactly once per set_time for a
        step-deterministic stream).
        """
        t = torch.as_tensor(t, dtype=torch.float32, device=self.device)
        if env_ids is None:
            self._tau = self._t_start + t.expand(self.n) * self._speed
            return
        ids = env_ids.to(self.device)
        tv = t.expand(ids.shape[0]) if t.numel() == 1 else t
        self._tau[ids] = self._t_start[ids] + tv * self._speed[ids]

    def phase(self) -> torch.Tensor:
        """Current per-env playback phase in trajectory-seconds (diagnostics)."""
        return self._tau.clone()

    @property
    def assignment(self) -> torch.Tensor:
        """Per-env index into ``.names``: which skill each env is replaying
        ((N,) long, drawn at reset() through the generator; constant between
        resets, so one reset-free endurance run = one skill family). Family
        protocol counterpart of ConflictMixSource.assignment; read-only."""
        return self._traj

    # ---- DeltaSource contract ----------------------------------------------

    def reset(self, env_ids: torch.Tensor,
              generator: "torch.Generator | None" = None) -> None:
        if generator is not None:
            self.gen = generator
        ids = env_ids.to(self.device)
        m = ids.shape[0]
        p = self.p

        def _u(*shape):
            return torch.rand(*shape, device=self.device, generator=self.gen)

        self._traj[ids] = torch.randint(len(self._lib), (m,), device=self.device,
                                        generator=self.gen)
        dur = (self._n_steps[self._traj[ids]].float()
               * self._dt_traj[self._traj[ids]])
        self._t_start[ids] = _u(m) * p.start_frac * dur
        self._t_end[ids] = (1.0 - _u(m) * p.end_frac) * dur
        self._speed[ids] = p.speed_range[0] + _u(m) * (
            p.speed_range[1] - p.speed_range[0])
        self._amp[ids] = p.amp_range[0] + _u(m) * (p.amp_range[1] - p.amp_range[0])
        self._jit[ids] = 0.0
        self._tau[ids] = self._t_start[ids]

    def _qref(self, tau: torch.Tensor) -> dict:
        """Linear interpolation of each env's trajectory at phase tau (N,)."""
        k = self._traj
        dt_tr = self._dt_traj[k]
        n_steps = self._n_steps[k]
        s = (tau / dt_tr).clamp(min=0.0)
        s = torch.minimum(s, n_steps.float())
        i0 = s.floor().long()                      # <= n_steps <= Tmax by clamp
        frac = (s - i0.float()).unsqueeze(-1)
        i1 = torch.minimum(i0 + 1, n_steps)
        out = {}
        for a in self.arms:
            lib = self._qlib[a]                    # (K, T+1, dof)
            q0 = lib[k, i0]
            q1 = lib[k, i1]
            out[a] = q0 + (q1 - q0) * frac
        return out

    def sample(self, state: SceneState) -> DeltaCmd:
        p, dt = self.p, state.dt
        # playback rate for this step: speed * (1 + OU jitter)
        if p.time_jitter > 0.0:
            import math

            theta = 2.0 * math.pi * p.jitter_bw_hz
            self._jit = self._jit + (
                -theta * self._jit * dt
                + p.time_jitter * math.sqrt(2.0 * theta * dt)
                * torch.randn(self.n, device=self.device, generator=self.gen))
        rate = self._speed * (1.0 + self._jit).clamp_min(0.0)

        t0 = self._tau
        t1 = t0 + rate * dt
        ended = t0 >= self._t_end
        if self.loop and ended.any():
            # restart the finished envs from their randomized start phase
            t0 = torch.where(ended, self._t_start, t0)
            t1 = torch.where(ended, self._t_start + rate * dt, t1)
            ended = torch.zeros_like(ended)
        t1_c = torch.minimum(t1, self._t_end)
        q0 = self._qref(t0)
        q1 = self._qref(t1_c)
        out = {}
        for a in self.arms:
            d = (q1[a] - q0[a]) * self._amp.unsqueeze(-1)
            if p.joint_noise > 0.0:
                d = d + p.joint_noise * torch.randn(
                    self.n, self.dof_of[a], device=self.device, generator=self.gen)
            d = torch.where(ended.unsqueeze(-1), torch.zeros_like(d), d)
            out[a] = d.clamp(-p.amp_max, p.amp_max)
        if self.auto_advance:
            self._tau = torch.where(ended, t0, t1_c)
        return DeltaCmd(delta_q=out)


def load_library(directory: "str | Path", pattern: str = "skill_*.npz") -> list:
    """All SkillTrajectory files under a directory, name-sorted (stable order
    matters: env->traj assignment is index-based through the generator).
    Default pattern skips the pipeline sidecars (segments_*/waypoints_*) that
    share artifacts/skill_trajs/."""
    files = sorted(Path(directory).glob(pattern))
    if not files:
        raise FileNotFoundError(f"no {pattern} under {directory}")
    return [SkillTrajectory.load(f) for f in files]


def load_manifest_library(manifest_path: "str | Path") -> list:
    """Skill library exactly as listed in a skills_manifest.json.

    Order follows the manifest, not the directory listing, so library
    indices -- and with them .names / .assignment and the generator-driven
    env->skill draw -- stay stable whatever else lands next to the npz files.
    Files resolve relative to the manifest; each entry's ``skill`` name must
    match the npz meta (guards against renamed or mispaired artifacts)."""
    mp = Path(manifest_path)
    man = json.loads(mp.read_text())
    if not man.get("skills"):
        raise ValueError(f"manifest {mp} lists no skills")
    lib = []
    for entry in man["skills"]:
        tr = SkillTrajectory.load(mp.parent / entry["file"])
        got = tr.meta.get("skill")
        if got != entry["skill"]:
            raise ValueError(
                f"{entry['file']}: manifest says skill {entry['skill']!r} "
                f"but npz meta says {got!r}")
        lib.append(tr)
    return lib
