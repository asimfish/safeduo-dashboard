"""Block 1 (M1) main-table harness: SafeDuo checkpoint vs baselines inside the
duo_env closed loop -- one shared L2 backstop, one scenario seed set, full
ACCEPTANCE section-3 metrics + Clopper-Pearson bound + bootstrap CIs + Pareto
point export, table formatted to PAPER_NOTES Tab.1 (method x flow x metrics).

Design (boundary-respecting: duo_env/train are Agent A's files, this module
only *drives* a live env instance):
  - every method runs through the env's own VelocityDamperBackstop, so all
    methods share the exact same L2 bottom layer and PD/physics stack;
  - baselines consume the same ConstraintRows the backstop consumes
    (EnvRowsProvider -> IsaacGeometryProvider.rows_from), apples to apples;
  - method pathways (measured 08-12 dry-run, see STATUS_C W5):
      safeduo / estop / speed / cbf_sched   emit the coordinator action
        (alpha^4, p); the env backstop projects the raw delta under that
        budget. For E-stop/speed this IS their native semantics.
      cbf (default = exec-passthrough)      the strengthened CBF-QP's whole
        point is its own QP *redirection* (intent extrapolation + delay
        compensation live in the exec, not in a scalar budget). Passing only
        its (alpha~0.94, p) schedule measurably breaks it: the env backstop
        then fights a sustained push on the permanently-active fat-sphere
        self rows and its finite-pass projection grinds through the d_min
        buffer (19/36 sub-mm self-graze episodes on L1 in the dry-run).
        So the cbf method feeds its own QP exec as the command (alpha=1 ->
        no budget rows) with its priority p kept for the cross-row budget
        split; the env backstop still projects it (the shared L2 floor).
        The schedule-only variant stays available as `cbf_sched` (it is the
        honest "CBF as coordinator-slot" ablation).
  - intervention is reported twice: declared (the method's own alpha gate)
    and effective (<exec, cmd>/||cmd||^2 against the ORIGINAL commanded
    delta, delta-nonzero steps only per ACCEPTANCE section 3) -- the
    effective one is method-uniform and is what Tab.1/Pareto should use;
  - flows: "l1" = non-conflict random traffic (fidelity regime), "l2eval" =
    the 8-family scripted conflict battery on the EVAL split (train split is
    the tuning battery, never used here);
  - same env cfg seed + same condition seed -> identical episode-1 scenario
    assignment for every method; methods with zero violations stay
    episode-synchronized for the entire run (resets all happen at the shared
    timeout), which is what makes the anticipation pairing valid.

Episode semantics: the env terminates on VIOLATION and truncates at the
horizon; the recorder logs the done flags and episodes are chopped at done
steps. Complete episodes go into a padded EpisodeBatch (in_tube padded with
its last value so a tube open at termination stays censored, everything else
padded inert) plus a step-valid mask; step-mean metrics (intervention,
tracking RMSE, backstop duty) are recomputed under the mask so padding never
dilutes them. The trailing unfinished segment of each env is dropped and
reported as censored episodes.

Run on the server (Isaac required; CPU physics is fine and leaves the GPUs
to training):
  source ~/safeduo_setup/env.sh && cd ~/safeduo && \
    python -m safeduo.eval.block1_harness --headless --device cpu \
      --ckpt artifacts/runs/a5_v1_4096/model_1400.pt \
      --methods safeduo estop speed cbf --flows l2eval l1 \
      --num-envs 32 --episodes 1 --seed 123 --out artifacts/block1/dryrun

The pure-torch pieces (checkpoint loader, episode chopper, masked metrics,
table/pareto formatting, action mapping) are Mac-importable and unit-tested;
only main()/run_* touch Isaac.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from safeduo.delta._contract_stub import ARM_KEYS, DeltaCmd
from safeduo.eval.metrics import (
    CollisionSplitConfig,
    EpisodeBatch,
    MetricsConfig,
    bootstrap_ci,
    collision_split_stats,
    compute_all_metrics,
)
from safeduo.eval.paired_report import anticipation_report, anticipation_table_row

FLOWS = ("l1", "l2eval")
METHOD_CHOICES = ("safeduo", "estop", "speed", "cbf", "cbf_sched", "passthrough")

# W5 grid winners (artifacts/grid*, baseline_matrix.md v0.4): the operating
# points frozen for the Block-1 main table.
#   estop  closed-loop zero-collision representative: 0.04/0.06 latches the F
#          arms at the 0.0322 m resting envelope (kinematic best-by-
#          intervention among latched configs). The releasable kinematic
#          winner 0.015/0.02 MOVES -- and lands on the universal PD-lag graze
#          floor in the physical loop (dress rehearsal: 9/32 collisions on
#          L1), breaking the zero-collision precondition. freeze-XOR-graze is
#          E-stop's honest dilemma here; use --baseline-cfg to run the
#          releasable variant as a trade-curve point.
#   speed  releasable-regime winner (floor=0 mandatory: alpha floors collide
#          standalone; crawling keeps it at zero collisions in-loop).
#   cbf    gamma=4 wins time-to-clear at equal intervention, matches the
#          shared L2 damper's gamma; consolidation defaults already optimal;
#          tau_delay stays 0 (closed-loop sweep = clean negative vs the
#          graze floor, STATUS_C W5).
BASELINE_FROZEN_W5 = {
    "estop": {"d_stop": 0.04, "d_resume": 0.06},
    "speed": {"d_slow": 0.075, "alpha_floor": 0.0, "ema_hz": 2.0},
    "cbf": {"gamma": 4.0, "extrap": "linear"},
}


# --------------------------------------------------------------------------
# checkpoint actor (pure torch; architecture inferred from the state dict)
# --------------------------------------------------------------------------

class _BetaActorPolicy(torch.nn.Module):
    """Deterministic eval wrapper for lagrangian_ppo_v1 checkpoints (A8-W8).

    v4 runs (train_lagrangian) save a Beta actor under ck["actor"] with
    backbone.*/parameter_head.* keys -- no actor.<i>.* rsl_rl layout, so the
    ActorMLP path cannot load them. forward(obs) returns the deterministic
    env-layout action (alpha*2-1, p) = to_env_action(distribution mean),
    matching the ActorMLP output contract so PolicyDriver and the demo
    recorders need no caller-side change.
    """

    def __init__(self, actor, hazard_detector: bool = False,
                 hazard_temp: float = 1.0):
        super().__init__()
        self.actor = actor
        # loop-4 detector-source switch: when True the alpha 4-dims of the
        # env action are replaced with 1 - sigmoid(hazard_logits / T)
        # (P(safe) semantics; ClutchDriver recovers it via (a+1)/2). The p
        # dim and execution semantics are untouched. Default False keeps the
        # legacy path bit-identical (tests/test_clutch.py pins).
        self.hazard_detector = bool(hazard_detector)
        self.hazard_temp = float(hazard_temp)
        if self.hazard_detector and getattr(actor, "hazard_head", None) is None:
            raise ValueError(
                "hazard_detector requires a ckpt with hazard_head (a25+)")
        if self.hazard_temp <= 0.0:
            raise ValueError("hazard_temp must be positive")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        from safeduo.algo.beta_actor import BetaCoordinatorActor
        raw = self.actor.distribution(obs).mean.clamp(1e-6, 1.0 - 1e-6)
        a = BetaCoordinatorActor.to_env_action(raw)
        if self.hazard_detector:
            z = self.actor.hazard_logits(obs) / self.hazard_temp
            alpha_det = 1.0 - torch.sigmoid(z)
            a = torch.cat([alpha_det * 2.0 - 1.0, a[:, 4:]], dim=-1)
        return a


class ActorMLP(torch.nn.Module):
    """rsl_rl ActorCritic actor rebuilt from a train_ppo checkpoint.

    Hidden dims differ across eras (v1 = [256,128,64], r2 = [256,256]) so the
    Linear stack is inferred from the `actor.<i>.weight` shapes; activation is
    ELU in every SafeDuo recipe to date. Deterministic eval = actor mean (the
    exploration `std` parameter is ignored).
    """

    def __init__(self, sizes: list):
        super().__init__()
        layers: list = []
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(torch.nn.Linear(n_in, n_out))
            if i < len(sizes) - 2:
                layers.append(torch.nn.ELU())
        self.net = torch.nn.Sequential(*layers)
        self.sizes = list(sizes)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @staticmethod
    def from_checkpoint(path: "str | Path", hazard_detector: bool = False,
                        hazard_temp: float = 1.0) -> "torch.nn.Module":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ck, dict) and str(ck.get("format", "")).startswith(
                "lagrangian_ppo"):
            # v4 Lagrangian run (Beta actor); see _BetaActorPolicy above
            from safeduo.algo.beta_actor import BetaCoordinatorActor
            lag_cfg = ck.get("config", {})
            actor = BetaCoordinatorActor(
                int(ck["obs_dim"]),
                hidden=tuple(lag_cfg.get("hidden", (256, 256))),
                min_concentration=float(lag_cfg.get("min_concentration", 1.0)),
                # R22 P2: hazard-head ckpts carry hazard_head.* weights; build
                # the matching arch or load_state_dict rejects the ckpt.
                hazard_head=bool(lag_cfg.get("hazard_head", False)))
            actor.load_state_dict(ck["actor"])
            actor.eval()
            wrapped = _BetaActorPolicy(actor, hazard_detector=hazard_detector,
                                       hazard_temp=hazard_temp)
            wrapped.eval()
            return wrapped
        if hazard_detector:
            raise ValueError(
                f"hazard_detector only supports lagrangian_ppo ckpts: {path}")
        sd = ck.get("model_state_dict", ck)
        weights: dict[int, torch.Tensor] = {}
        biases: dict[int, torch.Tensor] = {}
        for k, v in sd.items():
            if k.startswith("actor.") and k.endswith(".weight"):
                weights[int(k.split(".")[1])] = v
            elif k.startswith("actor.") and k.endswith(".bias"):
                biases[int(k.split(".")[1])] = v
        if not weights:
            raise ValueError(f"no actor.* weights in checkpoint {path}")
        order = sorted(weights)
        sizes = [weights[order[0]].shape[1]] + [weights[i].shape[0] for i in order]
        net = ActorMLP(sizes)
        with torch.no_grad():
            linears = [m for m in net.net if isinstance(m, torch.nn.Linear)]
            for lin, i in zip(linears, order):
                lin.weight.copy_(weights[i])
                lin.bias.copy_(biases[i])
        net.eval()
        return net


# --------------------------------------------------------------------------
# method drivers: anything -> coordinator action (alpha*2-1, p)
# --------------------------------------------------------------------------

def filter_output_to_action(alpha: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """FilterOutput (alpha in [0,1], p in [-1,1]) -> env action (N,5); the env
    maps action[:, :4] back via (a+1)/2 so this is exact round-trip."""
    return torch.cat([alpha.clamp(0.0, 1.0) * 2.0 - 1.0,
                      p.clamp(-1.0, 1.0).unsqueeze(-1)], dim=-1)


class PolicyDriver:
    """SafeDuo checkpoint: obs -> deterministic actor mean."""

    def __init__(self, ckpt_path: "str | Path", device: str = "cpu",
                 hazard_detector: bool = False, hazard_temp: float = 1.0):
        self.net = ActorMLP.from_checkpoint(
            ckpt_path, hazard_detector=hazard_detector,
            hazard_temp=hazard_temp).to(device)
        self.device = device

    def reset(self, env_ids: torch.Tensor) -> None:  # stateless
        pass

    def act(self, env, obs_dict: dict) -> torch.Tensor:
        obs = obs_dict["policy"].to(self.device)
        with torch.no_grad():
            a = self.net(obs)
        return a.clamp(-1.0, 1.0)


class EnvRowsProvider:
    """GeometryProvider contract backed by the live env: the same rows the
    env backstop is about to consume this step (IsaacGeometryProvider over
    the obs-time sphere compute cache)."""

    def __init__(self, env):
        self.env = env

    def rows(self, state):  # state arg unused: rows come from the step cache
        env = self.env
        assert env._last_out is not None, "compute_dist() must precede rows()"
        return env._provider.rows_from(env._last_out, env._body_pos_cache)


class BaselineDriver:
    """Wraps a BaselineFilter. Two pathways (module docstring):

    schedule mode        emit the filter's (alpha, p) as the env action.
    exec-passthrough     swap the env's pending command for the filter's own
                         delta_exec, emit alpha=1 (no budget rows) + the
                         filter's p; publish a `log_patch` so the recorder
                         logs the ORIGINAL command and the filter's declared
                         alpha instead of the swapped/unit values.
    """

    def __init__(self, filt, exec_passthrough: bool = False):
        self.filt = filt
        self.exec_passthrough = exec_passthrough
        self.log_patch: "dict | None" = None

    def reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel():
            self.filt.reset(env_ids)

    def act(self, env, obs_dict: dict) -> torch.Tensor:
        state = env.scene_state()
        cmd = env._pending_cmd
        assert cmd is not None, "env must be reset before act()"
        out = self.filt.filter(state, cmd)
        if not self.exec_passthrough:
            self.log_patch = None
            return filter_output_to_action(out.alpha, out.priority_p).to(env.device)
        self.log_patch = {
            "cmd": {a: cmd.delta_q[a].detach().clone() for a in ARM_KEYS},
            "alpha": out.alpha.detach().clone(),
        }
        env._pending_cmd = DeltaCmd(delta_q={
            a: out.delta_exec[a].detach().clone() for a in ARM_KEYS})
        action = filter_output_to_action(torch.ones_like(out.alpha),
                                         out.priority_p)
        return action.to(env.device)


class PassthroughDriver:
    """alpha=1, p=0 (B0 sanity row, not for the main table)."""

    def reset(self, env_ids: torch.Tensor) -> None:
        pass

    def act(self, env, obs_dict: dict) -> torch.Tensor:
        a = torch.ones(env.num_envs, 5, device=env.device)
        a[:, 4] = 0.0
        return a


def make_baseline_driver(method: str, env, overrides: "dict | None" = None
                         ) -> BaselineDriver:
    from safeduo.baselines.estop import EStopConfig, EStopFilter
    from safeduo.baselines.speed_scaling import SpeedScalingConfig, SpeedScalingFilter
    from safeduo.baselines.strong_cbf_qp import StrongCBFQPConfig, StrongCBFQPFilter

    kw = dict(overrides or {})
    provider = EnvRowsProvider(env)
    n, dev = env.num_envs, str(env.device)
    if method == "estop":
        filt = EStopFilter(n, provider, EStopConfig(**kw), device=dev)
    elif method == "speed":
        filt = SpeedScalingFilter(n, provider, SpeedScalingConfig(**kw), device=dev)
    elif method in ("cbf", "cbf_sched"):
        filt = StrongCBFQPFilter(n, provider, StrongCBFQPConfig(**kw), device=dev)
    else:
        raise ValueError(f"unknown baseline {method}")
    return BaselineDriver(filt, exec_passthrough=(method == "cbf"))


# --------------------------------------------------------------------------
# flow sources (eval battery construction)
# --------------------------------------------------------------------------

def make_flow_source(flow: str, env):
    """l1 = non-conflict random traffic; l2eval = 8-family conflict battery,
    EVAL split, uniform family weights, no curriculum stages."""
    amp = float(env.cfg.coordinator.get("delta_amp_max", 0.015))
    if flow == "l1":
        from safeduo.delta.l1_random import L1Params, L1RandomDelta

        return L1RandomDelta(env.num_envs, params=L1Params(amp_max=amp),
                             device=env.device)
    if flow == "l2eval":
        from safeduo.delta.l2_env_source import ConflictMixSource
        from safeduo.delta.l2_scenarios import SCENARIOS

        cfg = {"mix": {"l1": 0.0, "l2": 1.0},
               "scenarios": {name: 1.0 for name in SCENARIOS},
               "split": "eval", "n_variants": 100, "stages": None}
        return ConflictMixSource(env.num_envs, cfg, device=env.device,
                                 amp_max=amp)
    raise ValueError(f"unknown flow {flow}")


# --------------------------------------------------------------------------
# step recorder (reward-time hook, the train_ppo pattern: capture the step
# cache before resets wipe it)
# --------------------------------------------------------------------------

class StepRecorder:
    COLS = ("alpha", "p", "tube", "violation", "backstop", "ep_step",
            "mm_cross", "mm_self_f", "mm_self_u", "mm_table")

    def __init__(self, env, driver=None):
        self.env = env
        self.driver = driver          # exec-passthrough drivers publish a patch
        self.rows: dict[str, list] = {k: [] for k in self.COLS}
        self.cmd: dict[str, list] = {a: [] for a in ARM_KEYS}
        self.exec_: dict[str, list] = {a: [] for a in ARM_KEYS}
        self.done: list = []
        self._orig = None

    @staticmethod
    def select_logs(step_cache: dict, patch: "dict | None") -> tuple:
        """(alpha (N,4), cmd dict) to log: the driver's declared alpha and the
        ORIGINAL command when a patch is present (exec-passthrough swaps the
        env's pending command, so the step cache alone would log exec-vs-exec
        and unit alpha)."""
        if patch is None:
            return step_cache["alpha"], step_cache["cmd"].delta_q
        return patch["alpha"], patch["cmd"]

    def install(self):
        env = self.env
        self._orig = env._get_rewards

        def hooked():
            r = self._orig()
            c = env._step_cache
            out = env._last_out
            patch = getattr(self.driver, "log_patch", None)
            alpha_log, cmd_log = self.select_logs(c, patch)
            self.rows["alpha"].append(alpha_log.detach().cpu())
            self.rows["p"].append(c["p"].detach().cpu())
            self.rows["tube"].append(c["tube"].detach().cpu())
            self.rows["violation"].append(c["violation"].detach().cpu())
            self.rows["backstop"].append(c["bs_active"].detach().cpu())
            self.rows["ep_step"].append(env.episode_length_buf.detach().cpu().clone())
            mm = out.min_margin
            self.rows["mm_cross"].append(mm["cross"].detach().cpu())
            self.rows["mm_self_f"].append(mm["self_F"].detach().cpu())
            self.rows["mm_self_u"].append(mm["self_U"].detach().cpu())
            self.rows["mm_table"].append(mm["table"].detach().cpu())
            for a in ARM_KEYS:
                self.cmd[a].append(cmd_log[a].detach().cpu())
                self.exec_[a].append(c["exec"].delta_q[a].detach().cpu())
            return r

        env._get_rewards = hooked

    def uninstall(self):
        if self._orig is not None:
            self.env._get_rewards = self._orig
            self._orig = None

    def record_done(self, done: torch.Tensor) -> None:
        self.done.append(done.detach().cpu().clone())

    def stacked(self) -> dict:
        out = {k: torch.stack(v) for k, v in self.rows.items()}
        out["done"] = torch.stack(self.done)
        out["cmd"] = {a: torch.stack(self.cmd[a]) for a in ARM_KEYS}
        out["exec"] = {a: torch.stack(self.exec_[a]) for a in ARM_KEYS}
        return out


# --------------------------------------------------------------------------
# episode chopping -> padded EpisodeBatch + valid mask
# --------------------------------------------------------------------------

def episode_slices(done: torch.Tensor) -> tuple:
    """(T, N) done flags -> ([(env, s, e_inclusive)], n_dropped_tail).
    The single source of episode-boundary semantics: complete episodes end at
    a done step; the trailing never-done segment of each env is dropped and
    counted. Shared by chop_episodes and the offline collision re-cut."""
    T, N = done.shape
    episodes = []
    n_tail = 0
    for env in range(N):
        s = 0
        for t in range(T):
            if bool(done[t, env]):
                episodes.append((env, s, t))
                s = t + 1
        if s < T:
            n_tail += 1
    return episodes, n_tail


# margin-trace keys recorded by StepRecorder -> metrics.MARGIN_CLASSES names
MARGIN_TRACE_KEYS = (("mm_cross", "cross"), ("mm_self_f", "self_f"),
                     ("mm_self_u", "self_u"), ("mm_table", "table"))
MARGIN_PAD = 1.0    # inert pad: +1.0 m margin can never register as depth


def chop_episodes(trace: dict, dt: float, horizon: int) -> dict:
    """Segment the (T, N) trace at done steps into complete episodes.

    Returns {"batch": EpisodeBatch (H, E), "valid": (H, E) bool,
             "ep_env": (E,) source env ids, "ep_len": (E,) real lengths,
             "n_dropped_tail": trailing unfinished segments dropped,
             "margins": {class: (H, E)} when the trace carries mm_* columns}.
    Padding: in_tube repeats its last value (tube open at a violation
    termination stays open -> censored, never counted as cleared); alpha/p
    repeat their last value (no spurious gate/flip edges); violation/backstop
    pad False; cmd/exec pad zero; margins pad +1.0 (inert for the
    damaging/near-contact depth split).
    """
    done = trace["done"]
    T, N = done.shape
    episodes, n_tail = episode_slices(done)
    E = len(episodes)
    H = min(horizon, T)

    def pad_last(x: torch.Tensor, length: int) -> torch.Tensor:
        if x.shape[0] >= length:
            return x[:length]
        reps = x[-1:].expand(length - x.shape[0], *x.shape[1:])
        return torch.cat([x, reps], dim=0)

    def pad_zero(x: torch.Tensor, length: int) -> torch.Tensor:
        if x.shape[0] >= length:
            return x[:length]
        z = torch.zeros(length - x.shape[0], *x.shape[1:], dtype=x.dtype)
        return torch.cat([x, z], dim=0)

    cols = {"alpha": [], "p": [], "tube": [], "violation": [], "backstop": []}
    cmd = {a: [] for a in ARM_KEYS}
    exc = {a: [] for a in ARM_KEYS}
    margin_keys = [(tk, mk) for tk, mk in MARGIN_TRACE_KEYS if tk in trace]
    margins = {mk: [] for _, mk in margin_keys}
    valid = torch.zeros(H, E, dtype=torch.bool)
    ep_env = torch.zeros(E, dtype=torch.long)
    ep_len = torch.zeros(E, dtype=torch.long)
    for j, (env, s, e) in enumerate(episodes):
        ln = min(e - s + 1, H)
        ep_env[j], ep_len[j] = env, ln
        valid[:ln, j] = True
        sl = slice(s, s + ln)
        cols["alpha"].append(pad_last(trace["alpha"][sl, env], H))
        cols["p"].append(pad_last(trace["p"][sl, env], H))
        cols["tube"].append(pad_last(trace["tube"][sl, env], H))
        cols["violation"].append(pad_zero(trace["violation"][sl, env], H))
        cols["backstop"].append(pad_zero(trace["backstop"][sl, env], H))
        for tk, mk in margin_keys:
            seg = trace[tk][sl, env]
            if seg.shape[0] < H:
                pad = torch.full((H - seg.shape[0],), MARGIN_PAD,
                                 dtype=seg.dtype)
                seg = torch.cat([seg, pad], dim=0)
            margins[mk].append(seg)
        for a in ARM_KEYS:
            cmd[a].append(pad_zero(trace["cmd"][a][sl, env], H))
            exc[a].append(pad_zero(trace["exec"][a][sl, env], H))
    if E == 0:
        raise ValueError("no complete episodes in trace (run longer)")
    batch = EpisodeBatch(
        delta_cmd={a: torch.stack(cmd[a], dim=1) for a in ARM_KEYS},
        delta_exec={a: torch.stack(exc[a], dim=1) for a in ARM_KEYS},
        alpha=torch.stack(cols["alpha"], dim=1),
        priority_p=torch.stack(cols["p"], dim=1),
        in_tube=torch.stack(cols["tube"], dim=1).bool(),
        violation=torch.stack(cols["violation"], dim=1).bool(),
        backstop_active=torch.stack(cols["backstop"], dim=1).bool(),
        dt=dt,
    )
    out = {"batch": batch, "valid": valid, "ep_env": ep_env,
           "ep_len": ep_len, "n_dropped_tail": n_tail}
    if margin_keys:
        out["margins"] = {mk: torch.stack(v, dim=1) for mk, v in margins.items()}
    return out


def masked_step_metrics(batch: EpisodeBatch, valid: torch.Tensor) -> dict:
    """Recompute padding-sensitive step means under the valid mask:
    declared intervention rate (+ per arm), EFFECTIVE intervention rate
    (1 - <exec, cmd>/||cmd||^2, delta-nonzero steps only -- the ACCEPTANCE
    section 3 qualifier, method-uniform and robust to how a method expresses
    its filtering), tracking RMSE (+ outside conflict), backstop duty cycle,
    plus per-episode arrays for bootstrap CIs."""
    w = valid.float()                                  # (H, E)
    steps = w.sum().clamp_min(1.0)
    one_minus = (1.0 - batch.alpha) * w.unsqueeze(-1)
    out = {"intervention_rate": float(one_minus.sum().item() / (steps * 4))}
    per_arm = one_minus.sum(dim=(0, 1)) / steps
    out.update({f"intervention_rate_{a}": float(per_arm[i].item())
                for i, a in enumerate(ARM_KEYS)})
    sq_all = 0.0
    free = (~batch.in_tube).float() * w
    sq_free = 0.0
    eff_sum, eff_cnt = 0.0, 0.0
    eff_ep_sum = torch.zeros(batch.N)
    eff_ep_cnt = torch.zeros(batch.N)
    for a in ARM_KEYS:
        cmd, exc = batch.delta_cmd[a], batch.delta_exec[a]
        err = exc - cmd
        sq = (err * err).sum(dim=-1)
        sq_all += (sq * w).sum().item()
        sq_free += (sq * free).sum().item()
        den = (cmd * cmd).sum(dim=-1)
        nz = (den > 1e-12).float() * w                 # delta-nonzero steps
        alpha_eff = ((exc * cmd).sum(dim=-1) / den.clamp_min(1e-12)).clamp(0, 1)
        eff_sum += (alpha_eff * nz).sum().item()
        eff_cnt += nz.sum().item()
        eff_ep_sum += (alpha_eff * nz).sum(dim=0)
        eff_ep_cnt += nz.sum(dim=0)
    out["tracking_rmse"] = (sq_all / max(steps.item(), 1.0)) ** 0.5
    n_free = float(free.sum().item())
    out["tracking_rmse_outside_conflict"] = (sq_free / max(n_free, 1.0)) ** 0.5
    out["effective_intervention_rate"] = 1.0 - eff_sum / max(eff_cnt, 1.0)
    bs = batch.backstop_active.float() * w.unsqueeze(-1)
    out["backstop_duty_cycle"] = float(bs.sum().item() / (steps * 4))
    ep_steps = w.sum(dim=0).clamp_min(1.0)             # (E,)
    out["per_episode_intervention"] = (
        ((1.0 - batch.alpha) * w.unsqueeze(-1)).sum(dim=(0, 2)) / (ep_steps * 4))
    out["per_episode_effective_intervention"] = (
        1.0 - eff_ep_sum / eff_ep_cnt.clamp_min(1.0))
    out["per_episode_tube_dwell_s"] = (
        (batch.in_tube.float() * w).sum(dim=0) * batch.dt)
    return out


def condition_metrics(chop: dict, cfg: "MetricsConfig | None" = None,
                      split_cfg: "CollisionSplitConfig | None" = None) -> dict:
    """compute_all_metrics with the padding-sensitive means overridden, plus
    the G1 damaging/near-contact split when the chop carries margin traces
    (contact force is not in the recorder yet, so the split is depth-only;
    the force criterion activates automatically once a force trace exists)."""
    cfg = cfg or MetricsConfig()
    met = compute_all_metrics(chop["batch"], cfg)
    if "margins" in chop:
        met.update(collision_split_stats(
            chop["margins"], chop["batch"].violation,
            split_cfg or CollisionSplitConfig(), valid=chop["valid"]))
    masked = masked_step_metrics(chop["batch"], chop["valid"])
    eps = {k: masked.pop(k) for k in
           ("per_episode_intervention", "per_episode_effective_intervention",
            "per_episode_tube_dwell_s")}
    met.update(masked)
    met["episodes_dropped_tail"] = chop["n_dropped_tail"]
    lo, hi = bootstrap_ci(eps["per_episode_intervention"], conf=cfg.conf)
    met["intervention_ci95"] = (lo, hi)
    lo_e, hi_e = bootstrap_ci(eps["per_episode_effective_intervention"],
                              conf=cfg.conf)
    met["effective_intervention_ci95"] = (lo_e, hi_e)
    return {"metrics": met, "eps": eps}


# --------------------------------------------------------------------------
# Tab.1 table + Pareto export
# --------------------------------------------------------------------------

TAB1_HEADER = (
    "| method | flow | damaging (CP-UB95) | near-contact "
    "| sphere-layer (CP-UB95) | eff. intervention [CI95] "
    "| declared interv. | tracking RMSE | time-to-clear mean/p95 (s) "
    "| oscill./ep | proj. progress | backstop/ep | recovery (s) | stalls |")

TAB1_NCOL = 14


def _split_cells(met: dict) -> str:
    """G1-split table cells; n/a when the chop carried no margin traces
    (pre-W6 traces / synthetic tests)."""
    if "damaging_episodes" not in met:
        return "| n/a | n/a "
    return (f"| {met['damaging_episodes']}/{met['episodes']} "
            f"({met['damaging_rate_ub95']:.4f}) "
            f"| {met['near_contact_episodes']}/{met['episodes']} ")


def tab1_row(method: str, flow: str, met: dict) -> str:
    lo, hi = met["effective_intervention_ci95"]
    return (f"| {method} | {flow} "
            + _split_cells(met) +
            f"| {met['collision_episodes']}/{met['episodes']} "
            f"({met['collision_rate_ub95']:.4f}) "
            f"| {met['effective_intervention_rate']:.4f} [{lo:.4f},{hi:.4f}] "
            f"| {met['intervention_rate']:.4f} "
            f"| {met['tracking_rmse']:.5f} "
            f"| {met['time_to_clear_mean']:.2f}/{met['time_to_clear_p95']:.2f} "
            f"| {met['oscillations_per_episode']:.2f} "
            f"| {met['projected_progress_conflict']:.3f} "
            f"| {met['backstop_triggers_per_episode']:.2f} "
            f"| {met['recovery_latency_mean']:.2f} "
            f"| {met['stall_events']} |")


def render_tab1(results: dict) -> str:
    """results: (method, flow) -> {"metrics": ...}. Flow-major sections,
    PAPER_NOTES Tab.1 schema (conflict first: that is the headline regime)."""
    lines = ["# Block 1 main table (Tab.1 schema)", ""]
    for flow in ("l2eval", "l1"):
        rows = [(m, f) for (m, f) in results if f == flow]
        if not rows:
            continue
        regime = "conflict battery (L2 eval split)" if flow == "l2eval" \
            else "non-conflict random traffic (L1)"
        lines += [f"## {regime}", "", TAB1_HEADER, "|" + "---|" * TAB1_NCOL]
        lines += [tab1_row(m, f, results[(m, f)]["metrics"]) for m, f in rows]
        lines.append("")
    return "\n".join(lines)


def pareto_points(results: dict) -> list:
    """Fig.2 data points: one per (method, flow) -- intervention x conflict
    time under the zero-collision precondition (violations reported so a
    nonzero-collision point can be flagged/excluded downstream)."""
    pts = []
    for (m, f), r in sorted(results.items()):
        met = r["metrics"]
        lo, hi = met["effective_intervention_ci95"]
        pts.append({
            "method": m, "flow": f,
            "intervention_rate": met["effective_intervention_rate"],
            "intervention_ci_lo": lo, "intervention_ci_hi": hi,
            "intervention_rate_declared": met["intervention_rate"],
            "time_to_clear_mean_s": met["time_to_clear_mean"],
            "time_to_clear_p95_s": met["time_to_clear_p95"],
            "n_conflicts_cleared": met["n_conflicts_cleared"],
            "collision_episodes": met["collision_episodes"],
            "episodes": met["episodes"],
            "collision_rate_ub95": met["collision_rate_ub95"],
        })
    return pts


# --------------------------------------------------------------------------
# parquet step records (pyarrow-native: the server canonical env has pyarrow
# but not necessarily pandas on every node)
# --------------------------------------------------------------------------

def write_records_parquet(trace: dict, dt: float, method: str, flow: str,
                          path: "str | Path") -> None:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    T, N = trace["done"].shape
    t_idx = torch.arange(T).unsqueeze(-1).expand(T, N)
    env_idx = torch.arange(N).unsqueeze(0).expand(T, N)
    cols = {
        "t": (t_idx.flatten().float() * dt).numpy(),
        "env_id": env_idx.flatten().numpy().astype(np.int32),
        "ep_step": trace["ep_step"].flatten().numpy().astype(np.int32),
        "done": trace["done"].flatten().numpy(),
        "priority_p": trace["p"].flatten().numpy(),
        "in_conflict_tube": trace["tube"].flatten().numpy().astype(bool),
        "violation": trace["violation"].flatten().numpy().astype(bool),
        "min_margin_cross": trace["mm_cross"].flatten().numpy(),
        "min_margin_self_f": trace["mm_self_f"].flatten().numpy(),
        "min_margin_self_u": trace["mm_self_u"].flatten().numpy(),
        "min_margin_table": trace["mm_table"].flatten().numpy(),
    }
    for i, a in enumerate(ARM_KEYS):
        cols[f"alpha_{a}"] = trace["alpha"][..., i].flatten().numpy()
        cols[f"backstop_{a}"] = trace["backstop"][..., i].flatten().numpy().astype(bool)
        cols[f"cmd_norm_{a}"] = trace["cmd"][a].norm(dim=-1).flatten().numpy()
        cols[f"exec_norm_{a}"] = trace["exec"][a].norm(dim=-1).flatten().numpy()
    table = pa.table({k: pa.array(v) for k, v in cols.items()})
    table = table.append_column("method", pa.array([method] * (T * N)))
    table = table.append_column("delta_source", pa.array([flow] * (T * N)))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))


# --------------------------------------------------------------------------
# condition runner (Isaac side)
# --------------------------------------------------------------------------

def run_condition(env, driver, flow: str, steps: int, seed: int) -> dict:
    """One (method, flow) rollout on a live env: swap the delta source,
    re-seed, reset, drive, record. Returns the stacked trace."""
    env._delta_src = make_flow_source(flow, env)
    env._gen.manual_seed(seed)
    env._pending_cmd = None
    rec = StepRecorder(env, driver=driver)
    rec.install()
    try:
        obs, _ = env.reset()
        driver.reset(torch.arange(env.num_envs, device=env.device))
        t0 = time.time()
        for t in range(steps):
            action = driver.act(env, obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = (terminated | truncated)
            rec.record_done(done)
            done_ids = done.nonzero(as_tuple=True)[0]
            if done_ids.numel():
                driver.reset(done_ids)
            if t % 100 == 0:
                print(f"[block1] {flow} step {t}/{steps} "
                      f"({(t + 1) / max(time.time() - t0, 1e-9):.1f} steps/s)",
                      flush=True)
    finally:
        rec.uninstall()
    return rec.stacked()


def first_episode_batch(chop: dict) -> "EpisodeBatch | None":
    """Episode-0-per-env sub-batch (the paired-report guarantee holds only for
    the first episode of each env: identical scenario assignment + reset)."""
    ep_env, b = chop["ep_env"], chop["batch"]
    first_idx = []
    seen = set()
    for j, env in enumerate(ep_env.tolist()):
        if env not in seen:
            seen.add(env)
            first_idx.append(j)
    if not first_idx:
        return None
    sel = torch.tensor(first_idx, dtype=torch.long)
    return EpisodeBatch(
        delta_cmd={a: b.delta_cmd[a][:, sel] for a in ARM_KEYS},
        delta_exec={a: b.delta_exec[a][:, sel] for a in ARM_KEYS},
        alpha=b.alpha[:, sel], priority_p=b.priority_p[:, sel],
        in_tube=b.in_tube[:, sel], violation=b.violation[:, sel],
        backstop_active=b.backstop_active[:, sel], dt=b.dt)


def main() -> None:
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=str, default="",
                        help="SafeDuo checkpoint (required if 'safeduo' in --methods)")
    parser.add_argument("--run-dir", type=str, default="",
                        help="training run dir; replaces --ckpt with the "
                             "peak-harvest selection (C5-W7 default policy: "
                             "newest checkpoint inside the healthy window "
                             "judged from event shards -- the v3 m200 lesson;"
                             " the LAST checkpoint of a collapsed run is a "
                             "frozen statue with reward-flattering stats)")
    parser.add_argument("--ckpt-policy", choices=["peak", "last"],
                        default="peak",
                        help="with --run-dir: peak = freeze-aware harvest "
                             "(default), last = newest checkpoint (only for "
                             "deliberate post-collapse forensics)")
    parser.add_argument("--methods", nargs="+",
                        default=["safeduo", "estop", "speed", "cbf"],
                        choices=list(METHOD_CHOICES))
    parser.add_argument("--flows", nargs="+", default=["l2eval", "l1"],
                        choices=list(FLOWS))
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=1,
                        help="rollout horizon = episodes * max_episode_length")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--baseline-cfg", type=str, default="",
                        help='JSON: {"estop": {...}, "speed": {...}, "cbf": {...}}'
                             " ctor overrides (merged over --frozen-baselines)")
    parser.add_argument("--frozen-baselines", action="store_true",
                        help="start from the W5 grid winners "
                             "(BASELINE_FROZEN_W5) for estop/speed/cbf")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--env-yaml", type=str, default="duo_env.yaml",
                        help="env config yaml passed to make_duo_env_cfg; "
                             "v4 checkpoints MUST come with duo_env_v4.yaml "
                             "or the table is a wrong-scene table "
                             "(Round 116 debt, wired by A8)")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app  # noqa: F841  (kit must live for the run)

    from safeduo.envs.duo_env import DuoEnv, make_duo_env_cfg

    ckpt_pick = None
    if args.run_dir:
        if args.ckpt:
            raise SystemExit("--ckpt and --run-dir are mutually exclusive")
        if args.ckpt_policy == "peak":
            from safeduo.algo.freeze_sentinel import select_peak_from_events

            ckpt_pick = select_peak_from_events(args.run_dir)
            if ckpt_pick is None:
                raise SystemExit(
                    f"peak harvest found NO healthy checkpoint in "
                    f"{args.run_dir} (all-frozen run?) -- refusing to table "
                    f"a frozen statue; use --ckpt-policy last + --tag "
                    f"forensics if that is really intended")
            args.ckpt = ckpt_pick["path"]
            print(f"[block1] peak harvest: {args.ckpt} "
                  f"(healthy through iter {ckpt_pick['healthy_up_to_iter']})",
                  flush=True)
        else:
            cks = sorted((p for p in Path(args.run_dir).glob("model_*.pt")
                          if p.stem.split("_", 1)[1].isdigit()),
                         key=lambda p: int(p.stem.split("_", 1)[1]))
            if not cks:
                raise SystemExit(f"no model_*.pt in {args.run_dir}")
            args.ckpt = str(cks[-1])
            print(f"[block1] ckpt-policy last: {args.ckpt}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    overrides = {}
    if args.frozen_baselines:
        overrides = {k: dict(v) for k, v in BASELINE_FROZEN_W5.items()}
    for k, v in (json.loads(args.baseline_cfg) if args.baseline_cfg else {}).items():
        overrides[k] = {**overrides.get(k, {}), **v}
    cfg = make_duo_env_cfg(num_envs=args.num_envs, device=args.device,
                           yaml_name=args.env_yaml, coordinator=True)
    cfg.seed = args.seed
    env = DuoEnv(cfg)
    dt = cfg.sim.dt * cfg.decimation
    horizon = int(env.max_episode_length)
    steps = args.episodes * horizon

    results: dict = {}
    chops: dict = {}
    for method in args.methods:
        if method == "safeduo":
            if not args.ckpt:
                raise SystemExit("--ckpt required for the safeduo method")
            driver = PolicyDriver(args.ckpt, device=str(env.device))
        elif method == "passthrough":
            driver = PassthroughDriver()
        else:
            kw = overrides.get(method,
                               overrides.get("cbf") if method == "cbf_sched"
                               else None)
            driver = make_baseline_driver(method, env, kw)
        for flow in args.flows:
            print(f"[block1] === {method} x {flow} ({steps} steps x "
                  f"{args.num_envs} envs) ===", flush=True)
            trace = run_condition(env, driver, flow, steps, args.seed)
            write_records_parquet(trace, dt, method, flow,
                                  out_dir / f"records_{method}_{flow}.parquet")
            chop = chop_episodes(trace, dt, horizon)
            cm = condition_metrics(chop)
            results[(method, flow)] = cm
            chops[(method, flow)] = chop
            m = cm["metrics"]
            print(f"[block1] {method}/{flow}: coll {m['collision_episodes']}/"
                  f"{m['episodes']} interv {m['intervention_rate']:.3f} "
                  f"ttc {m['time_to_clear_mean']:.2f}s", flush=True)

    tab1 = render_tab1(results)
    (out_dir / "main_table.md").write_text(tab1, encoding="utf-8")
    pts = pareto_points(results)
    (out_dir / "pareto_points.json").write_text(json.dumps(pts, indent=2))

    anticip = None
    key_ours, key_base = ("safeduo", "l2eval"), ("cbf", "l2eval")
    if key_ours in chops and key_base in chops:
        b_ours = first_episode_batch(chops[key_ours])
        b_base = first_episode_batch(chops[key_base])
        if b_ours is not None and b_base is not None \
                and b_ours.N == b_base.N:
            anticip = anticipation_report(b_ours, b_base)
            anticip["paired_lead_values_s"] = \
                anticip["paired_lead_values_s"].tolist()
            (out_dir / "anticipation_paired.json").write_text(
                json.dumps(anticip, indent=2))
            with open(out_dir / "main_table.md", "a", encoding="utf-8") as f:
                f.write("\n## anticipation (paired, episode-1)\n\n"
                        "| pair | both-yield | paired lead mean [CI] (s) "
                        "| median | frac ours earlier | avoided |\n"
                        "|---|---|---|---|---|---|\n"
                        + anticipation_table_row(anticip) + "\n")

    manifest = {
        "tag": args.tag, "ckpt": args.ckpt, "methods": args.methods,
        "ckpt_policy": (args.ckpt_policy if args.run_dir else "explicit"),
        "ckpt_pick": ckpt_pick,
        "env_yaml": args.env_yaml,
        "flows": args.flows, "num_envs": args.num_envs,
        "episodes": args.episodes, "seed": args.seed, "dt": dt,
        "horizon": horizon, "baseline_cfg": overrides,
        "metrics": {f"{m}_{f}": r["metrics"] for (m, f), r in results.items()},
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for v in manifest["metrics"].values():
        v.pop("per_episode_intervention", None)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str))
    print(f"[block1] wrote {out_dir}/main_table.md", flush=True)
    print(tab1, flush=True)


if __name__ == "__main__":
    main()
