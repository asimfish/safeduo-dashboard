"""Self-contained PPO-Lagrangian trainer for the coordinator (v4 recipe, C5-W7).

Supervision ruling (STATUS Round 110 #5): the v4 anti-freeze mainline is
  PID-Lagrangian  margin/tube shaping costs move OFF the reward into
                  constraint channels with adaptive multipliers -- the
                  mechanistic fix for the v3 landscape disease where freezing
                  (reward -0.91) beat activity (-1.3) and PPO correctly
                  optimized the wrong landscape (both seeds collapsed
                  iter ~277-450, artifacts/analysis/p_head_v3_*).
  Beta actor      bounded support kills the clip-corner attractors the
                  Gaussian actor saturates into (alpha 4 + p 1 joint Beta).
  validity-aware GAE   true termination = no bootstrap, time-out = bootstrap
                  + broken trace, per stream (reward + 3 cost channels).
  peak harvest    checkpoints carry per-iter freeze stats so the harvester
                  picks the best HEALTHY checkpoint, not the last one
                  (v3 lesson: m200 beat m1999 by protection 0.45 vs frozen).

Why a self-built loop (the c2_lagrangian_v1 design, STATUS_C W3 sec.2c):
rsl_rl's OnPolicyRunner hardwires a Gaussian actor-critic, a single value
stream and its own GAE; swapping all three means forking the runner. train/
is A-line territory -- this loop lives entirely in algo/ and drives any env
with the DuoEnv step contract (KinematicDuoEnv locally, live DuoEnv on the
server via a thin launcher, see @A7 checklist in STATUS_C).

Cost channels: margin_cost_by_class (cross/self/table worst-pair proximity,
algo/reward_shaping) RECONSTRUCTED FROM THE STANDARD 275-dim OBSERVATION --
the P0 pair features are [dist, closing, class onehot] so the trainer needs
no env hooks and works identically on every env that speaks the obs layout.
Config numbers start from the double_hand reference (limits 0.05x3, PID
kp 0.5 / ki 0.05 / kd 0.1 / integral_limit 20, update once per iteration on
rollout-buffer means).

Local full-chain smoke (CPU, ~30 s):
  PYTHONPATH=src .venv/bin/python -m safeduo.algo.lagrangian_ppo --smoke
Acceptance: losses finite + critic learning, lambda responds to constraint
violation (rises while cost > limit, stays down below), no freeze signature
(alpha in band, p two-sided) -- summary lands in
artifacts/analysis/c5_lagrangian_smoke/. Reward mean is reported, not gated:
at 16 envs one terminal violation swings the batch mean ~0.26, see run_smoke.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from safeduo.algo.beta_actor import ACTION_DIM, BetaCoordinatorActor, MultiHeadCritic
from safeduo.algo.lagrangian import (
    DualGradientLagrangian,
    PIDLagrangian,
    combine_advantages,
    validity_aware_gae,
)
from safeduo.algo.reward_shaping import margin_cost_by_class
from safeduo.safety.semantics import semantics_with_safety_overrides
from safeduo.safety.types import (
    ARM_KEYS,
    CLASS_CROSS,
    CLASS_SELF,
    CLASS_TABLE,
    MAX_ACTIVE_PAIRS,
    PAIR_ARM_FEATURE_DIM,
    PAIR_FEATURE_DIM,
    TOTAL_DOF,
)

COST_HEADS = ("cross", "self", "table")
CRITIC_HEADS = ("reward",) + COST_HEADS
# duo_env._get_rewards shaped-reward breakdown, wandb task panel keys (C14);
# order/names = the summands cached in duo_env's step_cache (alpha_util only
# appears when the R15 w_alpha_util switch is on -- the wandb block skips
# absent keys, so schedule-off runs are unchanged)
REWARD_TERMS = ("track", "backstop", "tube", "progress", "smooth", "flip",
                "violation", "margin", "alpha_util")


# --------------------------------------------------------------------------
# cost channels straight from the standard observation
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ObsLayout:
    """Slice map of the duo_env coordinator observation (P0, 275 @ M=32):
    [q(26), qd(26), pair_feats(M*5), mask(M), cmd(26), alpha_prev(4), p_prev(1)].
    a25/P2: extra_dim counts the appended tail block (flags 8 + min-margin 4 +
    backlog 4 = 16 when duo_env p2_obs is on); it sits AFTER every existing
    slice so pair_start/mask_start stay valid for cost_channels_from_obs."""
    max_pairs: int = MAX_ACTIVE_PAIRS
    pair_dim: int = PAIR_FEATURE_DIM
    total_dof: int = TOTAL_DOF
    extra_dim: int = 0

    @property
    def pair_start(self) -> int:
        return 2 * self.total_dof

    @property
    def mask_start(self) -> int:
        return self.pair_start + self.max_pairs * self.pair_dim

    @property
    def obs_dim(self) -> int:
        return self.mask_start + self.max_pairs + self.total_dof + 5 + self.extra_dim


def cost_channels_from_obs(obs: Tensor, layout: "ObsLayout | None" = None,
                           d_warn: float = 0.05,
                           d_min_by_class: "dict | None" = None) -> Tensor:
    """(N, obs_dim) -> (N, 3) [cross, self, table] worst-pair proximity cost.

    Rebuilds the active-pair set from the pair-feature block ([dist, closing,
    onehot3], mask-zeroed by pair_obs_features) + mask block, then reuses
    margin_cost_by_class so the numbers are bit-identical to the env-side
    channels. Padding rows carry d=0 which would read as contact -- the mask
    keeps them out exactly as in the env.

    d_min_by_class (C14, C12_HANDOFF §2): {CLASS_* -> d_min} from the env
    profile; None keeps the v4 defaults (D_MIN_BY_CLASS_DEFAULT). Per-link
    overrides CANNOT be honored here (obs carries only the class onehot), so
    per-link rows are class-tier approximated -- exact per-row d_min comes
    from the env-side cost_channels hook when available (collect()).
    """
    lay = layout or ObsLayout()
    if obs.shape[-1] != lay.obs_dim:
        raise ValueError(f"obs dim {obs.shape[-1]} != layout {lay.obs_dim} "
                         "(M drifted? pass the right ObsLayout)")
    n = obs.shape[0]
    feats = obs[:, lay.pair_start:lay.mask_start].reshape(
        n, lay.max_pairs, lay.pair_dim)
    mask = obs[:, lay.mask_start:lay.mask_start + lay.max_pairs] > 0.5
    cls = feats[..., 2:5].argmax(dim=-1).to(feats.dtype)
    pairs = torch.stack([feats[..., 0], feats[..., 1], cls,
                         torch.zeros_like(cls)], dim=-1)
    return margin_cost_by_class(pairs, mask, d_warn=d_warn,
                                d_min_by_class=d_min_by_class)


# --------------------------------------------------------------------------
# R18 clutch-semantics primitives (S8 grid post-mortem recipe, 2026-08-21).
# Pure torch, unit-nailed in tests/test_r18_recipe.py; all trainer switches
# default off (zero drift, pinned by the R15 golden regression).
# --------------------------------------------------------------------------

def ste_binarize(alpha: Tensor, threshold: float = 0.5) -> Tensor:
    """Straight-through binarization: forward (alpha > threshold) exactly
    {0,1} (strict >, so exactly-at-threshold reads closed -- matches
    eval/clutch.py's `>= theta_hi` engage only up to the boundary point,
    which a continuous Beta sample hits with probability 0), backward
    identity. Numerically hard + (a - a.detach()): the parenthesized zero is
    exact in fp, so the forward value is the hard bit with no residue
    ((hard + a) - a would round through 1.501)."""
    hard = (alpha > threshold).to(alpha.dtype)
    return hard + (alpha - alpha.detach())


def hindsight_hazard_labels(hazard: Tensor, done: Tensor, horizon: int,
                            return_valid: bool = False):
    """(T, N, A) bool per-step hazard flags -> (T, N, A) float32 hindsight
    labels: 1 iff a hazard fires within the next `horizon` steps INCLUDING
    the current one, without crossing an episode boundary (done[t] = the
    transition at t ended the episode; flags on done rows are already zeroed
    by collect() because the post-step state there is the autoreset state).

    Backward scan, O(T) with (N, A) tensors -- rollout-buffer post-
    processing, zero physics overhead. Tail steps whose true future lies
    beyond the buffer read 0 (label noise at the last <horizon steps of each
    rollout; with T=24 < horizon=30 the labels degrade to "hazard visible
    before the buffer ends", documented in the a22 recipe notes).

    S14 (R18 hindsight-window fix): return_valid=True additionally returns
    a (T, N, A) bool mask, True where the label is fully determined inside
    the buffer -- a hazard is visible (definite 1), the episode truly ends
    within the window (boundary truncation is semantic, not an artifact),
    or the full `horizon` window lies inside the buffer. Ambiguous tail
    zeros (episode still running, window cut by the buffer end, no hazard
    seen) read False; cfg.hazard_tail_mask drops them from the BCE and
    zero-prices them in the dead-zone. NOTE capping horizon to T is an
    exact no-op on labels (in-buffer distances are <= T-1 < T), which is
    why the fix is a validity mask and not a cap. Default False = original
    single-tensor return, zero drift for every existing caller."""
    T = hazard.shape[0]
    far = float(T + horizon + 1)
    dist = torch.full(hazard.shape[1:], far, device=hazard.device)
    labels = torch.empty(hazard.shape, dtype=torch.float32,
                         device=hazard.device)
    zero = torch.zeros_like(dist)
    inf_t = torch.full_like(dist, far)
    std = valid = None
    if return_valid:
        # steps-to-episode-boundary (per env), same backward scan family
        std = torch.full(done.shape[1:], far, device=hazard.device)
        valid = torch.empty(hazard.shape, dtype=torch.bool,
                            device=hazard.device)
    for t in range(T - 1, -1, -1):
        dist = torch.where(hazard[t], zero,
                           torch.where(done[t].unsqueeze(-1), inf_t,
                                       dist + 1.0))
        labels[t] = (dist < float(horizon)).float()
        if return_valid:
            std = torch.where(done[t], torch.zeros_like(std), std + 1.0)
            valid[t] = ((dist < float(horizon))
                        | (std < float(horizon)).unsqueeze(-1)
                        | ((t + horizon) <= T))
    if return_valid:
        return labels, valid
    return labels


def alpha_deadzone_penalty(alpha_exec: Tensor, labels: Tensor, gray: Tensor,
                           w: float) -> Tensor:
    """(T, N, 4) EXECUTED alpha + hindsight labels + gray flags -> (T, N)
    reward add-on. Bidirectional dead-zone pricing (replaces w_alpha_util):
      hazard  (label 1): alpha^2      -- staying open into danger is priced;
      safe    (label 0, not gray): (1-alpha)^2 -- closing without danger is
                                       priced (the false-brake side);
      gray band (label 0, gray 1): 0  -- the hysteresis corridor is free, so
                                       the policy is not whipped across the
                                       threshold by two opposing quadratics.
    A hazard label overrides gray (danger must close, corridor or not)."""
    per_arm = (labels * alpha_exec.pow(2)
               + (1.0 - labels) * (1.0 - gray)
               * (1.0 - alpha_exec).pow(2))
    return -w * per_arm.sum(dim=-1)


def cost_sync_from_env_cfg(safety_cfg: dict, semantics_yaml: str) -> dict:
    """env yaml safety 节 -> 训练器 cost 通道参数（C14，C12_HANDOFF §1/§2）。

    与 duo_env.__init__ 消费同一 semantics_with_safety_overrides，保证塑形带
    （env）与约束通道（trainer）读同一组 d_warn / 逐类 d_min——v5 yaml 当年
    警告过的语义分叉（duo_env_v5.yaml L78-81）从机制上关死。v4 无 override 时
    产出 = 语义 yaml 基值 {0.05, cross 0.03, self 0.02, table 0.02}，与训练器
    旧硬编码逐位相同（零漂移）；v6 产出 {0.08, 0.038, 0.013, 0.020}。
    纯 python/yaml，无 isaac 依赖，本地可测。
    """
    sem = semantics_with_safety_overrides(semantics_yaml, safety_cfg)
    return {"d_warn": float(sem.d_warn),
            "d_min_by_class": {CLASS_CROSS: float(sem.d_min["cross"]),
                               CLASS_SELF: float(sem.d_min["self"]),
                               CLASS_TABLE: float(sem.d_min["table"])}}


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

@dataclass
class LagrangianPPOConfig:
    # rollout
    num_steps_per_env: int = 24            # = A5 runner cfg
    gamma: float = 0.995                   # v2 credit-assignment arbitration
    gae_lambda: float = 0.95
    # ppo
    clip_param: float = 0.2
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 5.0e-4
    entropy_coef: float = 0.003
    value_loss_coef: float = 1.0
    max_grad_norm: float = 1.0
    # nets
    hidden: tuple = (256, 256)
    min_concentration: float = 1.0
    # constraint controller (double_hand reference numbers)
    cost_limits: tuple = (0.05, 0.05, 0.05)
    # R10 (G3 analysis 2026-08-17, option 2): per-channel cost divisors that
    # bring the worst-pair quadratic proximity cost into the same dimension
    # as cost_limits. v6 evidence: self/table episode-mean cost sits at
    # 0.81/0.91 against limit 0.05 purely because the birth/hover geometry
    # lives inside the 80mm warn band -- the 0.05 target is unsatisfiable
    # and the PID integral pins at its cap from iter ~25 on (lambda becomes
    # a quasi-fixed penalty). Dividing the cost stream ONCE at collect()
    # scales the critic targets, the GAE advantages and the controller
    # comparison consistently (G3 execution constraint). (1,1,1) is an
    # exact float no-op -> zero drift for every existing caller.
    cost_scale: tuple = (1.0, 1.0, 1.0)
    # R13 (G3 RCA 2026-08-18) v6.2 anti-collapse knobs. All defaults are exact
    # no-ops (zero drift for existing callers):
    # - lambda_max: per-channel PID multiplier cap with anti-windup
    #   (v6.2_core: (0.35, 0, 0) -> cap lambda_cross at 0.35, others uncapped;
    #   entries <= 0 mean "no cap"). Rationale: v6.1 lambda_cross saturated
    #   near 1.0 vs a ~0.343 break-even shadow price (2.94x over-priced).
    # - entropy_coef_hi: when > 0, enables the adaptive entropy schedule --
    #   if the EMA(20) of policy entropy drops to <= entropy_ema_lo the
    #   effective coefficient latches to entropy_coef_hi until the EMA
    #   recovers to >= entropy_ema_hi (v6.2: base 0.006, hi 0.012, lo/hi
    #   thresholds -4.8/-4.5). Guards Beta-concentration blow-up before
    #   alpha_sat_lo takes off (v6.1 entropy collapsed -4.7 -> -7.7).
    lambda_max: "tuple | None" = None
    entropy_coef_hi: float = 0.0
    entropy_ema_lo: float = -4.8
    entropy_ema_hi: float = -4.5
    # R15 (A1/A2 v6.2 post-mortem 2026-08-19) utility-recovery knobs for the
    # v7 recipe. ALL defaults are exact no-ops (zero drift, pinned by the
    # golden regression in tests/test_r15_recipe.py):
    # - arm_aware_obs: pair rows carry two-arm one-hot identity (13 dims/row,
    #   obs 275 -> 531 @ M=32). Attacks the F_L constraint-sink bias
    #   (corr(lambda_cross, alpha_F_L) = -0.661, first-miss 76%): with shared
    #   lambdas and identity-blind rows the policy cannot bill cross pricing
    #   to the arm that causes it. MUST match the env-side switch
    #   (duo_env coordinator.arm_aware_obs) -- train_lagrangian sets both.
    # - w_armbal: per-arm alpha balance regularizer, adds
    #   w * (max_arm mean-alpha - min_arm mean-alpha) of the CURRENT policy
    #   (distribution mean) to the loss. Recommended start 0.01 -- small, so
    #   it equalizes by lifting the sunk arm rather than dragging good arms.
    # - ent_cost_gate: entropy latch engages only while
    #   cost_cross < gate * limit_cross. inf = no gate (pre-R15 behavior --
    #   NOTE 1.0 is NOT a no-op: v6.2 sat at/above the limit on 46.8% of
    #   iterations, where a 1.0 gate would veto latches the old code took).
    #   R15 recommended 0.9: no entropy injection while cross is at the
    #   limit -- exploration bonuses there just buy constraint violations.
    # - lambda_rate_max: PID multiplier slew limit per update (see
    #   lagrangian.PIDLagrangian). Recommended 0.01.
    arm_aware_obs: bool = False
    w_armbal: float = 0.0
    ent_cost_gate: float = float("inf")
    lambda_rate_max: float = float("inf")
    # R18 clutch-semantics retraining (S8 recipe, 2026-08-21). ALL defaults
    # are exact no-ops (zero drift, golden regression untouched):
    # - clutch_train: rollout alpha is binarized at clutch_threshold via the
    #   straight-through estimator BEFORE it reaches the env, killing the
    #   train-exec mismatch the S8 grid proved fatal (a continuous-alpha
    #   policy binarized post-hoc: false-brake <= 5% and full brake-stop are
    #   structurally exclusive). Stored raw samples stay continuous -- PPO
    #   log_prob replay and freeze_stats read the Beta sample as before.
    #   Training uses NO hysteresis (fixed 0.5); execution-side hysteresis
    #   stays in eval/clutch.py.
    # - w_hazard_bce: auxiliary BCE on the policy's mean alpha against
    #   hindsight hazard labels (margin below the lock line within
    #   hazard_horizon steps, env hazard_flags() hook + rollout-buffer
    #   backward scan). Teaches the head P(danger) semantics directly.
    # - w_alpha_deadzone: bidirectional dead-zone reward pricing on the
    #   EXECUTED alpha (see alpha_deadzone_penalty). Mutually exclusive with
    #   the env-side w_alpha_util (train_lagrangian asserts); the alpha_util
    #   code path stays intact for old recipes.
    # - hazard_tail_mask (S14, R18 hindsight-window fix): with rollout
    #   T=24 < hazard_horizon=30 EVERY step's lookahead is buffer-truncated
    #   and every 0-label is "no hazard seen before buffer end", not
    #   "safe for 0.5 s" (a22 regime, S10 known limitation). Capping the
    #   horizon to T is an exact label no-op (in-buffer distances <= T-1),
    #   so the fix is a validity mask: ambiguous tail samples (window cut
    #   by the buffer, episode still running, no hazard visible) leave the
    #   BCE (masked mean) and are priced as gray (zero) in the dead-zone.
    #   Pair with a horizon << T (recommended 12 = 0.2 s at 60 Hz with
    #   T=24: first 12 steps carry the full window) or a longer rollout
    #   (num_steps_per_env 48 + horizon 30). False = S10/a22 behavior.
    clutch_train: bool = False
    clutch_threshold: float = 0.5
    w_hazard_bce: float = 0.0
    w_alpha_deadzone: float = 0.0
    hazard_horizon: int = 30               # 0.5 s at 60 Hz control
    hazard_tail_mask: bool = False
    # a25/P3 (R22-F2): decouple BCE from the Beta head + soften the training
    # clutch. ALL defaults keep prior behavior bit-identical.
    # - hazard_head: BCE trains a dedicated per-arm hazard-logit head on the
    #   shared backbone (beta_actor.hazard_logits) instead of dragging the
    #   Beta mean; alpha then answers to RL signal only (deadzone/cost keep
    #   pricing danger) and entropy can recover. Fresh runs only -- the
    #   actor state_dict gains keys, old checkpoints won't load with it on.
    # - clutch_tau_start > 0: during TRAINING rollouts the executed alpha is
    #   sigmoid((alpha - thr)/tau) instead of the hard step. The hard clutch
    #   makes the env response (and thus reward) flat in alpha on each side
    #   of thr -- no preference gradient, alpha drifts to the walls. The
    #   soft ramp restores slope; tau anneals geometrically to
    #   clutch_tau_end over clutch_tau_anneal_iters so late training
    #   matches the product's hard-clutch semantics. 0.0 = hard (old path).
    hazard_head: bool = False
    clutch_tau_start: float = 0.0
    clutch_tau_end: float = 0.02
    clutch_tau_anneal_iters: int = 600
    # a25/P2 (R22-F3): policy obs carries the appended 16-dim safety tail
    # (per-arm hazard/gray flags + min-margin + target backlog) -- must
    # mirror duo_env's coordinator.p2_obs switch or the actor obs-dim check
    # fails loudly. train_lagrangian --p2-obs lights both sides.
    p2_obs: bool = False
    coupling_obs: bool = False
    w_coupling_sync: float = 0.0
    intervention_labels: bool = False
    # R22: class-balance the hazard BCE (pos/neg each carry half the loss).
    # With safe-heavy labels (~9:1 dense, ~1:1-but-tail-biased under S14)
    # a plain mean lets the majority class pin the alpha head.
    hazard_bce_balance: bool = False
    pid_kp: float = 0.5
    pid_ki: float = 0.05
    pid_kd: float = 0.1
    pid_integral_limit: float = 20.0
    controller: str = "pid"                # pid | dual | off (ablation arms)
    dual_lr: float = 0.05
    d_warn: float = 0.05
    # C14 (C12_HANDOFF §2): per-class d_min for the obs-side cost fallback;
    # None = D_MIN_BY_CLASS_DEFAULT (v4 semantics numbers -- zero drift for
    # every existing caller). train_lagrangian fills d_warn + this from the
    # env yaml via cost_sync_from_env_cfg (single source with the env side).
    d_min_by_class: "dict | None" = None
    # freeze signature gates (bc_smoke_check thresholds + A6 sentinel 0.15)
    freeze_alpha_mean: float = 0.15
    freeze_p_pin: float = 0.95
    # io
    save_interval: int = 50
    log_every: int = 1


# --------------------------------------------------------------------------
# trainer
# --------------------------------------------------------------------------

class LagrangianPPO:
    """PPO with a joint Beta policy, one reward + 3 cost critics, per-stream
    validity-aware GAE and a PID-controlled Lagrangian objective."""

    def __init__(self, env, cfg: "LagrangianPPOConfig | None" = None,
                 layout: "ObsLayout | None" = None,
                 log_dir: "str | Path | None" = None,
                 device: "str | torch.device" = "cpu", seed: int = 0,
                 wandb_logger=None):
        self.env = env
        # ---- wandb telemetry (R1, C13): purely additive sidecar. With the
        # default wandb_logger=None nothing below changes behavior -- no
        # telemetry frames are collected, stats.jsonl/events/checkpoints are
        # byte-identical to the pre-R1 loop.
        self.wandb = wandb_logger
        # env-side per-step diagnostics (tube/backstop/min_margin) come from
        # an OPTIONAL step_telemetry() hook on the env (train_lagrangian's
        # _EnvAdapter implements it by reading duo_env's existing _step_cache
        # /_last_out -- reuse, not recompute). Envs without the hook (e.g.
        # KinematicDuoEnv smoke) simply skip those panel keys.
        tel = getattr(env, "step_telemetry", None)
        self._telemetry = tel if (wandb_logger is not None
                                  and callable(tel)) else None
        # exemption-aware cost channels (C14, C12_HANDOFF §3 option b): an
        # OPTIONAL cost_channels() hook on the env serves the env-side
        # per-class cost (per-row d_min incl. near_table runtime exemptions,
        # computed on the post-step state duo_env caches in
        # _get_observations). Envs without the hook -- KinematicDuoEnv smoke,
        # test stubs -- fall back to the obs reconstruction in collect(),
        # i.e. the pre-C14 behavior, unchanged.
        ch = getattr(env, "cost_channels", None)
        self._cost_channels = ch if callable(ch) else None
        self._health: dict = {}
        self.cfg = cfg or LagrangianPPOConfig()
        # R18 (3)/(4): hindsight labels need per-arm hazard/gray flags from
        # an env hazard_flags() hook ((N, 4, 2) bool on the post-step state;
        # _EnvAdapter serves duo_env's cache, test stubs script their own).
        # Fail at construction, not mid-run: a silent no-label fallback
        # would train the BCE against garbage.
        self._r18_labels = (self.cfg.w_hazard_bce > 0.0
                            or self.cfg.w_alpha_deadzone > 0.0)
        hz = getattr(env, "hazard_flags", None)
        self._hazard_flags = hz if callable(hz) else None
        if self._r18_labels and self._hazard_flags is None:
            raise ValueError(
                "w_hazard_bce/w_alpha_deadzone require an env hazard_flags() "
                "hook (R18; duo_env fills it under coordinator."
                "r18_hazard_flags -- train_lagrangian sets both sides)")
        # R15 ①: with arm_aware_obs on and no explicit layout, expect the
        # extended pair rows (must mirror the env-side switch; obs-dim
        # mismatches fail loudly in cost_channels_from_obs / the actor)
        self.layout = layout or ObsLayout(
            pair_dim=PAIR_ARM_FEATURE_DIM if self.cfg.arm_aware_obs
            else PAIR_FEATURE_DIM,
            extra_dim=(16 if self.cfg.p2_obs else 0) + (6 if self.cfg.coupling_obs else 0))
        if self.cfg.w_coupling_sync > 0 and not self.cfg.coupling_obs:
            raise ValueError("w_coupling_sync requires coupling_obs")
        if self.cfg.intervention_labels and not callable(getattr(env, "intervention_flags", None)):
            raise ValueError("intervention_labels requires intervention_flags hook")
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self.actor = BetaCoordinatorActor(
            self.layout.obs_dim, hidden=tuple(self.cfg.hidden),
            min_concentration=self.cfg.min_concentration,
            hazard_head=self.cfg.hazard_head).to(self.device)
        self.critic = MultiHeadCritic(self.layout.obs_dim, heads=CRITIC_HEADS,
                                      hidden=tuple(self.cfg.hidden)).to(self.device)
        if self.cfg.controller == "pid":
            self.controller = PIDLagrangian(
                len(COST_HEADS), kp=self.cfg.pid_kp, ki=self.cfg.pid_ki,
                kd=self.cfg.pid_kd,
                integral_limit=self.cfg.pid_integral_limit,
                lambda_max=self.cfg.lambda_max,
                lambda_rate_max=self.cfg.lambda_rate_max).to(self.device)
        elif self.cfg.controller == "dual":
            self.controller = DualGradientLagrangian(
                len(COST_HEADS), learning_rate=self.cfg.dual_lr).to(self.device)
        elif self.cfg.controller == "off":
            self.controller = None
        else:
            raise ValueError(f"unknown controller {self.cfg.controller}")
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.cfg.learning_rate)
        self.cost_limits = torch.tensor(self.cfg.cost_limits,
                                        device=self.device)
        self.cost_scale = torch.tensor(self.cfg.cost_scale,
                                       device=self.device)
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.iteration = 0
        self._obs = None
        # adaptive entropy schedule state (inactive unless entropy_coef_hi > 0)
        self._ent_ema: "float | None" = None
        self._ent_hi_latched = False

    def _clutch_tau(self) -> float:
        """a25/P3 当前迭代的软离合温度；0.0 表示走硬二值旧路径。
        tau_start=0 时恒 0（零漂移）；否则从 tau_start 几何退火到 tau_end，
        anneal_iters 之后钉在 tau_end（不退到 0——最后一段語义交给评测侧
        的硬离合，训练侧保留极小斜率防止地形重新平坦化）。"""
        t0 = float(self.cfg.clutch_tau_start)
        if t0 <= 0.0:
            return 0.0
        t1 = max(float(self.cfg.clutch_tau_end), 1e-4)
        n = max(int(self.cfg.clutch_tau_anneal_iters), 1)
        frac = min(float(self.iteration) / float(n), 1.0)
        return float(t0 * ((t1 / t0) ** frac))

    # ---- rollout -----------------------------------------------------------
    @torch.no_grad()
    def collect(self) -> dict:
        T, N = self.cfg.num_steps_per_env, self.env.num_envs
        lay = self.layout
        obs_buf = torch.empty(T, N, lay.obs_dim, device=self.device)
        raw_buf = torch.empty(T, N, ACTION_DIM, device=self.device)
        logp_buf = torch.empty(T, N, device=self.device)
        rew_buf = torch.empty(T, N, device=self.device)
        cost_buf = torch.empty(T, N, len(COST_HEADS), device=self.device)
        term_buf = torch.empty(T, N, dtype=torch.bool, device=self.device)
        trunc_buf = torch.empty(T, N, dtype=torch.bool, device=self.device)
        val_buf = torch.empty(T + 1, N, len(CRITIC_HEADS), device=self.device)
        # R18 sidecar buffers, allocated only when a switch is on (default
        # path allocates nothing, roll dict keys unchanged -> zero drift)
        aexec_buf = (torch.empty(T, N, 4, device=self.device)
                     if (self.cfg.clutch_train
                         or self.cfg.w_alpha_deadzone > 0.0) else None)
        hz_buf = (torch.empty(T, N, 4, 2, dtype=torch.bool,
                              device=self.device)
                  if self._r18_labels else None)

        if self._obs is None:
            obs_d, _ = self.env.reset()
            self._obs = obs_d["policy"].to(self.device)
        obs = self._obs
        tel_frames = [] if self._telemetry is not None else None
        for t in range(T):
            sample = self.actor.act(obs)
            val_buf[t] = self.critic.stacked(obs)
            action = BetaCoordinatorActor.to_env_action(sample.raw)
            # R18 (2) clutch-train: the env EXECUTES the binarized alpha
            # (fixed 0.5, no hysteresis -- execution-side hysteresis lives
            # in eval/clutch.py); the STORED raw sample stays continuous so
            # PPO's Beta log_prob replay and freeze_stats are untouched.
            if self.cfg.clutch_train:
                tau = self._clutch_tau()
                if tau > 0.0:
                    # a25/P3 软离合：训练期用温度斜坡代替硬阶跃，让 env 响应
                    # （进而 reward）对 alpha 有斜率；tau 几何退火到贴近硬
                    # 离合。存储的 raw 样本不变，PPO replay 不受影响。
                    a_bin = torch.sigmoid(
                        (sample.raw[:, :4] - self.cfg.clutch_threshold) / tau)
                else:
                    a_bin = ste_binarize(sample.raw[:, :4],
                                         self.cfg.clutch_threshold)
                action = torch.cat([a_bin * 2.0 - 1.0, action[:, 4:]],
                                   dim=-1)
            if aexec_buf is not None:
                # what the env actually sees: (a+1)/2 inverts to_env_action
                aexec_buf[t] = (action[:, :4] + 1.0) * 0.5
            obs_d, reward, terminated, truncated, _extras = self.env.step(action)
            if self.cfg.coupling_obs and aexec_buf is not None and callable(getattr(self.env, "executed_alpha", None)):
                aexec_buf[t] = self.env.executed_alpha().to(self.device)
            nxt = obs_d["policy"].to(self.device)
            obs_buf[t] = obs
            raw_buf[t] = sample.raw
            logp_buf[t] = sample.log_prob
            rew_buf[t] = reward.to(self.device)
            term_buf[t] = terminated.to(self.device)
            trunc_buf[t] = truncated.to(self.device)
            # cost of action t = proximity of the post-step state, read from
            # obs_{t+1}. On done rows obs_{t+1} is the AUTO-RESET state of a
            # fresh episode -- charging it to the old episode would leak
            # birth proximity into the constraint account, so done rows are
            # zeroed (the violation itself is already charged through the
            # terminal reward penalty; the dying state's margins were charged
            # at t-1).
            done = term_buf[t] | trunc_buf[t]
            # cost source priority (C14): env-side exemption-aware channel
            # when the hook serves it (live DuoEnv), obs reconstruction
            # otherwise. The done-row zeroing applies to BOTH sources: the
            # env-side cache is also computed post-reset on done rows.
            cost = self._cost_channels() if self._cost_channels else None
            if cost is None:
                cost = cost_channels_from_obs(nxt, lay, self.cfg.d_warn,
                                              self.cfg.d_min_by_class)
            else:
                cost = cost.to(self.device)
            # R10 single scaling point: everything downstream (cost critics,
            # GAE, PID observed-vs-limit, stats/wandb cost_means) consumes
            # the scaled stream, keeping all views dimensionally consistent.
            cost = cost / self.cost_scale
            cost_buf[t] = torch.where(done.unsqueeze(-1),
                                      torch.zeros_like(cost), cost)
            # R18 (3): per-arm hazard/gray flags of the post-step state --
            # same convention as cost: done rows read the autoreset birth
            # state, which must not label the OLD episode's action -> zeroed
            if hz_buf is not None:
                flags = self._hazard_flags()
                if flags is None:
                    raise RuntimeError(
                        "env hazard_flags() returned None -- the R18 cache "
                        "was not filled (coordinator.r18_hazard_flags off?)")
                hz_buf[t] = flags.to(self.device) & ~done.reshape(N, 1, 1)
                if self.cfg.intervention_labels:
                    # Intervention belongs to transition t, even if it ended
                    # the episode. Do not erase it with the autoreset mask.
                    hz_buf[t, :, :, 0] |= self.env.intervention_flags().to(self.device)
            if tel_frames is not None:
                frame = self._telemetry()
                if frame:
                    tel_frames.append(frame)
            obs = nxt
        val_buf[T] = self.critic.stacked(obs)
        self._obs = obs
        roll = {"obs": obs_buf, "raw": raw_buf, "logp": logp_buf,
                "reward": rew_buf, "cost": cost_buf, "terminated": term_buf,
                "truncated": trunc_buf, "values": val_buf}
        if aexec_buf is not None:
            roll["alpha_exec"] = aexec_buf
        if hz_buf is not None:
            roll["hazard"] = hz_buf[..., 0]
            roll["gray"] = hz_buf[..., 1]
        if tel_frames:
            roll["telemetry"] = tel_frames
        return roll

    # ---- advantage assembly ---------------------------------------------------
    def _targets(self, roll: dict) -> dict:
        """Per-stream validity-aware GAE. next_values[t] = V(s_{t+1}); on
        time-outs the terminal observation is lost to autoreset, so the
        bootstrap falls back to V(s_t) -- the same approximation rsl_rl's
        time-out handling applies, semantics documented in lagrangian.py."""
        cfg = self.cfg
        term, trunc = roll["terminated"], roll["truncated"]
        valid = torch.ones_like(term)
        out = {}
        streams = {"reward": roll["reward"]}
        for i, c in enumerate(COST_HEADS):
            streams[c] = roll["cost"][..., i]
        for h, r in streams.items():
            k = CRITIC_HEADS.index(h)
            v, v_next_seq = roll["values"][:-1, :, k], roll["values"][1:, :, k]
            v_next = torch.where(trunc & ~term, v, v_next_seq)
            out[h] = validity_aware_gae(
                r, v, v_next, term, trunc, learning_sample_valid=valid,
                gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)
        return out

    # ---- update -------------------------------------------------------------
    def update(self, roll: dict) -> dict:
        cfg = self.cfg
        # R18 (3)/(4): hindsight labels + dead-zone reward relabel, computed
        # BEFORE GAE so the penalty flows into the reward critic/advantages.
        # `roll` is shallow-rebound, never mutated: learn()'s freeze_stats /
        # stats.jsonl reward_mean keep reporting the RAW env reward (shaping
        # must not fake the task metric).
        r18_stats: dict = {}
        # a25/P3：软离合温度入册（键只在功能开启时存在，零漂移）
        if self.cfg.clutch_train and self.cfg.clutch_tau_start > 0.0:
            r18_stats["clutch_tau"] = round(self._clutch_tau(), 5)
        labels_flat = None
        valid_flat = None
        if self._r18_labels:
            done = roll["terminated"] | roll["truncated"]
            valid = None
            if cfg.hazard_tail_mask:
                # S14: ambiguous buffer-tail labels leave the loss surfaces
                labels, valid = hindsight_hazard_labels(
                    roll["hazard"], done, cfg.hazard_horizon,
                    return_valid=True)
                r18_stats["hazard_valid_frac"] = round(
                    float(valid.float().mean().item()), 5)
            else:
                labels = hindsight_hazard_labels(roll["hazard"], done,
                                                 cfg.hazard_horizon)
            r18_stats["hazard_rate"] = round(float(labels.mean().item()), 5)
            if cfg.w_alpha_deadzone > 0.0:
                gray = roll["gray"].float()
                if valid is not None:
                    # ambiguous rows priced like the gray band: no side
                    # charged (a definite hazard label still overrides)
                    gray = torch.maximum(gray, 1.0 - valid.float())
                dz = alpha_deadzone_penalty(roll["alpha_exec"], labels,
                                            gray, cfg.w_alpha_deadzone)
                dz = torch.where(done, torch.zeros_like(dz), dz)
                r18_stats["deadzone_pen"] = round(float(dz.mean().item()), 5)
                roll = {**roll, "reward": roll["reward"] + dz}
            if cfg.w_hazard_bce > 0.0:
                labels_flat = labels.reshape(-1, 4)
                if valid is not None:
                    valid_flat = valid.reshape(-1, 4).float()
        targets = self._targets(roll)
        adv_r = targets["reward"].advantages
        adv_c = torch.stack([targets[c].advantages for c in COST_HEADS], -1)
        observed = roll["cost"].mean(dim=(0, 1))
        if self.controller is not None:
            multipliers = self.controller.update(observed, self.cost_limits)
        else:
            multipliers = torch.zeros(len(COST_HEADS), device=self.device)
        T, N = adv_r.shape
        flat = lambda x: x.reshape(T * N, *x.shape[2:])
        obs, raw = flat(roll["obs"]), flat(roll["raw"])
        old_logp = flat(roll["logp"])
        adv = combine_advantages(flat(adv_r), flat(adv_c), multipliers)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        returns = torch.stack([targets[h].returns for h in CRITIC_HEADS], -1)
        returns = flat(returns)

        # effective entropy coefficient for this iteration (R13 schedule).
        # The latch state was advanced at the END of the previous update()
        # from that iteration's entropy EMA -- coefficient is constant within
        # one iteration's minibatches.
        ent_coef = cfg.entropy_coef
        if cfg.entropy_coef_hi > 0.0 and self._ent_hi_latched:
            ent_coef = cfg.entropy_coef_hi

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        # R15 ①: additive stats key only when the regularizer is on, so
        # stats.jsonl stays byte-identical for w_armbal=0 runs
        if cfg.w_armbal > 0.0:
            stats["armbal"] = 0.0
        if cfg.w_coupling_sync > 0.0:
            stats["coupling_sync"] = 0.0
            r18_stats["coupled_sample_fraction"] = float((obs[:, -6:].sum(-1) > 0).float().mean())
        if labels_flat is not None:
            stats["hazard_bce"] = 0.0
        vl_head = torch.zeros(len(CRITIC_HEADS), device=self.device)
        # wandb health-panel extras (approx KL / clip fraction). Kept OUT of
        # the returned stats dict on purpose: stats feeds stats.jsonl and the
        # zero-drift discipline requires that file byte-identical when wandb
        # is off; these land in self._health for the wandb block only.
        kl_sum, clip_sum = 0.0, 0.0
        n_upd = 0
        batch = T * N
        mb = batch // cfg.num_mini_batches
        for _ in range(cfg.num_learning_epochs):
            perm = torch.randperm(batch, device=self.device)
            for i in range(cfg.num_mini_batches):
                idx = perm[i * mb:(i + 1) * mb]
                logp, entropy = self.actor.evaluate_actions(obs[idx], raw[idx])
                ratio = (logp - old_logp[idx]).exp()
                surr = ratio * adv[idx]
                surr_cl = ratio.clamp(1 - cfg.clip_param,
                                      1 + cfg.clip_param) * adv[idx]
                policy_loss = -torch.min(surr, surr_cl).mean()
                # R15 ① per-arm alpha balance regularizer: range (max - min)
                # of the CURRENT policy's per-arm mean alpha (distribution
                # mean over the minibatch -- differentiable in the actor
                # params, unlike the stored rollout samples). Off by default
                # (no extra forward pass, no loss term, no stats key).
                armbal = None
                head_alpha = None
                # one shared forward for armbal + BCE (R18); the armbal-only
                # path computes the same .mean[:, :4].mean(dim=0) as before.
                # a25/P3 hazard_head 开启时 BCE 不再吃 Beta 均值，distribution
                # 前向只为 armbal 保留。
                if cfg.w_armbal > 0.0 or cfg.w_coupling_sync > 0.0 or (labels_flat is not None
                                          and not cfg.hazard_head):
                    head_alpha = self.actor.distribution(obs[idx]).mean[:, :4]
                if cfg.w_armbal > 0.0:
                    mean_alpha = head_alpha.mean(dim=0)
                    armbal = mean_alpha.max() - mean_alpha.min()
                # R18 (3): auxiliary BCE -- the head's mean alpha is trained
                # toward 1 - hazard_label, i.e. alpha reads P(safe) and
                # (1 - alpha) reads P(danger within 0.5 s).
                # a25/P3: hazard_head 开启时改为专用 logit 头预测危险
                # （target = label 本身，不再取反），alpha 头从 BCE 中解放。
                bce = None
                if labels_flat is not None:
                    if cfg.hazard_head:
                        el = torch.nn.functional.binary_cross_entropy_with_logits(
                            self.actor.hazard_logits(obs[idx]),
                            labels_flat[idx], reduction="none")
                    else:
                        el = torch.nn.functional.binary_cross_entropy(
                            head_alpha.clamp(1e-6, 1.0 - 1e-6),
                            1.0 - labels_flat[idx], reduction="none")
                    if valid_flat is None:
                        w = torch.ones_like(el)
                    else:
                        # S14 masked mean: sum over valid samples only
                        # (weight= + reduction='mean' would divide by numel
                        # and shrink the loss with the invalid fraction)
                        w = valid_flat[idx]
                    if cfg.hazard_bce_balance:
                        # R22: hazard and safe halves contribute equally --
                        # per-class weights sum to tot/2 each, so a class
                        # missing from the batch simply drops out instead
                        # of letting the other one dominate the mean
                        lab = labels_flat[idx]
                        pos = ((lab > 0.5).float() * w).sum()
                        neg = ((lab <= 0.5).float() * w).sum()
                        tot = (pos + neg).clamp_min(1.0)
                        w = w * torch.where(
                            lab > 0.5,
                            tot / (2.0 * pos.clamp_min(1.0)),
                            tot / (2.0 * neg.clamp_min(1.0)))
                    bce = (el * w).sum() / w.sum().clamp_min(1.0)
                values = self.critic.stacked(obs[idx])
                per_head = (values - returns[idx]).pow(2).mean(dim=0)
                value_loss = per_head.mean()
                loss = (policy_loss + cfg.value_loss_coef * value_loss
                        - ent_coef * entropy.mean())
                if armbal is not None:
                    loss = loss + cfg.w_armbal * armbal
                if cfg.w_coupling_sync > 0.0:
                    from safeduo.safety.coupling import synchronization_loss
                    sync_loss = synchronization_loss(head_alpha, obs[idx, -6:])
                    loss = loss + cfg.w_coupling_sync * sync_loss
                    stats["coupling_sync"] += float(sync_loss.detach())
                if bce is not None:
                    loss = loss + cfg.w_hazard_bce * bce
                if cfg.coupling_obs and not torch.isfinite(loss):
                    raise FloatingPointError("a31 non-finite loss; refusing optimizer update")
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    cfg.max_grad_norm)
                self.optimizer.step()
                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy.mean().item())
                if armbal is not None:
                    stats["armbal"] += float(armbal.item())
                if bce is not None:
                    stats["hazard_bce"] += float(bce.item())
                with torch.no_grad():
                    log_ratio = logp - old_logp[idx]
                    kl_sum += float(((log_ratio.exp() - 1.0)
                                     - log_ratio).mean().item())
                    clip_sum += float(((ratio - 1.0).abs()
                                       > cfg.clip_param).float().mean().item())
                vl_head += per_head.detach()
                n_upd += 1
        for k in stats:
            stats[k] /= max(n_upd, 1)
        # R18 iteration-level scalars (already means, added after the
        # per-minibatch division; keys only exist when a switch is on)
        stats.update(r18_stats)
        self._health = {"approx_kl": kl_sum / max(n_upd, 1),
                        "clip_frac": clip_sum / max(n_upd, 1)}
        # advance the entropy-schedule latch from this iteration's mean
        # entropy (EMA over ~20 iterations). Only active when the schedule is
        # enabled; the extra stats key is additive and only appears then, so
        # stats.jsonl stays byte-identical for schedule-off runs.
        if cfg.entropy_coef_hi > 0.0:
            beta = 1.0 / 20.0
            e = stats["entropy"]
            self._ent_ema = e if self._ent_ema is None else (
                (1.0 - beta) * self._ent_ema + beta * e)
            if self._ent_hi_latched:
                if self._ent_ema >= cfg.entropy_ema_hi:
                    self._ent_hi_latched = False
            elif self._ent_ema <= cfg.entropy_ema_lo and (
                    # R15 ②: cost-linked engage gate -- only inject entropy
                    # while cross has margin to the limit (v6.2 lesson: the
                    # -4.8 trigger never fired -- EMA floor -4.274 -- and had
                    # it fired near the limit, the extra exploration would be
                    # spent on constraint violations). inf = no gate, the
                    # pre-R15 trigger bit-identically; release is unchanged.
                    float(observed[COST_HEADS.index("cross")])
                    < cfg.ent_cost_gate
                    * float(self.cost_limits[COST_HEADS.index("cross")])):
                self._ent_hi_latched = True
            stats["entropy_coef"] = round(ent_coef, 5)
            stats["entropy_ema"] = round(self._ent_ema, 5)
            self._health["entropy_coef"] = ent_coef
            self._health["entropy_ema"] = self._ent_ema
        stats["value_loss_by_head"] = {
            h: round(float(vl_head[i].item()) / max(n_upd, 1), 5)
            for i, h in enumerate(CRITIC_HEADS)}
        stats["multipliers"] = [round(float(v), 5) for v in multipliers]
        stats["cost_means"] = [round(float(v), 5) for v in observed]
        return stats

    # ---- freeze signature (the v3 disease, per-iteration) --------------------
    @staticmethod
    def freeze_stats(roll: dict, cfg: LagrangianPPOConfig) -> dict:
        raw = roll["raw"]
        alpha = raw[..., :4].reshape(-1, 4)
        p = raw[..., 4].reshape(-1) * 2.0 - 1.0
        arm_means = alpha.mean(dim=0)
        a_mean = float(alpha.mean().item())
        p_pin = float((p.abs() > 0.9).float().mean().item())
        out = {
            "alpha_mean": round(a_mean, 4),
            "alpha_arm_means": [round(float(v), 4) for v in arm_means],
            "alpha_std": round(float(alpha.std().item()), 4),
            "alpha_sat_lo": round(float((alpha < 0.05).float().mean().item()), 4),
            "p_pinned_frac": round(p_pin, 4),
            "p_frac_pos": round(float((p > 0.05).float().mean().item()), 4),
            "p_frac_neg": round(float((p < -0.05).float().mean().item()), 4),
            "reward_mean": round(float(roll["reward"].mean().item()), 4),
            "frozen": bool(a_mean < cfg.freeze_alpha_mean
                           or p_pin > cfg.freeze_p_pin),
        }
        # R18 (2): EXECUTED alpha mean (binary open-fraction under
        # clutch_train). Additive key, absent for default runs. NOTE the
        # freeze gates above stay on the RAW Beta sample on purpose --
        # under clutch_train the executed stream is 0/1 by design and would
        # trip alpha_sat_lo-style reasoning; the sentinel reads these raw
        # keys and needs no change.
        if "alpha_exec" in roll:
            out["alpha_exec_mean"] = round(
                float(roll["alpha_exec"].mean().item()), 4)
        return out

    # ---- checkpointing ---------------------------------------------------------
    def save(self, tag: "int | str | None" = None, stats: "dict | None" = None) -> Path:
        assert self.log_dir is not None, "log_dir required for save()"
        tag = self.iteration if tag is None else tag
        path = self.log_dir / f"model_{tag}.pt"
        torch.save({
            "format": "lagrangian_ppo_v1",
            "iteration": self.iteration,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "controller": (self.controller.state_dict()
                           if self.controller is not None else None),
            "optimizer": self.optimizer.state_dict(),
            "config": asdict(self.cfg),
            "obs_dim": self.layout.obs_dim,
            "stats": stats or {},
        }, path)
        return path

    def load(self, path: "str | Path") -> dict:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        if self.controller is not None and ck.get("controller"):
            self.controller.load_state_dict(ck["controller"])
        if "optimizer" in ck:
            self.optimizer.load_state_dict(ck["optimizer"])
        self.iteration = int(ck.get("iteration", 0))
        return ck

    # ---- curriculum hook (R2/T3 trainer-side wiring, C13) ---------------------
    def _delta_source(self):
        """The env's delta source, unwrapping one adapter level if present
        (train_lagrangian._EnvAdapter wraps the live DuoEnv)."""
        src = getattr(self.env, "_delta_src", None)
        if src is None:
            src = getattr(getattr(self.env, "env", None), "_delta_src", None)
        return src

    # ---- main loop -----------------------------------------------------------
    def learn(self, iterations: int, quiet: bool = False, max_wall_seconds: float = 0.0) -> list:
        history = []
        started = time.monotonic()
        stats_file = (self.log_dir / "stats.jsonl").open("a") \
            if self.log_dir else None
        # T3 amp/mix curriculum (R2 trainer half): sources exposing
        # set_progress (ConflictMixSource) advance per iteration on
        # it/max_it. Under the default delta_curriculum.yaml (stages: null,
        # amp_stages: null) set_progress recomputes the same flat weights and
        # skips the amp update -- a true no-op, so existing recipes drift
        # zero; stage schedules only engage via --curriculum-yaml (t3).
        src = self._delta_source()
        curriculum = src if (src is not None
                             and hasattr(src, "set_progress")) else None
        total = max(self.iteration + iterations, 1)
        for _ in range(iterations):
            if history and max_wall_seconds > 0 and time.monotonic() - started >= max_wall_seconds:
                break
            frac = self.iteration / total
            if curriculum is not None:
                curriculum.set_progress(frac)
            t0 = time.time()
            roll = self.collect()
            upd = self.update(roll)
            fz = self.freeze_stats(roll, self.cfg)
            rec = {"iter": self.iteration, **upd, **fz,
                   "wall_s": round(time.time() - t0, 2)}
            history.append(rec)
            if stats_file:
                stats_file.write(json.dumps(rec) + "\n")
                stats_file.flush()
            if not quiet and self.iteration % self.cfg.log_every == 0:
                print("LAG_PPO " + json.dumps(rec), flush=True)
            if self.wandb is not None and self.wandb.active:
                self.wandb.log(self.iteration,
                               self._wandb_metrics(rec, roll, frac, curriculum))
            self.iteration += 1
            if self.log_dir and self.iteration % self.cfg.save_interval == 0:
                path = self.save(stats=rec)
                if self.wandb is not None:
                    self.wandb.log_checkpoint(self.iteration, path,
                                              score=rec.get("reward_mean"))
        if stats_file:
            stats_file.close()
        if self.log_dir:
            path = self.save(tag="last", stats=history[-1] if history else None)
            if self.wandb is not None and history:
                self.wandb.log_checkpoint(self.iteration, path,
                                          score=history[-1].get("reward_mean"))
        return history

    # ---- wandb panels (MASTER_REPORT §5, R1) ---------------------------------
    def _wandb_metrics(self, rec: dict, roll: dict, frac: float,
                       curriculum) -> dict:
        """Assemble the six §5 panel groups from what the loop already has.

        Known §5 gaps, on purpose (do not enlarge the rollout structure):
        - task panel reward breakdown (task/reward_{term} eight keys, C14):
          served by duo_env's step_cache["reward_terms"] cache through the
          step_telemetry hook -- live DuoEnv only; KinematicDuoEnv smoke has
          no hook and reports just the totals, as before.
        - geometry + tube/backstop rates come from the optional env
          step_telemetry() hook, so they only light up on the live DuoEnv
          run (KinematicDuoEnv smoke has no hook).
        - eval-video sampling (artifacts panel) is an eval-side job, not a
          trainer-side one; checkpoints are covered via log_checkpoint.
        """
        m = {
            # -- task group ---------------------------------------------------
            "task/reward_mean": rec["reward_mean"],
            "task/reward_std": float(roll["reward"].std().item()),
            # -- policy health group -------------------------------------------
            "health/alpha_mean": rec["alpha_mean"],
            "health/alpha_std": rec["alpha_std"],
            "health/p_pinned_frac": rec["p_pinned_frac"],
            "health/p_frac_pos": rec["p_frac_pos"],
            "health/p_frac_neg": rec["p_frac_neg"],
            "health/entropy": rec["entropy"],
            "health/frozen": float(rec["frozen"]),
            # -- system group ---------------------------------------------------
            "sys/iter_wall_s": rec["wall_s"],
            "sys/fps_ctrl_steps": round(
                self.cfg.num_steps_per_env * self.env.num_envs
                / max(rec["wall_s"], 1e-9), 1),
            "sys/curriculum_progress": round(frac, 5),
            # loss curves ride along (not a §5 panel but free)
            "loss/policy": rec["policy_loss"],
            "loss/value": rec["value_loss"],
        }
        m.update({f"health/{k}": round(v, 6)
                  for k, v in self._health.items()})
        # R18 panels, only when the switches emitted the keys
        for src_k, dst_k in (("hazard_rate", "health/hazard_rate"),
                             ("hazard_valid_frac", "health/hazard_valid_frac"),
                             ("hazard_bce", "loss/hazard_bce"),
                             ("deadzone_pen", "task/reward_deadzone"),
                             ("alpha_exec_mean", "health/alpha_exec_mean")):
            if src_k in rec:
                m[dst_k] = rec[src_k]
        # safety group: cost channels vs limits + lambdas
        for i, h in enumerate(COST_HEADS):
            m[f"safety/cost_{h}"] = rec["cost_means"][i]
            m[f"safety/limit_{h}"] = float(self.cfg.cost_limits[i])
            m[f"safety/lambda_{h}"] = rec["multipliers"][i]
        # per-arm alpha distribution (P10/P90 catch one-arm freezes the
        # global mean hides)
        raw = roll["raw"]
        alpha = raw[..., :4].reshape(-1, 4)
        qs = torch.quantile(
            alpha.float(), torch.tensor([0.1, 0.9], device=alpha.device), dim=0)
        for i, arm in enumerate(ARM_KEYS):
            m[f"health/alpha_{arm}_mean"] = rec["alpha_arm_means"][i]
            m[f"health/alpha_{arm}_p10"] = round(float(qs[0, i].item()), 4)
            m[f"health/alpha_{arm}_p90"] = round(float(qs[1, i].item()), 4)
        m["health/p_mean"] = round(
            float((raw[..., 4].reshape(-1) * 2.0 - 1.0).mean().item()), 4)
        # env-side telemetry (live DuoEnv only, see docstring)
        tel = roll.get("telemetry")
        if tel:
            cat = lambda k: torch.cat([f[k].reshape(f[k].shape[0], -1).float()
                                       for f in tel if k in f])
            if "violation" in tel[0]:
                m["safety/violation_rate"] = round(
                    float(cat("violation").mean().item()), 5)
            if "tube" in tel[0]:
                m["safety/tube_fraction"] = round(
                    float(cat("tube").mean().item()), 5)
            if "bs_active" in tel[0]:
                bs = cat("bs_active").reshape(-1, len(ARM_KEYS)).mean(dim=0)
                for i, arm in enumerate(ARM_KEYS):
                    m[f"safety/backstop_rate_{arm}"] = round(
                        float(bs[i].item()), 5)
            for k in ("mm_cross", "mm_table", "mm_self_f", "mm_self_u"):
                if k in tel[0]:
                    v = cat(k).reshape(-1)
                    p = torch.quantile(
                        v, torch.tensor([0.01, 0.5], device=v.device))
                    m[f"geom/{k}_p1"] = round(float(p[0].item()), 5)
                    m[f"geom/{k}_p50"] = round(float(p[1].item()), 5)
            # task panel reward breakdown (C14): rew_* frames are 0-dim
            # env-means cached by duo_env._get_rewards; iteration value =
            # mean over the rollout's steps (same averaging as reward_mean)
            for name in REWARD_TERMS:
                k = f"rew_{name}"
                if k in tel[0]:
                    m[f"task/reward_{name}"] = round(
                        sum(float(f[k]) for f in tel if k in f) / len(tel), 6)
        else:
            # violation fallback: both DuoEnv and KinematicDuoEnv terminate
            # on violation, so the terminated flag is the violation flag
            # (only diverges under terminate_on_violation=False measurement
            # protocols, which never train).
            m["safety/violation_rate"] = round(
                float(roll["terminated"].float().mean().item()), 5)
        # system group: current amp curriculum stage + GPU memory
        amp = getattr(curriculum, "amp_max", None) if curriculum else None
        if amp is not None:
            m["sys/amp_max"] = float(amp)
        if self.device.type == "cuda" and torch.cuda.is_available():
            m["sys/gpu_mem_gb"] = round(
                torch.cuda.memory_allocated(self.device) / 2 ** 30, 3)
        return m


# --------------------------------------------------------------------------
# peak-harvest discipline (the m200 lesson, mechanized)
# --------------------------------------------------------------------------

def select_peak_checkpoint(run_dir: "str | Path",
                           alpha_band: tuple = (0.3, 0.9),
                           p_pin_max: float = 0.95) -> "dict | None":
    """Best HEALTHY checkpoint of a run, not the last one.

    v3 lesson (STATUS_A W6 sec.1): both seeds collapsed mid-training and the
    final checkpoints were frozen statues; the iter-200 peaks beat r2's
    2000-iter finals. 'Peak' here = the newest saved checkpoint whose
    training-time freeze stats are healthy (alpha mean in band, p not
    pinned); reward is NOT the criterion because the freeze attractor
    RAISES reward (-1.3 -> -0.91) while killing the policy.

    Reads stats.jsonl written by LagrangianPPO.learn (or any compatible
    sidecar). Returns {"path", "iter", "stats"} or None if no checkpoint
    qualifies (all-frozen run -> caller must treat the run as failed).

    Tag semantics: learn() saves model_N AFTER finishing iteration N-1, so
    checkpoint N qualifies iff iteration N-1 was still healthy.
    """
    run_dir = Path(run_dir)
    stats_path = run_dir / "stats.jsonl"
    if not stats_path.exists():
        return None
    healthy_up_to = -1
    by_iter = {}
    for line in stats_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        by_iter[int(rec["iter"])] = rec
        a = rec.get("alpha_mean")
        pin = rec.get("p_pinned_frac", 0.0)
        if a is not None and alpha_band[0] <= a <= alpha_band[1] \
                and pin <= p_pin_max:
            healthy_up_to = max(healthy_up_to, int(rec["iter"]))
    if healthy_up_to < 0:
        return None
    best = None
    for ck in run_dir.glob("model_*.pt"):
        tag = ck.stem.split("_", 1)[1]
        if not tag.isdigit():
            continue
        it = int(tag)
        if it <= healthy_up_to + 1 and (best is None or it > best[0]):
            best = (it, ck)
    if best is None:
        return None
    near = by_iter.get(best[0], by_iter.get(best[0] - 1, {}))
    return {"path": str(best[1]), "iter": best[0], "stats": near}


# --------------------------------------------------------------------------
# local full-chain smoke
# --------------------------------------------------------------------------

def run_smoke(n_envs: int = 16, iterations: int = 25, seed: int = 0,
              out_dir: "str | Path | None" = None, quiet: bool = False) -> dict:
    """KinematicDuoEnv full chain: real geometry + conflict traffic + Beta
    actor + PID-Lagrangian. Acceptance gates returned in the summary."""
    from safeduo.algo.kinematic_env import KinematicDuoEnv, KinematicEnvConfig

    env = KinematicDuoEnv(
        n_envs, cfg=KinematicEnvConfig(episode_length=120), seed=seed)
    out = Path(out_dir) if out_dir else None
    trainer = LagrangianPPO(env, LagrangianPPOConfig(save_interval=10),
                            log_dir=out, seed=seed)
    history = trainer.learn(iterations, quiet=quiet)
    k = max(3, iterations // 5)
    late = history[-k:]
    lam_series = [h["multipliers"] for h in history]
    cost_series = [h["cost_means"] for h in history]
    lam_max = [max(l[c] for l in lam_series) for c in range(3)]
    # lambda responds = a channel above its limit drives its multiplier off
    # zero; regulating = by the end every channel is either under 1.5x limit
    # or still actively priced
    responded = any(m > 1e-4 for m in lam_max)
    ended_regulated = all(
        cost_series[-1][c] <= trainer.cfg.cost_limits[c] * 1.5
        or lam_series[-1][c] > 1e-4 for c in range(3))
    # critic gate rides the SELF cost head: it is the dense spike-free
    # stream at this traffic (rest pose sits inside the self warn band,
    # C3-W5 geometry fact), so its fit must improve if learning works at
    # all. reward/total value losses spike two orders of magnitude on the
    # iterations containing a terminal violation (-100 is unpredictable by
    # construction, measured vl_r 0.07 -> 127 on violation iters) -- medians
    # over thirds keep the gate robust to those.
    from statistics import median

    vl_self = [h["value_loss_by_head"]["self"] for h in history]
    early_vl = median(vl_self[:k])
    late_vl = median(vl_self[-k:])
    late_alpha = sum(h["alpha_mean"] for h in late) / k
    # NOTE reward_mean is deliberately NOT a gate at smoke scale: with 16
    # envs a single terminal violation (-100) swings the batch mean by ~0.26
    # (measured), so first-vs-last reward is coin-flip noise; the trend is
    # reported for eyeballing and the real reward gate belongs to the
    # 4096-env server run's first-shard checks.
    checks = {
        "loss_finite": all(map(lambda h: abs(h["policy_loss"]) < 1e6, history)),
        "critic_learning": late_vl < early_vl,
        "lambda_responded": responded,
        "lambda_regulating": ended_regulated,
        "no_freeze_signature": not any(h["frozen"] for h in late),
        "alpha_in_band_late": 0.3 <= late_alpha <= 0.9,
        "p_two_sided": late[-1]["p_frac_pos"] >= 0.01
                       and late[-1]["p_frac_neg"] >= 0.01,
    }
    summary = {
        "n_envs": n_envs, "iterations": iterations, "seed": seed,
        "value_loss_self_median_early_vs_late": [round(early_vl, 4),
                                                 round(late_vl, 4)],
        "reward_trend_first_last": [history[0]["reward_mean"],
                                    history[-1]["reward_mean"]],
        "final": history[-1], "lambda_max": lam_max,
        "checks": checks, "pass": all(checks.values()),
    }
    if out:
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="local kinematic full-chain smoke (CPU)")
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--iterations", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/analysis/c5_lagrangian_smoke")
    args = ap.parse_args()
    if not args.smoke:
        ap.error("only --smoke is wired locally; server training goes "
                 "through the A-line launcher (see STATUS_C @A7)")
    summary = run_smoke(args.n_envs, args.iterations, args.seed, args.out)
    print(json.dumps({k: v for k, v in summary.items() if k != "final"},
                     indent=2), flush=True)
    print("SMOKE " + ("PASS" if summary["pass"] else "FAIL"), flush=True)
    raise SystemExit(0 if summary["pass"] else 1)


if __name__ == "__main__":
    main()
