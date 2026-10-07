"""PPO-Lagrangian components for the coordinator (credit-assignment fix #3).

PROVENANCE: ported from the double_hand workspace on bjxy_5090 NAS
(/mnt/nas/data/lyf/double_hand, pure-torch RL stack, 137 tests green):
  - algorithms/constraints.py  -> PIDLagrangian / DualGradientLagrangian /
                                  combine_advantages
  - algorithms/targets.py      -> validity_aware_gae
Port adaptations for the SafeDuo contract (documented per class):
  - n-channel controllers (double_hand hardcoded 2 channels [robot, table];
    SafeDuo default 3 channels [cross, self, table] fed by
    algo/reward_shaping.margin_cost_by_class episode means)
  - lightweight local finite/shape checks instead of the full numerics module
  - update semantics, clamping, integral/derivative handling kept EXACTLY
    (unit tests pin the original double_hand numbers)

Why Lagrangian (STATUS_SERVER 08-12 03:20 finding): with a fixed penalty
weight the 600-step discounted return barely feels terminal violations
(0.99^600 ~= 0.002) and sweeping the weight is fragile; a constrained
formulation regulates the *observed episode cost* against an explicit limit
and auto-tunes the multiplier.

Recommended starting config (registered as planned runs in experiments.csv):
  channels     = episode-mean of margin_cost_by_class (cross/self/table)
  cost_limits  = [0.05, 0.05, 0.05]   (worst-pair proximity cost per step)
  PID gains    = kp 0.5, ki 0.05, kd 0.1, integral_limit 20
  update cadence = once per PPO iteration on the rollout-buffer episode means
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


def _finite(name: str, t: Tensor) -> None:
    if not bool(torch.isfinite(t).all().item()):
        raise ValueError(f"{name} contains NaN or Inf")


def _check_costs(observed: Tensor, limits: Tensor, n: int) -> None:
    if observed.shape != (n,) or limits.shape != (n,):
        raise ValueError(f"cost controllers require ({n},) tensors")
    _finite("observed_costs", observed)
    _finite("cost_limits", limits)
    if bool((observed < 0).any().item()) or bool((limits < 0).any().item()):
        raise ValueError("costs and limits must be non-negative")


class PIDLagrangian(nn.Module):
    """N-channel PID multiplier controller (double_hand PIDLagrangeController).

    multipliers = clamp_min(kp*e + ki*integral(e) + kd*d(e), 0) with
    e = observed_costs - cost_limits; derivative is zero on the first update;
    integral optionally clamped to +-integral_limit. State lives in buffers so
    it rides checkpoints.
    """

    def __init__(self, n_channels: int = 3, *, kp: float = 0.5, ki: float = 0.05,
                 kd: float = 0.1, initial_value: float = 0.0,
                 integral_limit: "float | None" = 20.0,
                 lambda_max: "tuple | None" = None,
                 lambda_rate_max: float = float("inf")):
        super().__init__()
        if any(v < 0.0 for v in (kp, ki, kd)):
            raise ValueError("PID gains must be non-negative")
        if initial_value < 0.0:
            raise ValueError("initial_value must be non-negative")
        if integral_limit is not None and integral_limit <= 0.0:
            raise ValueError("integral_limit must be positive when provided")
        if lambda_rate_max <= 0.0:
            raise ValueError("lambda_rate_max must be positive (inf = off)")
        self.n_channels = n_channels
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral_limit = integral_limit
        # R15 ③b: per-iteration slew limit on the multiplier output,
        # |lambda_t - lambda_{t-1}| <= lambda_rate_max (all channels). v6.2
        # evidence: lambda_cross jumped 0 -> cap 0.35 within a few updates at
        # curriculum stage transitions, repricing the whole batch at once and
        # yanking alpha down before the policy could adapt. Default inf is an
        # exact no-op (the rate branches are skipped entirely). The rate
        # limit applies BEFORE the lambda_max cap; while the upward rate
        # bound binds, anti-windup is conditional integration (integral
        # frozen), so lambda rides the exact rate ramp AND releases within
        # ~lambda/rate updates -- see update() for why back-calculation is
        # the wrong tool against a moving bound.
        self.lambda_rate_max = float(lambda_rate_max)
        self.register_buffer("_multipliers", torch.full((n_channels,), initial_value))
        self.register_buffer("_integral", torch.zeros(n_channels))
        self.register_buffer("_previous_error", torch.zeros(n_channels))
        self.register_buffer("_has_previous", torch.tensor(False))
        # R13 (G3 RCA 2026-08-18): optional per-channel multiplier cap with
        # anti-windup. v6.1 evidence: lambda_cross saturated near 1.0 (integral
        # pinned at +20 from early training) while the late-window break-even
        # shadow price was only ~0.343 -- a ~2.94x over-pricing that made
        # "move less" the dominant constrained-objective direction and drove
        # the inversion collapse. A cap bounds the shadow price; conditional
        # integration stops the integral from winding further while capped.
        # None entries (encoded as +inf) leave a channel uncapped; the default
        # lambda_max=None is an exact no-op for every existing caller.
        if lambda_max is not None:
            lm = torch.tensor([float("inf") if v is None or v <= 0.0 else float(v)
                               for v in lambda_max])
            if lm.shape != (n_channels,):
                raise ValueError(f"lambda_max must have {n_channels} entries")
            self.register_buffer("_lambda_max", lm, persistent=False)
        else:
            self._lambda_max = None

    @property
    def multipliers(self) -> Tensor:
        return self._multipliers.detach().clone()

    @torch.no_grad()
    def update(self, observed_costs: Tensor, cost_limits: Tensor) -> Tensor:
        _check_costs(observed_costs, cost_limits, self.n_channels)
        error = observed_costs - cost_limits
        integral = self._integral + error
        if self.integral_limit is not None:
            integral = integral.clamp(-self.integral_limit, self.integral_limit)
        derivative = torch.where(self._has_previous,
                                 error - self._previous_error,
                                 torch.zeros_like(error))
        update = self.kp * error + self.ki * integral + self.kd * derivative
        rate_on = self.lambda_rate_max != float("inf")
        if rate_on:
            # R15 ③b slew limit, applied BEFORE the lambda_max cap. Anti-
            # windup here is CONDITIONAL INTEGRATION, not back-calculation:
            # while the upward rate bound binds and the error is positive,
            # the integral does not accumulate this step (back-calculation
            # against a MOVING bound would crush the integral so deep on the
            # first clamped step that lambda afterwards crawls at ki*e per
            # update instead of riding the rate ramp). The frozen integral
            # keeps the release fast: once the constraint is satisfied the
            # raw update drops immediately and lambda walks down at <= rate
            # per update (downward clamp below, no integral inflation).
            hi = self._multipliers + self.lambda_rate_max
            if self.ki > 0.0:
                integral = torch.where((update > hi) & (error > 0.0),
                                       integral - error, integral)
            update = torch.minimum(update, hi)
        if self._lambda_max is not None:
            # back-calculation anti-windup: where the raw update exceeds the
            # cap, shrink the integral so the recomputed output sits exactly
            # AT the cap (bounded shadow price while saturated) instead of
            # letting it wind toward integral_limit -- so the multiplier
            # releases within a few updates once the constraint is satisfied.
            # (uncapped channels have cap=+inf -> excess clamps to exactly 0)
            # With the slew limit on, `update` is already rate-clamped, so
            # during the ramp (update < cap) the cap block is a no-op and the
            # steady state at the cap is bit-identical to the pre-R15 cap.
            excess = (update - self._lambda_max).clamp_min(0.0)
            if self.ki > 0.0:
                integral = integral - excess / self.ki
            update = torch.minimum(update, self._lambda_max)
        if rate_on:
            update = torch.maximum(
                update, self._multipliers - self.lambda_rate_max)
        _finite("constraint_multipliers", update)
        self._multipliers.copy_(update.clamp_min(0.0))
        self._integral.copy_(integral)
        self._previous_error.copy_(error)
        self._has_previous.fill_(True)
        return self.multipliers


class DualGradientLagrangian(nn.Module):
    """Projected dual ascent baseline (double_hand DualGradientController)."""

    def __init__(self, n_channels: int = 3, *, learning_rate: float,
                 initial_value: float = 0.0):
        super().__init__()
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if initial_value < 0.0:
            raise ValueError("initial_value must be non-negative")
        self.n_channels = n_channels
        self.learning_rate = learning_rate
        self.register_buffer("_multipliers", torch.full((n_channels,), initial_value))

    @property
    def multipliers(self) -> Tensor:
        return self._multipliers.detach().clone()

    @torch.no_grad()
    def update(self, observed_costs: Tensor, cost_limits: Tensor) -> Tensor:
        _check_costs(observed_costs, cost_limits, self.n_channels)
        candidate = self._multipliers + self.learning_rate * (observed_costs - cost_limits)
        _finite("constraint_multipliers", candidate)
        self._multipliers.copy_(candidate.clamp_min(0.0))
        return self.multipliers


def combine_advantages(reward_advantage: Tensor, cost_advantages: Tensor,
                       multipliers: Tensor) -> Tensor:
    """Constrained objective: A_reward - sum_c lambda_c * A_cost_c.

    (double_hand combine_policy_advantages generalized to (B, C) cost streams;
    multipliers are detached -- gradients never flow into the controller.)
    """
    if cost_advantages.ndim != 2 or cost_advantages.shape[0] != reward_advantage.shape[0]:
        raise ValueError("cost_advantages must have shape [B, C]")
    if multipliers.shape != (cost_advantages.shape[1],):
        raise ValueError("multipliers must have shape [C]")
    _finite("reward_advantage", reward_advantage)
    _finite("cost_advantages", cost_advantages)
    if bool((multipliers < 0).any().item()):
        raise ValueError("multipliers must be non-negative")
    return reward_advantage - (cost_advantages * multipliers.detach()).sum(dim=-1)


@dataclass(frozen=True)
class AdvantageTargets:
    advantages: Tensor
    returns: Tensor


@torch.no_grad()
def validity_aware_gae(rewards: Tensor, values: Tensor, next_values: Tensor,
                       terminated: Tensor, truncated: Tensor, *,
                       learning_sample_valid: Tensor,
                       gamma: float, gae_lambda: float) -> AdvantageTargets:
    """GAE with explicit episode-end semantics (double_hand targets.py, exact).

    True terminations disable the value bootstrap. Time-limit truncations keep
    the one-step bootstrap from their terminal observation but break the
    recursive trace so it cannot leak into the following episode. Invalid rows
    produce zero advantages/returns and break the backward trace.

    All tensors (T, N); terminated/truncated/learning_sample_valid bool.
    SafeDuo note: duo_env terminates on VIOLATION and truncates on time-out,
    exactly the two channels this estimator distinguishes; use it for the
    reward stream and each cost stream alike.
    """
    if rewards.ndim != 2 or rewards.shape[0] == 0 or rewards.shape[1] == 0:
        raise ValueError("GAE tensors must have shape [T, N] with T,N >= 1")
    for name, t in (("values", values), ("next_values", next_values),
                    ("terminated", terminated), ("truncated", truncated),
                    ("learning_sample_valid", learning_sample_valid)):
        if t.shape != rewards.shape:
            raise ValueError(f"{name} must have shape {tuple(rewards.shape)}")
    for name, t in (("terminated", terminated), ("truncated", truncated),
                    ("learning_sample_valid", learning_sample_valid)):
        if t.dtype != torch.bool:
            raise ValueError(f"{name} must use bool")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must lie in [0, 1]")
    _finite("rewards", rewards)
    _finite("values", values)
    _finite("next_values", next_values)

    zero = torch.zeros_like(rewards)
    safe_rewards = torch.where(learning_sample_valid, rewards, zero)
    safe_values = torch.where(learning_sample_valid, values, zero)
    safe_next_values = torch.where(learning_sample_valid, next_values, zero)
    bootstrap_mask = (learning_sample_valid & ~terminated).to(rewards.dtype)
    trace_continues = learning_sample_valid & ~(terminated | truncated)
    raw_deltas = safe_rewards + gamma * bootstrap_mask * safe_next_values - safe_values

    advantages = torch.empty_like(rewards)
    running = torch.zeros_like(rewards[-1])
    for t in range(rewards.shape[0] - 1, -1, -1):
        safe_continuation = torch.where(trace_continues[t], running,
                                        torch.zeros_like(running))
        candidate = raw_deltas[t] + gamma * gae_lambda * safe_continuation
        running = torch.where(learning_sample_valid[t], candidate,
                              torch.zeros_like(candidate))
        advantages[t] = running
    returns = torch.where(learning_sample_valid, advantages + safe_values,
                          torch.zeros_like(advantages))
    _finite("advantages", advantages)
    return AdvantageTargets(advantages=advantages, returns=returns)
