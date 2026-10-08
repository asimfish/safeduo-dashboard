"""R17 S8: clutch execution semantics (eval-side driver wrapper, zero drift).

Owner product correction (2026-08-20): the s0 model is a SAFETY FILTER for
teleop data collection -- when safe the output must equal the input bit-for-
bit (no continuous attenuation), when unsafe the at-risk arm hand-brakes
(freezes), and the user unlocks arms one at a time to recover. The learned
alpha head therefore becomes a LOOKAHEAD DANGER DETECTOR whose continuous
value is binarized per arm by a hysteresis state machine:

  ENGAGED (alpha_exec = 1, bitwise passthrough)
      -> LOCKED   when alpha_policy <  theta_lo
  LOCKED  (alpha_exec = 0, per-arm delta zeroed = frozen; the env's
           persistent-target integration makes PD converge to a fixed point)
      -> ENGAGED  when alpha_policy >  theta_hi

Boundary semantics are pinned by tests: alpha == theta_lo stays ENGAGED
(transition needs strict <), alpha == theta_hi stays LOCKED (needs strict >).
The transition uses the CURRENT step's alpha_policy and alpha_exec reflects
the post-transition state in the same step (detector fires -> the very same
control step freezes).

The p arbitration head passes through untouched (recovery-order arbitration
downstream). The wrapper lives entirely on the eval/execution side: training
code paths are not imported, not modified, and `wrap_clutch(enabled=False)`
returns the inner driver object itself (bit-identical default, pinned by
tests/test_clutch.py).

Manual unlock (scripted "user" support for the R17 recovery metric): a held
per-(env, arm) `hold_engaged` latch forces ENGAGED regardless of
alpha_policy -- the analytic safety stack (tube/damper/backstop) below still
bottoms out, matching the product semantics where an explicit user unlock
takes responsibility for the arm while the rule stack stays armed.
"""

from __future__ import annotations

import torch

N_ARMS = 4


class ClutchDriver:
    """Hysteresis binarizer around any block1-style driver (act/reset)."""

    def __init__(self, inner, num_envs: int, theta_hi: float = 0.7,
                 theta_lo: float = 0.4, device: "str | torch.device" = "cpu"):
        if not (0.0 <= theta_lo < theta_hi <= 1.0):
            raise ValueError(
                f"need 0 <= theta_lo < theta_hi <= 1, got "
                f"theta_lo={theta_lo} theta_hi={theta_hi}")
        self.inner = inner
        self.theta_hi = float(theta_hi)
        self.theta_lo = float(theta_lo)
        self.device = torch.device(device)
        self.engaged = torch.ones(num_envs, N_ARMS, dtype=torch.bool,
                                  device=self.device)
        self.hold_engaged = torch.zeros(num_envs, N_ARMS, dtype=torch.bool,
                                        device=self.device)
        # diagnostics for the eval tracer (state that produced the last action)
        self.last_alpha_policy: "torch.Tensor | None" = None
        self.last_p: "torch.Tensor | None" = None

    def reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel():
            self.engaged[env_ids] = True
            self.hold_engaged[env_ids] = False
        self.inner.reset(env_ids)

    def set_hold(self, env_ids: torch.Tensor, arm_idx: int,
                 hold: bool) -> None:
        """Scripted user unlock: while held, (env, arm) is forced ENGAGED."""
        ids = torch.as_tensor(env_ids, dtype=torch.long,
                              device=self.hold_engaged.device)
        self.hold_engaged[ids, int(arm_idx)] = bool(hold)

    def act(self, env, obs_dict: dict) -> torch.Tensor:
        a = self.inner.act(env, obs_dict)
        alpha = (a[:, :N_ARMS].clamp(-1.0, 1.0) + 1.0) * 0.5
        p = a[:, N_ARMS:N_ARMS + 1]
        # hysteresis on the current detector readout; strict comparisons pin
        # the boundary semantics (== theta_lo stays, == theta_hi stays)
        self.engaged = torch.where(self.engaged,
                                   alpha >= self.theta_lo,
                                   alpha > self.theta_hi)
        self.engaged = self.engaged | self.hold_engaged
        self.last_alpha_policy = alpha
        self.last_p = p.squeeze(-1)
        alpha_exec = self.engaged.to(alpha.dtype)
        return torch.cat([alpha_exec * 2.0 - 1.0, p], dim=-1)


def wrap_clutch(driver, enabled: bool, theta_hi: float, theta_lo: float,
                num_envs: int, device: "str | torch.device" = "cpu"):
    """CLI wiring helper: enabled=False returns `driver` ITSELF (the default
    eval path stays bit-identical -- no wrapper object, no state)."""
    if not enabled:
        return driver
    return ClutchDriver(driver, num_envs, theta_hi=theta_hi,
                        theta_lo=theta_lo, device=device)
