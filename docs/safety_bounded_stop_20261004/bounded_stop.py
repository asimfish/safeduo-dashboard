"""Evaluation-only bounded approach to a once-sampled measured pose."""
import math

import torch


class BoundedStopLatch:
    def __init__(self, trigger_steps, increment_limit, lower, upper):
        if not math.isfinite(increment_limit) or increment_limit <= 0:
            raise ValueError('positive finite increment limit required')
        self.trigger_steps = torch.as_tensor(trigger_steps, dtype=torch.long)
        if self.trigger_steps.ndim != 1 or (self.trigger_steps < -1).any():
            raise ValueError('one trigger per environment; -1 means no intervention')
        if lower.shape != upper.shape or lower.ndim != 2 or lower.shape[0] != len(self.trigger_steps):
            raise ValueError('matching (environments, controlled joints) limits required')
        if not torch.isfinite(lower).all() or not torch.isfinite(upper).all() or (lower >= upper).any():
            raise ValueError('finite ordered position limits required')
        self.increment_limit = increment_limit
        self.lower, self.upper = lower.clone(), upper.clone()
        self.fixed = None
        self.active = None

    def update(self, step, prior_target, measured_q, proposed_target):
        if proposed_target.shape != self.lower.shape or prior_target.shape != self.lower.shape or measured_q.shape != self.lower.shape:
            raise ValueError('targets and measured q must match limit shape')
        if self.fixed is None:
            self.fixed = torch.zeros_like(proposed_target)
            self.active = torch.zeros(len(self.trigger_steps), device=proposed_target.device, dtype=torch.bool)
            self.trigger_steps = self.trigger_steps.to(proposed_target.device)
            self.lower, self.upper = self.lower.to(proposed_target), self.upper.to(proposed_target)
        newly = (self.trigger_steps >= 0) & (step >= self.trigger_steps) & ~self.active
        self.fixed[newly] = torch.clamp(measured_q[newly], self.lower[newly], self.upper[newly])
        self.active |= newly
        delta = (self.fixed - prior_target).clamp(-self.increment_limit, self.increment_limit)
        bounded = torch.clamp(prior_target + delta, self.lower, self.upper)
        return torch.where(self.active[:, None], bounded, proposed_target)
