"""Evaluation-only one-shot fixed targets; no queue access or robot interface."""
import torch


class StopLatch:
    def __init__(self, trigger_steps, mode):
        if mode not in ('stored', 'measured'):
            raise ValueError('mode must be stored or measured')
        self.trigger_steps = torch.as_tensor(trigger_steps, dtype=torch.long)
        if self.trigger_steps.ndim != 1 or (self.trigger_steps < -1).any():
            raise ValueError('one trigger per environment; -1 means no intervention')
        self.mode = mode
        self.fixed = None
        self.active = None

    def update(self, step, prior_target, measured_q, proposed_target):
        if self.fixed is None:
            self.fixed = torch.zeros_like(proposed_target)
            self.active = torch.zeros(len(self.trigger_steps), device=proposed_target.device, dtype=torch.bool)
            self.trigger_steps = self.trigger_steps.to(proposed_target.device)
        if proposed_target.shape != self.fixed.shape or proposed_target.ndim != 2:
            raise ValueError('target shape must remain (environments, controlled joints)')
        if prior_target.shape != proposed_target.shape or measured_q.shape != proposed_target.shape:
            raise ValueError('matching prior target and measured q required')
        if proposed_target.shape[0] != len(self.trigger_steps):
            raise ValueError('trigger count differs from environment count')
        newly = (self.trigger_steps >= 0) & (step >= self.trigger_steps) & ~self.active
        source = prior_target if self.mode == 'stored' else measured_q
        self.fixed[newly] = source[newly]
        self.active |= newly
        return torch.where(self.active[:, None], self.fixed, proposed_target)
