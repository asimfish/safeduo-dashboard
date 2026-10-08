"""Evaluation-only, matched initial poses and post-safety actuator latency."""
from __future__ import annotations

from collections import deque

import torch

from safeduo.safety.types import ARM_KEYS


class InitialPoseSource:
    """Independent uniform joint jitter, with explicit soft-limit clipping audit.

    No collision rejection is performed. Initial overlap is measured separately
    by the battery and must not enter collision-prevention success claims.
    """
    def __init__(self, source, baseline, limits, jitter_rad, seed):
        if jitter_rad < 0:
            raise ValueError('initial jitter must be nonnegative')
        self.source=source
        self.seed=int(seed)
        self.jitter_rad=float(jitter_rad)
        self.positions={}
        self.requested={}
        self.baseline={a:q.detach().clone() for a,q in baseline.items()}
        gen=torch.Generator(device='cpu').manual_seed(self.seed)
        self.clipped=[]
        for arm in ARM_KEYS:
            base=self.baseline[arm]
            jitter=(torch.rand(base.shape,generator=gen)*2-1)*self.jitter_rad
            self.requested[arm]=jitter.to(base.device)
            requested=base+self.requested[arm]
            position=requested.clamp(limits[arm][...,0],limits[arm][...,1])
            self.positions[arm]=position
            self.clipped.append((position!=requested).sum(-1))

    def reset(self,env_ids,generator=None):
        self.source.reset(env_ids,generator)

    def sample(self,state):
        return self.source.sample(state)

    def initial_positions(self,env_ids):
        return env_ids,{a:q[env_ids] for a,q in self.positions.items()}

    def coverage_metadata(self):
        requested=torch.cat(list(self.requested.values()),-1)
        effective=torch.cat([self.positions[a]-self.baseline[a] for a in ARM_KEYS],-1)
        clipped=torch.stack(self.clipped,-1).sum(-1).cpu().tolist()
        max_abs=effective.abs().amax(-1).cpu().tolist()
        requested_max=requested.abs().amax(-1).cpu().tolist()
        rms=effective.square().mean(-1).sqrt().cpu().tolist()
        return [dict(initial_seed=self.seed,initial_jitter_rad=self.jitter_rad,
                     initial_clipped_joints=c,initial_effective_max_abs_rad=m,
                     initial_requested_max_abs_rad=q,initial_effective_rms_rad=s)
                for c,m,q,s in zip(clipped,max_abs,requested_max,rms)]


class TargetDelayQueue:
    """Hold the initial target for k steps, then emit target[t-k] exactly."""
    def __init__(self,steps):
        if steps < 0 or int(steps)!=steps:
            raise ValueError('target delay must be a nonnegative integer')
        self.steps=int(steps)
        self.pending=deque()

    @staticmethod
    def clone(target):
        return {a:q.detach().clone() for a,q in target.items()}

    def reset(self,initial):
        self.pending=deque(self.clone(initial) for _ in range(self.steps))

    def push(self,target):
        self.pending.append(self.clone(target))
        return self.pending.popleft()


class TargetDelayAdapter:
    """Inject latency after safety projection, once per control step.

    The controller target integrator and measurements remain current. Only
    the target sent to PD actuators is delayed; this is not sensor latency or
    a command-source delay. Repeat physics substeps reuse the same target.
    """
    def __init__(self,env,steps):
        self.env=env
        self.queue=TargetDelayQueue(steps)
        self.original_pre=env._pre_physics_step
        self.original_apply=env._apply_action
        self.applied=None
        if steps:
            env._pre_physics_step=self.pre
            env._apply_action=self.apply

    def reset(self,initial):
        self.queue.reset(initial)
        self.applied=self.queue.clone(initial)

    def pre(self,actions):
        self.original_pre(actions)
        self.applied=self.queue.push(self.env._targets)

    def apply(self):
        if self.applied is None:
            raise RuntimeError('actuator delay must be reset before stepping')
        for arm in ARM_KEYS:
            self.env._arms[arm].set_joint_position_target(
                self.applied[arm],joint_ids=self.env._joint_idx[arm].tolist())

    def close(self):
        if self.queue.steps:
            self.env._pre_physics_step=self.original_pre
            self.env._apply_action=self.original_apply
