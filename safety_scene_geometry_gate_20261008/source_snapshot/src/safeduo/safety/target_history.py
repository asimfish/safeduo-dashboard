"""Recent issued targets that may remain pending under bounded delivery lag."""
from collections import deque


class PendingTargetHistory:
    def __init__(self,steps):
        if steps<1 or int(steps)!=steps:
            raise ValueError('pending target history requires positive integer steps')
        self.steps=int(steps)
        self.targets=deque(maxlen=self.steps)

    @staticmethod
    def clone(target):
        return {a:q.detach().clone() for a,q in target.items()}

    def reset(self,target,env_ids=None):
        if not self.targets or env_ids is None:
            self.targets=deque((self.clone(target) for _ in range(self.steps)),maxlen=self.steps)
        else:
            for past in self.targets:
                for a,q in target.items(): past[a][env_ids]=q[env_ids]

    def append(self,target):
        self.targets.append(self.clone(target))
