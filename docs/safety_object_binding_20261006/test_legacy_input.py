"""A discriminating baseline: extra object state cannot repair fixed replay."""
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from safeduo.delta.skill_replay import SkillTrajectory,SkillReplayDelta,SkillNoiseParams

class LegacyObjectBinding(unittest.TestCase):
    def test_sealed_replay_is_object_invariant(self):
        traj=SkillTrajectory.load(Path('/home/liyufeng/safeduo/artifacts/forensics/20261004_randomized_safety_campaign_v4/block_00/task_two_pair_pdz_uax07.npz'))
        source=SkillReplayDelta(1,[traj],params=SkillNoiseParams.tier(0),auto_advance=False,device='cpu')
        source.reset(torch.tensor([0]));source.set_time(4.1)
        original=source.sample(SimpleNamespace(dt=.016666,objects=torch.tensor([[[.53,0,.84],[-.4,0,.84]]])))
        perturbed=source.sample(SimpleNamespace(dt=.016666,objects=torch.tensor([[[.55,.02,.84],[-.42,-.02,.84]]])))
        self.assertTrue(any(bool(v.abs().any()) for v in original.delta_q.values()))
        for arm in original.delta_q:self.assertTrue(torch.equal(original.delta_q[arm],perturbed.delta_q[arm]))

if __name__=='__main__':unittest.main()
