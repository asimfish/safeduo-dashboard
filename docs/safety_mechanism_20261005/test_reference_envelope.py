"""CPU contracts: the reference change stays feasible and removes target windup."""
import unittest
import torch
from reference_envelope import reference_bounds


class ReferenceContracts(unittest.TestCase):
    def test_integrated_target_windup_is_bounded(self):
        q = torch.zeros(1)
        target = q.clone()
        for _ in range(20):
            lo, hi = reference_bounds(q, target, q-2, q+2, .025, .050)
            target += torch.full_like(q, .025).maximum(lo).minimum(hi)
        self.assertAlmostEqual(target.item(), .050, places=6)
        # Red-capable old integration accumulates .500 for the same inputs.
        self.assertGreater(20 * .025, target.item() + .4)

    def test_outside_envelope_recovers_without_target_snap(self):
        q = torch.zeros(1)
        target = torch.full_like(q, .20)
        updates = []
        for _ in range(8):
            lo, hi = reference_bounds(q, target, q-2, q+2, .025, .050)
            step = torch.zeros_like(q).maximum(lo).minimum(hi)
            updates.append(step.item())
            target += step
        self.assertLessEqual(max(abs(x) for x in updates), .02500001)
        self.assertAlmostEqual(target.item(), .050, places=6)

    def test_random_limits_and_reachability_match_scalar_oracle(self):
        g = torch.Generator().manual_seed(491701)
        lower = torch.randn(500, generator=g) - 2
        upper = lower + .1 + torch.rand(500, generator=g)*5
        target = lower + torch.rand(500, generator=g)*(upper-lower)
        q = target + torch.randn(500, generator=g)*.3
        lo, hi = reference_bounds(q, target, lower, upper, .025, .05)
        for i in range(500):
            reachable = (max(lower[i].item(), target[i].item()-.025),
                         min(upper[i].item(), target[i].item()+.025))
            desired = (q[i].item()-.05, q[i].item()+.05)
            closest = tuple(min(max(x, reachable[0]), reachable[1]) for x in desired)
            self.assertAlmostEqual((target+lo)[i].item(), closest[0], places=5)
            self.assertAlmostEqual((target+hi)[i].item(), closest[1], places=5)
            self.assertLessEqual(lo[i].item(), hi[i].item())
        self.assertTrue(((target+lo)>=lower-1e-6).all())
        self.assertTrue(((target+hi)<=upper+1e-6).all())

    def test_invalid_state_is_rejected(self):
        with self.assertRaises(ValueError):
            reference_bounds(torch.zeros(1), torch.tensor([float('nan')]),
                             torch.tensor([-1.]), torch.tensor([1.]), .025, .05)
        with self.assertRaises(ValueError):
            reference_bounds(torch.zeros(1), torch.tensor([2.]),
                             torch.tensor([-1.]), torch.tensor([1.]), .025, .05)


if __name__ == '__main__':
    unittest.main()
