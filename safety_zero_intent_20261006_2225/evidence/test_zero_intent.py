"""CPU target-reference contract tests; no physical-safety assertion."""
import unittest
import torch
from reference_envelope import reference_bounds, install
from zero_intent import reference_gap

class ReferenceContract(unittest.TestCase):
    def bounds(self,q,target,lower=None,upper=None,**kw):
        return reference_bounds(q,target,torch.full_like(q,-1) if lower is None else lower,
                                torch.full_like(q,1) if upper is None else upper,.025,.010,**kw)
    def test_old_forced_zero_counterexample_and_candidate_holds(self):
        q=torch.tensor([[.03,-.03]],dtype=torch.float32);target=torch.zeros_like(q)
        old=self.bounds(q,target);new=self.bounds(q,target,zero_inclusive=True)
        raw=torch.zeros_like(q)
        self.assertFalse(torch.equal(raw.maximum(old[0]).minimum(old[1]),raw))
        self.assertTrue(torch.equal(raw.maximum(new[0]).minimum(new[1]),raw))
    def test_original_interval_is_preserved_when_flag_false(self):
        q=torch.tensor([[.2,-.2,0]],dtype=torch.float32);target=torch.zeros_like(q)
        a=self.bounds(q,target);b=self.bounds(q,target,zero_inclusive=False)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,b)))
    def test_candidate_contains_original_envelope_inside_original_limits(self):
        gen=torch.Generator().manual_seed(135901)
        q=torch.rand((512,26),generator=gen)*4-2;target=torch.rand((512,26),generator=gen)*2-1
        old=self.bounds(q,target);new=self.bounds(q,target,zero_inclusive=True)
        box=reference_bounds(q,target,torch.full_like(q,-1),torch.full_like(q,1),.025)
        self.assertTrue(bool((new[0]<=0).all() and (new[1]>=0).all()))
        self.assertTrue(bool((new[0]<=old[0]).all() and (new[1]>=old[1]).all()))
        self.assertTrue(bool((new[0]>=box[0]).all() and (new[1]<=box[1]).all()))
    def test_target_exactly_on_limits_allows_holding(self):
        q=torch.tensor([[1.2,-1.2]]);target=torch.tensor([[1.,-1.]])
        lo,hi=self.bounds(q,target,zero_inclusive=True)
        self.assertTrue(bool((lo<=0).all() and (hi>=0).all()))
        self.assertEqual(float(hi[0,0]),0);self.assertEqual(float(lo[0,1]),0)
    def test_outside_original_softlimit_rejects_hold(self):
        with self.assertRaisesRegex(ValueError,'holding target'):
            self.bounds(torch.zeros((1,1)),torch.tensor([[1.01]]),zero_inclusive=True)
    def test_nonfinite_inputs_rejected(self):
        for value in [float('nan'),float('inf')]:
            with self.assertRaises(ValueError):self.bounds(torch.tensor([[value]]),torch.zeros((1,1)),zero_inclusive=True)
    def test_shape_and_gap_failure(self):
        with self.assertRaises(ValueError):self.bounds(torch.zeros((2,1)),torch.zeros((1,1)),zero_inclusive=True)
        with self.assertRaises(ValueError):reference_bounds(torch.zeros((1,1)),torch.zeros((1,1)),torch.full((1,1),-1),torch.ones((1,1)),.025,0,zero_inclusive=True)
    def test_registered_modes_have_fixed_gap(self):
        self.assertEqual(reference_gap('joint_reference'),.050)
        self.assertEqual(reference_gap('tight_reference'),.010)
        self.assertEqual(reference_gap('zero_inclusive'),.010)
        with self.assertRaises(ValueError):reference_gap('delay_reserve')

if __name__=='__main__':unittest.main(verbosity=2)
