import unittest
import torch
from multirow_response_v3 import bounded_projection


class MultirowOracles(unittest.TestCase):
    def test_feasible_intersection_and_box(self):
        A=torch.tensor([[[1.,0.],[0.,1.],[-1.,-1.]]],dtype=torch.float64)
        b=torch.tensor([[.1,.2,-.5]],dtype=torch.float64)
        delta,info=bounded_projection(A,b,torch.full((1,2),-.3),torch.full((1,2),.3),40)
        self.assertLessEqual(info['residual'].max().item(),1e-10)
        self.assertAlmostEqual(delta[0,0].item(),.1,places=12)
        self.assertAlmostEqual(delta[0,1].item(),.2,places=12)

    def test_contradictory_rows_remain_failed(self):
        delta,info=bounded_projection(torch.tensor([[[1.],[-1.]]]),torch.tensor([[1.,1.]]),
            torch.tensor([[-.05]]),torch.tensor([[.05]]),40)
        self.assertGreater(info['residual'].max().item(),.9)
        self.assertLessEqual(abs(delta[0,0].item()),.050001)

    def test_impossible_row_does_not_hide_feasible_row(self):
        A=torch.tensor([[[0.,0.],[1.,0.]]],dtype=torch.float64)
        b=torch.tensor([[1.,.02]],dtype=torch.float64)
        delta,info=bounded_projection(A,b,torch.full((1,2),-.05),torch.full((1,2),.05),40)
        self.assertTrue(info['missing_direction'][0])
        self.assertTrue(info['impossible_in_box'][0,0])
        self.assertGreater(info['residual'][0,0].item(),.99)
        self.assertAlmostEqual(delta[0,0].item(),.02,places=12)

    def test_uncontrolled_constraint_is_unknown(self):
        _,info=bounded_projection(torch.zeros(1,1,2),torch.ones(1,1),
            torch.full((1,2),-.05),torch.full((1,2),.05),40)
        self.assertTrue(info['missing_direction'][0])


if __name__=='__main__':unittest.main()
