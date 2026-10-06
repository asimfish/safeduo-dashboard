"""Reachable target contract, frozen FIFO and nominal-risk boundary checks."""
import unittest
import torch

from tracking_reserve import reserve_forecast, gap_from_margin, ARM_KEYS
from reference_envelope import reference_bounds


class TrackingTests(unittest.TestCase):
    def inputs(self):
        q={a:torch.zeros(3,d) for a,d in zip(ARM_KEYS,[7,7,6,6])}
        j={'F':torch.zeros(3,2,14),'U':torch.zeros(3,2,12)}
        j['F'][:,:,0]=1
        return dict(distance=torch.full((3,2),.080),dmin=torch.full((3,2),.005),
                    exempt=torch.zeros(3,2,dtype=torch.bool),jacobian=j,q=q,
                    qd={a:v.clone() for a,v in q.items()},pending=[{a:v.clone() for a,v in q.items()} for _ in range(6)],dt=.016666)

    def test_baseline_and_fixed_exact(self):
        m=torch.tensor([-.2,0.,.03,.1])
        torch.testing.assert_close(gap_from_margin(m,'joint_reference'),torch.full_like(m,.05),rtol=0,atol=0)
        torch.testing.assert_close(gap_from_margin(m,'tight_reference'),torch.full_like(m,.01),rtol=0,atol=0)

    def test_gap_monotone_bounded(self):
        m=torch.linspace(-2,2,10001);g=gap_from_margin(m,'delay_reserve')
        self.assertTrue((g[1:]>=g[:-1]).all());self.assertTrue((g>=.01).all() and (g<=.05).all())

    def test_velocity_with_zero_target_debt(self):
        k=self.inputs();k['qd']['F_L'][:,0]=-1
        z=reserve_forecast(**k)
        torch.testing.assert_close(z['risk_margin'],torch.full((3,),.075-18*.016666),rtol=0,atol=2e-8)

    def test_actual_last_pending_can_trigger(self):
        k=self.inputs();k['pending'][-1]['F_L'][:,0]=-.1
        z=reserve_forecast(**k);torch.testing.assert_close(z['risk_margin'],torch.full((3,),-.025),rtol=0,atol=1e-8)

    def test_original_exemption(self):
        k=self.inputs();k['distance'][:,0]=-.4;k['exempt'][:,0]=True
        z=reserve_forecast(**k);torch.testing.assert_close(z['risk_margin'],torch.full((3,),.075),rtol=0,atol=1e-8)

    def test_no_pending_mutation(self):
        k=self.inputs();saved=[{a:v.clone() for a,v in t.items()} for t in k['pending']]
        reserve_forecast(**k)
        for actual,old in zip(k['pending'],saved):
            for a in ARM_KEYS:self.assertTrue(torch.equal(actual[a],old[a]))

    def test_unreachable_tight_gap_stays_in_original_box(self):
        q=torch.tensor([[0.,0.]]);target=torch.tensor([[1.,-1.]])
        lo,hi=reference_bounds(q,target,torch.full_like(q,-2),torch.full_like(q,2),.016,.01)
        torch.testing.assert_close(lo,torch.tensor([[-.016,.016]]),rtol=0,atol=0)
        self.assertTrue(torch.equal(lo,hi))

    def test_tensor_gaps_per_environment(self):
        q=torch.zeros(2,7);gap=torch.tensor([[.01],[.05]])
        lo,hi=reference_bounds(q,q,torch.full_like(q,-2),torch.full_like(q,2),.1,gap)
        torch.testing.assert_close(hi,gap.expand_as(q),rtol=0,atol=0);self.assertTrue(torch.equal(lo,-hi))

    def test_fail_closed(self):
        for field in ['distance','dmin']:
            k=self.inputs();k[field][0,0]=float('nan')
            with self.assertRaises(ValueError):reserve_forecast(**k)
        k=self.inputs();k['pending']=k['pending'][:5]
        with self.assertRaises(ValueError):reserve_forecast(**k)
        k=self.inputs();k['dt']=float('inf')
        with self.assertRaises(ValueError):reserve_forecast(**k)


if __name__=='__main__':unittest.main()
