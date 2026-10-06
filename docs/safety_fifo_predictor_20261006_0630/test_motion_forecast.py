"""Pre-outcome checks against independent NumPy FIFO oracle and old baseline."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np
import torch
from motion_forecast import (motion_displacements, model_coefficients, distance_forecasts,
                             governed_proposal, MODES, HORIZONS)
from target_forecast import full_forecast
from astra_fifo_oracle import DiagonalDynamics, rollout, fifo_targets
from safeduo.safety.types import ARM_KEYS, DeltaCmd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('prior_target_forecast',
        HERE.parent/'safety_feasible_guard_20261005_2100/full_finite_guard.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


def arms(value):
    return dict(zip(ARM_KEYS, value.split([7,7,6,6], -1)))


class MotionTests(unittest.TestCase):
    def setUp(self):
        gen = torch.Generator().manual_seed(870014)
        self.q = torch.randn((3,26), generator=gen)*.2
        self.qd = torch.randn((3,26), generator=gen)*.3
        self.issued = self.q + torch.randn((3,26), generator=gen)*.05
        self.pending = self.issued[None] + torch.randn((6,3,26), generator=gen)*.02
        self.cmd = DeltaCmd(arms(torch.randn((3,26), generator=gen)*.03))
        self.state = NS(q=arms(self.q), qd=arms(self.qd), dt=.016666)
        self.limits = {a: torch.stack([torch.full_like(v,-2.),torch.full_like(v,2.)],-1)
                       for a,v in self.state.q.items()}
        self.full = NS(dists=torch.rand((3,9021), generator=gen)*.1,
                       full_dmin=torch.full((3,9021),.01), closing=torch.zeros(3,9021))
        self.rows = NS(J={'F':torch.randn((3,9021,14),generator=gen)*.1,
                          'U':torch.randn((3,9021,12),generator=gen)*.1})

    def compute(self, mode):
        pending = [arms(t) for t in self.pending]
        base = full_forecast(self.full,self.rows,self.state,arms(self.issued),pending,
                             self.cmd,self.limits,.01)
        return distance_forecasts(self.full,self.rows,self.state,arms(self.issued),pending,
                                  self.cmd,self.limits,.01,base,mode),base

    def test_independent_numpy_on_six_actual_first_failures(self):
        model = json.loads((HERE/'MODEL_FIT.json').read_text())
        oracle = DiagonalDynamics(model['c_q'],model['c_v'],model['dt'])
        choices = json.loads((HERE/'astra_fifo_selection.json').read_text())['choices']
        for c in choices:
            with np.load(c['snapshot'],allow_pickle=False) as z:
                i=c['index'];q=torch.tensor(z['pre_q'][i:i+1]);v=torch.tensor(z['snapshot_qd'][i:i+1])
                pending=torch.tensor(z['pre_pending_actuator_targets'][i]).unsqueeze(1)
                proposal=torch.tensor(z['pre_issued_target'][i:i+1]+z['actual_project_return'][i:i+1])
            cq,cv,dt=model_coefficients(q)
            cvd,pdd=motion_displacements(q,v,pending,proposal,cq,cv,dt)
            applied,_=fifo_targets(pending[:,0].numpy(),np.repeat(proposal.numpy(),18,axis=0))
            expected,_=rollout(oracle,q[0].numpy(),v[0].numpy(),applied)
            np.testing.assert_allclose((pdd+q).numpy()[:,0],expected[np.array(HORIZONS)-1],atol=8e-6,rtol=8e-6)
            np.testing.assert_allclose(cvd.numpy()[:,0],np.array(HORIZONS)[:,None]*dt*v[0].numpy(),atol=1e-7)

    def test_new_proposal_cannot_affect_first_six_steps(self):
        cq,cv,dt=model_coefficients(self.q)
        h=(1,6,7,12,18)
        a=motion_displacements(self.q,self.qd,self.pending,self.issued,cq,cv,dt,h)[1]
        b=motion_displacements(self.q,self.qd,self.pending,self.issued+.3,cq,cv,dt,h)[1]
        self.assertTrue(torch.equal(a[:2],b[:2]))
        self.assertGreater(float((a[2]-b[2]).abs().max()),0)

    def test_baseline_and_source_copies_exact(self):
        (result,components),base=self.compute('joint_reference')
        old=prior.full_forecast(self.full,self.rows,self.state,arms(self.issued),
              [arms(t) for t in self.pending],self.cmd,self.limits,.01)
        self.assertTrue(torch.equal(old,base))
        self.assertTrue(torch.equal(result,base))
        for oldname,newname in [('full_finite_guard.py','target_forecast.py'),
                               ('reference_envelope.py','reference_envelope.py'),
                               ('projection_diagnostics.py','projection_diagnostics.py')]:
            self.assertEqual((HERE.parent/'safety_feasible_guard_20261005_2100'/oldname).read_bytes(),
                             (HERE/newname).read_bytes())

    def test_all_modes_add_only_and_exact_factor_membership(self):
        for mode in MODES:
            (result,c),base=self.compute(mode)
            expected=base
            if mode in ('velocity_admission','motion_admission'):expected=torch.minimum(expected,c['cv_forecast'])
            if mode in ('pd_admission','motion_admission'):expected=torch.minimum(expected,c['pd_forecast'])
            self.assertTrue(torch.equal(result,expected))
            self.assertTrue((result<=base).all())
            band=self.full.full_dmin+.010
            self.assertTrue(((result<=band)|~(base<=band)).all())

    def test_h6_distance_matches_independent_frozen_j(self):
        (result,c),_=self.compute('motion_admission')
        for name in ('cv','pd'):
            displacement=c[name+'_q_horizons'][1]-self.q
            expected=self.full.dists+sum(torch.einsum('nmd,nd->nm',self.rows.J[r],d)
                       for r,d in [('F',displacement[:,:14]),('U',displacement[:,14:])])
            torch.testing.assert_close(c[name+'_h6'],expected,atol=1e-7,rtol=1e-5)

    def test_hidden_last_row_nan_and_pending_inf_abort(self):
        self.rows.J['U'][2,9020,11]=float('nan')
        with self.assertRaisesRegex(ValueError,'nonfinite jacobian'):self.compute('joint_reference')
        self.rows.J['U'][2,9020,11]=0
        self.pending[5,2,25]=float('inf')
        with self.assertRaisesRegex(ValueError,'nonfinite target_5'):self.compute('joint_reference')

    def test_no_input_mutation_or_queue_reordering(self):
        tensors=[self.q,self.qd,self.issued,self.pending,self.full.dists,
                 *self.rows.J.values(),*self.cmd.delta_q.values()]
        before=[t.clone() for t in tensors]
        self.compute('motion_admission')
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(tensors,before)))

    def test_model_interval_and_mode_mismatch_abort(self):
        with self.assertRaisesRegex(ValueError,'unregistered'):self.compute('unexpected')
        self.state.dt=.008333
        with self.assertRaisesRegex(ValueError,'control interval'):self.compute('motion_admission')

    def test_proposal_in_original_speed_and_reference_bounds(self):
        p=governed_proposal(self.state,arms(self.issued),self.cmd,self.limits,.01)
        for a in ARM_KEYS:
            self.assertTrue(((p[a]-arms(self.issued)[a]).abs()<=.010001).all())
            self.assertTrue((p[a]>=-2).all() and (p[a]<=2).all())


if __name__=='__main__':unittest.main(verbosity=2)
