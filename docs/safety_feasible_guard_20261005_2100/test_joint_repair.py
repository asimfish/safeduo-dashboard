import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'safety_joint_guard_20261005_1005'))
import unittest
import numpy as np
import torch
from scipy.optimize import linprog
from joint_repair import solve_joint, install
from test_projection_diagnostics import inputs
from safeduo.safety.backstop import BackstopConfig
from projection_diagnostics import projection_record

class JointRepairTests(unittest.TestCase):
    def test_real_solver_feasible_nonconvergence_red_then_repaired(self):
        env,cmd,rows,a,p,dt=inputs(m=2,cfg=BackstopConfig(max_passes=1,row_authority_clamp=False))
        rows.J['F'].zero_();rows.J['F'][0,0,0]=-1.;rows.J['F'][0,1,0]=1.;rows.J['F'][0,1,1]=-.1
        rows.d[0,1]=.015 # h=-.001
        cmd.delta_q['F_L'][0,:2]=.02
        bounds={k:(torch.full_like(v,-.025),torch.full_like(v,.025)) for k,v in cmd.delta_q.items()}
        kwargs=dict(delta_bounds=bounds)
        old=env._backstop.project(cmd,rows,a,p,dt,**kwargs)
        before=projection_record(env._backstop,cmd,rows,a,p,dt,kwargs,old)
        self.assertGreater(float(before['returned_safety_residual_F']),1e-5)
        install(env);new=env._backstop.project(cmd,rows,a,p,dt,**kwargs)
        after=projection_record(env._backstop,cmd,rows,a,p,dt,kwargs,new)
        self.assertLess(float(after['returned_safety_residual_F']),2e-7)
        self.assertLess(float(after['returned_bound_residual_F']),2e-7)
        self.assertEqual(float(new[2]['repair_safety_min_slack_m_F']),0.)
    def test_individually_feasible_jointly_infeasible(self):
        u,m=solve_joint([.02],[[1.],[-1.]],[-.01,-.01],np.empty((0,1)),[],[-.025],[.025])
        self.assertAlmostEqual(m['safety_min_slack_m'],.01,places=7)
        self.assertAlmostEqual(float(u[0]),0.,places=6)
        self.assertGreater(m['safety_residual_m'],.009999)
    def test_positive_singleton_is_not_fake_safe_zero(self):
        u,m=solve_joint([.025],[[1.]],[.0225],[[1.]],[0.],[.025],[.025])
        self.assertEqual(float(u[0]),.025)
        self.assertAlmostEqual(m['alpha_min_slack_rad'],.025)
        self.assertAlmostEqual(m['safety_min_slack_m'],.0025)
    def test_original_feasible_output_exact(self):
        env,cmd,rows,a,p,dt=inputs();cmd.delta_q['F_L'][0,0]=-.02
        old=env._backstop.project(cmd,rows,a,p,dt);install(env);new=env._backstop.project(cmd,rows,a,p,dt)
        self.assertTrue(torch.equal(old[0].stacked(),new[0].stacked()))
        self.assertEqual(float(new[2]['repair_applied_F']),0.)
    def test_nonfinite_and_empty_bounds_abort(self):
        for c,lo,hi in [([float('nan')],[-1],[1]),([0],[1],[-1])]:
            with self.assertRaises(ValueError):solve_joint(c,[[1.]],[0.],np.empty((0,1)),[],lo,hi)
    def test_random_feasible_intersections_independent_witness(self):
        rng=np.random.default_rng(8912727)
        for _ in range(24):
            d=5;w=rng.uniform(-.02,.02,d);G=rng.normal(size=(12,d));h=G@w+rng.uniform(0,.003,12)
            ag=rng.normal(size=(2,d));ah=ag@w+rng.uniform(0,.002,2);c=rng.uniform(-.025,.025,d)
            u,m=solve_joint(c,G,h,ag,ah,np.full(d,-.025),np.full(d,.025))
            self.assertLess(m['safety_min_slack_m'],1e-8);self.assertLess(m['alpha_min_slack_rad'],1e-8)
            self.assertLess(np.max(G@u-h),2e-7);self.assertLess(np.max(ag@u-ah),2e-7)

if __name__=='__main__':unittest.main(verbosity=2)
