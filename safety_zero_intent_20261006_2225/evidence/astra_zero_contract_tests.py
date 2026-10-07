"""Independent CPU tests of notified source, exact scalar oracles and mutants.

No actor, Isaac, simulation, outcome files, parent tests or parent scores.
Generic API hazards are reported separately from the fixed registered call path.
"""
import argparse
import ast
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from astra_zero_oracle import reference_interval_oracle, zero_interval_oracle
import astra_zero_guard_fixture as fixture

H = Path(__file__).resolve().parent
REPO = Path('/home/liyufeng/safeduo')
PRIOR = H.parent/'safety_tracking_reserve_20261006_1535'
NAMES = ('reference_envelope.py', 'zero_intent.py', 'guard_runner.py',
         'target_forecast.py', 'projection_diagnostics.py', 'tracking_reserve.py')
ARMS, DOFS = fixture.ARMS, fixture.DOFS
SOURCE = {}
REFERENCE = None
OLD = None


def module(text, name):
    scope = {'__file__':str(H/name), '__name__':'astra_zero_test_'+name[:-3]}
    exec(compile(text, str(H/name), 'exec'), scope)
    return SimpleNamespace(**scope)


def values(q=0., target=0., lower=-1., upper=1., shape=(2, 7), dtype=torch.float32):
    return [torch.full(shape, v, dtype=dtype, device='cpu') for v in (q,target,lower,upper)]


class ZeroBoundsTests(unittest.TestCase):
    def test_zero_is_admitted_both_drift_signs_with_correction_freedom(self):
        for sign in (-1, 1):
            x=values(q=sign*.03)
            lo,hi=REFERENCE.reference_bounds(*x,.025,.01,zero_inclusive=True)
            self.assertTrue(((lo<=0)&(hi>=0)).all())
            self.assertTrue((hi-lo>0).all(), 'must preserve nonzero safety correction freedom')
            raw=torch.zeros_like(lo)
            self.assertTrue(torch.equal(raw.maximum(lo).minimum(hi),raw))
            oldlo,oldhi=OLD.reference_bounds(*x,.025,.01)
            self.assertTrue((torch.abs(raw.maximum(oldlo).minimum(oldhi))>0).all())

    def test_invalid_zero_outside_original_bounds_rejects_both_sides(self):
        for target in (-1.01,1.01):
            x=values(target=target)
            lo,hi=OLD.reference_bounds(*x,.025,.01)
            self.assertTrue((lo<=hi).all(), 'original reachable interval must be nonempty')
            with self.assertRaisesRegex(ValueError,'outside original reachable'):
                REFERENCE.reference_bounds(*x,.025,.01,zero_inclusive=True)

    def test_original_empty_interval_rejects_without_holding_invalid_target(self):
        for flag in (False,True):
            with self.assertRaisesRegex(ValueError,'unreachable'):
                REFERENCE.reference_bounds(*values(target=5.),.025,.01,zero_inclusive=flag)

    def test_boundary_targets_and_gap_none_preserve_original_bounds(self):
        for target in (-1.,1.):
            x=values(target=target,q=-target)
            lo,hi=REFERENCE.reference_bounds(*x,.025,.01,zero_inclusive=True)
            self.assertTrue(((lo<=0)&(hi>=0)).all())
            self.assertTrue(((target+lo>=-1)&(target+hi<=1)).all())
            for a,b in zip(REFERENCE.reference_bounds(*x,.025,None,zero_inclusive=True),
                           OLD.reference_bounds(*x,.025,None)):
                self.assertTrue(torch.equal(a,b))

    def test_exact_breakpoint_oracle_on_binary_inputs_float32_and_float64(self):
        rng=np.random.default_rng(22252225)
        for dtype in (torch.float32,torch.float64):
            for t,q,b,g in rng.integers([-32,-48,1,1],[33,49,8,12],size=(160,4)):
                target,measured,box,gap=t/32,q/32,b/64,g/128
                x=values(q=measured,target=target,shape=(1,1),dtype=dtype)
                actual=REFERENCE.reference_bounds(*x,box,gap,zero_inclusive=True)
                expected=zero_interval_oracle(target,measured,box,gap,-1.,1.)
                # All arithmetic for this fixture is exactly representable binary.
                self.assertEqual(tuple(float(a.item()) for a in actual),tuple(map(float,expected)))

    def test_false_flag_controls_match_prior_bitwise_including_unreachable_envelopes(self):
        gen=torch.Generator(device='cpu').manual_seed(2225)
        for dtype in (torch.float32,torch.float64):
            q=torch.rand((64,7),generator=gen,dtype=dtype)*4-2
            target=torch.rand((64,7),generator=gen,dtype=dtype)*2-1
            x=[q,target,torch.full_like(q,-1),torch.full_like(q,1)]
            for gap in (None,.050,.010,torch.full((64,1),.010,dtype=dtype)):
                a=REFERENCE.reference_bounds(*x,.024999,gap,zero_inclusive=False)
                b=OLD.reference_bounds(*x,.024999,gap)
                for left,right in zip(a,b):
                    self.assertEqual(left.numpy().tobytes(),right.numpy().tobytes())

    def test_registered_batch_shape_dtype_device_and_input_immutability(self):
        for dtype in (torch.float32,torch.float64):
            for batch,joints in ((64,7),(64,6),(7,7),(6,6)):
                x=values(q=.03,shape=(batch,joints),dtype=dtype)
                gap=torch.full((batch,1),.01,dtype=dtype)
                before=[a.clone() for a in x+[gap]]
                lo,hi=REFERENCE.reference_bounds(*x,.025,gap,zero_inclusive=True)
                for a in (lo,hi):
                    self.assertEqual(a.shape,x[0].shape)
                    self.assertEqual(a.dtype,dtype)
                    self.assertEqual(a.device.type,'cpu')
                    self.assertTrue(torch.isfinite(a).all())
                for a,b in zip(x+[gap],before): self.assertTrue(torch.equal(a,b))

    def test_finite_positive_parameters_shapes_and_limit_order(self):
        for field in range(6):
            for invalid in (float('nan'),float('inf'),float('-inf')):
                x=values()+[.025,.01]
                if field<4: x[field][0,0]=invalid
                else: x[field]=invalid
                with self.subTest(field=field,invalid=invalid),self.assertRaises((ValueError,RuntimeError)):
                    REFERENCE.reference_bounds(*x,zero_inclusive=True)
        for index in (4,5):
            for invalid in (0.,-.01):
                x=values()+[.025,.01];x[index]=invalid
                with self.assertRaises((ValueError,RuntimeError)):
                    REFERENCE.reference_bounds(*x,zero_inclusive=True)
        for index in range(4):
            x=values();x[index]=x[index][:1]
            with self.assertRaises(ValueError): REFERENCE.reference_bounds(*x,.025,.01,zero_inclusive=True)
        with self.assertRaises(ValueError):
            REFERENCE.reference_bounds(*values(lower=2.),.025,.01,zero_inclusive=True)

    def test_gap_expansion_joint_axis_and_mixed_dtype_device_are_rejected(self):
        for shape in ((7,7),(64,7),(64,6)):
            x=values(shape=shape)
            for gap in (torch.full((shape[0],),.01),torch.full((2,1,1),.01),torch.full((1,shape[1]),.01)):
                with self.subTest(shape=shape,gap=list(gap.shape)),self.assertRaisesRegex(ValueError,'gap requires'):
                    REFERENCE.reference_bounds(*x,.025,gap,zero_inclusive=True)
            a=REFERENCE.reference_bounds(*x,.025,torch.full(shape,.01),zero_inclusive=True)
            b=REFERENCE.reference_bounds(*x,.025,.01,zero_inclusive=True)
            self.assertTrue(all(torch.equal(l,r) for l,r in zip(a,b)))
        for idx in (1,2,3):
            x=values();x[idx]=x[idx].to(torch.float64)
            with self.assertRaisesRegex(ValueError,'identical floating dtype and device'):
                REFERENCE.reference_bounds(*x,.025,.01,zero_inclusive=True)
            x=values();x[idx]=torch.empty(x[idx].shape,device='meta')
            with self.assertRaisesRegex(ValueError,'identical floating dtype and device'):
                REFERENCE.reference_bounds(*x,.025,.01,zero_inclusive=True)
        for flag in (0,1,'yes',torch.tensor(True)):
            with self.assertRaisesRegex(ValueError,'flag must be boolean'):
                REFERENCE.reference_bounds(*values(),.025,.01,zero_inclusive=flag)
        for box in (torch.tensor([.025]),torch.tensor([[.025]])):
            with self.assertRaisesRegex(ValueError,'box must be scalar'):
                REFERENCE.reference_bounds(*values(),box,.01,zero_inclusive=True)
        with self.assertRaisesRegex(ValueError,'floating batch-by-joint'):
            REFERENCE.reference_bounds(*values(dtype=torch.int32),.025,.01,zero_inclusive=True)


def make_env(project):
    q={a:torch.full((2,d),.03) for a,d in zip(ARMS,DOFS)}
    targets={a:torch.zeros_like(v) for a,v in q.items()}
    pending=[{a:v.clone() for a,v in targets.items()} for _ in range(6)]
    return SimpleNamespace(scene_state=lambda:SimpleNamespace(q=q),_targets=targets,
        pending=pending, root=torch.tensor([[0.,0.,0.,1.,0.,0.,0.]]),
        qd={a:torch.zeros_like(v) for a,v in q.items()},
        _q_soft_limits={a:torch.stack([torch.full_like(v,-1),torch.full_like(v,1)],-1) for a,v in q.items()},
        _backstop=SimpleNamespace(project=project,cfg=SimpleNamespace(vmax=1.5)))


class ProjectionIntegrationTests(unittest.TestCase):
    def test_r19_input_is_limited_before_verbatim_return_and_no_state_changes(self):
        from safeduo.safety.types import DeltaCmd
        captured=[]
        def original(cmd,rows,alpha,p,dt,**kwargs):
            captured.append((cmd,kwargs))
            return cmd,torch.zeros(2,4,dtype=torch.bool),{}
        env=make_env(original)
        before=fixture.tensor_tree(dict(q=env.scene_state().q,targets=env._targets,
            pending=env.pending,root=env.root,qd=env.qd))
        REFERENCE.install(env,'envelope_050',gap_provider=lambda:torch.full((2,1),.01),zero_inclusive=True)
        bypass=torch.ones(2,4,dtype=torch.bool)
        for raw in (0.,-.125,.125):
            cmd=DeltaCmd({a:torch.full_like(v,raw) for a,v in env._targets.items()})
            result,active,info=env._backstop.project(cmd,None,None,None,.016666,bypass_arm=bypass)
            self.assertIs(captured[-1][1]['bypass_arm'],bypass)
            for a in ARMS:
                lo,hi=captured[-1][1]['delta_bounds'][a]
                self.assertTrue(torch.equal(captured[-1][0].delta_q[a],cmd.delta_q[a].maximum(lo).minimum(hi)))
                self.assertTrue(torch.equal(result.delta_q[a],captured[-1][0].delta_q[a]))
            self.assertEqual(bool(info['reference_governor_changed'].any()),raw!=0.)
        fixture.tree_equal(self,before,dict(q=env.scene_state().q,targets=env._targets,
            pending=env.pending,root=env.root,qd=env.qd))

    def test_nonzero_projector_correction_is_not_overwritten_on_zero_intent(self):
        from safeduo.safety.types import DeltaCmd
        corrections=[]
        def correct(cmd,rows,alpha,p,dt,**kwargs):
            out=DeltaCmd({a:torch.full_like(v,.0125) for a,v in cmd.delta_q.items()})
            corrections.append(out)
            return out,torch.ones(2,4,dtype=torch.bool),{}
        env=make_env(correct)
        REFERENCE.install(env,'envelope_050',gap_provider=lambda:.01,zero_inclusive=True)
        cmd=DeltaCmd({a:torch.zeros_like(v) for a,v in env._targets.items()})
        result,active,info=env._backstop.project(cmd,None,None,None,.016666)
        self.assertIs(result,corrections[0])
        self.assertTrue(all((v==.0125).all() for v in result.delta_q.values()))
        self.assertTrue(active.all())

    def test_real_finite_projection_can_open_on_raw_zero_and_retains_r19(self):
        from safeduo.safety.backstop import BackstopConfig,VelocityDamperBackstop
        from safeduo.safety.types import DeltaCmd
        cfg=BackstopConfig(vmax=1.5,max_passes=30)
        solver=VelocityDamperBackstop(cfg)
        env=make_env(solver.project)
        env._backstop.cfg=cfg
        j={'F':torch.zeros(2,1,14),'U':torch.zeros(2,1,12)}
        j['F'][:,:,0]=1
        rows=SimpleNamespace(d=torch.zeros(2,1),d_min=torch.full((2,1),.01),
            cls=torch.ones(2,1),valid=torch.ones(2,1,dtype=torch.bool),J=j,
            arm_mask=torch.tensor([[[True,False,False,False]]]).expand(2,-1,-1))
        REFERENCE.install(env,'envelope_050',gap_provider=lambda:.01,zero_inclusive=True)
        cmd=DeltaCmd({a:torch.zeros_like(v) for a,v in env._targets.items()})
        result,active,info=env._backstop.project(cmd,rows,torch.ones(2,4),torch.zeros(2),.016666)
        self.assertTrue((result.delta_q['F_L'][:,0]>0).all())
        self.assertTrue(all(torch.isfinite(v).all() for v in result.delta_q.values()))
        self.assertLessEqual(info['passes_F'],30)
        result,_,_=env._backstop.project(cmd,rows,torch.ones(2,4),torch.zeros(2),.016666,
                                       bypass_arm=torch.ones(2,4,dtype=torch.bool))
        self.assertTrue(all(torch.equal(v,torch.zeros_like(v)) for v in result.delta_q.values()))

    def test_common_finite_gate_rejects_nonfinite_correction(self):
        from projection_diagnostics import checked_project
        from safeduo.safety.types import DeltaCmd
        cmd=DeltaCmd({a:torch.zeros(2,d) for a,d in zip(ARMS,DOFS)})
        rows=SimpleNamespace(d=torch.ones(2,1),J={'F':torch.zeros(2,1,14),'U':torch.zeros(2,1,12)},
            cls=torch.ones(2,1),arm_mask=torch.ones(2,1,4,dtype=torch.bool),
            valid=torch.ones(2,1,dtype=torch.bool),d_min=torch.zeros(2,1))
        def corrupt(*args,**kwargs):
            bad=cmd.clone();bad.delta_q['U_R'][0,0]=float('nan')
            return bad,torch.zeros(2,4,dtype=torch.bool),{}
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            checked_project(corrupt,cmd,rows,torch.ones(2,4),torch.zeros(2),.016666)


class FixedGapWiringTests(unittest.TestCase):
    def test_all_modes_fixed_even_when_diagnostic_risk_changes(self):
        for mode,want in [('joint_reference',.05),('tight_reference',.01),('zero_inclusive',.01)]:
            safety,trace,env,j,aborted,_=fixture.guard_fixture(mode)
            safety();first=trace.reserve_gap.clone();risk=trace.last_chunk['reserve_margin'].copy()
            env.scene_state().qd['F_L'][:,0]=8.
            safety()
            self.assertFalse(np.array_equal(risk,trace.last_chunk['reserve_margin']))
            self.assertTrue(torch.equal(first,trace.reserve_gap))
            self.assertTrue(torch.equal(first,torch.full_like(first,want)))

    def test_exact_install_call_selects_zero_flag_only_for_candidate(self):
        tree=ast.parse(SOURCE['guard_runner.py'])
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='install']
        self.assertEqual(len(calls),1)
        keywords={k.arg:k.value for k in calls[0].keywords}
        expression=compile(ast.Expression(keywords['zero_inclusive']),'<registered zero flag>','eval')
        for mode in fixture.CANDIDATE.zero.MODES:
            self.assertEqual(eval(expression,{},dict(self=SimpleNamespace(mode=mode))),mode=='zero_inclusive')


def mutation_checks():
    global REFERENCE
    original=REFERENCE
    variants=[
        ('drop_zero_hull', [('reference_lo = reference_lo.minimum(torch.zeros_like(reference_lo))','reference_lo = reference_lo'),
                           ('reference_hi = reference_hi.maximum(torch.zeros_like(reference_hi))','reference_hi = reference_hi')],
         ZeroBoundsTests,'test_zero_is_admitted_both_drift_signs_with_correction_freedom'),
        ('admit_invalid_zero', [("if zero_inclusive and ((lo > 0).any() or (hi < 0).any()):","if False:")],
         ZeroBoundsTests,'test_invalid_zero_outside_original_bounds_rejects_both_sides'),
        ('collapse_correction_interval', [('reference_hi = reference_hi.maximum(torch.zeros_like(reference_hi))','reference_hi = torch.zeros_like(reference_hi)')],
         ZeroBoundsTests,'test_zero_is_admitted_both_drift_signs_with_correction_freedom'),
        ('remove_prelimit', [('cmd.delta_q[a].maximum(bounds[a][0]).minimum(bounds[a][1])','cmd.delta_q[a]')],
         ProjectionIntegrationTests,'test_r19_input_is_limited_before_verbatim_return_and_no_state_changes'),
        ('force_zero_after_projection', [('return result, active | changed, info','result.delta_q = {a: torch.zeros_like(v) for a,v in result.delta_q.items()}\n        return result, active | changed, info')],
         ProjectionIntegrationTests,'test_nonzero_projector_correction_is_not_overwritten_on_zero_intent')]
    results=[]
    try:
        for name,replacements,cls,test in variants:
            text=SOURCE['reference_envelope.py']
            for old,new in replacements:
                if text.count(old)!=1: raise RuntimeError('mutation anchor changed: '+name)
                text=text.replace(old,new)
            REFERENCE=module(text,'reference_envelope.py')
            stream=io.StringIO()
            result=unittest.TextTestRunner(stream=stream).run(unittest.TestSuite([cls(test)]))
            results.append(dict(name=name,killed=not result.wasSuccessful(),tests=result.testsRun,
                                failures=len(result.failures),errors=len(result.errors),evidence=stream.getvalue()))
    finally: REFERENCE=original
    if not all(r['killed'] for r in results): raise AssertionError('surviving mutant')
    print('MUTATION_RESULTS',json.dumps(results),flush=True)


def api_observations():
    print('API_OBSERVATIONS',json.dumps(dict(scope='Current notified validation is exercised for scalar, batch-by-one and exact-state gaps; rejected joint-axis/expansion gaps and mixed dtype/meta-device inputs. No CUDA device execution.')),flush=True)


def main():
    global SOURCE,REFERENCE,OLD
    parser=argparse.ArgumentParser();parser.add_argument('--mutations',action='store_true')
    args=parser.parse_args()
    sys.dont_write_bytecode=True
    sys.path[:0]=[str(H),str(REPO/'src')]
    torch.set_num_threads(1)
    SOURCE={name:(H/name).read_text() for name in NAMES}
    hashes={str(H/name):hashlib.sha256(text.encode()).hexdigest() for name,text in SOURCE.items()}
    print('TESTED_SOURCE_SHA256',json.dumps(hashes,sort_keys=True),flush=True)
    REFERENCE=module(SOURCE['reference_envelope.py'],'reference_envelope.py')
    OLD=module((PRIOR/'reference_envelope.py').read_text(),'sealed_prior_reference_envelope.py')
    fixture.CANDIDATE=SimpleNamespace(reference=REFERENCE,zero=module(SOURCE['zero_intent.py'],'zero_intent.py'),
        tracking=module(SOURCE['tracking_reserve.py'],'tracking_reserve.py'),text=SOURCE)
    classes=[ZeroBoundsTests,ProjectionIntegrationTests,FixedGapWiringTests,fixture.GuardWiringTests]
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in classes)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if args.mutations: mutation_checks()
    api_observations()
    for path,digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('reviewed source changed during CPU tests: '+path)
    print('CPU_SCOPE',json.dumps(dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        no_gpu=True,no_simulation=True,no_outcomes=True,source_stable=True)),flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__': raise SystemExit(main())
