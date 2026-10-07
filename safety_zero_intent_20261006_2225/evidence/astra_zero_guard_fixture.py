"""Read-only source-derived CPU fixture for exact full-row admission guard.
Derived from sealed astra_tracking_tests.py; no simulation or source writes.
"""
import ast
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
H=Path(__file__).resolve().parent
REPO=Path('/home/liyufeng/safeduo')
ARMS=('F_L','F_R','U_L','U_R')
DOFS=(7,7,6,6)
CANDIDATE=None
def tensor_tree(value):
    if torch.is_tensor(value):return value.detach().clone()
    if isinstance(value,dict):return {k:tensor_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [tensor_tree(v) for v in value]
    return value


def tree_equal(test,a,b):
    if torch.is_tensor(a):test.assertTrue(torch.equal(a,b));return
    if isinstance(a,dict):
        test.assertEqual(set(a),set(b))
        for k in a:tree_equal(test,a[k],b[k])
    elif isinstance(a,list):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):tree_equal(test,x,y)
    else:test.assertEqual(a,b)


def extract_function(source,name,namespace):
    nodes=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==name]
    if len(nodes)!=1:raise ValueError('unique source function required: '+name)
    exec(compile(ast.Module(body=[nodes[0]],type_ignores=[]),'<exact extracted '+name+'>','exec'),namespace)
    return namespace[name]


@dataclass
class FullGeometryFixture:
    dists: object
    closing: object
    full_dmin: object
    full_viol_exempt: object
    active_pairs: object
    active_mask: object
    active_idx: object
    active_dmin: object
    viol_exempt: object


def guard_fixture(mode,proposal=.125,target_value=.03125,all_admitted=False):
    """Execute the exact nested safety function with CPU state/provider fixtures.

    Both historical union/select functions are exact extracted source. No Isaac,
    trace.start, actor, filesystem writer or physics update is invoked.
    """
    n,width=64,9021
    q={a:torch.zeros(n,d) for a,d in zip(ARMS,DOFS)}
    qd={a:v.clone() for a,v in q.items()};qd['F_L'][:,0]=.5
    d=torch.full((n,width),.0546875)
    if all_admitted:d[:]=0
    dm=torch.full_like(d,.03125);exempt=torch.zeros_like(d,dtype=torch.bool)
    idx=torch.arange(32).expand(n,-1);active=torch.ones(n,32,dtype=torch.bool)
    geometry=FullGeometryFixture(d,torch.zeros_like(d),dm,exempt,
        torch.zeros(n,32,4),active,idx,dm[:,:32],exempt[:,:32])
    j={'F':torch.zeros(n,width,14),'U':torch.zeros(n,width,12)};j['F'][:,-1,0]=-1
    state=SimpleNamespace(q=q,qd=qd,dt=.015625)
    from safeduo.safety.types import DeltaCmd
    pending=[{a:v.clone() for a,v in q.items()} for _ in range(6)]
    env=SimpleNamespace(_last_out=geometry,scene_state=lambda:state,num_envs=n,device='cpu',
        _sph=SimpleNamespace(class_id=torch.zeros(width),pair_id=torch.arange(width,dtype=torch.float32)),
        _body_pos_cache=None,_pending_cmd=DeltaCmd({a:torch.full_like(v,proposal) for a,v in q.items()}),
        _targets={a:torch.full_like(v,target_value) for a,v in q.items()},
        _evaluation_actuator_delay=SimpleNamespace(queue=SimpleNamespace(pending=pending)),
        _q_soft_limits={a:torch.stack([torch.full_like(v,-2),torch.full_like(v,2)],-1) for a,v in q.items()},
        _backstop=SimpleNamespace(cfg=SimpleNamespace(vmax=1.)))
    trace=SimpleNamespace(original_selected=lambda:geometry,original_rows=lambda out,body:SimpleNamespace(J=j),
        mode=mode,admission_on=True,capacity=9021,guard_calls=0,last_unreachable=torch.zeros(n,4,dtype=torch.bool))
    source=(H.parent/'safety_mechanism_20261005_causal_obs/mechanism_runner.py').read_text()
    adm_ns=dict(torch=torch,replace=replace,CAPACITY=1024)
    union=extract_function(source,'union_mask',adm_ns);select=extract_function(source,'select',adm_ns)
    from target_forecast import full_forecast,require_finite
    from safeduo.baselines.base import stack_robot
    aborted=[]
    namespace=dict(self=trace,env=env,torch=torch,np=np,replace=replace,ARM_KEYS=ARMS,
        admission=SimpleNamespace(union_mask=union,select=select),full_forecast=full_forecast,
        require_finite=require_finite,stack_robot=stack_robot,
        reserve_forecast=CANDIDATE.tracking.reserve_forecast,reference_gap=CANDIDATE.zero.reference_gap,
        reference_bounds=CANDIDATE.reference.reference_bounds,abort=lambda error,stage:aborted.append((type(error).__name__,stage)))
    namespace['checked_rows']=extract_function(CANDIDATE.text['guard_runner.py'],'checked_rows',namespace)
    safety=extract_function(CANDIDATE.text['guard_runner.py'],'safety',namespace)
    return safety,trace,env,j,aborted,select


class GuardWiringTests(unittest.TestCase):
    def test_exact_guard_modes_have_identical_old_target_admission(self):
        masks=[]
        for mode in ['joint_reference','tight_reference','zero_inclusive']:
            safety,trace,env,j,aborted,_=guard_fixture(mode)
            queue_before=tensor_tree(env._evaluation_actuator_delay.queue.pending)
            result=safety();masks.append(result.active_idx.clone())
            self.assertFalse(aborted);self.assertEqual(trace.guard_calls,1)
            self.assertEqual(tuple(trace.last_chunk['reserve_margin'].shape),(64,))
            self.assertEqual(tuple(trace.last_chunk['reserve_forecast'].shape),(64,9021))
            self.assertEqual(tuple(trace.reserve_gap.shape),(64,1))
            self.assertEqual(trace.capacity,9021)
            self.assertTrue((result.active_idx[:,0]==0).all())
            self.assertTrue((result.active_idx[:,-1]==9020).all())
            self.assertTrue((trace.last_unreachable==(mode!='joint_reference')).all(),
                'actual tight/zero-inclusive gap must determine archived unreachability')
            tree_equal(self,env._evaluation_actuator_delay.queue.pending,queue_before)
        self.assertTrue(all(torch.equal(masks[0],v) for v in masks[1:]))

    def test_current_proposal_changes_admission_but_not_prequeue_reserve(self):
        safety,trace,env,j,aborted,_=guard_fixture('zero_inclusive',proposal=.125,target_value=0.)
        one=safety();risk1=trace.last_chunk['reserve_margin'].copy();gap1=trace.reserve_gap.clone()
        for v in env._pending_cmd.delta_q.values():v[:]=-.125
        two=safety()
        self.assertTrue(np.array_equal(risk1,trace.last_chunk['reserve_margin']))
        self.assertTrue(torch.equal(gap1,trace.reserve_gap))
        self.assertFalse(torch.equal(one.active_idx,two.active_idx),'fixture must expose real proposal admission sensitivity')

    def test_all9021_rows_retained_and_insufficient_budget_aborts(self):
        safety,trace,env,j,aborted,select=guard_fixture('zero_inclusive',all_admitted=True)
        selected=safety()
        self.assertEqual(tuple(selected.active_idx.shape),(64,9021))
        self.assertTrue((selected.active_mask.sum(-1)==9021).all())
        with self.assertRaises(ValueError):
            select(env._last_out,env._last_out,torch.ones(64,9021,dtype=torch.bool),env._sph,capacity=9020)

    def test_hidden_nonfinite_last_row_aborts_before_projection(self):
        safety,trace,env,j,aborted,_=guard_fixture('zero_inclusive')
        j['U'][0,-1,-1]=float('nan')
        with self.assertRaises(ValueError):safety()
        self.assertEqual(trace.guard_calls,0)
        self.assertEqual(aborted,[('ValueError','before projection/physics')])

    def test_producer_physics_order_and_no_queue_writes_in_guard(self):
        source=(REPO/'src/safeduo/envs/duo_env.py').read_text()
        tree=ast.parse(source)
        method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_pre_physics_step')
        calls=[n for n in ast.walk(method) if isinstance(n,ast.Call)]
        safety=min(n.lineno for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='safety_dist_out')
        project=min(n.lineno for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='project')
        self.assertLess(safety,project)
        guard_tree=ast.parse(CANDIDATE.text['guard_runner.py'])
        writes=[ast.unparse(n) for n in ast.walk(guard_tree) if isinstance(n,(ast.Assign,ast.AugAssign)) and
                any('queue' in ast.unparse(t) or '_targets[' in ast.unparse(t) for t in (n.targets if isinstance(n,ast.Assign) else [n.target]))]
        self.assertEqual(writes,[])
        proposal_calls=[n for n in ast.walk(guard_tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='reserve_forecast']
        self.assertEqual(len(proposal_calls),1)
        self.assertNotIn('_pending_cmd',ast.unparse(proposal_calls[0]))
        self.assertIn('queue.pending',ast.unparse(proposal_calls[0]))
