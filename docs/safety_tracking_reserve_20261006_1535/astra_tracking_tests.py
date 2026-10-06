"""Synthetic CPU oracle/candidate contracts; no candidate imported at intake.

Default candidate phase is deliberately RED while registration/interface is
unavailable. --phase oracle runs only independent counterexample self-checks.
The candidate adapter will be bound to observed registered source, not guessed.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import ast
import copy
from dataclasses import dataclass, replace
import hashlib
import json
import sys
from types import SimpleNamespace
import unittest
import numpy as np
from astra_tracking_oracle import adaptive_gap,fixtures,reference_interval_oracle,risk_oracle

H=Path(__file__).resolve().parent
REPO=Path('/home/liyufeng/safeduo')
ARMS=('F_L','F_R','U_L','U_R')
DOFS=(7,7,6,6)
CANDIDATE=None
torch=None


def split_state(value):
    return dict(zip(ARMS,torch.split(value,DOFS,dim=-1)))


def adapt(case,dtype=None):
    dtype=dtype or torch.float64
    tensor=lambda v:torch.tensor(v,dtype=dtype,device='cpu')
    j=tensor(case['jacobian'])
    return dict(distance=tensor(case['distance']),dmin=tensor(case['dmin']),
        exempt=torch.tensor(case['exempt'],dtype=torch.bool),jacobian={'F':j[...,:14],'U':j[...,14:]},
        q=split_state(tensor(case['q'])),qd=split_state(tensor(case['qd'])),
        pending=[split_state(tensor(slot)) for slot in case['pending']],dt=case['dt'])


def finite_case(n=2,rows=4):
    rng=np.random.default_rng(1031)
    q=rng.integers(-8,9,size=(n,26))/32
    return dict(distance=rng.integers(2,12,size=(n,rows))/64,dmin=np.full((n,rows),1/64),
        jacobian=rng.integers(-4,5,size=(n,rows,26))/16,q=q,qd=rng.integers(-4,5,size=q.shape)/32,
        pending=np.repeat(q[None],6,axis=0)+rng.integers(-3,4,size=(6,n,26))/32,
        exempt=np.zeros((n,rows),bool),dt=.03125)


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


def load_candidate(revision):
    global torch
    sys.path[:0]=[str(H),str(REPO/'src')]
    import torch as cpu_torch
    torch=cpu_torch;torch.set_num_threads(1)
    names=['tracking_reserve.py','reference_envelope.py','guard_runner.py','target_forecast.py','projection_diagnostics.py']
    texts={n:(H/n).read_text() for n in names}
    if revision=='initial':
        saved=json.loads((H/'astra_tracking_sources_initial.json').read_text())
        texts={n:saved['sources'][n]['text'] for n in names}
    modules={}
    for name in ['tracking_reserve.py','reference_envelope.py']:
        ns={'__file__':str(H/name),'__name__':'astra_test_'+name[:-3]}
        exec(compile(texts[name],str(H/name),'exec'),ns)
        modules[name]=SimpleNamespace(**ns)
    return SimpleNamespace(tracking=modules['tracking_reserve.py'],reference=modules['reference_envelope.py'],
        text=texts,revision=revision,source_sha256={n:hashlib.sha256(v.encode()).hexdigest() for n,v in texts.items()})


class IndependentOracleTests(unittest.TestCase):
    def test_velocity_without_target_debt_rejects_static_only_mutant(self):
        case=fixtures()['velocity_without_target_debt']
        actual=risk_oracle(**case)[0]
        self.assertEqual(actual['risk'],Fraction(-7,64))
        self.assertEqual(actual['endpoint'],'velocity_18dt')
        mutant=copy.deepcopy(case);mutant['qd'][:]=0
        self.assertGreater(risk_oracle(**mutant)[0]['risk'],actual['risk'])

    def test_intermediate_pending_rejects_last_only_mutant(self):
        case=fixtures()['intermediate_pending_hazard']
        actual=risk_oracle(**case)[0]
        self.assertEqual(actual['risk'],Fraction(-3,32))
        self.assertEqual(actual['endpoint'],'pending_2')
        mutant=copy.deepcopy(case);mutant['pending'][:]=mutant['pending'][-1]
        self.assertGreater(risk_oracle(**mutant)[0]['risk'],actual['risk'])

    def test_joint_arms_reject_single_arm_mutant(self):
        case=fixtures()['four_arm_joint_closing']
        self.assertEqual(risk_oracle(**case)[0]['risk'],Fraction(-1,32))
        mutant=copy.deepcopy(case);mutant['jacobian'][...,7:]=0
        self.assertEqual(risk_oracle(**mutant)[0]['risk'],Fraction(1,64))

    def test_opening_cannot_hide_measured_minimum(self):
        case=fixtures()['velocity_without_target_debt'];case['qd']*=-1
        self.assertEqual(risk_oracle(**case)[0]['risk'],Fraction(1,32))

    def test_exemption_and_dmin_have_distinct_meanings(self):
        case=fixtures()['intermediate_pending_hazard']
        original=risk_oracle(**case)[0]['risk']
        case['dmin'][:]=0
        self.assertEqual(risk_oracle(**case)[0]['risk']-original,Fraction(1,32))
        case['exempt'][:]=True
        self.assertIsNone(risk_oracle(**case)[0]['risk'])

    def test_gap_exact_decimal_boundary_and_monotonicity(self):
        self.assertEqual(adaptive_gap(Fraction(1,100)),Fraction(1,100))
        self.assertEqual(adaptive_gap(Fraction(5,100)),Fraction(5,100))
        self.assertEqual(adaptive_gap(Fraction(3,100)),Fraction(3,100))
        values=[adaptive_gap(Fraction(i,1000)) for i in range(-10,71)]
        self.assertEqual(values,sorted(values))

    def test_reference_unreachable_tight_gap_never_snaps(self):
        left,right,reachable=reference_interval_oracle(.25,0,.03125,.01)
        self.assertEqual((left,right,reachable),(Fraction(-1,32),Fraction(-1,32),False))
        self.assertGreater(Fraction(1,4)+left,Fraction(1,100))
        a,b,ok=reference_interval_oracle(-.25,0,.03125,.01)
        self.assertEqual((a,b,ok),(Fraction(1,32),Fraction(1,32),False))

    def test_reference_reachable_intersection_and_tangent(self):
        self.assertEqual(reference_interval_oracle(0,0,.125,.0625),(-Fraction(1,16),Fraction(1,16),True))
        self.assertEqual(reference_interval_oracle(.125,0,.0625,.0625),(-Fraction(1,16),-Fraction(1,16),True))

    def test_corrupt_exempt_inputs_and_bad_queue_do_not_turn_safe(self):
        case=fixtures()['intermediate_pending_hazard']
        case['exempt'][:]=True;case['jacobian'][0,0,0]=np.nan
        with self.assertRaises(ValueError):risk_oracle(**case)
        for count in [0,5,7]:
            case=fixtures()['intermediate_pending_hazard'];case['pending']=np.zeros((count,1,26))
            with self.assertRaises(ValueError):risk_oracle(**case)

    def test_new_target_delivered_only_on_transition_seven(self):
        queue=[f'old{i}' for i in range(6)];new='new0';seen=[]
        for step in range(7):
            seen.append(queue[0]);queue=queue[1:]+[new if step==0 else f'later{step}']
        self.assertEqual(seen,[f'old{i}' for i in range(6)]+[new])


class ReserveContractTests(unittest.TestCase):
    def check_oracle(self,case,dtype=None):
        inputs=adapt(case,dtype);before=tensor_tree(inputs)
        result=CANDIDATE.tracking.reserve_forecast(**inputs)
        expected=risk_oracle(**case)
        want=torch.tensor([float(r['risk']) for r in expected],dtype=inputs['distance'].dtype)
        # All synthetic reference inputs in these comparisons are dyadic, and
        # their bounded scalar sums are exactly representable in both dtypes.
        torch.testing.assert_close(result['risk_margin'],want,rtol=0,atol=0)
        for row in range(case['distance'].shape[1]):
            single={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in case.items()}
            for key in ['distance','dmin','exempt']:single[key]=single[key][:,row:row+1]
            single['exempt'][:]=False
            single['jacobian']=single['jacobian'][:,row:row+1]
            exact=risk_oracle(**single)
            want=torch.tensor([float(e['risk']+Fraction.from_float(float(single['dmin'][i,0]))) for i,e in enumerate(exact)],dtype=inputs['distance'].dtype)
            torch.testing.assert_close(result['distance'][:,row],want,rtol=0,atol=0)
        tree_equal(self,inputs,before)
        return result

    def test_velocity_at_zero_debt_and_direction(self):
        case=fixtures()['velocity_without_target_debt'];self.check_oracle(case)
        case['qd']*=-1;self.check_oracle(case)

    def test_every_pending_slot_can_determine_minimum(self):
        for slot in range(6):
            case=fixtures()['intermediate_pending_hazard'];case['pending'][:]=0
            case['pending'][slot,0,0]=.125
            with self.subTest(slot=slot):self.check_oracle(case)

    def test_absolute_targets_not_summed_as_increments(self):
        case=fixtures()['intermediate_pending_hazard'];case['q'][:]=1
        case['pending'][:]=case['q'];case['pending'][:,:,0]+=.125
        self.check_oracle(case)

    def test_each_joint_all_four_arm_slices(self):
        for joint in range(26):
            case=fixtures()['intermediate_pending_hazard'];case['jacobian'][:]=0;case['pending'][:]=0
            case['jacobian'][0,0,joint]=-1;case['pending'][4,0,joint]=.125
            with self.subTest(joint=joint):self.check_oracle(case)

    def test_joint_reinforcement_and_cancellation(self):
        case=fixtures()['four_arm_joint_closing'];self.check_oracle(case)
        case['jacobian'][0,0,[7,20]]=1;self.check_oracle(case)

    def test_seeded_multirow_multienv_oracle_float32_and64(self):
        for n,rows in [(1,1),(2,4),(3,7)]:
            for dtype in [torch.float32,torch.float64]:
                with self.subTest(n=n,rows=rows,dtype=dtype):self.check_oracle(finite_case(n,rows),dtype)

    def test_row_permutation_duplication_and_batch_isolation(self):
        case=finite_case();original=self.check_oracle(case)
        for key in ['distance','dmin','exempt','jacobian']:case[key]=case[key][:,[3,0,1,2,3]]
        changed=self.check_oracle(case)
        torch.testing.assert_close(original['risk_margin'],changed['risk_margin'],rtol=0,atol=0)
        single={k:(v[:1].copy() if isinstance(v,np.ndarray) and k!='pending' else v[:,:1].copy() if k=='pending' else v) for k,v in case.items()}
        result=self.check_oracle(single)
        torch.testing.assert_close(result['risk_margin'],original['risk_margin'][:1],rtol=0,atol=0)

    def test_exact_original_exemption_and_per_row_dmin(self):
        case=finite_case();case['distance'][0,1]=-4;case['exempt'][0,1]=True
        case['dmin'][1,3]=.125
        self.check_oracle(case)
        case['exempt'][0,1]=False;self.check_oracle(case)

    def test_all_exempt_or_empty_rows_abort(self):
        case=finite_case();case['exempt'][:]=True
        with self.assertRaises((ValueError,RuntimeError)):CANDIDATE.tracking.reserve_forecast(**adapt(case))
        case=finite_case(rows=0)
        with self.assertRaises((ValueError,RuntimeError)):CANDIDATE.tracking.reserve_forecast(**adapt(case))

    def test_final_row_of9021_is_not_dropped(self):
        x=adapt(finite_case(n=1,rows=9021),torch.float32)
        x['distance'][:]=1;x['distance'][0,-1]=-.25;x['dmin'][:]=0
        for v in x['jacobian'].values():v[:]=0
        z=CANDIDATE.tracking.reserve_forecast(**x)
        self.assertEqual(z['risk_margin'].item(),-.25)

    def test_nonfinite_inputs_including_exempt_entries_abort(self):
        for value in [float('nan'),float('inf'),float('-inf')]:
            for field in ['distance','dmin','q','qd','jacobian','pending','dt']:
                x=adapt(finite_case());x['exempt'][0,0]=True
                if field in ['distance','dmin']:x[field][0,0]=value
                elif field in ['q','qd']:x[field]['F_L'][0,0]=value
                elif field=='jacobian':x[field]['F'][0,0,0]=value
                elif field=='pending':x[field][2]['U_R'][0,0]=value
                else:x['dt']=value
                with self.subTest(value=value,field=field),self.assertRaises((ValueError,RuntimeError)):
                    CANDIDATE.tracking.reserve_forecast(**x)

    def test_finite_overflow_in_displacement_projection_and_margin_aborts(self):
        for location in ['displacement','projection','margin']:
            x=adapt(fixtures()['velocity_without_target_debt'],torch.float32)
            if location=='displacement':x['qd']['F_L'][0,0]=3e38;x['dt']=1.
            elif location=='projection':x['jacobian']['F'][0,0,0]=3e38;x['pending'][0]['F_L'][0,0]=3e38
            else:x['distance'][0,0]=-3e38;x['dmin'][0,0]=3e38
            with self.subTest(location=location),self.assertRaises((ValueError,RuntimeError)):
                CANDIDATE.tracking.reserve_forecast(**x)

    def test_malformed_batch_or_row_broadcast_is_rejected(self):
        for field in ['q','qd','jacobian','pending']:
            x=adapt(finite_case())
            if field in ['q','qd']:x[field]={a:v[:1] for a,v in x[field].items()}
            elif field=='jacobian':x[field]={a:v[:1] for a,v in x[field].items()}
            else:x[field]=[{a:v[:1] for a,v in t.items()} for t in x[field]]
            with self.subTest(field=field),self.assertRaises((ValueError,RuntimeError)):
                CANDIDATE.tracking.reserve_forecast(**x)
        x=adapt(finite_case());x['jacobian']={a:v[:,:1] for a,v in x['jacobian'].items()}
        with self.assertRaises((ValueError,RuntimeError)):CANDIDATE.tracking.reserve_forecast(**x)

    def test_missing_extra_components_and_queue_length_abort(self):
        for field in ['q','qd','jacobian','pending']:
            for change in ['missing','extra']:
                x=adapt(finite_case());part=x[field][0] if field=='pending' else x[field]
                if change=='missing':del part[next(iter(part))]
                else:part['unexpected']=torch.zeros(2,1)
                with self.subTest(field=field,change=change),self.assertRaises((ValueError,RuntimeError,KeyError)):
                    CANDIDATE.tracking.reserve_forecast(**x)
        for count in [0,5,7]:
            x=adapt(finite_case());x['pending']=[copy.deepcopy(x['q']) for _ in range(count)]
            with self.subTest(count=count),self.assertRaises((ValueError,RuntimeError)):
                CANDIDATE.tracking.reserve_forecast(**x)
        for dt in [0,-.01]:
            x=adapt(finite_case());x['dt']=dt
            with self.assertRaises((ValueError,RuntimeError)):CANDIDATE.tracking.reserve_forecast(**x)

    def test_actual_six_slot_fifo_first_new_delivery_transition7(self):
        from safeduo.eval.perturbations import TargetDelayQueue
        x=adapt(finite_case(n=1));queue=TargetDelayQueue(6);queue.reset(x['q'])
        fixed=[copy.deepcopy(t) for t in queue.pending]
        x['pending']=queue.pending
        first=CANDIDATE.tracking.reserve_forecast(**x)['risk_margin'].clone()
        delivered=[]
        for step in range(7):
            new={a:torch.full_like(v,100.+step) for a,v in x['q'].items()}
            if step==0:
                # Proposed target has no place in reserve inputs; no queue push yet.
                torch.testing.assert_close(CANDIDATE.tracking.reserve_forecast(**x)['risk_margin'],first,rtol=0,atol=0)
            delivered.append(queue.push(new))
        for step in range(6):tree_equal(self,delivered[step],fixed[step])
        for v in delivered[6].values():self.assertTrue((v==100.).all())

    def test_gap_modes_boundaries_and_ulp_neighbors(self):
        for dtype in [torch.float32,torch.float64]:
            edge=torch.tensor([.01,.05],dtype=dtype)
            values=torch.cat([torch.tensor([-1.,0.,.03,1.],dtype=dtype),edge,
                torch.nextafter(edge,torch.full_like(edge,float('-inf'))),torch.nextafter(edge,torch.full_like(edge,float('inf')))])
            for mode,want in [('joint_reference',.05),('tight_reference',.01)]:
                torch.testing.assert_close(CANDIDATE.tracking.gap_from_margin(values,mode),torch.full_like(values,want),rtol=0,atol=0)
            actual=CANDIDATE.tracking.gap_from_margin(values,'delay_reserve')
            expected=torch.tensor([float(adaptive_gap(v)) for v in values.tolist()],dtype=dtype)
            # Arithmetic rounding only: 8 elementary scalar operations at .05rad.
            torch.testing.assert_close(actual,expected,rtol=0,atol=8*torch.finfo(dtype).eps*.05)
            ordered=values.sort().values;result=CANDIDATE.tracking.gap_from_margin(ordered,'delay_reserve')
            self.assertTrue((result[1:]>=result[:-1]).all())
            self.assertTrue((result>=torch.tensor(.01,dtype=dtype)).all())
            self.assertTrue((result<=torch.tensor(.05,dtype=dtype)).all())

    def test_gap_invalid_mode_or_nonfinite_never_returns_safe(self):
        with self.assertRaises(ValueError):CANDIDATE.tracking.gap_from_margin(torch.tensor(1.),'other')
        for mode in CANDIDATE.tracking.MODES:
            for value in [float('nan'),float('inf'),float('-inf')]:
                with self.subTest(mode=mode,value=value),self.assertRaises(ValueError):
                    CANDIDATE.tracking.gap_from_margin(torch.tensor(value),mode)


class ReferenceContractTests(unittest.TestCase):
    def test_scalar_geometric_interval_oracle_and_soft_limits(self):
        cases=[(.25,0,.03125,.01,-2,2),(-.25,0,.03125,.01,-2,2),(0,0,.125,.0625,-2,2),
               (.125,0,.0625,.0625,-2,2),(.96875,.9375,.125,.05,-1,1),(-.96875,-.9375,.125,.01,-1,1)]
        rng=np.random.default_rng(617)
        cases += [(t/32,q/32,b/64,g/128,-1.,1.) for t,q,b,g in rng.integers([ -30,-40,1,1],[31,41,8,12],size=(80,4))]
        for target,q,box,gap,lower,upper in cases:
            lo,hi=CANDIDATE.reference.reference_bounds(*[torch.tensor([[v]],dtype=torch.float64) for v in [q,target,lower,upper]],box,gap)
            a,b,reachable=reference_interval_oracle(target,q,box,gap,lower,upper)
            self.assertLessEqual(abs(lo.item()-float(a)),4*np.finfo(float).eps*max(1,abs(target),abs(q)))
            self.assertLessEqual(abs(hi.item()-float(b)),4*np.finfo(float).eps*max(1,abs(target),abs(q)))
            self.assertTrue(-box<=lo<=hi<=box)
            if not reachable:self.assertTrue(torch.equal(lo,hi))

    def test_baseline050_matches_sealed_original_on_valid_inputs(self):
        old=H.parent/'safety_fifo_predictor_20261006_0630/reference_envelope.py'
        original=extract_function(old.read_text(),'reference_bounds',dict(torch=torch))
        q=torch.tensor([[0.,.2,-.2],[.5,-.5,1.]],dtype=torch.float64);target=q+torch.tensor([[.08,-.08,0.],[.2,0.,-.2]])
        limits=(torch.full_like(q,-2),torch.full_like(q,2))
        for gap in [.05,None]:
            old_bounds=original(q,target,*limits,.03125,gap)
            new_bounds=CANDIDATE.reference.reference_bounds(q,target,*limits,.03125,gap)
            for a,b in zip(old_bounds,new_bounds):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_perenv_gap_and_unreachable_does_not_snap(self):
        q=torch.zeros(2,7,dtype=torch.float64);target=torch.full_like(q,.03125)
        gap=torch.tensor([[.01],[.05]],dtype=q.dtype)
        lo,hi=CANDIDATE.reference.reference_bounds(q,target,torch.full_like(q,-2),torch.full_like(q,2),.015625,gap)
        self.assertTrue((lo[0]==-.015625).all() and torch.equal(lo[0],hi[0]))
        self.assertTrue((target[0]+hi[0]>.01).all())
        self.assertTrue((hi[1]==.015625).all())

    def test_nonfinite_state_limit_box_and_gap_abort(self):
        for key in ['q','target','lower','upper','box','gap']:
            for value in [float('nan'),float('inf'),float('-inf')]:
                x=dict(q=torch.zeros(2,2),target=torch.zeros(2,2),lower=torch.full((2,2),-1.),upper=torch.ones(2,2),box=.05,gap=.01)
                if key in ['box','gap']:x[key]=value
                else:x[key][0,0]=value
                with self.subTest(key=key,value=value),self.assertRaises((ValueError,RuntimeError)):
                    CANDIDATE.reference.reference_bounds(**x)

    def test_invalid_shapes_order_or_original_unreachable_abort(self):
        base=dict(q=torch.zeros(2,2),target=torch.zeros(2,2),lower=torch.full((2,2),-1.),upper=torch.ones(2,2),box=.05,gap=.01)
        for key in ['q','target','lower','upper']:
            x=copy.deepcopy(base);x[key]=x[key][:1]
            with self.subTest(key=key),self.assertRaises((ValueError,RuntimeError)):CANDIDATE.reference.reference_bounds(**x)
        for key,value in [('box',0),('box',-.01),('gap',0),('gap',-.01)]:
            x=copy.deepcopy(base);x[key]=value
            with self.assertRaises((ValueError,RuntimeError)):CANDIDATE.reference.reference_bounds(**x)
        x=copy.deepcopy(base);x['lower'][:]=2
        with self.assertRaises(ValueError):CANDIDATE.reference.reference_bounds(**x)
        x=copy.deepcopy(base);x['target'][:]=5
        with self.assertRaises(ValueError):CANDIDATE.reference.reference_bounds(**x)

    def test_real_install_dynamic_provider_limits_r19_passthrough(self):
        from safeduo.safety.types import DeltaCmd
        q={a:torch.zeros(2,d,dtype=torch.float64) for a,d in zip(ARMS,DOFS)}
        targets={a:torch.full_like(v,.03125) for a,v in q.items()};before=tensor_tree(targets)
        gap=torch.tensor([[.01],[.05]],dtype=torch.float64);captured=[]
        def original(cmd,rows,alpha,p,dt,**kwargs):
            captured.append((cmd,rows,alpha,p,dt,kwargs))
            return cmd,torch.zeros(2,4,dtype=torch.bool),{}
        env=SimpleNamespace(scene_state=lambda:SimpleNamespace(q=q),_targets=targets,
            _q_soft_limits={a:torch.stack([torch.full_like(v,-2),torch.full_like(v,2)],-1) for a,v in q.items()},
            _backstop=SimpleNamespace(project=original,cfg=SimpleNamespace(vmax=1.)))
        old=CANDIDATE.reference.install(env,'envelope_050',gap_provider=lambda:gap)
        self.assertIs(old,original)
        command=DeltaCmd(delta_q={a:torch.full_like(v,.125) for a,v in q.items()})
        rows=object();alpha=torch.ones(2,4);p=torch.zeros(2);bypass=torch.ones(2,4,dtype=torch.bool)
        result,active,info=env._backstop.project(command,rows,alpha,p,.015625,bypass_arm=bypass)
        self.assertIs(captured[0][1],rows);self.assertIs(captured[0][2],alpha);self.assertIs(captured[0][5]['bypass_arm'],bypass)
        for value in result.delta_q.values():
            self.assertTrue((value[0]==-.015625).all());self.assertTrue((value[1]==.015625).all())
        gap[:]=.05
        result,_,_=env._backstop.project(command,rows,alpha,p,.015625,bypass_arm=bypass)
        self.assertTrue(all((v==.015625).all() for v in result.delta_q.values()))
        tree_equal(self,targets,before)


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
        reserve_forecast=CANDIDATE.tracking.reserve_forecast,gap_from_margin=CANDIDATE.tracking.gap_from_margin,
        reference_bounds=CANDIDATE.reference.reference_bounds,abort=lambda error,stage:aborted.append((type(error).__name__,stage)))
    namespace['checked_rows']=extract_function(CANDIDATE.text['guard_runner.py'],'checked_rows',namespace)
    safety=extract_function(CANDIDATE.text['guard_runner.py'],'safety',namespace)
    return safety,trace,env,j,aborted,select


class GuardWiringTests(unittest.TestCase):
    def test_exact_guard_modes_have_identical_old_target_admission(self):
        masks=[]
        for mode in ['joint_reference','tight_reference','delay_reserve']:
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
                'actual tight/adaptive gap must determine archived unreachability')
            tree_equal(self,env._evaluation_actuator_delay.queue.pending,queue_before)
        self.assertTrue(all(torch.equal(masks[0],v) for v in masks[1:]))

    def test_current_proposal_changes_admission_but_not_prequeue_reserve(self):
        safety,trace,env,j,aborted,_=guard_fixture('delay_reserve',proposal=.125,target_value=0.)
        one=safety();risk1=trace.last_chunk['reserve_margin'].copy();gap1=trace.reserve_gap.clone()
        for v in env._pending_cmd.delta_q.values():v[:]=-.125
        two=safety()
        self.assertTrue(np.array_equal(risk1,trace.last_chunk['reserve_margin']))
        self.assertTrue(torch.equal(gap1,trace.reserve_gap))
        self.assertFalse(torch.equal(one.active_idx,two.active_idx),'fixture must expose real proposal admission sensitivity')

    def test_all9021_rows_retained_and_insufficient_budget_aborts(self):
        safety,trace,env,j,aborted,select=guard_fixture('delay_reserve',all_admitted=True)
        selected=safety()
        self.assertEqual(tuple(selected.active_idx.shape),(64,9021))
        self.assertTrue((selected.active_mask.sum(-1)==9021).all())
        with self.assertRaises(ValueError):
            select(env._last_out,env._last_out,torch.ones(64,9021,dtype=torch.bool),env._sph,capacity=9020)

    def test_hidden_nonfinite_last_row_aborts_before_projection(self):
        safety,trace,env,j,aborted,_=guard_fixture('delay_reserve')
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


def main():
    global CANDIDATE
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['oracle','candidate'],default='candidate')
    parser.add_argument('--revision',choices=['current','initial'],default='current')
    args=parser.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(IndependentOracleTests)
    if args.phase=='candidate':
        CANDIDATE=load_candidate(args.revision)
        for cls in [ReserveContractTests,ReferenceContractTests,GuardWiringTests]:
            suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if CANDIDATE is not None and args.revision=='current':
        for name,digest in CANDIDATE.source_sha256.items():
            if hashlib.sha256((H/name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('Candidate source changed during tests: '+name)
    if CANDIDATE is not None:print('TESTED_SOURCE_SHA256',json.dumps(CANDIDATE.source_sha256,sort_keys=True),flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':main()
