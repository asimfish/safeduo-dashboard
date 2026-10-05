"""Independent CPU captured-exit/LP/QP seam checks; no physics or parent tests.

Reads only the four cases already selected before the independent oracle LP.
Synthetic R19/alpha/failure fixtures are labelled separately from captured data.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

H = Path(__file__).resolve().parent
OLD = H.parent / 'safety_joint_guard_20261005_1005'
CELL = Path('/mnt/nas/data/lyf/double_hand/safety_joint_guard_20261005_1005/holdout/joint_guard_1701627244')
sys.path[:0] = [str(H), '/home/liyufeng/safeduo/src', str(OLD)]
import torch
import osqp
import joint_repair as candidate
from safeduo.safety.backstop import BackstopConfig, VelocityDamperBackstop
from safeduo.safety.types import ARM_KEYS, DeltaCmd
from safeduo.baselines.base import ConstraintRows
from projection_diagnostics import projection_record

torch.set_num_threads(1)
WIDTHS = [7, 7, 6, 6]
SLICES = {'F': slice(0, 14), 'U': slice(14, 26)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor(a):
    return torch.from_numpy(np.asarray(a).copy())


def arms(a):
    return dict(zip(ARM_KEYS, a.split(WIDTHS, -1)))


def raw_load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def positive(x):
    return float(np.maximum(x, 0).max(initial=0))


def actual_matrices(s, e, r):
    rel, ar = s[f'snapshot_rel_{r}'][e], s[f'snapshot_alpha_rel_{r}'][e]
    return (s[f'snapshot_G_{r}'][e, rel].astype(float),
            s[f'snapshot_h_after_authority_{r}'][e, rel].astype(float),
            s[f'snapshot_alpha_G_{r}'][e, ar].astype(float),
            s[f'snapshot_alpha_h_{r}'][e, ar].astype(float))


def replay_seam(s, e, cfg):
    """Original exit is a recorded tensor; this does not rerun original Dykstra."""
    bs = VelocityDamperBackstop(BackstopConfig(**cfg))
    rows = ConstraintRows(tensor(s['snapshot_d'][e:e+1]),
        {r: tensor(s[f'snapshot_J_{r}'][e:e+1]) for r in ('F', 'U')},
        tensor(s['snapshot_cls'][e:e+1]), tensor(s['snapshot_arm_mask'][e:e+1]),
        tensor(s['snapshot_valid'][e:e+1]), tensor(s['snapshot_dmin'][e:e+1]))
    cmd = DeltaCmd(arms(tensor(s['project_input_cmd'][e:e+1])))
    alpha, p = tensor(s['alpha'][e:e+1]), tensor(s['p'][e:e+1])
    bounds = dict(zip(ARM_KEYS, zip(tensor(s['bounds_lower'][e:e+1]).split(WIDTHS, -1),
                                  tensor(s['bounds_upper'][e:e+1]).split(WIDTHS, -1))))
    kwargs = dict(delta_bounds=bounds, bypass_arm=tensor(s['bypass_arm'][e:e+1]),
                  qd=arms(tensor(s['snapshot_qd'][e:e+1])),
                  backlog=arms(tensor(s['snapshot_backlog'][e:e+1])),
                  past_backlogs=[arms(tensor(v)[None]) for v in s['snapshot_past_backlogs'][e]],
                  struct_exempt=tensor(s['snapshot_struct_exempt'][e:e+1]),
                  contact_exempt=tensor(s['snapshot_contact_exempt'][e:e+1]))
    original = (DeltaCmd(arms(tensor(s['returned_cmd'][e:e+1]))),
                tensor(s['original_active'][e:e+1]),
                dict(cap=tensor(s['snapshot_cap'][e:e+1]),
                     **{f'{k}_{r}': tensor(s[f'original_{n}_{r}'][e:e+1])
                        for r in ('F', 'U') for k,n in [('residual','safety_residual'), ('passes','passes')]}))
    bs.project = lambda *a, **kw: original
    env = SimpleNamespace(_backstop=bs)
    candidate.install(env)
    before = projection_record(bs, cmd, rows, alpha, p, .016666, kwargs, original)
    initial = {k: v.clone() for k,v in dict(cmd=cmd.stacked(), d=rows.d, JF=rows.J['F'], JU=rows.J['U'], alpha=alpha, p=p).items()}
    result = bs.project(cmd, rows, alpha, p, .016666, **kwargs)
    after = projection_record(bs, cmd, rows, alpha, p, .016666, kwargs, result)
    current = dict(cmd=cmd.stacked(), d=rows.d, JF=rows.J['F'], JU=rows.J['U'], alpha=alpha, p=p)
    assert all(torch.equal(v, current[k]) for k,v in initial.items())
    values = {k: float(v[0]) for k,v in result[2].items() if k.startswith('repair_')}
    for r in ('F', 'U'):
        for k in ('safety', 'alpha', 'bound'):
            values[f'pre_{k}_{r}'] = float(before[f'returned_{k}_residual_{r}'][0])
            values[f'post_{k}_{r}'] = float(after[f'returned_{k}_residual_{r}'][0])
        values[f'original_field_post_{r}'] = float(after[f'original_safety_residual_{r}'][0])
    values['returned_float32'] = result[0].stacked()[0].tolist()
    return values


def main():
    assert not torch.cuda.is_available(), 'CPU-only test process must hide CUDA'
    plan_names = ['development_plan.json', 'holdout_0_plan.json', 'holdout_1_plan.json', 'holdout_2_plan.json']
    bound = {}
    for name in plan_names:
        path = H / name
        plan = json.loads(path.read_bytes())
        bound[str(path)] = sha(path)
        assert plan['sources_frozen'] and not plan['production_promoted']
        for file in ['joint_repair.py', 'guard_runner.py', 'reference_envelope.py', 'projection_diagnostics.py', 'full_finite_guard.py']:
            source = H / file
            assert sha(source) == plan['research_source_sha256'][str(source)]
            bound[str(source)] = sha(source)
    assert Path(candidate.__file__).resolve() == H / 'joint_repair.py'
    oracle = json.loads((H / 'ASTRA_JOINT_ORACLE.json').read_bytes())
    bound[str(H / 'ASTRA_JOINT_ORACLE.json')] = sha(H / 'ASTRA_JOINT_ORACLE.json')
    cfg = json.loads((CELL / 'protocol.json').read_bytes())['effective_backstop']
    outcomes = []
    for t,e,expected in [(71,3,'FEASIBLE'), (75,40,'FEASIBLE'), (480,31,'FEASIBLE'), (66,23,'INFEASIBLE')]:
        old = next(r for r in oracle['outcomes'] if r['choice']['step'] == t and r['choice']['env'] == e)
        assert old['full_constraints']['status'] == ('FEASIBLE' if expected == 'FEASIBLE' else 'INFEASIBLE_CERTIFIED')
        path = CELL / 'projection_snapshots' / f'step_{t:04d}.npz'
        assert sha(path) == oracle['input_sha256'][str(path)]
        bound[str(path)] = sha(path)
        s = raw_load(path)
        returned = s['returned_cmd'][e].astype(float).copy()
        records = []
        for r in ('F', 'U'):
            G,h,ag,ah = actual_matrices(s,e,r)
            sl = SLICES[r]
            pre = dict(safety=positive(G @ returned[sl] - h), alpha=positive(ag @ returned[sl] - ah),
                       bounds=positive(np.r_[s['bounds_lower'][e,sl]-returned[sl], returned[sl]-s['bounds_upper'][e,sl]]))
            if max(pre.values()) <= candidate.TRIGGER:
                continue
            u,meta = candidate.solve_joint(s['project_input_cmd'][e,sl], G,h,ag,ah,
                                          s['bounds_lower'][e,sl], s['bounds_upper'][e,sl])
            independent = dict(safety=positive(G@u-h), alpha=positive(ag@u-ah),
                               bounds=positive(np.r_[s['bounds_lower'][e,sl]-u,u-s['bounds_upper'][e,sl]]))
            assert independent['bounds'] == 0
            assert abs(independent['safety']-meta['safety_residual_m']) < 1e-15
            assert abs(independent['alpha']-meta['alpha_residual_rad']) < 1e-15
            if expected == 'FEASIBLE':
                assert meta['alpha_min_slack_rad'] == 0 and meta['safety_min_slack_m'] == 0
                assert max(independent.values()) <= candidate.CHECK_TOL
            else:
                assert meta['safety_min_slack_m'] > 0 and independent['safety'] > 0
            returned[sl] = u
            records.append(dict(robot=r, pre=pre, post=independent, meta=meta, repaired_double=u.tolist()))
        assert len(records) == 1
        native_dual = None
        if expected == 'INFEASIBLE':
            # Convert the independent normalized-row certificate into a native
            # metre minimax lower bound. Its active rows have no alpha terms.
            certificate = old['full_constraints']['certificate_rows']
            assert all(v['label']['kind'] == 'safety' and v['label']['robot'] == 'U' for v in certificate)
            weight_sum = sum(-v['dual'] / np.linalg.norm(s['snapshot_G_U'][e,v['label']['position']].astype(float))
                             for v in certificate)
            native_dual = old['full_constraints']['dual_lower_bound'] / weight_sum
            assert abs(records[0]['meta']['safety_min_slack_m'] - native_dual) < 1e-12
            assert abs(records[0]['post']['safety'] - native_dual) < 1e-12
        seam = replay_seam(s,e,cfg)
        r=records[0]['robot']
        assert seam[f'repair_applied_{r}'] == 1
        assert abs(seam[f'repair_pre_safety_residual_m_{r}'] - seam[f'pre_safety_{r}']) < 1e-12
        assert abs(seam[f'original_field_post_{r}'] - seam[f'post_safety_{r}']) < 1e-12
        if expected == 'FEASIBLE':
            assert max(seam[f'post_{k}_{r}'] for k in ('safety','alpha','bound')) <= candidate.CHECK_TOL
        else:
            assert seam[f'repair_safety_min_slack_m_{r}'] > 0 and seam[f'post_safety_{r}'] > 0
        outcomes.append(dict(step=t,env=e,status='PASS',expected_joint_set=expected,pure_solve=records,captured_exit_seam=seam,
                             independent_native_minimax_dual_lower_bound_m=native_dual))
    # Exact pass-through of a captured zero-residual exit; no original solver rerun.
    path = CELL / 'projection_snapshots/step_0480.npz'
    s = raw_load(path)
    zero = replay_seam(s,49,cfg)
    assert np.array_equal(np.asarray(zero['returned_float32'],dtype=np.float32),s['returned_cmd'][49])
    assert zero['repair_applied_F'] == zero['repair_applied_U'] == 0

    # Synthetic priority counterexample: hard alpha x<=0, safety x>=.02.
    u,lex = candidate.solve_joint([.02],[[-1]],[-.02],[[1]],[0],[-.025],[.025])
    assert lex['alpha_min_slack_rad']==0 and abs(lex['safety_min_slack_m']-(.02-candidate.SOLVE_ALLOWANCE))<1e-12
    assert lex['safety_residual_m']>.019999 and lex['alpha_residual_rad']<=candidate.CHECK_TOL

    # Synthetic forced reference endpoint, alpha itself cannot be met.
    u,single = candidate.solve_joint([.025],[[1]],[.0225],[[1]],[0],[.025],[.025])
    assert float(u[0])==.025 and abs(single['alpha_min_slack_rad']-.025)<1e-12 and abs(single['safety_min_slack_m']-.0025)<1e-12

    # Actual wrapper around CPU production backstop, synthetic R19 override.
    bs=VelocityDamperBackstop(BackstopConfig())
    cmd=DeltaCmd({a:torch.zeros(1,w) for a,w in zip(ARM_KEYS,WIDTHS)});cmd.delta_q['F_L'][0,0]=.01
    jf=torch.zeros(1,1,14);jf[0,0,0]=-1
    rows=ConstraintRows(torch.tensor([[.03]]),{'F':jf,'U':torch.zeros(1,1,12)},torch.tensor([[2]]),
                        torch.tensor([[[True,False,False,False]]]),torch.tensor([[True]]))
    alpha=torch.ones(1,4);p=torch.zeros(1);bypass=torch.tensor([[True,False,False,False]])
    bounds={a:(torch.full_like(v,-.025),torch.full_like(v,.025)) for a,v in cmd.delta_q.items()}
    old=bs.project(cmd,rows,alpha,p,.016666,delta_bounds=bounds,bypass_arm=bypass)
    candidate.install(SimpleNamespace(_backstop=bs))
    new=bs.project(cmd,rows,alpha,p,.016666,delta_bounds=bounds,bypass_arm=bypass)
    assert torch.equal(old[0].delta_q['F_L'],cmd.delta_q['F_L'])
    assert not torch.equal(new[0].delta_q['F_L'],cmd.delta_q['F_L']) and new[2]['repair_applied_F'][0]==1
    assert new[2]['bypass_arm'][0,0] and new[1][0,0]
    r19=dict(pre_returned=float(old[0].delta_q['F_L'][0,0]),post_returned=float(new[0].delta_q['F_L'][0,0]),
             bypass_flag_still_true=bool(new[2]['bypass_arm'][0,0]),active_post=bool(new[1][0,0]),
             pre_residual=float(new[2]['repair_pre_safety_residual_m_F'][0]),post_info_residual=float(new[2]['residual_F'][0]))

    # QP nonfinite failure must select and independently check the actual LP exit.
    class BadQP:
        def setup(self, **kwargs):
            self.d=len(kwargs['q'])
        def solve(self):
            return SimpleNamespace(info=SimpleNamespace(status_val=7),x=np.full(self.d,np.nan))
    s=raw_load(CELL/'projection_snapshots/step_0075.npz');G,h,ag,ah=actual_matrices(s,40,'U')
    with patch.object(candidate.osqp,'OSQP',BadQP):
        u,fallback=candidate.solve_joint(s['project_input_cmd'][40,14:],G,h,ag,ah,s['bounds_lower'][40,14:],s['bounds_upper'][40,14:])
    assert fallback['qp_fallback'] and positive(G@u-h)<=candidate.CHECK_TOL and positive(ag@u-ah)<=candidate.CHECK_TOL
    assert positive(np.r_[s['bounds_lower'][40,14:]-u,u-s['bounds_upper'][40,14:]])==0

    for path,expected in bound.items():
        assert sha(path)==expected, f'input/source mutation during review: {path}'
    output=dict(status='PASS_BOUNDED_CANDIDATE_CPU_CHECKS',oracle_test_source_sha256=sha(__file__),
                source_input_sha256=bound,torch_version=torch.__version__,cuda_available=False,
                osqp_version=osqp.__version__,captured_cases=outcomes,
                exact_pass_through_captured_zero='480/49 PASS',synthetic_alpha_priority=lex,
                synthetic_unavoidable_alpha=single,synthetic_R19=r19,forced_QP_failure=fallback,
                safety_strategy_approved=False,hardware_authorized=False,
                scope='Four previously frozen captured cases plus captured-zero exit; synthetic fixtures distinct; CPU recorded-exit seam is not a physics replay.')
    print(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
