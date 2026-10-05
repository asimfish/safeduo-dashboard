"""Evaluation-only residual-triggered joint solve. Positive slack is NOT safety.

Production rows, cap, authority clamp, row gates and target bounds remain intact.
Hard bounds first; minimum unavoidable alpha slack second; minimum maximum safety
residual third; closest command QP last. Actual violations use original geometry.
No simulator/target/queue writes; the returned delta still enters the real FIFO.
"""
import numpy as np
import torch
from scipy.optimize import linprog
from scipy import sparse
import osqp
from safeduo.safety.types import ARM_KEYS, DeltaCmd
from projection_diagnostics import projection_record, require_finite_tree

TRIGGER = 1e-6
SOLVE_ALLOWANCE = 2e-8
CHECK_TOL = 2e-7


def solve_joint(c, G, h, aG, ah, lower, upper):
    """Pure float64 LP oracle and QP tie breaker, with separately typed units."""
    c, G, h, aG, ah, lower, upper = [np.asarray(x, dtype=np.float64) for x in (c,G,h,aG,ah,lower,upper)]
    if any(not np.isfinite(x).all() for x in (c,G,h,aG,ah,lower,upper)) or (lower>upper).any():
        raise ValueError('invalid joint solve input')
    d=len(c); bounds=list(zip(lower,upper))
    opts=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9)
    # Alpha has units rad/step, safety rows metres; never add these quantities.
    if len(ah):
        A=np.column_stack([aG,-np.ones(len(ah))])
        alpha_lp=linprog(np.r_[np.zeros(d),1.],A_ub=A,b_ub=ah,bounds=bounds+[(0,None)],method='highs',options=opts)
        if not alpha_lp.success:raise ValueError('alpha phase LP failed: '+alpha_lp.message)
        alpha_slack=max(0.,float(alpha_lp.x[-1]))
    else:alpha_slack=0.
    alpha_limit=ah+alpha_slack+SOLVE_ALLOWANCE
    if len(h):
        A=np.vstack([np.column_stack([G,-np.ones(len(h))]),np.column_stack([aG,np.zeros(len(ah))])])
        b=np.r_[h,alpha_limit]
        lp=linprog(np.r_[np.zeros(d),1.],A_ub=A,b_ub=b,bounds=bounds+[(0,None)],method='highs',options=opts)
        if not lp.success:raise ValueError('safety phase LP failed: '+lp.message)
        safety_slack=max(0.,float(lp.x[-1])); lp_u=lp.x[:d]
    else:
        lp=linprog(np.zeros(d),A_ub=aG if len(ah) else None,b_ub=alpha_limit if len(ah) else None,bounds=bounds,method='highs',options=opts)
        if not lp.success:raise ValueError('empty safety phase LP failed')
        safety_slack=0.;lp_u=lp.x
    A=sparse.csc_matrix(np.vstack([np.eye(d),aG,G]))
    low=np.r_[lower,np.full(len(ah)+len(h),-np.inf)]
    high=np.r_[upper,alpha_limit,h+safety_slack+SOLVE_ALLOWANCE]
    qp=osqp.OSQP()
    qp.setup(P=sparse.eye(d,format='csc'),q=-c,A=A,l=low,u=high,verbose=False,
             eps_abs=1e-8,eps_rel=1e-8,max_iter=4000,polish=True,adaptive_rho=True)
    result=qp.solve()
    qp_ok=result.info.status_val in (1,2) and result.x is not None and np.isfinite(result.x).all()
    u=np.clip(result.x if qp_ok else lp_u,lower,upper)
    sr=float(np.maximum(G@u-h,0).max(initial=0.))
    ar=float(np.maximum(aG@u-ah,0).max(initial=0.))
    if sr>safety_slack+CHECK_TOL or ar>alpha_slack+CHECK_TOL:
        # Retain a certified LP feasible point if QP termination is insufficient.
        u=np.clip(lp_u,lower,upper);qp_ok=False
        sr=float(np.maximum(G@u-h,0).max(initial=0.));ar=float(np.maximum(aG@u-ah,0).max(initial=0.))
    if sr>safety_slack+CHECK_TOL or ar>alpha_slack+CHECK_TOL:raise ValueError('joint result exceeds declared unavoidable slack')
    return u,dict(alpha_min_slack_rad=alpha_slack,safety_min_slack_m=safety_slack,
                  safety_residual_m=sr,alpha_residual_rad=ar,qp_fallback=not qp_ok)


def install(env):
    original=env._backstop.project
    def project(cmd,rows,alpha,p,dt,**kwargs):
        result=original(cmd,rows,alpha,p,dt,**kwargs)
        record=projection_record(env._backstop,cmd,rows,alpha,p,dt,kwargs,result)
        masks={r:torch.stack([record[f'returned_{k}_residual_{r}'] for k in ('safety','alpha','bound')]).amax(0)>TRIGGER for r in ('F','U')}
        n=rows.d.shape[0];dev=rows.d.device
        meta={f'repair_{key}_{r}':torch.zeros(n,device=dev,dtype=rows.d.dtype) for r in ('F','U')
              for key in ('applied','alpha_min_slack_rad','safety_min_slack_m','qp_fallback','pre_safety_residual_m')}
        if any(mask.any().item() for mask in masks.values()):
            detail=projection_record(env._backstop,cmd,rows,alpha,p,dt,kwargs,result,snapshot=True)
            require_finite_tree('joint_repair_snapshot',detail)
            stacked=result[0].stacked().clone()
            for r,sl in [('F',slice(0,14)),('U',slice(14,26))]:
                for e in torch.where(masks[r])[0].cpu().tolist():
                    rel=detail[f'snapshot_rel_{r}'][e].cpu().numpy()
                    arel=detail[f'snapshot_alpha_rel_{r}'][e].cpu().numpy()
                    def arr(k):return detail[k][e].cpu().numpy()
                    u,m=solve_joint(arr('project_input_cmd')[sl],arr(f'snapshot_G_{r}')[rel],arr(f'snapshot_h_after_authority_{r}')[rel],
                        arr(f'snapshot_alpha_G_{r}')[arel],arr(f'snapshot_alpha_h_{r}')[arel],arr('bounds_lower')[sl],arr('bounds_upper')[sl])
                    stacked[e,sl]=torch.as_tensor(u,device=dev,dtype=stacked.dtype)
                    meta[f'repair_applied_{r}'][e]=1
                    meta[f'repair_pre_safety_residual_m_{r}'][e]=record[f'returned_safety_residual_{r}'][e]
                    for key in ('alpha_min_slack_rad','safety_min_slack_m','qp_fallback'):meta[f'repair_{key}_{r}'][e]=m[key]
            output=DeltaCmd(dict(zip(ARM_KEYS,stacked.split([7,7,6,6],-1))))
            changed=torch.stack([(output.delta_q[a]-result[0].delta_q[a]).abs().amax(-1)>1e-6 for a in ARM_KEYS],-1)
            info={**result[2],**meta}
            result=(output,result[1]|changed,info)
            final=projection_record(env._backstop,cmd,rows,alpha,p,dt,kwargs,result)
            for r in ('F','U'):info['residual_'+r]=final[f'returned_safety_residual_{r}']
        else:result=(result[0],result[1],{**result[2],**meta})
        require_finite_tree('joint_repair_output',dict(cmd=result[0].delta_q,info=result[2]))
        return result
    env._backstop.project=project
    return original
