"""Search bounded nominal commands, then independently admit actual float32 targets."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch
from scipy.optimize import linprog
from full_row_model_v1 import construct,ARMS
sys.path.insert(0,str(Path(__file__).resolve().parent/'astra'))
from runtime_action_admission import ModelConstraints,LinearRows,Candidate,admit_action,fifo_digest
def quantize_inside(reference,delta,lower,upper):
    centre=np.asarray(reference,dtype=np.float64)
    low=centre+np.asarray(lower,dtype=np.float64);high=centre+np.asarray(upper,dtype=np.float64)
    target=(centre+np.asarray(delta,dtype=np.float64)).astype(np.float32)
    for _ in range(8):
        target=np.where(target.astype(float)<low,np.nextafter(target,np.float32(np.inf)),target)
        target=np.where(target.astype(float)>high,np.nextafter(target,np.float32(-np.inf)),target)
    return target
def project_goal(A,b,lower,upper,goal,iterations):
    norm=A.square().sum(-1).sqrt();unit=A/norm.clamp_min(1e-12)[...,None];rhs=b/norm.clamp_min(1e-12)
    delta=goal.clamp(lower,upper);lanes=torch.arange(len(A),device=A.device)
    for _ in range(iterations):
        violation=rhs-torch.einsum('nmd,nd->nm',unit,delta)
        worst=violation.argmax(-1);amount=violation[lanes,worst].clamp_min(0)
        delta=(delta+unit[lanes,worst]*amount[:,None]).clamp(lower,upper)
    return delta
def lp_l1(A,b,lower,upper,goal,tol):
    if not all(np.isfinite(v).all() for v in [A,b,lower,upper,goal,tol]) or (lower>upper).any():return None
    # Rows omitted from search are algebraically satisfied throughout the box.
    # ALL original rows still reach the independent admission gate afterward.
    minimum=np.sum(A*np.where(A>=0,lower,upper),-1)
    keep=b-minimum>tol*.5
    margin=np.minimum(1e-6,(upper-lower)/4);lo=lower+margin;hi=upper-margin
    G=A[keep];rhs=b[keep]-tol[keep]*.5
    if len(G) and (rhs-np.sum(G*np.where(G>=0,hi,lo),-1)>0).any():return None
    I=np.eye(26);constraint=np.r_[np.c_[-G,np.zeros((len(G),26))],np.c_[I,-I],np.c_[-I,-I]]
    bounds=np.r_[-rhs,goal,-goal]
    result=linprog(np.r_[np.zeros(26),np.ones(26)],A_ub=constraint,b_ub=bounds,bounds=list(zip(lo,hi))+[(0,None)]*26,method='highs')
    return result.x[:26] if result.success else None
def make_issue(env,provider,data,queue,requested,settings,step):
    centre={a:queue.pending[-1][a].clone() for a in ARMS}
    m=construct(env,provider,data,centre,settings)
    goal=torch.cat([requested[a]-centre[a] for a in ARMS],-1)
    delta=project_goal(m['A'],m['b'],m['lower'],m['upper'],goal,settings['projection_passes'])
    cpu={k:m[k].detach().cpu().numpy() for k in ['A','b','lower','upper','tol','reference']}
    proposed=quantize_inside(cpu['reference'],delta.cpu().numpy(),cpu['lower'],cpu['upper'])
    hold=cpu['reference'].astype(np.float32)
    fifo=np.stack([np.concatenate([v[a].cpu().numpy() for a in ARMS],-1) for v in queue.pending])
    raw=data.dists.cpu().numpy();chosen=hold.copy();code=np.zeros(64,np.int8);model_pass=np.zeros(64,bool);lp_attempted=np.zeros(64,bool);no_action=np.zeros(64,bool)
    residual=np.zeros((64,3),np.float64);candidate_pass=np.zeros((64,3),bool)
    for lane in range(64):
        ctxhash=hashlib.sha256((settings['context_id']+':'+str(step)+':'+str(lane)+':'+fifo_digest(fifo[:,lane])).encode())
        for key in ['A','b','reference','lower','upper']:ctxhash.update(cpu[key][lane].tobytes(order='C'))
        ctx=ctxhash.hexdigest()
        model=ModelConstraints(ctx,fifo_digest(fifo[:,lane]),cpu['reference'][lane],cpu['lower'][lane],cpu['upper'][lane],
            LinearRows(cpu['A'][lane,:9021],cpu['b'][lane,:9021],tuple(range(9021))),
            LinearRows(cpu['A'][lane,9021:9169],cpu['b'][lane,9021:9169],m['velocity_ids']),
            LinearRows(cpu['A'][lane,9169:],cpu['b'][lane,9169:],m['effort_ids']),m['controlled_ids'],m['native_ids'])
        candidates=[Candidate('projected_goal',proposed[lane],ctx),Candidate('queue_tail_hold',hold[lane],ctx)]
        result=admit_action(model,candidates,context_id=ctx,fifo6=fifo[:,lane],raw_geometry_gaps=raw[lane],measurements={})
        if not result['evaluations'][0]['full_model_constraints_satisfied']:
            lp_attempted[lane]=True
            answer=lp_l1(cpu['A'][lane].astype(float),cpu['b'][lane].astype(float),cpu['lower'][lane].astype(float),cpu['upper'][lane].astype(float),goal[lane].cpu().numpy().astype(float),cpu['tol'].astype(float))
            if answer is not None:
                target=quantize_inside(cpu['reference'][lane],answer,cpu['lower'][lane],cpu['upper'][lane])
                candidates=[candidates[0],Candidate('lp_goal',target,ctx),candidates[1]]
                result=admit_action(model,candidates,context_id=ctx,fifo6=fifo[:,lane],raw_geometry_gaps=raw[lane],measurements={})
        if result['status']=='MODEL_FULL_ROW_FEASIBLE' and result['full_model_constraints_satisfied']:
            chosen[lane]=result['selected_target'];model_pass[lane]=True;code[lane]={'projected_goal':1,'lp_goal':2,'queue_tail_hold':3}[result['selected_name']]
        else:no_action[lane]=True
        for ev in result['evaluations']:
            slot={'projected_goal':0,'lp_goal':1,'queue_tail_hold':2}[ev['name']];candidate_pass[lane,slot]=ev['full_model_constraints_satisfied']
        actual_delta=chosen[lane].astype(float)-cpu['reference'][lane].astype(float)
        rr=cpu['b'][lane].astype(float)-cpu['A'][lane].astype(float)@actual_delta
        residual[lane]=[max(0,float(rr[:9021].max())),max(0,float(rr[9021:9169].max())),max(0,float(rr[9169:].max()))]
    assert np.array_equal(fifo,np.stack([np.concatenate([v[a].cpu().numpy() for a in ARMS],-1) for v in queue.pending]))
    output={a:torch.as_tensor(chosen[:,off:off+width],device=env.device) for a,(off,width) in m['offsets'].items()}
    records={k:cpu[k] for k in ['A','b','lower','upper','reference']}
    records.update(quantized_projected_target=proposed,quantized_hold_target=hold,actual_issued_target=chosen,full_model_pass=model_pass,candidate_pass=candidate_pass,decision_code=code,no_model_action=no_action,lp_attempted=lp_attempted)
    info=dict(model_constraints_satisfied=model_pass,geometry_velocity_residual=residual[:,0],full_native_velocity_limit_residual=residual[:,1],full_drive_effort_residual=residual[:,2],no_model_feasible_action=no_action,decision_code=code,candidate_pass=candidate_pass,lp_attempted=lp_attempted)
    return output,info,records
