"""Frozen streaming metrics; strict negative geometry and mask changes retained."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from predictors import Predictor,NAMES
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def miss_flags(pred,actual,pre_ex,post_ex):
    miss=(pred>=0)&(actual<0)&~post_ex
    return miss,miss&~pre_ex
def row_classes(identity):
    original=np.array(identity['class_id']);cls=original.copy()
    pair=np.array(identity['pair_sphere_idx']);arm=np.array(identity['sphere_arm_id'])
    cls[original==2]=3
    cls[(original==1)&(arm[pair[:,0]]>=2)]=2
    return cls
class Metric:
    def __init__(self):self.n=0;self.abs=0.;self.sq=0.;self.signed=0.;self.maximum=0.;self.hist=np.zeros(2002,np.int64)
    def add(self,error):
        e=np.asarray(error,np.float64).ravel();a=np.abs(e)
        self.n+=len(e);self.abs+=a.sum();self.sq+=(e*e).sum();self.signed+=e.sum();self.maximum=max(self.maximum,float(a.max(initial=0)))
        bins=np.minimum(np.floor(a/.0001),2001).astype(np.int32)
        self.hist+=np.bincount(bins,minlength=2002)
    def result(self):
        def quantile(p):
            i=int(np.searchsorted(self.hist.cumsum(),np.ceil(self.n*p)))
            return None if i==2001 else (i+1)*.0001
        return dict(values=self.n,mae_m=self.abs/self.n if self.n else None,
                    rmse_m=np.sqrt(self.sq/self.n) if self.n else None,
                    signed_mean_m=self.signed/self.n if self.n else None,max_m=self.maximum,
                    p95_upper_m=quantile(.95) if self.n else None,p99_upper_m=quantile(.99) if self.n else None,
                    histogram_step_m=.0001,histogram_overflow=int(self.hist[-1]))
def inspect(root,job):
    path=root/job['id'];p=json.loads((path/'protocol.json').read_text());assert p['status']=='complete'
    z=load(path/'cell_001.npz');recipe=load(path/'regime_recipe.npz');inp=load(path/'input_recipe.npz')
    assert z['q'].shape==(960,64,26) and z['predicted_delta_q'].shape==(960,64,4,26)
    for k,v in z.items():
        if v.dtype.kind=='f':assert np.isfinite(v).all(),k
    qprev=np.r_[z['q_initial'][None],z['q'][:-1]]
    delivered=np.r_[np.repeat(z['q_initial'][None],6,0),z['controller_target'][:-6]]
    assert np.array_equal(delivered,z['actuator_target']) and np.array_equal(delivered,z['delivered_model_target'])
    assert np.array_equal(qprev,z['passive_model_q']) and np.array_equal(z['pre_qd_compact'],z['passive_model_qd'])
    model=json.loads((HERE/'model.json').read_text())
    for i,name in enumerate(NAMES):
        out=Predictor(name,model['coefficients']).predict(torch.from_numpy(qprev),torch.from_numpy(z['pre_qd_compact']),torch.from_numpy(delivered),p['dt']).numpy()
        # CPU/GPU elementary float32 rounding check, fixed before outcomes.
        np.testing.assert_allclose(out,z['predicted_delta_q'][:,:,i],rtol=0,atol=2e-7)
    dq=z['q']-qprev;qe=z['predicted_delta_q']-dq[:,:,None]
    joint=[dict(model=name,mae_rad=float(np.abs(qe[:,:,i].astype(float)).mean()),rmse_rad=float(np.sqrt(np.square(qe[:,:,i].astype(float)).mean())),max_rad=float(np.abs(qe[:,:,i]).max())) for i,name in enumerate(NAMES)]
    cls=row_classes(json.loads((path/'full_row_identity.json').read_text()))
    metrics=[[Metric(),Metric()] for _ in NAMES];geom=Metric();miss_windows=np.zeros((4,64),bool);stable_windows=miss_windows.copy()
    regime_metrics=[[Metric() for _ in range(3)] for _ in NAMES]
    regime_steps=np.zeros((4,3),np.int64);regime_bad=np.zeros((3,64),bool)
    actual_steps=0;negative_rows=0;stable_negative_rows=0
    miss_steps=np.zeros(4,np.int64);miss_rows=miss_steps.copy();stable_steps=miss_steps.copy();newmask_steps=0;end=0
    receipt=json.loads((path/'prediction_receipts.json').read_text());assert receipt['steps']==960 and len(receipt['chunks'])==30
    for r in receipt['chunks']:
        f=path/r['path'];assert sha(f)==r['sha256'];c=load(f);start,stop=r['start'],r['stop'];assert start==end
        pred=c['predicted'];actual=c['post_d'];assert pred.shape==(stop-start,64,9021,4)
        assert np.isfinite(pred).all() and np.isfinite(actual).all()
        pre_ex=np.unpackbits(c['pre_exempt'],axis=-1)[...,:9021].astype(bool);post_ex=np.unpackbits(c['post_exempt'],axis=-1)[...,:9021].astype(bool)
        mask=~post_ex;near=mask&(c['pre_d']<=.080)
        regime=recipe['regime'][start:stop]
        actual_bad=(actual<0)&mask
        actual_steps+=int(actual_bad.any(-1).sum());negative_rows+=int(actual_bad.sum())
        stable_negative_rows+=int((actual_bad&~pre_ex).sum())
        for g in range(3):regime_bad[g]|=(actual_bad.any(-1)&(regime==g)).any(0)
        actual_margin=np.stack([np.where(post_ex|(cls!=k),np.inf,actual).min(-1) for k in range(4)],-1)
        assert np.array_equal(actual_margin,z['official_margins'][start:stop])
        newmask_steps+=int((pre_ex&~post_ex).any(-1).sum())
        for i,name in enumerate(NAMES):
            error=pred[...,i].astype(float)-actual.astype(float)
            metrics[i][0].add(error[mask]);metrics[i][1].add(error[near])
            miss,stable=miss_flags(pred[...,i],actual,pre_ex,post_ex)
            miss_rows[i]+=miss.sum();steps=miss.any(-1);ss=stable.any(-1)
            miss_steps[i]+=steps.sum();stable_steps[i]+=ss.sum();miss_windows[i]|=steps.any(0);stable_windows[i]|=ss.any(0)
            for g in range(3):
                regime_metrics[i][g].add(error[near&(regime==g)[...,None]])
                regime_steps[i,g]+=int((steps&(regime==g)).sum())
        residual=actual.astype(float)-c['pre_d'].astype(float)-c['hindsight_controlled_delta'].astype(float)-c['passive_rate_delta'].astype(float)
        geom.add(residual[near]);assert sha(f)==r['sha256'];end=stop
    assert end==960
    bad=(z['official_margins']<0).any((0,2));deep=(z['official_margins']<-.005).any((0,2))
    rows=[dict(model=name,all_nonexempt=metrics[i][0].result(),near_pre80mm=metrics[i][1].result(),
               missed_negative_rows=int(miss_rows[i]),missed_env_steps=int(miss_steps[i]),missed_windows=int(miss_windows[i].sum()),
               stable_mask_missed_env_steps=int(stable_steps[i]),stable_mask_missed_windows=int(stable_windows[i].sum()),
               missed_case_ids=np.flatnonzero(miss_windows[i]).tolist(),
               stable_mask_missed_case_ids=np.flatnonzero(stable_windows[i]).tolist(),
               regimes=[dict(regime=g,near_pre80mm=regime_metrics[i][g].result(),missed_env_steps=int(regime_steps[i,g])) for g in range(3)]) for i,name in enumerate(NAMES)]
    return dict(id=job['id'],mode=job['mode'],windows=64,violations=int(bad.sum()),deep=int(deep.sum()),
                predictions=rows,joint_errors=joint,newly_nonexempt_env_steps=newmask_steps,
                hindsight_near_residual=geom.result(),actual_fifo_exact=True,online_model_input_exact=True,
                actual_negative_rows=negative_rows,stable_mask_actual_negative_rows=stable_negative_rows,
                actual_negative_env_steps=actual_steps,bad_case_ids=np.flatnonzero(bad).tolist(),
                regime_bad_windows=[int(x.sum()) for x in regime_bad],
                dense_sha256=sha(path/'cell_001.npz')),inp
def main():
    plans=[json.loads(f.read_text()) for f in sorted((HERE/'registered').glob('block*.json'))];results=[];inputs={}
    for plan in plans:
        for path,h in plan['research_source_sha256'].items():assert sha(path)==h,path
        for rel,h in plan['source_sha256'].items():assert sha(Path(plan['cwd'])/rel)==h
        root=Path(plan['output_root']);campaign=json.loads((root/'campaign.json').read_text());assert campaign['status']=='complete'
        for job in plan['jobs']:
            r,inp=inspect(root,job);results.append(r);inputs[job['id']]=inp;print('AUDITED',job['id'],r['violations'],flush=True)
        first,second=plan['jobs'];a,b=inputs[first['id']],inputs[second['id']]
        assert np.array_equal(a['q_initial'],b['q_initial']) and np.array_equal(a['tape'],b['tape'])
    out=dict(status='complete',method_windows=384,new_command_tapes=192,new_initials=0,rows=results,
             model_sha256=sha(HERE/'model.json'),control_enabled=False,
             scope='prospective passive point predictions; no conservative or physical-safety certification; joint/static-mask and full-mask observations separated')
    (HERE/'results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print('COMPLETE',out['method_windows'])
if __name__=='__main__':main()
