"""Read frozen old trajectories; fit once and score a separately named old cohort."""
from pathlib import Path
import hashlib,json
import numpy as np
from predictors import nonnegative_fit,NAMES
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in ('q','q_initial','pre_qd_compact','actuator_target','controller_target')}
def transitions(d):
    pre=np.concatenate([d['q_initial'][None],d['q'][:-1]]).astype(float)
    assert np.array_equal(d['actuator_target'],np.concatenate([np.repeat(d['q_initial'][None],6,axis=0),d['controller_target'][:-6]]))
    return pre,d['pre_qd_compact'].astype(float),d['actuator_target'].astype(float),d['q'].astype(float)-pre
def main():
    plan=json.loads((HERE/'CALIBRATION.json').read_text());dt=plan['dt']
    for name,h in plan['sources'].items():assert sha(name)==h
    gram=np.zeros((26,2,2));xy=np.zeros((26,2));yy=np.zeros(26);n=0
    for f,h in plan['training'].items():
        assert sha(f)==h
        q,v,t,y=transitions(load(f));x=np.stack([v*dt,t-q],-1)
        gram+=np.einsum('tnji,tnjk->jik',x,x);xy+=np.einsum('tnji,tnj->ji',x,y);yy+=(y*y).sum((0,1));n+=y.shape[0]*y.shape[1]
        assert sha(f)==h
    c=np.stack([nonnegative_fit(gram[j],xy[j],yy[j]) for j in range(26)])
    model=dict(schema='safeduo.one_step_pd.v1',dt=dt,coefficients=c.tolist(),training_transitions=n,
        calibration_sha256=sha(HERE/'CALIBRATION.json'),formula='a*qd*dt+b*(oldest_actual_fifo_target-q)',control_enabled=False)
    with (HERE/'model.json').open('x') as f:json.dump(model,f,indent=2);f.write('\n')
    out={}
    for phase in ('training','historical_external'):
        absum=np.zeros((4,26));sqsum=absum.copy();maximum=absum.copy();count=0
        for file,h in plan[phase].items():
            assert sha(file)==h
            q,v,t,y=transitions(load(file));pred=[np.zeros_like(q),t-q,v*dt,c[:,0]*(v*dt)+c[:,1]*(t-q)]
            for i,p in enumerate(pred):
                e=p-y;absum[i]+=np.abs(e).sum((0,1));sqsum[i]+=(e*e).sum((0,1));maximum[i]=np.maximum(maximum[i],np.abs(e).max((0,1)))
            count+=y.shape[0]*y.shape[1];assert sha(file)==h
        out[phase]={name:dict(transitions=count,mae_rad=float((absum[i]/count).mean()),rmse_rad=float(np.sqrt(sqsum[i].sum()/(count*26))),max_rad=float(maximum[i].max()),per_joint_mae_rad=(absum[i]/count).tolist()) for i,name in enumerate(NAMES)}
    with (HERE/'calibration_results.json').open('x') as f:json.dump(dict(status='complete',results=out,model_sha256=sha(HERE/'model.json'),historical_outcomes_already_known=True,new_physical_windows=0),f,indent=2);f.write('\n')
    print(json.dumps({phase:{name:{k:r[k] for k in ('mae_rad','rmse_rad','max_rad')} for name,r in rows.items()} for phase,rows in out.items()}))
if __name__=='__main__':main()
