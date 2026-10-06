"""Causal six-step forecast error against this run's own actual future geometry.

First six applied targets are known at forecast time. Endpoint actual geometry
is pre[t+6]=post[t+5]; first 954 endpoints available. No optimism epsilon.
"""
from pathlib import Path
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def inspect(path):
    receipt=json.loads((path/'forecast_receipts.json').read_text())
    data=load(path/'cell_001.npz')
    sums={k:dict(n=0,sq=0.,max_abs_m=0.,optimistic_raw_negative_row_endpoints=0,
                actual_raw_negative_row_endpoints=0,predictor_negative_row_endpoints=0) for k in ('cv','pd')}
    joint={k:dict(n=0,sq=0.,max_abs_rad=0.) for k in ('cv','pd')}
    chunks=receipt['chunks'];previous=None
    checked=0;bindings={str(path/'cell_001.npz'):sha(path/'cell_001.npz')}
    for i,c in enumerate(chunks):
        current=previous if previous is not None else load(path/c['path'])
        nextz=load(path/chunks[i+1]['path']) if i+1<len(chunks) else None
        file=path/c['path'];assert sha(file)==c['sha256'];bindings[str(file)]=c['sha256']
        combined=np.concatenate([current['measured_d'],nextz['measured_d'][:6]],0) if nextz is not None else current['measured_d']
        masks=np.concatenate([current['exempt'],nextz['exempt'][:6]],0) if nextz is not None else current['exempt']
        length=min(c['stop']-c['start'],954-c['start'])
        if length<=0:previous=nextz;continue
        actual=combined[6:6+length]
        exempt=np.unpackbits(masks[6:6+length],axis=-1)[...,:9021].astype(bool)
        bad=(actual<0)&~exempt
        for k in ('cv','pd'):
            predicted=current[k+'_h6'][:length]
            error=(predicted-actual).astype(np.float64)
            stat=sums[k];stat['n']+=error.size;stat['sq']+=float(np.square(error).sum())
            stat['max_abs_m']=max(stat['max_abs_m'],float(np.abs(error).max()))
            stat['optimistic_raw_negative_row_endpoints']+=int((bad&(predicted>=0)).sum())
            stat['actual_raw_negative_row_endpoints']+=int(bad.sum())
            stat['predictor_negative_row_endpoints']+=int(((predicted<0)&~exempt).sum())
            p=current[k+'_q_horizons'][:length,1]
            q=data['q'][c['start']+5:c['start']+5+length]
            err=(p-q).astype(np.float64);stat=joint[k]
            stat['n']+=err.size;stat['sq']+=float(np.square(err).sum())
            stat['max_abs_rad']=max(stat['max_abs_rad'],float(np.abs(err).max()))
        checked+=length;previous=nextz
    assert checked==954
    assert all(sha(Path(p))==v for p,v in bindings.items())
    for stat in sums.values():stat['rms_m']=(stat.pop('sq')/stat['n'])**.5
    for stat in joint.values():stat['rms_rad']=(stat.pop('sq')/stat['n'])**.5
    return dict(path=str(path),actual_steps=954,geometry=sums,joints=joint,input_sha256=bindings,
                scope='own executed trajectory; frozen-J and empirical dynamics errors; first6targets causally known; no futurepolicyactions used')

if __name__=='__main__':
    rows=[]
    for p in sorted((HERE/'plans').glob('holdout_*_plan.json')):
        plan=json.loads(p.read_text());campaign=json.loads((Path(plan['output_root'])/'campaign.json').read_text())
        assert campaign['status']=='complete'
        for job in plan['jobs']:
            r=inspect(Path(plan['output_root'])/job['id']);r.update(mode=job['env']['SAFEDUO_JOINT_MODE'],id=job['id'])
            rows.append(r);print('H6_AUDITED',job['id'],r['geometry']['pd']['optimistic_raw_negative_row_endpoints'],flush=True)
    with (HERE/'H6_PREDICTION_AUDIT.json').open('x') as f:json.dump(dict(status='COMPLETE_OWN_TRAJECTORY_AUDIT',rows=rows,physical_safety_certified=False),f,indent=2);f.write('\n')
