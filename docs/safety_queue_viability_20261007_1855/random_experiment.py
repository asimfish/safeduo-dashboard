"""Registered fresh synthetic linear tests, explicitly not robot trajectories."""
from pathlib import Path
import hashlib,json
import numpy as np
from queue_guard import diagnose_set,queue_linear_probe
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    reg=json.loads((HERE/'RANDOM_REGISTRATION.json').read_text())
    for p,v in reg['sources'].items():assert sha(p)==v
    raw=Path(reg['raw_root']);total=0;results=[];bindings=[]
    for seed in reg['seeds']:
        rng=np.random.default_rng(seed)
        for n in reg['dimensions']:
            fixtures={k:[] for k in ['G','h','lower','upper','expected','stratum','q','qd','pending']}
            summary={s:dict(cases=0,unexpected=0,all_individual_pass_but_infeasible=0,zero_in_box_but_violates=0) for s in reg['strata']}
            for st in reg['strata']:
                for _ in range(reg['cases_per_stratum_dimension_seed']):
                    center=rng.uniform(-.03,.03,n);width=rng.uniform(.002,.08,n)
                    lo,hi=center-width,center+width
                    G=rng.normal(size=(16,n));G/=np.linalg.norm(G,axis=1)[:,None]
                    witness=rng.uniform(lo,hi);h=G@witness+rng.uniform(.005,.05,16)
                    expected=0
                    if st=='individual_contradiction':
                        rowmin=np.where(G[0]>=0,G[0]*lo,G[0]*hi).sum()
                        h[0]=rowmin-rng.uniform(.001,.01);expected=2
                    elif st=='coupled_contradiction':
                        g=G[0].copy();rowmin=np.where(g>=0,g*lo,g*hi).sum();rowmax=np.where(g>=0,g*hi,g*lo).sum()
                        a=(rowmin+rowmax)/2;gap=.15*(rowmax-rowmin)
                        G[1]=-g;h[0]=a;h[1]=-a-gap;expected=2
                        # Other rows are permissive; impossibility comes from the pair.
                        h[2:]=np.where(G[2:]>=0,G[2:]*hi,G[2:]*lo).sum(axis=1)+.01
                    elif st=='zero_inside_but_unsafe':
                        lo=-width;hi=width;g=G[0].copy()
                        rowmin=np.where(g>=0,g*lo,g*hi).sum()
                        h[0]=.3*rowmin
                        h[1:]=np.where(G[1:]>=0,G[1:]*hi,G[1:]*lo).sum(axis=1)+.01
                    # Save actual float32 inputs; known gaps exceed rounding uncertainty.
                    G,h,lo,hi=[v.astype(np.float32) for v in (G,h,lo,hi)]
                    d=diagnose_set(G,h,lo,hi)
                    assert d['highs_status']==expected,(seed,n,st,d)
                    assert d['future_safety'].startswith('UNKNOWN') and not d['physical_safety_certified']
                    if expected==0:assert d['witness_max_residual']<1e-7
                    if st=='coupled_contradiction':assert d['individual_impossible_rows']==0
                    if st=='zero_inside_but_unsafe':assert d['zero_in_box'] and not d['zero_satisfies_current_rows']
                    q=rng.uniform(-3,3,n).astype(np.float32);qd=rng.uniform(-5,5,n).astype(np.float32)
                    pending=(q+np.cumsum(rng.normal(size=(6,n))*rng.choice([.005,.015,.025,.05]),axis=0)).astype(np.float32)
                    distance=rng.uniform(.001,.08,16);dm=rng.uniform(0,.025,16)
                    probe=queue_linear_probe(distance,dm,G,q,qd,pending,1/60,np.ones(16,dtype=bool))
                    direct=distance[:,None]+np.einsum('md,kd->mk',G.astype(float),pending.astype(float)-q.astype(float))-dm[:,None]
                    np.testing.assert_allclose(probe['pending_min_margin_m'],direct.min(axis=0),rtol=0,atol=1e-12)
                    assert probe['future_safety'].startswith('UNKNOWN')
                    row=summary[st];row['cases']+=1
                    row['all_individual_pass_but_infeasible']+=int(expected==2 and d['individual_impossible_rows']==0)
                    row['zero_in_box_but_violates']+=int(d['zero_in_box'] and not d['zero_satisfies_current_rows'])
                    for k,v in dict(G=G,h=h,lower=lo,upper=hi,expected=expected,stratum=reg['strata'].index(st),q=q,qd=qd,pending=pending).items():fixtures[k].append(v)
                    total+=1
            path=raw/f'random_{seed}_{n}.npz'
            with path.open('xb') as f:np.savez_compressed(f,**{k:np.stack(v) for k,v in fixtures.items()})
            bindings.append(dict(path=str(path),sha256=sha(path),cases=len(fixtures['expected']),dimension=n,seed=seed))
            results.append(dict(seed=seed,dimension=n,strata=summary));print('CLOSED_RANDOM',seed,n,len(fixtures['expected']),flush=True)
    for p,v in reg['sources'].items():assert sha(p)==v
    with (HERE/'RANDOM_RESULTS.json').open('x') as f:
        json.dump(dict(status='PASS_FRESH_SYNTHETIC_LINEAR_CONTRACTS',cases=total,results=results,bindings=bindings,
                       registration_sha256=sha(HERE/'RANDOM_REGISTRATION.json'),new_robot_windows=0,
                       physical_safety_certified=False,scope='constructed random linear sets, not Isaac/servo/hardware or full 26D robot coverage'),f,indent=2);f.write('\n')


if __name__=='__main__':main()
