"""Read-only audit of closed files; old outcomes and new coverage not conflated."""
from pathlib import Path
import json,hashlib,numpy as np
from coverage import histograms,pair_summary,ARMS,SLICES,PAIRS
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def inspect(row):
    root=Path(row['root']);raw=root/'cell_001.npz';inp=root/'input_recipe.npz'
    assert sha(raw)==row['dense_sha256'] and sha(inp)==row['input_sha256']
    protocol=json.loads((root/'protocol.json').read_text());assert protocol['status']=='complete' and protocol['steps']==960 and protocol['completed_cells']==1
    with np.load(raw,allow_pickle=False) as z:
        q=z['q'].copy();q0=z['q_initial'].copy();ee=z['ee'].copy();v=z['pre_qd_compact'].copy();margins=z['official_margins'].copy()
    with np.load(inp,allow_pickle=False) as z:limits=z['joint_soft_limits'].copy();ee0=z['ee_initial'].copy()
    assert q.shape==(960,64,26) and margins.shape==(960,64,4)
    assert all(np.isfinite(x).all() for x in (q,q0,ee,ee0,v,margins,limits))
    full=np.r_[q0[None],q];all_a,all_b,outs,states=histograms(full,limits)
    negative=(margins<0).any(-1);bad=negative.any(0);first=np.where(bad,negative.argmax(0),960)
    safe_a=np.zeros_like(all_a);safe_b=np.zeros_like(all_b);safe_out=np.zeros(26,np.int64)
    safe_states=0;span=[];prefix_span=[];path=[];prefix_path=[]
    for e,t in enumerate(first):
        trajectory=full[:,e];prefix=full[:int(t)+1,e]
        a,b,o,s=histograms(prefix,limits[e]);safe_a+=a;safe_b+=b;safe_out+=o;safe_states+=s
        width=limits[e,:,1].astype(float)-limits[e,:,0].astype(float)
        span.append(float((np.ptp(trajectory.astype(float),axis=0)/width).mean()*100))
        prefix_span.append(float((np.ptp(prefix.astype(float),axis=0)/width).mean()*100))
        path.append(float(np.abs(np.diff(trajectory.astype(float),axis=0)).sum()))
        prefix_path.append(float(np.abs(np.diff(prefix.astype(float),axis=0)).sum()))
    pressure_rms=np.stack([np.sqrt(np.square(v[60:,...,s].astype(float)).mean(-1)) for s in SLICES],-1)
    voxels=[int(len(np.unique(np.floor(np.r_[ee0[None],ee][:,:,a,:].reshape(-1,3)/.1).astype(np.int64),axis=0))) for a in range(4)]
    result=dict(id=row['id'],mode=row['mode'],seed=row['seed'],windows=64,violations=int(bad.sum()),deep=int((margins<-.005).any((0,2)).sum()),
        observed_joint_values=int(full.size),outside_soft_limits_joint_values=outs.tolist(),outside_soft_limits_states=int(states),
        full_marginal_cells=(all_a>0).sum(-1).tolist(),full_pairs=pair_summary(all_b),
        nonnegative_prefix_marginal_cells=(safe_a>0).sum(-1).tolist(),nonnegative_prefix_pairs=pair_summary(safe_b),
        prefix_outside_soft_limits_states=int(safe_states),prefix_outside_soft_limits_joint_values=safe_out.tolist(),
        normalized_span_percent=span,nonnegative_prefix_span_percent=prefix_span,joint_l1_path_rad=path,nonnegative_prefix_joint_l1_path_rad=prefix_path,
        actual_all_four_active_fraction=float((pressure_rms>.01).all(-1).mean()),
        ee_100mm_local_voxels=voxels,ee_denominator=None,orientation_coverage=None,
        outside_limit_scope='soft target bounds, not independently specified hardware limits',
        native_float32_physical_endpoints=True,dense_sha256=row['dense_sha256'],input_sha256=row['input_sha256'])
    assert result['violations']==row['expected_violations'] and result['deep']==row['expected_deep']
    assert sha(raw)==row['dense_sha256'] and sha(inp)==row['input_sha256']
    return result,dict(full_marginal=all_a,full_pairs=all_b,prefix_marginal=safe_a,prefix_pairs=safe_b)
def main():
    registration=json.loads((HERE/'AUDIT_REGISTRATION.json').read_text())
    for path,h in registration['source_sha256'].items():assert sha(path)==h,path
    results=[];totals={}
    for row in registration['rows']:
        result,counts=inspect(row);results.append(result)
        target=totals.setdefault(row['mode'],{k:np.zeros_like(v) for k,v in counts.items()})
        for key in counts:target[key]+=counts[key]
        print('AUDITED',row['id'],result['violations'],'pair',result['full_pairs']['mean_percent'],flush=True)
    groups=[];directory=HERE/'histograms';directory.mkdir(exist_ok=False)
    for mode,counts in totals.items():
        rows=[r for r in results if r['mode']==mode];file=directory/(mode+'.npz');np.savez_compressed(file,pairs=PAIRS,**counts)
        groups.append(dict(mode=mode,windows=192,violations=sum(r['violations'] for r in rows),deep=sum(r['deep'] for r in rows),
            full_marginal_cells=(counts['full_marginal']>0).sum(-1).tolist(),full_pairs=pair_summary(counts['full_pairs']),
            nonnegative_prefix_marginal_cells=(counts['prefix_marginal']>0).sum(-1).tolist(),nonnegative_prefix_pairs=pair_summary(counts['prefix_pairs']),
            mean_normalized_span_percent=float(np.mean([r['normalized_span_percent'] for r in rows])),
            mean_nonnegative_prefix_span_percent=float(np.mean([r['nonnegative_prefix_span_percent'] for r in rows])),
            actual_all_four_active_fraction=float(np.mean([r['actual_all_four_active_fraction'] for r in rows])),
            outside_soft_limits_states=sum(r['outside_soft_limits_states'] for r in rows),histogram_sha256=sha(file)))
    for path,h in registration['source_sha256'].items():assert sha(path)==h
    write(HERE/'coverage_audit.json',dict(status='complete',audited_historical_windows=768,new_physical_windows=0,unique_old_initials=192,
        rows=results,groups=groups,contact_oracle=None,orientation_coverage=None,reachable_space_reference=None,
        operational_reliability_ci=None,scope='retrospective lower-dimensional coverage; signed sphere events only; no26D/hardware/independent-probability claim'))
if __name__=='__main__':main()
