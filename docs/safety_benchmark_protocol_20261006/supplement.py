"""Separate registered supplement: distinguish initial poses from later occupancy."""
from pathlib import Path
import hashlib,json,numpy as np
from coverage import histograms,pair_summary
HERE=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def main():
    reg=json.loads((HERE/'INITIAL_SUPPLEMENT_REGISTRATION.json').read_text())
    for name,h in reg['source_sha256'].items():assert sha(HERE/name)==h,name
    old=json.loads((HERE/'AUDIT_REGISTRATION.json').read_text())
    totals={};seen={};equal=True
    for row in old['rows']:
        root=Path(row['root']);raw=root/'cell_001.npz';inp=root/'input_recipe.npz'
        assert sha(raw)==row['dense_sha256'] and sha(inp)==row['input_sha256']
        with np.load(raw,allow_pickle=False) as z:q=z['q_initial'].copy()
        with np.load(inp,allow_pickle=False) as z:limits=z['joint_soft_limits'].copy()
        assert q.shape==(64,26)
        if row['seed'] in seen:equal=equal and np.array_equal(q,seen[row['seed']])
        else:seen[row['seed']]=q.copy()
        counts=histograms(q,limits)
        target=totals.setdefault(row['mode'],[np.zeros_like(v) for v in counts])
        for i,v in enumerate(counts):target[i]+=v
        assert sha(raw)==row['dense_sha256'] and sha(inp)==row['input_sha256']
        print('INITIAL_AUDITED',row['id'],flush=True)
    groups=[]
    for mode,(m,p,o,s) in totals.items():
        path=HERE/'histograms'/(mode+'.npz')
        with np.load(path,allow_pickle=False) as z:
            full=z['full_pairs'];prefix=z['prefix_pairs']
            assert np.all((p>0)<=(prefix>0)) and np.all(prefix<=full)
            added_full=((full>0)&(p==0)).sum(-1)
            added_prefix=((prefix>0)&(p==0)).sum(-1)
        groups.append(dict(mode=mode,initial_states=192,initial_pairs=pair_summary(p),initial_marginal_cells=(m>0).sum(-1).tolist(),
            initial_outside_soft_limit_states=int(s),initial_outside_joint_values=o.tolist(),
            full_added_pair_cells=added_full.tolist(),nonnegative_prefix_added_pair_cells=added_prefix.tolist(),
            mean_full_added_pair_cells=float(added_full.mean()),mean_prefix_added_pair_cells=float(added_prefix.mean()),
            initial_pairs_counts=p.tolist()))
    for name,h in reg['source_sha256'].items():assert sha(HERE/name)==h,name
    result=dict(status='complete',scope='retrospective initial-state occupancy supplement; no new physics or reachable-volume claim',
        actual_initial_states_bitexact_between_methods=bool(equal),unique_old_initials=192,groups=groups)
    with (HERE/'initial_coverage.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
if __name__=='__main__':main()
