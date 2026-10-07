"""Widening also changes authority RHS; isolate that mathematical dependency."""
from pathlib import Path
import hashlib,json
import numpy as np
from queue_guard import diagnose_set
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    reg=json.loads((HERE/'AUTHORITY_REGISTRATION.json').read_text())
    for p,v in reg['sources'].items():assert sha(p)==v
    rows=json.loads((HERE/'CONSTRAINT_DECOMPOSITION.json').read_text())['records'];output=[];cache={}
    for r in rows:
        path=Path(r['snapshot_path'])
        assert sha(path)==r['snapshot_sha256']
        if path not in cache:
            with np.load(path,allow_pickle=False) as z:cache[path]={k:z[k].copy() for k in z.files}
        s=cache[path];i=int(np.flatnonzero(s['env_ids']==r['env'])[0])
        cell=path.parent.parent/'cell_001.npz'
        if cell not in cache:
            with np.load(cell,allow_pickle=False) as z:cache[cell]=z['joint_soft_limits'].copy()
        limits=cache[cell][r['env']].astype(float);target=s['pre_issued_target'][i].astype(float)
        lo=np.maximum(limits[:,0]-target,-1.5/60);hi=np.minimum(limits[:,1]-target,1.5/60)
        checks={}
        for robot,sl in [('F',slice(0,14)),('U',slice(14,26))]:
            rel=s['snapshot_rel_'+robot][i];ar=s['snapshot_alpha_rel_'+robot][i]
            G=s['snapshot_G_'+robot][i,rel].astype(float)
            h_before=s['snapshot_h_before_authority_'+robot][i,rel].astype(float)
            rowmin=np.where(G>=0,G*lo[sl],G*hi[sl]).sum(-1)
            h_recomputed=np.maximum(h_before,.9*rowmin)
            aG=s['snapshot_alpha_G_'+robot][i,ar];ah=s['snapshot_alpha_h_'+robot][i,ar]
            checks[robot]=dict(widen_recompute_authority_keep_alpha=diagnose_set(G,h_recomputed,lo[sl],hi[sl],aG,ah),
                              widen_before_authority_keep_alpha=diagnose_set(G,h_before,lo[sl],hi[sl],aG,ah))
        output.append(dict(condition=r['condition'],mode=r['mode'],seed=r['seed'],env=r['env'],first_failure_step=r['first_failure_step'],checks=checks))
    counts=[]
    for mode in reg['modes']:
        for robot in ['F','U']:
            group=[r for r in output if r['mode']==mode]
            counts.append(dict(mode=mode,robot=robot,cases=len(group),infeasible={k:sum(r['checks'][robot][k]['highs_status']==2 for r in group) for k in group[0]['checks'][robot]}))
    for p,v in reg['sources'].items():assert sha(p)==v
    with (HERE/'AUTHORITY_AUDIT.json').open('x') as f:
        json.dump(dict(status='PASS_MATHEMATICAL_AUTHORITY_DEPENDENCY_PROBE',cases=211,lp_sets=844,counts=counts,records=output,
            scope='Post-outcome binary64 arithmetic on saved float32 rows. Recomputed .9*minimum budget only; selected rows/gates/alpha/input remain frozen. Not an execution of widened production policy or physical counterfactual.',physical_safety_certified=False),f,indent=2);f.write('\n')
    print('CLOSED_AUTHORITY',len(output),844,flush=True)


if __name__=='__main__':main()
