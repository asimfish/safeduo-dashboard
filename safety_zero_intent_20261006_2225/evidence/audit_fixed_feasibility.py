"""All nine predeclared time snapshots, all64 environments, exact selected sets."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json
import numpy as np
from scipy.optimize import linprog

HERE=Path(__file__).resolve().parent
TIMES=[0,60,62,66,71,75,180,480,959]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def inspect(arg):
    root,job=arg;records=[]
    for t in TIMES:
        file=root/'projection_snapshots'/f'step_{t:04d}.npz';digest=sha(file)
        with np.load(file,allow_pickle=False) as z:s={k:z[k].copy() for k in z.files}
        for e in range(64):
            checks={}
            for r,sl in [('F',slice(0,14)),('U',slice(14,26))]:
                rel=s['snapshot_rel_'+r][e];ar=s['snapshot_alpha_rel_'+r][e]
                G=s['snapshot_G_'+r][e][rel].astype(np.float64);h=s['snapshot_h_after_authority_'+r][e][rel].astype(np.float64)
                aG=s['snapshot_alpha_G_'+r][e][ar].astype(np.float64);ah=s['snapshot_alpha_h_'+r][e][ar].astype(np.float64)
                A=np.vstack([G,aG]);b=np.r_[h,ah];lo=s['bounds_lower'][e,sl].astype(np.float64);hi=s['bounds_upper'][e,sl].astype(np.float64)
                assert np.isfinite(A).all() and np.isfinite(b).all() and (lo<=hi).all()
                norm=np.linalg.norm(A,axis=1);norm=np.where(norm>0,norm,1.)
                lp=linprog(np.zeros(len(lo)),A_ub=A/norm[:,None] if len(b) else None,b_ub=b/norm if len(b) else None,bounds=list(zip(lo,hi)),method='highs',options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
                assert lp.status in [0,2],lp.message
                witness=None if not lp.success else float(np.maximum(A@lp.x-b,0).max(initial=0.))
                checks[r]=dict(feasible=bool(lp.success),status=int(lp.status),witness_max_residual=witness)
            records.append(dict(mode=job['env']['SAFEDUO_JOINT_MODE'],seed=job['expected']['seed'],env=e,pre_step=t,checks=checks,snapshot_sha256=digest))
        assert sha(file)==digest
    print('FIXED_LP_CLOSED',job['id'],len(records),flush=True)
    return records


def main():
    reg=json.loads((HERE/'FIXED_LP_REGISTRATION.json').read_text());assert sha(Path(__file__))==reg['source_sha256']
    plans=[json.loads(p.read_text()) for p in sorted((HERE/'plans').glob('holdout_*_plan.json'))]
    args=[(Path(p['output_root'])/j['id'],j) for p in plans for j in p['jobs']]
    with ThreadPoolExecutor(max_workers=2) as pool:rows=[r for records in pool.map(inspect,args) for r in records]
    assert len(rows)==5184 and sha(Path(__file__))==reg['source_sha256']
    summary=[]
    for mode in reg['modes']:
        for t in TIMES:
            group=[r for r in rows if r['mode']==mode and r['pre_step']==t]
            assert len(group)==192
            summary.append(dict(mode=mode,pre_step=t,cases=192,F_infeasible=sum(not r['checks']['F']['feasible'] for r in group),U_infeasible=sum(not r['checks']['U']['feasible'] for r in group)))
    with (HERE/'FIXED_FEASIBILITY_AUDIT.json').open('x') as f:
        json.dump(dict(status='PASS_ALL_FIXED_SELECTED_SET_SNAPSHOTS',states=5184,robot_sets=10368,summary=summary,records=rows,source_sha256=reg['source_sha256'],physical_safety_certified=False,scope='nine predeclared preceding selected-set snapshots; binary64 HiGHS bounded decision, no physical viability or continuous all9021 constraint guarantee'),f,indent=2,allow_nan=False)
    print('ALL_FIXED_FEASIBILITY_CLOSED',flush=True)


if __name__=='__main__':main()
