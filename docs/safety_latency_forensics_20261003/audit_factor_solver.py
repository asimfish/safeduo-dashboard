"""Audit actual priority-weighted full solver constraints near factor failures."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from analyze import constraints,load


OUT=Path(__file__).resolve().parent
reports=[]
for name in ['nominal_debit','nominal_critical','nominal_both',
             'delayed_debit','delayed_critical','delayed_both']:
    p=json.loads((OUT/name/'protocol.json').read_text())
    assert p['status']=='complete'
    d=load(OUT/name/'cell_001.npz');bad=(d['official_margins']<0).any(-1)
    observations=[]
    for env in np.flatnonzero(bad.any(0)):
        first=int(np.flatnonzero(bad[:,env])[0])
        for t in sorted({max(first-5,0),max(first-2,0),first}):
            for robot in [0,1]:
                G,h,rel,lo,hi,aG,ah,debit,h0=constraints(d,p['effective_backstop'],t,int(env),robot,include_cross=True)
                g,b=G[rel],h[rel];nd=len(lo)
                aug=np.vstack([np.column_stack([g,-np.ones(len(g))]),
                               np.column_stack([aG,np.zeros(len(aG))])])
                rhs=np.r_[b,ah]
                res=linprog(np.r_[np.zeros(nd),1.],A_ub=aug,b_ub=rhs,
                    bounds=list(zip(lo,hi))+[(0,None)],method='highs')
                assert res.success,res.message
                u=d['exec'][t,env,:14] if robot==0 else d['exec'][t,env,14:]
                actual=float(np.maximum(g@u-b,0).max(initial=0))
                recorded=float(d['solver_residual'][t,env,robot])
                assert abs(actual-recorded)<2e-6,(name,t,env,robot,actual,recorded)
                observations.append(dict(env=int(env),step=t,robot=['F','U'][robot],
                    full_row_slack_mm=float(res.x[-1]*1000),
                    reproduced_residual_mm=actual*1000,recorded_residual_mm=recorded*1000))
    reports.append(dict(run=name,observations=observations,
        sampled_full_set_infeasible=sum(x['full_row_slack_mm']>1e-3 for x in observations),
        sampled_snapshots=len(observations)))
(OUT/'factor_solver_audit.json').write_text(json.dumps(dict(
    scope='All actual selected rows, fixed recorded p, alpha and target/speed bounds; sampled before/at failure only. Linear feasibility is not physical safety.',
    reports=reports),indent=2)+'\n')
for r in reports: print(r['run'],'infeasible',r['sampled_full_set_infeasible'],'/',r['sampled_snapshots'])
