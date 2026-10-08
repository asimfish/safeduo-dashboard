"""Full original reachable-target box support; frozen J is a heuristic only."""
import torch
from safeduo.safety.types import ARM_KEYS
from safeduo.baselines.base import stack_robot
from reference_envelope import reference_bounds
from target_forecast import require_finite

def box_support_forecast(full,rows,state,issued,limits,box):
 require_finite('box support full geometry',dict(d=full.dists,dmin=full.full_dmin));require_finite('box support J',rows.J)
 require_finite('box support q',state.q);require_finite('box support issued',issued)
 bounds={a:reference_bounds(state.q[a],issued[a],limits[a][...,0],limits[a][...,1],box) for a in ARM_KEYS}
 debt={a:issued[a]-state.q[a] for a in ARM_KEYS}
 prediction=full.dists.clone()
 for robot in ('F','U'):
  J=rows.J[robot];lo=stack_robot({a:v[0] for a,v in bounds.items()},robot);hi=stack_robot({a:v[1] for a,v in bounds.items()},robot)
  prediction=prediction+torch.einsum('nmd,nd->nm',J,stack_robot(debt,robot))
  prediction=prediction+torch.where(J>=0,J*lo[:,None,:],J*hi[:,None,:]).sum(-1)
 require_finite('box support result',dict(prediction=prediction))
 return prediction
