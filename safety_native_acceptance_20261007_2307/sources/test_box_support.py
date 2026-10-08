from types import SimpleNamespace as S
import numpy as np,torch
from scipy.optimize import linprog
from box_support import box_support_forecast
from reference_envelope import reference_bounds
from safeduo.safety.types import ARM_KEYS
from safeduo.baselines.base import stack_robot
rng=np.random.default_rng(784671170203);n,m=4,32
q={a:torch.tensor(rng.uniform(-.5,.5,(n,w)),dtype=torch.float64) for a,w in zip(ARM_KEYS,[7,7,6,6])}
target={a:x+torch.tensor(rng.uniform(-.03,.03,x.shape)) for a,x in q.items()}
limits={a:torch.stack([x-.4,x+.4],-1) for a,x in q.items()};J={r:torch.tensor(rng.normal(size=(n,m,w))) for r,w in [('F',14),('U',12)]};full=S(dists=torch.tensor(rng.uniform(.001,.2,(n,m))),full_dmin=torch.full((n,m),.038,dtype=torch.float64));rows=S(J=J);state=S(q=q)
y=box_support_forecast(full,rows,state,target,limits,.025).numpy();bounds={a:reference_bounds(q[a],target[a],limits[a][...,0],limits[a][...,1],.025) for a in ARM_KEYS}
lo=np.concatenate([bounds[a][0].numpy() for a in ARM_KEYS],-1);hi=np.concatenate([bounds[a][1].numpy() for a in ARM_KEYS],-1);debt=np.concatenate([(target[a]-q[a]).numpy() for a in ARM_KEYS],-1);jac=np.concatenate([J[r].numpy() for r in ['F','U']],-1)
for e in range(n):
 for i in range(m):
  res=linprog(jac[e,i],bounds=list(zip(lo[e],hi[e])),method='highs');assert res.success
  expected=float(full.dists[e,i])+jac[e,i]@debt[e]+res.fun;assert abs(y[e,i]-expected)<1e-10
# Raw proposal alone can miss a harmful corner; box support must include it.
sampled=full.dists.numpy()+np.einsum('nmd,nd->nm',jac,debt);assert (y<=sampled+1e-12).all() and (y<sampled-.001).any()
J['F'][0,0,0]=float('nan')
try:box_support_forecast(full,rows,state,target,limits,.025)
except ValueError:pass
else:raise AssertionError('nonfinite J accepted')
print('PASS128 independent signed-box LP oracles; rawproposal omission negative control; nonfinite rejected')
