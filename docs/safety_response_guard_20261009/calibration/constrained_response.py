"""Implicit clipped PD plus coupled native joint stops; nominal, not certified."""
import numpy as np

def advance_constrained(q,v,target,mass,kp,kd,effort,vmax,bias,hard,dt,max_iterations=80,tolerance=1e-7):
    q,v,target,mass,kp,kd,effort,vmax,bias,hard=[np.asarray(x,dtype=float) for x in (q,v,target,mass,kp,kd,effort,vmax,bias,hard)]
    if q.ndim!=2 or any(x.shape!=q.shape for x in (v,target,kp,kd,effort,vmax,bias)) or hard.shape!=(*q.shape,2) or mass.shape!=(*q.shape,q.shape[-1]):raise ValueError('native batch shape mismatch')
    if not all(np.isfinite(x).all() for x in (q,v,target,mass,kp,kd,effort,vmax,bias,hard)) or not np.isfinite(dt) or dt<=0:raise ValueError('invalid native data')
    if (kp<0).any() or (kd<0).any() or (effort<=0).any() or (vmax<=0).any() or (hard[...,0]>hard[...,1]).any():raise ValueError('invalid native limits')
    if not np.allclose(mass,mass.swapaxes(-1,-2),atol=1e-5,rtol=1e-5):raise ValueError('mass is not symmetric')
    mass=(mass+mass.swapaxes(-1,-2))/2;np.linalg.cholesky(mass)
    lo=np.maximum(-vmax,(hard[...,0]-q)/dt);hi=np.minimum(vmax,(hard[...,1]-q)/dt)
    if (lo>hi).any():raise ValueError('joint/velocity box infeasible')
    slope=kp*dt+kd;offset=kp*(target-q)
    momentum=np.einsum('nij,nj->ni',mass,v)-dt*bias
    def gradient(w):return np.einsum('nij,nj->ni',mass,w)-momentum-dt*np.clip(offset-slope*w,-effort,effort)
    def objective(w):
        z=offset-slope*w;h=np.where(np.abs(z)<=effort,.5*z*z,effort*(np.abs(z)-.5*effort))
        potential=np.where(slope>0,h/np.maximum(slope,1e-30),-np.clip(offset,-effort,effort)*w)
        return .5*np.einsum('ni,nij,nj->n',w,mass,w)-np.sum(momentum*w,-1)+dt*potential.sum(-1)
    def fixed(w,g):return ((w<=lo+1e-12)&(g>=0))|((w>=hi-1e-12)&(g<=0))|(hi-lo<=1e-12)
    w=np.clip(v,lo,hi);done=np.zeros(len(q),bool);diag=np.arange(q.shape[-1])
    for _ in range(max_iterations):
        g=gradient(w);bound=fixed(w,g);res=np.where(bound,0,g);norm=np.abs(res).max(-1);done|=norm<=tolerance
        if done.all():break
        free=~bound;H=mass.copy();H[:,diag,diag]+=dt*slope*(np.abs(offset-slope*w)<effort)
        H*=free[:,:,None]*free[:,None,:];H[:,diag,diag]+=bound
        direction=np.linalg.solve(H,-res[...,None])[...,0];base=objective(w);scale=np.ones(len(q))
        accepted=done.copy();proposal=w.copy()
        for _ in range(32):
            p=np.clip(w+scale[:,None]*direction,lo,hi);descent=np.sum(g*(p-w),-1)
            good=objective(p)<=base+1e-4*descent+1e-12
            use=good&~accepted;proposal[use]=p[use];accepted|=good
            if accepted.all():break
            scale[~accepted]*=.5
        w=np.where(done[:,None],w,proposal)
    g=gradient(w);res=np.where(fixed(w,g),0,g);norm=np.abs(res).max(-1)
    return q+dt*w,w,dict(converged=norm<=tolerance,kkt_residual=norm,
        lower_active=w<=lo+1e-10,upper_active=w>=hi-1e-10,
        torque=np.clip(offset-slope*w,-effort,effort),safety_certificate=False)
