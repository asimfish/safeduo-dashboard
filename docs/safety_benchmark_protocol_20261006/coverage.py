"""Fixed-limit retrospective coverage, preserving out-of-domain observations."""
import numpy as np
from itertools import combinations
ARMS=('F_L','F_R','U_L','U_R');SLICES=(slice(0,7),slice(7,14),slice(14,20),slice(20,26))
PAIRS=np.array(list(combinations(range(26),2)),np.int32)
ARM_OF=np.repeat(np.arange(4),[7,7,6,6])
def histograms(q,limits,bins=10):
    native=np.asarray(q,np.float64)
    lim=np.broadcast_to(np.asarray(limits,np.float64),native.shape+(2,)).reshape(-1,26,2)
    q=native.reshape(-1,26)
    assert lim.shape==q.shape+(2,) and np.isfinite(q).all() and np.isfinite(lim).all()
    width=lim[...,1]-lim[...,0];assert (width>0).all()
    valid=(q>=lim[...,0])&(q<=lim[...,1])
    ids=np.minimum(np.floor((q-lim[...,0])/width*bins),bins-1).astype(np.int32)
    marginal=np.stack([np.bincount(ids[valid[:,j],j],minlength=bins) for j in range(26)])
    joint=np.stack([np.bincount(ids[valid[:,a]&valid[:,b],a]*bins+ids[valid[:,a]&valid[:,b],b],minlength=bins*bins) for a,b in PAIRS])
    return marginal,joint,(~valid).sum(0),((~valid).any(-1)).sum()
def pair_summary(counts):
    assert counts.shape==(325,100),'frozen10x10 grid required'
    visited=(counts>0).sum(-1);cross=ARM_OF[PAIRS[:,0]]!=ARM_OF[PAIRS[:,1]]
    return dict(pairs=325,within_arm_pairs=72,cross_arm_pairs=253,
                visited_limit_rectangle_cells=visited.tolist(),
                mean_percent=float(visited.mean()),min_percent=int(visited.min()),max_percent=int(visited.max()),
                within_arm_mean_percent=float(visited[~cross].mean()),cross_arm_mean_percent=float(visited[cross].mean()),
                denominator='100 cells of normalized soft-limit rectangle per pair; not certified reachable area or26D volume')
def zero_failure_upper(n,confidence=.95):
    if not isinstance(n,int) or n<=0 or not 0<confidence<1:raise ValueError('invalid sample/confidence')
    return -np.expm1(np.log1p(-confidence)/n)
def zero_failure_required(probability,confidence=.95):
    if not 0<probability<1 or not 0<confidence<1:raise ValueError('invalid risk/confidence')
    return int(np.ceil(np.log1p(-confidence)/np.log1p(-probability)))
