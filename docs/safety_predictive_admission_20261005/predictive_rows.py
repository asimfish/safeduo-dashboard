"""Evaluation-only exact union; class horizons are frozen, not a dynamics proof."""
from dataclasses import replace
import torch

HORIZONS=(.16,.40,.30)
CAPACITY=2048

def merge_predictive_rows(full,selected,class_id,pair_id,capacity=CAPACITY):
    values=(full.dists,full.closing,full.full_dmin,full.full_viol_exempt)
    if any(x is None for x in values):raise ValueError('full predictive admission requires complete measurements')
    d,closing,dm,exempt=values
    if not all(torch.isfinite(x).all() for x in (d,closing,dm)):
        raise ValueError('nonfinite predictive admission measurements')
    if capacity<1:raise ValueError('predictive row capacity must be positive')
    n,p=d.shape
    classes=class_id.long()
    if ((classes<0)|(classes>2)).any():raise ValueError('unknown geometry class')
    horizon=torch.tensor(HORIZONS,device=d.device,dtype=d.dtype)[classes]
    predicted=(d-horizon*closing.clamp_min(0))<=dm+.010
    instantaneous=d<=dm+.010
    selected_membership=torch.zeros((n,p),device=d.device,dtype=torch.long)
    selected_membership.scatter_add_(1,selected.active_idx.clamp_min(0),selected.active_mask.long())
    legacy=selected_membership>0
    union=legacy|instantaneous|predicted
    count=union.sum(-1);required=int(count.max().item())
    if required>capacity:raise ValueError(f'predictive-row union exceeds declared capacity: {required} > {capacity}')
    width=max(required,1)
    ids=torch.arange(p,device=d.device).expand(n,-1)
    chosen=ids.masked_fill(~union,p).topk(width,dim=1,largest=False).values
    valid=chosen<p;idx=chosen.clamp_max(p-1)
    features=torch.stack([d.gather(1,idx),closing.gather(1,idx),class_id[idx],pair_id[idx]],-1)
    features=features.masked_fill(~valid.unsqueeze(-1),0);features[...,3]=features[...,3].masked_fill(~valid,-1)
    out=replace(selected,active_pairs=features,active_mask=valid,active_idx=idx.masked_fill(~valid,-1),
                active_dmin=dm.gather(1,idx).masked_fill(~valid,0),viol_exempt=exempt.gather(1,idx)&valid)
    # Independent membership after packing verifies the actual returned row set.
    membership=torch.zeros_like(selected_membership);membership.scatter_add_(1,idx,valid.long())
    returned=membership>0
    stats=dict(required=count,additional=(union&~legacy).sum(-1),
               predicted_missing=(predicted&~returned).sum(-1),legacy_missing=(legacy&~returned).sum(-1))
    if stats['predicted_missing'].any() or stats['legacy_missing'].any():
        raise ValueError('packed predictive row union dropped required rows')
    return out,stats
