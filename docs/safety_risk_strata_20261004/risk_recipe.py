"""Pure input recipe: fixed zero prefix, unchanged independent pressure tape."""
import hashlib
import numpy as np
import torch

PREFIX=60

def prepend_zero(tape,info):
    tape=torch.cat([torch.zeros_like(tape[:PREFIX]),tape],0)
    out={**info, 'kind':'zero_prefix_then_mixed_hold', 'zero_prefix_steps':PREFIX,
         'random_steps':len(tape)-PREFIX-2}
    for key in ['updates','holds','segment_amplitudes']:
        out[key]=np.concatenate([np.zeros_like(info[key][:PREFIX]),info[key]],0)
    out['tape_sha256']=hashlib.sha256(tape.cpu().numpy().tobytes()).hexdigest()
    return tape,out

def validate_bank(q,labels,meta):
    if q.shape!=(64,26) or not np.isfinite(q).all():raise ValueError('exactly 64 finite initial poses required')
    if labels.shape!=(64,) or [(labels==i).sum() for i in range(6)]!=[8]*6 or (labels==-1).sum()!=16:
        raise ValueError('six risk quotas8 plus general16 required; no padding')
    if meta['status']!='complete' or meta['selected_count']!=64 or meta['policy_outcomes_used']:
        raise ValueError('only complete pre-policy bank admitted')
