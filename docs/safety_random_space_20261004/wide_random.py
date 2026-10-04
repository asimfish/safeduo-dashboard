"""State-independent random tapes and auditable wide initial-pose sampling."""
import hashlib

import numpy as np
import torch

DOFS=(7,7,6,6)
HOLD_CHOICES=(1,4,15,30,90,180)
AMP_CHOICES=(.005,.015,.025,.05)


def generator(seed,stream):
    value=int.from_bytes(hashlib.sha256(f'wide-random-v1|{seed}|{stream}'.encode()).digest()[:8],'little')%(2**63-1)
    return torch.Generator(device='cpu').manual_seed(value)


def make_initial(baseline,limits,kind,seed):
    if baseline.ndim!=2 or limits.shape!=baseline.shape+(2,):
        raise ValueError('batched joint positions and limits required')
    if not torch.isfinite(limits).all() or (limits[...,0]>=limits[...,1]).any():
        raise ValueError('finite ordered limits required')
    base=baseline.detach().cpu();lim=limits.detach().cpu()
    n,d=base.shape;gen=generator(seed,'initial')
    if kind=='global_lhs':
        u=torch.empty_like(base)
        for j in range(d):u[:,j]=(torch.randperm(n,generator=gen)+torch.rand(n,generator=gen))/n
        requested=lim[...,0]+(.025+.95*u)*(lim[...,1]-lim[...,0])
    elif kind=='wide_jitter':
        requested=base+(torch.rand(base.shape,generator=gen)*2-1)*1.2
    else:
        raise ValueError('initial kind must be wide_jitter or global_lhs')
    positions=requested.clamp(lim[...,0],lim[...,1])
    clipped=(positions!=requested).sum(-1).tolist()
    metadata=dict(kind=kind,seed=int(seed),clipped_joints=clipped,
                  requested_max_abs_rad=float((requested-base).abs().max()),
                  effective_max_abs_rad=float((positions-base).abs().max()),
                  effective_rms_rad=(positions-base).square().mean(-1).sqrt().tolist(),
                  sampled_fraction=(.025,.975) if kind=='global_lhs' else None)
    return positions.to(baseline.device),metadata


def make_tape(n,steps,amp,seed,kind):
    if n<=0 or steps<=0 or amp!=.05:
        raise ValueError('positive sizes and registered maximum amplitude .05 required')
    length=steps+2
    directions=generator(seed,'directions');times=generator(seed,'timing');scales=generator(seed,'amplitudes')
    updates=np.zeros((length,n,4),dtype=bool)
    holds=np.zeros((length,n,4),dtype=np.int16)
    amplitudes=np.zeros((length,n,4),dtype=np.float32)
    if kind=='iid':
        tape=(torch.rand((length,n,26),generator=directions)*2-1)*amp
        updates[:]=True;holds[:]=1;amplitudes[:]=amp
        hc=[1];ac=[amp]
    elif kind=='mixed_hold':
        tape=torch.empty((length,n,26))
        offset=0
        for arm,dof in enumerate(DOFS):
            for e in range(n):
                t=0
                while t<length:
                    hold=HOLD_CHOICES[int(torch.randint(len(HOLD_CHOICES),(1,),generator=times))]
                    scale=AMP_CHOICES[int(torch.randint(len(AMP_CHOICES),(1,),generator=scales))]
                    q=(torch.rand(dof,generator=directions)*2-1)*scale
                    tape[t:min(t+hold,length),e,offset:offset+dof]=q
                    updates[t,e,arm]=True;holds[t,e,arm]=hold;amplitudes[t,e,arm]=scale
                    t+=hold
            offset+=dof
        hc=list(HOLD_CHOICES);ac=list(AMP_CHOICES)
    else:
        raise ValueError('temporal kind must be iid or mixed_hold')
    info=dict(kind=kind,seed=int(seed),hold_choices=hc,amplitude_choices=ac,
              hold_counts={str(x):int((holds==x).sum()) for x in hc},
              amplitude_counts={str(x):int(((np.abs(amplitudes-x)<1e-8)&updates).sum()) for x in ac},
              updates=updates,holds=holds,segment_amplitudes=amplitudes,
              tape_sha256=hashlib.sha256(tape.numpy().tobytes()).hexdigest(),
              streams='independent direction/timing/amplitude CPU generators; no global RNG or state feedback')
    return tape,info
