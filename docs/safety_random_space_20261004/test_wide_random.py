import numpy as np
import pytest
import torch

from wide_random import make_initial, make_tape


def test_global_random_strata_and_seeded_limits():
    base=torch.zeros((64,26)); limits=torch.stack([base-2,base+3],-1)
    a,meta=make_initial(base,limits,'global_lhs',91)
    b,_=make_initial(base,limits,'global_lhs',91)
    c,_=make_initial(base,limits,'global_lhs',92)
    assert torch.equal(a,b) and not torch.equal(a,c)
    f=(a-limits[...,0])/(limits[...,1]-limits[...,0])
    assert (f>=.025).all() and (f<.975).all()
    bins=torch.floor((f-.025)/.95*64).long()
    for j in range(26):assert sorted(bins[:,j].tolist())==list(range(64))
    assert not np.array_equal(bins[:,0],bins[:,1])
    assert meta['clipped_joints']==[0]*64


def test_local_jitter_is_matched_and_clipping_visible():
    base=torch.full((64,26),.8); limits=torch.stack([base*0-1,base*0+1],-1)
    a,meta=make_initial(base,limits,'wide_jitter',73)
    assert a.min()>=-1 and a.max()<=1
    assert sum(meta['clipped_joints'])>0
    assert meta['requested_max_abs_rad']>.99


@pytest.mark.parametrize('kind',['iid','mixed_hold'])
def test_tape_reproducible_seed_separation_and_bounds(kind):
    a,info=make_tape(32,400,.05,7,kind)
    b,_=make_tape(32,400,.05,7,kind)
    c,_=make_tape(32,400,.05,8,kind)
    assert torch.equal(a,b) and not torch.equal(a,c)
    assert a.shape==(402,32,26) and a.abs().max()<=.05
    assert (a.reshape(402,32,26).abs().sum(-1)>0).all()
    if kind=='iid':
        assert (a[1:]!=a[:-1]).any(-1).all()
        assert info['hold_choices']==[1]
    else:
        assert set(info['hold_choices'])=={1,4,15,30,90,180}
        assert set(info['amplitude_choices'])=={.005,.015,.025,.05}
        updates=info['updates']
        holds=info['holds'];amps=info['segment_amplitudes']
        assert all(info['hold_counts'][str(x)]>0 for x in info['hold_choices'])
        assert all(info['amplitude_counts'][str(x)]>0 for x in info['amplitude_choices'])
        assert not np.array_equal(updates[:,0,0],updates[:,0,1])
        assert not np.array_equal(updates[:,0,0],updates[:,1,0])
        for e in range(32):
            for arm,sl in enumerate([slice(0,7),slice(7,14),slice(14,20),slice(20,26)]):
                starts=np.flatnonzero(updates[:,e,arm])
                for t,nxt in zip(starts[:-1],starts[1:]):
                    assert nxt-t==holds[t,e,arm]
                    assert torch.all(a[t:nxt,e,sl]==a[t,e,sl])
                    assert a[t,e,sl].abs().max()<=amps[t,e,arm]+1e-7
