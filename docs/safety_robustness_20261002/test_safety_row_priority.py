"""Safety selection must include closing threats without changing actor rows."""
from types import SimpleNamespace

import pytest
import torch

from safeduo.safety.sphere_distance import SphereDistanceModule, SphereDistOut


def crowded_snapshot():
    p = 40
    module = SimpleNamespace(n_pairs=p, max_active=32, quota_cross=8,
        _slice_cross=slice(0, 8), d_soft=.05, tau_ttc=.5,
        class_id=torch.tensor([0.] * 8 + [1.] * 32), pair_id=torch.arange(p).float())
    d = torch.full((1,p), .023)
    closing = torch.zeros_like(d)
    dm = torch.full_like(d, .013)
    d[0,:8] = .2
    dm[0,:8] = .03
    d[0,-1] = .08
    closing[0,-1] = .6
    ids = torch.arange(32)[None]
    out = SphereDistOut(active_pairs=torch.zeros(1,32,4), active_mask=torch.ones(1,32,dtype=torch.bool),
        active_idx=ids, active_dmin=dm[:,:32].clone(), viol_exempt=torch.zeros(1,32,dtype=torch.bool),
        min_margin={}, violation=torch.zeros(1,dtype=torch.bool), dists=d, closing=closing,
        full_dmin=dm, full_viol_exempt=torch.zeros(1,p,dtype=torch.bool))
    return module, out


def test_far_closing_threat_displaces_quiet_nearer_rows():
    module, out = crowded_snapshot()
    original = out.active_idx.clone()
    safety = SphereDistanceModule.prioritized_out(module, out, .15)
    assert 39 in safety.active_idx[safety.active_mask]
    assert 39 not in original
    assert torch.equal(out.active_idx, original)
    k = torch.where(safety.active_idx[0] == 39)[0].item()
    assert safety.active_pairs[0,k,0] == .08  # actual measured distance, not prediction
    assert safety.active_pairs[0,k,1] == .6
    assert safety.active_dmin[0,k] == .013
    selected = safety.active_idx[0][safety.active_mask[0]]
    assert len(selected.unique()) == len(selected)


def test_cross_quota_and_exemptions_survive_priority_selection():
    module, out = crowded_snapshot()
    out.dists[0,:8] = .025
    out.full_viol_exempt[0,39] = True
    out.full_dmin[0,39] = .005
    safety = SphereDistanceModule.prioritized_out(module, out, .15)
    assert (safety.active_pairs[0,:,2][safety.active_mask[0]] == 0).sum() >= 8
    k = torch.where(safety.active_idx[0] == 39)[0].item()
    assert safety.viol_exempt[0,k]
    assert safety.active_dmin[0,k] == .005
    assert len(safety.active_idx[0].unique()) == 32


def test_empty_eligible_set_has_valid_padding():
    module, out = crowded_snapshot()
    out.dists.fill_(1.)
    out.closing.zero_()
    safety = SphereDistanceModule.prioritized_out(module, out, .15)
    assert not safety.active_mask.any()
    assert (safety.active_idx == -1).all()
    assert (safety.active_pairs[...,3] == -1).all()
    assert (safety.active_dmin == 0).all()
    assert not safety.viol_exempt.any()


def test_priority_requires_current_full_snapshot():
    module, out = crowded_snapshot()
    out.full_dmin = None
    with pytest.raises(ValueError, match='full distance snapshot'):
        SphereDistanceModule.prioritized_out(module, out, .15)
