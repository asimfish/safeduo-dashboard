from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent/'dependencies'))
from critical_rows import merge_critical_rows
from safeduo.safety.sphere_distance import SphereDistOut


def test_131_required_rows_survive_dynamic_allocation_with_original_actor32():
    d=torch.full((2,160),.20);d[0,:131]=.001
    dm=torch.full_like(d,.038)
    selected=SphereDistOut(active_pairs=torch.zeros(2,32,4),active_mask=torch.ones(2,32,dtype=torch.bool),
                          active_idx=torch.arange(32).expand(2,-1),active_dmin=dm[:,:32],
                          viol_exempt=torch.zeros(2,32,dtype=torch.bool),min_margin={},
                          violation=torch.zeros(2,dtype=torch.bool),dists=d,closing=torch.zeros_like(d),
                          full_dmin=dm,full_viol_exempt=torch.zeros_like(d,dtype=torch.bool))
    cls=torch.zeros(160);ids=torch.arange(160.)
    with pytest.raises(ValueError,match='131 > 128'):
        merge_critical_rows(selected,selected,cls,ids,capacity=128)
    dynamic=merge_critical_rows(selected,selected,cls,ids,capacity=None)
    assert dynamic.active_idx.shape==(2,131)
    assert dynamic.active_mask.sum(-1).tolist()==[131,32]
    assert set(dynamic.active_idx[0].tolist())==set(range(131))
    assert set(dynamic.active_idx[1][dynamic.active_mask[1]].tolist())==set(range(32))
    assert selected.active_idx.shape==(2,32)
    assert (dynamic.active_idx[~dynamic.active_mask]==-1).all()
