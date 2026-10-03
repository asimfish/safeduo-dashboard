import pytest
import torch

from critical_rows import merge_critical_rows
from safeduo.baselines.base import ConstraintRows
from safeduo.configs import load_config
from safeduo.safety.backstop import BackstopConfig, VelocityDamperBackstop
from safeduo.safety.sphere_distance import SphereDistOut
from safeduo.safety.types import ARM_KEYS, DOF_OF, zeros_delta


def test_actual_debit_profile_enforces_historical_closing_budget():
    cfg=load_config('duo_env_a31_pending_debit_guard.yaml')
    declared=cfg['safety']['backstop']
    bs=VelocityDamperBackstop(BackstopConfig(**declared))
    rows=ConstraintRows(d=torch.tensor([[.06]]),
        J={'F':torch.zeros(1,1,14),'U':torch.zeros(1,1,12)},
        cls=torch.ones(1,1),arm_mask=torch.tensor([[[True,False,False,False]]]),
        valid=torch.ones(1,1,dtype=torch.bool),d_min=torch.tensor([[.013]]))
    rows.J['F'][0,0,0]=-1
    current={a:torch.zeros(1,DOF_OF[a]) for a in ARM_KEYS};current['F_L'][0,0]=-.10
    old={a:torch.zeros(1,DOF_OF[a]) for a in ARM_KEYS};old['F_L'][0,0]=.10
    cmd=zeros_delta(1);cmd.delta_q['F_L'][0,0]=.02
    result,_,_=bs.project(cmd,rows,torch.ones(1,4),torch.zeros(1),1/60,
        backlog=current,past_backlogs=[old]*6)
    # Exact one-row endpoint: the retained closing debit reaches the existing
    # 90% authority floor. The previously published profile yields -0.003533.
    assert result.delta_q['F_L'][0,0].item()==pytest.approx(-.0225,abs=1e-6)
    assert declared['backlog_aware'] and declared['predict_backlog']
    assert declared['pending_target_steps']==6


def full_and_selected():
    distance=torch.full((2,41),.20);distance[:,40]=.0066
    dm=torch.full_like(distance,.038);dm[:,40]=.005
    idx=torch.arange(32).expand(2,-1)
    valid=torch.ones((2,32),dtype=torch.bool)
    selected=SphereDistOut(active_pairs=torch.zeros(2,32,4),active_mask=valid,
        active_idx=idx,active_dmin=dm[:,:32],viol_exempt=~valid,
        min_margin={},violation=torch.zeros(2,dtype=torch.bool),
        dists=distance,closing=torch.zeros_like(distance),full_dmin=dm,
        full_viol_exempt=torch.zeros_like(distance,dtype=torch.bool))
    selected.full_viol_exempt[:,40]=True
    return selected,torch.cat([torch.zeros(40),torch.tensor([2.])]),torch.arange(41.)


def test_critical_table_row_survives_full_cross_quota_even_when_contact_allowed():
    original,cls,pids=full_and_selected()
    result=merge_critical_rows(original,original,cls,pids)
    assert result.active_mask.sum(-1).tolist()==[33,33]
    for e in range(2):
        assert set(result.active_idx[e].tolist())==set(range(32))|{40}
        col=torch.where(result.active_idx[e]==40)[0].item()
        assert result.viol_exempt[e,col]
        assert result.active_pairs[e,col,0]==pytest.approx(.0066)
    assert original.active_idx.shape==(2,32)


def test_union_preserves_multiple_critical_pairs_and_masks_padding_without_duplicates():
    original,cls,pids=full_and_selected()
    original.dists[0,35:40]=0
    result=merge_critical_rows(original,original,cls,pids)
    assert result.active_mask.sum(-1).tolist()==[38,33]
    for e in range(2):
        valid_ids=result.active_idx[e][result.active_mask[e]]
        assert valid_ids.unique().numel()==valid_ids.numel()
    assert (result.active_idx[~result.active_mask]==-1).all()
    assert not result.viol_exempt[~result.active_mask].any()


def test_union_is_idempotent_and_requires_full_measurements():
    original,cls,pids=full_and_selected()
    once=merge_critical_rows(original,original,cls,pids)
    twice=merge_critical_rows(original,once,cls,pids)
    assert torch.equal(once.active_idx,twice.active_idx)
    original.dists=None
    with pytest.raises(ValueError,match='complete'): merge_critical_rows(original,once,cls,pids)


def test_capacity_overflow_aborts_instead_of_dropping_critical_rows():
    original,cls,pids=full_and_selected()
    with pytest.raises(ValueError,match='exceeds declared capacity'):
        merge_critical_rows(original,original,cls,pids,capacity=32)
    padded=merge_critical_rows(original,original,cls,pids,capacity=40)
    assert padded.active_idx.shape==(2,40)
    assert padded.active_mask.sum(-1).tolist()==[33,33]
