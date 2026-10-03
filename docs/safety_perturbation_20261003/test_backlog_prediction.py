import pytest
import torch

from safeduo.baselines.base import ConstraintRows
from safeduo.safety.backstop import BackstopConfig, VelocityDamperBackstop
from safeduo.safety.types import ARM_KEYS, DOF_OF, zeros_delta


def project(predict,store,velocity=-.1):
    rows=ConstraintRows(d=torch.tensor([[.06]]),
        J={'F':torch.zeros(1,1,14),'U':torch.zeros(1,1,12)},
        cls=torch.ones(1,1),arm_mask=torch.tensor([[[True,False,False,False]]]),
        valid=torch.ones(1,1,dtype=torch.bool),d_min=torch.tensor([[.013]]))
    rows.J['F'][0,0,0]=-1
    cmd=zeros_delta(1);cmd.delta_q['F_L'][0,0]=.02
    qd={a:torch.zeros(1,DOF_OF[a]) for a in ARM_KEYS};qd['F_L'][0,0]=velocity
    backlog={a:torch.zeros(1,DOF_OF[a]) for a in ARM_KEYS};backlog['F_L'][0,0]=store
    bs=VelocityDamperBackstop(BackstopConfig(lookahead_s=.4,engage_dist=.04,
                                             backlog_aware=True,predict_backlog=predict))
    return bs.project(cmd,rows,torch.ones(1,4),torch.zeros(1),1/60,
                      qd=qd,backlog=backlog)


def test_opening_velocity_does_not_hide_closing_stored_target():
    old,_,_=project(False,.10)
    new,active,_=project(True,.10)
    assert old.delta_q['F_L'][0,0]==pytest.approx(.02)
    assert new.delta_q['F_L'][0,0]<0
    assert active[0,0]


def test_opening_target_cannot_relax_velocity_braking():
    old,_,_=project(False,-.10,velocity=.3)
    new,_,_=project(True,-.10,velocity=.3)
    for a in ARM_KEYS: assert torch.equal(old.delta_q[a],new.delta_q[a])


def test_zero_backlog_retains_legacy_output():
    old,_,_=project(False,0)
    new,_,_=project(True,0)
    for a in ARM_KEYS: assert torch.equal(old.delta_q[a],new.delta_q[a])
