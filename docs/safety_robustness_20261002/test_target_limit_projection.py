"""A safety increment must remain safe after actuator target saturation."""
import torch

from safeduo.baselines.base import ConstraintRows
from safeduo.safety.backstop import BackstopConfig, VelocityDamperBackstop
from safeduo.safety.types import ARM_KEYS, DOF_OF, zeros_delta


def saturated_opening_joint():
    rows=ConstraintRows(d=torch.tensor([[.010]]),d_min=torch.tensor([[.013]]),
        J={'F':torch.zeros(1,1,14),'U':torch.zeros(1,1,12)},cls=torch.ones(1,1),
        arm_mask=torch.tensor([[[True,False,False,False]]]),valid=torch.ones(1,1,dtype=torch.bool))
    rows.J['F'][0,0,:2]=torch.tensor([-1.,1.])
    cmd=zeros_delta(1);cmd.delta_q['F_L'][0,0]=.02
    bounds={a:(torch.full((1,DOF_OF[a]),-.025),torch.full((1,DOF_OF[a]),.025)) for a in ARM_KEYS}
    bounds['F_L'][1][0,1]=0  # the joint that could open is already at target upper limit
    return rows,cmd,bounds


def test_projection_is_safe_in_the_box_actually_applied_to_targets():
    rows,cmd,bounds=saturated_opening_joint()
    out,_,info=VelocityDamperBackstop(BackstopConfig(max_passes=30)).project(
        cmd,rows,torch.ones(1,4),torch.zeros(1),1/60,delta_bounds=bounds)
    u=out.delta_q['F_L']
    assert u[0,1] <= 0
    assert -u[0,0]+u[0,1] >= 4*(.013-.010)/60-1e-6
    assert info['residual_F'][0]<1e-6
    for a in ARM_KEYS:
        lo,hi=bounds[a]
        assert ((out.delta_q[a]>=lo)&(out.delta_q[a]<=hi)).all()


def test_legacy_post_clipping_can_reverse_an_approved_opening_increment():
    rows,cmd,bounds=saturated_opening_joint()
    out,_,_=VelocityDamperBackstop().project(cmd,rows,torch.ones(1,4),torch.zeros(1),1/60)
    u=out.delta_q['F_L'];applied=u.maximum(bounds['F_L'][0]).minimum(bounds['F_L'][1])
    assert -u[0,0]+u[0,1]>0
    assert -applied[0,0]+applied[0,1]<0


def test_unrestricted_target_box_matches_legacy_projection():
    rows,cmd,_=saturated_opening_joint()
    args=(cmd,rows,torch.ones(1,4),torch.zeros(1),1/60)
    bs=VelocityDamperBackstop()
    old,_,_=bs.project(*args)
    bounds={a:(torch.full((1,DOF_OF[a]),-10.),torch.full((1,DOF_OF[a]),10.)) for a in ARM_KEYS}
    new,_,_=bs.project(*args,delta_bounds=bounds)
    for a in ARM_KEYS:
        torch.testing.assert_close(old.delta_q[a],new.delta_q[a],atol=1e-7,rtol=0)


def test_target_box_preserves_the_original_alpha_budget_direction():
    rows,cmd,bounds=saturated_opening_joint()
    rows.J['F'][0,0,:2]=torch.tensor([0.,1.])
    cmd.delta_q['F_L'][0,:2]=.02
    bounds['F_L'][1][0,:2]=torch.tensor([0.,.025])
    alpha=torch.ones(1,4);alpha[0,0]=0
    out,_,_=VelocityDamperBackstop(BackstopConfig(max_passes=30)).project(
        cmd,rows,alpha,torch.zeros(1),1/60,delta_bounds=bounds)
    u=out.delta_q['F_L'][0,:2]
    assert u[1] >= 4*(.013-.010)/60-1e-6
    assert u.sum()<=1e-6
