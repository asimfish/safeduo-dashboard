import torch

from safeduo.safety.target_history import PendingTargetHistory
from safeduo.safety.backstop import BackstopConfig, VelocityDamperBackstop
from safeduo.baselines.base import ConstraintRows
from safeduo.safety.types import ARM_KEYS, DOF_OF, zeros_delta


def test_history_keeps_only_pending_window_and_resets_selected_envs():
    h=PendingTargetHistory(2)
    initial={'a':torch.tensor([[0.],[10.]])}
    h.reset(initial)
    q={'a':torch.tensor([[1.],[11.]])};h.append(q);q['a'].fill_(99)
    h.append({'a':torch.tensor([[2.],[12.]])})
    assert [x['a'][:,0].tolist() for x in h.targets]==[[1,11],[2,12]]
    h.reset({'a':torch.tensor([[3.],[13.]])},torch.tensor([1]))
    assert [x['a'][:,0].tolist() for x in h.targets]==[[1,13],[2,13]]


def test_safe_current_target_does_not_hide_older_closing_target():
    rows=ConstraintRows(d=torch.tensor([[.06]]),
        J={'F':torch.zeros(1,1,14),'U':torch.zeros(1,1,12)},
        cls=torch.ones(1,1),arm_mask=torch.tensor([[[True,False,False,False]]]),
        valid=torch.ones(1,1,dtype=torch.bool),d_min=torch.tensor([[.013]]))
    rows.J['F'][0,0,0]=-1
    cmd=zeros_delta(1);cmd.delta_q['F_L'][0,0]=.02
    current={a:torch.zeros(1,DOF_OF[a]) for a in ARM_KEYS};current['F_L'][0,0]=-.10
    old={a:torch.zeros(1,DOF_OF[a]) for a in ARM_KEYS};old['F_L'][0,0]=.10
    bs=VelocityDamperBackstop(BackstopConfig(engage_dist=.04,backlog_aware=True,predict_backlog=True))
    baseline,_,_=bs.project(cmd,rows,torch.ones(1,4),torch.zeros(1),1/60,backlog=current)
    pending,_,_=bs.project(cmd,rows,torch.ones(1,4),torch.zeros(1),1/60,
                           backlog=current,past_backlogs=[old])
    assert baseline.delta_q['F_L'][0,0]>0
    assert pending.delta_q['F_L'][0,0]<0
