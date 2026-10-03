import torch
import pytest

from safeduo.eval.perturbations import InitialPoseSource, TargetDelayQueue, TargetDelayAdapter
from safeduo.eval.research_battery import UniformRandomTape
from safeduo.safety.types import ARM_KEYS, DOF_OF


def poses(n=3):
    return {a: torch.zeros(n, DOF_OF[a]) for a in ARM_KEYS}


def test_initial_poses_are_matched_and_do_not_consume_command_randomness():
    base=poses()
    limits={a: torch.stack([q-1, q+1], -1) for a,q in base.items()}
    a=InitialPoseSource(UniformRandomTape(3, 5, .015, 7), base, limits, .3, 13)
    b=InitialPoseSource(UniformRandomTape(3, 5, .015, 7), base, limits, .3, 13)
    assert torch.equal(a.source.tape,UniformRandomTape(3,5,.015,7).tape)
    ids=torch.arange(3)
    a.reset(ids); b.reset(ids)
    _,pa=a.initial_positions(ids); _,pb=b.initial_positions(ids)
    for arm in ARM_KEYS:
        torch.testing.assert_close(pa[arm],pb[arm],rtol=0,atol=0)
        assert pa[arm].abs().max()<=.3
        assert not torch.equal(pa[arm][0],pa[arm][1])
        torch.testing.assert_close(a.sample(None).delta_q[arm],b.sample(None).delta_q[arm],rtol=0,atol=0)


def test_initial_pose_clipping_is_recorded_and_reset_replays():
    base=poses()
    limits={a: torch.stack([q-.02, q+.02], -1) for a,q in base.items()}
    source=InitialPoseSource(UniformRandomTape(3, 5, .015, 7), base, limits, .3, 13)
    ids=torch.arange(3)
    _,position=source.initial_positions(ids)
    assert all(q.abs().max()<=.02 for q in position.values())
    meta=source.coverage_metadata()
    assert all(x['initial_clipped_joints']>0 for x in meta)
    assert all(x['initial_effective_max_abs_rad']<=.020001 for x in meta)
    source.sample(None); source.reset(ids)
    for arm,q in source.initial_positions(ids)[1].items():
        assert torch.equal(q,position[arm])


def test_delay_emits_exact_prior_target_and_copies_queue_entries():
    queue=TargetDelayQueue(2)
    initial={'a':torch.tensor([[0.]])}
    queue.reset(initial)
    first={'a':torch.tensor([[1.]])}
    assert queue.push(first)['a'].item()==0
    first['a'].fill_(99)
    assert queue.push({'a':torch.tensor([[2.]])})['a'].item()==0
    assert queue.push({'a':torch.tensor([[3.]])})['a'].item()==1
    assert queue.push({'a':torch.tensor([[4.]])})['a'].item()==2
    queue.reset(initial)
    assert queue.push({'a':torch.tensor([[5.]])})['a'].item()==0


def test_zero_delay_and_adapter_decimation_preserve_target_sequence():
    queue=TargetDelayQueue(0); queue.reset({'a':torch.tensor([[0.]])})
    assert queue.push({'a':torch.tensor([[3.]])})['a'].item()==3
    class Arm:
        def __init__(self): self.received=[]
        def set_joint_position_target(self,q,joint_ids): self.received.append(q.clone())
    class Env:
        def __init__(self):
            self._targets=poses(1)
            self._arms={a:Arm() for a in ARM_KEYS}
            self._joint_idx={a:torch.arange(DOF_OF[a]) for a in ARM_KEYS}
        def _pre_physics_step(self,actions):
            for a in ARM_KEYS: self._targets[a]=self._targets[a]+actions
        def _apply_action(self):
            for a in ARM_KEYS: self._arms[a].set_joint_position_target(self._targets[a],joint_ids=[])
    env=Env()
    adapter=TargetDelayAdapter(env,1); adapter.reset(env._targets)
    for action in [1.,2.,3.]:
        env._pre_physics_step(action)
        env._apply_action(); env._apply_action()
    assert [q[0,0].item() for q in env._arms['F_L'].received]==[0,0,1,1,3,3]
    assert env._targets['F_L'][0,0].item()==6
    adapter.close()
    env._pre_physics_step(4.); env._apply_action()
    assert env._arms['F_L'].received[-1][0,0].item()==10


def test_zero_delay_adapter_keeps_original_env_methods():
    class Env:
        def _pre_physics_step(self,actions): pass
        def _apply_action(self): pass
    env=Env(); pre=env._pre_physics_step; apply=env._apply_action
    adapter=TargetDelayAdapter(env,0)
    assert env._pre_physics_step==pre and env._apply_action==apply
    adapter.close()
    assert env._pre_physics_step==pre and env._apply_action==apply


@pytest.mark.parametrize('delay',[-1,1.5])
def test_invalid_delay_rejected(delay):
    with pytest.raises(ValueError): TargetDelayQueue(delay)
