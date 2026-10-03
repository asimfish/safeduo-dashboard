import pytest
import torch

from bounded_stop import BoundedStopLatch
from safeduo.eval.perturbations import TargetDelayQueue


def test_fixed_measured_goal_bounded_approach_and_real_fifo():
    latch = BoundedStopLatch([2, -1], .025, torch.full((2, 2), -1.), torch.full((2, 2), 1.))
    queue = TargetDelayQueue(6)
    prior = torch.tensor([[.7, -.3], [0., 0.]])
    initial = prior.clone()
    queue.reset({'arm': prior})
    sent, delivered = [], []
    for t in range(30):
        measured = torch.tensor([[.2, 2.], [.4, .4]]) if t == 2 else torch.zeros((2, 2))
        proposed = prior + .01
        actual = latch.update(t, prior, measured, proposed)
        if t >= 2:
            assert torch.max(torch.abs(actual[0] - prior[0])) <= .0250005
            assert (actual[0] <= 1).all() and (actual[0] >= -1).all()
            assert torch.equal(latch.fixed[0], torch.tensor([.2, 1.]))
        else:
            assert torch.equal(actual, proposed)
        assert torch.equal(actual[1], proposed[1])
        sent.append(actual.clone())
        delivered.append(queue.push({'arm': actual})['arm'].clone())
        prior = actual
    assert torch.equal(sent[-1][0], torch.tensor([.2, .4])) is False
    assert sent[-1][0, 0].item() == pytest.approx(.2)
    assert sent[-1][0, 1] < 1  # longer axis still approaches; no direct jump
    assert all(torch.equal(delivered[t], initial if t < 6 else sent[t-6]) for t in range(30))
    assert torch.equal(delivered[7], sent[1])
    assert not torch.equal(sent[2][0], latch.fixed[0])


def test_goal_converges_then_stays_fixed_despite_new_measurements():
    latch = BoundedStopLatch([0], .025, torch.tensor([[-1.]]), torch.tensor([[1.]]))
    prior = torch.tensor([[.2]])
    for t in range(10):
        measured = torch.tensor([[.1 if t == 0 else -.8]])
        prior = latch.update(t, prior, measured, torch.tensor([[.9]]))
    assert prior.item() == pytest.approx(.1)
    assert latch.fixed.item() == pytest.approx(.1)


@pytest.mark.parametrize('cap', [0, -.1, float('nan'), float('inf')])
def test_invalid_bound_fails(cap):
    with pytest.raises(ValueError):
        BoundedStopLatch([0], cap, torch.tensor([[-1.]]), torch.tensor([[1.]]))
