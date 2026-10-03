import pytest
import torch

from stop_latch import StopLatch
from safeduo.eval.perturbations import TargetDelayQueue


@pytest.mark.parametrize('mode', ['stored', 'measured'])
def test_one_shot_fixed_target_waits_entire_fifo(mode):
    triggers = [2, -1]
    latch = StopLatch(triggers, mode)
    queue = TargetDelayQueue(6)
    initial = torch.tensor([[0.], [0.]])
    queue.reset({'arm': initial})
    issued, delivered = [], []
    for t in range(12):
        prior = torch.full((2, 1), float(t))
        measured = torch.full((2, 1), float(t + 20))
        proposed = torch.full((2, 1), float(t + 1))
        target = latch.update(t, prior, measured, proposed)
        issued.append(target.clone())
        delivered.append(queue.push({'arm': target})['arm'])
    fixed = 2. if mode == 'stored' else 22.
    assert all(x[0, 0] == fixed for x in issued[2:])
    assert all(x[1, 0] == t + 1 for t, x in enumerate(issued))
    assert delivered[7][0, 0] == issued[1][0, 0]
    assert delivered[8][0, 0] == fixed
    assert all(torch.equal(delivered[t], initial if t < 6 else issued[t - 6]) for t in range(12))


def test_stop_target_is_a_clone_of_trigger_state():
    latch = StopLatch([0], 'measured')
    q = torch.tensor([[.3, -.7]])
    latch.update(0, q + 1, q, q + 2)
    q.zero_()
    assert torch.equal(latch.update(1, q, q, q), torch.tensor([[.3, -.7]]))


@pytest.mark.parametrize('mode,steps', [('unknown', [1]), ('measured', [-2]), ('stored', [[1]])])
def test_invalid_recipe_rejected(mode, steps):
    with pytest.raises(ValueError):
        StopLatch(steps, mode)
