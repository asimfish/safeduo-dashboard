import torch
import pytest

from safeduo.eval.research_battery import UniformRandomTape, cell_seed
from safeduo.safety.types import DOF_OF


def test_uniform_tape_is_reproducible_and_bounded():
    a = UniformRandomTape(3, 4, 0.015, cell_seed(2, "uniform_random", 0.015))
    b = UniformRandomTape(3, 4, 0.015, cell_seed(2, "uniform_random", 0.015))
    torch.testing.assert_close(a.tape, b.tape)
    assert a.sha256 == b.sha256
    assert tuple(a.tape.shape) == (6, 3, sum(DOF_OF.values()))
    assert float(a.tape.abs().max()) <= 0.015


def test_cell_seed_does_not_depend_on_python_hash_randomization():
    assert cell_seed(0, "uniform_random", 0.005) == cell_seed(0, "uniform_random", 0.005)
    assert cell_seed(0, "uniform_random", 0.005) != cell_seed(1, "uniform_random", 0.005)
    assert cell_seed(0, "uniform_random", 0.005) != cell_seed(0, "uniform_random", 0.015)


def test_held_random_preserves_independent_joint_intent_during_each_hold():
    tape = UniformRandomTape(3, 31, .015, cell_seed(9, 'held_random', .015), hold_steps=10)
    assert tape.tape.shape == (33, 3, 26)
    for first in (0, 10, 20, 30):
        block = tape.tape[first:first + 10]
        assert torch.equal(block, block[0].expand_as(block))
    assert not torch.equal(tape.tape[0], tape.tape[10])
    assert (tape.tape > 0).any() and (tape.tape < 0).any()
    assert tape.tape.abs().max() <= .015
    repeat = UniformRandomTape(3, 31, .015, cell_seed(9, 'held_random', .015), hold_steps=10)
    assert repeat.sha256 == tape.sha256


def test_held_tape_reset_replays_the_same_commands():
    tape = UniformRandomTape(2, 8, .015, 19, hold_steps=3)
    first = tape.sample(None).delta_q
    tape.sample(None)
    tape.reset(torch.arange(2))
    for arm, value in tape.sample(None).delta_q.items():
        assert torch.equal(value, first[arm])


def test_iid_tape_retains_original_sampling_sequence():
    seed = cell_seed(8, 'uniform_random', .015)
    expected = (torch.rand(12, 2, 26, generator=torch.Generator().manual_seed(seed)) * 2 - 1) * .015
    assert torch.equal(UniformRandomTape(2, 10, .015, seed).tape, expected)


@pytest.mark.parametrize('hold', [0, -1, 1.5])
def test_invalid_hold_duration_rejected(hold):
    with pytest.raises(ValueError):
        UniformRandomTape(2, 10, .015, 19, hold_steps=hold)
