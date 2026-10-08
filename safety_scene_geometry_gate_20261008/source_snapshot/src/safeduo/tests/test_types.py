"""契约冒烟测试：types.py 形状/复现性/门控直通（Mac cpu 可跑）。"""

import torch

from safeduo.delta.smoke_noise import SmokeNoiseDelta
from safeduo.safety.types import (
    ARM_DOF,
    ARM_KEYS,
    DOF_OF,
    TOTAL_DOF,
    SceneState,
    backstop_project,
    split_stacked,
    zeros_delta,
)


def make_state(n=4):
    q = {k: torch.zeros(n, DOF_OF[k]) for k in ARM_KEYS}
    qd = {k: torch.zeros(n, DOF_OF[k]) for k in ARM_KEYS}
    return SceneState(q=q, qd=qd, dt=0.02)


def test_constants():
    assert ARM_KEYS == ("F_L", "F_R", "U_L", "U_R")
    assert ARM_DOF == (7, 7, 6, 6)
    assert TOTAL_DOF == 26


def test_delta_shapes_roundtrip():
    d = zeros_delta(3)
    s = d.stacked()
    assert s.shape == (3, 26)
    d2 = split_stacked(s + 1.0)
    for k in ARM_KEYS:
        assert d2.delta_q[k].shape == (3, DOF_OF[k])
        assert (d2.delta_q[k] == 1.0).all()


def test_smoke_noise_reproducible_and_bounded():
    state = make_state(8)
    outs = []
    for _ in range(2):
        src = SmokeNoiseDelta(8, amp=0.05)
        src.reset(torch.arange(8), torch.Generator().manual_seed(42))
        outs.append(src.sample(state).stacked())
    assert torch.equal(outs[0], outs[1])
    assert outs[0].abs().max() <= 0.05


def test_backstop_stub_passthrough():
    # v2：契约桩只保形状直通（alpha 语义改为 L2 内部预算，真实现在 safety/backstop.py）
    n = 5
    state = make_state(n)
    delta = zeros_delta(n)
    for k in ARM_KEYS:
        delta.delta_q[k] += 1.0
    alpha = torch.tensor([0.0, 0.5, 1.0, 0.25]).repeat(n, 1)
    out, active = backstop_project(delta, state, alpha, torch.zeros(n))
    for k in ARM_KEYS:
        assert torch.allclose(out.delta_q[k], delta.delta_q[k])
    assert active.shape == (n, 4) and not active.any()
