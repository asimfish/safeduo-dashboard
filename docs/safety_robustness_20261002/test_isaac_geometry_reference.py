"""Independent rotating-point oracle for PhysX COM Jacobian reference."""
from types import SimpleNamespace

import torch

from safeduo.safety.geometry import IsaacGeometryProvider
from safeduo.safety.types import ARM_KEYS


def rotating_fixture():
    # One z-axis hinge, link origin at the hinge, COM=(.2,.3,0),
    # sphere=(.4,.3,0). PhysX linear Jacobian at COM is (-.3,.2,0).
    jac = torch.zeros(1, 1, 6, 1)
    jac[0, 0, :3, 0] = torch.tensor([-.3, .2, 0.])
    jac[0, 0, 5, 0] = 1.
    arms = {a: SimpleNamespace(
        root_physx_view=SimpleNamespace(get_jacobians=lambda: jac),
        data=SimpleNamespace(body_com_pos_w=torch.tensor([[[0., 0., 0.], [.2, .3, 0.]]])))
        for a in ARM_KEYS}
    module = SimpleNamespace(
        n_spheres=4,
        _arm_slices={a: slice(i, i + 1) for i, a in enumerate(ARM_KEYS)},
        _body_idx={a: torch.tensor([1]) for a in ARM_KEYS},
        arm_id=torch.arange(4), pair_table=torch.tensor([[0, 0]]),
        _slice_table=slice(0, 1),
        last_centers=torch.tensor([[[.4, .3, 0.]] * 4]),
        last_table_grad=torch.tensor([[[0., 1., 0.]]]),
    )
    out = SimpleNamespace(
        active_idx=torch.tensor([[0]]), active_mask=torch.tensor([[True]]),
        active_pairs=torch.tensor([[[.1, 0., 2., 0.]]]), active_dmin=torch.tensor([[.02]]),
    )
    provider = IsaacGeometryProvider(arms, {a: torch.tensor([0]) for a in ARM_KEYS}, module, 'cpu')
    return provider, out, {a: torch.zeros(1, 2, 3) for a in ARM_KEYS}


def test_com_referenced_row_matches_rotating_point_finite_difference():
    provider, out, body_pos = rotating_fixture()
    provider.jacobian_reference = 'com'
    actual = provider.rows_from(out, body_pos).J['F'][0, 0, 0]
    eps = 1e-5
    def rotated_y(theta):
        theta = torch.tensor(theta, dtype=torch.float64)
        return .4 * theta.sin() + .3 * theta.cos()
    expected = (rotated_y(eps) - rotated_y(-eps)) / (2 * eps)
    torch.testing.assert_close(actual.double(), expected, atol=1e-7, rtol=1e-7)


def test_legacy_reference_remains_explicitly_reproducible():
    provider, out, body_pos = rotating_fixture()
    provider.jacobian_reference = 'link'
    assert torch.isclose(provider.rows_from(out, body_pos).J['F'][0, 0, 0], torch.tensor(.6))


def test_fixed_base_has_zero_jacobian_even_with_offset():
    provider, _, _ = rotating_fixture()
    jac = provider._sphere_jacobian('F_L', torch.tensor([[0]]), torch.tensor([[[2., 3., 4.]]]))
    assert torch.count_nonzero(jac) == 0
