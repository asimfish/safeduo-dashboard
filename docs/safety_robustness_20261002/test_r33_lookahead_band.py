"""R33: velocity-aware damper band (BackstopConfig.lookahead_s).

A cross row at 60 mm sits outside the 40 mm engage gate. With the arm closing
at 0.7 m/s the row must engage (and go into forced retreat) when the predicted
margin over the lag horizon is below the line; a slow approach must stay a
bit-identical passthrough, and lookahead_s=None must be bit-identical to the
legacy damper regardless of qd.
"""
import pytest
import torch

from safeduo.baselines.base import ConstraintRows
from safeduo.safety.backstop import BackstopConfig, VelocityDamperBackstop
from safeduo.safety.types import ARM_KEYS, DOF_OF, zeros_delta

DT = 1 / 60


def one_cross_row(d: float):
    rows = ConstraintRows(
        d=torch.full((1, 1), d),
        J={"F": torch.zeros(1, 1, 14), "U": torch.zeros(1, 1, 12)},
        cls=torch.zeros(1, 1),
        arm_mask=torch.zeros(1, 1, 4, dtype=torch.bool),
        valid=torch.ones(1, 1, dtype=torch.bool),
    )
    rows.J["F"][0, 0, 0] = -1.0     # F_L joint 0 moving + closes the row 1 m per rad
    rows.arm_mask[0, 0, 0] = True
    return rows


def cmd_const(val=0.02):
    d = zeros_delta(1)
    for k in ARM_KEYS:
        d.delta_q[k] += val
    return d


def qd_closing(v: float):
    qd = {a: torch.zeros(1, DOF_OF[a]) for a in ARM_KEYS}
    qd["F_L"][0, 0] = v          # + joint velocity -> margin rate -v (J = -1)
    return qd


def project(cfg, d, qd):
    bs = VelocityDamperBackstop(cfg)
    return bs.project(cmd_const(), one_cross_row(d), torch.ones(1, 4), torch.zeros(1), DT,
                      torch.full((1, 1), 0.03), qd=qd)


def test_none_is_bit_identical_even_with_qd():
    a, act_a, _ = project(BackstopConfig(max_passes=3, engage_dist=0.04), 0.06, None)
    b, act_b, _ = project(BackstopConfig(max_passes=3, engage_dist=0.04), 0.06, qd_closing(0.7))
    assert torch.equal(a.delta_q["F_L"], b.delta_q["F_L"]) and torch.equal(act_a, act_b)
    assert a.delta_q["F_L"][0, 0].item() == pytest.approx(0.02)        # far row: passthrough


def test_fast_closing_row_engages_beyond_gate():
    cfg = BackstopConfig(max_passes=3, engage_dist=0.04, lookahead_s=0.15)
    out, act, _ = project(cfg, 0.06, qd_closing(0.7))
    # predicted margin 0.06 - 0.15*0.7 < 0 -> forced retreat: the closing
    # command is cut (and reversed within the velocity box)
    assert act[0, 0]
    assert out.delta_q["F_L"][0, 0].item() < 0.0


def test_slow_approach_untouched():
    cfg = BackstopConfig(max_passes=3, engage_dist=0.04, lookahead_s=0.15)
    out, act, _ = project(cfg, 0.06, qd_closing(0.05))       # 5 cm/s: predicted 52.5 mm
    assert not act.any()
    assert out.delta_q["F_L"][0, 0].item() == pytest.approx(0.02)


def test_opening_motion_never_relaxes_a_near_row():
    cfg = BackstopConfig(max_passes=3, engage_dist=0.04, lookahead_s=0.15)
    ref, _, _ = project(BackstopConfig(max_passes=3, engage_dist=0.04), 0.035, None)
    out, act, _ = project(cfg, 0.035, qd_closing(-0.7))       # moving away
    assert act[0, 0]
    assert torch.equal(out.delta_q["F_L"], ref.delta_q["F_L"])


def test_self_collision_brakes_before_measured_lag_consumes_margin():
    # Captured failure: self pair 6081, env 6, pre-step 61. The old 60 ms
    # prediction is 104.83 mm and drops the row; 150 ms predicts 39.03 mm
    # and must engage before the stored target drives the pair through zero.
    rows = one_cross_row(0.14869)
    rows.cls.fill_(1)
    rows.d_min = torch.full((1, 1), 0.013)
    cfg = BackstopConfig(max_passes=3, engage_dist=0.04, lookahead_s=0.06,
                         self_lookahead_s=0.15)
    out, active, _ = VelocityDamperBackstop(cfg).project(
        cmd_const(), rows, torch.ones(1, 4), torch.zeros(1), DT,
        qd=qd_closing(0.731))
    assert active[0, 0]
    assert out.delta_q["F_L"][0, 0] < 0.02


def test_self_horizon_keeps_cross_and_table_outputs_unchanged():
    base = BackstopConfig(max_passes=3, engage_dist=0.04, lookahead_s=0.06)
    candidate = BackstopConfig(max_passes=3, engage_dist=0.04, lookahead_s=0.06,
                               self_lookahead_s=0.15)
    for cls in (0, 2):
        rows = one_cross_row(0.08)
        rows.cls.fill_(cls)
        args = (cmd_const(), rows, torch.ones(1, 4), torch.zeros(1), DT)
        old, a, _ = VelocityDamperBackstop(base).project(*args, qd=qd_closing(0.7))
        new, b, _ = VelocityDamperBackstop(candidate).project(*args, qd=qd_closing(0.7))
        assert torch.equal(a, b)
        for arm in ARM_KEYS:
            assert torch.equal(old.delta_q[arm], new.delta_q[arm])


@pytest.mark.parametrize('speed', [0.05, -0.7])
def test_safe_or_opening_self_pair_preserves_commands(speed):
    rows = one_cross_row(0.06)
    rows.cls.fill_(1)
    cfg = BackstopConfig(max_passes=3, engage_dist=0.04, lookahead_s=0.06,
                         self_lookahead_s=0.15)
    out, active, _ = VelocityDamperBackstop(cfg).project(
        cmd_const(), rows, torch.ones(1, 4), torch.zeros(1), DT,
        qd=qd_closing(speed))
    assert not active.any()
    for arm in ARM_KEYS:
        assert torch.equal(out.delta_q[arm], cmd_const().delta_q[arm])


def test_self_override_cannot_shorten_existing_horizon():
    rows = one_cross_row(0.06)
    rows.cls.fill_(1)
    args = (cmd_const(), rows, torch.ones(1, 4), torch.zeros(1), DT)
    base = BackstopConfig(max_passes=3, lookahead_s=0.15)
    candidate = BackstopConfig(max_passes=3, lookahead_s=0.15, self_lookahead_s=0.06)
    old, a, _ = VelocityDamperBackstop(base).project(*args, qd=qd_closing(0.7))
    new, b, _ = VelocityDamperBackstop(candidate).project(*args, qd=qd_closing(0.7))
    assert torch.equal(a, b)
    for arm in ARM_KEYS:
        assert torch.equal(old.delta_q[arm], new.delta_q[arm])


@pytest.mark.parametrize('contact_exempt,structural,d,dm,speed', [
    (False, False, .04170, .020, .253),
    (True, True, .012, .005, .070),
])
def test_table_lag_and_conditional_contact_brake_before_collision(contact_exempt,structural,d,dm,speed):
    rows = one_cross_row(d)
    rows.cls.fill_(2)
    rows.d_min = torch.full((1, 1), dm)
    cfg = BackstopConfig(max_passes=3, engage_dist=.04, lookahead_s=.06,
                         table_lookahead_s=.15, exempt_structural_rows=True)
    out, active, _ = VelocityDamperBackstop(cfg).project(
        cmd_const(), rows, torch.ones(1,4), torch.zeros(1), DT,
        qd=qd_closing(speed), struct_exempt=torch.tensor([[structural]]),
        contact_exempt=torch.tensor([[contact_exempt]]))
    assert active[0,0]
    assert out.delta_q['F_L'][0,0] < 0


def test_table_horizon_preserves_permanent_structural_response():
    rows=one_cross_row(.00176)
    rows.cls.fill_(2)
    rows.d_min=torch.full((1,1),.001)
    kw=dict(max_passes=3,engage_dist=.04,lookahead_s=.06,exempt_structural_rows=True)
    args=(cmd_const(),rows,torch.ones(1,4),torch.zeros(1),DT)
    extra=dict(qd=qd_closing(.015),struct_exempt=torch.ones(1,1,dtype=torch.bool),
               contact_exempt=torch.zeros(1,1,dtype=torch.bool))
    old,a,_=VelocityDamperBackstop(BackstopConfig(**kw)).project(*args,**extra)
    new,b,_=VelocityDamperBackstop(BackstopConfig(**kw,table_lookahead_s=.15)).project(*args,**extra)
    assert torch.equal(a,b)
    for arm in ARM_KEYS:
        assert torch.equal(old.delta_q[arm],new.delta_q[arm])


@pytest.mark.parametrize('d,speed',[(.06,.05),(.035,-.7)])
def test_table_safe_and_opening_motion_matches_existing_damper(d,speed):
    rows=one_cross_row(d)
    rows.cls.fill_(2)
    rows.d_min=torch.full((1,1),.020)
    args=(cmd_const(),rows,torch.ones(1,4),torch.zeros(1),DT)
    kw=dict(max_passes=3,engage_dist=.04,lookahead_s=.06)
    old,a,_=VelocityDamperBackstop(BackstopConfig(**kw)).project(*args,qd=qd_closing(speed))
    new,b,_=VelocityDamperBackstop(BackstopConfig(**kw,table_lookahead_s=.15)).project(*args,qd=qd_closing(speed))
    assert torch.equal(a,b)
    for arm in ARM_KEYS:
        assert torch.equal(old.delta_q[arm],new.delta_q[arm])


def test_conditional_contact_keeps_positive_cap_damping_before_velocity_exemption_ends():
    rows = one_cross_row(.012)
    rows.cls.fill_(2)
    rows.d_min = torch.tensor([[.005]])
    cfg = BackstopConfig(max_passes=3, exempt_structural_rows=True, lookahead_s=.06)
    cfg.retain_conditional_rows = True
    out, active, _ = VelocityDamperBackstop(cfg).project(
        cmd_const(), rows, torch.ones(1,4), torch.zeros(1), DT,
        qd=qd_closing(0), struct_exempt=torch.ones(1,1,dtype=torch.bool),
        contact_exempt=torch.ones(1,1,dtype=torch.bool))
    # An allowed slow contact does not authorize a new 1.2 m/s closing command.
    assert active[0,0]
    assert out.delta_q['F_L'][0,0] <= 4 * (.012 - .005) * DT + 1e-7
