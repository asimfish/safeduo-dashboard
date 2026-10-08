"""Opt-in clearance separates hands along the beam before raising them."""
from types import SimpleNamespace
import math
import pytest
from safeduo.skills import spec as S
from safeduo.skills.compile import compile_task, objects_for


def task(distance=None, yaw=90):
    group = {'arms': ['U_L', 'U_R'], 'obj': 'beam', 'half_station': 0.12,
             'place_dz': -0.013}
    if distance is not None:
        group['clear_axial_m'] = distance
    sp = S.from_dict({'family': 'clear_test', 'category': 2,
        'objects': [{'name': 'beam', 'catalog': 'alu_profile_4080_300', 'xy': [-0.4, 0.], 'yaw': yaw}],
        'arms': {'U_L': {'do': 'colift'}, 'U_R': {'do': 'colift'}}, 'groups': [group]})
    positions = ({'U_L': (-0.52,0.,1.), 'U_R':(-0.28,0.,1.)} if yaw == 90 else
                 {'U_L': (-0.4,0.12,1.), 'U_R':(-0.4,-0.12,1.)})
    cands = {a: SimpleNamespace(tcp_grasp=p, flange_grasp=p, flange_pre=(p[0], p[1], 1.1),
        R_flange=((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)), notes={})
        for a,p in positions.items()}
    return compile_task(sp, objects_for(sp), cands)


def test_default_has_no_extra_phase():
    assert 'clear_axial' not in [p.name for p in task().spec.phases]


def test_axial_exit_precedes_lift_and_preserves_prior_targets():
    old = {p.name:p for p in task().spec.phases}
    new_task=task(0.07)
    new={p.name:p for p in new_task.spec.phases}
    assert list(new).index('settle') < list(new).index('clear_axial') < list(new).index('clear')
    for name in ['approach','descend','lift','carry','place','hold','release','settle']:
        assert new[name].targets == old[name].targets
    for arm,sign in [('U_L',-1),('U_R',1)]:
        place=old['place'].targets[arm]
        expected=(place[0]+sign*0.07,place[1],place[2])
        assert new['clear_axial'].targets[arm] == pytest.approx(expected)
        assert new['clear'].targets[arm][:2] == pytest.approx(expected[:2])
        assert new['clear'].targets[arm][2] == old['clear'].targets[arm][2]
    assert new_task.hand_events == task().hand_events


@pytest.mark.parametrize('distance',[-0.01,math.nan,math.inf])
def test_invalid_clearance_rejected(distance):
    with pytest.raises((AssertionError,ValueError)):
        task(distance)


def test_y_long_beam_exits_according_to_station_side():
    phases={p.name:p for p in task(0.07,yaw=0).spec.phases}
    for arm,sign in [('U_L',1),('U_R',-1)]:
        place=phases['place'].targets[arm]
        assert phases['clear_axial'].targets[arm] == pytest.approx((place[0],place[1]+sign*0.07,place[2]))
    assert set(phases['clear_axial'].targets) == {'U_L','U_R'}


def test_explicit_zero_preserves_entire_programme():
    assert task(0).spec == task().spec
    assert task(0).hand_events == task().hand_events
