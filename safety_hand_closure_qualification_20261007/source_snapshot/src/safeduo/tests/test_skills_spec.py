"""GroupSpec knobs added for the R37 co-lift diagnosis (jig rails, place_dz hang compensation)."""
from safeduo.skills import spec as S

_BASE = {
    "family": "t", "category": 4,
    "objects": [{"name": "beam", "catalog": "alu_profile_4080_700", "xy": [0.53, 0.0]}],
    "arms": {"F_L": {"do": "colift", "obj": "beam"}, "F_R": {"do": "colift", "obj": "beam"}},
}


def _spec(group):
    d = dict(_BASE)
    d["groups"] = [dict({"arms": ["F_L", "F_R"], "obj": "beam"}, **group)]
    return S.from_dict(d)


def test_group_defaults_keep_no_overshoot_and_no_jig():
    g = _spec({}).groups[0]
    assert g.jig is False
    assert g.place_dz == 0.0
    assert g.carry_dxy == (-0.16, 0.0)


def test_group_knobs_are_parsed():
    g = _spec({"jig": True, "place_dz": -0.015, "carry_dxy": [-0.1, 0.0]}).groups[0]
    assert g.jig is True
    assert abs(g.place_dz + 0.015) < 1e-12
    assert g.carry_dxy == (-0.1, 0.0)
