"""CPU regressions for six-arm-pair exposure and empty-subset statistics."""
import json

import pytest
import torch

from safeduo.eval.random_battery import (
    ARM_PAIRS, CLASS_KEYS, PairMarginProbe, RowViolationAudit, aggregate, clopper_pearson_upper,
    summarize_window,
)
from safeduo.safety.sphere_distance import ArmSpheres, LinkSpheres, SphereDistanceModule
from safeduo.safety.types import ARM_KEYS


def test_probe_measures_same_robot_arm_pairs():
    specs = {k: ArmSpheres(links=[LinkSpheres("b0", [(0., 0., 0.)], [0.1])])
             for k in ARM_KEYS}
    sph = SphereDistanceModule(specs)
    centers = torch.tensor([[[0., 0., 1.], [0.25, 0., 1.],
                             [1., 0., 1.], [1.3, 0., 1.]]])
    sph.last_centers = centers
    actual = PairMarginProbe(sph).pair_min()
    expected = torch.tensor([[(centers[0, i] - centers[0, j]).norm().item() - 0.2
                              for i, j in ARM_PAIRS]])
    torch.testing.assert_close(actual, expected)


def test_empty_exposure_is_unavailable_in_window_and_aggregate():
    z = lambda *shape: torch.zeros(*shape, dtype=torch.long)
    acc = {"steps": 3, "wall_s": 1., "safe": z(2), "viol_any": z(2),
           "viol_cross": z(2), "near": z(2), "pair_warn": z(2, 6),
           "locked": z(2, 4), "false_brake": z(2, 4),
           "min_pair": torch.ones(2, 6),
           "min_cls": {k: torch.ones(2) for k in CLASS_KEYS},
           "viol_cls": {k: z(2) for k in CLASS_KEYS}}
    row = summarize_window(acc, {"flow": "l1_full", "dt": 1 / 60}, 3)
    assert row["exposure"]["violation_cp95_upper_exposed"] is None
    summary = aggregate([row])
    assert summary["violation_cp95_upper_exposed"] is None
    json.dumps([row, summary], allow_nan=False)


def test_cp_boundary_values():
    assert clopper_pearson_upper(0, 0) is None
    assert clopper_pearson_upper(0, 100) == pytest.approx(1 - 0.05 ** 0.01)
    assert clopper_pearson_upper(100, 100) == 1.


def test_missing_arm_pair_rejects_incomplete_audit():
    specs = {k: ArmSpheres(links=[LinkSpheres("b0", [(0., 0., 0.)], [0.1])])
             for k in ARM_KEYS}
    sph = SphereDistanceModule(specs)
    sph.pairs_self = sph.pairs_self[:0]
    with pytest.raises(ValueError, match="F_L-F_R"):
        PairMarginProbe(sph)


def _one_sphere_module():
    specs = {k: ArmSpheres(links=[LinkSpheres("b0", [(0., 0., 0.)], [0.1])])
             for k in ARM_KEYS}
    return SphereDistanceModule(specs)


def test_row_audit_official_matches_geometry_and_has_no_structural_rows_by_default():
    sph = _one_sphere_module()
    # F_L/F_R overlap (self_F row), U_L/U_R far apart, F-U cross rows far apart
    sph.last_centers = torch.tensor([[[0., 0., 1.], [0.15, 0., 1.],
                                      [2., 0., 1.], [2.5, 0., 1.]]])
    audit = RowViolationAudit(sph)
    assert audit.n_rows == len(sph.pairs_cross) + len(sph.pairs_self) == 6
    assert not bool(audit.struct.any())            # legacy path: every row sits on the class d_min
    d, viol = audit.step()
    assert viol.shape == (1, 6)
    assert int(viol.sum()) == 1
    hit = int(torch.nonzero(viol[0]).item())
    assert audit.cls[hit] == 1                    # self_F
    assert d[0, hit] == pytest.approx(0.15 - 0.2)
    top = audit.top_rows(viol[0].long())
    assert top["self_F"][0]["steps"] == 1 and top["self_F"][0]["structural"] is False
    assert top["cross"] == [] and top["self_U"] == []


def test_row_audit_excludes_structural_rows_from_second_caliber():
    sph = _one_sphere_module()
    sph.last_centers = torch.tensor([[[0., 0., 1.], [0.15, 0., 1.],
                                      [2., 0., 1.], [2.5, 0., 1.]]])
    probe_rows = torch.cat([sph.pairs_cross, sph.pairs_self], dim=0)
    # mark the overlapping F_L-F_R row as an R29 structural row (per-row d_min override)
    self_f = [r for r in range(len(probe_rows)) if set(probe_rows[r].tolist()) == {0, 1}][0]
    sph.pair_dmin[self_f] = sph.dmin_default["self"] + 0.005
    audit = RowViolationAudit(sph)
    assert int(audit.struct.sum()) == 1 and bool(audit.struct[self_f])
    _, viol = audit.step()
    ns = viol & ~audit.struct.unsqueeze(0)
    assert int(viol.sum()) == 1 and int(ns.sum()) == 0
    assert audit.top_rows(viol[0].long())["self_F"][0]["structural"] is True


def test_row_audit_table_structural_rows_are_second_caliber_only():
    sph = _one_sphere_module()
    pt = len(sph.pairs_table)
    assert pt > 0
    audit = RowViolationAudit(sph)
    assert audit.n_table_rows == pt and not bool(audit.struct_table.any())
    # per-link table d_min override on row 0 (the v6 structural hover link) -> structural
    sph.pair_dmin[sph._slice_table][0] = sph.dmin_default["table"] + 0.005
    audit = RowViolationAudit(sph)
    assert int(audit.struct_table.sum()) == 1 and bool(audit.struct_table[0])
    sph.last_table_margin = torch.full((1, pt), 0.05)
    sph.last_table_margin[0, 0] = -0.002                       # the structural row hovers inside
    sph.last_table_viol_exempt = torch.zeros(1, pt, dtype=torch.bool)
    d, viol = audit.step_table()
    assert int(viol.sum()) == 1 and bool(viol[0, 0])
    assert int((viol & ~audit.struct_table.unsqueeze(0)).sum()) == 0
    # the near_table exemption is still honoured by the official per-row flag
    sph.last_table_viol_exempt[0, 0] = True
    _, viol = audit.step_table()
    assert int(viol.sum()) == 0
    top = audit.top_rows(torch.zeros(audit.n_rows, dtype=torch.long), table_steps=torch.tensor([3] + [0] * (pt - 1)))
    assert top["table"][0]["structural"] is True and top["table"][0]["steps"] == 3
    assert "|" in top["table"][0]["pair"]


def test_summary_reports_both_calibers_and_serializes():
    sph = _one_sphere_module()
    sph.last_centers = torch.tensor([[[0., 0., 1.], [0.15, 0., 1.],
                                      [2., 0., 1.], [2.5, 0., 1.]]] * 2)
    audit = RowViolationAudit(sph)
    z = lambda *shape: torch.zeros(*shape, dtype=torch.long)
    acc = {"steps": 5, "wall_s": 1., "safe": z(2), "viol_any": torch.tensor([5, 0]),
           "viol_cross": z(2), "near": z(2), "pair_warn": z(2, 6),
           "locked": z(2, 4), "false_brake": z(2, 4),
           "min_pair": torch.ones(2, 6),
           "min_cls": {k: torch.ones(2) for k in CLASS_KEYS},
           "viol_cls": {k: z(2) for k in CLASS_KEYS},
           "viol_ns_any": torch.tensor([0, 0]),
           "viol_ns_cls": {k: z(2) for k in CLASS_KEYS},
           "min_ns_cls": {k: torch.ones(2) for k in CLASS_KEYS},
           "row_viol": torch.tensor([0, 0, 0, 0, 5, 0]), "row_viol_table": z(audit.n_table_rows),
           "audit_mismatch": z(2), "audit": audit}
    row = summarize_window(acc, {"flow": "l1_full", "dt": 1 / 60}, 3)
    assert row["violation_episodes"] == 1
    assert row["non_structural"]["violation_episodes"] == 0
    assert row["non_structural"]["table_rows_total"] == audit.n_table_rows
    assert row["non_structural"]["top_rows"]["table"] == []
    assert row["non_structural"]["violation_cp95_upper"] == pytest.approx(1 - 0.05 ** 0.5)
    assert row["non_structural"]["official_recompute_mismatch_steps"] == 0
    summary = aggregate([row])
    assert summary["non_structural"]["violation_episodes"] == 0
    assert summary["non_structural"]["by_flow"]["l1_full"]["episodes"] == 2
    json.dumps([row, summary], allow_nan=False)


def test_summary_without_row_audit_keys_is_backward_compatible():
    z = lambda *shape: torch.zeros(*shape, dtype=torch.long)
    acc = {"steps": 3, "wall_s": 1., "safe": z(2), "viol_any": z(2),
           "viol_cross": z(2), "near": z(2), "pair_warn": z(2, 6),
           "locked": z(2, 4), "false_brake": z(2, 4),
           "min_pair": torch.ones(2, 6),
           "min_cls": {k: torch.ones(2) for k in CLASS_KEYS},
           "viol_cls": {k: z(2) for k in CLASS_KEYS}}
    row = summarize_window(acc, {"flow": "l1_full", "dt": 1 / 60}, 3)
    assert "non_structural" not in row
    assert aggregate([row])["non_structural"] is None
