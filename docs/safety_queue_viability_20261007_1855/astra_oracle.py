#!/home/liyufeng/miniforge3/envs/safeduo/bin/python -B
"""Independent, passive CPU oracle. No safeduo/torch/robot imports.

G,h are already-selected float32 rows, including alpha rows when applicable.
Their exact binary values define this CURRENT set; no physical model is inferred.
min_x max_i (G_i x-h_i)/s_i is solved numerically on the supplied finite box.
Only exact rational checks can promote its output from UNKNOWN:
  feasible: an explicit x satisfies Gx<=h and lo<=x<=hi;
  infeasible: y>=0 and min_box (y^T G)x - y^T h > 0.
The second check does not require exactly stationary floating-point duals.
Row scaling is only a solver aid. It never changes certification inequalities.

Run with -B to avoid creating __pycache__ outside the permitted filename scope.
Default CLI prints a demo; --bounded-review reads only two preregistered NPZs.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parent
FEASIBLE = "SELECTED_FEASIBLE"
INFEASIBLE = "SELECTED_INFEASIBLE"
UNKNOWN = "UNKNOWN"


def fraction(x):
    return Fraction.from_float(float(x))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite(name, value, shape=None, dtype32=False):
    a = np.asarray(value)
    if a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise ValueError(f"{name}: finite numeric array required")
    if shape is not None and a.shape != shape:
        raise ValueError(f"{name}: shape {shape} required, got {a.shape}")
    if dtype32 and a.dtype != np.dtype("float32"):
        raise ValueError(f"{name}: original float32 coefficients required")
    return a


def _inputs(G, h, lower, upper):
    G = _finite("G", G, dtype32=True)
    if G.ndim != 2 or not 1 <= G.shape[1] <= 26:
        raise ValueError("G: matrix with 1..26 columns required")
    m, n = G.shape
    h = _finite("h", h, (m,), dtype32=True)
    lower = _finite("lower", lower, (n,))
    upper = _finite("upper", upper, (n,))
    return G, h, lower, upper


def _rational(G, h, lower, upper):
    return ([[fraction(v) for v in row] for row in G],
            [fraction(v) for v in h], [fraction(v) for v in lower],
            [fraction(v) for v in upper])


def _primal(data, x):
    A, b, lo, hi = data
    z = [fraction(v) for v in x]
    residuals = [sum((a*v for a, v in zip(row, z)), Fraction()) - rhs
                 for row, rhs in zip(A, b)]
    bound_errors = [max(l-v, v-u) for l, u, v in zip(lo, hi, z)]
    error = max(residuals + bound_errors, default=Fraction())
    return {"verified": error <= 0, "max_residual_exact": str(error),
            "max_residual": float(error)}


def verify_witness(G, h, lower, upper, x):
    try:
        G, h, lower, upper = _inputs(G, h, lower, upper)
        x = _finite("witness", x, (G.shape[1],))
        return _primal(_rational(G, h, lower, upper), x)
    except (ValueError, TypeError, OverflowError) as exc:
        return {"verified": False, "reason": str(exc)}


def _separation(data, weights):
    A, b, lo, hi = data
    y = [fraction(v) for v in weights]
    if any(v < 0 for v in y) or not any(y) or any(l > u for l, u in zip(lo, hi)):
        return {"verified": False, "reason": "nonnegative nonzero y and nonempty box required"}
    nz = [i for i, v in enumerate(y) if v]
    combined = [sum((y[i]*A[i][j] for i in nz), Fraction()) for j in range(len(lo))]
    rhs = sum((y[i]*b[i] for i in nz), Fraction())
    support_min = sum((a*(l if a >= 0 else u) for a, l, u in zip(combined, lo, hi)), Fraction())
    gap = support_min - rhs
    return {"verified": gap > 0, "gap_exact": str(gap), "gap": float(gap),
            "box_min_exact": str(support_min), "weighted_h_exact": str(rhs),
            "weighted_G_exact": [str(v) for v in combined], "support_rows": nz,
            "weights": [float(v) for v in weights],
            "theorem": "y>=0 and min_box(y^T G x)>y^T h excludes every x in the current box"}


def verify_separation(G, h, lower, upper, weights):
    try:
        G, h, lower, upper = _inputs(G, h, lower, upper)
        y = _finite("weights", weights, (len(h),))
        return _separation(_rational(G, h, lower, upper), y)
    except (ValueError, TypeError, OverflowError) as exc:
        return {"verified": False, "reason": str(exc)}


def selected_set(G, h, lower, upper, *, row_labels=None, _solver=linprog):
    """CURRENT selected set only; unknown/error is never silently feasible.

    row_labels are metadata, in exactly the same order as the supplied rows.
    _solver is a test seam for failure semantics, not part of production API.
    """
    result = {"schema": "safeduo.astra.current_set.v1", "status": UNKNOWN,
              "input_state": "INVALID", "scope": "current_supplied_selected_linear_set_only",
              "future_physical_safety": UNKNOWN}
    try:
        G, h, lower, upper = _inputs(G, h, lower, upper)
        m, n = G.shape
        if row_labels is not None and len(row_labels) != m:
            raise ValueError("row_labels length mismatch")
        labels = list(row_labels) if row_labels is not None else list(range(m))
        data = _rational(G, h, lower, upper)
        A, b, lo, hi = data
        result.update(input_state="VALID", row_count=m, dimension=n, row_labels=labels)
        empty = [j for j in range(n) if lo[j] > hi[j]]
        if empty:
            result.update(status=INFEASIBLE, certificate={"kind": "EMPTY_BOX", "joints": empty},
                          reason="supplied command interval is empty")
            return result
        row_slacks = [rhs - sum((a*(l if a >= 0 else u) for a, l, u in zip(row, lo, hi)), Fraction())
                      for row, rhs in zip(A, b)]
        result["individual_infeasible_rows"] = [i for i, v in enumerate(row_slacks) if v < 0]
        result["individual_min_slack_exact"] = [str(v) for v in row_slacks]
        result["all_rows_individually_feasible"] = all(v >= 0 for v in row_slacks)
        low, high = lower.astype(np.float64), upper.astype(np.float64)
        center = low*.5 + high*.5
        if not m:
            witness = center
            check = _primal(data, witness)
            result.update(status=FEASIBLE if check["verified"] else UNKNOWN,
                          witness=witness.tolist(), witness_check=check, lp_status="NO_ROWS_BOX_ONLY")
            return result
        scale = np.max(np.abs(G.astype(np.float64)), axis=1)
        scale[scale == 0] = 1.0
        aug = np.column_stack((G.astype(np.float64)/scale[:, None], -np.ones(m)))
        solve = _solver(np.r_[np.zeros(n), 1.0], A_ub=aug, b_ub=h.astype(np.float64)/scale,
                        bounds=list(zip(low, high)) + [(None, None)], method="highs-ds",
                        options={"primal_feasibility_tolerance": 1e-9,
                                 "dual_feasibility_tolerance": 1e-9})
        result.update(lp_status=int(solve.status), lp_message=str(solve.message))
        if not solve.success or solve.x is None or not np.isfinite(solve.x).all():
            result["reason"] = "no certified solver candidate"
            return result
        result["normalized_phase_value"] = float(solve.x[-1])
        # Clipping a candidate to the ORIGINAL box is permitted; every resulting
        # candidate is rechecked exactly. No row RHS/endpoint is relaxed.
        for kind, witness in (("lp_box_clipped", np.clip(solve.x[:-1], low, high)),
                              ("zero_box_clipped", np.clip(np.zeros(n), low, high)),
                              ("box_center", center)):
            check = _primal(data, witness)
            if check["verified"]:
                result.update(status=FEASIBLE, witness=witness.tolist(), witness_kind=kind,
                              witness_check=check)
                return result
        y = np.maximum(0.0, -np.asarray(solve.ineqlin.marginals, dtype=np.float64))/scale
        if np.isfinite(y).all():
            certificate = _separation(data, y)
            result["certificate"] = certificate
            if certificate["verified"]:
                certificate["support_labels"] = [labels[i] for i in certificate["support_rows"]]
                result.update(status=INFEASIBLE,
                              conflict_kind="INDIVIDUAL" if result["individual_infeasible_rows"] else "JOINT_ONLY")
                return result
        result["reason"] = "numerical LP result has neither exact primal nor separating certificate"
    except Exception as exc:
        # Explicit UNKNOWN preserves fail-closed diagnostic semantics for solver,
        # dtype/shape, overflow and nonfinite faults. This does not stop a robot.
        result["reason"] = f"{type(exc).__name__}: {exc}"
    return result


def diagnose_current(blocks, lower, upper):
    """F14/U12 product; caller supplies actual saved rows including alpha.

    This product assertion requires disjoint command columns with fixed h and p.
    It does not certify an unsplit physical cross-robot constraint.
    """
    out = {"status": UNKNOWN, "scope": "current_fixed_F_U_product_only",
           "future_physical_safety": UNKNOWN}
    try:
        lo, hi = _finite("lower", lower, (26,)), _finite("upper", upper, (26,))
        if set(blocks) != {"F", "U"}:
            raise ValueError("exact F and U block keys required")
        for robot, sl, width in (("F", slice(0, 14), 14), ("U", slice(14, 26), 12)):
            if np.asarray(blocks[robot]["G"]).ndim != 2 or np.asarray(blocks[robot]["G"]).shape[1] != width:
                raise ValueError(f"{robot}: {width} columns required")
        out["robots"] = {r: selected_set(blocks[r]["G"], blocks[r]["h"], lo[s], hi[s],
                                        row_labels=blocks[r].get("row_labels"))
                         for r, s in (("F", slice(0, 14)), ("U", slice(14, 26)))}
        if any(v["input_state"] != "VALID" for v in out["robots"].values()):
            out["reason"] = "invalid block"
        elif any(v["status"] == INFEASIBLE for v in out["robots"].values()):
            out["status"] = INFEASIBLE
        elif all(v["status"] == FEASIBLE for v in out["robots"].values()):
            out["status"] = FEASIBLE
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        out["reason"] = str(exc)
    return out


def queue_local_risk(J, d, dmin, q, pending_targets):
    """Six immutable endpoint probes d-dmin+J(target-q), exact rational sign.

    Negative -> LOCAL_LINEAR_UNSAFE, never a claim of physical violation.
    Nonnegative -> LOCAL_ENDPOINTS_NONNEGATIVE, never a safety guarantee.
    FIFO index is supplied index, not a calibrated time or predicted actual q.
    No interpolation, dynamics, row change, or cancellation is assumed.
    """
    out = {"status": UNKNOWN, "input_state": "INVALID", "future_physical_safety": UNKNOWN,
           "scope": "frozen_J_selected_rows_six_pending_endpoints",
           "pending_count_required": 6, "physical_time_to_failure": None}
    try:
        J = _finite("J", J, dtype32=True)
        if J.ndim != 2 or J.shape[1] != 26 or J.shape[0] == 0:
            raise ValueError("nonempty J with 26 columns required; empty coverage is UNKNOWN")
        m = J.shape[0]
        d = _finite("d", d, (m,), dtype32=True)
        dmin = _finite("dmin", dmin, (m,), dtype32=True)
        q = _finite("q", q, (26,))
        pending = _finite("pending_targets", pending_targets, (6, 26))
        A = [[fraction(v) for v in row] for row in J]
        margin = [fraction(a)-fraction(b) for a, b in zip(d, dmin)]
        z = [fraction(v) for v in q]
        values = []
        for target in pending:
            delta = [fraction(v)-a for v, a in zip(target, z)]
            values.append([b+sum((a*x for a, x in zip(row, delta)), Fraction())
                           for row, b in zip(A, margin)])
        negative = [{"slot": k, "rows": [i for i, v in enumerate(row) if v < 0]}
                    for k, row in enumerate(values) if any(v < 0 for v in row)]
        observed = [i for i, v in enumerate(margin) if v < 0]
        minimum = min(min(row) for row in values)
        out.update(input_state="VALID", status="LOCAL_LINEAR_UNSAFE" if negative else "LOCAL_ENDPOINTS_NONNEGATIVE",
                   current_geometry_status="OBSERVED_UNSAFE" if observed else "OBSERVED_NONNEGATIVE",
                   observed_negative_rows=observed, negative_pending=negative,
                   first_negative_slot=negative[0]["slot"] if negative else None,
                   min_pending_margin=float(minimum), min_pending_margin_exact=str(minimum),
                   minimum_by_slot=[float(min(row)) for row in values],
                   endpoint_margins=[[float(v) for v in row] for row in values])
    except (ValueError, TypeError, OverflowError) as exc:
        out["reason"] = str(exc)
    return out


def bounded_snapshot_review(registration):
    """Only the two named post-outcome failure snapshots, four robot sets."""
    records = registration["bounded_snapshot_review"]["snapshots"]
    if len(records) != 2:
        raise ValueError("bounded protocol requires exactly the two registered snapshots")
    reviewed = []
    for record in records:
        path = Path(record["path"])
        digest = sha256(path)
        if digest != record["sha256"]:
            raise ValueError("frozen snapshot hash mismatch")
        with np.load(path, allow_pickle=False) as data:
            matches = np.flatnonzero(data["env_ids"] == record["env"])
            if len(matches) != 1:
                raise ValueError("one registered env required")
            e = int(matches[0])
            blocks, returned = {}, {}
            for robot, sl in (("F", slice(0, 14)), ("U", slice(14, 26))):
                rel = data[f"snapshot_rel_{robot}"][e]
                ar = data[f"snapshot_alpha_rel_{robot}"][e]
                positions = np.flatnonzero(rel)
                ids = data["selected_ids"][e][:len(rel)]
                if np.any(ids[positions] < 0):
                    raise ValueError("relevant row missing saved global ID")
                G = np.concatenate((data[f"snapshot_G_{robot}"][e][rel], data[f"snapshot_alpha_G_{robot}"][e][ar]))
                h = np.concatenate((data[f"snapshot_h_after_authority_{robot}"][e][rel], data[f"snapshot_alpha_h_{robot}"][e][ar]))
                labels = [f"safety:global={int(ids[p])}:position={int(p)}" for p in positions]
                labels += [f"alpha:local_arm={int(i)}" for i in np.flatnonzero(ar)]
                blocks[robot] = {"G": G, "h": h, "row_labels": labels}
                returned[robot] = verify_witness(G, h, data["bounds_lower"][e][sl], data["bounds_upper"][e][sl],
                                                  data["actual_project_return"][e][sl])
            current = diagnose_current(blocks, data["bounds_lower"][e], data["bounds_upper"][e])
            mask = data["snapshot_rel_F"][e] | data["snapshot_rel_U"][e]
            J = np.concatenate((data["snapshot_J_F"][e], data["snapshot_J_U"][e]), axis=1)[mask]
            queue = queue_local_risk(J, data["snapshot_d"][e][mask], data["snapshot_dmin"][e][mask],
                                     data["pre_q"][e], data["pre_pending_actuator_targets"][e])
            queue["global_ids"] = data["selected_ids"][e][:len(mask)][mask].tolist()
            queue["coverage"] = "union of saved F/U relevant rows only; no re-selection, full-row or leadup audit"
            reviewed.append({"path": str(path), "sha256": digest, "env": record["env"],
                             "step": int(data["step"]), "current": current, "returned_command": returned,
                             "queue_local": queue,
                             "maximum_measured_qd": float(np.max(np.abs(data["snapshot_qd"][e]))) if data["snapshot_qd_available"][e] else None,
                             "original_bytes_unchanged_after_read": sha256(path) == digest})
    return {"schema": "safeduo.astra.bounded_review.v1", "snapshot_count": 2, "robot_set_count": 4,
            "scope": "post_outcome_development_recheck_only", "records": reviewed,
            "future_physical_safety": UNKNOWN}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bounded-review", action="store_true")
    args = parser.parse_args()
    if args.bounded_review:
        result = bounded_snapshot_review(json.loads((ROOT/"ASTRA_PREREGISTRATION.json").read_text()))
    else:
        g = np.ones((2, 26), dtype=np.float32)
        g[1] *= -1
        result = selected_set(g, np.array([-.25, -.25], dtype=np.float32),
                              np.full(26, -1., dtype=np.float32), np.full(26, 1., dtype=np.float32))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
