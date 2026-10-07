#!/home/liyufeng/miniforge3/envs/safeduo/bin/python -B
"""Run preregistered CPU properties; persist every case and retain failed runs.

No robot/simulator execution. No snapshot re-review in this suite.
Output is restricted to astra_test_results*.json beside this script.
"""
import sys
sys.dont_write_bytecode = True

from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import traceback
from types import SimpleNamespace

import numpy as np
import scipy

from astra_oracle import (ROOT, FEASIBLE, INFEASIBLE, UNKNOWN, diagnose_current,
                          fraction, queue_local_risk, selected_set, sha256,
                          verify_separation, verify_witness)

F32 = np.float32


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def vec(value, n=26):
    return np.full(n, value, dtype=F32)


def digest_inputs(data):
    digest = hashlib.sha256()
    for k, value in sorted(data.items()):
        digest.update(k.encode())
        if isinstance(value, np.ndarray):
            digest.update(str(value.dtype).encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes(order="C"))
        else:
            digest.update(json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()


def brief(result):
    cert = result.get("certificate", {})
    return {"status": result["status"], "lp_status": result.get("lp_status"),
            "phase_value": result.get("normalized_phase_value"),
            "all_rows_individually_feasible": result.get("all_rows_individually_feasible"),
            "witness_verified": result.get("witness_check", {}).get("verified"),
            "certificate_verified": cert.get("verified"), "gap_exact": cert.get("gap_exact"),
            "reason": result.get("reason")}


def certified(result, G, h, lo, hi, expected):
    require(result["status"] == expected, f"expected {expected}: {brief(result)}")
    if expected == FEASIBLE:
        require(verify_witness(G, h, lo, hi, result["witness"])["verified"], "primal replay failed")
    else:
        require(verify_separation(G, h, lo, hi, result["certificate"]["weights"])["verified"],
                "separation replay failed")


def generate(name, rng):
    if name == "dense_feasible":
        G = rng.uniform(-1, 1, (40, 26)).astype(F32)
        c, w = rng.uniform(-.2, .2, 26), rng.uniform(.05, .2, 26)
        x = c + rng.uniform(-.2, .2, 26)*w
        h = (G.astype(float)@x + rng.uniform(.02, .08, 40)).astype(F32)
        return dict(G=G, h=h, lo=(c-w).astype(F32), hi=(c+w).astype(F32), planted=x)
    if name == "collective_infeasible":
        g = rng.uniform(-1, 1, 26).astype(F32)
        c, w = rng.uniform(-.05, .05, 26), rng.uniform(.1, .3, 26)
        R, t = np.abs(g.astype(float))@w, g.astype(float)@c
        eta = rng.uniform(.1, .25)*R
        lo, hi = (c-w).astype(F32), (c+w).astype(F32)
        extra = rng.uniform(-1, 1, (24, 26)).astype(F32)
        rhs = np.where(extra >= 0, extra.astype(float)*hi, extra.astype(float)*lo).sum(1) + rng.uniform(.01, .03, 24)
        return dict(G=np.vstack((g, -g, extra)), h=np.r_[t-eta, -t-eta, rhs].astype(F32), lo=lo, hi=hi)
    if name == "zero_hull":
        lo, hi = -rng.uniform(.02, .08, 26).astype(F32), rng.uniform(.02, .08, 26).astype(F32)
        rlo, rhi = rng.uniform(.1, .3, 26)*hi, rng.uniform(.5, .9, 26)*hi
        g = rng.uniform(.2, 1.2, 26).astype(F32)
        eta = rng.uniform(.2, .5)*(g.astype(float)@(-lo))
        return dict(G=g[None], h=np.array([-eta], dtype=F32), lo=lo, hi=hi,
                    ref_lo=rlo.astype(F32), ref_hi=rhi.astype(F32))
    if name == "command_not_velocity":
        q = rng.uniform(-.25, .25, 26).astype(F32)
        v = (rng.choice([-1., 1.], 26)*rng.uniform(2, 4, 26)).astype(F32)
        J = (-v.astype(float)/np.linalg.norm(v)).astype(F32)[None]
        fake, true = 1.5*np.abs(J.astype(float)).sum(), float(-J.astype(float)[0]@v)
        margin = .12*(fake + rng.uniform(.2, .8)*(true-fake))
        command = rng.uniform(-.03, .03, 26).astype(F32)
        pending = (q.astype(float) + rng.uniform(-.001, .001, (6, 26))).astype(F32)
        return dict(J=J, d=np.array([margin], dtype=F32), dmin=np.zeros(1, dtype=F32),
                    q=q, v=v, pending=pending, command=command, dt=.02, horizon=.12, vmax=1.5)
    if name == "current_not_queue":
        q = rng.uniform(-.3, .3, 26).astype(F32)
        J = rng.normal(size=26)
        J = (J/np.linalg.norm(J)).astype(F32)[None]
        margin, gap = rng.uniform(.1, .2), rng.uniform(.15, .2)
        g = J.astype(float)[0]
        jitter = rng.uniform(-.001, .001, 6)
        noise = rng.uniform(-.0001, .0001, (6, 26))
        tangent = noise - (noise@g)[:, None]*g[None]/(g@g)
        pending = (q - (margin+gap+jitter)[:, None]*g[None] + tangent).astype(F32)
        return dict(J=J, G=-J, h=np.array([margin], dtype=F32), d=np.array([margin], dtype=F32),
                    dmin=np.zeros(1, dtype=F32), q=q, pending=pending, lo=vec(-.01), hi=vec(.01))
    if name == "benign_local_unknown_future":
        q = rng.uniform(-.3, .3, 26).astype(F32)
        J = rng.normal(size=(8, 26)).astype(F32)
        pending = (q + rng.uniform(-.01, .01, (6, 26))).astype(F32)
        d = (.02*np.abs(J.astype(float)).sum(1) + rng.uniform(.01, .03, 8)).astype(F32)
        return dict(q=q, J=J, pending=pending, d=d, dmin=np.zeros(8, dtype=F32))
    raise ValueError(name)


def evaluate(name, a):
    if name in ("dense_feasible", "collective_infeasible"):
        G, h, lo, hi = (a[k] for k in ("G", "h", "lo", "hi"))
        expected = FEASIBLE if name == "dense_feasible" else INFEASIBLE
        r = selected_set(G, h, lo, hi)
        certified(r, G, h, lo, hi, expected)
        require(r["all_rows_individually_feasible"], "construction must keep every row individually feasible")
        require(np.all(lo < hi) and np.all(np.linalg.norm(G, axis=1) > 0), "degenerate input")
        if name == "dense_feasible":
            require(verify_witness(G, h, lo, hi, a["planted"])["verified"], "analytic planted oracle failed")
        else:
            require(fraction(h[0])+fraction(h[1]) < 0 and np.array_equal(G[0], -G[1]),
                    "independent opposing-row proof failed")
        return {"current": brief(r), "mutant_detected": name == "collective_infeasible"}
    if name == "zero_hull":
        G, h, lo, hi = (a[k] for k in ("G", "h", "lo", "hi"))
        baseline = selected_set(G, h, lo, hi)
        hull_lo, hull_hi = np.minimum(a["ref_lo"], 0), np.maximum(a["ref_hi"], 0)
        bad = selected_set(G, h, hull_lo, hull_hi)
        certified(baseline, G, h, lo, hi, FEASIBLE)
        certified(bad, G, h, hull_lo, hull_hi, INFEASIBLE)
        require(np.all(hull_lo <= 0) and np.all(hull_hi >= 0), "hull must include zero")
        require(np.all(G > 0) and h[0] < 0, "independent positive-orthant contradiction failed")
        return {"original": brief(baseline), "hull": brief(bad), "mutant_detected": True}
    r = queue_local_risk(a["J"], a["d"], a["dmin"], a["q"], a["pending"])
    require(r["input_state"] == "VALID" and r["future_physical_safety"] == UNKNOWN, "scope escalation")
    require(np.all(a["q"] != 0) and np.all(np.ptp(a["pending"], axis=0) > 0), "random state degenerate")
    if name == "command_not_velocity":
        j, v = [fraction(x) for x in a["J"][0]], [fraction(x) for x in a["v"]]
        tau, vmax = fraction(a["horizon"]), fraction(a["vmax"])
        fake = fraction(a["d"][0]) - tau*vmax*sum(map(abs, j), Fraction())
        actual = fraction(a["d"][0]) + tau*sum((x*y for x, y in zip(j, v)), Fraction())
        require(fake > 0 and actual < 0, "velocity-box false-safe proof failed")
        require(np.all(np.abs(a["command"]) <= .03) and np.all(np.abs(a["v"]) > 1.5), "rate construction failed")
        require(r["status"] == "LOCAL_ENDPOINTS_NONNEGATIVE", "benign target endpoints required")
        return {"local_status": r["status"], "fake_margin_exact": str(fake), "actual_toy_margin_exact": str(actual),
                "max_qd": float(np.abs(a["v"]).max()), "future": UNKNOWN, "mutant_detected": True}
    if name == "current_not_queue":
        current = selected_set(a["G"], a["h"], a["lo"], a["hi"])
        certified(current, a["G"], a["h"], a["lo"], a["hi"], FEASIBLE)
        require(verify_witness(a["G"], a["h"], a["lo"], a["hi"], vec(0))["verified"], "zero must be feasible now")
        require(r["status"] == "LOCAL_LINEAR_UNSAFE" and r["first_negative_slot"] == 0, "immutable prefix risk missing")
        require(np.max(np.abs(np.diff(a["pending"].astype(float), axis=0))) < .01, "queued targets exceed rate box")
        next_h = np.array([r["endpoint_margins"][0][0]], dtype=F32)
        future = selected_set(a["G"], next_h, a["lo"], a["hi"])
        certified(future, a["G"], next_h, a["lo"], a["hi"], INFEASIBLE)
        return {"current": brief(current), "toy_next": brief(future), "local_status": r["status"],
                "first_negative_slot": r["first_negative_slot"], "future_physical_safety": UNKNOWN, "mutant_detected": True}
    require(r["status"] == "LOCAL_ENDPOINTS_NONNEGATIVE", "registered benign endpoint property failed")
    return {"local_status": r["status"], "minimum": r["min_pending_margin"], "future": UNKNOWN, "mutant_detected": True}


def test_collective_fixed():
    G = np.zeros((2, 26), dtype=F32)
    G[:, 0] = [1, -1]
    h = np.array([-.25, -.25], dtype=F32)
    lo, hi = vec(-1), vec(1)
    r = selected_set(G, h, lo, hi)
    certified(r, G, h, lo, hi, INFEASIBLE)
    require(r["all_rows_individually_feasible"], "row authority is insufficient for joint feasibility")
    return brief(r)


def test_empty_box():
    lo, hi = vec(-1), vec(1)
    lo[4] = 2
    r = selected_set(np.empty((0, 26), dtype=F32), np.empty(0, dtype=F32), lo, hi)
    require(r["status"] == INFEASIBLE and r["certificate"]["joints"] == [4], "empty box missed")
    return brief(r)


def test_no_rows_singleton():
    x = np.arange(26, dtype=F32)/32
    r = selected_set(np.empty((0, 26), dtype=F32), np.empty(0, dtype=F32), x, x)
    require(r["status"] == FEASIBLE and r["witness"] == x.tolist(), "singleton box failed")
    return brief(r)


def test_exact_tiny_negative():
    G = np.zeros((1, 26), dtype=F32)
    h = np.array([np.nextafter(F32(0), F32(-1))], dtype=F32)
    r = selected_set(G, h, vec(-1), vec(1))
    certified(r, G, h, vec(-1), vec(1), INFEASIBLE)
    return brief(r)


def test_geometry_endpoints():
    J, q, p = np.zeros((1, 26), dtype=F32), vec(0), np.zeros((6, 26), dtype=F32)
    dmin = np.array([.02], dtype=F32)
    equal = queue_local_risk(J, dmin.copy(), dmin, q, p)
    below = queue_local_risk(J, np.nextafter(dmin, F32(-np.inf)), dmin, q, p)
    above = queue_local_risk(J, np.nextafter(dmin, F32(np.inf)), dmin, q, p)
    require(equal["status"] == above["status"] == "LOCAL_ENDPOINTS_NONNEGATIVE", "equality not strict-negative")
    require(equal["min_pending_margin_exact"] == "0", "endpoint changed")
    require(below["status"] == "LOCAL_LINEAR_UNSAFE" and below["current_geometry_status"] == "OBSERVED_UNSAFE", "negative ULP hidden")
    return {"equal": equal["status"], "one_ulp_below": below["min_pending_margin_exact"], "future": UNKNOWN}


def test_invalid_inputs():
    G, h, lo, hi = np.ones((1, 26), dtype=F32), np.ones(1, dtype=F32), vec(-1), vec(1)
    bad = [(G.astype(float), h, lo, hi), (G, h.astype(float), lo, hi),
           (G, h, lo[:-1], hi), (G*np.nan, h, lo, hi), (G, h*np.inf, lo, hi)]
    for args in bad:
        r = selected_set(*args)
        require(r["status"] == UNKNOWN and r["input_state"] == "INVALID", "invalid current input accepted")
    for p in (np.zeros((5, 26), dtype=F32), np.full((6, 26), np.nan, dtype=F32)):
        require(queue_local_risk(G, h, h, vec(0), p)["status"] == UNKNOWN, "invalid queue accepted")
    require(queue_local_risk(np.empty((0, 26), dtype=F32), h[:0], h[:0], vec(0), np.zeros((6, 26), dtype=F32))["status"] == UNKNOWN,
            "empty monitored coverage must be unknown")
    return {"invalid_cases": 8, "status": "PASS"}


def test_solver_error_unknown():
    def broken(*args, **kwargs):
        raise RuntimeError("injected solver failure")
    r = selected_set(np.ones((1, 26), dtype=F32), np.ones(1, dtype=F32), vec(-1), vec(1), _solver=broken)
    require(r["status"] == UNKNOWN and "injected solver failure" in r["reason"], "solver error accepted")
    return brief(r)


def test_unverified_unknown():
    def liar(*args, **kwargs):
        return SimpleNamespace(status=0, message="injected false optimizer", success=True,
                               x=np.zeros(27), ineqlin=SimpleNamespace(marginals=np.zeros(2)))
    G = np.zeros((2, 26), dtype=F32)
    G[:, 0] = [1, -1]
    r = selected_set(G, np.array([-.25, -.25], dtype=F32), vec(-1), vec(1), _solver=liar)
    require(r["status"] == UNKNOWN, "solver success bit accepted as proof")
    return brief(r)


def test_corrupted_proofs():
    G, h = np.ones((1, 26), dtype=F32), np.array([-1], dtype=F32)
    require(not verify_witness(G, h, vec(0), vec(1), vec(0))["verified"], "bad witness accepted")
    require(not verify_separation(G, h, vec(0), vec(1), [-1])["verified"], "negative multiplier accepted")
    require(not verify_separation(G, h, vec(-1), vec(1), [1])["verified"], "invalid separating gap accepted")
    require(verify_separation(G, h, vec(0), vec(1), [1])["verified"], "valid separating gap rejected")
    return {"corruptions_rejected": 3, "valid_proof_accepted": True}


def test_F_U_alpha_product():
    F = dict(G=np.zeros((0, 14), dtype=F32), h=np.zeros(0, dtype=F32))
    U = dict(G=np.zeros((2, 12), dtype=F32), h=np.array([-.2, -.2], dtype=F32),
             row_labels=["safety", "alpha"])
    U["G"][:, 0] = [1, -1]
    r = diagnose_current(dict(F=F, U=U), vec(-1), vec(1))
    require(r["status"] == INFEASIBLE and r["robots"]["F"]["status"] == FEASIBLE, "product decomposition failed")
    require(r["robots"]["U"]["certificate"]["support_labels"] == ["safety", "alpha"], "alpha conflict omitted")
    return {"status": r["status"], "F": r["robots"]["F"]["status"], "U": r["robots"]["U"]["status"]}


def test_no_input_mutation():
    G, h, lo, hi = np.ones((1, 26), dtype=F32), np.ones(1, dtype=F32), vec(-1), vec(1)
    q, p = vec(.1), np.full((6, 26), .2, dtype=F32)
    arrays = dict(G=G, h=h, lo=lo, hi=hi, q=q, p=p)
    before = digest_inputs(arrays)
    selected_set(G, h, lo, hi)
    queue_local_risk(G, h, h, q, p)
    require(digest_inputs(arrays) == before, "oracle mutated inputs")
    return {"unchanged": True}


def test_three_fixed_counterexamples():
    # Exact scalar mathematics embedded in 26D.
    # 1) u=0 obeys .03 command bound but qdot=2 exceeds vmax=1.5.
    # 2) reference=[.25,.5], hull=[0,.5], x<=-.25: original [-1,1]
    #    feasible while the hull is not.
    # 3) q_next=pending[0]=-1, clearance=.5+q, u in [-.1,.1].
    #    Current -u<=.5 is feasible; next -u<=-.5 is infeasible.
    g = np.zeros((1, 26), dtype=F32)
    g[0, 0] = 1
    hull = selected_set(g, np.array([-.25], dtype=F32), vec(0), vec(.5))
    require(hull["status"] == INFEASIBLE, "hull counterexample failed")
    p = np.zeros((6, 26), dtype=F32)
    p[:, 0] = -1
    queue = queue_local_risk(g, np.array([.5], dtype=F32), np.array([0], dtype=F32), vec(0), p)
    current = selected_set(-g, np.array([.5], dtype=F32), vec(-.1), vec(.1))
    next_set = selected_set(-g, np.array([-.5], dtype=F32), vec(-.1), vec(.1))
    require(current["status"] == FEASIBLE and next_set["status"] == INFEASIBLE, "time-index confusion")
    require(queue["status"] == "LOCAL_LINEAR_UNSAFE", "queue counterexample failed")
    return {"vmax": 1.5, "command_u": 0, "allowed_toy_qdot": 2,
            "hull": hull["status"], "current": current["status"], "toy_next": next_set["status"],
            "future_physical_safety": UNKNOWN}


FIXED = [test_collective_fixed, test_empty_box, test_no_rows_singleton, test_exact_tiny_negative,
         test_geometry_endpoints, test_invalid_inputs, test_solver_error_unknown,
         test_unverified_unknown, test_corrupted_proofs, test_F_U_alpha_product,
         test_no_input_mutation, test_three_fixed_counterexamples]


def main():
    reg_path = ROOT/"ASTRA_PREREGISTRATION.json"
    registration = json.loads(reg_path.read_text())
    results = {"schema": "safeduo.astra.test_results.v1", "started_utc": datetime.now(timezone.utc).isoformat(),
               "python": sys.executable, "python_version": sys.version, "numpy": np.__version__,
               "scipy": scipy.__version__, "preregistration_sha256": sha256(reg_path),
               "source_sha256": {p.name: sha256(p) for p in (ROOT/"astra_oracle.py", ROOT/"astra_tests.py")},
               "deterministic": [], "random": [], "failures": []}
    require(scipy.__version__ == "1.15.3" and np.__version__ == "1.26.0", "registered dependency version mismatch")
    for fn in FIXED:
        record = {"name": fn.__name__}
        try:
            record.update(status="PASS", observed=fn())
        except Exception:
            record.update(status="FAIL", traceback=traceback.format_exc())
            results["failures"].append(record["name"])
        results["deterministic"].append(record)
    for family in registration["families"]:
        rng = np.random.Generator(np.random.PCG64(family["seed"]))
        for index in range(registration["count_per_family"]):
            record = {"family": family["name"], "seed": family["seed"], "index": index}
            data = None
            try:
                data = generate(family["name"], rng)
                record["input_sha256"] = digest_inputs(data)
                before = record["input_sha256"]
                observed = evaluate(family["name"], data)
                require(digest_inputs(data) == before, "random input mutated")
                record.update(status="PASS", observed=observed)
            except Exception:
                record.update(status="FAIL", traceback=traceback.format_exc())
                if data is not None:
                    record["failed_inputs"] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in data.items()}
                results["failures"].append(f"{family['name']}:{index}")
            results["random"].append(record)
    results["summary"] = {"deterministic_executed": len(results["deterministic"]),
                          "random_executed": len(results["random"]), "failure_count": len(results["failures"]),
                          "mutants_detected": sum(r.get("observed", {}).get("mutant_detected", False) for r in results["random"])}
    results["finished_utc"] = datetime.now(timezone.utc).isoformat()
    results["exit_code"] = int(bool(results["failures"]))
    output = ROOT/"astra_test_results.json"
    i = 1
    while output.exists():
        i += 1
        output = ROOT/f"astra_test_results_{i:02d}.json"
    with output.open("x") as f:
        json.dump(results, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"output": str(output), "summary": results["summary"], "exit_code": results["exit_code"],
                      "failures": results["failures"]}, indent=2))
    return results["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
