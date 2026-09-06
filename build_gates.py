#!/usr/bin/env python3
"""Build gates.json (four-gate evaluation matrix) from clutch_eval cell JSON files.

Usage: build_gates.py <cells_root> <out_json>
<cells_root>/<grid_dir>/cell_*.json  (mirrored from bjxy_5090:~/safeduo/artifacts/clutch/)

Columns follow MASTER_REPORT sec.9 R28-R33: brake / viol-free(cross,any) / min cross margin /
no-deadlock(hys) / done(hys,man) / false-brake(arm,any) / fidelity / cmd-stop p50,p95.
"""
import glob
import json
import os
import sys
import time


def g(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def env_tags(env_yaml, cell):
    """Human-readable execution-stack tags derived from the env yaml name + cell options."""
    name = (env_yaml or "").replace("duo_env_v7_", "").replace(".yaml", "")
    tags = []
    if "r16a" in name:
        tags.append("r16 壳+活腕")
    elif "r16" in name:
        tags.append("r16 壳")
    if "fix2" in name:
        tags.append("双臂重力补偿")
    elif "fix" in name:
        tags.append("活腕(armature)+F 重力补偿")
    if "stack" in name:
        tags.append("R29+R30b 栈")
    if "_bl" in name:
        tags.append("backlog-aware")
    if "_la06" in name:
        tags.append("lookahead 0.06 s")
    elif "_la" in name:
        tags.append("lookahead 0.15 s")
    if cell.get("bypass_mm"):
        tags.append(f"bypass {cell['bypass_mm']} mm")
    return name, tags


def row_from_cell(grid_dir, path):
    with open(path, encoding="utf-8") as fh:
        c = json.load(fh)
    ckpt = c.get("ckpt") or ""
    policy = os.path.basename(os.path.dirname(ckpt)) if ckpt else "?"
    short = policy.split("_")[0] if policy != "?" else "?"
    env_name, tags = env_tags(c.get("env_yaml"), c)
    br = g(c, "brake", "aggregate", default={}) or {}
    hys = g(c, "recovery_hysteresis", "aggregate", default={}) or {}
    man = g(c, "recovery_manual", "aggregate", default={}) or {}
    fid = c.get("fidelity") or {}
    fb = c.get("false_brake") or {}
    mm = br.get("min_cross_margin")
    fid_bitwise = fid.get("fidelity_bitwise")
    if fid_bitwise is None and isinstance(fid.get("fidelity_at_tol"), dict):
        fid_bitwise = fid["fidelity_at_tol"].get("1e-06")
    return {
        "grid": grid_dir,
        "cell": os.path.basename(path)[5:-5],
        "policy": short,
        "policy_run": policy,
        "env": env_name,
        "env_yaml": c.get("env_yaml"),
        "tags": tags,
        "theta": f"{c.get('theta_hi')}:{c.get('theta_lo')}",
        "clutch": bool(c.get("clutch", True)),
        "date": c.get("date"),
        "num_envs": c.get("num_envs"),
        "duration_s": c.get("duration_s"),
        "brake_rate": br.get("brake_rate"),
        "viol_free_cross": br.get("violation_free_rate"),
        "viol_free_any": br.get("violation_free_any_class_rate"),
        "min_cross_mm": None if mm is None else round(1000.0 * mm, 1),
        "cmd_stop_p50": g(br, "cmd_stop_delay_steps", "p50"),
        "cmd_stop_p95": g(br, "cmd_stop_delay_steps", "p95"),
        "lock_delay_p50": g(br, "lock_delay_steps", "p50"),
        "no_deadlock_hys": hys.get("no_deadlock_rate"),
        "done_hys": hys.get("done_rate"),
        "done_man": man.get("done_rate"),
        "froze_hys": hys.get("froze_frac"),
        "froze_man": man.get("froze_frac"),
        "false_brake_arm": fb.get("false_brake_arm_rate"),
        "false_brake_any": fb.get("false_brake_any_rate"),
        "false_brake_per_arm": fb.get("false_brake_per_arm"),
        "fidelity": fid_bitwise,
        "mean_atten": fid.get("mean_attenuation_safe"),
        "safe_step_frac": fid.get("safe_step_frac"),
    }


def hard_gates_pass(r):
    """Paper hard gates (R28): brake 1.0, viol-free(cross) 1.0, no-deadlock(hys) 1.0."""
    vals = [r.get("brake_rate"), r.get("viol_free_cross"), r.get("no_deadlock_hys")]
    if any(v is None for v in vals):
        return None
    return all(v >= 0.999 for v in vals)


def main():
    root, out = sys.argv[1], sys.argv[2]
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(os.path.join(d, "cell_*.json"))):
            try:
                r = row_from_cell(os.path.basename(d), p)
                r["hard_pass"] = hard_gates_pass(r)
                rows.append(r)
            except Exception as e:  # keep the matrix alive even if one cell is malformed
                rows.append({"grid": os.path.basename(d), "cell": os.path.basename(p), "error": str(e)})
    rows.sort(key=lambda r: (r.get("date") or ""), reverse=True)
    json.dump({"updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "n": len(rows), "rows": rows},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"gates.json: {len(rows)} cells")


if __name__ == "__main__":
    main()
