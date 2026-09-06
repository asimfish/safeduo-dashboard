#!/usr/bin/env python3
"""Merge plan.json (planned four-gate cells) with gates.json (finished cells) and runs.json
(in-progress gate dirs) into matrix.json.  Usage: build_matrix.py <data_dir>"""
import json
import os
import sys
import time

d = sys.argv[1]
plan = json.load(open(os.path.join(d, "plan.json"), encoding="utf-8"))
gates = json.load(open(os.path.join(d, "gates.json"), encoding="utf-8"))
try:
    runs = json.load(open(os.path.join(d, "runs.json"), encoding="utf-8"))
except Exception:
    runs = {}

done = {}
for r in gates.get("rows", []):
    if "error" in r:
        continue
    key = (r["policy"], r["env"], r["theta"])
    if key not in done or (r.get("date") or "") > (done[key].get("date") or ""):
        done[key] = r

active_dirs = {g["dir"] for g in runs.get("gates", []) if g.get("active") and g.get("n_cells", 0) == 0}
active_labels = [p["label"] for g in runs.get("gpus", []) for p in g.get("procs", []) if p.get("tag") == "safeduo"]

rows = []
for p in plan["rows"]:
    key = (p["policy"], p["env"], p["theta"])
    row = dict(p)
    if key in done:
        g = done[key]
        row.update({"status": "done", "date": g.get("date"), "cell": f'{g["grid"]}/{g["cell"]}',
                    "brake_rate": g.get("brake_rate"), "viol_free_cross": g.get("viol_free_cross"),
                    "min_cross_mm": g.get("min_cross_mm"), "no_deadlock_hys": g.get("no_deadlock_hys"),
                    "done_hys": g.get("done_hys"), "done_man": g.get("done_man"),
                    "false_brake_arm": g.get("false_brake_arm"), "fidelity": g.get("fidelity"), "hard_pass": g.get("hard_pass")})
    else:
        tag = f'{p["policy"]}_grid_{p["env"]}'.replace("r18_", "")
        running = any(tag in a or (p["policy"] in a and p["env"].split("_")[-1] in a) for a in active_dirs) or \
                  any(("gate " + p["policy"]) in a and p["env"].split("_")[-1] in a for a in active_labels)
        training = any(("train " + p["policy"] + "_") in a for a in active_labels)
        row["status"] = "run" if running else ("training" if training else ("planned" if "提案" in (p["policy"] + p["env"] + p.get("owner", "")) else "wait"))
    rows.append(row)

summary = {"total": len(rows), "done": sum(r["status"] == "done" for r in rows), "run": sum(r["status"] in ("run", "training") for r in rows),
           "planned": sum(r["status"] in ("planned", "wait") for r in rows)}
json.dump({"updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "protocol": plan.get("protocol", ""), "summary": summary, "rows": rows},
          open(os.path.join(d, "matrix.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("matrix.json:", summary)
