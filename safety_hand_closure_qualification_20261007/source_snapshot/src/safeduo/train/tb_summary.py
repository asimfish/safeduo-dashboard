"""tfevents summary for G0 evidence: iteration count, per-tag first/last/min/max, NaN scan.

Usage: python -m safeduo.train.tb_summary <run_dir> [--json out.json]
"""

from __future__ import annotations

import argparse
import json

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser = argparse.ArgumentParser()
parser.add_argument("run_dir")
parser.add_argument("--json", default="")
args = parser.parse_args()

ea = EventAccumulator(args.run_dir, size_guidance={"scalars": 0})
ea.Reload()
out = {}
for tag in sorted(ea.Tags()["scalars"]):
    vals = [e.value for e in ea.Scalars(tag)]
    out[tag] = {
        "n": len(vals),
        "first": round(vals[0], 6),
        "last": round(vals[-1], 6),
        "min": round(min(vals), 6),
        "max": round(max(vals), 6),
        "has_nan": any(v != v for v in vals),
    }
payload = json.dumps(out, indent=1)
if args.json:
    with open(args.json, "w") as f:
        f.write(payload)
print("TB_SUMMARY " + payload, flush=True)
