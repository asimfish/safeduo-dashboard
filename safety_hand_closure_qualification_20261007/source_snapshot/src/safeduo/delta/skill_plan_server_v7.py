"""Server-side cuRobo segment planner for the v7 (R14) skill pipeline.

Thin wrapper over skill_plan_server: identical waypoint/segment protocol, but
the U arms use the UR5 config built by RobotBuilder from the owner-provided
UR5_only.urdf (arm-only, 6 dof; the RH56DFX hand is NOT in the planner model,
matching the F-side precedent where franka.yml carries no F2 hand -- the
authoritative hand-aware margins are re-audited offline by skill_record_v7
compose against the repaired real_v7 sphere pack).

  source ~/safeduo_setup/env.sh && source ~/venvs/curobo_t1/bin/activate
  CUDA_VISIBLE_DEVICES=0 timeout 540 python -m \
      safeduo.delta.skill_plan_server_v7 --wp-dir ~/safeduo/artifacts/skill_trajs_v7
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from safeduo.delta.skill_plan_server import build_planner, plan_edge

UR5_CFG = str(Path.home()
              / "safeduo/artifacts/skill_trajs_v7/curobo_cfgs/ur5_auto.yml")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wp-dir", required=True)
    ap.add_argument("--skill", default=None)
    args = ap.parse_args()
    wp_dir = Path(args.wp_dir).expanduser()

    t0 = time.time()
    planners = {"U": build_planner(UR5_CFG), "F": build_planner("franka.yml")}
    print(f"[{time.time()-t0:.1f}s] planners built (ur5 + franka)")

    docs = sorted(wp_dir.glob("waypoints_*.json"))
    if args.skill:
        docs = [d for d in docs if d.stem == f"waypoints_{args.skill}"]
    n_planned = 0
    for doc_path in docs:
        doc = json.loads(doc_path.read_text())
        skill = doc["skill"]
        wps = {a: np.asarray(v, dtype=np.float64)
               for a, v in doc["waypoints"].items()}
        out = {}
        for k, ph in enumerate(doc["phases"]):
            for arm, mode in ph["mode"].items():
                if mode != "plan":
                    continue
                q0, q1 = wps[arm][k], wps[arm][k + 1]
                if np.abs(q1 - q0).max() < 1e-9:
                    continue
                path = plan_edge(planners[arm[0]], q0, q1, arm_dof=len(q0))
                out[f"{arm}_seg{k}"] = path
                n_planned += 1
                print(f"  {skill} {ph['name']} {arm}: {path.shape[0]} pts "
                      f"[{time.time()-t0:.1f}s]")
        np.savez_compressed(wp_dir / f"segments_{skill}.npz", **out)
        print(f"wrote segments_{skill}.npz ({len(out)} segments)")
    print(f"[{time.time()-t0:.1f}s] done: {n_planned} plans, "
          f"mem {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
