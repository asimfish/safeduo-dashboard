"""Server-side cuRobo segment planner for the T1 skill pipeline (stage B).

Runs INSIDE the curobo overlay venv (~/venvs/curobo_t1, --system-site-packages
over conda safeduo) on bjxy_5090 -- NOT importable in the plain safeduo env.
GPU discipline: fragment-window job, single run well under 10 minutes.

  source ~/safeduo_setup/env.sh && source ~/venvs/curobo_t1/bin/activate
  CUDA_VISIBLE_DEVICES=<fragment card> timeout 540 python -m \
      safeduo.delta.skill_plan_server --wp-dir ~/safeduo/artifacts/skill_trajs

For every waypoints_<skill>.json and every phase edge whose mode is "plan",
plans a joint-space trajectory (plan_cspace: PRM graph seed + trajopt with
velocity/accel/jerk limits + self-collision cost) and stores the dense path
as segments_<skill>.npz key "<arm>_seg<k>". The composer (skill_record.py,
local CPU) retimes each segment onto the choreography phase duration, so the
planner's own timing is irrelevant -- only the geometric path matters.

Robot configs:
  U arms  jaka_zu7_auto.yml -- RobotBuilder output from the project URDF
          (assets_real jaka_zu7_clean.urdf), auto-fitted spheres + ignore
          matrix. Planner-side self-collision only; authoritative margins are
          re-audited offline against B's v5 spheres by the composer.
  F arms  curobo bundled franka.yml (FR3 kinematics == Panda is the standing
          project convention, see baselines/real_geometry.py FR3_JOINTS);
          fingers pinned at 0.04 and stripped from the output.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

JAKA_CFG = str(Path.home()
               / "safeduo/artifacts/skill_trajs/curobo_cfgs/jaka_zu7_auto.yml")
FRANKA_FINGER_Q = 0.04


def build_planner(robot_cfg: str):
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

    cfg = MotionPlannerCfg.create(robot=robot_cfg, collision_cache={"cuboid": 4})
    return MotionPlanner(cfg)


def plan_edge(mp, q0: np.ndarray, q1: np.ndarray, arm_dof: int) -> np.ndarray:
    from curobo.types import JointState

    names = mp.joint_names
    q0p, q1p = q0, q1
    if len(names) > len(q0):                      # e.g. franka arm + fingers
        pad = [FRANKA_FINGER_Q] * (len(names) - len(q0))
        q0p = np.concatenate([q0, pad])
        q1p = np.concatenate([q1, pad])
    js0 = JointState.from_position(
        torch.tensor(q0p, dtype=torch.float32, device="cuda").unsqueeze(0),
        joint_names=names)
    js1 = JointState.from_position(
        torch.tensor(q1p, dtype=torch.float32, device="cuda").unsqueeze(0),
        joint_names=names)
    res = mp.plan_cspace(js1, js0)
    if res is None or not bool(res.success.any()):
        raise RuntimeError("plan_cspace failed")
    pos = res.get_interpolated_plan().position   # (.., M, dof_out); dof_out may
    path = pos.reshape(-1, pos.shape[-1])        # include mimic/fixed joints
    path = path.detach().cpu().numpy().astype(np.float64)[:, :arm_dof]
    # exact endpoint pinning (planner is ~1e-7 off; composer asserts 1e-3)
    path[0], path[-1] = q0, q1
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wp-dir", required=True)
    ap.add_argument("--skill", default=None)
    args = ap.parse_args()
    wp_dir = Path(args.wp_dir).expanduser()

    t0 = time.time()
    planners = {"U": build_planner(JAKA_CFG), "F": build_planner("franka.yml")}
    print(f"[{time.time()-t0:.1f}s] planners built (jaka + franka)")

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
                    continue                      # degenerate plan edge = hold
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
