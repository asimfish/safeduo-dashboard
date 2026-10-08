"""Build spec-defined task families through the existing R34 machinery.

  python -m safeduo.skills.build --spec src/safeduo/skills/specs/mix_pick_center.yaml \
      --out artifacts/task_trajs_spec --hand-calib v7 --spheres r16 --struct-exempt --multiseed

Same knobs as `task_library_r35 build` (design IK projected onto the joint
limits by default, physical-grasp candidates, r16 finger shell).  Prints
SPEC_BUILD_OK / SPEC_BUILD_FAILED per family, writes the recording env yaml
and the joint-limit report; the recorder is then
  a22_record_s9task.py --task <family> --task_dir <out>/<family> --env_yaml duo_env_v7_spec_<family>.yaml ...
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SafeDuo spec-defined task families")
    ap.add_argument("--spec", nargs="+", required=True, help="TaskSpec yaml file(s)")
    ap.add_argument("--out", default="artifacts/task_trajs_spec")
    ap.add_argument("--hand-calib", default="v7")
    ap.add_argument("--spheres", default="r16", choices=["v7", "r16"])
    ap.add_argument("--struct-exempt", action="store_true")
    ap.add_argument("--multiseed", action="store_true")
    ap.add_argument("--no-dressing", action="store_true")
    ap.add_argument("--no-joint-limits", action="store_true")
    ap.add_argument("--describe-only", action="store_true", help="parse + describe the specs, no IK")
    args = ap.parse_args(argv)

    from safeduo.skills import compile as C
    from safeduo.skills import spec as S

    specs = [S.load(p) for p in args.spec]
    for sp in specs:
        C.register(sp)
        print(C.describe(sp))
    if args.describe_only:
        return 0

    from safeduo.delta import task_library_r26 as R26
    from safeduo.delta import task_library_r34 as R34
    from safeduo.delta import task_library_r35 as R35
    from safeduo.delta.task_record_s9 import make_s9_provider

    if args.struct_exempt:
        from safeduo.delta.skill_record import DESIGN_STRUCT_EXEMPT
        DESIGN_STRUCT_EXEMPT["on"] = True
    if args.multiseed:
        from safeduo.delta.skill_record_v7 import IK_MULTISEED
        IK_MULTISEED["on"] = True
        IK_MULTISEED["verbose"] = True
    if not args.no_joint_limits:
        from safeduo.delta.skill_record import IK_JOINT_LIMITS
        IK_JOINT_LIMITS["on"] = True
    R26.PHYS_GRASP["on"] = True
    R26.PHYS_GRASP["calib"] = args.hand_calib
    provider = make_s9_provider(1, device="cpu", spheres=args.spheres)
    rc = 0
    for sp in specs:
        t0 = time.time()
        try:
            rep = R34.build_task_r34(provider, sp.family, Path(args.out) / sp.family)
        except (RuntimeError, AssertionError, ValueError) as e:
            print(f"SPEC_BUILD_FAILED {sp.family}: {e}", flush=True)
            rc = 1
            continue
        print(f"SPEC_BUILD_OK {sp.family} in {time.time() - t0:.1f}s gate={rep.get('hard_gate_margin_gt0')}",
              flush=True)
        R35.joint_limit_report(Path(args.out) / sp.family / f"task_{sp.family}.npz")
        print(C.write_env_yaml(sp, dressing=not args.no_dressing), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
