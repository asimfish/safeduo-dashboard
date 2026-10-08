"""v7 wiring for the skill task-success judge (R14) -- NO judgment logic here.

skill_success.py stays untouched (redline); its SkillSuccessJudge already
accepts an injected FK and config path. This module only binds the v7 pieces:
V7SkillFK (UR5+DFX / FR3 flange FK on SceneLayoutV7, same caliber as the
skill_record_v7 hard-gate audit) + configs/skill_success_v7.yaml +
artifacts/skill_trajs_v7, and reuses the unmodified tier_table verifier.

    PYTHONPATH=src python -m safeduo.eval.skill_success_v7 \
        --skill-dir artifacts/skill_trajs_v7 --tiers 0 1 --seeds 25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safeduo.baselines.real_geometry_v7 import V7SkillFK
from safeduo.eval.skill_success import (
    REPO_ROOT,
    SkillSuccessJudge,
    render_tier_table,
    tier_table,
)

DEFAULT_CONFIG_V7 = (REPO_ROOT / "src" / "safeduo" / "configs"
                     / "skill_success_v7.yaml")
DEFAULT_SKILL_DIR_V7 = REPO_ROOT / "artifacts" / "skill_trajs_v7"


def v7_judge(skill_dir: "str | Path" = DEFAULT_SKILL_DIR_V7,
             config_path: "str | Path" = DEFAULT_CONFIG_V7,
             device: str = "cpu") -> SkillSuccessJudge:
    return SkillSuccessJudge(skill_dir=skill_dir, config_path=config_path,
                             fk=V7SkillFK(device), device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-dir", type=str,
                        default=str(DEFAULT_SKILL_DIR_V7))
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_V7))
    parser.add_argument("--tiers", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--amp-max", type=float, default=0.06)
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    judge = v7_judge(skill_dir=args.skill_dir, config_path=args.config)
    table = tier_table(judge, args.skill_dir, tiers=args.tiers,
                       n_seeds=args.seeds, seed=args.seed,
                       amp_max=args.amp_max, no_crop=args.no_crop)
    md = render_tier_table(table)
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"schema": "skill_task_success_tiers_v1", "layout": "v7",
             "seeds": args.seeds, "seed_base": args.seed,
             "amp_max": args.amp_max, "no_crop": args.no_crop,
             "table": table}, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
