"""Compatibility shim over the canonical contract (safeduo/safety/types.py, owner A).

W1 history: this file carried a full stub while A's types.py did not exist yet.
W2 (2026-08-11 arbitration): types.py is published, so the stub definitions are
deleted; this module now only re-exports the canonical symbols so existing
C-side imports keep working, plus a few C-side helpers A does not define.

Contract notes consumed here (see types.py for authority):
- class_id is 3-class: CLASS_CROSS=0 / CLASS_SELF=1 / CLASS_TABLE=2.
  (The W1 4-class guess with separate self_F/self_U is dead; distinguish the
  robot of a self pair via pair_id -> SphereDistanceModule.pair_table.)
- observation is the ACTIVE PAIR SET: SceneState.active_pairs (N, M, 4) with
  [dist, closing_vel, class_id, pair_id], margin-ascending, zero-filled, plus
  active_mask (N, M); closing_vel is POSITIVE when approaching. The
  conflict-tube test is margin < d_soft AND closing_vel > 0 on a cross row.
"""

from __future__ import annotations

from safeduo.safety.types import (  # noqa: F401
    ARM_DOF,
    ARM_KEYS,
    ARMS_OF_ROBOT,
    CLASS_CROSS,
    CLASS_SELF,
    CLASS_TABLE,
    DOF_OF,
    MAX_ACTIVE_PAIRS,
    PAIR_FEATURE_DIM,
    ROBOT_OF,
    TOTAL_DOF,
    DeltaCmd,
    DeltaSource,
    SceneState,
    backstop_project,
    pair_obs_features,
    split_stacked,
    zeros_delta,
)


def using_stub() -> bool:
    """Canonical types.py is now a hard dependency; the W1 stub is gone."""
    return False
