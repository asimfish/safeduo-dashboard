"""Workspace-spanning L1 roaming: the T2-C spatial expansion of the random
intent stream (owner 2026-08-13 20:58: longer random episodes, wider range).

The stock l1 flow roams in JOINT space (JointSpaceMapper: waypoints are
q + U[-1.2, 1.2] offsets), which has no geometric awareness -- over minutes it
random-walks wherever the joint space takes it and visits the cross-machine
overlap band only incidentally. This module keeps the L1RandomDelta engine
(OU + segment machine, unchanged sample() contract) but samples waypoints in
EE space over the FULL shared workspace, so a 5-10 minute episode keeps
producing deliberate cross-workspace traffic.

Roam-box rationale (source of truth: duo_env_v5.yaml -> scene.layout_yaml =
assets_src/real/scene_layout_v5.yaml, loaded through RealScenePoses):

- workspace AABB x [-0.31, 0.31] / y [-0.85, 0.85] / z [0.80, 1.40] is B's
  reach-derived shared-workspace definition ("aabb ??(?? reach ??)") --
  the roam boxes take the FULL AABB, not the scenario half-boxes.
- z floor is lifted to table_top + 0.05 m (the workspace_spec convention):
  waypoints are flange-frame intent targets and a target inside the tabletop
  is not meaningful intent; the F2 hand still hangs ~0.19 m below the flange,
  so z 0.85 keeps plenty of genuine table pressure.
- per-arm x is clamped by full-extension reach (STATUS_B: FR3 855 mm from
  x=+-0.55 reaches x >= -0.305; JAKA Zu7 819 mm reaches x <= +0.269). The
  full-extension overlap band x in [-0.305, +0.269] (width 0.574 m, the
  cross-machine conflict main stage) lies inside BOTH per-arm boxes by
  construction, so every waypoint pair can produce overlap-band traffic.
- waypoints are additionally projected onto the arm's practical reach ball
  (F 0.75 / U 0.72 m, the singularity-avoiding numbers from STATUS_B):
  an unreachable waypoint would park the arm saturated at full stretch for
  the whole segment, wasting the segment's coverage.

Joint limits: the mapper emits deltas through the damped pseudo-inverse and
duo_env integrates targets under soft joint-limit clamps, so EE-space
waypoints never command past the limits; no per-joint box is needed here.
"""

from __future__ import annotations

import torch

from safeduo.delta._contract_stub import ARM_KEYS, SceneState
from safeduo.delta.l1_random import L1Params, L1RandomDelta
from safeduo.delta.l2_env_source import (
    CachedEEMapper,
    EnvEEBackend,
    RealScenePoses,
)

FULL_REACH = {"F": 0.855, "U": 0.819}       # data-sheet full extension (m)
PRACTICAL_REACH = {"F": 0.75, "U": 0.72}    # singularity-avoiding (STATUS_B)
Z_TABLE_MARGIN = 0.05                       # flange-target floor above table


def roam_boxes(poses: RealScenePoses) -> dict:
    """Per-arm waypoint AABB: full shared workspace, x clamped by reach.

    Returns arm -> (lo(3), hi(3)) float32 tensors. The cross-machine overlap
    band [-0.305, +0.269] is inside every box by construction (see module
    docstring for the derivation).
    """
    lo, hi = poses.ws_lo, poses.ws_hi
    z_lo = max(lo[2], poses.table_top_z + Z_TABLE_MARGIN)
    boxes = {}
    for arm in ARM_KEYS:
        bx = poses.base_pos[arm][0]
        if bx > 0:      # F side (base +x, reaches toward -x)
            x_lo, x_hi = max(lo[0], bx - FULL_REACH[arm[0]]), hi[0]
        else:           # U side
            x_lo, x_hi = lo[0], min(hi[0], bx + FULL_REACH[arm[0]])
        boxes[arm] = (
            torch.tensor([x_lo, lo[1], z_lo], dtype=torch.float32),
            torch.tensor([x_hi, hi[1], hi[2]], dtype=torch.float32),
        )
    return boxes


class WorkspaceRoamMapper(CachedEEMapper):
    """CachedEEMapper whose waypoints span the full shared workspace.

    sample_waypoint: uniform draw in the arm's roam box, then projected onto
    the practical-reach ball around the arm's base (radial clamp -- keeps the
    waypoint direction, pulls the distance inside reach).
    """

    def __init__(self, backend: EnvEEBackend, ee_speed: float = 0.25,
                 boxes: "dict | None" = None,
                 reach: "dict | None" = None):
        super().__init__(backend, ee_speed)
        self.boxes = boxes or roam_boxes(backend.poses)
        reach = reach or PRACTICAL_REACH
        self._r_max = {a: float(reach[a[0]]) for a in ARM_KEYS}
        self._base = {a: torch.tensor(backend.poses.base_pos[a],
                                      dtype=torch.float32)
                      for a in ARM_KEYS}

    def sample_waypoint(self, arm, state, generator):
        n = state.ee_pos[arm].shape[0]
        dev = state.ee_pos[arm].device
        lo, hi = (t.to(dev) for t in self.boxes[arm])
        u = torch.rand(n, 3, device=dev, generator=generator)
        wp = lo + u * (hi - lo)
        base = self._base[arm].to(dev)
        v = wp - base
        r = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        wp = base + v * torch.clamp(self._r_max[arm] / r, max=1.0)
        return wp


class WorkspaceRoamDelta(L1RandomDelta):
    """Standalone workspace-roaming L1 source (owns its FK backend).

    Same DeltaSource contract as L1RandomDelta -- reset(env_ids, generator) /
    sample(state) -> DeltaCmd, unbounded horizon (the segment machine
    resamples forever; nothing accumulates per step), so 5-10+ minute
    reset-free episodes need no special handling.

    Inside ConflictMixSource do NOT use this class: build a plain
    L1RandomDelta with a WorkspaceRoamMapper sharing the mix's backend (the
    mix refreshes the FK cache once per step already; a second owner would
    double the FK work).
    """

    def __init__(self, n_envs: int, params: "L1Params | None" = None,
                 device: "str | torch.device" = "cpu",
                 env_yaml: str = "duo_env_v5.yaml",
                 backend: "EnvEEBackend | None" = None,
                 mapper: "WorkspaceRoamMapper | None" = None,
                 arm_keys: tuple = ARM_KEYS,
                 dof_of: "dict | None" = None):
        if mapper is None:
            backend = backend or EnvEEBackend(RealScenePoses(env_yaml),
                                              device=device)
            mapper = WorkspaceRoamMapper(backend)
        self._backend = backend
        self._token = 0
        super().__init__(n_envs, params=params, mapper=mapper, device=device,
                         arm_keys=arm_keys, dof_of=dof_of)

    def sample(self, state: SceneState):
        self._token += 1
        if self._backend is not None:
            self._backend.refresh(state, self._token)
        return super().sample(state)
