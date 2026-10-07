"""Four-arm reachable-space and overlap audit.

This is a geometry audit, not a collision or task-success metric.  It samples
joint limits with the same FK chains used by the v7 coverage source and reports
per-arm reachable clouds, radial shells, and pairwise occupied-voxel overlap.
The result makes the distinction between a waypoint box and a physically
reachable operation region explicit.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import torch

from safeduo.baselines.real_geometry import ArmKinematics, FR3_FLANGE, FR3_JOINTS, FR3_LINKS
from safeduo.delta._contract_stub import ARM_KEYS
from safeduo.delta.l1_workspace_v7 import V7_FULL_REACH, full_boxes_v7
from safeduo.delta.l2_env_source import RealScenePoses
from safeduo.envs.birth_pose_search_v7 import FR3_LIMITS, UR_LIMITS
from safeduo.envs.v7_offline_geometry import ur5_v7_kin


def _sample(arm: str, n: int, seed: int, base_pos, base_yaw: float,
            device: str = "cpu") -> torch.Tensor:
    kin = ArmKinematics(FR3_JOINTS, FR3_LINKS, True, FR3_FLANGE, 0.0, device=device) \
        if arm[0] == "F" else ur5_v7_kin(device=device)
    limits = FR3_LIMITS if arm[0] == "F" else UR_LIMITS
    g = torch.Generator(device=device).manual_seed(int(seed))
    lo, hi = limits[:, 0].to(device), limits[:, 1].to(device)
    q = lo + torch.rand((n, limits.shape[0]), generator=g, device=device) * (hi - lo)
    return kin.fk(q, base_pos, base_yaw)["t_flange"]


def _voxel(points: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, bins: int) -> torch.Tensor:
    valid = ((points >= lo) & (points <= hi)).all(dim=1)
    p = points[valid]
    idx = ((p - lo) / (hi - lo) * bins).floor().long().clamp(0, bins - 1)
    out = torch.zeros((bins, bins, bins), dtype=torch.bool)
    if len(idx):
        out[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return out


def analyze(env_yaml: str = "duo_env_v7.yaml", n: int = 100_000,
            bins: int = 48, seed: int = 20260919) -> dict:
    poses = RealScenePoses(env_yaml)
    lo = torch.tensor(poses.ws_lo, dtype=torch.float32)
    hi = torch.tensor(poses.ws_hi, dtype=torch.float32)
    lo[2] = max(float(lo[2]), poses.table_top_z + 0.05)
    extent = hi - lo
    boxes = full_boxes_v7(poses)
    clouds = {a: _sample(a, n, seed + i * 7919, poses.base_pos[a], poses.base_yaw[a])
              for i, a in enumerate(ARM_KEYS)}
    vox = {a: _voxel(p, lo, hi, bins) for a, p in clouds.items()}
    out = {
        "schema": "safeduo.four_arm_workspace.v1",
        "env_yaml": env_yaml, "sample_count_per_arm": n, "voxel_bins": bins,
        "workspace_aabb": {"lo": lo.tolist(), "hi": hi.tolist(),
                            "volume_m3": float(extent.prod())},
        "arms": {}, "pairs": {},
    }
    for arm in ARM_KEYS:
        p = clouds[arm]
        r = (p - torch.tensor(poses.base_pos[arm])).norm(dim=1)
        q01, q99 = torch.quantile(p, 0.01, dim=0), torch.quantile(p, 0.99, dim=0)
        in_ws = ((p >= lo) & (p <= hi)).all(dim=1)
        b_lo, b_hi = boxes[arm]
        out["arms"][arm] = {
            "base_m": list(poses.base_pos[arm]),
            "full_reach_m": V7_FULL_REACH[arm[0]],
            "workspace_hit_rate": float(in_ws.float().mean()),
            "cloud_q01_m": q01.tolist(), "cloud_q99_m": q99.tolist(),
            "radius_q05_q50_q95_m": [float(torch.quantile(r, q)) for q in (.05, .5, .95)],
            "radius_le_080_rate": float((r <= .80).float().mean()),
            "full_box_volume_m3": float((b_hi - b_lo).prod()),
            "full_box_volume_over_workspace": float((b_hi - b_lo).prod() / extent.prod()),
            "occupied_voxels": int(vox[arm].sum()),
            "occupied_volume_m3": float(vox[arm].sum()) * float(extent.prod()) / bins**3,
            "occupied_fraction_of_workspace": float(vox[arm].sum() / bins**3),
        }
    for a, b in combinations(ARM_KEYS, 2):
        inter = vox[a] & vox[b]
        union = vox[a] | vox[b]
        key = f"{a}-{b}"
        out["pairs"][key] = {
            "intersection_voxels": int(inter.sum()),
            "intersection_volume_m3": float(inter.sum()) * float(extent.prod()) / bins**3,
            "intersection_over_union": float(inter.sum() / union.sum()) if union.any() else 0.0,
            "overlap_fraction_of_workspace": float(inter.sum() / bins**3),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-yaml", default="duo_env_v7.yaml")
    ap.add_argument("--samples", type=int, default=100_000)
    ap.add_argument("--bins", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    result = analyze(args.env_yaml, args.samples, args.bins, args.seed)
    data = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(data + "\n")
    print(data)


if __name__ == "__main__":
    main()
