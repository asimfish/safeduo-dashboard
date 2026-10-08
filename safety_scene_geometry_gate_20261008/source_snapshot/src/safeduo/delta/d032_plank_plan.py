"""CPU-only D-032 four-arm plank qualification command candidate.

This module designs and audits one *scripted* trajectory on the actual-v6
FR3 + JAKA Zu7 + RH56F2 identity.  It is intentionally not a glove recording,
not a contact result, and not evidence that the dynamic plank moved.  The
candidate must still pass live Isaac/PhysX contact, constraint, safety, and
task predicates before it can be used as a qualification episode.

The arm path reuses the existing v5 position-IK design/composition pipeline.
Its low interaction lane keeps the F/U flange x gap at 0.40 m; four named
object grasp frames remain a runtime contact target rather than an offline IK
claim because the provider does not model articulated finger contact.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from safeduo.baselines.real_geometry import make_v5_provider
from safeduo.delta.skill_record import (
    CONTROL_DT,
    DELTA_HARD_CAP,
    Phase,
    SkillSpec,
    compose_skill,
    design_skill,
    validate_trajectory,
)
from safeduo.delta.skill_replay import SkillTrajectory
from safeduo.safety.types import ARM_KEYS

ENVIRONMENT_PROFILE = "duo_env_v6"
ASSET_PROFILE = "actual_asset_v6"
GEOMETRY_PROFILE = "geometry_v5"
ASSET_IDENTITIES = {"F": "fr3", "U": "jaka_zu7", "hand": "rh56f2"}
SKILL_NAME = "d032_scripted_plank_colift_v0"

ACTIVITY_NORM_MIN = 0.10
RETENTION_MIN = 0.25
MIN_ALL_FOUR_CONTIGUOUS_S = 1.0

PLANK_SIZE_M = (0.25, 0.90, 0.02)
TABLE_TOP_Z_M = 0.80
# The live v2 dynamic-object probe observed the plank settle at center z=0.81
# on the 0.80 m tabletop.  This is a center pose (not the USD child's local
# bottom/root transform), so center_z - half_height is exactly table_top.
PLANK_INITIAL_CENTER_POSE_WXYZ = (0.0, 0.0, 0.81, 1.0, 0.0, 0.0, 0.0)
PLANK_GRASP_FRAMES = {
    "F_L": (0.075, -0.38, 0.02),
    "F_R": (0.075, 0.38, 0.02),
    "U_L": (-0.075, 0.38, 0.02),
    "U_R": (-0.075, -0.38, 0.02),
}


def _hand_joint_names(side: str) -> tuple[str, ...]:
    return (
        f"{side}_thumb_1_joint",
        f"{side}_thumb_2_joint",
        f"{side}_thumb_3_joint",
        f"{side}_thumb_4_joint",
        f"{side}_index_1_joint",
        f"{side}_index_2_joint",
        f"{side}_middle_1_joint",
        f"{side}_middle_2_joint",
        f"{side}_ring_1_joint",
        f"{side}_ring_2_joint",
        f"{side}_pinky_1_joint",
        f"{side}_pinky_2_joint",
    )


HAND_JOINT_NAMES = {
    "left": _hand_joint_names("left"),
    "right": _hand_joint_names("right"),
}

# The dependent joints respect the RH56F2 URDF mimic ratios.  These are
# conservative command targets below every URDF upper limit, not evidence of
# force closure or object contact.
HAND_CLOSED = np.asarray(
    [
        1.0,
        0.55,
        0.55 * 0.7222,
        0.55 * 0.7222 * 0.69754,
        0.90,
        0.90 * 1.1545,
        0.90,
        0.90 * 1.1545,
        0.90,
        0.90 * 1.1545,
        0.90,
        0.90 * 1.1545,
    ],
    dtype=np.float32,
)


class CandidateValidationError(ValueError):
    """The scripted qualification candidate violates an offline hard gate."""


@dataclass
class PlankPlanCandidate:
    """Designed arm/hand commands plus complete CPU audit evidence."""

    trajectory: SkillTrajectory
    waypoints: dict[str, Any]
    validation: dict[str, Any]
    phase_boundaries: list[dict[str, Any]]
    margins: dict[str, np.ndarray]
    deltas: dict[str, np.ndarray]
    normalized_raw_command_norm: dict[str, np.ndarray]
    active: dict[str, np.ndarray]
    all_four_active: np.ndarray
    hands: dict[str, dict[str, Any]]
    activity_report: dict[str, Any]
    metadata: dict[str, Any]


def _all_modes(mode: str) -> dict[str, str]:
    return {arm: mode for arm in ARM_KEYS}


def _interaction_targets(
    y_shift_m: float,
    f_z_m: float,
    u_z_m: float | None = None,
) -> dict[str, tuple[float, float, float]]:
    """World flange targets surrounding the four named plank regions."""

    u_z_m = f_z_m if u_z_m is None else u_z_m
    return {
        "F_L": (0.21, -0.38 + y_shift_m, f_z_m),
        "F_R": (0.21, 0.38 + y_shift_m, f_z_m),
        "U_L": (-0.19, 0.38 + y_shift_m, u_z_m),
        "U_R": (-0.19, -0.38 + y_shift_m, u_z_m),
    }


def plank_skill_spec() -> SkillSpec:
    """Return the frozen v0 scripted choreography used by this candidate."""

    move = _all_modes("plan")
    hold = _all_modes("hold")
    return SkillSpec(
        SKILL_NAME,
        [
            Phase(
                "approach_high",
                3.0,
                targets=_interaction_targets(0.0, 1.16),
                mode=dict(move),
            ),
            Phase(
                "descend_to_grasp_regions",
                2.0,
                targets=_interaction_targets(0.0, 1.05, 1.06),
                mode=dict(move),
            ),
            Phase("close_hands_contact_candidate", 1.0, mode=dict(hold)),
            # A 60 mm flange lift (plus 50 mm y lead-in) precedes a 2.0 s
            # hold.  Dynamic-object lift remains a live PhysX predicate.
            Phase(
                "cooperative_lift",
                1.5,
                targets=_interaction_targets(0.05, 1.11, 1.12),
                mode=dict(move),
            ),
            Phase("lift_hold", 2.0, mode=dict(hold)),
            # Total y displacement from grasp is 280 mm.  The shorter,
            # synchronized phase provides a >1 s all-four-active interval.
            Phase(
                "cooperative_transport",
                1.8,
                targets=_interaction_targets(0.28, 1.11, 1.12),
                mode=dict(move),
            ),
            Phase("transport_hold", 1.0, mode=dict(hold)),
            Phase(
                "place",
                2.0,
                targets=_interaction_targets(0.28, 1.05, 1.06),
                mode=dict(move),
            ),
            Phase("open_hands_release_candidate", 1.0, mode=dict(hold)),
            Phase(
                "retreat_up",
                2.0,
                targets=_interaction_targets(0.28, 1.18),
                mode=dict(move),
            ),
            Phase(
                "retreat_out",
                2.5,
                targets={
                    "F_L": (0.22, -0.10, 1.28),
                    "F_R": (0.22, 0.66, 1.28),
                    "U_L": (-0.12, 0.66, 1.18),
                    "U_R": (-0.12, -0.10, 1.18),
                },
                mode=dict(move),
            ),
        ],
    )


def _validate_identity(
    environment_profile: str,
    geometry_profile: str,
    asset_identities: dict[str, str],
) -> None:
    if environment_profile != ENVIRONMENT_PROFILE:
        raise CandidateValidationError(
            "D-032 requires actual_asset_v6 through duo_env_v6; "
            f"got {environment_profile!r}"
        )
    if geometry_profile != GEOMETRY_PROFILE:
        raise CandidateValidationError(
            f"D-032 requires geometry_v5; got {geometry_profile!r}"
        )
    if asset_identities != ASSET_IDENTITIES:
        raise CandidateValidationError(
            "asset identities must be FR3 + JAKA Zu7 + RH56F2; "
            f"got {asset_identities!r}"
        )


def _phase_boundaries(spec: SkillSpec, dt: float) -> list[dict[str, Any]]:
    boundaries: list[dict[str, Any]] = []
    cursor = 0
    for phase in spec.phases:
        transitions = max(1, round(float(phase.dur) / dt))
        end = cursor + transitions
        boundaries.append(
            {
                "name": phase.name,
                "start_transition": cursor,
                "end_transition_exclusive": end,
                "start_state": cursor,
                "end_state": end,
                "transition_count": transitions,
                "duration_s": transitions * dt,
            }
        )
        cursor = end
    return boundaries


def _margin_audit(provider: Any, trajectory: SkillTrajectory) -> dict[str, np.ndarray]:
    """Compute all T+1 clearances, splitting the provider's self class by robot."""

    class_id = provider.pair_class.detach().cpu().numpy()
    arm_i = provider.pair_arm_i.detach().cpu().numpy()
    masks = {
        "cross": class_id == 0.0,
        "self_F": (class_id == 1.0) & (arm_i < 2),
        "self_U": (class_id == 1.0) & (arm_i >= 2),
        "table": class_id == 2.0,
    }
    if not all(mask.any() for mask in masks.values()):
        missing = [name for name, mask in masks.items() if not mask.any()]
        raise CandidateValidationError(
            f"provider has no pairs for margin classes {missing}"
        )

    chunks: dict[str, list[np.ndarray]] = {name: [] for name in masks}
    rows = trajectory.n_steps + 1
    with torch.no_grad():
        for start in range(0, rows, 128):
            stop = min(rows, start + 128)
            q = {
                arm: torch.as_tensor(
                    trajectory.q[arm][start:stop],
                    dtype=torch.float32,
                    device=provider.device,
                )
                for arm in ARM_KEYS
            }
            centers = provider.fk_all(q)["centers"]
            distances = provider._all_margins(centers)[0].detach().cpu().numpy()
            for name, mask in masks.items():
                chunks[name].append(distances[:, mask].min(axis=1))
    return {
        name: np.concatenate(parts).astype(np.float32, copy=False)
        for name, parts in chunks.items()
    }


def _longest_true_run(values: np.ndarray) -> int:
    padded = np.concatenate(
        [np.asarray([False]), values.astype(np.bool_, copy=False), np.asarray([False])]
    )
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return int((edges[1::2] - edges[::2]).max(initial=0))


def _activity_audit(
    trajectory: SkillTrajectory,
    boundaries: list[dict[str, Any]],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    dict[str, Any],
]:
    deltas = {
        arm: np.diff(trajectory.q[arm], axis=0).astype(np.float32, copy=False)
        for arm in ARM_KEYS
    }
    normalized = {
        arm: np.linalg.norm(delta / DELTA_HARD_CAP, axis=1).astype(np.float32)
        for arm, delta in deltas.items()
    }
    # This is an unfiltered command candidate, so executed==raw by
    # construction.  The value is not reused as a physical-execution claim.
    retention = 1.0
    active = {
        arm: (normalized[arm] >= ACTIVITY_NORM_MIN) & (retention >= RETENTION_MIN)
        for arm in ARM_KEYS
    }
    all_four = np.logical_and.reduce([active[arm] for arm in ARM_KEYS])
    f_active = active["F_L"] | active["F_R"]
    u_active = active["U_L"] | active["U_R"]

    by_name = {boundary["name"]: boundary for boundary in boundaries}
    start = int(by_name["cooperative_lift"]["start_transition"])
    stop = int(by_name["cooperative_transport"]["end_transition_exclusive"])
    interaction = slice(start, stop)
    count = stop - start
    if count <= 0:
        raise CandidateValidationError("interaction window is empty")
    interaction_all = all_four[interaction]
    report = {
        "normalization": "l2(delta_q / per_joint_hard_cap)",
        "per_joint_hard_cap_rad": DELTA_HARD_CAP,
        "normalized_raw_command_norm_min": ACTIVITY_NORM_MIN,
        "executed_to_raw_retention": retention,
        "retention_claim_boundary": "offline_unfiltered_candidate_only",
        "interaction_window": {
            "start_phase": "cooperative_lift",
            "end_phase": "cooperative_transport",
            "start_transition": start,
            "end_transition_exclusive": stop,
            "duration_s": count * trajectory.dt,
        },
        "per_arm_active_fraction": {
            arm: float(active[arm][interaction].mean()) for arm in ARM_KEYS
        },
        "two_robot_concurrent_fraction": float(
            (f_active[interaction] & u_active[interaction]).mean()
        ),
        "all_four_active_fraction": float(interaction_all.mean()),
        "longest_all_four_active_steps": _longest_true_run(interaction_all),
        "longest_all_four_active_seconds": (
            _longest_true_run(interaction_all) * trajectory.dt
        ),
    }
    return deltas, normalized, active, all_four, report


def _cosine_schedule(q0: np.ndarray, q1: np.ndarray, states: int) -> np.ndarray:
    phase = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, states, dtype=np.float64))
    return (q0[None] + phase[:, None] * (q1 - q0)[None]).astype(np.float32)


def _hand_schedules(
    trajectory: SkillTrajectory,
    boundaries: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_name = {boundary["name"]: boundary for boundary in boundaries}
    close = by_name["close_hands_contact_candidate"]
    release = by_name["open_hands_release_candidate"]
    close_start = int(close["start_state"])
    close_end = int(close["end_state"])
    release_start = int(release["start_state"])
    release_end = int(release["end_state"])
    open_q = np.zeros_like(HAND_CLOSED)
    result: dict[str, dict[str, Any]] = {}
    for arm in ARM_KEYS:
        q = np.zeros((trajectory.n_steps + 1, HAND_CLOSED.size), dtype=np.float32)
        q[close_start : close_end + 1] = _cosine_schedule(
            open_q, HAND_CLOSED, close_end - close_start + 1
        )
        q[close_end : release_start + 1] = HAND_CLOSED
        q[release_start : release_end + 1] = _cosine_schedule(
            HAND_CLOSED, open_q, release_end - release_start + 1
        )
        side = "left" if arm.endswith("_L") else "right"
        result[arm] = {
            "joint_names": HAND_JOINT_NAMES[side],
            "q": q,
            "valid": np.ones_like(q, dtype=np.bool_),
            "closed_state_start": close_end,
            "release_state_end": release_end,
            "claim": "command_schedule_only_contact_not_verified",
        }
    return result


def _enforce_offline_gates(candidate: PlankPlanCandidate) -> None:
    report = candidate.validation
    if not report.get("hard_gate_margin_gt0") or "FAIL" in report:
        raise CandidateValidationError(f"skill validation failed: {report}")
    if max(report["max_delta_per_arm"].values()) > DELTA_HARD_CAP:
        raise CandidateValidationError("trajectory delta exceeds the hard cap")
    bad_margin = {
        name: float(values.min())
        for name, values in candidate.margins.items()
        if float(values.min()) <= 0.0
    }
    if bad_margin:
        raise CandidateValidationError(f"margin <= 0 in candidate: {bad_margin}")

    activity = candidate.activity_report
    if min(activity["per_arm_active_fraction"].values()) < 0.10:
        raise CandidateValidationError("an interaction arm is active for less than 10%")
    if activity["two_robot_concurrent_fraction"] < 0.25:
        raise CandidateValidationError("two-robot concurrent activity is below 25%")
    if activity["all_four_active_fraction"] < 0.05:
        raise CandidateValidationError("all-four activity is below 5%")
    if (
        activity["longest_all_four_active_seconds"] + 1.0e-12
        < MIN_ALL_FOUR_CONTIGUOUS_S
    ):
        raise CandidateValidationError(
            "no contiguous one-second all-four-active interval"
        )


def build_candidate(
    *,
    environment_profile: str = ENVIRONMENT_PROFILE,
    geometry_profile: str = GEOMETRY_PROFILE,
    asset_identities: dict[str, str] | None = None,
) -> PlankPlanCandidate:
    """Design and fully audit the frozen command candidate on CPU."""

    identities = dict(
        ASSET_IDENTITIES if asset_identities is None else asset_identities
    )
    _validate_identity(environment_profile, geometry_profile, identities)
    provider = make_v5_provider(1, device="cpu")
    spec = plank_skill_spec()
    waypoints = design_skill(provider, spec, verbose=False)
    trajectory = compose_skill(waypoints, segments=None, dt=CONTROL_DT)
    validation = validate_trajectory(provider, trajectory)
    boundaries = _phase_boundaries(spec, trajectory.dt)
    if boundaries[-1]["end_transition_exclusive"] != trajectory.n_steps:
        raise CandidateValidationError("phase boundaries do not cover trajectory")
    margins = _margin_audit(provider, trajectory)
    deltas, normalized, active, all_four, activity = _activity_audit(
        trajectory, boundaries
    )
    hands = _hand_schedules(trajectory, boundaries)
    metadata: dict[str, Any] = {
        "skill": SKILL_NAME,
        "environment_profile": environment_profile,
        "asset_profile": ASSET_PROFILE,
        "geometry_profile": geometry_profile,
        "asset_identities": identities,
        "source_kind": "scripted_qualification_candidate",
        "formal_glove_recording": False,
        "physical_contact_verified": False,
        "task_success_verified": False,
        "planner": "skill_record.design_skill+compose_skill.cosine_ease_fallback",
        "offline_validator": "skill_record.validate_trajectory+four_class_margin_audit",
        "object": {
            "asset_id": "obj_plank",
            "dynamic_rigid_body_required": True,
            "gravity_enabled_required": True,
            "size_m": PLANK_SIZE_M,
            "initial_center_pose_world_wxyz": PLANK_INITIAL_CENTER_POSE_WXYZ,
            "initial_bottom_z_m": (
                PLANK_INITIAL_CENTER_POSE_WXYZ[2] - 0.5 * PLANK_SIZE_M[2]
            ),
            "table_top_z_m": TABLE_TOP_Z_M,
            "grasp_frames_object": PLANK_GRASP_FRAMES,
            "post_initialization_pose_write_allowed": False,
            "kinematic_follow_allowed": False,
        },
        "nominal_command_targets": {
            "plank_lift_m": 0.06,
            "lift_hold_s": 2.0,
            "plank_transport_m": 0.28,
            "object_motion_claim": "NOT_RUN_REQUIRES_LIVE_PHYSX",
        },
        "runtime_blockers": [
            "hand grasp-frame transforms and contact attribution require live PhysX",
            "four contact-gated constraints require live closed-loop stability evidence",
            "object lift, hold, translation, setdown, release, and gravity response are NOT_RUN",
            "this scripted trace does not replace a recorded two-operator glove episode",
        ],
    }
    candidate = PlankPlanCandidate(
        trajectory=trajectory,
        waypoints=waypoints,
        validation=validation,
        phase_boundaries=boundaries,
        margins=margins,
        deltas=deltas,
        normalized_raw_command_norm=normalized,
        active=active,
        all_four_active=all_four,
        hands=hands,
        activity_report=activity,
        metadata=metadata,
    )
    _enforce_offline_gates(candidate)
    return candidate


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _deterministic_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """Encode NPZ with sorted members and fixed ZIP metadata."""

    output = io.BytesIO()
    with zipfile.ZipFile(
        output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(arrays):
            npy = io.BytesIO()
            np.lib.format.write_array(
                npy, np.ascontiguousarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, npy.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
    return output.getvalue()


def _payload_entry(
    kind: str, payload: bytes, suffix: str
) -> tuple[dict[str, Any], bytes]:
    digest = hashlib.sha256(payload).hexdigest()
    return (
        {
            "file": f"{kind}_{digest}.{suffix}",
            "sha256": digest,
            "bytes": len(payload),
        },
        payload,
    )


def _exclusive_write(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)


def write_candidate_artifacts(
    candidate: PlankPlanCandidate,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Write a new content-hashed candidate bundle without overwriting files."""

    destination = Path(out_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"artifact destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    trajectory_arrays = {
        **{f"q_{arm}": candidate.trajectory.q[arm] for arm in ARM_KEYS},
        "dt": np.asarray(candidate.trajectory.dt, dtype=np.float64),
    }
    audit_arrays: dict[str, np.ndarray] = {
        **{f"margin_{name}": values for name, values in candidate.margins.items()},
        **{f"delta_{arm}": values for arm, values in candidate.deltas.items()},
        **{
            f"normalized_raw_command_norm_{arm}": values
            for arm, values in candidate.normalized_raw_command_norm.items()
        },
        **{
            f"active_{arm}": values.astype(np.bool_, copy=False)
            for arm, values in candidate.active.items()
        },
        "all_four_active": candidate.all_four_active.astype(np.bool_, copy=False),
    }
    phase_id = np.empty(candidate.trajectory.n_steps, dtype=np.int16)
    for index, boundary in enumerate(candidate.phase_boundaries):
        phase_id[
            boundary["start_transition"] : boundary["end_transition_exclusive"]
        ] = index
    audit_arrays["phase_id_transition"] = phase_id
    hand_arrays = {
        **{f"q_{arm}": candidate.hands[arm]["q"] for arm in ARM_KEYS},
        **{f"valid_{arm}": candidate.hands[arm]["valid"] for arm in ARM_KEYS},
    }

    payloads = {
        "trajectory": _payload_entry(
            "trajectory", _deterministic_npz_bytes(trajectory_arrays), "npz"
        ),
        "audit": _payload_entry("audit", _deterministic_npz_bytes(audit_arrays), "npz"),
        "hands": _payload_entry("hands", _deterministic_npz_bytes(hand_arrays), "npz"),
        "waypoints": _payload_entry(
            "waypoints", _canonical_json_bytes(candidate.waypoints), "json"
        ),
    }
    files = {name: entry for name, (entry, _) in payloads.items()}
    manifest = {
        "schema_version": "d032.scripted_plank_plan.v0",
        "claim_boundary": (
            "scripted_qualification_command_candidate_only_live_physx_not_run"
        ),
        "profile": ENVIRONMENT_PROFILE,
        "asset_profile": ASSET_PROFILE,
        "geometry": GEOMETRY_PROFILE,
        "asset_identities": ASSET_IDENTITIES,
        "metadata": candidate.metadata,
        "phase_boundaries": candidate.phase_boundaries,
        "validation": candidate.validation,
        "activity_report": candidate.activity_report,
        "hand_joint_names": {
            arm: list(candidate.hands[arm]["joint_names"]) for arm in ARM_KEYS
        },
        "files": files,
    }
    manifest_payload = _canonical_json_bytes(manifest)
    manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
    manifest_path = destination / f"manifest_{manifest_sha}.json"

    for _, (entry, payload) in payloads.items():
        _exclusive_write(destination / entry["file"], payload)
    _exclusive_write(manifest_path, manifest_payload)
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha,
        "files": files,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default="artifacts/d032/plank_plan_v0",
        help="new/empty output directory",
    )
    args = parser.parse_args(argv)
    candidate = build_candidate()
    result = write_candidate_artifacts(candidate, args.out)
    print(
        json.dumps(
            {
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "validation": candidate.validation,
                "activity_report": candidate.activity_report,
                "claim": "NOT_RUN_REQUIRES_LIVE_PHYSX",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTIVITY_NORM_MIN",
    "ARM_KEYS",
    "ASSET_IDENTITIES",
    "ASSET_PROFILE",
    "CandidateValidationError",
    "ENVIRONMENT_PROFILE",
    "GEOMETRY_PROFILE",
    "HAND_CLOSED",
    "HAND_JOINT_NAMES",
    "PLANK_INITIAL_CENTER_POSE_WXYZ",
    "PLANK_GRASP_FRAMES",
    "PLANK_SIZE_M",
    "PlankPlanCandidate",
    "TABLE_TOP_Z_M",
    "build_candidate",
    "plank_skill_spec",
    "write_candidate_artifacts",
]
