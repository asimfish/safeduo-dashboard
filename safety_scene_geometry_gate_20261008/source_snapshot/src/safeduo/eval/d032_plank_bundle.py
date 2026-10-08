"""Strict conversion of the frozen D-032 plank plan to a nominal v6.2 bundle.

The source artifact is a scripted, offline qualification candidate.  This
module verifies every content-addressed byte before converting it; it does not
upgrade the candidate into contact, lift, or task-success evidence.  The
resulting bundle contains a truly zero perturbation channel under the explicit
``d032.actual_asset_v6.2`` nominal contract.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from safeduo.eval.trajectory_task_bundle import (
    ARM_DIMS,
    ASSET_IDENTITIES,
    GEOMETRY,
    NOMINAL_SCHEMA_VERSION,
    PROFILE,
    load_episode_bundle,
    validate_exogenous_input,
    write_episode_bundle,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CANDIDATE_DIR = REPO_ROOT / "artifacts" / "d032" / "plank_plan_v0"
ENV_CONFIG_PATH = REPO_ROOT / "src" / "safeduo" / "configs" / "duo_env_v6.yaml"
PLAN_SOURCE_PATH = REPO_ROOT / "src" / "safeduo" / "delta" / "d032_plank_plan.py"
CORE_BUNDLE_SOURCE_PATH = (
    REPO_ROOT / "src" / "safeduo" / "eval" / "trajectory_task_bundle.py"
)
REPLAY_SOURCE_PATH = (
    REPO_ROOT / "src" / "safeduo" / "delta" / "trajectory_task_replay.py"
)
PLANK_ASSET_PATH = REPO_ROOT / "assets_src" / "tasks" / "usd" / "obj_plank.usda"

EXPECTED_CANDIDATE_MANIFEST_SHA256 = (
    "b5df6ffc15fd3961cb2f10bcecfca870049eda95f06085d8f762cc73fcac54fc"
)
EXPECTED_PLAN_SOURCE_SHA256 = (
    "b8672b6c4d127d5a71d533066a3fa3b77bc50ba72021e1c0d254b33e3ba4ec39"
)
EXPECTED_CANDIDATE_SCHEMA = "d032.scripted_plank_plan.v0"
EXPECTED_CANDIDATE_FILES = {"trajectory", "audit", "hands", "waypoints"}
EXPECTED_OBJECT_PATH = "/World/envs/env_0/Obj_plank"
PLANK_CENTER_POSE_WXYZ = (0.0, 0.0, 0.81, 1.0, 0.0, 0.0, 0.0)
PLANK_SIZE_M = (0.25, 0.90, 0.02)
TABLE_TOP_Z_M = 0.80
DELTA_HARD_CAP_RAD = 0.03

_ASSET_COMPONENTS = {
    "geometry_manifest": "assets_src/real/v5_bundle_manifest.yaml",
    "plank_usda": "assets_src/tasks/usd/obj_plank.usda",
    "spheres_fr3": "assets_src/spheres/fr3.yaml",
    "spheres_jaka_v5": "assets_src/real/jaka_zu7_v5.yaml",
    "spheres_hand_left_v5": "assets_src/real/inspire_rh56f2_left_v5.yaml",
    "spheres_hand_right_v5": "assets_src/real/inspire_rh56f2_right_v5.yaml",
}
_EXPECTED_ARM_USD = {
    "F_L": "assets_real/usd/combined/fr3_f2_left_v5.usd",
    "F_R": "assets_real/usd/combined/fr3_f2_right_v5.usd",
    "U_L": "assets_real/usd/combined/jaka_f2_left_v5.usd",
    "U_R": "assets_real/usd/combined/jaka_f2_right_v5.usd",
}


class PlankBundleError(ValueError):
    """Candidate provenance or nominal bundle conversion failed closed."""


@dataclass(frozen=True)
class PlankPlanArtifact:
    manifest: dict[str, Any]
    manifest_sha256: str
    file_sha256: dict[str, str]
    q_ref: dict[str, np.ndarray]
    raw_delta: dict[str, np.ndarray]
    hand_q: dict[str, np.ndarray]
    hand_valid: dict[str, np.ndarray]
    hand_joint_names: dict[str, tuple[str, ...]]
    margins: dict[str, np.ndarray]
    steps: int
    dt: float


@dataclass(frozen=True)
class NominalPlankBundle:
    candidate: PlankPlanArtifact
    exogenous: dict[str, Any]
    manifest: dict[str, Any]
    provenance: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    def jsonable(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, dict):
            return {str(key): jsonable(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [jsonable(child) for child in item]
        return item

    return (
        json.dumps(
            jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _aggregate_digest(domain: str, records: dict[str, dict[str, Any]]) -> str:
    payload = {
        "domain": domain,
        "records": {
            name: {
                "path": record["path"],
                "bytes": record["bytes"],
                "sha256": record["sha256"],
            }
            for name, record in sorted(records.items())
        },
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _relative_path(path: Path) -> str:
    resolved = Path(path).resolve(strict=True)
    try:
        return resolved.relative_to(REPO_ROOT.resolve(strict=True)).as_posix()
    except ValueError as exc:
        raise PlankBundleError(
            f"provenance path escapes repository: {resolved}"
        ) from exc


def _file_record(path: Path) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise PlankBundleError(
            f"provenance file is missing, non-regular, or symlinked: {candidate}"
        )
    return {
        "path": _relative_path(candidate),
        "bytes": candidate.stat().st_size,
        "sha256": sha256_file(candidate),
    }


def _load_npz(payload: bytes, label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError, EOFError) as exc:
        raise PlankBundleError(f"candidate {label} NPZ cannot be read: {exc}") from exc
    for name, array in arrays.items():
        if array.dtype.hasobject:
            raise PlankBundleError(f"candidate {label}.{name} uses object dtype")
        if array.dtype.kind in "fci" and not np.isfinite(array).all():
            raise PlankBundleError(f"candidate {label}.{name} is non-finite")
    return arrays


def _longest_true_run(values: np.ndarray) -> int:
    padded = np.concatenate(
        (np.asarray([False]), np.asarray(values, dtype=np.bool_), np.asarray([False]))
    )
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return int((edges[1::2] - edges[::2]).max(initial=0))


def _load_candidate_directory(
    root: str | Path,
    *,
    required_root: str | Path,
) -> PlankPlanArtifact:
    """Load one exact candidate directory; exposed privately for tamper tests."""

    raw_root = Path(root).expanduser()
    if raw_root.is_symlink():
        raise PlankBundleError("candidate root must not be a symlink")
    try:
        resolved = raw_root.resolve(strict=True)
        required = Path(required_root).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise PlankBundleError(f"candidate root is missing: {raw_root}") from exc
    if not resolved.is_dir() or resolved != required:
        raise PlankBundleError(
            f"candidate root differs from required root: {resolved} != {required}"
        )
    entries = list(resolved.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise PlankBundleError("candidate files must be regular non-symlink files")

    manifest_name = f"manifest_{EXPECTED_CANDIDATE_MANIFEST_SHA256}.json"
    manifest_path = resolved / manifest_name
    if not manifest_path.is_file():
        raise PlankBundleError(
            f"required candidate manifest is missing: {manifest_name}"
        )
    manifest_payload = manifest_path.read_bytes()
    if (
        hashlib.sha256(manifest_payload).hexdigest()
        != EXPECTED_CANDIDATE_MANIFEST_SHA256
    ):
        raise PlankBundleError("candidate manifest sha256 differs from frozen identity")
    try:
        manifest = json.loads(manifest_payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PlankBundleError(f"candidate manifest cannot be decoded: {exc}") from exc
    if not isinstance(manifest, dict):
        raise PlankBundleError("candidate manifest must be a mapping")
    expected_manifest_keys = {
        "schema_version",
        "claim_boundary",
        "profile",
        "asset_profile",
        "geometry",
        "asset_identities",
        "metadata",
        "phase_boundaries",
        "validation",
        "activity_report",
        "hand_joint_names",
        "files",
    }
    if set(manifest) != expected_manifest_keys:
        raise PlankBundleError("candidate manifest keys differ from frozen schema")
    if (
        manifest["schema_version"] != EXPECTED_CANDIDATE_SCHEMA
        or manifest["profile"] != "duo_env_v6"
        or manifest["asset_profile"] != PROFILE
        or manifest["geometry"] != "geometry_v5"
        or manifest["asset_identities"] != ASSET_IDENTITIES
    ):
        raise PlankBundleError("candidate identity is not actual-v6 geometry-v5")
    if manifest["claim_boundary"] != (
        "scripted_qualification_command_candidate_only_live_physx_not_run"
    ):
        raise PlankBundleError("candidate claim boundary was widened")

    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != EXPECTED_CANDIDATE_FILES:
        raise PlankBundleError("candidate content file set differs from frozen schema")
    suffix_of = {
        "trajectory": "npz",
        "audit": "npz",
        "hands": "npz",
        "waypoints": "json",
    }
    payloads: dict[str, bytes] = {}
    file_sha256: dict[str, str] = {}
    expected_names = {manifest_name}
    for kind in sorted(EXPECTED_CANDIDATE_FILES):
        entry = files[kind]
        if not isinstance(entry, dict) or set(entry) != {"file", "sha256", "bytes"}:
            raise PlankBundleError(f"candidate files.{kind} entry is malformed")
        digest = entry["sha256"]
        name = entry["file"]
        expected_name = f"{kind}_{digest}.{suffix_of[kind]}"
        if not isinstance(digest, str) or len(digest) != 64 or name != expected_name:
            raise PlankBundleError(
                f"candidate files.{kind} filename/hash path confusion"
            )
        if Path(name).name != name:
            raise PlankBundleError(f"candidate files.{kind} must be a basename")
        path = resolved / name
        if path.is_symlink() or not path.is_file() or path.parent != resolved:
            raise PlankBundleError(f"candidate files.{kind} escapes its root")
        payload = path.read_bytes()
        if len(payload) != entry["bytes"]:
            raise PlankBundleError(f"candidate files.{kind} bytes mismatch")
        if hashlib.sha256(payload).hexdigest() != digest:
            raise PlankBundleError(f"candidate files.{kind} sha256 mismatch")
        expected_names.add(name)
        payloads[kind] = payload
        file_sha256[kind] = digest
    actual_names = {entry.name for entry in entries}
    if actual_names != expected_names:
        raise PlankBundleError(
            f"candidate files differ from manifest: {sorted(actual_names)}"
        )

    trajectory = _load_npz(payloads["trajectory"], "trajectory")
    audit = _load_npz(payloads["audit"], "audit")
    hands = _load_npz(payloads["hands"], "hands")
    try:
        waypoints = json.loads(payloads["waypoints"])
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PlankBundleError(f"candidate waypoints cannot be decoded: {exc}") from exc
    if (
        not isinstance(waypoints, dict)
        or set(waypoints) != {"dt", "meta", "phases", "skill", "waypoints"}
        or not isinstance(waypoints["waypoints"], dict)
        or set(waypoints["waypoints"]) != set(ARM_DIMS)
    ):
        raise PlankBundleError("candidate waypoints do not cover exactly four arms")

    trajectory_keys = {"dt", *(f"q_{arm}" for arm in ARM_DIMS)}
    if set(trajectory) != trajectory_keys:
        raise PlankBundleError("candidate trajectory keys differ from schema")
    dt_array = trajectory["dt"]
    if dt_array.shape != (1,):
        raise PlankBundleError("candidate dt must contain exactly one value")
    dt = float(dt_array[0])
    if not math.isclose(dt, 1.0 / 60.0, rel_tol=0.0, abs_tol=0.0):
        raise PlankBundleError("candidate dt is not exactly 1/60 second")
    steps = int(trajectory["q_F_L"].shape[0] - 1)
    if steps != 1188:
        raise PlankBundleError(f"candidate transition count differs: {steps}")
    q_ref: dict[str, np.ndarray] = {}
    raw_delta: dict[str, np.ndarray] = {}
    for arm, dim in ARM_DIMS.items():
        q = trajectory[f"q_{arm}"]
        delta = audit.get(f"delta_{arm}")
        if q.shape != (steps + 1, dim) or delta is None or delta.shape != (steps, dim):
            raise PlankBundleError(f"candidate arm shape differs for {arm}")
        if not np.array_equal(delta, np.diff(q, axis=0)):
            raise PlankBundleError(
                f"candidate raw delta does not equal q diff for {arm}"
            )
        if float(np.abs(delta).max(initial=0.0)) > DELTA_HARD_CAP_RAD:
            raise PlankBundleError(f"candidate delta exceeds hard cap for {arm}")
        q_ref[arm] = q.copy()
        raw_delta[arm] = delta.copy()

    margin_names = ("cross", "self_F", "self_U", "table")
    margins: dict[str, np.ndarray] = {}
    for name in margin_names:
        values = audit.get(f"margin_{name}")
        if values is None or values.shape != (steps + 1,) or float(values.min()) <= 0.0:
            raise PlankBundleError(
                f"candidate margin is non-positive or malformed: {name}"
            )
        margins[name] = values.copy()
    all_four = audit.get("all_four_active")
    if (
        all_four is None
        or all_four.shape != (steps,)
        or _longest_true_run(all_four) < 60
    ):
        raise PlankBundleError("candidate lacks a one-second all-four-active interval")

    joint_names_raw = manifest["hand_joint_names"]
    if not isinstance(joint_names_raw, dict) or set(joint_names_raw) != set(ARM_DIMS):
        raise PlankBundleError("candidate hand names do not cover exactly four arms")
    hand_q: dict[str, np.ndarray] = {}
    hand_valid: dict[str, np.ndarray] = {}
    hand_joint_names: dict[str, tuple[str, ...]] = {}
    for arm in ARM_DIMS:
        q = hands.get(f"q_{arm}")
        valid = hands.get(f"valid_{arm}")
        names = tuple(joint_names_raw[arm])
        side = "left_" if arm.endswith("_L") else "right_"
        if (
            q is None
            or valid is None
            or q.shape != (steps + 1, 12)
            or valid.shape != q.shape
            or valid.dtype.kind != "b"
            or len(names) != 12
            or len(set(names)) != 12
            or any(not str(name).startswith(side) for name in names)
        ):
            raise PlankBundleError(f"candidate RH56F2 schedule is malformed for {arm}")
        hand_q[arm] = q.copy()
        hand_valid[arm] = valid.copy()
        hand_joint_names[arm] = names

    metadata = manifest["metadata"]
    if not isinstance(metadata, dict):
        raise PlankBundleError("candidate metadata must be a mapping")
    obj = metadata.get("object")
    if not isinstance(obj, dict):
        raise PlankBundleError("candidate object metadata is missing")
    pose = tuple(
        float(value) for value in obj.get("initial_center_pose_world_wxyz", ())
    )
    size = tuple(float(value) for value in obj.get("size_m", ()))
    if pose != PLANK_CENTER_POSE_WXYZ or size != PLANK_SIZE_M:
        raise PlankBundleError(
            "candidate plank pose/size differs from frozen actual-v6 target"
        )
    bottom = pose[2] - 0.5 * size[2]
    if bottom < TABLE_TOP_Z_M or not math.isclose(bottom, TABLE_TOP_Z_M, abs_tol=1e-12):
        raise PlankBundleError(
            "candidate plank bottom penetrates or floats above the table"
        )
    if (
        metadata.get("physical_contact_verified") is not False
        or metadata.get("task_success_verified") is not False
        or metadata.get("formal_glove_recording") is not False
    ):
        raise PlankBundleError(
            "candidate improperly claims glove/contact/task evidence"
        )
    return PlankPlanArtifact(
        manifest=manifest,
        manifest_sha256=EXPECTED_CANDIDATE_MANIFEST_SHA256,
        file_sha256=file_sha256,
        q_ref=q_ref,
        raw_delta=raw_delta,
        hand_q=hand_q,
        hand_valid=hand_valid,
        hand_joint_names=hand_joint_names,
        margins=margins,
        steps=steps,
        dt=dt,
    )


def load_plank_plan_candidate() -> PlankPlanArtifact:
    """Load only the canonical content-addressed plank candidate."""

    return _load_candidate_directory(
        DEFAULT_CANDIDATE_DIR,
        required_root=DEFAULT_CANDIDATE_DIR,
    )


def _provenance(candidate: PlankPlanArtifact) -> dict[str, Any]:
    canonical_plan_source = (
        REPO_ROOT / "src" / "safeduo" / "delta" / "d032_plank_plan.py"
    )
    if PLAN_SOURCE_PATH.resolve(strict=True) != canonical_plan_source.resolve(
        strict=True
    ):
        raise PlankBundleError("plank plan source path differs from canonical source")
    plan_source = _file_record(PLAN_SOURCE_PATH)
    if plan_source["sha256"] != EXPECTED_PLAN_SOURCE_SHA256:
        raise PlankBundleError(
            "plank plan source changed after the frozen candidate was generated"
        )
    config_record = _file_record(ENV_CONFIG_PATH)
    try:
        config = yaml.safe_load(ENV_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PlankBundleError(f"actual-v6 config cannot be read: {exc}") from exc
    assets_cfg = config.get("assets") if isinstance(config, dict) else None
    if not isinstance(assets_cfg, dict):
        raise PlankBundleError("actual-v6 config assets section is missing")
    if (
        assets_cfg.get("franka", {}).get("variant") != "fr3"
        or assets_cfg.get("ur", {}).get("variant") != "jaka_zu7"
        or assets_cfg.get("spheres") != "real_v5"
        or assets_cfg.get("arm_usd_overrides") != _EXPECTED_ARM_USD
    ):
        raise PlankBundleError("actual-v6 config asset identity drifted")

    asset_paths = dict(_ASSET_COMPONENTS)
    asset_paths.update(
        {f"{arm}_combined_usd": relative for arm, relative in _EXPECTED_ARM_USD.items()}
    )
    assets = {
        name: _file_record(REPO_ROOT / relative)
        for name, relative in sorted(asset_paths.items())
    }
    sources = {
        "plank_plan": plan_source,
        "plank_bundle_converter": _file_record(Path(__file__)),
        "trajectory_task_bundle": _file_record(CORE_BUNDLE_SOURCE_PATH),
        "trajectory_task_replay": _file_record(REPLAY_SOURCE_PATH),
    }
    candidate_manifest = _file_record(
        DEFAULT_CANDIDATE_DIR / f"manifest_{EXPECTED_CANDIDATE_MANIFEST_SHA256}.json"
    )
    if candidate_manifest["sha256"] != candidate.manifest_sha256:
        raise PlankBundleError(
            "candidate manifest provenance changed during conversion"
        )
    source_records = {"candidate_manifest": candidate_manifest, **sources}
    binding_hashes = {
        "asset": _aggregate_digest("d032-actual-v6-assets-v1", assets),
        "config": config_record["sha256"],
        "checkpoint": hashlib.sha256(
            b"d032-scripted-nominal-no-checkpoint-v1\n"
        ).hexdigest(),
        "source": _aggregate_digest("d032-plank-bundle-sources-v1", source_records),
    }
    return {
        "candidate_manifest": candidate_manifest,
        "candidate_files": {
            name: {
                "path": candidate.manifest["files"][name]["file"],
                "bytes": candidate.manifest["files"][name]["bytes"],
                "sha256": digest,
            }
            for name, digest in sorted(candidate.file_sha256.items())
        },
        "sources": sources,
        "config": config_record,
        "assets": assets,
        "checkpoint_semantics": "NO_CHECKPOINT_SCRIPTED_NOMINAL_IDENTITY_RAW",
        "binding_hashes": binding_hashes,
    }


def build_nominal_bundle() -> NominalPlankBundle:
    """Build an in-memory zero-perturbation actual-v6 episode input."""

    candidate = load_plank_plan_candidate()
    provenance = _provenance(candidate)
    perturbation_delta = {
        arm: np.zeros((candidate.steps, dim), dtype=np.float32)
        for arm, dim in ARM_DIMS.items()
    }
    hands = {
        f"{arm}_rh56f2": {
            "arm": arm,
            "kind": "q",
            "joint_names": list(candidate.hand_joint_names[arm]),
            "values": candidate.hand_q[arm].copy(),
            "valid": candidate.hand_valid[arm].copy(),
        }
        for arm in ARM_DIMS
    }
    exogenous = {
        "schema_version": NOMINAL_SCHEMA_VERSION,
        "q_ref": {arm: values.copy() for arm, values in candidate.q_ref.items()},
        "raw_delta": {
            arm: values.copy() for arm, values in candidate.raw_delta.items()
        },
        "hands": hands,
        "objects": {
            "shared_plank": {
                "asset_id": "obj_plank",
                "asset_sha256": provenance["assets"]["plank_usda"]["sha256"],
                "prim_path": EXPECTED_OBJECT_PATH,
                "pose_world": np.asarray(PLANK_CENTER_POSE_WXYZ, dtype=np.float64),
                "linear_velocity": np.zeros(3, dtype=np.float64),
                "angular_velocity": np.zeros(3, dtype=np.float64),
                "physics": {
                    "mass_kg": 1.2,
                    "collision_shape": "cuboid",
                    "static_friction": 1.2,
                    "dynamic_friction": 0.9,
                    "restitution": 0.0,
                },
            }
        },
        "perturbation": {
            "delta": perturbation_delta,
            "target_class": "none",
            "affected_arms": [],
            "onset": 0,
            "release": 0,
        },
    }
    manifest = {
        "schema_version": NOMINAL_SCHEMA_VERSION,
        "profile": PROFILE,
        "geometry": GEOMETRY,
        "asset_identities": dict(ASSET_IDENTITIES),
        "pair_id": f"d032-plank-nominal-{candidate.manifest_sha256}",
        "dt": candidate.dt,
        "seeds": {"trajectory": 0, "perturbation": 0, "simulation": 32032},
        "hashes": dict(provenance["binding_hashes"]),
    }
    validate_exogenous_input(exogenous)
    return NominalPlankBundle(
        candidate=candidate,
        exogenous=exogenous,
        manifest=manifest,
        provenance=provenance,
    )


def write_nominal_bundle(path: str | Path) -> dict[str, Any]:
    """Write one immutable core episode directory and return its bindings."""

    built = build_nominal_bundle()
    persisted = write_episode_bundle(
        path,
        exogenous=built.exogenous,
        manifest=built.manifest,
    )
    return {"manifest": persisted, "provenance": built.provenance}


def verify_nominal_bundle(path: str | Path) -> dict[str, Any]:
    """Reload a core bundle and rebind it to current canonical source bytes."""

    loaded = load_episode_bundle(path)
    built = build_nominal_bundle()
    expected_normalized = validate_exogenous_input(built.exogenous)
    expected_digest = hashlib.sha256(_canonical_bytes(expected_normalized)).hexdigest()
    manifest = loaded["manifest"]
    if (
        manifest["schema_version"] != NOMINAL_SCHEMA_VERSION
        or manifest["pair_id"] != built.manifest["pair_id"]
        or manifest["hashes"] != built.manifest["hashes"]
        or manifest["dt"] != built.manifest["dt"]
        or manifest["seeds"] != built.manifest["seeds"]
        or manifest["exogenous_digest"] != expected_digest
    ):
        raise PlankBundleError(
            "nominal bundle binding differs from canonical candidate"
        )
    perturbation = loaded["exogenous"]["perturbation"]
    if (
        perturbation["target_class"] != "none"
        or perturbation["affected_arms"]
        or perturbation["onset"] != 0
        or perturbation["release"] != 0
        or any(np.any(delta != 0.0) for delta in perturbation["delta"].values())
    ):
        raise PlankBundleError("nominal bundle perturbation is not exactly zero")
    return {"manifest": manifest, "provenance": built.provenance}


__all__ = [
    "DEFAULT_CANDIDATE_DIR",
    "EXPECTED_CANDIDATE_MANIFEST_SHA256",
    "EXPECTED_OBJECT_PATH",
    "EXPECTED_PLAN_SOURCE_SHA256",
    "NominalPlankBundle",
    "PLANK_CENTER_POSE_WXYZ",
    "PLANK_SIZE_M",
    "PlankBundleError",
    "PlankPlanArtifact",
    "build_nominal_bundle",
    "load_plank_plan_candidate",
    "sha256_file",
    "verify_nominal_bundle",
    "write_nominal_bundle",
]
