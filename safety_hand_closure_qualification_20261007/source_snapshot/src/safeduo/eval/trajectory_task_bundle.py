"""Strict, CPU-only D-032 trajectory-task episode bundle contract.

The bundle stores only exogenous replay inputs plus a content-addressed
manifest.  Runtime outcomes (contacts, attachments, safety decisions, and
task-success results) deliberately do not belong in the exogenous payload.
This keeps the ``raw`` and ``s0`` branches causally paired: both consume the
same immutable input digest and may record their outcomes elsewhere.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "d032.actual_asset_v6.1"
NOMINAL_SCHEMA_VERSION = "d032.actual_asset_v6.2"
SUPPORTED_SCHEMA_VERSIONS = {SCHEMA_VERSION, NOMINAL_SCHEMA_VERSION}
PROFILE = "actual_asset_v6"
GEOMETRY = "v5"
ARM_DIMS = {"F_L": 7, "F_R": 7, "U_L": 6, "U_R": 6}
ASSET_IDENTITIES = {"F": "fr3", "U": "jaka_zu7", "hand": "rh56f2"}
TARGET_CLASSES = {"cross", "self_F", "self_U", "table"}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BANNED_IDENTITY_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:v7|ur5e?|rh56dfx)(?:$|[^a-z0-9])", re.I
)
_RESULT_KEY_RE = re.compile(
    r"(?:attachment|contact).*(?:result|actual)|(?:result|actual).*(?:attachment|contact)",
    re.I,
)


class BundleValidationError(ValueError):
    """The episode bundle violates the frozen D-032 input contract."""


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BundleValidationError(f"{label} must be a mapping")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise BundleValidationError(
            f"{label} keys differ: missing={missing}, extra={extra}"
        )


def _numeric_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise BundleValidationError(f"{label} must be numeric")
    if array.shape != shape:
        raise BundleValidationError(f"{label} shape {array.shape} != {shape}")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise BundleValidationError(f"{label} must contain only finite values")
    return array.copy()


def _valid_mask(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind != "b":
        raise BundleValidationError(f"{label} valid mask must be boolean")
    if array.shape != shape:
        raise BundleValidationError(
            f"{label} valid mask shape {array.shape} != {shape}"
        )
    return np.asarray(array, dtype=np.bool_).copy()


def _reject_outcome_channels(value: Any, path: str = "exogenous") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if _RESULT_KEY_RE.search(key_text):
                raise BundleValidationError(
                    f"{path}.{key_text} is an attachment/contact result channel"
                )
            _reject_outcome_channels(child, f"{path}.{key_text}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_outcome_channels(child, f"{path}[{index}]")


def _reject_deprecated_identity(value: str, label: str) -> None:
    if _BANNED_IDENTITY_RE.search(value):
        raise BundleValidationError(f"{label} contains a v7/UR5/RH56DFX identity")


def _validate_objects(value: Any) -> dict[str, dict[str, Any]]:
    objects = _require_mapping(value, "objects")
    if not objects:
        raise BundleValidationError("objects must contain at least one physical object")
    normalized: dict[str, dict[str, Any]] = {}
    expected = {
        "asset_id",
        "asset_sha256",
        "prim_path",
        "pose_world",
        "linear_velocity",
        "angular_velocity",
        "physics",
    }
    physics_expected = {
        "mass_kg",
        "collision_shape",
        "static_friction",
        "dynamic_friction",
        "restitution",
    }
    for name, raw in objects.items():
        if not isinstance(name, str) or not name:
            raise BundleValidationError("object names must be non-empty strings")
        obj = _require_mapping(raw, f"objects.{name}")
        _require_exact_keys(obj, expected, f"objects.{name}")
        asset_id = str(obj["asset_id"])
        prim_path = str(obj["prim_path"])
        if not asset_id or not prim_path.startswith("/"):
            raise BundleValidationError(f"objects.{name} identity/path is invalid")
        _reject_deprecated_identity(asset_id, f"objects.{name}.asset_id")
        _reject_deprecated_identity(prim_path, f"objects.{name}.prim_path")
        asset_sha = str(obj["asset_sha256"])
        if not _SHA256_RE.fullmatch(asset_sha):
            raise BundleValidationError(f"objects.{name}.asset_sha256 is not sha256")

        pose = _numeric_array(obj["pose_world"], (7,), f"objects.{name}.pose_world")
        if not math.isclose(float(np.linalg.norm(pose[3:])), 1.0, abs_tol=1e-6):
            raise BundleValidationError(
                f"objects.{name} quaternion must be unit length"
            )
        linear = _numeric_array(
            obj["linear_velocity"], (3,), f"objects.{name}.linear_velocity"
        )
        angular = _numeric_array(
            obj["angular_velocity"], (3,), f"objects.{name}.angular_velocity"
        )
        physics = _require_mapping(obj["physics"], f"objects.{name}.physics")
        _require_exact_keys(physics, physics_expected, f"objects.{name}.physics")
        collision_shape = str(physics["collision_shape"])
        if not collision_shape:
            raise BundleValidationError(
                f"objects.{name}.physics.collision_shape is empty"
            )
        numeric_physics: dict[str, float] = {}
        for field in (
            "mass_kg",
            "static_friction",
            "dynamic_friction",
            "restitution",
        ):
            raw_number = physics[field]
            if isinstance(raw_number, bool) or not isinstance(
                raw_number, (int, float, np.integer, np.floating)
            ):
                raise BundleValidationError(
                    f"objects.{name}.physics.{field} is not numeric"
                )
            number = float(raw_number)
            if not math.isfinite(number) or number < 0.0:
                raise BundleValidationError(
                    f"objects.{name}.physics.{field} must be finite and non-negative"
                )
            numeric_physics[field] = number
        if numeric_physics["mass_kg"] <= 0.0:
            raise BundleValidationError(
                f"objects.{name}.physics.mass_kg must be positive"
            )
        normalized[name] = {
            "asset_id": asset_id,
            "asset_sha256": asset_sha,
            "prim_path": prim_path,
            "pose_world": pose,
            "linear_velocity": linear,
            "angular_velocity": angular,
            "physics": {"collision_shape": collision_shape, **numeric_physics},
        }
    return normalized


def validate_exogenous_input(exogenous: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a D-032 replay input without mutating the caller."""
    raw = _require_mapping(exogenous, "exogenous")
    _reject_outcome_channels(raw)
    _require_exact_keys(
        raw,
        {"schema_version", "q_ref", "raw_delta", "hands", "objects", "perturbation"},
        "exogenous",
    )
    schema_version = raw["schema_version"]
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise BundleValidationError(
            f"schema_version must be one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )

    q_ref_raw = _require_mapping(raw["q_ref"], "q_ref")
    delta_raw = _require_mapping(raw["raw_delta"], "raw_delta")
    if set(q_ref_raw) != set(ARM_DIMS) or set(delta_raw) != set(ARM_DIMS):
        raise BundleValidationError(
            "q_ref/raw_delta arm names must exactly match four arms"
        )

    first = np.asarray(q_ref_raw["F_L"])
    if first.ndim != 2 or first.shape[0] < 2:
        raise BundleValidationError("q_ref must have T+1 rows with T >= 1")
    steps = int(first.shape[0] - 1)
    q_ref = {
        arm: _numeric_array(q_ref_raw[arm], (steps + 1, dim), f"q_ref.{arm}")
        for arm, dim in ARM_DIMS.items()
    }
    raw_delta = {
        arm: _numeric_array(delta_raw[arm], (steps, dim), f"raw_delta.{arm}")
        for arm, dim in ARM_DIMS.items()
    }

    hands_raw = _require_mapping(raw["hands"], "hands")
    if len(hands_raw) != len(ARM_DIMS):
        raise BundleValidationError("hands must name exactly one hand for each arm")
    hands: dict[str, dict[str, Any]] = {}
    seen_arms: set[str] = set()
    for name, raw_hand in hands_raw.items():
        if not isinstance(name, str) or not name:
            raise BundleValidationError("hand names must be non-empty strings")
        hand = _require_mapping(raw_hand, f"hands.{name}")
        if "kind" not in hand:
            raise BundleValidationError(f"hands.{name}.kind is required")
        kind = hand["kind"]
        expected = {"arm", "kind", "joint_names", "values", "valid"}
        if kind == "delta":
            expected.add("initial")
        _require_exact_keys(hand, expected, f"hands.{name}")
        arm = hand["arm"]
        if arm not in ARM_DIMS:
            raise BundleValidationError(f"hands.{name} has unknown arm {arm!r}")
        if arm in seen_arms:
            raise BundleValidationError(f"multiple named hands target arm {arm}")
        seen_arms.add(arm)
        values_probe = np.asarray(hand["values"])
        if values_probe.ndim != 2 or values_probe.shape[1] < 1:
            raise BundleValidationError(f"hands.{name}.values must be rank-2")
        hand_dim = int(values_probe.shape[1])
        joint_names_raw = hand["joint_names"]
        if not isinstance(joint_names_raw, (list, tuple)):
            raise BundleValidationError(f"hands.{name}.joint_names must be a list")
        joint_names = [str(joint_name) for joint_name in joint_names_raw]
        if (
            len(joint_names) != hand_dim
            or len(set(joint_names)) != hand_dim
            or any(not joint_name for joint_name in joint_names)
        ):
            raise BundleValidationError(
                f"hands.{name}.joint_names must be unique non-empty names "
                f"matching hand dimension {hand_dim}"
            )
        rows = steps + 1 if kind == "q" else steps if kind == "delta" else -1
        if rows < 0:
            raise BundleValidationError(f"hands.{name}.kind must be q or delta")
        values = _numeric_array(
            hand["values"], (rows, hand_dim), f"hands.{name}.values"
        )
        valid = _valid_mask(hand["valid"], values.shape, f"hands.{name}.valid")
        normalized_hand: dict[str, Any] = {
            "arm": arm,
            "kind": kind,
            "joint_names": joint_names,
            "values": values,
            "valid": valid,
        }
        if kind == "delta":
            normalized_hand["initial"] = _numeric_array(
                hand["initial"], (hand_dim,), f"hands.{name}.initial"
            )
        hands[name] = normalized_hand
    if seen_arms != set(ARM_DIMS):
        raise BundleValidationError("hands must cover exactly the four known arms")

    objects = _validate_objects(raw["objects"])

    perturb_raw = _require_mapping(raw["perturbation"], "perturbation")
    _require_exact_keys(
        perturb_raw,
        {"delta", "target_class", "affected_arms", "onset", "release"},
        "perturbation",
    )
    target_class = perturb_raw["target_class"]
    is_nominal = schema_version == NOMINAL_SCHEMA_VERSION and target_class == "none"
    if not is_nominal and target_class not in TARGET_CLASSES:
        raise BundleValidationError(
            f"unknown perturbation target class {target_class!r}"
        )
    affected_raw = perturb_raw["affected_arms"]
    if not isinstance(affected_raw, (list, tuple)):
        raise BundleValidationError("perturbation affected_arms must be a list")
    affected = list(affected_raw)
    onset = perturb_raw["onset"]
    release = perturb_raw["release"]
    if (
        isinstance(onset, bool)
        or isinstance(release, bool)
        or not isinstance(onset, (int, np.integer))
        or not isinstance(release, (int, np.integer))
    ):
        raise BundleValidationError("perturbation onset/release must be integer steps")
    onset = int(onset)
    release = int(release)
    perturb_delta_raw = _require_mapping(perturb_raw["delta"], "perturbation.delta")
    if set(perturb_delta_raw) != set(ARM_DIMS):
        raise BundleValidationError(
            "perturbation.delta arm names must exactly match four arms"
        )
    perturb_delta = {
        arm: _numeric_array(
            perturb_delta_raw[arm], (steps, dim), f"perturbation.delta.{arm}"
        )
        for arm, dim in ARM_DIMS.items()
    }
    if is_nominal:
        if affected or onset != 0 or release != 0:
            raise BundleValidationError(
                "v6.2 nominal perturbation requires affected_arms=[] and onset=release=0"
            )
        if any(np.any(delta != 0.0) for delta in perturb_delta.values()):
            raise BundleValidationError(
                "v6.2 nominal perturbation requires all four delta arrays to be zero"
            )
    else:
        if not affected:
            raise BundleValidationError(
                "perturbation affected_arms must be a non-empty list"
            )
        if len(set(affected)) != len(affected) or not set(affected).issubset(ARM_DIMS):
            raise BundleValidationError(
                "perturbation contains duplicate or unknown arm"
            )
        if not 0 <= onset < release <= steps:
            raise BundleValidationError(
                "perturbation onset/release must satisfy "
                f"0 <= onset < release <= T ({steps})"
            )
        for arm, delta in perturb_delta.items():
            if arm not in affected and np.any(delta != 0.0):
                raise BundleValidationError(
                    f"unaffected arm {arm} has non-zero perturbation"
                )
            if np.any(delta[:onset] != 0.0) or np.any(delta[release:] != 0.0):
                raise BundleValidationError(
                    f"perturbation for {arm} is non-zero outside window"
                )
        if not any(
            np.any(perturb_delta[arm][onset:release] != 0.0) for arm in affected
        ):
            raise BundleValidationError(
                "affected arms have no non-zero perturbation in window"
            )

    return {
        "schema_version": schema_version,
        "q_ref": q_ref,
        "raw_delta": raw_delta,
        "hands": hands,
        "objects": objects,
        "perturbation": {
            "delta": perturb_delta,
            "target_class": target_class,
            "affected_arms": affected,
            "onset": onset,
            "release": release,
        },
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise BundleValidationError(
            f"value is not canonical-JSON serializable: {exc}"
        ) from exc
    return (text + "\n").encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _initial_state(normalized: Mapping[str, Any]) -> dict[str, Any]:
    hand_initial: dict[str, Any] = {}
    for name, hand in normalized["hands"].items():
        initial = hand["values"][0] if hand["kind"] == "q" else hand["initial"]
        hand_initial[name] = {
            "arm": hand["arm"],
            "q": initial,
            "valid": hand["valid"][0],
        }
    return {
        "arms": {arm: q[0] for arm, q in normalized["q_ref"].items()},
        "hands": hand_initial,
        "objects": normalized["objects"],
    }


_BASE_MANIFEST_KEYS = {
    "schema_version",
    "profile",
    "geometry",
    "asset_identities",
    "pair_id",
    "dt",
    "seeds",
    "hashes",
}
_PERSISTED_MANIFEST_KEYS = _BASE_MANIFEST_KEYS | {
    "arm_dims",
    "steps",
    "initial_state_digest",
    "exogenous_digest",
    "branches",
}


def _validate_manifest_base(manifest: Any) -> dict[str, Any]:
    raw = _require_mapping(manifest, "manifest")
    _require_exact_keys(raw, _BASE_MANIFEST_KEYS, "manifest")
    schema_version = raw["schema_version"]
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise BundleValidationError(
            "manifest schema_version must be one of "
            f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    if raw["profile"] != PROFILE:
        raise BundleValidationError(f"manifest profile must be {PROFILE}")
    if raw["geometry"] != GEOMETRY:
        raise BundleValidationError(f"manifest geometry must be {GEOMETRY}")
    identities = _require_mapping(raw["asset_identities"], "manifest.asset_identities")
    if dict(identities) != ASSET_IDENTITIES:
        raise BundleValidationError(
            "asset identities must be FR3 + JAKA Zu7 + RH56F2 (no v7/UR5/RH56DFX)"
        )
    for label, identity in identities.items():
        _reject_deprecated_identity(str(identity), f"asset_identities.{label}")
    pair_id = raw["pair_id"]
    if not isinstance(pair_id, str) or not pair_id.strip():
        raise BundleValidationError("manifest pair_id must be a non-empty string")
    _reject_deprecated_identity(pair_id, "manifest.pair_id")
    dt = raw["dt"]
    if isinstance(dt, bool) or not isinstance(
        dt, (int, float, np.integer, np.floating)
    ):
        raise BundleValidationError("manifest dt must be numeric")
    dt = float(dt)
    if not math.isfinite(dt) or dt <= 0.0:
        raise BundleValidationError("manifest dt must be finite and positive")
    seeds_raw = _require_mapping(raw["seeds"], "manifest.seeds")
    _require_exact_keys(
        seeds_raw, {"trajectory", "perturbation", "simulation"}, "seeds"
    )
    seeds: dict[str, int] = {}
    for name, value in seeds_raw.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < 0
        ):
            raise BundleValidationError(
                f"manifest seed {name} must be a non-negative int"
            )
        seeds[name] = int(value)
    hashes_raw = _require_mapping(raw["hashes"], "manifest.hashes")
    _require_exact_keys(
        hashes_raw, {"asset", "config", "checkpoint", "source"}, "hashes"
    )
    hashes = {name: str(value) for name, value in hashes_raw.items()}
    for name, value in hashes.items():
        if not _SHA256_RE.fullmatch(value):
            raise BundleValidationError(f"manifest hash {name} is not lowercase sha256")
    return {
        "schema_version": schema_version,
        "profile": PROFILE,
        "geometry": GEOMETRY,
        "asset_identities": dict(ASSET_IDENTITIES),
        "pair_id": pair_id,
        "dt": dt,
        "seeds": seeds,
        "hashes": hashes,
    }


def _persisted_manifest(
    base: Mapping[str, Any], normalized: Mapping[str, Any]
) -> dict[str, Any]:
    exogenous_digest = _digest(normalized)
    initial_state_digest = _digest(_initial_state(normalized))
    steps = int(normalized["raw_delta"]["F_L"].shape[0])
    return {
        **base,
        "arm_dims": dict(ARM_DIMS),
        "steps": steps,
        "initial_state_digest": initial_state_digest,
        "exogenous_digest": exogenous_digest,
        "branches": {
            "raw": {"exogenous_digest": exogenous_digest},
            "s0": {"exogenous_digest": exogenous_digest},
        },
    }


def _write_fsync(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def write_episode_bundle(
    path: str | os.PathLike[str],
    *,
    exogenous: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a fail-closed bundle under an atomic no-overwrite directory claim.

    The target directory is claimed with ``mkdir(exist_ok=False)`` before any
    persisted file is published.  ``manifest.json`` is moved last, so a
    concurrent reader can only see an incomplete bundle (which the strict
    loader rejects) or the complete pair; an existing target is never replaced.
    """
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"episode bundle already exists: {target}")
    normalized = validate_exogenous_input(exogenous)
    base = _validate_manifest_base(manifest)
    if base["schema_version"] != normalized["schema_version"]:
        raise BundleValidationError("manifest and exogenous schema_version must match")
    persisted = _persisted_manifest(base, normalized)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    claimed = False
    try:
        _write_fsync(temporary / "exogenous.json", _canonical_bytes(normalized))
        _write_fsync(temporary / "manifest.json", _canonical_bytes(persisted))
        directory_fd = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        target.mkdir(mode=0o700, exist_ok=False)
        claimed = True
        (temporary / "exogenous.json").replace(target / "exogenous.json")
        (temporary / "manifest.json").replace(target / "manifest.json")
        target_fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(target_fd)
        finally:
            os.close(target_fd)
        parent_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        temporary.rmdir()
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        if claimed and target.exists():
            shutil.rmtree(target)
        raise
    return persisted


def load_episode_bundle(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a bundle, revalidate every field, and reject digest tampering."""
    root = Path(path)
    if not root.is_dir():
        raise BundleValidationError(f"episode bundle is not a directory: {root}")
    names = {entry.name for entry in root.iterdir()}
    if names != {"exogenous.json", "manifest.json"}:
        raise BundleValidationError(
            f"bundle files differ from strict schema: {sorted(names)}"
        )
    try:
        exogenous_raw = json.loads(
            (root / "exogenous.json").read_text(encoding="utf-8")
        )
        manifest_raw = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleValidationError(f"bundle JSON cannot be read: {exc}") from exc
    normalized = validate_exogenous_input(exogenous_raw)
    persisted_raw = _require_mapping(manifest_raw, "manifest")
    _require_exact_keys(persisted_raw, _PERSISTED_MANIFEST_KEYS, "persisted manifest")
    base = {key: persisted_raw[key] for key in _BASE_MANIFEST_KEYS}
    normalized_base = _validate_manifest_base(base)
    if normalized_base["schema_version"] != normalized["schema_version"]:
        raise BundleValidationError("manifest and exogenous schema_version must match")
    expected = _persisted_manifest(normalized_base, normalized)
    if persisted_raw != _jsonable(expected):
        if persisted_raw.get("exogenous_digest") != expected["exogenous_digest"]:
            raise BundleValidationError("exogenous digest mismatch")
        raise BundleValidationError("persisted manifest binding mismatch")
    return {"manifest": expected, "exogenous": normalized}


__all__ = [
    "ARM_DIMS",
    "BundleValidationError",
    "GEOMETRY",
    "NOMINAL_SCHEMA_VERSION",
    "PROFILE",
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "load_episode_bundle",
    "validate_exogenous_input",
    "write_episode_bundle",
]
