"""Strict, transport-neutral Manus input adapter for D-032 recordings.

The audited legacy wire protocol is a packed SHM v1 block with one ``seq``, one
epoch ``timestamp_us``, and two hand slots in ``(right, left)`` order.  Each
valid hand exposes exactly 25 node positions and ``xyzw`` quaternions.  This
module does not open that transport.  A CLI or transport process must decode
two independent operator streams into the typed values below and supply a
receipt timestamp from the same epoch clock.

The adapter deliberately does not import the legacy RH56DFX retargeter.  Arm
increments and RH56F2 absolute hand targets are produced only from an explicit,
content-hashed actual-RH56F2 calibration.  Missing, stale, reordered, or
out-of-calibration input fails before either operator's state is advanced.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import re
import threading
import weakref
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from safeduo.eval.glove_episode_recorder import (
    ARM_ORDER,
    OPERATOR_ARM_ORDER,
    OPERATOR_ORDER,
    RH56F2_LEFT_JOINT_NAMES,
    RH56F2_RIGHT_JOINT_NAMES,
    ArmDelta,
    HandTarget,
    OperatorFrame,
    RecorderContractError,
)
from safeduo.eval.manus_shm_v1 import (
    ManusShmV1ContractError,
    ManusShmV1Receipt,
    require_decoder_issued_receipt,
)
from safeduo.eval.trajectory_task_bundle import ARM_DIMS, validate_exogenous_input


CALIBRATION_SCHEMA_VERSION = "d032.manus_rh56f2_calibration.v2"
ADAPTER_PROVENANCE_SCHEMA_VERSION = "d032.manus_adapter.v2"
MANUS_NODE_IDS = tuple(range(25))
MANUS_WIRE_HAND_ORDER = ("right", "left")

_EXPECTED_SOURCE_PROTOCOL = {
    "magic": "MANU",
    "version": 1,
    "block_size_bytes": 1448,
    "header_size_bytes": 32,
    "hand_size_bytes": 708,
    "node_size_bytes": 28,
    "max_nodes": 25,
    "node_ids": list(MANUS_NODE_IDS),
    "side_codes": {"left": 1, "right": 2},
    "hand_count_required": 2,
    "wire_hand_slot_order": list(MANUS_WIRE_HAND_ORDER),
    "quaternion_order": "xyzw",
    "timestamp_unit": "microseconds_since_epoch",
    "skeleton_position_frame": "manus_local",
    "wrist_pose_frame": "operator_calibrated_world",
}
_EXPECTED_OWNERSHIP = {"F": ["F_L", "F_R"], "U": ["U_L", "U_R"]}
_JOINT_NAMES_BY_SIDE = {
    "left": RH56F2_LEFT_JOINT_NAMES,
    "right": RH56F2_RIGHT_JOINT_NAMES,
}
_SIDE_BY_ARM = {
    "F_L": "left",
    "F_R": "right",
    "U_L": "left",
    "U_R": "right",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_OPERATOR_SAMPLE_ISSUER = object()
_WRIST_POSE_ISSUER = object()
FROZEN_ACTUAL_ARM_ASSET_SHA256 = MappingProxyType(
    {
        "F_L": "f97e1e24d1c99cd87a0061fe42e0a675d7d675d1a65fd1f743aca88a3b2ebedb",
        "F_R": "b5c24e25f71736cd51024d3d2314ed5e8a85802aeba46354d44d66b2e389838b",
        "U_L": "07fc75b3da410662905adb1d02671562a971420655c8838ee46cfeba2083cfb3",
        "U_R": "535998ec046f55e1bb9f769a341a4988f6bdcf88041dd4f777072640e5d54529",
    }
)
FROZEN_ACTUAL_HAND_EVIDENCE_SHA256 = MappingProxyType(
    {
        "left": MappingProxyType(
            {
                "urdf_sha256": "18f0d684cc003d24121560f28cf4e240e5a1e28d8b86c9b96e7f6c25154d8bf3",
                "inventory_sha256": "590e8af5b3ce04bf7009327c548771137ab13c8d077c4e8e3c690a273361ce99",
            }
        ),
        "right": MappingProxyType(
            {
                "urdf_sha256": "2dfa011010e81f46e30c8456ab9bf9e662d1964f1a2548d51244a25de2cca66a",
                "inventory_sha256": "70ae00f2abfaf47aff0e598a0e1dcc951344834adfdc08232d63e0811ac23cf0",
            }
        ),
    }
)


class AdapterContractError(ValueError):
    """The calibration, source frame, or adapter state violates the contract."""


@dataclass(frozen=True)
class CalibrationAssetEvidence:
    """Caller-supplied bytes for every asset bound by calibration v2.

    Paths are intentionally excluded: calibration authority comes from exact
    content and its digest, not a mutable filesystem location.
    """

    arm_asset_bytes: Mapping[str, bytes]
    hand_urdf_bytes: Mapping[str, bytes]
    hand_inventory_bytes: Mapping[str, bytes]


@dataclass(frozen=True)
class ManusHandSample:
    """One decoded Manus hand plus a separately timestamped world wrist pose."""

    side: str
    valid: bool
    node_ids: Sequence[int]
    positions_m: Any
    quaternions_xyzw: Any
    pose_position_m: Any
    pose_quaternion_xyzw: Any
    pose_timestamp_us: int
    glove_device_id: str


@dataclass(frozen=True, eq=False)
class ManusOperatorSample:
    """One complete decoded SHM-v1-style frame for one named operator."""

    operator: str
    seq: int
    timestamp_us: int
    hands: Sequence[ManusHandSample]
    stream_id: str
    operator_device_id: str
    source_id: str
    producer_identity: str
    decoder_identity: str
    pose_source: str
    clock_domain: str
    raw_frame_sha256: str
    formal_live_eligible: bool
    decoder_receipt: ManusShmV1Receipt | None = None
    wrist_pose_receipts: Sequence[WristPoseReceipt] = ()
    _issuance_token: Any = None


@dataclass(frozen=True, eq=False)
class WristPoseReceipt:
    """Independently issued wrist-pose evidence for one operator hand."""

    operator: str
    side: str
    pose_position_m: Any
    pose_quaternion_xyzw: Any
    pose_timestamp_us: int
    receipt_timestamp_us: int
    raw_pose_sha256: str
    pose_source: str
    clock_domain: str
    capture_class: str
    formal_live_eligible: bool
    external_blockers: tuple[str, ...]
    _provenance: Mapping[str, Any]
    _raw_pose_payload: bytes
    _issuance_token: Any = None

    @property
    def provenance(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self._provenance))

    @property
    def raw_pose_payload(self) -> bytes:
        """Return the exact immutable input bytes bound by this receipt."""
        return self._raw_pose_payload


_ISSUED_OPERATOR_SAMPLES: weakref.WeakKeyDictionary[ManusOperatorSample, object] = (
    weakref.WeakKeyDictionary()
)
_ISSUED_WRIST_POSES: weakref.WeakKeyDictionary[WristPoseReceipt, object] = (
    weakref.WeakKeyDictionary()
)


@dataclass(frozen=True, eq=False)
class PreparedManusPair:
    """Opaque generation-fenced adapter output awaiting recorder acceptance."""

    generation: int
    _issuance_token: Any


@dataclass(frozen=True)
class _PreparedRecord:
    generation: int
    token: object
    pair: dict[str, _ValidatedOperator]
    frames: tuple[OperatorFrame, OperatorFrame]
    output_trace: dict[str, Any]


@dataclass(frozen=True)
class _AdapterTimeline:
    primed: bool
    last: Mapping[str, _ValidatedOperator]
    accepted_transitions: int
    generation: int
    baseline_receipts: Mapping[str, Any] | None
    transition_receipts: tuple[Mapping[str, Any], ...]
    initial_output: Mapping[str, Any] | None
    transition_outputs: tuple[Mapping[str, Any], ...]
    diagnostic_direct_commits: int
    bound_recorder: Any | None
    baseline_raw_evidence: Mapping[str, Any] | None
    transition_raw_evidence: tuple[Mapping[str, Any], ...]


def _staging_fault_point(_name: str) -> None:
    """No-op seam for deterministic tests of pre-commit allocation failures."""


def _raw_evidence_from_pair(
    pair: Mapping[str, _ValidatedOperator],
) -> dict[str, Any]:
    return {
        operator: {
            "manus_raw_payload": bytes(pair[operator].raw_frame_payload),
            "wrist_raw_payloads": {
                side: bytes(pair[operator].wrist_raw_payloads[side])
                for side in MANUS_WIRE_HAND_ORDER
            },
        }
        for operator in OPERATOR_ORDER
    }


@dataclass(frozen=True)
class _ArmCalibration:
    side: str
    twist_to_delta: np.ndarray
    max_abs_delta: np.ndarray
    max_translation_step_m: float
    max_rotation_step_rad: float


@dataclass(frozen=True)
class _JointContract:
    name: str
    lower: float
    upper: float
    mimic_source: str | None
    mimic_multiplier: float
    mimic_offset: float


@dataclass(frozen=True)
class _HandCalibration:
    joint_names: tuple[str, ...]
    independent_joint_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_max_abs_deviation: np.ndarray
    feature_to_independent_q: np.ndarray
    independent_q_bias: np.ndarray
    joint_contract: tuple[_JointContract, ...]


@dataclass(frozen=True)
class _Calibration:
    calibration_id: str
    arms: dict[str, _ArmCalibration]
    hands: dict[str, _HandCalibration]
    asset_contract: dict[str, Any]
    device_bindings: dict[str, Any]


@dataclass(frozen=True)
class _ValidatedHand:
    side: str
    glove_device_id: str
    positions_m: np.ndarray
    quaternions_xyzw: np.ndarray
    pose_position_m: np.ndarray
    pose_quaternion_xyzw: np.ndarray
    pose_timestamp_us: int


@dataclass(frozen=True)
class _ValidatedOperator:
    operator: str
    seq: int
    timestamp_us: int
    hands: dict[str, _ValidatedHand]
    raw_frame_sha256: str
    formal_live_eligible: bool
    evidence_trace: dict[str, Any]
    raw_frame_payload: bytes
    wrist_raw_payloads: Mapping[str, bytes]


def _jsonable_finite(value: Any, path: str = "calibration") -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable_finite(value.tolist(), path)
    if isinstance(value, np.generic):
        return _jsonable_finite(value.item(), path)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise AdapterContractError(f"{path} keys must be strings")
            normalized[key] = _jsonable_finite(child, f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _jsonable_finite(child, f"{path}[{index}]")
            for index, child in enumerate(value)
        ]
    if isinstance(value, float) and not math.isfinite(value):
        raise AdapterContractError(f"{path} must contain only finite values")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise AdapterContractError(
        f"{path} contains non-canonical JSON value {type(value).__name__}"
    )


def _canonical_calibration_bytes(calibration: Mapping[str, Any]) -> bytes:
    if not isinstance(calibration, Mapping):
        raise AdapterContractError("calibration must be a mapping")
    try:
        normalized = _jsonable_finite(calibration)
        encoded = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AdapterContractError(
            f"calibration is not canonical finite JSON: {exc}"
        ) from exc
    return (encoded + "\n").encode("utf-8")


def calibration_content_sha256(calibration: Mapping[str, Any]) -> str:
    """Return the deterministic digest that must accompany a calibration."""
    return hashlib.sha256(_canonical_calibration_bytes(calibration)).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AdapterContractError(f"{label} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise AdapterContractError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _reject_rh56dfx(value: Any, path: str = "calibration") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_rh56dfx(str(key), f"{path}.key")
            _reject_rh56dfx(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_rh56dfx(child, f"{path}[{index}]")
    elif isinstance(value, str) and "rh56dfx" in value.lower():
        raise AdapterContractError(
            f"{path} references RH56DFX; no RH56DFX-to-RH56F2 fallback is allowed"
        )


def _finite_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise AdapterContractError(f"{label} must be numeric")
    if array.shape != shape:
        raise AdapterContractError(f"{label} shape {array.shape} != {shape}")
    normalized = np.asarray(array, dtype=np.float64)
    if not np.isfinite(normalized).all():
        raise AdapterContractError(f"{label} must contain only finite values")
    return normalized.copy()


def _positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise AdapterContractError(f"{label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise AdapterContractError(f"{label} must be finite and positive")
    return normalized


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise AdapterContractError(f"{label} must be an integer")
    normalized = int(value)
    if normalized < 0:
        raise AdapterContractError(f"{label} must be non-negative")
    return normalized


def _runtime_limit(value: Any, label: str, *, allow_zero: bool) -> int:
    normalized = _nonnegative_int(value, label)
    if not allow_zero and normalized == 0:
        raise AdapterContractError(f"{label} must be positive")
    return normalized


def _sequence(value: Any, label: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AdapterContractError(f"{label} must be a sequence")
    return tuple(value)


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdapterContractError(f"{label} must be a non-empty string")
    return value


def _sha256(value: Any, label: str) -> str:
    normalized = _nonempty_string(value, label)
    if not _SHA256_RE.fullmatch(normalized):
        raise AdapterContractError(f"{label} must be a lowercase sha256")
    return normalized


def _finite_scalar(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise AdapterContractError(f"{label} must be numeric")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise AdapterContractError(f"{label} must be numeric") from exc
    if not math.isfinite(normalized):
        raise AdapterContractError(f"{label} must be finite")
    return normalized


def _joint_from_mapping(value: Any, label: str) -> _JointContract:
    raw = _mapping(value, label)
    _exact_keys(raw, {"name", "lower", "upper", "mimic"}, label)
    name = _nonempty_string(raw["name"], f"{label}.name")
    lower = _finite_scalar(raw["lower"], f"{label}.lower")
    upper = _finite_scalar(raw["upper"], f"{label}.upper")
    if lower >= upper:
        raise AdapterContractError(f"{label} asset joint limit is invalid")
    mimic_source: str | None = None
    mimic_multiplier = 1.0
    mimic_offset = 0.0
    if raw["mimic"] is not None:
        mimic = _mapping(raw["mimic"], f"{label}.mimic")
        _exact_keys(mimic, {"source", "multiplier", "offset"}, f"{label}.mimic")
        mimic_source = _nonempty_string(mimic["source"], f"{label}.mimic.source")
        mimic_multiplier = _finite_scalar(
            mimic["multiplier"], f"{label}.mimic.multiplier"
        )
        mimic_offset = _finite_scalar(mimic["offset"], f"{label}.mimic.offset")
    return _JointContract(
        name=name,
        lower=lower,
        upper=upper,
        mimic_source=mimic_source,
        mimic_multiplier=mimic_multiplier,
        mimic_offset=mimic_offset,
    )


def _parse_urdf_contract(payload: bytes, side: str) -> tuple[_JointContract, ...]:
    try:
        root = ET.fromstring(payload)
    except (ET.ParseError, ValueError) as exc:
        raise AdapterContractError(
            f"{side} RH56F2 URDF cannot be parsed: {exc}"
        ) from exc
    parsed: list[_JointContract] = []
    for index, joint in enumerate(root.findall("joint")):
        if joint.attrib.get("type") != "revolute":
            continue
        name = joint.attrib.get("name")
        limit = joint.find("limit")
        if not name or limit is None:
            raise AdapterContractError(
                f"{side} URDF revolute joint {index} is incomplete"
            )
        mimic_element = joint.find("mimic")
        mimic = None
        if mimic_element is not None:
            mimic = {
                "source": mimic_element.attrib.get("joint"),
                "multiplier": mimic_element.attrib.get("multiplier", "1"),
                "offset": mimic_element.attrib.get("offset", "0"),
            }
        parsed.append(
            _joint_from_mapping(
                {
                    "name": name,
                    "lower": limit.attrib.get("lower"),
                    "upper": limit.attrib.get("upper"),
                    "mimic": mimic,
                },
                f"{side} URDF joint[{len(parsed)}]",
            )
        )
    return tuple(parsed)


def _parse_inventory_contract(payload: bytes, side: str) -> tuple[_JointContract, ...]:
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AdapterContractError(
            f"{side} RH56F2 inventory cannot be parsed: {exc}"
        ) from exc
    if not isinstance(decoded, list):
        raise AdapterContractError(f"{side} inventory must contain a robot list")
    expected_robot = f"RH56F2_{'L' if side == 'left' else 'R'}"
    candidates = [
        item
        for item in decoded
        if isinstance(item, Mapping) and item.get("robot_name") == expected_robot
    ]
    if len(candidates) != 1:
        raise AdapterContractError(
            f"{side} inventory must contain exactly one {expected_robot} robot"
        )
    robot = _mapping(candidates[0], f"{side} inventory robot")
    joints = robot.get("joints")
    if not isinstance(joints, list):
        raise AdapterContractError(f"{side} inventory joints are missing")
    parsed: list[_JointContract] = []
    for index, joint_value in enumerate(joints):
        joint = _mapping(joint_value, f"{side} inventory joint[{index}]")
        if joint.get("type") != "revolute":
            continue
        limit = _mapping(joint.get("limit"), f"{side} inventory joint[{index}].limit")
        mimic_raw = joint.get("mimic")
        mimic = None
        if mimic_raw is not None:
            source = _mapping(mimic_raw, f"{side} inventory joint[{index}].mimic")
            mimic = {
                "source": source.get("joint"),
                "multiplier": source.get("multiplier", "1"),
                "offset": source.get("offset", "0"),
            }
        parsed.append(
            _joint_from_mapping(
                {
                    "name": joint.get("name"),
                    "lower": limit.get("lower"),
                    "upper": limit.get("upper"),
                    "mimic": mimic,
                },
                f"{side} inventory revolute joint[{len(parsed)}]",
            )
        )
    return tuple(parsed)


def _bytes_map(value: Any, expected: set[str], label: str) -> dict[str, bytes]:
    raw = _mapping(value, label)
    if set(raw) != expected:
        raise AdapterContractError(f"{label} asset evidence keys differ")
    normalized: dict[str, bytes] = {}
    for key in sorted(expected):
        payload = raw[key]
        if not isinstance(payload, (bytes, bytearray, memoryview)) or len(payload) == 0:
            raise AdapterContractError(
                f"{label}.{key} asset evidence must be non-empty bytes"
            )
        normalized[key] = bytes(payload)
    return normalized


def _validate_asset_contract(
    value: Any,
    evidence: CalibrationAssetEvidence,
) -> tuple[dict[str, Any], dict[str, tuple[_JointContract, ...]]]:
    if not isinstance(evidence, CalibrationAssetEvidence):
        raise AdapterContractError("asset_evidence must be CalibrationAssetEvidence")
    asset_contract = _mapping(value, "calibration.asset_contract")
    _exact_keys(asset_contract, {"arms", "hands"}, "calibration.asset_contract")
    arm_bytes = _bytes_map(
        evidence.arm_asset_bytes, set(ARM_ORDER), "arm asset evidence"
    )
    urdf_bytes = _bytes_map(
        evidence.hand_urdf_bytes, {"left", "right"}, "hand URDF evidence"
    )
    inventory_bytes = _bytes_map(
        evidence.hand_inventory_bytes,
        {"left", "right"},
        "hand inventory evidence",
    )

    arms_raw = _mapping(asset_contract["arms"], "asset_contract.arms")
    if set(arms_raw) != set(ARM_ORDER):
        raise AdapterContractError("asset contract must bind all four arm assets")
    for arm in ARM_ORDER:
        binding = _mapping(arms_raw[arm], f"asset_contract.arms.{arm}")
        _exact_keys(binding, {"asset_sha256"}, f"asset_contract.arms.{arm}")
        expected = _sha256(
            binding["asset_sha256"], f"asset_contract.arms.{arm}.asset_sha256"
        )
        observed = hashlib.sha256(arm_bytes[arm]).hexdigest()
        if expected != FROZEN_ACTUAL_ARM_ASSET_SHA256[arm]:
            raise AdapterContractError(
                f"arm asset {arm} sha256 differs from the frozen actual v5 asset"
            )
        if not hmac.compare_digest(expected, observed):
            raise AdapterContractError(f"arm asset {arm} sha256 evidence mismatch")

    hands_raw = _mapping(asset_contract["hands"], "asset_contract.hands")
    if set(hands_raw) != {"left", "right"}:
        raise AdapterContractError(
            "asset contract must bind left and right RH56F2 hands"
        )
    contracts: dict[str, tuple[_JointContract, ...]] = {}
    for side in ("left", "right"):
        binding = _mapping(hands_raw[side], f"asset_contract.hands.{side}")
        _exact_keys(
            binding,
            {
                "urdf_sha256",
                "inventory_sha256",
                "joint_order",
                "independent_joint_order",
                "joints",
            },
            f"asset_contract.hands.{side}",
        )
        expected_urdf = _sha256(binding["urdf_sha256"], f"{side} URDF sha256")
        expected_inventory = _sha256(
            binding["inventory_sha256"], f"{side} inventory sha256"
        )
        frozen = FROZEN_ACTUAL_HAND_EVIDENCE_SHA256[side]
        if (
            expected_urdf != frozen["urdf_sha256"]
            or expected_inventory != frozen["inventory_sha256"]
        ):
            raise AdapterContractError(
                f"{side} hand evidence sha256 differs from frozen actual RH56F2 assets"
            )
        if not hmac.compare_digest(
            expected_urdf, hashlib.sha256(urdf_bytes[side]).hexdigest()
        ):
            raise AdapterContractError(f"{side} RH56F2 URDF sha256 evidence mismatch")
        if not hmac.compare_digest(
            expected_inventory, hashlib.sha256(inventory_bytes[side]).hexdigest()
        ):
            raise AdapterContractError(
                f"{side} RH56F2 inventory sha256 evidence mismatch"
            )
        urdf_contract = _parse_urdf_contract(urdf_bytes[side], side)
        inventory_contract = _parse_inventory_contract(inventory_bytes[side], side)
        declared_values = _sequence(binding["joints"], f"{side} asset contract joints")
        declared_contract = tuple(
            _joint_from_mapping(item, f"{side} asset contract joint[{index}]")
            for index, item in enumerate(declared_values)
        )
        canonical_order = _JOINT_NAMES_BY_SIDE[side]
        if tuple(joint.name for joint in urdf_contract) != canonical_order:
            raise AdapterContractError(
                f"{side} URDF revolute joint order differs from actual RH56F2"
            )
        if inventory_contract != urdf_contract:
            raise AdapterContractError(
                f"{side} inventory limits/mimic differ from the RH56F2 URDF"
            )
        if declared_contract != urdf_contract:
            raise AdapterContractError(
                f"{side} asset contract limits/mimic differ from the RH56F2 URDF"
            )
        joint_order = tuple(_sequence(binding["joint_order"], f"{side} joint_order"))
        if joint_order != canonical_order:
            raise AdapterContractError(f"{side} asset joint order differs")
        independent = tuple(
            joint.name for joint in urdf_contract if joint.mimic_source is None
        )
        declared_independent = tuple(
            _sequence(binding["independent_joint_order"], f"{side} independent order")
        )
        if declared_independent != independent or len(independent) != 6:
            raise AdapterContractError(
                f"{side} independent joint order must be the six actual RH56F2 DOFs"
            )
        seen: set[str] = set()
        for joint in urdf_contract:
            if joint.mimic_source is not None and joint.mimic_source not in seen:
                raise AdapterContractError(
                    f"{side} URDF mimic source must precede its dependent joint"
                )
            seen.add(joint.name)
        contracts[side] = urdf_contract
    return copy.deepcopy(dict(asset_contract)), contracts


def _validate_device_bindings(value: Any) -> dict[str, Any]:
    bindings = _mapping(value, "calibration.device_bindings")
    _exact_keys(bindings, {"operators", "arms"}, "calibration.device_bindings")
    operators = _mapping(bindings["operators"], "device_bindings.operators")
    if tuple(operators) != OPERATOR_ORDER:
        raise AdapterContractError("device bindings operator order must be F then U")
    operator_keys = {
        "operator_device_id",
        "stream_id",
        "source_id",
        "left_glove_device_id",
        "right_glove_device_id",
        "producer_identity",
        "decoder_identity",
        "pose_source",
        "clock_domain",
    }
    unique_fields = {
        "operator_device_id": [],
        "stream_id": [],
        "source_id": [],
        "glove_device_id": [],
    }
    for operator in OPERATOR_ORDER:
        binding = _mapping(operators[operator], f"device_bindings.operators.{operator}")
        _exact_keys(binding, operator_keys, f"device_bindings.operators.{operator}")
        for key in operator_keys:
            _nonempty_string(binding[key], f"device_bindings.{operator}.{key}")
        for key in ("operator_device_id", "stream_id", "source_id"):
            unique_fields[key].append(binding[key])
        unique_fields["glove_device_id"].extend(
            [binding["left_glove_device_id"], binding["right_glove_device_id"]]
        )
    for label, values in unique_fields.items():
        if len(values) != len(set(values)):
            raise AdapterContractError(f"{label} device bindings must be unique")

    arms = _mapping(bindings["arms"], "device_bindings.arms")
    if set(arms) != set(ARM_ORDER):
        raise AdapterContractError("device bindings must identify all four robot arms")
    robot_ids: list[str] = []
    for arm in ARM_ORDER:
        binding = _mapping(arms[arm], f"device_bindings.arms.{arm}")
        _exact_keys(binding, {"robot_device_id"}, f"device_bindings.arms.{arm}")
        robot_ids.append(
            _nonempty_string(binding["robot_device_id"], f"{arm}.robot_device_id")
        )
    if len(set(robot_ids)) != len(ARM_ORDER):
        raise AdapterContractError("four robot arm device bindings must be unique")
    return copy.deepcopy(dict(bindings))


def _expand_mimic_targets(
    contract: tuple[_JointContract, ...],
    independent_names: tuple[str, ...],
    independent_values: np.ndarray,
    label: str,
) -> np.ndarray:
    independent = dict(zip(independent_names, independent_values, strict=True))
    expanded: dict[str, float] = {}
    for joint in contract:
        if joint.mimic_source is None:
            if joint.name not in independent:
                raise AdapterContractError(
                    f"{label} independent joint binding is missing"
                )
            value = float(independent[joint.name])
        else:
            if joint.mimic_source not in expanded:
                raise AdapterContractError(f"{label} mimic source is unresolved")
            value = (
                expanded[joint.mimic_source] * joint.mimic_multiplier
                + joint.mimic_offset
            )
        if (
            not math.isfinite(value)
            or value < joint.lower - 1e-12
            or value > joint.upper + 1e-12
        ):
            raise AdapterContractError(
                f"{label} target for {joint.name} exceeds RH56F2 URDF joint limit"
            )
        expanded[joint.name] = value
    result = np.asarray([expanded[joint.name] for joint in contract], dtype=np.float64)
    for index, joint in enumerate(contract):
        if joint.mimic_source is not None:
            expected = (
                expanded[joint.mimic_source] * joint.mimic_multiplier
                + joint.mimic_offset
            )
            if not math.isclose(result[index], expected, rel_tol=0.0, abs_tol=1e-12):
                raise AdapterContractError(f"{label} mimic expansion validation failed")
    return result


def _validate_calibration(
    calibration: Mapping[str, Any],
    asset_evidence: CalibrationAssetEvidence,
) -> _Calibration:
    raw = _mapping(calibration, "calibration")
    _reject_rh56dfx(raw)
    _exact_keys(
        raw,
        {
            "schema_version",
            "status",
            "calibration_id",
            "target_hand",
            "asset_contract",
            "device_bindings",
            "source_protocol",
            "ownership",
            "arms",
            "hands",
        },
        "calibration",
    )
    if raw["schema_version"] != CALIBRATION_SCHEMA_VERSION:
        raise AdapterContractError(
            f"calibration schema_version must be {CALIBRATION_SCHEMA_VERSION}"
        )
    if raw["status"] != "calibrated":
        raise AdapterContractError("calibration status must be explicitly calibrated")
    calibration_id = _nonempty_string(raw["calibration_id"], "calibration_id")
    if raw["target_hand"] != "rh56f2":
        raise AdapterContractError("calibration target_hand must be actual rh56f2")
    asset_contract, joint_contracts = _validate_asset_contract(
        raw["asset_contract"], asset_evidence
    )
    device_bindings = _validate_device_bindings(raw["device_bindings"])

    source = _mapping(raw["source_protocol"], "calibration.source_protocol")
    if _jsonable_finite(source) != _EXPECTED_SOURCE_PROTOCOL:
        raise AdapterContractError(
            "calibration source_protocol must exactly bind audited MANU SHM v1"
        )
    ownership = _mapping(raw["ownership"], "calibration.ownership")
    if _jsonable_finite(ownership) != _EXPECTED_OWNERSHIP:
        raise AdapterContractError(
            "calibration ownership must be F=(F_L,F_R), U=(U_L,U_R)"
        )

    arms_raw = _mapping(raw["arms"], "calibration.arms")
    if set(arms_raw) != set(ARM_ORDER):
        raise AdapterContractError(
            "calibration arms must be exactly the four actual arms"
        )
    arms: dict[str, _ArmCalibration] = {}
    arm_keys = {
        "side",
        "twist_convention",
        "twist_to_delta",
        "max_abs_delta",
        "max_translation_step_m",
        "max_rotation_step_rad",
    }
    for arm in ARM_ORDER:
        arm_raw = _mapping(arms_raw[arm], f"calibration.arms.{arm}")
        _exact_keys(arm_raw, arm_keys, f"calibration.arms.{arm}")
        expected_side = _SIDE_BY_ARM[arm]
        if arm_raw["side"] != expected_side:
            raise AdapterContractError(
                f"calibration arm {arm} side must be {expected_side}"
            )
        if arm_raw["twist_convention"] != "world_spatial_v1":
            raise AdapterContractError(
                f"calibration arm {arm} twist convention must be world_spatial_v1"
            )
        dimension = ARM_DIMS[arm]
        matrix = _finite_array(
            arm_raw["twist_to_delta"],
            (dimension, 6),
            f"calibration.arms.{arm}.twist_to_delta",
        )
        if np.linalg.matrix_rank(matrix) != 6:
            raise AdapterContractError(
                f"calibration arm {arm} twist_to_delta must have rank 6"
            )
        max_abs_delta = _finite_array(
            arm_raw["max_abs_delta"],
            (dimension,),
            f"calibration.arms.{arm}.max_abs_delta",
        )
        if np.any(max_abs_delta <= 0.0):
            raise AdapterContractError(
                f"calibration arm {arm} max_abs_delta must be positive"
            )
        arms[arm] = _ArmCalibration(
            side=expected_side,
            twist_to_delta=matrix,
            max_abs_delta=max_abs_delta,
            max_translation_step_m=_positive_float(
                arm_raw["max_translation_step_m"],
                f"calibration.arms.{arm}.max_translation_step_m",
            ),
            max_rotation_step_rad=_positive_float(
                arm_raw["max_rotation_step_rad"],
                f"calibration.arms.{arm}.max_rotation_step_rad",
            ),
        )

    hands_raw = _mapping(raw["hands"], "calibration.hands")
    if set(hands_raw) != {"left", "right"}:
        raise AdapterContractError("calibration hands must be exactly left and right")
    hands: dict[str, _HandCalibration] = {}
    hand_keys = {
        "joint_names",
        "independent_joint_names",
        "centered_position_mean",
        "centered_position_max_abs_deviation",
        "centered_position_to_independent_q",
        "independent_q_bias",
    }
    for side in ("left", "right"):
        hand_raw = _mapping(hands_raw[side], f"calibration.hands.{side}")
        _exact_keys(hand_raw, hand_keys, f"calibration.hands.{side}")
        joint_names = tuple(
            _sequence(hand_raw["joint_names"], f"hands.{side}.joint_names")
        )
        if joint_names != _JOINT_NAMES_BY_SIDE[side]:
            raise AdapterContractError(
                f"calibration {side} joint name/order must exactly match actual RH56F2"
            )
        independent_names = tuple(
            _sequence(
                hand_raw["independent_joint_names"],
                f"hands.{side}.independent_joint_names",
            )
        )
        asset_independent = tuple(
            joint.name for joint in joint_contracts[side] if joint.mimic_source is None
        )
        if independent_names != asset_independent:
            raise AdapterContractError(
                f"calibration {side} independent joint order differs from URDF"
            )
        mean = _finite_array(
            hand_raw["centered_position_mean"],
            (75,),
            f"calibration.hands.{side}.centered_position_mean",
        )
        deviation = _finite_array(
            hand_raw["centered_position_max_abs_deviation"],
            (75,),
            f"calibration.hands.{side}.centered_position_max_abs_deviation",
        )
        if np.any(deviation <= 0.0):
            raise AdapterContractError(
                f"calibration {side} feature deviations must be positive"
            )
        matrix = _finite_array(
            hand_raw["centered_position_to_independent_q"],
            (6, 75),
            f"calibration.hands.{side}.centered_position_to_independent_q",
        )
        if np.linalg.matrix_rank(matrix) != 6:
            raise AdapterContractError(
                f"calibration {side} centered_position_to_independent_q must have rank 6"
            )
        bias = _finite_array(
            hand_raw["independent_q_bias"], (6,), f"hands.{side}.independent_q_bias"
        )
        _expand_mimic_targets(
            joint_contracts[side], independent_names, bias, f"calibration {side} bias"
        )
        hands[side] = _HandCalibration(
            joint_names=joint_names,
            independent_joint_names=independent_names,
            feature_mean=mean,
            feature_max_abs_deviation=deviation,
            feature_to_independent_q=matrix,
            independent_q_bias=bias,
            joint_contract=joint_contracts[side],
        )
    return _Calibration(
        calibration_id=calibration_id,
        arms=arms,
        hands=hands,
        asset_contract=asset_contract,
        device_bindings=device_bindings,
    )


def _unit_quaternion(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = _finite_array(value, shape, label)
    norms = np.linalg.norm(array, axis=-1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-4):
        raise AdapterContractError(f"{label} must contain unit quaternions")
    return array / np.expand_dims(norms, axis=-1)


def issue_wrist_pose_receipt(
    *,
    operator: str,
    side: str,
    pose_position_m: Any,
    pose_quaternion_xyzw: Any,
    pose_timestamp_us: int,
    receipt_timestamp_us: int,
    raw_pose_payload: bytes,
    pose_source: str,
    clock_domain: str,
    capture_class: str,
) -> WristPoseReceipt:
    """Issue a typed wrist receipt while its live wire protocol is unverified.

    The raw payload is content-bound, but this repository has no audited wrist
    producer/decoder protocol.  Consequently every receipt remains formal-live
    ineligible even when it represents a real-time source.
    """
    if operator not in OPERATOR_ORDER:
        raise AdapterContractError("wrist receipt operator must be F or U")
    if side not in {"left", "right"}:
        raise AdapterContractError("wrist receipt side must be left or right")
    position = _finite_array(pose_position_m, (3,), "wrist pose_position_m")
    quaternion = _unit_quaternion(
        pose_quaternion_xyzw, (4,), "wrist pose_quaternion_xyzw"
    )
    pose_timestamp = _nonnegative_int(pose_timestamp_us, "wrist pose_timestamp_us")
    receipt_timestamp = _nonnegative_int(
        receipt_timestamp_us, "wrist receipt_timestamp_us"
    )
    if receipt_timestamp < pose_timestamp:
        raise AdapterContractError("wrist receipt timestamp regresses")
    if (
        not isinstance(raw_pose_payload, (bytes, bytearray, memoryview))
        or not raw_pose_payload
    ):
        raise AdapterContractError("wrist raw_pose_payload must be non-empty bytes")
    if capture_class not in {"synthetic_test", "live", "formal_live"}:
        raise AdapterContractError("wrist capture_class is unsupported")
    pose_source = _nonempty_string(pose_source, "wrist pose_source")
    clock_domain = _nonempty_string(clock_domain, "wrist clock_domain")
    raw_sha = hashlib.sha256(bytes(raw_pose_payload)).hexdigest()
    blockers = ["wrist_pose_protocol_unverified"]
    if capture_class == "synthetic_test":
        blockers.insert(0, "synthetic_input")
    provenance = {
        "schema_version": "d032.wrist_pose.receipt.v1",
        "operator": operator,
        "side": side,
        "pose_position_m": position.tolist(),
        "pose_quaternion_xyzw": quaternion.tolist(),
        "pose_timestamp_us": pose_timestamp,
        "receipt_timestamp_us": receipt_timestamp,
        "raw_pose_sha256": raw_sha,
        "pose_source": pose_source,
        "clock_domain": clock_domain,
        "capture_class": capture_class,
        "formal_live_eligible": False,
        "external_blockers": blockers,
    }
    token = object()
    receipt = WristPoseReceipt(
        operator=operator,
        side=side,
        pose_position_m=position.copy(),
        pose_quaternion_xyzw=quaternion.copy(),
        pose_timestamp_us=pose_timestamp,
        receipt_timestamp_us=receipt_timestamp,
        raw_pose_sha256=raw_sha,
        pose_source=pose_source,
        clock_domain=clock_domain,
        capture_class=capture_class,
        formal_live_eligible=False,
        external_blockers=tuple(blockers),
        _provenance=copy.deepcopy(provenance),
        _raw_pose_payload=bytes(raw_pose_payload),
        _issuance_token=token,
    )
    _ISSUED_WRIST_POSES[receipt] = token
    return receipt


def _require_issued_wrist_pose(value: Any) -> WristPoseReceipt:
    if (
        not isinstance(value, WristPoseReceipt)
        or _ISSUED_WRIST_POSES.get(value) is not value._issuance_token
    ):
        raise AdapterContractError("wrist pose must carry an issued receipt token")
    return value


def make_manus_operator_sample(
    *,
    decoder_receipt: ManusShmV1Receipt,
    wrist_pose_receipts: Sequence[WristPoseReceipt],
) -> ManusOperatorSample:
    """Assemble a sample only from decoder-issued skeleton and wrist receipts."""
    try:
        decoder_receipt = require_decoder_issued_receipt(decoder_receipt)
    except ManusShmV1ContractError as exc:
        raise AdapterContractError(str(exc)) from exc
    manus = decoder_receipt.provenance
    identity = manus["identity"]
    raw_wrists = _sequence(wrist_pose_receipts, "wrist_pose_receipts")
    if len(raw_wrists) != 2:
        raise AdapterContractError(
            "operator sample requires right and left wrist receipts"
        )
    wrists: dict[str, WristPoseReceipt] = {}
    for expected_side, receipt in zip(MANUS_WIRE_HAND_ORDER, raw_wrists, strict=True):
        receipt = _require_issued_wrist_pose(receipt)
        if receipt.operator != manus["operator"] or receipt.side != expected_side:
            raise AdapterContractError("wrist receipt operator/side order differs")
        if (
            receipt.pose_source != identity["pose_source"]
            or receipt.clock_domain != identity["clock_domain"]
        ):
            raise AdapterContractError("wrist receipt identity/clock binding differs")
        if receipt.capture_class != manus["capture_class"]:
            raise AdapterContractError("wrist and MANU capture classes differ")
        wrists[expected_side] = receipt

    hands: list[ManusHandSample] = []
    for decoded_hand in decoder_receipt.hands:
        side = decoded_hand.side
        wrist = wrists[side]
        hands.append(
            ManusHandSample(
                side=side,
                valid=True,
                node_ids=MANUS_NODE_IDS,
                positions_m=decoded_hand.positions_m,
                quaternions_xyzw=decoded_hand.quaternions_xyzw,
                pose_position_m=np.asarray(
                    wrist.pose_position_m, dtype=np.float64
                ).copy(),
                pose_quaternion_xyzw=np.asarray(
                    wrist.pose_quaternion_xyzw, dtype=np.float64
                ).copy(),
                pose_timestamp_us=wrist.pose_timestamp_us,
                glove_device_id=identity[f"{side}_glove_device_id"],
            )
        )
    token = object()
    sample = ManusOperatorSample(
        operator=manus["operator"],
        seq=manus["seq"],
        timestamp_us=manus["timestamp_us"],
        hands=tuple(hands),
        stream_id=identity["stream_id"],
        operator_device_id=identity["operator_device_id"],
        source_id=identity["source_id"],
        producer_identity=identity["producer_identity"],
        decoder_identity=identity["decoder_identity"],
        pose_source=identity["pose_source"],
        clock_domain=identity["clock_domain"],
        raw_frame_sha256=manus["raw_frame_sha256"],
        formal_live_eligible=(
            manus["formal_live_eligible"]
            and all(wrist.formal_live_eligible for wrist in wrists.values())
        ),
        decoder_receipt=decoder_receipt,
        wrist_pose_receipts=tuple(wrists[side] for side in MANUS_WIRE_HAND_ORDER),
        _issuance_token=token,
    )
    _ISSUED_OPERATOR_SAMPLES[sample] = token
    return sample


def _validate_hand(
    value: Any,
    expected_side: str,
    *,
    expected_glove_device_id: str,
    frame_timestamp_us: int,
    received_timestamp_us: int,
    max_latency_us: int,
    max_pose_skew_us: int,
    label: str,
) -> _ValidatedHand:
    if not isinstance(value, ManusHandSample):
        raise AdapterContractError(f"{label} must be a ManusHandSample")
    if value.side != expected_side:
        raise AdapterContractError(
            f"{label} hand slot order/side must be exactly {MANUS_WIRE_HAND_ORDER}"
        )
    if value.glove_device_id != expected_glove_device_id:
        raise AdapterContractError(
            f"{label} glove_device_id differs from the calibrated device binding"
        )
    if not isinstance(value.valid, (bool, np.bool_)) or not bool(value.valid):
        raise AdapterContractError(f"{label} must be explicitly valid")
    node_ids = _sequence(value.node_ids, f"{label}.node_ids")
    if any(
        isinstance(node, bool) or not isinstance(node, (int, np.integer))
        for node in node_ids
    ):
        raise AdapterContractError(f"{label}.node_ids must be integers")
    if tuple(int(node) for node in node_ids) != MANUS_NODE_IDS:
        raise AdapterContractError(
            f"{label}.node_ids must contain exactly known Manus nodes 0..24 in order"
        )
    positions = _finite_array(value.positions_m, (25, 3), f"{label}.positions_m")
    quaternions = _unit_quaternion(
        value.quaternions_xyzw,
        (25, 4),
        f"{label}.quaternions_xyzw",
    )
    pose_position = _finite_array(
        value.pose_position_m, (3,), f"{label}.pose_position_m"
    )
    pose_quaternion = _unit_quaternion(
        value.pose_quaternion_xyzw,
        (4,),
        f"{label}.pose_quaternion_xyzw",
    )
    pose_timestamp_us = _nonnegative_int(
        value.pose_timestamp_us, f"{label}.pose_timestamp_us"
    )
    pose_skew = abs(pose_timestamp_us - frame_timestamp_us)
    if pose_skew > max_pose_skew_us:
        raise AdapterContractError(
            f"{label} pose/frame skew {pose_skew}us exceeds {max_pose_skew_us}us"
        )
    pose_age = received_timestamp_us - pose_timestamp_us
    if pose_age < 0:
        raise AdapterContractError(f"{label} pose timestamp is in the future")
    if pose_age > max_latency_us:
        raise AdapterContractError(
            f"{label} pose is stale by {pose_age}us (limit {max_latency_us}us)"
        )
    return _ValidatedHand(
        side=expected_side,
        glove_device_id=expected_glove_device_id,
        positions_m=positions,
        quaternions_xyzw=quaternions,
        pose_position_m=pose_position,
        pose_quaternion_xyzw=pose_quaternion,
        pose_timestamp_us=pose_timestamp_us,
    )


def _validate_operator(
    value: Any,
    expected_operator: str,
    *,
    received_timestamp_us: int,
    max_latency_us: int,
    max_pose_skew_us: int,
    device_binding: Mapping[str, Any],
) -> _ValidatedOperator:
    if not isinstance(value, ManusOperatorSample):
        raise AdapterContractError(
            f"operator {expected_operator} input must be a ManusOperatorSample"
        )
    if value.operator != expected_operator:
        raise AdapterContractError(
            "operator pair order/identity must be exactly F then U with no duplicates"
        )
    for field in (
        "stream_id",
        "operator_device_id",
        "source_id",
        "producer_identity",
        "decoder_identity",
        "pose_source",
        "clock_domain",
    ):
        if getattr(value, field) != device_binding[field]:
            raise AdapterContractError(
                f"operator {expected_operator} {field} differs from device binding"
            )
    _sha256(value.raw_frame_sha256, f"operator {expected_operator} raw_frame_sha256")
    if not isinstance(value.formal_live_eligible, bool):
        raise AdapterContractError(
            f"operator {expected_operator} formal_live_eligible must be boolean"
        )
    seq = _nonnegative_int(value.seq, f"operator {expected_operator} seq")
    timestamp_us = _nonnegative_int(
        value.timestamp_us, f"operator {expected_operator} timestamp_us"
    )
    age = received_timestamp_us - timestamp_us
    if age < 0:
        raise AdapterContractError(
            f"operator {expected_operator} timestamp is in the future"
        )
    if age > max_latency_us:
        raise AdapterContractError(
            f"operator {expected_operator} sample is stale by {age}us "
            f"(limit {max_latency_us}us)"
        )
    hands_raw = _sequence(value.hands, f"operator {expected_operator} hands")
    if len(hands_raw) != 2:
        raise AdapterContractError(
            f"operator {expected_operator} must contain exactly two hands"
        )
    hands = {
        side: _validate_hand(
            hand,
            side,
            expected_glove_device_id=device_binding[f"{side}_glove_device_id"],
            frame_timestamp_us=timestamp_us,
            received_timestamp_us=received_timestamp_us,
            max_latency_us=max_latency_us,
            max_pose_skew_us=max_pose_skew_us,
            label=f"operator {expected_operator}.{side}",
        )
        for side, hand in zip(MANUS_WIRE_HAND_ORDER, hands_raw, strict=True)
    }
    if _ISSUED_OPERATOR_SAMPLES.get(value) is not value._issuance_token:
        raise AdapterContractError(
            f"operator {expected_operator} sample is not decoder-issued; receipt token missing"
        )
    try:
        decoder_receipt = require_decoder_issued_receipt(value.decoder_receipt)
    except ManusShmV1ContractError as exc:
        raise AdapterContractError(str(exc)) from exc
    manus = decoder_receipt.provenance
    if (
        manus["operator"] != expected_operator
        or manus["seq"] != seq
        or manus["timestamp_us"] != timestamp_us
        or manus["raw_frame_sha256"] != value.raw_frame_sha256
    ):
        raise AdapterContractError(
            f"operator {expected_operator} self-reported MANU fields differ from decoder receipt"
        )
    identity = manus["identity"]
    for field in (
        "stream_id",
        "operator_device_id",
        "source_id",
        "producer_identity",
        "decoder_identity",
        "pose_source",
        "clock_domain",
    ):
        if getattr(value, field) != identity[field]:
            raise AdapterContractError(
                f"operator {expected_operator} {field} differs from decoder receipt"
            )
    receipt_age = received_timestamp_us - manus["receipt_timestamp_us"]
    if receipt_age < 0:
        raise AdapterContractError(
            f"operator {expected_operator} decoder receipt is in the future"
        )
    if receipt_age > max_latency_us:
        raise AdapterContractError(
            f"operator {expected_operator} decoder receipt is stale by {receipt_age}us"
        )
    decoded_hands = {hand.side: hand for hand in decoder_receipt.hands}
    wrist_values = _sequence(
        value.wrist_pose_receipts,
        f"operator {expected_operator} wrist_pose_receipts",
    )
    if len(wrist_values) != 2:
        raise AdapterContractError(
            f"operator {expected_operator} must bind two wrist pose receipts"
        )
    wrist_provenance: dict[str, Any] = {}
    wrist_raw_payloads: dict[str, bytes] = {}
    for expected_side, raw_hand, wrist_value in zip(
        MANUS_WIRE_HAND_ORDER, hands_raw, wrist_values, strict=True
    ):
        wrist = _require_issued_wrist_pose(wrist_value)
        wrist_trace = wrist.provenance
        if wrist.operator != expected_operator or wrist.side != expected_side:
            raise AdapterContractError(
                f"operator {expected_operator} wrist receipt side/order differs"
            )
        if (
            wrist_trace["pose_source"] != identity["pose_source"]
            or wrist_trace["clock_domain"] != identity["clock_domain"]
        ):
            raise AdapterContractError(
                f"operator {expected_operator} wrist receipt identity differs"
            )
        if not np.array_equal(
            np.asarray(raw_hand.positions_m, dtype=np.float64),
            decoded_hands[expected_side].positions_m,
        ) or not np.array_equal(
            np.asarray(raw_hand.quaternions_xyzw, dtype=np.float64),
            decoded_hands[expected_side].quaternions_xyzw,
        ):
            raise AdapterContractError(
                f"operator {expected_operator}.{expected_side} skeleton differs from decoder receipt"
            )
        if not np.array_equal(
            np.asarray(raw_hand.pose_position_m, dtype=np.float64),
            np.asarray(wrist_trace["pose_position_m"], dtype=np.float64),
        ) or not np.array_equal(
            np.asarray(raw_hand.pose_quaternion_xyzw, dtype=np.float64),
            np.asarray(wrist_trace["pose_quaternion_xyzw"], dtype=np.float64),
        ):
            raise AdapterContractError(
                f"operator {expected_operator}.{expected_side} pose differs from wrist receipt"
            )
        if raw_hand.pose_timestamp_us != wrist_trace["pose_timestamp_us"]:
            raise AdapterContractError(
                f"operator {expected_operator}.{expected_side} pose timestamp differs from receipt"
            )
        wrist_receipt_age = received_timestamp_us - wrist_trace["receipt_timestamp_us"]
        if wrist_receipt_age < 0 or wrist_receipt_age > max_latency_us:
            raise AdapterContractError(
                f"operator {expected_operator}.{expected_side} wrist receipt is stale/future"
            )
        wrist_provenance[expected_side] = wrist_trace
        wrist_payload = wrist.raw_pose_payload
        if hashlib.sha256(wrist_payload).hexdigest() != wrist.raw_pose_sha256:
            raise AdapterContractError(
                f"operator {expected_operator}.{expected_side} wrist raw payload digest differs"
            )
        wrist_raw_payloads[expected_side] = wrist_payload
    raw_frame_payload = decoder_receipt.raw_payload
    if hashlib.sha256(raw_frame_payload).hexdigest() != value.raw_frame_sha256:
        raise AdapterContractError(
            f"operator {expected_operator} MANU raw payload digest differs"
        )
    return _ValidatedOperator(
        operator=expected_operator,
        seq=seq,
        timestamp_us=timestamp_us,
        hands=hands,
        raw_frame_sha256=value.raw_frame_sha256,
        formal_live_eligible=value.formal_live_eligible,
        evidence_trace={
            "adapter_received_timestamp_us": received_timestamp_us,
            "manus": manus,
            "wrist_pose": wrist_provenance,
        },
        raw_frame_payload=raw_frame_payload,
        wrist_raw_payloads=wrist_raw_payloads,
    )


def _quaternion_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.array(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )


def _relative_rotation_vector(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    conjugate = np.array([-previous[0], -previous[1], -previous[2], previous[3]])
    # World/spatial convention: translation is expressed in world coordinates,
    # therefore orientation change must use q_current * q_previous^-1.
    relative = _quaternion_multiply_xyzw(current, conjugate)
    relative /= np.linalg.norm(relative)
    if relative[3] < 0.0:
        relative = -relative
    vector_norm = float(np.linalg.norm(relative[:3]))
    if vector_norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, float(np.clip(relative[3], -1.0, 1.0)))
    return relative[:3] * (angle / vector_norm)


def _adapter_output_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _jsonable_finite(value, "adapter_output_trace"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(b"d032-manus-adapter-output-v2\0" + encoded).hexdigest()


def _initial_output_trace(targets: Sequence[HandTarget]) -> dict[str, Any]:
    return {
        target.arm: {
            "kind": "q",
            "joint_names": list(target.joint_names),
            "q": np.asarray(target.q, dtype=np.float64).tolist(),
            "valid": [True] * len(target.joint_names),
        }
        for target in targets
    }


def _transition_output_trace(
    frames: Sequence[OperatorFrame],
) -> dict[str, Any]:
    return {
        "arm_deltas": {
            delta.arm: np.asarray(delta.delta, dtype=np.float64).tolist()
            for frame in frames
            for delta in frame.arm_deltas
        },
        "hand_targets": {
            target.arm: {
                "kind": "q",
                "joint_names": list(target.joint_names),
                "q": np.asarray(target.q, dtype=np.float64).tolist(),
                "valid": [True] * len(target.joint_names),
            }
            for frame in frames
            for target in frame.hand_targets
        },
    }


def adapter_output_trace_sha256_from_exogenous(exogenous: Mapping[str, Any]) -> str:
    """Recompute the adapter-output digest from a validated core exogenous input."""
    try:
        normalized = validate_exogenous_input(exogenous)
    except Exception as exc:
        raise AdapterContractError(
            f"core exogenous output trace is invalid: {exc}"
        ) from exc
    hands_by_arm = {hand["arm"]: hand for hand in normalized["hands"].values()}
    if set(hands_by_arm) != set(ARM_ORDER):
        raise AdapterContractError(
            "core hands do not bind exactly the four adapter arms"
        )
    for arm in ARM_ORDER:
        if hands_by_arm[arm]["kind"] != "q":
            raise AdapterContractError(
                "formal adapter output requires absolute q hand streams"
            )
        if not np.all(hands_by_arm[arm]["valid"]):
            raise AdapterContractError(
                "formal adapter output requires every hand-valid bit to be true"
            )
    steps = normalized["raw_delta"]["F_L"].shape[0]
    trace = {
        "initial_hand_targets": {
            arm: {
                "kind": hands_by_arm[arm]["kind"],
                "joint_names": list(hands_by_arm[arm]["joint_names"]),
                "q": hands_by_arm[arm]["values"][0].tolist(),
                "valid": hands_by_arm[arm]["valid"][0].tolist(),
            }
            for arm in ARM_ORDER
        },
        "transitions": [
            {
                "arm_deltas": {
                    arm: normalized["raw_delta"][arm][index].tolist()
                    for arm in ARM_ORDER
                },
                "hand_targets": {
                    arm: {
                        "kind": hands_by_arm[arm]["kind"],
                        "joint_names": list(hands_by_arm[arm]["joint_names"]),
                        "q": hands_by_arm[arm]["values"][index + 1].tolist(),
                        "valid": hands_by_arm[arm]["valid"][index + 1].tolist(),
                    }
                    for arm in ARM_ORDER
                },
            }
            for index in range(steps)
        ],
    }
    return _adapter_output_digest(trace)


class ManusGloveAdapter:
    """Map two complete Manus operator streams to recorder-native typed frames."""

    def __init__(
        self,
        *,
        calibration: Mapping[str, Any],
        calibration_sha256: str,
        asset_evidence: CalibrationAssetEvidence,
        max_latency_us: int,
        max_pair_skew_us: int,
        max_pose_skew_us: int,
    ) -> None:
        if not isinstance(calibration_sha256, str) or not _SHA256_RE.fullmatch(
            calibration_sha256
        ):
            raise AdapterContractError(
                "calibration_sha256 must be an explicit lowercase sha256"
            )
        canonical_bytes = _canonical_calibration_bytes(calibration)
        observed_hash = hashlib.sha256(canonical_bytes).hexdigest()
        if not hmac.compare_digest(observed_hash, calibration_sha256):
            raise AdapterContractError(
                f"calibration content hash mismatch: expected {calibration_sha256}, "
                f"observed {observed_hash}"
            )
        calibration_snapshot = json.loads(canonical_bytes.decode("utf-8"))
        self._calibration = _validate_calibration(calibration_snapshot, asset_evidence)
        self._calibration_sha256 = observed_hash
        self._max_latency_us = _runtime_limit(
            max_latency_us, "max_latency_us", allow_zero=False
        )
        self._max_pair_skew_us = _runtime_limit(
            max_pair_skew_us, "max_pair_skew_us", allow_zero=True
        )
        self._max_pose_skew_us = _runtime_limit(
            max_pose_skew_us, "max_pose_skew_us", allow_zero=True
        )
        self._state = _AdapterTimeline(
            primed=False,
            last={},
            accepted_transitions=0,
            generation=0,
            baseline_receipts=None,
            transition_receipts=(),
            initial_output=None,
            transition_outputs=(),
            diagnostic_direct_commits=0,
            bound_recorder=None,
            baseline_raw_evidence=None,
            transition_raw_evidence=(),
        )
        self._lock = threading.RLock()
        self._prepared: weakref.WeakKeyDictionary[
            PreparedManusPair, _PreparedRecord
        ] = weakref.WeakKeyDictionary()

    @property
    def calibration_sha256(self) -> str:
        return self._calibration_sha256

    @property
    def is_primed(self) -> bool:
        with self._lock:
            return self._state.primed

    @property
    def generation(self) -> int:
        """Current commit generation used to fence prepared pairs."""
        with self._lock:
            return self._state.generation

    @property
    def provenance(self) -> dict[str, Any]:
        """Return adapter evidence for a sidecar, never for bundle exogenous data."""
        with self._lock:
            state = self._state
            output_digest = None
            if state.initial_output is not None:
                output_digest = _adapter_output_digest(
                    {
                        "initial_hand_targets": state.initial_output,
                        "transitions": state.transition_outputs,
                    }
                )
            provenance = {
                "schema_version": ADAPTER_PROVENANCE_SCHEMA_VERSION,
                "calibration_id": self._calibration.calibration_id,
                "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
                "calibration_sha256": self._calibration_sha256,
                "asset_contract": self._calibration.asset_contract,
                "device_bindings": self._calibration.device_bindings,
                "source_protocol": _EXPECTED_SOURCE_PROTOCOL,
                "operator_streams_required": list(OPERATOR_ORDER),
                "wire_hand_slot_order": list(MANUS_WIRE_HAND_ORDER),
                "runtime_limits_us": {
                    "max_latency_us": self._max_latency_us,
                    "max_pair_skew_us": self._max_pair_skew_us,
                    "max_pose_skew_us": self._max_pose_skew_us,
                },
                "transaction_mode": (
                    "diagnostic_direct"
                    if state.diagnostic_direct_commits
                    else "two_phase_generation_fenced"
                ),
                "formal_capture_eligible": bool(
                    state.primed
                    and state.accepted_transitions > 0
                    and state.diagnostic_direct_commits == 0
                ),
                "generation": state.generation,
                "capture_receipts": {
                    "baseline": state.baseline_receipts,
                    "transitions": state.transition_receipts,
                },
                "output_trace_sha256": output_digest,
                "accepted_control_transitions": state.accepted_transitions,
                "interpolation_applied": False,
                "synthesized_frames": 0,
                "rh56dfx_fallback": False,
                "formal_live_eligible": False,
                "external_blockers": [
                    "producer_binary_evidence_missing",
                    "producer_source_sha256_missing",
                    "producer_seqlock_proof_missing",
                    "wrist_pose_protocol_unverified",
                ],
            }
            return copy.deepcopy(provenance)

    def reset(self) -> None:
        """Forget transport chronology and require a new synchronized prime pair."""
        with self._lock:
            next_prepared = weakref.WeakKeyDictionary()
            next_state = _AdapterTimeline(
                primed=False,
                last={},
                accepted_transitions=0,
                generation=self._state.generation + 1,
                baseline_receipts=None,
                transition_receipts=(),
                initial_output=None,
                transition_outputs=(),
                diagnostic_direct_commits=0,
                bound_recorder=None,
                baseline_raw_evidence=None,
                transition_raw_evidence=(),
            )
            self._prepared = next_prepared
            self._state = next_state

    def _capture_material_unlocked(
        self, *, recorder: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return adapter-issued material only for its committed recorder peer."""
        state = self._state
        if (
            state.bound_recorder is not recorder
            or not state.primed
            or state.accepted_transitions < 1
            or state.diagnostic_direct_commits != 0
            or state.baseline_raw_evidence is None
            or len(state.transition_raw_evidence) != state.accepted_transitions
        ):
            raise AdapterContractError(
                "adapter has no formal capture jointly bound to this recorder"
            )
        return (
            self.provenance,
            {
                "baseline": copy.deepcopy(state.baseline_raw_evidence),
                "transitions": copy.deepcopy(state.transition_raw_evidence),
            },
        )

    def _validated_pair(
        self,
        samples: Sequence[ManusOperatorSample],
        *,
        received_timestamp_us: int,
        require_monotonic: bool,
    ) -> dict[str, _ValidatedOperator]:
        receipt = _nonnegative_int(received_timestamp_us, "received_timestamp_us")
        pair_raw = _sequence(samples, "operator samples")
        if len(pair_raw) != 2:
            raise AdapterContractError(
                "each adapter input must contain exactly operator F and operator U"
            )
        pair = {
            operator: _validate_operator(
                sample,
                operator,
                received_timestamp_us=receipt,
                max_latency_us=self._max_latency_us,
                max_pose_skew_us=self._max_pose_skew_us,
                device_binding=self._calibration.device_bindings["operators"][operator],
            )
            for operator, sample in zip(OPERATOR_ORDER, pair_raw, strict=True)
        }
        pair_skew = abs(pair["F"].timestamp_us - pair["U"].timestamp_us)
        if pair_skew > self._max_pair_skew_us:
            raise AdapterContractError(
                f"operator pair clock skew {pair_skew}us exceeds "
                f"{self._max_pair_skew_us}us"
            )
        if require_monotonic:
            for operator in OPERATOR_ORDER:
                current = pair[operator]
                previous = self._state.last[operator]
                if current.seq <= previous.seq:
                    raise AdapterContractError(
                        f"operator {operator} seq must be strictly monotonic: "
                        f"{current.seq} <= {previous.seq}"
                    )
                if current.timestamp_us <= previous.timestamp_us:
                    raise AdapterContractError(
                        f"operator {operator} timestamp must be strictly monotonic: "
                        f"{current.timestamp_us} <= {previous.timestamp_us}"
                    )
        return pair

    def _hand_q(self, hand: _ValidatedHand) -> np.ndarray:
        calibration = self._calibration.hands[hand.side]
        feature = (hand.positions_m - hand.positions_m[0]).reshape(75)
        centered = feature - calibration.feature_mean
        if np.any(np.abs(centered) > calibration.feature_max_abs_deviation + 1e-12):
            raise AdapterContractError(
                f"{hand.side} skeleton lies outside the calibrated domain"
            )
        independent_q = (
            calibration.feature_to_independent_q @ centered
            + calibration.independent_q_bias
        )
        if not np.isfinite(independent_q).all():
            raise AdapterContractError(
                f"{hand.side} calibrated hand target is non-finite"
            )
        return _expand_mimic_targets(
            calibration.joint_contract,
            calibration.independent_joint_names,
            independent_q,
            f"{hand.side} calibrated hand target",
        )

    def _arm_delta(
        self,
        arm: str,
        previous: _ValidatedHand,
        current: _ValidatedHand,
    ) -> np.ndarray:
        calibration = self._calibration.arms[arm]
        translation = current.pose_position_m - previous.pose_position_m
        rotation = _relative_rotation_vector(
            previous.pose_quaternion_xyzw,
            current.pose_quaternion_xyzw,
        )
        translation_norm = float(np.linalg.norm(translation))
        rotation_norm = float(np.linalg.norm(rotation))
        if translation_norm > calibration.max_translation_step_m + 1e-12:
            raise AdapterContractError(
                f"arm {arm} wrist translation step {translation_norm:.9g}m exceeds "
                f"{calibration.max_translation_step_m:.9g}m"
            )
        if rotation_norm > calibration.max_rotation_step_rad + 1e-12:
            raise AdapterContractError(
                f"arm {arm} wrist rotation step {rotation_norm:.9g}rad exceeds "
                f"{calibration.max_rotation_step_rad:.9g}rad"
            )
        twist = np.concatenate((translation, rotation))
        delta = calibration.twist_to_delta @ twist
        if not np.isfinite(delta).all():
            raise AdapterContractError(f"arm {arm} delta is non-finite")
        if np.any(np.abs(delta) > calibration.max_abs_delta + 1e-12):
            raise AdapterContractError(
                f"arm delta for {arm} exceeds calibrated per-joint bounds"
            )
        return np.asarray(delta, dtype=np.float64).copy()

    def prime_pair(
        self,
        samples: Sequence[ManusOperatorSample],
        *,
        received_timestamp_us: int,
    ) -> tuple[HandTarget, ...]:
        """Validate the synchronized baseline and return four initial hand targets."""
        with self._lock:
            if self._state.primed:
                raise AdapterContractError(
                    "adapter is already primed; reset before re-priming"
                )
            pair = self._validated_pair(
                samples,
                received_timestamp_us=received_timestamp_us,
                require_monotonic=False,
            )
            hand_q = {
                (operator, side): self._hand_q(pair[operator].hands[side])
                for operator in OPERATOR_ORDER
                for side in ("left", "right")
            }
            targets = tuple(
                HandTarget(
                    arm=arm,
                    joint_names=_JOINT_NAMES_BY_SIDE[_SIDE_BY_ARM[arm]],
                    q=hand_q[(arm[0], _SIDE_BY_ARM[arm])].copy(),
                )
                for arm in ARM_ORDER
            )
            baseline_receipts = {
                operator: copy.deepcopy(pair[operator].evidence_trace)
                for operator in OPERATOR_ORDER
            }
            initial_output = _initial_output_trace(targets)
            baseline_raw_evidence = _raw_evidence_from_pair(pair)
            next_prepared = weakref.WeakKeyDictionary()
            next_state = _AdapterTimeline(
                primed=True,
                last=pair,
                accepted_transitions=0,
                generation=self._state.generation + 1,
                baseline_receipts=baseline_receipts,
                transition_receipts=(),
                initial_output=initial_output,
                transition_outputs=(),
                diagnostic_direct_commits=0,
                bound_recorder=None,
                baseline_raw_evidence=baseline_raw_evidence,
                transition_raw_evidence=(),
            )
            self._prepared = next_prepared
            self._state = next_state
            return targets

    def _frames_from_pair(
        self, pair: Mapping[str, _ValidatedOperator]
    ) -> tuple[OperatorFrame, OperatorFrame]:
        frames: list[OperatorFrame] = []
        for operator in OPERATOR_ORDER:
            arm_deltas: list[ArmDelta] = []
            hand_targets: list[HandTarget] = []
            for arm in OPERATOR_ARM_ORDER[operator]:
                side = _SIDE_BY_ARM[arm]
                current_hand = pair[operator].hands[side]
                previous_hand = self._state.last[operator].hands[side]
                arm_deltas.append(
                    ArmDelta(
                        arm=arm,
                        delta=self._arm_delta(arm, previous_hand, current_hand),
                    )
                )
                hand_targets.append(
                    HandTarget(
                        arm=arm,
                        joint_names=_JOINT_NAMES_BY_SIDE[side],
                        q=self._hand_q(current_hand),
                    )
                )
            frames.append(
                OperatorFrame(
                    operator=operator,
                    seq=pair[operator].seq,
                    timestamp_s=pair[operator].timestamp_us / 1_000_000.0,
                    arm_deltas=tuple(arm_deltas),
                    hand_targets=tuple(hand_targets),
                )
            )
        return frames[0], frames[1]

    def prepare_pair(
        self,
        samples: Sequence[ManusOperatorSample],
        *,
        received_timestamp_us: int,
    ) -> PreparedManusPair:
        """Validate and adapt a pair without advancing committed chronology."""
        with self._lock:
            if not self._state.primed:
                raise AdapterContractError(
                    "adapter must prime a synchronized pair first"
                )
            pair = self._validated_pair(
                samples,
                received_timestamp_us=received_timestamp_us,
                require_monotonic=True,
            )
            frames = self._frames_from_pair(pair)
            token = object()
            prepared = PreparedManusPair(
                generation=self._state.generation,
                _issuance_token=token,
            )
            self._prepared[prepared] = _PreparedRecord(
                generation=self._state.generation,
                token=token,
                pair=pair,
                frames=frames,
                output_trace=_transition_output_trace(frames),
            )
            return prepared

    def _prepared_record(self, prepared: PreparedManusPair) -> _PreparedRecord:
        if not isinstance(prepared, PreparedManusPair):
            raise AdapterContractError("adapter transaction requires a prepared token")
        record = self._prepared.get(prepared)
        if (
            record is None
            or record.token is not prepared._issuance_token
            or record.generation != self._state.generation
            or prepared.generation != self._state.generation
        ):
            raise AdapterContractError(
                "prepared adapter token is stale or has the wrong generation"
            )
        return record

    def _stage_committed_state(
        self,
        record: _PreparedRecord,
        *,
        diagnostic_direct: bool,
        bound_recorder: Any | None,
    ) -> _AdapterTimeline:
        """Allocate every next-state value before either timeline is written."""
        current = self._state
        if (
            current.bound_recorder is not None
            and bound_recorder is not current.bound_recorder
        ):
            raise AdapterContractError(
                "adapter is bound to one recorder transaction coordinator"
            )
        _staging_fault_point("before_evidence_copy")
        evidence = {
            operator: copy.deepcopy(record.pair[operator].evidence_trace)
            for operator in OPERATOR_ORDER
        }
        _staging_fault_point("after_evidence_copy")
        output = copy.deepcopy(record.output_trace)
        _staging_fault_point("after_output_copy")
        raw_evidence = _raw_evidence_from_pair(record.pair)
        _staging_fault_point("after_raw_evidence_build")
        next_state = _AdapterTimeline(
            primed=True,
            last=record.pair,
            accepted_transitions=current.accepted_transitions + 1,
            generation=current.generation + 1,
            baseline_receipts=current.baseline_receipts,
            transition_receipts=(*current.transition_receipts, evidence),
            initial_output=current.initial_output,
            transition_outputs=(*current.transition_outputs, output),
            diagnostic_direct_commits=(
                current.diagnostic_direct_commits + int(diagnostic_direct)
            ),
            bound_recorder=(
                current.bound_recorder
                if current.bound_recorder is not None
                else bound_recorder
            ),
            baseline_raw_evidence=current.baseline_raw_evidence,
            transition_raw_evidence=(
                *current.transition_raw_evidence,
                raw_evidence,
            ),
        )
        _staging_fault_point("after_state_build")
        return next_state

    def _commit_prepared_to_recorder(
        self,
        prepared: PreparedManusPair,
        *,
        recorder: Any,
        post_q_ref: Mapping[str, Any],
    ) -> None:
        """Stage both timelines, then install them while both locks are held."""
        with self._lock:
            record = self._prepared_record(prepared)
            next_adapter_state = self._stage_committed_state(
                record, diagnostic_direct=False, bound_recorder=recorder
            )
            with recorder._lock:
                try:
                    recorder_transition = recorder._prepare_transition_unlocked(
                        record.frames,
                        post_q_ref=post_q_ref,
                        bound_adapter=self,
                    )
                except RecorderContractError as exc:
                    raise AdapterContractError(
                        f"recorder transaction rejected without adapter commit: {exc}"
                    ) from exc
                _staging_fault_point("after_recorder_prepare")

                # Unified commit section: both locks are held and every
                # allocation/deepcopy/validation has completed.  Only pointer
                # stores occur from here until both timelines are installed.
                recorder._install_prepared_transition_unlocked(recorder_transition)
                self._state = next_adapter_state

    def adapt_pair(
        self,
        samples: Sequence[ManusOperatorSample],
        *,
        received_timestamp_us: int,
    ) -> tuple[OperatorFrame, OperatorFrame]:
        """Diagnostic direct commit; never eligible for a formal capture sidecar."""
        prepared = self.prepare_pair(
            samples, received_timestamp_us=received_timestamp_us
        )
        with self._lock:
            record = self._prepared_record(prepared)
            next_state = self._stage_committed_state(
                record, diagnostic_direct=True, bound_recorder=None
            )
            self._state = next_state
            return record.frames


__all__ = [
    "ADAPTER_PROVENANCE_SCHEMA_VERSION",
    "CALIBRATION_SCHEMA_VERSION",
    "FROZEN_ACTUAL_ARM_ASSET_SHA256",
    "FROZEN_ACTUAL_HAND_EVIDENCE_SHA256",
    "MANUS_NODE_IDS",
    "MANUS_WIRE_HAND_ORDER",
    "AdapterContractError",
    "CalibrationAssetEvidence",
    "ManusGloveAdapter",
    "ManusHandSample",
    "ManusOperatorSample",
    "PreparedManusPair",
    "WristPoseReceipt",
    "adapter_output_trace_sha256_from_exogenous",
    "calibration_content_sha256",
    "issue_wrist_pose_receipt",
    "make_manus_operator_sample",
]
