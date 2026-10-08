"""Content-bound outer wrapper for a formal glove-capture evidence set.

The planner-owned episode bundle is treated as an opaque, immutable file set.
This module copies that set and binds it to recorder, adapter, protocol-receipt,
and cadence sidecars without changing the core bundle's schema.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import json
import math
import os
import re
import shutil
import tempfile
import weakref
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from safeduo.eval.manus_glove_adapter import (
    AdapterContractError,
    FROZEN_ACTUAL_ARM_ASSET_SHA256,
    FROZEN_ACTUAL_HAND_EVIDENCE_SHA256,
    ManusGloveAdapter,
    adapter_output_trace_sha256_from_exogenous,
)
from safeduo.eval.glove_episode_recorder import (
    RH56F2_LEFT_JOINT_NAMES,
    RH56F2_RIGHT_JOINT_NAMES,
    GloveEpisodeRecorder,
    RecorderContractError,
    recorder_source_trace_sha256,
)
from safeduo.eval.trajectory_task_bundle import (
    BundleValidationError,
    load_episode_bundle,
)
from safeduo.eval.manus_shm_v1 import (
    ManusShmV1ContractError,
    ManusStreamIdentity,
    capture_stable_snapshot,
    decode_manus_shm_v1,
)


SCHEMA_VERSION = "d032.formal_capture_bundle.v1"
_CAPTURE_SNAPSHOT_SCHEMA = "d032.formal_capture_snapshot.v1"
_RECORDER_SCHEMA = "d032.glove_recorder.v2"
_ADAPTER_SCHEMA = "d032.manus_adapter.v2"
_CALIBRATION_SCHEMA = "d032.manus_rh56f2_calibration.v2"
_RECEIPT_SCHEMA = "d032.manus_shm_v1.receipt.v1"
_WRIST_RECEIPT_SCHEMA = "d032.wrist_pose.receipt.v1"
_CADENCE_SCHEMA = "d032.capture_cadence.v1"
_CAPTURE_CLASSES = {"synthetic_test", "formal_live"}
_OPERATORS = ("F", "U")
_OWNED_ARMS = {"F": ["F_L", "F_R"], "U": ["U_L", "U_R"]}
_ARMS = ("F_L", "F_R", "U_L", "U_R")
_SOURCE_PROTOCOL = {
    "magic": "MANU",
    "version": 1,
    "block_size_bytes": 1448,
    "header_size_bytes": 32,
    "hand_size_bytes": 708,
    "node_size_bytes": 28,
    "max_nodes": 25,
    "node_ids": list(range(25)),
    "side_codes": {"left": 1, "right": 2},
    "hand_count_required": 2,
    "wire_hand_slot_order": ["right", "left"],
    "quaternion_order": "xyzw",
    "timestamp_unit": "microseconds_since_epoch",
    "skeleton_position_frame": "manus_local",
    "wrist_pose_frame": "operator_calibrated_world",
}
_RH56F2_JOINT_VALUES = (
    ("thumb_1_joint", 0.0, 2.0, None),
    ("thumb_2_joint", 0.0, 0.7854, None),
    ("thumb_3_joint", 0.0, 0.567232, ("thumb_2_joint", 0.7222, 0.0)),
    ("thumb_4_joint", 0.0, 0.3957, ("thumb_3_joint", 0.69754, 0.0)),
    ("index_1_joint", 0.0, 1.466, None),
    ("index_2_joint", 0.0, 1.6925, ("index_1_joint", 1.1545, 0.0)),
    ("middle_1_joint", 0.0, 1.466, None),
    ("middle_2_joint", 0.0, 1.6925, ("middle_1_joint", 1.1545, 0.0)),
    ("ring_1_joint", 0.0, 1.466, None),
    ("ring_2_joint", 0.0, 1.6925, ("ring_1_joint", 1.1545, 0.0)),
    ("pinky_1_joint", 0.0, 1.466, None),
    ("pinky_2_joint", 0.0, 1.6925, ("pinky_1_joint", 1.1545, 0.0)),
)
_SIDECAR_FILES = {
    "capture_snapshot": "capture_snapshot.json",
    "recorder": "recorder.json",
    "adapter": "adapter.json",
    "protocol_receipts": "protocol_receipts.json",
    "cadence": "cadence.json",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FormalCaptureBundleError(ValueError):
    """The outer capture wrapper or one of its evidence bindings is invalid."""


_CAPABILITY_ISSUER = object()
_ISSUED_CAPABILITIES: weakref.WeakKeyDictionary[FormalCaptureCapability, Any]


class FormalCaptureCapability:
    """Opaque in-process authority for one jointly sealed recorder/adapter capture."""

    __slots__ = ("__weakref__", "_closure_sha256", "_issuer_token")

    def __init__(self, *, closure_sha256: str, issuer_token: object) -> None:
        if issuer_token is not _CAPABILITY_ISSUER:
            raise FormalCaptureBundleError(
                "formal capture capability must be jointly issued in process"
            )
        self._closure_sha256 = closure_sha256
        self._issuer_token = issuer_token

    @property
    def closure_sha256(self) -> str:
        return self._closure_sha256

    @property
    def snapshot(self) -> dict[str, Any]:
        """Return a non-authoritative copy for inspection and archival tooling."""
        snapshot = _ISSUED_CAPABILITIES.get(self)
        if snapshot is None:
            raise FormalCaptureBundleError("capture capability is not issued")
        return copy.deepcopy(snapshot)

    def __copy__(self) -> FormalCaptureCapability:
        raise TypeError("formal capture capabilities cannot be copied")

    def __deepcopy__(self, _memo: Any) -> FormalCaptureCapability:
        raise TypeError("formal capture capabilities cannot be deep-copied")

    def __reduce_ex__(self, _protocol: int) -> Any:
        raise TypeError("formal capture capabilities cannot be pickled")


_ISSUED_CAPABILITIES = weakref.WeakKeyDictionary()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FormalCaptureBundleError(f"{label} must be a mapping")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise FormalCaptureBundleError(
            f"{label} keys/binding differ: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FormalCaptureBundleError(f"{label} must be an integer")
    if value < minimum:
        raise FormalCaptureBundleError(f"{label} must be >= {minimum}")
    return value


def _finite_float(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FormalCaptureBundleError(f"{label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or (positive and normalized <= 0.0):
        qualifier = "positive and " if positive else ""
        raise FormalCaptureBundleError(f"{label} must be {qualifier}finite")
    return normalized


def _sha256(value: Any, label: str) -> str:
    normalized = str(value)
    if not _SHA256_RE.fullmatch(normalized):
        raise FormalCaptureBundleError(f"{label} must be a lowercase sha256")
    return normalized


def _canonical_bytes(value: Any) -> bytes:
    try:
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
    except (TypeError, ValueError) as exc:
        raise FormalCaptureBundleError(
            f"capture evidence is not strict JSON: {exc}"
        ) from exc


def _json_snapshot(value: Any, label: str) -> Any:
    try:
        return json.loads(_canonical_bytes(value).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:  # defensive; writer made it
        raise FormalCaptureBundleError(f"{label} cannot be snapshotted: {exc}") from exc


def _file_binding(data: bytes) -> dict[str, Any]:
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def _binding_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        b"d032-formal-capture-bundle-v1\0" + _canonical_bytes(value)
    ).hexdigest()


def _capture_closure_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        b"d032-formal-capture-snapshot-v1\0" + _canonical_bytes(value)
    ).hexdigest()


def _encode_raw_payload(payload: bytes, sha256: str) -> dict[str, str]:
    return {
        "payload_base64": base64.b64encode(payload).decode("ascii"),
        "sha256": sha256,
    }


def _decode_raw_payload(value: Any, label: str) -> tuple[bytes, str]:
    encoded = _require_mapping(value, label)
    _require_exact_keys(encoded, {"payload_base64", "sha256"}, label)
    payload_base64 = encoded["payload_base64"]
    if not isinstance(payload_base64, str) or not payload_base64:
        raise FormalCaptureBundleError(f"{label}.payload_base64 is missing")
    try:
        payload = base64.b64decode(payload_base64, validate=True)
    except (ValueError, UnicodeError) as exc:
        raise FormalCaptureBundleError(
            f"{label}.payload_base64 is not canonical base64"
        ) from exc
    if base64.b64encode(payload).decode("ascii") != payload_base64:
        raise FormalCaptureBundleError(f"{label}.payload_base64 is not canonical")
    expected_sha = _sha256(encoded["sha256"], f"{label}.sha256")
    if not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), expected_sha):
        raise FormalCaptureBundleError(f"{label} raw payload sha256 differs")
    return payload, expected_sha


def _expected_hand_asset_contract(side: str) -> dict[str, Any]:
    prefix = f"{side}_"
    joints = [
        {
            "name": prefix + name,
            "lower": lower,
            "upper": upper,
            "mimic": (
                None
                if mimic is None
                else {
                    "source": prefix + mimic[0],
                    "multiplier": mimic[1],
                    "offset": mimic[2],
                }
            ),
        }
        for name, lower, upper, mimic in _RH56F2_JOINT_VALUES
    ]
    joint_names = (
        RH56F2_LEFT_JOINT_NAMES if side == "left" else RH56F2_RIGHT_JOINT_NAMES
    )
    return {
        "joint_order": list(joint_names),
        "independent_joint_order": [
            prefix + name
            for name, _lower, _upper, mimic in _RH56F2_JOINT_VALUES
            if mimic is None
        ],
        "joints": joints,
    }


def _scan_regular_tree(root: Path) -> tuple[dict[str, bytes], set[str]]:
    if root.is_symlink() or not root.is_dir():
        raise FormalCaptureBundleError(
            f"core bundle must be a real directory without a symlink: {root}"
        )
    files: dict[str, bytes] = {}
    directories: set[str] = set()
    pending = [(root, Path())]
    while pending:
        directory, relative_directory = pending.pop()
        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            raise FormalCaptureBundleError(
                f"core bundle cannot be scanned: {exc}"
            ) from exc
        for entry in entries:
            relative = relative_directory / entry.name
            relative_text = relative.as_posix()
            if entry.is_symlink():
                raise FormalCaptureBundleError(
                    f"core bundle symlink is forbidden: {relative_text}"
                )
            if entry.is_dir():
                directories.add(relative_text)
                pending.append((entry, relative))
            elif entry.is_file():
                try:
                    files[relative_text] = entry.read_bytes()
                except OSError as exc:
                    raise FormalCaptureBundleError(
                        f"core file cannot be read: {relative_text}: {exc}"
                    ) from exc
            else:
                raise FormalCaptureBundleError(
                    f"core entry must be a regular file/directory: {relative_text}"
                )
    return files, directories


def _snapshot_core(root: Path) -> tuple[dict[str, bytes], set[str]]:
    first_files, first_directories = _scan_regular_tree(root)
    if "manifest.json" not in first_files or "exogenous.json" not in first_files:
        raise FormalCaptureBundleError(
            "opaque core must contain regular manifest.json and exogenous.json files"
        )
    second_files, second_directories = _scan_regular_tree(root)
    if first_files != second_files or first_directories != second_directories:
        raise FormalCaptureBundleError("opaque core changed during snapshot binding")
    return first_files, first_directories


def _validate_recorder(value: Any) -> dict[str, Any]:
    recorder = _require_mapping(
        _json_snapshot(value, "recorder provenance"), "recorder"
    )
    if recorder.get("schema_version") != _RECORDER_SCHEMA:
        raise FormalCaptureBundleError("recorder provenance schema is not formal v2")
    if (
        recorder.get("capture_mode") != "formal"
        or recorder.get("formal_eligible") is not True
    ):
        raise FormalCaptureBundleError(
            "diagnostic recorder provenance cannot be written as a formal capture"
        )
    transitions = _integer(
        recorder.get("control_transitions"), "recorder.control_transitions", minimum=1
    )
    if recorder.get("interpolation_applied") is not False:
        raise FormalCaptureBundleError("formal recorder interpolation must be false")
    if recorder.get("synthesized_frames") != 0:
        raise FormalCaptureBundleError(
            "formal recorder synthesized_frames must be zero"
        )
    _sha256(recorder.get("source_trace_sha256"), "recorder.source_trace_sha256")

    clock = _require_mapping(recorder.get("replay_clock_policy"), "replay clock policy")
    _require_exact_keys(
        clock,
        {"clock", "replay_dt_s", "cadence_tolerance_s", "sequence_policy"},
        "replay clock policy",
    )
    if clock["clock"] != "capture_timestamp_s":
        raise FormalCaptureBundleError(
            "formal recorder clock must be capture_timestamp_s"
        )
    dt = _finite_float(clock["replay_dt_s"], "replay_dt_s", positive=True)
    tolerance = _finite_float(clock["cadence_tolerance_s"], "cadence_tolerance_s")
    if tolerance < 0.0 or tolerance >= dt:
        raise FormalCaptureBundleError("cadence tolerance must be in [0, replay_dt_s)")
    if clock["sequence_policy"] != "exact_prev_plus_one_per_operator":
        raise FormalCaptureBundleError("formal recorder sequence policy is not exact")

    max_skew = _finite_float(
        recorder.get("max_allowed_clock_skew_s"),
        "recorder.max_allowed_clock_skew_s",
    )
    observed_skew = _finite_float(
        recorder.get("max_observed_clock_skew_s"),
        "recorder.max_observed_clock_skew_s",
    )
    if max_skew < 0.0 or observed_skew < 0.0 or observed_skew > max_skew + 1e-12:
        raise FormalCaptureBundleError("recorder clock-skew bound differs")

    operators = _require_mapping(recorder.get("operators"), "recorder.operators")
    if tuple(operators) != _OPERATORS:
        raise FormalCaptureBundleError(
            "recorder operator order/binding must be F then U"
        )
    for operator in _OPERATORS:
        stream = _require_mapping(operators[operator], f"recorder.operators.{operator}")
        if stream.get("owned_arms") != _OWNED_ARMS[operator]:
            raise FormalCaptureBundleError(f"operator {operator} arm binding differs")
        seq = stream.get("seq")
        timestamps = stream.get("timestamp_s")
        if not isinstance(seq, list) or not isinstance(timestamps, list):
            raise FormalCaptureBundleError(
                f"operator {operator} cadence arrays are missing"
            )
        if len(seq) != transitions or len(timestamps) != transitions:
            raise FormalCaptureBundleError(
                f"operator {operator} cadence length differs"
            )
        normalized_seq = [
            _integer(item, f"operator {operator} seq[{index}]")
            for index, item in enumerate(seq)
        ]
        if any(
            current != previous + 1
            for previous, current in zip(normalized_seq, normalized_seq[1:])
        ):
            raise FormalCaptureBundleError(
                f"operator {operator} receipt sequence has a gap"
            )
        normalized_times = [
            _finite_float(item, f"operator {operator} timestamp[{index}]")
            for index, item in enumerate(timestamps)
        ]
        for previous, current in zip(normalized_times, normalized_times[1:]):
            if (
                current <= previous
                or abs((current - previous) - dt) > tolerance + 1e-12
            ):
                raise FormalCaptureBundleError(f"operator {operator} cadence differs")
    recomputed_skews = [
        abs(float(left) - float(right))
        for left, right in zip(
            operators["F"]["timestamp_s"],
            operators["U"]["timestamp_s"],
            strict=True,
        )
    ]
    if any(skew > max_skew + 1e-12 for skew in recomputed_skews):
        raise FormalCaptureBundleError("recorder operator clock skew exceeds bound")
    if not math.isclose(
        max(recomputed_skews), observed_skew, rel_tol=0.0, abs_tol=1e-12
    ):
        raise FormalCaptureBundleError(
            "recorder max observed clock skew was not recomputed exactly"
        )
    return dict(recorder)


def _validate_adapter(value: Any, transitions: int) -> dict[str, Any]:
    adapter = _require_mapping(_json_snapshot(value, "adapter provenance"), "adapter")
    _require_exact_keys(
        adapter,
        {
            "schema_version",
            "calibration_id",
            "calibration_schema_version",
            "calibration_sha256",
            "asset_contract",
            "device_bindings",
            "source_protocol",
            "operator_streams_required",
            "wire_hand_slot_order",
            "runtime_limits_us",
            "transaction_mode",
            "formal_capture_eligible",
            "generation",
            "capture_receipts",
            "output_trace_sha256",
            "accepted_control_transitions",
            "interpolation_applied",
            "synthesized_frames",
            "rh56dfx_fallback",
            "formal_live_eligible",
            "external_blockers",
        },
        "adapter",
    )
    if adapter.get("schema_version") != _ADAPTER_SCHEMA:
        raise FormalCaptureBundleError("adapter schema/binding is not v2")
    if adapter.get("calibration_schema_version") != _CALIBRATION_SCHEMA:
        raise FormalCaptureBundleError("adapter calibration schema/binding is not v2")
    if (
        not isinstance(adapter.get("calibration_id"), str)
        or not adapter["calibration_id"]
    ):
        raise FormalCaptureBundleError("adapter calibration_id is missing")
    _sha256(adapter.get("calibration_sha256"), "adapter.calibration_sha256")
    asset_contract = _require_mapping(
        adapter.get("asset_contract"), "adapter.asset_contract"
    )
    _require_exact_keys(asset_contract, {"arms", "hands"}, "adapter.asset_contract")
    asset_arms = _require_mapping(asset_contract.get("arms"), "adapter asset arms")
    asset_hands = _require_mapping(asset_contract.get("hands"), "adapter asset hands")
    if set(asset_arms) != set(_ARMS) or set(asset_hands) != {"left", "right"}:
        raise FormalCaptureBundleError("adapter actual asset sha256 binding differs")
    for arm in _ARMS:
        arm_binding = _require_mapping(asset_arms[arm], f"adapter asset arm {arm}")
        _require_exact_keys(arm_binding, {"asset_sha256"}, f"adapter asset arm {arm}")
        observed = _sha256(
            arm_binding.get("asset_sha256"), f"adapter asset arm {arm} sha256"
        )
        if observed != FROZEN_ACTUAL_ARM_ASSET_SHA256[arm]:
            raise FormalCaptureBundleError(
                f"adapter arm {arm} actual asset sha256 binding differs"
            )
    for side in ("left", "right"):
        hand_binding = _require_mapping(asset_hands[side], f"adapter asset hand {side}")
        _require_exact_keys(
            hand_binding,
            {
                "urdf_sha256",
                "inventory_sha256",
                "joint_order",
                "independent_joint_order",
                "joints",
            },
            f"adapter asset hand {side}",
        )
        urdf_sha = _sha256(
            hand_binding.get("urdf_sha256"), f"adapter {side} URDF sha256"
        )
        inventory_sha = _sha256(
            hand_binding.get("inventory_sha256"),
            f"adapter {side} inventory sha256",
        )
        frozen = FROZEN_ACTUAL_HAND_EVIDENCE_SHA256[side]
        if (
            urdf_sha != frozen["urdf_sha256"]
            or inventory_sha != frozen["inventory_sha256"]
        ):
            raise FormalCaptureBundleError(
                f"adapter {side} actual RH56F2 evidence sha256 binding differs"
            )
        expected_hand = _expected_hand_asset_contract(side)
        for field in ("joint_order", "independent_joint_order", "joints"):
            if hand_binding[field] != expected_hand[field]:
                raise FormalCaptureBundleError(
                    f"adapter {side} RH56F2 {field} limits/mimic binding differs"
                )
    if adapter.get("accepted_control_transitions") != transitions:
        raise FormalCaptureBundleError("adapter/recorder transition binding differs")
    if (
        adapter.get("interpolation_applied") is not False
        or adapter.get("synthesized_frames") != 0
    ):
        raise FormalCaptureBundleError(
            "adapter must not interpolate or synthesize frames"
        )
    if adapter.get("rh56dfx_fallback") is not False:
        raise FormalCaptureBundleError("RH56DFX fallback is forbidden")
    limits = _require_mapping(
        adapter.get("runtime_limits_us"), "adapter.runtime_limits_us"
    )
    _require_exact_keys(
        limits,
        {"max_latency_us", "max_pair_skew_us", "max_pose_skew_us"},
        "adapter.runtime_limits_us",
    )
    _integer(limits["max_latency_us"], "adapter.max_latency_us", minimum=1)
    _integer(limits["max_pair_skew_us"], "adapter.max_pair_skew_us")
    _integer(limits["max_pose_skew_us"], "adapter.max_pose_skew_us")
    if adapter.get("transaction_mode") != "two_phase_generation_fenced":
        raise FormalCaptureBundleError(
            "formal capture requires two-phase generation-fenced adapter commits"
        )
    if adapter.get("formal_capture_eligible") is not True:
        raise FormalCaptureBundleError(
            "adapter is not eligible for a recorder-bound formal capture"
        )
    _integer(adapter.get("generation"), "adapter.generation", minimum=1)
    _sha256(adapter.get("output_trace_sha256"), "adapter.output_trace_sha256")
    capture_receipts = _require_mapping(
        adapter.get("capture_receipts"), "adapter.capture_receipts"
    )
    _require_exact_keys(
        capture_receipts, {"baseline", "transitions"}, "adapter.capture_receipts"
    )
    baseline = _require_mapping(
        capture_receipts["baseline"], "adapter.capture_receipts.baseline"
    )
    if tuple(baseline) != _OPERATORS:
        raise FormalCaptureBundleError(
            "adapter baseline receipt order/binding must be F then U"
        )
    transition_receipts = capture_receipts["transitions"]
    if (
        not isinstance(transition_receipts, list)
        or len(transition_receipts) != transitions
    ):
        raise FormalCaptureBundleError(
            "adapter per-transition receipt count differs from recorder"
        )
    for index, pair in enumerate(transition_receipts):
        pair_mapping = _require_mapping(
            pair, f"adapter.capture_receipts.transitions[{index}]"
        )
        if tuple(pair_mapping) != _OPERATORS:
            raise FormalCaptureBundleError(
                f"adapter transition receipt {index} order/binding must be F then U"
            )
    protocol = _require_mapping(
        adapter.get("source_protocol"), "adapter.source_protocol"
    )
    if dict(protocol) != _SOURCE_PROTOCOL:
        raise FormalCaptureBundleError("adapter audited MANU protocol binding differs")
    if adapter["operator_streams_required"] != ["F", "U"]:
        raise FormalCaptureBundleError("adapter operator stream binding differs")
    if adapter["wire_hand_slot_order"] != ["right", "left"]:
        raise FormalCaptureBundleError("adapter MANU hand slot binding differs")
    bindings = _require_mapping(
        adapter.get("device_bindings"), "adapter.device_bindings"
    )
    _require_exact_keys(bindings, {"operators", "arms"}, "adapter.device_bindings")
    operators = _require_mapping(bindings.get("operators"), "adapter operator bindings")
    arms = _require_mapping(bindings.get("arms"), "adapter arm bindings")
    if set(operators) != set(_OPERATORS) or set(arms) != set(_ARMS):
        raise FormalCaptureBundleError("adapter device binding identities differ")
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
    unique_operator_ids: dict[str, list[str]] = {
        "operator_device_id": [],
        "stream_id": [],
        "source_id": [],
        "glove_device_id": [],
    }
    robot_ids: list[str] = []
    for operator in _OPERATORS:
        binding = _require_mapping(operators[operator], f"adapter operator {operator}")
        _require_exact_keys(binding, operator_keys, f"adapter operator {operator}")
        for key in operator_keys:
            if not isinstance(binding[key], str) or not binding[key]:
                raise FormalCaptureBundleError(
                    f"adapter operator {operator} {key} is missing"
                )
        for key in ("operator_device_id", "stream_id", "source_id"):
            unique_operator_ids[key].append(binding[key])
        unique_operator_ids["glove_device_id"].extend(
            [binding["left_glove_device_id"], binding["right_glove_device_id"]]
        )
    for arm in _ARMS:
        binding = _require_mapping(arms[arm], f"adapter arm {arm}")
        _require_exact_keys(binding, {"robot_device_id"}, f"adapter arm {arm}")
        robot_id = binding.get("robot_device_id")
        if not isinstance(robot_id, str) or not robot_id:
            raise FormalCaptureBundleError(f"adapter arm {arm} device is missing")
        robot_ids.append(robot_id)
    if any(len(values) != len(set(values)) for values in unique_operator_ids.values()):
        raise FormalCaptureBundleError(
            "adapter operator/device identities must be unique"
        )
    if len(set(robot_ids)) != 4:
        raise FormalCaptureBundleError("adapter robot device identities must be unique")
    if not isinstance(adapter.get("formal_live_eligible"), bool):
        raise FormalCaptureBundleError(
            "adapter formal_live_eligible binding is missing"
        )
    blockers = adapter.get("external_blockers")
    if not isinstance(blockers, list) or any(
        not isinstance(item, str) or not item for item in blockers
    ):
        raise FormalCaptureBundleError("adapter external blocker binding differs")
    return dict(adapter)


def _validate_receipts(
    value: Any,
    recorder: Mapping[str, Any],
    adapter: Mapping[str, Any],
    capture_class: str,
) -> list[dict[str, Any]]:
    receipts = _json_snapshot(value, "protocol receipts")
    if not isinstance(receipts, list):
        raise FormalCaptureBundleError("protocol receipts must be a list")
    transitions = recorder["control_transitions"]
    expected_count = 2 * (transitions + 1)
    if len(receipts) != expected_count:
        raise FormalCaptureBundleError(
            f"protocol receipt pair count {len(receipts)} != {expected_count}"
        )
    streams = {
        operator: adapter["device_bindings"]["operators"][operator]["stream_id"]
        for operator in _OPERATORS
    }
    clock = recorder["replay_clock_policy"]
    dt_us = float(clock["replay_dt_s"]) * 1_000_000.0
    tolerance_us = float(clock["cadence_tolerance_s"]) * 1_000_000.0 + 1e-6
    operator_seq = {
        operator: recorder["operators"][operator]["seq"] for operator in _OPERATORS
    }
    operator_time_us = {
        operator: [
            timestamp * 1_000_000.0
            for timestamp in recorder["operators"][operator]["timestamp_s"]
        ]
        for operator in _OPERATORS
    }
    seen_raw: set[str] = set()
    producer_builds: dict[str, tuple[str, str | None]] = {}
    decoder_builds: dict[str, str] = {}
    normalized: list[dict[str, Any]] = []
    required = {
        "schema_version",
        "operator",
        "seq",
        "timestamp_us",
        "receipt_timestamp_us",
        "raw_frame_sha256",
        "identity",
        "snapshot",
        "capture_class",
        "formal_live_eligible",
        "external_blockers",
    }
    identity_keys = {
        "stream_id",
        "operator_device_id",
        "source_id",
        "left_glove_device_id",
        "right_glove_device_id",
        "producer_identity",
        "producer_binary_sha256",
        "producer_source_sha256",
        "decoder_identity",
        "decoder_source_sha256",
        "pose_source",
        "clock_domain",
    }
    for pair_index in range(transitions + 1):
        for operator_offset, operator in enumerate(_OPERATORS):
            index = pair_index * 2 + operator_offset
            receipt = _require_mapping(receipts[index], f"receipt[{index}]")
            _require_exact_keys(receipt, required, f"receipt[{index}]")
            if receipt["schema_version"] != _RECEIPT_SCHEMA:
                raise FormalCaptureBundleError(f"receipt[{index}] schema differs")
            if receipt["operator"] != operator:
                raise FormalCaptureBundleError(
                    f"receipt pair {pair_index} operator order differs"
                )
            seq = _integer(receipt["seq"], f"receipt[{index}].seq")
            timestamp_us = _integer(
                receipt["timestamp_us"], f"receipt[{index}].timestamp_us"
            )
            receipt_timestamp = _integer(
                receipt["receipt_timestamp_us"],
                f"receipt[{index}].receipt_timestamp_us",
            )
            if receipt_timestamp < timestamp_us:
                raise FormalCaptureBundleError(
                    f"receipt[{index}] receipt clock regressed"
                )
            raw_sha = _sha256(
                receipt["raw_frame_sha256"], f"receipt[{index}].raw_frame_sha256"
            )
            if raw_sha in seen_raw:
                raise FormalCaptureBundleError("duplicate raw-frame receipt binding")
            seen_raw.add(raw_sha)
            identity = _require_mapping(
                receipt["identity"], f"receipt[{index}].identity"
            )
            _require_exact_keys(identity, identity_keys, f"receipt[{index}].identity")
            for key in identity_keys - {"producer_source_sha256"}:
                if not isinstance(identity[key], str) or not identity[key]:
                    raise FormalCaptureBundleError(
                        f"receipt[{index}] identity.{key} is missing"
                    )
            _sha256(
                identity["producer_binary_sha256"],
                f"receipt[{index}] producer binary sha256",
            )
            if identity["producer_source_sha256"] is not None:
                _sha256(
                    identity["producer_source_sha256"],
                    f"receipt[{index}] producer source sha256",
                )
            _sha256(
                identity["decoder_source_sha256"],
                f"receipt[{index}] decoder source sha256",
            )
            producer_build = (
                identity["producer_binary_sha256"],
                identity["producer_source_sha256"],
            )
            previous_producer = producer_builds.setdefault(
                identity["producer_identity"], producer_build
            )
            if previous_producer != producer_build:
                raise FormalCaptureBundleError(
                    "producer identity changed binary/source hash during capture"
                )
            previous_decoder = decoder_builds.setdefault(
                identity["decoder_identity"], identity["decoder_source_sha256"]
            )
            if previous_decoder != identity["decoder_source_sha256"]:
                raise FormalCaptureBundleError(
                    "decoder identity changed source hash during capture"
                )
            if identity["left_glove_device_id"] == identity["right_glove_device_id"]:
                raise FormalCaptureBundleError(
                    f"receipt[{index}] glove device identities must be distinct"
                )
            if identity["stream_id"] != streams[operator]:
                raise FormalCaptureBundleError(
                    f"receipt[{index}] stream binding differs"
                )
            adapter_identity = adapter["device_bindings"]["operators"][operator]
            for identity_field in (
                "operator_device_id",
                "source_id",
                "left_glove_device_id",
                "right_glove_device_id",
                "producer_identity",
                "decoder_identity",
                "pose_source",
                "clock_domain",
            ):
                if identity[identity_field] != adapter_identity[identity_field]:
                    raise FormalCaptureBundleError(
                        f"receipt[{index}] {identity_field} adapter binding differs"
                    )
            snapshot = _require_mapping(
                receipt["snapshot"], f"receipt[{index}].snapshot"
            )
            read_count = snapshot.get("read_count")
            if (
                snapshot.get("method") != "byte_identical_double_read"
                or isinstance(read_count, bool)
                or not isinstance(read_count, int)
                or read_count < 2
                or read_count % 2 != 0
            ):
                raise FormalCaptureBundleError(
                    f"receipt[{index}] stable snapshot proof differs"
                )
            if receipt["capture_class"] != capture_class:
                qualifier = (
                    "formal live receipt is not eligible; "
                    if capture_class == "formal_live"
                    else ""
                )
                raise FormalCaptureBundleError(
                    f"{qualifier}receipt[{index}] capture class differs"
                )
            blockers = receipt["external_blockers"]
            if not isinstance(blockers, list) or any(
                not isinstance(item, str) or not item for item in blockers
            ):
                raise FormalCaptureBundleError(
                    f"receipt[{index}] blocker evidence differs"
                )

            recorded_index = pair_index - 1
            if pair_index == 0:
                expected_seq = operator_seq[operator][0] - 1
                expected_timestamp = operator_time_us[operator][0] - dt_us
            else:
                expected_seq = operator_seq[operator][recorded_index]
                expected_timestamp = operator_time_us[operator][recorded_index]
            if seq != expected_seq:
                raise FormalCaptureBundleError(
                    f"receipt[{index}] sequence binding differs"
                )
            if abs(timestamp_us - expected_timestamp) > tolerance_us:
                raise FormalCaptureBundleError(
                    f"receipt[{index}] cadence timestamp binding differs"
                )
            normalized.append(dict(receipt))
    return normalized


def _finite_vector(value: Any, size: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise FormalCaptureBundleError(f"{label} must contain exactly {size} values")
    return [
        _finite_float(item, f"{label}[{index}]") for index, item in enumerate(value)
    ]


def _validate_capture_evidence(
    adapter: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    capture_class: str,
) -> None:
    """Revalidate adapter runtime limits against every persisted receipt."""
    limits = adapter["runtime_limits_us"]
    max_latency = int(limits["max_latency_us"])
    max_pair_skew = int(limits["max_pair_skew_us"])
    max_pose_skew = int(limits["max_pose_skew_us"])
    capture_receipts = adapter["capture_receipts"]
    pairs = [capture_receipts["baseline"], *capture_receipts["transitions"]]
    if len(pairs) * 2 != len(receipts):  # defensive; both validators also bind count
        raise FormalCaptureBundleError("adapter/protocol receipt count differs")

    wrist_keys = {
        "schema_version",
        "operator",
        "side",
        "pose_position_m",
        "pose_quaternion_xyzw",
        "pose_timestamp_us",
        "receipt_timestamp_us",
        "raw_pose_sha256",
        "pose_source",
        "clock_domain",
        "capture_class",
        "formal_live_eligible",
        "external_blockers",
    }
    seen_wrist_raw: set[str] = set()
    for pair_index, raw_pair in enumerate(pairs):
        pair = _require_mapping(raw_pair, f"adapter capture pair[{pair_index}]")
        received_times: list[int] = []
        source_times: list[int] = []
        for operator_offset, operator in enumerate(_OPERATORS):
            evidence = _require_mapping(
                pair[operator], f"adapter capture pair[{pair_index}].{operator}"
            )
            _require_exact_keys(
                evidence,
                {"adapter_received_timestamp_us", "manus", "wrist_pose"},
                f"adapter capture pair[{pair_index}].{operator}",
            )
            received = _integer(
                evidence["adapter_received_timestamp_us"],
                f"adapter capture pair[{pair_index}].{operator}.received",
            )
            expected_receipt = receipts[pair_index * 2 + operator_offset]
            manus = _require_mapping(
                evidence["manus"],
                f"adapter capture pair[{pair_index}].{operator}.manus",
            )
            if dict(manus) != dict(expected_receipt):
                raise FormalCaptureBundleError(
                    "adapter decoder-issued receipt must exactly match the protocol receipt binding"
                )
            source_timestamp = int(manus["timestamp_us"])
            decoder_receipt_timestamp = int(manus["receipt_timestamp_us"])
            for observed_timestamp, label in (
                (source_timestamp, "source frame"),
                (decoder_receipt_timestamp, "decoder receipt"),
            ):
                age = received - observed_timestamp
                if age < 0 or age > max_latency:
                    raise FormalCaptureBundleError(
                        f"adapter {label} latency exceeds persisted runtime limit"
                    )
            received_times.append(received)
            source_times.append(source_timestamp)

            wrists = _require_mapping(
                evidence["wrist_pose"],
                f"adapter capture pair[{pair_index}].{operator}.wrist_pose",
            )
            if set(wrists) != {"right", "left"}:
                raise FormalCaptureBundleError(
                    "adapter wrist receipts must bind exactly right and left"
                )
            identity = manus["identity"]
            for side in ("right", "left"):
                wrist = _require_mapping(
                    wrists[side],
                    f"adapter capture pair[{pair_index}].{operator}.wrist.{side}",
                )
                _require_exact_keys(
                    wrist,
                    wrist_keys,
                    f"adapter capture pair[{pair_index}].{operator}.wrist.{side}",
                )
                if (
                    wrist["schema_version"] != _WRIST_RECEIPT_SCHEMA
                    or wrist["operator"] != operator
                    or wrist["side"] != side
                ):
                    raise FormalCaptureBundleError(
                        "adapter wrist receipt operator/side/schema binding differs"
                    )
                position = _finite_vector(
                    wrist["pose_position_m"], 3, "wrist pose_position_m"
                )
                quaternion = _finite_vector(
                    wrist["pose_quaternion_xyzw"], 4, "wrist pose_quaternion_xyzw"
                )
                del position
                if not math.isclose(
                    math.sqrt(sum(component * component for component in quaternion)),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    raise FormalCaptureBundleError(
                        "adapter wrist quaternion must be unit xyzw"
                    )
                pose_timestamp = _integer(
                    wrist["pose_timestamp_us"], "wrist pose_timestamp_us"
                )
                wrist_receipt_timestamp = _integer(
                    wrist["receipt_timestamp_us"], "wrist receipt_timestamp_us"
                )
                if wrist_receipt_timestamp < pose_timestamp:
                    raise FormalCaptureBundleError("wrist receipt clock regressed")
                if (
                    received - pose_timestamp < 0
                    or received - pose_timestamp > max_latency
                    or received - wrist_receipt_timestamp < 0
                    or received - wrist_receipt_timestamp > max_latency
                ):
                    raise FormalCaptureBundleError(
                        "adapter wrist latency exceeds persisted runtime limit"
                    )
                if abs(pose_timestamp - source_timestamp) > max_pose_skew:
                    raise FormalCaptureBundleError(
                        "adapter wrist/frame skew exceeds persisted runtime limit"
                    )
                wrist_raw_sha = _sha256(
                    wrist["raw_pose_sha256"], "wrist raw_pose_sha256"
                )
                if wrist_raw_sha in seen_wrist_raw:
                    raise FormalCaptureBundleError(
                        "duplicate wrist raw-pose receipt binding"
                    )
                seen_wrist_raw.add(wrist_raw_sha)
                if (
                    wrist["pose_source"] != identity["pose_source"]
                    or wrist["clock_domain"] != identity["clock_domain"]
                    or wrist["capture_class"] != capture_class
                ):
                    raise FormalCaptureBundleError(
                        "adapter wrist source/clock/capture binding differs"
                    )
                blockers = wrist["external_blockers"]
                if not isinstance(blockers, list) or any(
                    not isinstance(item, str) or not item for item in blockers
                ):
                    raise FormalCaptureBundleError(
                        "adapter wrist blocker evidence differs"
                    )
                if not isinstance(wrist["formal_live_eligible"], bool):
                    raise FormalCaptureBundleError(
                        "adapter wrist formal-live eligibility is missing"
                    )
        if len(set(received_times)) != 1:
            raise FormalCaptureBundleError(
                "atomic operator pair must have one adapter receipt timestamp"
            )
        if abs(source_times[0] - source_times[1]) > max_pair_skew:
            raise FormalCaptureBundleError(
                "adapter operator pair skew exceeds persisted runtime limit"
            )


def _float_binding_equal(left: float, right: float) -> bool:
    tolerance = 4.0 * max(math.ulp(left), math.ulp(right))
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def _validate_core_semantics(
    core_path: Path,
    recorder: Mapping[str, Any],
    adapter: Mapping[str, Any],
) -> dict[str, Any]:
    """Load the opaque core through its public validator and bind its meaning."""
    try:
        core = load_episode_bundle(core_path)
    except (BundleValidationError, OSError) as exc:
        raise FormalCaptureBundleError(
            f"opaque core bundle failed strict validation: {exc}"
        ) from exc
    manifest = core["manifest"]
    exogenous = core["exogenous"]
    steps = int(manifest["steps"])
    if steps != recorder["control_transitions"]:
        raise FormalCaptureBundleError("core/recorder step count binding differs")
    core_dt = float(manifest["dt"])
    recorder_dt = float(recorder["replay_clock_policy"]["replay_dt_s"])
    if not _float_binding_equal(core_dt, recorder_dt):
        raise FormalCaptureBundleError("core dt differs from recorder replay cadence")
    try:
        source_trace = recorder_source_trace_sha256(exogenous, recorder["operators"])
    except RecorderContractError as exc:
        raise FormalCaptureBundleError(
            f"core recorder source trace cannot be recomputed: {exc}"
        ) from exc
    if source_trace != recorder["source_trace_sha256"]:
        raise FormalCaptureBundleError(
            "recorder source trace differs from recomputed core binding"
        )
    try:
        adapter_trace = adapter_output_trace_sha256_from_exogenous(exogenous)
    except AdapterContractError as exc:
        raise FormalCaptureBundleError(
            f"core adapter output trace cannot be recomputed: {exc}"
        ) from exc
    if adapter_trace != adapter["output_trace_sha256"]:
        raise FormalCaptureBundleError(
            "adapter output trace differs from recomputed core binding"
        )
    return {
        "steps": steps,
        "dt": core_dt,
        "initial_state_digest": manifest["initial_state_digest"],
        "exogenous_digest": manifest["exogenous_digest"],
        "recorder_source_trace_sha256": source_trace,
        "adapter_output_trace_sha256": adapter_trace,
    }


def _flatten_adapter_receipts(adapter: Mapping[str, Any]) -> list[dict[str, Any]]:
    capture_receipts = adapter["capture_receipts"]
    phases = [capture_receipts["baseline"], *capture_receipts["transitions"]]
    return [
        copy.deepcopy(phase[operator]["manus"])
        for phase in phases
        for operator in _OPERATORS
    ]


def _raw_evidence_snapshot(
    raw_evidence: Mapping[str, Any], adapter: Mapping[str, Any]
) -> dict[str, Any]:
    receipt_phases = [
        adapter["capture_receipts"]["baseline"],
        *adapter["capture_receipts"]["transitions"],
    ]
    raw_phases = [raw_evidence["baseline"], *raw_evidence["transitions"]]
    if len(raw_phases) != len(receipt_phases):
        raise FormalCaptureBundleError("raw evidence transition count differs")
    encoded_phases: list[dict[str, Any]] = []
    for raw_phase, receipt_phase in zip(raw_phases, receipt_phases, strict=True):
        encoded_pair: dict[str, Any] = {}
        for operator in _OPERATORS:
            operator_raw = raw_phase[operator]
            operator_receipt = receipt_phase[operator]
            encoded_pair[operator] = {
                "manus": _encode_raw_payload(
                    operator_raw["manus_raw_payload"],
                    operator_receipt["manus"]["raw_frame_sha256"],
                ),
                "wrist_pose": {
                    side: _encode_raw_payload(
                        operator_raw["wrist_raw_payloads"][side],
                        operator_receipt["wrist_pose"][side]["raw_pose_sha256"],
                    )
                    for side in ("right", "left")
                },
            }
        encoded_phases.append(encoded_pair)
    return {
        "baseline": encoded_phases[0],
        "transitions": encoded_phases[1:],
    }


def _redecode_persisted_manus(
    payload: bytes,
    expected_receipt: Mapping[str, Any],
    label: str,
) -> None:
    identity = expected_receipt["identity"]
    try:
        snapshot = capture_stable_snapshot(lambda: payload, max_attempts=1)
        decoded = decode_manus_shm_v1(
            snapshot,
            identity=ManusStreamIdentity(
                operator=expected_receipt["operator"],
                stream_id=identity["stream_id"],
                operator_device_id=identity["operator_device_id"],
                source_id=identity["source_id"],
                left_glove_device_id=identity["left_glove_device_id"],
                right_glove_device_id=identity["right_glove_device_id"],
                producer_identity=identity["producer_identity"],
                producer_binary_sha256=identity["producer_binary_sha256"],
                producer_source_sha256=identity["producer_source_sha256"],
                decoder_identity=identity["decoder_identity"],
                decoder_source_sha256=identity["decoder_source_sha256"],
                pose_source=identity["pose_source"],
                clock_domain=identity["clock_domain"],
            ),
            receipt_timestamp_us=expected_receipt["receipt_timestamp_us"],
            capture_class=expected_receipt["capture_class"],
        )
    except ManusShmV1ContractError as exc:
        raise FormalCaptureBundleError(
            f"{label} failed strict MANU re-decode: {exc}"
        ) from exc
    recomputed = decoded.provenance
    for field in set(expected_receipt) - {"snapshot"}:
        if recomputed[field] != expected_receipt[field]:
            raise FormalCaptureBundleError(
                f"{label} re-decoded {field} differs from persisted receipt"
            )


def _validate_raw_evidence(
    value: Any,
    adapter: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = _require_mapping(_json_snapshot(value, "raw evidence"), "raw evidence")
    _require_exact_keys(evidence, {"baseline", "transitions"}, "raw evidence")
    receipt_phases = [
        adapter["capture_receipts"]["baseline"],
        *adapter["capture_receipts"]["transitions"],
    ]
    transitions = evidence["transitions"]
    if not isinstance(transitions, list):
        raise FormalCaptureBundleError("raw evidence transitions must be a list")
    raw_phases = [evidence["baseline"], *transitions]
    if len(raw_phases) != len(receipt_phases):
        raise FormalCaptureBundleError("raw evidence transition count differs")
    for pair_index, (raw_phase, receipt_phase) in enumerate(
        zip(raw_phases, receipt_phases, strict=True)
    ):
        raw_pair = _require_mapping(raw_phase, f"raw evidence pair[{pair_index}]")
        if set(raw_pair) != set(_OPERATORS):
            raise FormalCaptureBundleError("raw evidence operator binding differs")
        for operator in _OPERATORS:
            operator_raw = _require_mapping(
                raw_pair[operator], f"raw evidence pair[{pair_index}].{operator}"
            )
            _require_exact_keys(
                operator_raw,
                {"manus", "wrist_pose"},
                f"raw evidence pair[{pair_index}].{operator}",
            )
            manus_payload, manus_sha = _decode_raw_payload(
                operator_raw["manus"],
                f"raw evidence pair[{pair_index}].{operator}.manus",
            )
            if len(manus_payload) != 1448:
                raise FormalCaptureBundleError(
                    "raw MANU evidence must contain exactly 1448 bytes"
                )
            expected_manus_sha = receipt_phase[operator]["manus"]["raw_frame_sha256"]
            if not hmac.compare_digest(manus_sha, expected_manus_sha):
                raise FormalCaptureBundleError(
                    "raw MANU evidence differs from decoder-issued receipt"
                )
            _redecode_persisted_manus(
                manus_payload,
                receipt_phase[operator]["manus"],
                f"raw evidence pair[{pair_index}].{operator}.manus",
            )
            wrists = _require_mapping(
                operator_raw["wrist_pose"],
                f"raw evidence pair[{pair_index}].{operator}.wrist_pose",
            )
            if set(wrists) != {"right", "left"}:
                raise FormalCaptureBundleError("raw wrist evidence binding differs")
            for side in ("right", "left"):
                wrist_payload, wrist_sha = _decode_raw_payload(
                    wrists[side],
                    f"raw evidence pair[{pair_index}].{operator}.wrist.{side}",
                )
                if not wrist_payload:
                    raise FormalCaptureBundleError(
                        "raw wrist evidence payload must be non-empty"
                    )
                expected_wrist_sha = receipt_phase[operator]["wrist_pose"][side][
                    "raw_pose_sha256"
                ]
                if not hmac.compare_digest(wrist_sha, expected_wrist_sha):
                    raise FormalCaptureBundleError(
                        "raw wrist evidence differs from issued wrist receipt"
                    )
    return dict(evidence)


def _validate_capture_snapshot(
    value: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    snapshot = _require_mapping(
        _json_snapshot(value, "capture snapshot"), "capture snapshot"
    )
    keys = {
        "schema_version",
        "capture_class",
        "recorder",
        "adapter",
        "protocol_receipts",
        "raw_evidence",
        "closure_sha256",
    }
    _require_exact_keys(snapshot, keys, "capture snapshot")
    if snapshot["schema_version"] != _CAPTURE_SNAPSHOT_SCHEMA:
        raise FormalCaptureBundleError("capture snapshot schema differs")
    capture_class = snapshot["capture_class"]
    if capture_class not in _CAPTURE_CLASSES:
        raise FormalCaptureBundleError("capture snapshot class is not formal")
    closure = _sha256(snapshot["closure_sha256"], "capture snapshot closure")
    closure_payload = {key: snapshot[key] for key in keys - {"closure_sha256"}}
    if not hmac.compare_digest(_capture_closure_digest(closure_payload), closure):
        raise FormalCaptureBundleError("capture snapshot closure digest mismatch")
    recorder = _validate_recorder(snapshot["recorder"])
    adapter = _validate_adapter(snapshot["adapter"], recorder["control_transitions"])
    receipts = _validate_receipts(
        snapshot["protocol_receipts"], recorder, adapter, capture_class
    )
    _validate_capture_evidence(adapter, receipts, capture_class)
    _validate_raw_evidence(snapshot["raw_evidence"], adapter)
    return dict(snapshot), recorder, adapter, receipts


def issue_formal_capture_capability(
    *,
    recorder: GloveEpisodeRecorder,
    adapter: ManusGloveAdapter,
) -> FormalCaptureCapability:
    """Jointly seal one completed in-process recorder/adapter capture."""
    if not isinstance(recorder, GloveEpisodeRecorder) or not isinstance(
        adapter, ManusGloveAdapter
    ):
        raise FormalCaptureBundleError(
            "capture capability requires concrete recorder and adapter instances"
        )
    try:
        # Commit lock order is adapter then recorder; issuance uses the same order.
        with adapter._lock:
            with recorder._lock:
                exogenous, recorder_provenance = recorder._capture_material_unlocked(
                    adapter=adapter
                )
                adapter_provenance, raw_evidence = adapter._capture_material_unlocked(
                    recorder=recorder
                )
    except (RecorderContractError, AdapterContractError) as exc:
        raise FormalCaptureBundleError(
            f"capture capability cannot be jointly issued: {exc}"
        ) from exc

    recorder_snapshot = _validate_recorder(recorder_provenance)
    adapter_snapshot = _validate_adapter(
        adapter_provenance, recorder_snapshot["control_transitions"]
    )
    if (
        recorder_source_trace_sha256(exogenous, recorder_snapshot["operators"])
        != recorder_snapshot["source_trace_sha256"]
    ):
        raise FormalCaptureBundleError(
            "joint recorder source trace differs before capability issuance"
        )
    if (
        adapter_output_trace_sha256_from_exogenous(exogenous)
        != adapter_snapshot["output_trace_sha256"]
    ):
        raise FormalCaptureBundleError(
            "joint adapter output trace differs before capability issuance"
        )
    receipts = _flatten_adapter_receipts(adapter_snapshot)
    capture_classes = {receipt["capture_class"] for receipt in receipts}
    if len(capture_classes) != 1:
        raise FormalCaptureBundleError("joint capture classes differ")
    capture_class = next(iter(capture_classes))
    raw_snapshot = _raw_evidence_snapshot(raw_evidence, adapter_snapshot)
    closure_payload = {
        "schema_version": _CAPTURE_SNAPSHOT_SCHEMA,
        "capture_class": capture_class,
        "recorder": recorder_snapshot,
        "adapter": adapter_snapshot,
        "protocol_receipts": receipts,
        "raw_evidence": raw_snapshot,
    }
    snapshot = {
        **closure_payload,
        "closure_sha256": _capture_closure_digest(closure_payload),
    }
    normalized, _recorder, _adapter, _receipts = _validate_capture_snapshot(snapshot)
    capability = FormalCaptureCapability(
        closure_sha256=normalized["closure_sha256"],
        issuer_token=_CAPABILITY_ISSUER,
    )
    _ISSUED_CAPABILITIES[capability] = normalized
    return capability


def _require_issued_capability(
    value: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if (
        not isinstance(value, FormalCaptureCapability)
        or getattr(value, "_issuer_token", None) is not _CAPABILITY_ISSUER
    ):
        raise FormalCaptureBundleError(
            "writer requires an opaque jointly issued capture capability"
        )
    snapshot = _ISSUED_CAPABILITIES.get(value)
    closure = getattr(value, "_closure_sha256", None)
    if (
        snapshot is None
        or not isinstance(closure, str)
        or not hmac.compare_digest(closure, snapshot["closure_sha256"])
    ):
        raise FormalCaptureBundleError(
            "capture capability is unissued, copied, stale, or forged"
        )
    return _validate_capture_snapshot(snapshot)


def _cadence_sidecar(recorder: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": _CADENCE_SCHEMA,
        "control_transitions": recorder["control_transitions"],
        "replay_clock_policy": recorder["replay_clock_policy"],
        "operators": {
            operator: {
                "seq": recorder["operators"][operator]["seq"],
                "timestamp_s": recorder["operators"][operator]["timestamp_s"],
            }
            for operator in _OPERATORS
        },
    }


def _external_blockers(
    capture_class: str,
    adapter: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
) -> list[str]:
    # This repository does not contain the producer binary bytes/source audit
    # and producer-side seqlock proof.  Self-declared flags in a sidecar cannot
    # turn that absent evidence into a formal-live authorization.
    blockers: set[str] = {"formal_live_producer_evidence_unavailable"}
    if capture_class == "synthetic_test":
        blockers.add("synthetic_capture_not_formal_live")
    if adapter.get("formal_live_eligible") is not True:
        blockers.add("adapter_not_formal_live_eligible")
    blockers.update(str(item) for item in adapter.get("external_blockers", []))
    for receipt in receipts:
        blockers.update(str(item) for item in receipt["external_blockers"])
        if receipt["formal_live_eligible"] is not True:
            blockers.add("receipt_not_formal_live_eligible")
    return sorted(blockers)


def _write_fsync(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_formal_capture_bundle(
    path: str | os.PathLike[str],
    *,
    core_bundle_path: str | os.PathLike[str],
    capture_capability: FormalCaptureCapability | None = None,
    recorder_provenance: Mapping[str, Any] | None = None,
    adapter_provenance: Mapping[str, Any] | None = None,
    protocol_receipts: Sequence[Mapping[str, Any]] | None = None,
    capture_class: str | None = None,
) -> dict[str, Any]:
    """Atomically snapshot and content-bind a core bundle plus capture evidence."""
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"formal capture bundle already exists: {target}")
    if any(
        legacy is not None
        for legacy in (
            recorder_provenance,
            adapter_provenance,
            protocol_receipts,
            capture_class,
        )
    ):
        raise FormalCaptureBundleError(
            "ordinary provenance/receipt dictionaries cannot authorize the writer; "
            "use one opaque jointly issued capture capability"
        )
    capture_snapshot, recorder, adapter, receipts = _require_issued_capability(
        capture_capability
    )
    capture_class = capture_snapshot["capture_class"]
    core_files, core_directories = _snapshot_core(Path(core_bundle_path))
    core_semantics = _validate_core_semantics(Path(core_bundle_path), recorder, adapter)
    if _scan_regular_tree(Path(core_bundle_path)) != (core_files, core_directories):
        raise FormalCaptureBundleError(
            "opaque core changed while its semantic binding was validated"
        )
    cadence = _cadence_sidecar(recorder)
    blockers = _external_blockers(capture_class, adapter, receipts)
    formal_live_ready = capture_class == "formal_live" and not blockers
    if capture_class == "formal_live" and not formal_live_ready:
        raise FormalCaptureBundleError(
            "formal live capture is not eligible; external blocker evidence remains: "
            + ", ".join(blockers)
        )

    sidecars = {
        "capture_snapshot": capture_snapshot,
        "recorder": recorder,
        "adapter": adapter,
        "protocol_receipts": receipts,
        "cadence": cadence,
    }
    sidecar_bytes = {name: _canonical_bytes(value) for name, value in sidecars.items()}
    core_bindings = {
        name: _file_binding(data) for name, data in sorted(core_files.items())
    }
    sidecar_bindings = {
        _SIDECAR_FILES[name]: _file_binding(data)
        for name, data in sorted(sidecar_bytes.items())
    }
    binding = {
        "schema_version": SCHEMA_VERSION,
        "capture_class": capture_class,
        "formal_live_ready": formal_live_ready,
        "external_blockers": blockers,
        "core_files": core_bindings,
        "core_directories": sorted(core_directories),
        "core_manifest_sha256": core_bindings["manifest.json"]["sha256"],
        "core_exogenous_sha256": core_bindings["exogenous.json"]["sha256"],
        "core_semantics": core_semantics,
        "sidecar_files": sidecar_bindings,
    }
    manifest = {**binding, "capture_digest": _binding_digest(binding)}
    manifest_bytes = _canonical_bytes(manifest)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    claimed = False
    try:
        for relative in sorted(core_directories):
            (temporary / "core" / relative).mkdir(parents=True, exist_ok=True)
        for relative, data in core_files.items():
            _write_fsync(temporary / "core" / relative, data)
        for name, data in sidecar_bytes.items():
            _write_fsync(temporary / "sidecars" / _SIDECAR_FILES[name], data)
        _write_fsync(temporary / "formal_manifest.json", manifest_bytes)
        for directory in sorted(
            (item for item in temporary.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        _fsync_directory(temporary)

        target.mkdir(mode=0o700, exist_ok=False)
        claimed = True
        # The manifest is the publication marker and is moved only after the
        # opaque core and every sidecar are already in the claimed directory.
        (temporary / "core").replace(target / "core")
        (temporary / "sidecars").replace(target / "sidecars")
        (temporary / "formal_manifest.json").replace(target / "formal_manifest.json")
        _fsync_directory(target)
        _fsync_directory(target.parent)
        temporary.rmdir()
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        if claimed and target.exists():
            shutil.rmtree(target)
        raise
    return _json_snapshot(manifest, "formal manifest")


def _read_json_canonical(path: Path, label: str) -> tuple[Any, bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FormalCaptureBundleError(f"{label} JSON cannot be read: {exc}") from exc
    if raw != _canonical_bytes(value):
        raise FormalCaptureBundleError(f"{label} JSON canonical binding differs")
    return value, raw


def load_formal_capture_bundle(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and revalidate every opaque-core and capture-sidecar byte binding."""
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise FormalCaptureBundleError(
            f"formal capture bundle is not a real directory: {root}"
        )
    manifest_value, manifest_bytes = _read_json_canonical(
        root / "formal_manifest.json", "formal manifest"
    )
    manifest = _require_mapping(manifest_value, "formal manifest")
    manifest_keys = {
        "schema_version",
        "capture_class",
        "formal_live_ready",
        "external_blockers",
        "core_files",
        "core_directories",
        "core_manifest_sha256",
        "core_exogenous_sha256",
        "core_semantics",
        "sidecar_files",
        "capture_digest",
    }
    _require_exact_keys(manifest, manifest_keys, "formal manifest")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise FormalCaptureBundleError("formal manifest schema differs")
    capture_digest = _sha256(manifest["capture_digest"], "capture_digest")
    binding = {key: manifest[key] for key in manifest_keys - {"capture_digest"}}
    if _binding_digest(binding) != capture_digest:
        raise FormalCaptureBundleError("formal manifest capture digest mismatch")

    core_bindings = _require_mapping(manifest["core_files"], "core file bindings")
    sidecar_bindings = _require_mapping(
        manifest["sidecar_files"], "sidecar file bindings"
    )
    if set(sidecar_bindings) != set(_SIDECAR_FILES.values()):
        raise FormalCaptureBundleError("sidecar files/binding differ")
    expected_files = {"formal_manifest.json"}
    expected_files.update(f"core/{relative}" for relative in core_bindings)
    expected_files.update(f"sidecars/{relative}" for relative in sidecar_bindings)
    expected_directories = {"core", "sidecars"}
    expected_directories.update(
        f"core/{relative}" for relative in manifest["core_directories"]
    )
    actual_files, actual_directories = _scan_regular_tree(root)
    if (
        set(actual_files) != expected_files
        or actual_directories != expected_directories
    ):
        raise FormalCaptureBundleError(
            "formal capture files/directories differ from content binding"
        )
    if actual_files["formal_manifest.json"] != manifest_bytes:
        raise FormalCaptureBundleError("formal manifest byte binding differs")

    for relative, raw_binding in core_bindings.items():
        expected = _require_mapping(raw_binding, f"core binding {relative}")
        _require_exact_keys(
            expected, {"sha256", "size_bytes"}, f"core binding {relative}"
        )
        data = actual_files[f"core/{relative}"]
        if _file_binding(data) != dict(expected):
            raise FormalCaptureBundleError(
                f"core/{relative} sha256/digest binding mismatch"
            )
    if manifest["core_manifest_sha256"] != core_bindings["manifest.json"]["sha256"]:
        raise FormalCaptureBundleError("core manifest sha256 binding differs")
    if manifest["core_exogenous_sha256"] != core_bindings["exogenous.json"]["sha256"]:
        raise FormalCaptureBundleError("core exogenous sha256 binding differs")

    loaded_sidecars: dict[str, Any] = {}
    for name, filename in _SIDECAR_FILES.items():
        value, data = _read_json_canonical(
            root / "sidecars" / filename, f"sidecar {name}"
        )
        expected = _require_mapping(
            sidecar_bindings[filename], f"sidecar binding {filename}"
        )
        _require_exact_keys(
            expected, {"sha256", "size_bytes"}, f"sidecar binding {filename}"
        )
        if _file_binding(data) != dict(expected):
            raise FormalCaptureBundleError(
                f"sidecar {name} sha256/digest binding mismatch"
            )
        loaded_sidecars[name] = value

    capture_snapshot, recorder, adapter, receipts = _validate_capture_snapshot(
        loaded_sidecars["capture_snapshot"]
    )
    if manifest["capture_class"] != capture_snapshot["capture_class"]:
        raise FormalCaptureBundleError("manifest/capture capability class differs")
    if (
        loaded_sidecars["recorder"] != recorder
        or loaded_sidecars["adapter"] != adapter
        or loaded_sidecars["protocol_receipts"] != receipts
    ):
        raise FormalCaptureBundleError(
            "derived sidecars differ from the sealed capture snapshot"
        )
    core_semantics = _validate_core_semantics(root / "core", recorder, adapter)
    if manifest["core_semantics"] != core_semantics:
        raise FormalCaptureBundleError("core semantic manifest binding differs")
    if loaded_sidecars["cadence"] != _cadence_sidecar(recorder):
        raise FormalCaptureBundleError("cadence sidecar binding differs")
    blockers = _external_blockers(manifest["capture_class"], adapter, receipts)
    expected_ready = manifest["capture_class"] == "formal_live" and not blockers
    if (
        manifest["external_blockers"] != blockers
        or manifest["formal_live_ready"] is not expected_ready
    ):
        raise FormalCaptureBundleError("formal-live blocker binding differs")
    return {
        "manifest": _json_snapshot(manifest, "formal manifest"),
        "sidecars": _json_snapshot(loaded_sidecars, "sidecars"),
    }


__all__ = [
    "FormalCaptureCapability",
    "FormalCaptureBundleError",
    "SCHEMA_VERSION",
    "issue_formal_capture_capability",
    "load_formal_capture_bundle",
    "write_formal_capture_bundle",
]
