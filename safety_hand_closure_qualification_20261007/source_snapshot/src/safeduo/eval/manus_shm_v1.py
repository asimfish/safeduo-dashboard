"""Strict, transport-neutral decoder for the audited MANU SHM-v1 byte layout.

This module never opens shared memory.  A transport supplies a callable which
returns byte snapshots; :func:`capture_stable_snapshot` performs the required
double-read seam.  Double-read equality detects common torn reads but is not a
producer-side seqlock proof, so live receipts deliberately remain ineligible
for formal capture.
"""

from __future__ import annotations

import copy
import hashlib
import math
import struct
import weakref
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


MANUS_SHM_V1_MAGIC = 0x4D414E55
MANUS_SHM_V1_VERSION = 1
MANUS_SHM_V1_BLOCK_SIZE = 1448
MANUS_SHM_V1_HAND_COUNT = 2
MANUS_SHM_V1_NODE_COUNT = 25

_HEADER_SIZE = 32
_HAND_SIZE = 708
_HAND_HEADER_SIZE = 8
_NODE_SIZE = 28
_SLOTS = (("right", 2), ("left", 1))
_CAPTURE_CLASSES = {"synthetic_test", "live", "formal_live"}
_SHA256_LENGTH = 64
_RECEIPT_SCHEMA = "d032.manus_shm_v1.receipt.v1"
_RECEIPT_ISSUER = object()
_ISSUED_RECEIPTS: weakref.WeakSet[Any] = weakref.WeakSet()


class ManusShmV1ContractError(ValueError):
    """A MANU byte frame, identity, or receipt boundary is invalid."""


class TornSnapshotError(ManusShmV1ContractError):
    """No byte-identical double read was observed within the attempt budget."""


@dataclass(frozen=True)
class ManusStreamIdentity:
    """Explicit identity and provenance binding for one operator stream."""

    operator: str
    stream_id: str
    operator_device_id: str
    source_id: str
    left_glove_device_id: str
    right_glove_device_id: str
    producer_identity: str
    producer_binary_sha256: str
    producer_source_sha256: str | None
    decoder_identity: str
    decoder_source_sha256: str
    pose_source: str
    clock_domain: str


@dataclass(frozen=True)
class _StableSnapshot:
    payload: bytes
    read_count: int


class DecodedManusHand:
    """Defensive decoded copy of one fixed MANU hand slot."""

    __slots__ = ("_positions_m", "_quaternions_xyzw", "node_count", "side")

    def __init__(
        self,
        *,
        side: str,
        positions_m: np.ndarray,
        quaternions_xyzw: np.ndarray,
    ) -> None:
        self.side = side
        self.node_count = MANUS_SHM_V1_NODE_COUNT
        self._positions_m = np.asarray(positions_m, dtype=np.float64).copy()
        self._quaternions_xyzw = np.asarray(quaternions_xyzw, dtype=np.float64).copy()

    @property
    def positions_m(self) -> np.ndarray:
        return self._positions_m.copy()

    @property
    def quaternions_xyzw(self) -> np.ndarray:
        return self._quaternions_xyzw.copy()


class ManusShmV1Receipt:
    """Decoded data plus an immutable, separately copyable evidence receipt."""

    __slots__ = (
        "__weakref__",
        "_external_blockers",
        "_hands",
        "_provenance",
        "_raw_payload",
        "_issuer_token",
        "formal_live_eligible",
        "raw_frame_sha256",
        "seq",
        "timestamp_us",
    )

    def __init__(
        self,
        *,
        seq: int,
        timestamp_us: int,
        hands: tuple[DecodedManusHand, DecodedManusHand],
        raw_frame_sha256: str,
        raw_payload: bytes,
        provenance: dict[str, Any],
        issuer_token: object,
    ) -> None:
        if issuer_token is not _RECEIPT_ISSUER:
            raise ManusShmV1ContractError(
                "Manus receipt must be issued by decode_manus_shm_v1"
            )
        self.seq = seq
        self.timestamp_us = timestamp_us
        self._hands = hands
        self.raw_frame_sha256 = raw_frame_sha256
        self._raw_payload = bytes(raw_payload)
        if hashlib.sha256(self._raw_payload).hexdigest() != raw_frame_sha256:
            raise ManusShmV1ContractError(
                "decoder receipt raw payload sha256 binding differs"
            )
        self.formal_live_eligible = bool(provenance["formal_live_eligible"])
        self._external_blockers = tuple(provenance["external_blockers"])
        self._provenance = copy.deepcopy(provenance)
        self._issuer_token = issuer_token

    @property
    def hands(self) -> tuple[DecodedManusHand, DecodedManusHand]:
        return self._hands

    @property
    def external_blockers(self) -> tuple[str, ...]:
        return self._external_blockers

    @property
    def provenance(self) -> dict[str, Any]:
        return copy.deepcopy(self._provenance)

    @property
    def raw_payload(self) -> bytes:
        """Return the exact immutable 1448-byte decoder input."""
        return self._raw_payload


def require_decoder_issued_receipt(value: Any) -> ManusShmV1Receipt:
    """Return an authentic in-process decoder receipt or fail closed."""
    if (
        not isinstance(value, ManusShmV1Receipt)
        or value._issuer_token is not _RECEIPT_ISSUER
        or value not in _ISSUED_RECEIPTS
    ):
        raise ManusShmV1ContractError(
            "sample must carry a decoder-issued MANU SHM-v1 receipt token"
        )
    return value


def _strict_bytes(value: Any, label: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ManusShmV1ContractError(f"{label} must return bytes-like data")
    return bytes(value)


def capture_stable_snapshot(
    read_once: Callable[[], bytes],
    *,
    max_attempts: int = 3,
) -> _StableSnapshot:
    """Return one byte-identical double read or fail without decoding either read."""
    if not callable(read_once):
        raise ManusShmV1ContractError("stable snapshot read_once must be callable")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
        raise ManusShmV1ContractError("max_attempts must be an integer")
    if max_attempts < 1:
        raise ManusShmV1ContractError("max_attempts must be positive")
    for attempt in range(1, max_attempts + 1):
        first = _strict_bytes(read_once(), "stable snapshot first read")
        second = _strict_bytes(read_once(), "stable snapshot second read")
        if first == second:
            return _StableSnapshot(payload=bytes(first), read_count=attempt * 2)
    raise TornSnapshotError(
        f"torn snapshot: no stable byte-identical double read in {max_attempts} attempts"
    )


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManusShmV1ContractError(f"identity {label} must be a non-empty string")
    return value


def _sha256(value: Any, label: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    normalized = _nonempty_string(value, label)
    if len(normalized) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ManusShmV1ContractError(f"identity {label} must be a lowercase sha256")
    return normalized


def _validate_identity(identity: Any) -> ManusStreamIdentity:
    if not isinstance(identity, ManusStreamIdentity):
        raise ManusShmV1ContractError("identity must be ManusStreamIdentity")
    if identity.operator not in {"F", "U"}:
        raise ManusShmV1ContractError("identity operator must be F or U")
    for field in (
        "stream_id",
        "operator_device_id",
        "source_id",
        "left_glove_device_id",
        "right_glove_device_id",
        "producer_identity",
        "decoder_identity",
        "pose_source",
        "clock_domain",
    ):
        _nonempty_string(getattr(identity, field), field)
    if identity.left_glove_device_id == identity.right_glove_device_id:
        raise ManusShmV1ContractError(
            "identity left/right glove devices must be distinct"
        )
    _sha256(identity.producer_binary_sha256, "producer_binary_sha256")
    _sha256(
        identity.producer_source_sha256,
        "producer_source_sha256",
        optional=True,
    )
    _sha256(identity.decoder_source_sha256, "decoder_source_sha256")
    return identity


def _identity_provenance(identity: ManusStreamIdentity) -> dict[str, Any]:
    return {
        "stream_id": identity.stream_id,
        "operator_device_id": identity.operator_device_id,
        "source_id": identity.source_id,
        "left_glove_device_id": identity.left_glove_device_id,
        "right_glove_device_id": identity.right_glove_device_id,
        "producer_identity": identity.producer_identity,
        "producer_binary_sha256": identity.producer_binary_sha256,
        "producer_source_sha256": identity.producer_source_sha256,
        "decoder_identity": identity.decoder_identity,
        "decoder_source_sha256": identity.decoder_source_sha256,
        "pose_source": identity.pose_source,
        "clock_domain": identity.clock_domain,
    }


def _decode_hand(payload: bytes, slot: int) -> DecodedManusHand:
    expected_side, expected_side_code = _SLOTS[slot]
    hand_offset = _HEADER_SIZE + slot * _HAND_SIZE
    valid, side_code = struct.unpack_from("<BB", payload, hand_offset)
    if payload[hand_offset + 2 : hand_offset + 4] != b"\0\0":
        raise ManusShmV1ContractError(f"hand slot {slot} padding must be zero")
    (node_count,) = struct.unpack_from("<I", payload, hand_offset + 4)
    if valid != 1:
        raise ManusShmV1ContractError(f"hand slot {slot} valid must be exactly 1")
    if side_code != expected_side_code:
        raise ManusShmV1ContractError(
            f"hand slot {slot} side must be {expected_side} ({expected_side_code})"
        )
    if node_count != MANUS_SHM_V1_NODE_COUNT:
        raise ManusShmV1ContractError(
            f"hand slot {slot} node_count must be exactly 25; no clamp is permitted"
        )
    positions = np.empty((MANUS_SHM_V1_NODE_COUNT, 3), dtype=np.float64)
    quaternions = np.empty((MANUS_SHM_V1_NODE_COUNT, 4), dtype=np.float64)
    for node in range(MANUS_SHM_V1_NODE_COUNT):
        node_offset = hand_offset + _HAND_HEADER_SIZE + node * _NODE_SIZE
        values = np.asarray(
            struct.unpack_from("<7f", payload, node_offset), dtype=np.float64
        )
        if not np.isfinite(values).all():
            raise ManusShmV1ContractError(
                f"hand slot {slot} node {node} values must be finite"
            )
        quaternion = values[3:]
        if not math.isclose(
            float(np.linalg.norm(quaternion)), 1.0, rel_tol=0.0, abs_tol=1e-5
        ):
            raise ManusShmV1ContractError(
                f"hand slot {slot} node {node} must have a unit quaternion in xyzw order"
            )
        positions[node] = values[:3]
        quaternions[node] = quaternion
    return DecodedManusHand(
        side=expected_side,
        positions_m=positions,
        quaternions_xyzw=quaternions,
    )


def decode_manus_shm_v1(
    snapshot: Any,
    *,
    identity: ManusStreamIdentity,
    receipt_timestamp_us: int,
    capture_class: str,
) -> ManusShmV1Receipt:
    """Decode one stable MANU frame and bind it to explicit stream evidence."""
    if not isinstance(snapshot, _StableSnapshot):
        raise ManusShmV1ContractError(
            "decoder requires a stable snapshot from capture_stable_snapshot"
        )
    identity = _validate_identity(identity)
    if capture_class not in _CAPTURE_CLASSES:
        raise ManusShmV1ContractError(
            f"capture_class must be one of {sorted(_CAPTURE_CLASSES)}"
        )
    if isinstance(receipt_timestamp_us, bool) or not isinstance(
        receipt_timestamp_us, int
    ):
        raise ManusShmV1ContractError("receipt_timestamp_us must be an integer")
    payload = snapshot.payload
    if len(payload) != MANUS_SHM_V1_BLOCK_SIZE:
        raise ManusShmV1ContractError(
            f"MANU frame length must be exactly {MANUS_SHM_V1_BLOCK_SIZE} bytes"
        )
    magic, version, seq, timestamp_us, hand_count = struct.unpack_from(
        "<IIQQB", payload, 0
    )
    if magic != MANUS_SHM_V1_MAGIC:
        raise ManusShmV1ContractError("MANU magic differs")
    if version != MANUS_SHM_V1_VERSION:
        raise ManusShmV1ContractError("MANU version differs")
    if hand_count != MANUS_SHM_V1_HAND_COUNT:
        raise ManusShmV1ContractError("MANU hand_count must be exactly 2")
    if payload[25:_HEADER_SIZE] != b"\0" * 7:
        raise ManusShmV1ContractError("MANU header padding must be zero")
    if receipt_timestamp_us < timestamp_us:
        raise ManusShmV1ContractError(
            "receipt timestamp regresses relative to the producer clock"
        )
    hands = (_decode_hand(payload, 0), _decode_hand(payload, 1))
    raw_sha = hashlib.sha256(payload).hexdigest()

    blockers: list[str] = []
    if capture_class == "synthetic_test":
        blockers.append("synthetic_input")
    # A digest stated by the transport is an identity claim, not proof that the
    # audited bytes for that binary were supplied and matched.
    blockers.append("producer_binary_evidence_missing")
    if identity.producer_source_sha256 is None:
        blockers.append("producer_source_sha256_missing")
    # Byte-identical reads are a useful transport seam but cannot prove that a
    # producer did not publish an ABA/torn state.  No accepted identity field is
    # allowed to self-assert this proof; audited producer source is required.
    blockers.append("producer_seqlock_proof_missing")
    provenance = {
        "schema_version": _RECEIPT_SCHEMA,
        "operator": identity.operator,
        "seq": seq,
        "timestamp_us": timestamp_us,
        "receipt_timestamp_us": receipt_timestamp_us,
        "raw_frame_sha256": raw_sha,
        "identity": _identity_provenance(identity),
        "snapshot": {
            "method": "byte_identical_double_read",
            "read_count": snapshot.read_count,
        },
        "capture_class": capture_class,
        "formal_live_eligible": False,
        "external_blockers": blockers,
    }
    receipt = ManusShmV1Receipt(
        seq=seq,
        timestamp_us=timestamp_us,
        hands=hands,
        raw_frame_sha256=raw_sha,
        raw_payload=payload,
        provenance=provenance,
        issuer_token=_RECEIPT_ISSUER,
    )
    _ISSUED_RECEIPTS.add(receipt)
    return receipt


__all__ = [
    "MANUS_SHM_V1_BLOCK_SIZE",
    "MANUS_SHM_V1_HAND_COUNT",
    "MANUS_SHM_V1_MAGIC",
    "MANUS_SHM_V1_NODE_COUNT",
    "MANUS_SHM_V1_VERSION",
    "DecodedManusHand",
    "ManusShmV1ContractError",
    "ManusShmV1Receipt",
    "ManusStreamIdentity",
    "TornSnapshotError",
    "capture_stable_snapshot",
    "decode_manus_shm_v1",
    "require_decoder_issued_receipt",
]
