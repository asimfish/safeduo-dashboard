"""Pickle-free evidence boundary for D-033 target-guard emergencies.

The environment stores the exact tensors consumed by the production guard in
its step cache.  If the guard aborts, an upstream runner can convert that cache
to this closed NumPy schema and persist it before destroying the simulator.
Conversion is intentionally outside the normal step hot path.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import torch

from safeduo.baselines.base import ConstraintRows
from safeduo.safety.target_guard import (
    TargetGuardBatchAbort,
    TargetGuardContractError,
    TargetGuardDecision,
    TargetGuardEmergency,
)
from safeduo.safety.types import ARM_KEYS, CLASS_TABLE


EMERGENCY_TRACE_SCHEMA = "safeduo.target_guard_emergency.v3"
CONTRACT_ERROR_TRACE_SCHEMA = "safeduo.target_guard_contract_error.v1"
EMERGENCY_TRACE_PHASE = "PRE_TARGET_BUFFER_COMMIT"
PAIR_MAP_SCHEMA = "safeduo.target_guard_pair_map.v2"
EMERGENCY_TRACE_AUTHORITY_SCHEMA = "safeduo.target_guard_trace_authority.v1"
TERMINATION_SNAPSHOT_AUTHORITY_SCHEMA = "safeduo.target_guard_termination_snapshot_authority.v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class PairMapEntry:
    """One exact static sphere-pair identity from ``SphereDistanceModule``."""

    global_pair_id: int
    left_sphere_ordinal: int
    left_qualified_link: str
    right_kind: Literal["sphere", "table"]
    right_ordinal: int
    right_qualified_name: str
    class_id: int
    d_min: float
    conditional: bool

    def __post_init__(self) -> None:
        if type(self.global_pair_id) is not int or self.global_pair_id < 0:
            raise ValueError("global_pair_id must be a non-negative int")
        if type(self.left_sphere_ordinal) is not int or self.left_sphere_ordinal < 0:
            raise ValueError("left sphere ordinal must be a non-negative int")
        if self.right_kind not in ("sphere", "table"):
            raise ValueError("right kind must be sphere or table")
        if type(self.right_ordinal) is not int or self.right_ordinal < 0:
            raise ValueError("right ordinal must be a non-negative int")
        if self.right_kind == "sphere" and self.right_ordinal <= self.left_sphere_ordinal:
            raise ValueError("sphere endpoint ordinals must be strictly increasing")
        for name, value in (
            ("left qualified link", self.left_qualified_link),
            ("right qualified name", self.right_qualified_name),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if type(self.class_id) is not int or self.class_id not in (0, 1, 2):
            raise ValueError("class_id must be one of 0, 1, 2")
        if self.right_kind == "table" and self.class_id != int(CLASS_TABLE):
            raise ValueError("table endpoint must use the table class")
        if self.right_kind == "sphere" and self.class_id == int(CLASS_TABLE):
            raise ValueError("sphere endpoint cannot use the table class")
        if not math.isfinite(self.d_min) or self.d_min < 0.0:
            raise ValueError("d_min must be finite and non-negative")
        if type(self.conditional) is not bool:
            raise ValueError("conditional must be a bool")

    @property
    def endpoint_identity(self) -> tuple[int, str, int]:
        return (self.left_sphere_ordinal, self.right_kind, self.right_ordinal)

    @property
    def pair_name(self) -> str:
        return (
            f"{self.left_qualified_link}#sphere:{self.left_sphere_ordinal}|"
            f"{self.right_qualified_name}#{self.right_kind}:{self.right_ordinal}"
        )

    def canonical_record(self) -> list[object]:
        return [
            self.global_pair_id,
            self.left_sphere_ordinal,
            self.left_qualified_link,
            self.right_kind,
            self.right_ordinal,
            self.right_qualified_name,
            self.class_id,
            self.d_min.hex(),
            self.conditional,
        ]


@dataclass(frozen=True, eq=False)
class PairMapAuthority:
    """Immutable pair map; authorization additionally requires issued identity."""

    entries: tuple[PairMapEntry, ...]
    pair_names: tuple[str, ...] = field(init=False)
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if not entries or any(not isinstance(entry, PairMapEntry) for entry in entries):
            raise ValueError("pair map must contain typed entries")
        expected_ids = tuple(range(len(entries)))
        if tuple(entry.global_pair_id for entry in entries) != expected_ids:
            raise ValueError("global pair IDs must be contiguous and ordered")
        endpoints = tuple(entry.endpoint_identity for entry in entries)
        if len(set(endpoints)) != len(endpoints):
            raise ValueError("duplicate pair endpoint identity")
        names = tuple(entry.pair_name for entry in entries)
        if len(set(names)) != len(names):
            raise ValueError("duplicate exact pair name")
        payload = json.dumps(
            {
                "schema": PAIR_MAP_SCHEMA,
                "pairs": [entry.canonical_record() for entry in entries],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "pair_names", names)
        object.__setattr__(self, "sha256", hashlib.sha256(payload).hexdigest())

    @property
    def pair_count(self) -> int:
        return len(self.entries)


@dataclass(frozen=True, eq=False)
class EmergencyTraceAuthority:
    """Opaque external binding for one complete emergency trace snapshot."""

    pair_map_authority: PairMapAuthority
    pair_relation_sha256: str
    snapshot_sha256: str

    def __post_init__(self) -> None:
        for name, value in (
            ("pair_relation_sha256", self.pair_relation_sha256),
            ("snapshot_sha256", self.snapshot_sha256),
        ):
            if not _SHA256_RE.fullmatch(value):
                raise ValueError(f"{name} must be lowercase 64-hex")


@dataclass(frozen=True, eq=False)
class TargetGuardTerminationSnapshot:
    """One immutable, process-local authority over a pre-commit termination."""

    kind: Literal["numeric_emergency", "contract_error"]
    trace: Mapping[str, np.ndarray]
    trace_authority: EmergencyTraceAuthority | None
    snapshot_sha256: str
    pair_relation_sha256: str | None
    pair_map_sha256: str | None
    attempted_transition_index: int
    pre_state_index: int
    committed_transitions: int
    sim_step_counter_before: int

    def __post_init__(self) -> None:
        if self.kind not in ("numeric_emergency", "contract_error"):
            raise ValueError("unknown target-guard termination kind")
        if not isinstance(self.trace, Mapping):
            raise TypeError("termination trace must be a mapping")
        if not _SHA256_RE.fullmatch(self.snapshot_sha256):
            raise ValueError("snapshot_sha256 must be lowercase 64-hex")
        for name, value in (
            ("attempted_transition_index", self.attempted_transition_index),
            ("pre_state_index", self.pre_state_index),
            ("committed_transitions", self.committed_transitions),
            ("sim_step_counter_before", self.sim_step_counter_before),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative native int")
        if not (
            self.attempted_transition_index == self.pre_state_index == self.committed_transitions
        ):
            raise ValueError("pre-commit transition counters must be identical")
        if self.kind == "numeric_emergency":
            if not isinstance(self.trace_authority, EmergencyTraceAuthority):
                raise ValueError("numeric emergency requires its trace authority")
            if not _SHA256_RE.fullmatch(str(self.pair_relation_sha256)):
                raise ValueError("numeric emergency requires pair_relation_sha256")
            if not _SHA256_RE.fullmatch(str(self.pair_map_sha256)):
                raise ValueError("numeric emergency requires pair_map_sha256")
        elif any(
            value is not None
            for value in (
                self.trace_authority,
                self.pair_relation_sha256,
                self.pair_map_sha256,
            )
        ):
            raise ValueError("contract-error snapshot cannot claim numeric authority")


_ISSUED_PAIR_MAP_AUTHORITIES: weakref.WeakSet[PairMapAuthority] = weakref.WeakSet()
_ISSUED_TRACE_AUTHORITIES: weakref.WeakSet[EmergencyTraceAuthority] = weakref.WeakSet()
_ISSUED_TERMINATION_SNAPSHOTS: weakref.WeakSet[TargetGuardTerminationSnapshot] = weakref.WeakSet()


def _require_issued_pair_map_authority(authority: object) -> PairMapAuthority:
    if not isinstance(authority, PairMapAuthority):
        raise ValueError("typed pair-map authority is required")
    if authority not in _ISSUED_PAIR_MAP_AUTHORITIES:
        raise ValueError("pair-map authority was not issued from SphereDistanceModule")
    return authority


def _pair_map_authority_from_sphere_distance(
    sphere_module: object,
) -> PairMapAuthority:
    """Privately issue one authority from the module's frozen static pair order."""

    from safeduo.safety.sphere_distance import SphereDistanceModule, TABLE_NAMES

    if type(sphere_module) is not SphereDistanceModule:
        raise ValueError("pair-map authority requires an exact SphereDistanceModule")

    pair_table_t = sphere_module.pair_table
    class_id_t = sphere_module.class_id
    pair_dmin_t = sphere_module.pair_dmin
    conditional_t = sphere_module.pair_conditional
    if pair_table_t.dtype != torch.int64 or pair_table_t.ndim != 2 or pair_table_t.shape[1] != 2:
        raise ValueError("sphere pair table must have shape (P, 2) and dtype int64")
    pair_count = pair_table_t.shape[0]
    if (
        class_id_t.shape != (pair_count,)
        or pair_dmin_t.shape != (pair_count,)
        or conditional_t.shape != (pair_count,)
        or conditional_t.dtype != torch.bool
    ):
        raise ValueError("sphere pair metadata must exactly match the pair table")

    qualified_names = tuple(sphere_module.qualified_names)
    if len(qualified_names) != sphere_module.n_spheres or any(
        type(name) is not str or not name for name in qualified_names
    ):
        raise ValueError("sphere qualified names do not match sphere ordinals")
    table_start = sphere_module._slice_table.start
    if (
        type(table_start) is not int
        or not 0 <= table_start <= pair_count
        or sphere_module._slice_table.stop != pair_count
    ):
        raise ValueError("table pair slice does not close the static pair table")

    pair_table = pair_table_t.detach().cpu().tolist()
    classes = class_id_t.detach().cpu().tolist()
    dmins = pair_dmin_t.detach().cpu().tolist()
    conditionals = conditional_t.detach().cpu().tolist()
    entries: list[PairMapEntry] = []
    for pair_id, ((left, right), raw_class, raw_dmin, conditional) in enumerate(
        zip(pair_table, classes, dmins, conditionals, strict=True)
    ):
        if type(left) is not int or not 0 <= left < sphere_module.n_spheres:
            raise ValueError("left sphere ordinal is outside the frozen sphere table")
        class_id = int(raw_class)
        if float(class_id) != float(raw_class):
            raise ValueError("pair class must be an exact integer class ID")
        if pair_id >= table_start:
            if type(right) is not int or not 0 <= right < len(TABLE_NAMES):
                raise ValueError("table ordinal is outside the frozen table map")
            right_kind: Literal["sphere", "table"] = "table"
            right_name = TABLE_NAMES[right]
        else:
            if type(right) is not int or not 0 <= right < sphere_module.n_spheres:
                raise ValueError("right sphere ordinal is outside the frozen sphere table")
            right_kind = "sphere"
            right_name = qualified_names[right]
        entries.append(
            PairMapEntry(
                global_pair_id=pair_id,
                left_sphere_ordinal=left,
                left_qualified_link=qualified_names[left],
                right_kind=right_kind,
                right_ordinal=right,
                right_qualified_name=right_name,
                class_id=class_id,
                d_min=float(raw_dmin),
                conditional=conditional,
            )
        )

    authority = PairMapAuthority(tuple(entries))
    _ISSUED_PAIR_MAP_AUTHORITIES.add(authority)
    return authority


def build_target_guard_step_cache(
    *,
    decision: TargetGuardDecision,
    rows: ConstraintRows,
    pair_ids: torch.Tensor,
    exempt: torch.Tensor,
    closing: torch.Tensor,
    q: dict[str, torch.Tensor],
    qd: dict[str, torch.Tensor],
    target_before: dict[str, torch.Tensor],
    exec_delta: dict[str, torch.Tensor],
    soft_limits: dict[str, torch.Tensor],
    dt: float,
    priority_p: torch.Tensor,
    pair_map_authority: PairMapAuthority,
) -> dict[str, Any]:
    """Retain the exact guard inputs until the step is committed or aborted.

    No tensor is copied here: the caller invokes this immediately after the
    guard decision and before any target-buffer write.  Emergency conversion
    happens synchronously in the abort handler, before those references can be
    reused by a later step.
    """

    pair_map_authority = _require_issued_pair_map_authority(pair_map_authority)
    emergency = isinstance(decision, TargetGuardEmergency)
    return {
        "status": "emergency" if emergency else "output",
        "target_write_authorized": not emergency,
        "decision": decision,
        "rows": rows,
        "pair_ids": pair_ids,
        "exempt": exempt,
        "closing": closing,
        "q": q,
        "qd": qd,
        "target_before": target_before,
        "exec_delta": exec_delta,
        "soft_limits": soft_limits,
        "dt": dt,
        "priority_p": priority_p,
        "pair_map_authority": pair_map_authority,
    }


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _arm_fields(cache: Mapping[str, Any]) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    for arm in ARM_KEYS:
        for name in ("q", "qd", "target_before", "exec_delta", "soft_limits"):
            fields[f"{name}_{arm}"] = _numpy(cache[name][arm])
    return fields


def emergency_trace_arrays(
    cache: Mapping[str, Any],
    *,
    sim_step_counter: int,
    episode_length: torch.Tensor,
    expected_pair_map_authority: PairMapAuthority,
) -> tuple[dict[str, np.ndarray], EmergencyTraceAuthority]:
    """Materialize a closed, ``allow_pickle=False`` emergency record.

    The four write-attempt fields describe the point at which the typed abort
    was raised.  An upstream owner must persist and validate this record before
    closing the simulator; this module deliberately does not perform I/O.
    """

    decision = cache.get("decision")
    if cache.get("status") != "emergency" or not isinstance(decision, TargetGuardEmergency):
        raise ValueError("cache must contain a TargetGuardEmergency")
    if cache.get("target_write_authorized") is not False:
        raise ValueError("emergency cache cannot authorize a target write")
    expected_pair_map_authority = _require_issued_pair_map_authority(expected_pair_map_authority)
    cache_authority = cache.get("pair_map_authority")
    if cache_authority is not expected_pair_map_authority:
        raise ValueError("emergency cache pair-map authority differs")

    rows = cache["rows"]
    if not isinstance(rows, ConstraintRows):
        raise ValueError("cache rows must be ConstraintRows")
    d_min = rows.d_min
    if d_min is None:
        raise ValueError("emergency trace requires the exact per-row d_min")

    pair_ids = _numpy(cache["pair_ids"])
    pair_names = _pair_names_for_ids(pair_ids, expected_pair_map_authority)
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray(EMERGENCY_TRACE_SCHEMA),
        "phase": np.asarray(EMERGENCY_TRACE_PHASE),
        "pair_map_sha256": np.asarray(expected_pair_map_authority.sha256),
        "pair_map_count": np.asarray(expected_pair_map_authority.pair_count, dtype=np.int64),
        "pair_name": pair_names,
        "sim_step_counter": np.asarray(sim_step_counter, dtype=np.int64),
        "episode_length": _numpy(episode_length).astype(np.int64, copy=False),
        "dt": np.asarray(cache["dt"], dtype=np.float64),
        "target_write_authorized": np.asarray(False, dtype=np.bool_),
        "target_write_attempted": np.asarray(False, dtype=np.bool_),
        "scene_write_attempted": np.asarray(False, dtype=np.bool_),
        "physics_step_attempted": np.asarray(False, dtype=np.bool_),
        "latch_committed": np.asarray(False, dtype=np.bool_),
        "infeasible": _numpy(decision.infeasible),
        "infeasible_reason": _numpy(decision.info["infeasible_reason"]),
        "held_arm": _numpy(decision.held_arm),
        "selected_mask": _numpy(decision.selected_mask),
        "trigger_pair_ids": _numpy(decision.trigger_pair_ids),
        "trigger_row": _numpy(decision.info["trigger_row"]),
        "pair_ids": pair_ids,
        "exempt": _numpy(cache["exempt"]),
        "closing": _numpy(cache["closing"]),
        "priority_p": _numpy(cache["priority_p"]),
        "row_d": _numpy(rows.d),
        "row_d_min": _numpy(d_min),
        "row_cls": _numpy(rows.cls),
        "row_valid": _numpy(rows.valid),
        "row_arm_mask": _numpy(rows.arm_mask),
        "row_J_F": _numpy(rows.J["F"]),
        "row_J_U": _numpy(rows.J["U"]),
    }
    arrays.update(_arm_fields(cache))
    _validate_emergency_trace_contents(
        arrays, expected_pair_map_authority=expected_pair_map_authority
    )
    trace_authority = EmergencyTraceAuthority(
        pair_map_authority=expected_pair_map_authority,
        pair_relation_sha256=_canonical_trace_sha256(arrays, field_names=_PAIR_RELATION_FIELDS),
        snapshot_sha256=_canonical_trace_sha256(arrays),
    )
    _ISSUED_TRACE_AUTHORITIES.add(trace_authority)
    validate_emergency_trace(arrays, expected_trace_authority=trace_authority)
    return arrays, trace_authority


def _pair_names_for_ids(pair_ids: np.ndarray, authority: PairMapAuthority) -> np.ndarray:
    if pair_ids.dtype != np.int64:
        raise ValueError("pair_ids must have dtype int64")
    if ((pair_ids < -1) | (pair_ids >= authority.pair_count)).any():
        raise ValueError("pair ID is outside the pinned pair map")
    names = [
        "" if pair_id < 0 else authority.pair_names[int(pair_id)]
        for pair_id in pair_ids.reshape(-1)
    ]
    return np.asarray(names, dtype=np.str_).reshape(pair_ids.shape)


_PAIR_RELATION_FIELDS = (
    "pair_map_sha256",
    "pair_map_count",
    "pair_ids",
    "pair_name",
    "row_valid",
    "exempt",
    "trigger_row",
    "trigger_pair_ids",
)


def _update_length_prefixed(hasher: Any, payload: bytes) -> None:
    hasher.update(len(payload).to_bytes(8, byteorder="big", signed=False))
    hasher.update(payload)


def _canonical_trace_sha256(
    trace: Mapping[str, np.ndarray],
    *,
    field_names: tuple[str, ...] | None = None,
) -> str:
    """Hash an exact, pickle-free canonical representation of trace arrays."""

    names = tuple(sorted(trace)) if field_names is None else field_names
    hasher = hashlib.sha256()
    _update_length_prefixed(
        hasher,
        (
            EMERGENCY_TRACE_AUTHORITY_SCHEMA
            if field_names is None
            else f"{EMERGENCY_TRACE_AUTHORITY_SCHEMA}.pair_relation"
        ).encode("utf-8"),
    )
    for name in names:
        if name not in trace:
            raise ValueError(f"trace authority field is missing: {name}")
        value = trace[name]
        if not isinstance(value, np.ndarray) or value.dtype == object:
            raise ValueError("trace authority requires non-object numpy arrays")
        _update_length_prefixed(hasher, name.encode("utf-8"))
        _update_length_prefixed(hasher, value.dtype.str.encode("ascii"))
        _update_length_prefixed(
            hasher,
            json.dumps(value.shape, separators=(",", ":")).encode("ascii"),
        )
        if value.dtype.kind == "U":
            for item in value.reshape(-1, order="C"):
                _update_length_prefixed(hasher, str(item).encode("utf-8"))
            continue
        canonical_dtype = value.dtype.newbyteorder("<")
        canonical = np.ascontiguousarray(value.astype(canonical_dtype, copy=False))
        _update_length_prefixed(hasher, canonical.tobytes(order="C"))
    return hasher.hexdigest()


def _canonical_contract_trace_sha256(trace: Mapping[str, np.ndarray]) -> str:
    """Hash the exact pickle-free contract-error sidecar representation."""

    hasher = hashlib.sha256()
    _update_length_prefixed(hasher, TERMINATION_SNAPSHOT_AUTHORITY_SCHEMA.encode("utf-8"))
    for name in sorted(trace):
        value = trace[name]
        if not isinstance(value, np.ndarray) or value.dtype == object:
            raise ValueError("contract trace authority requires non-object arrays")
        _update_length_prefixed(hasher, name.encode("utf-8"))
        _update_length_prefixed(hasher, value.dtype.str.encode("ascii"))
        _update_length_prefixed(
            hasher,
            json.dumps(value.shape, separators=(",", ":")).encode("ascii"),
        )
        if value.dtype.kind == "U":
            for item in value.reshape(-1, order="C"):
                _update_length_prefixed(hasher, str(item).encode("utf-8"))
            continue
        canonical_dtype = value.dtype.newbyteorder("<")
        canonical = np.ascontiguousarray(value.astype(canonical_dtype, copy=False))
        _update_length_prefixed(hasher, canonical.tobytes(order="C"))
    return hasher.hexdigest()


def _freeze_trace(trace: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    frozen: dict[str, np.ndarray] = {}
    for name, value in trace.items():
        if type(name) is not str or not name:
            raise ValueError("trace field names must be non-empty native strings")
        if not isinstance(value, np.ndarray) or value.dtype == object:
            raise ValueError("termination trace requires non-object numpy arrays")
        copy = np.array(value, copy=True)
        copy.setflags(write=False)
        frozen[name] = copy
    return MappingProxyType(frozen)


_CONTRACT_ERROR_FIELDS = {
    "schema",
    "phase",
    "contract_code",
    "field",
    "expected",
    "observed",
    "exception_module",
    "exception_qualname",
    "attempted_transition_index",
    "pre_state_index",
    "committed_transitions",
    "sim_step_counter_before",
    "episode_length_before",
    "runtime_poisoned",
    "target_write_authorized",
    "target_write_attempted",
    "scene_write_attempted",
    "physics_step_attempted",
    "latch_committed",
}


def _validate_contract_error_trace_contents(
    trace: Mapping[str, np.ndarray],
) -> dict[str, int]:
    if set(trace) != _CONTRACT_ERROR_FIELDS:
        missing = sorted(_CONTRACT_ERROR_FIELDS - set(trace))
        extra = sorted(set(trace) - _CONTRACT_ERROR_FIELDS)
        raise ValueError(
            f"contract-error trace field set mismatch: missing={missing}, extra={extra}"
        )
    if any(not isinstance(value, np.ndarray) for value in trace.values()):
        raise ValueError("every contract-error trace field must be a numpy array")
    if any(value.dtype == object for value in trace.values()):
        raise ValueError("contract-error trace cannot contain object dtype")

    string_fields = (
        "schema",
        "phase",
        "contract_code",
        "field",
        "expected",
        "observed",
        "exception_module",
        "exception_qualname",
    )
    for key in string_fields:
        value = trace[key]
        if value.shape != () or value.dtype.kind != "U":
            raise ValueError(f"{key} must be one scalar unicode string")
        text = str(value)
        if not text or len(text) > 1024 or "\x00" in text:
            raise ValueError(f"{key} is empty, oversized, or contains NUL")
    if str(trace["schema"]) != CONTRACT_ERROR_TRACE_SCHEMA:
        raise ValueError("unexpected contract-error trace schema")
    if str(trace["phase"]) != EMERGENCY_TRACE_PHASE:
        raise ValueError("contract error must be captured before target commit")
    if str(trace["contract_code"]) not in (
        "dtype",
        "device",
        "lifecycle_signature",
    ):
        raise ValueError("unknown target-guard contract code")
    if str(trace["exception_module"]) != "safeduo.safety.target_guard":
        raise ValueError("contract error module differs from the production type")
    if str(trace["exception_qualname"]) != "TargetGuardContractError":
        raise ValueError("contract error qualname differs from the production type")

    for key in (
        "attempted_transition_index",
        "pre_state_index",
        "committed_transitions",
        "sim_step_counter_before",
    ):
        value = trace[key]
        if value.shape != () or value.dtype != np.int64 or int(value) < 0:
            raise ValueError(f"{key} must be one non-negative int64")
    attempted = int(trace["attempted_transition_index"])
    if not (attempted == int(trace["pre_state_index"]) == int(trace["committed_transitions"])):
        raise ValueError("pre-commit transition counters must be identical")

    episode_length = trace["episode_length_before"]
    if (
        episode_length.dtype != np.int64
        or episode_length.ndim != 1
        or episode_length.size == 0
        or (episode_length < 0).any()
    ):
        raise ValueError("episode_length_before must be a non-empty int64 vector")
    for key in (
        "runtime_poisoned",
        "target_write_authorized",
        "target_write_attempted",
        "scene_write_attempted",
        "physics_step_attempted",
        "latch_committed",
    ):
        value = trace[key]
        if value.shape != () or value.dtype != np.bool_:
            raise ValueError(f"{key} must be one scalar bool")
    if not bool(trace["runtime_poisoned"]):
        raise ValueError("contract-error runtime must be permanently poisoned")
    for key in (
        "target_write_authorized",
        "target_write_attempted",
        "scene_write_attempted",
        "physics_step_attempted",
        "latch_committed",
    ):
        if bool(trace[key]):
            raise ValueError(f"{key} must be false before target commit")
    return {"num_envs": int(episode_length.size), "num_rows": 0}


def _expected_fields() -> set[str]:
    fields = {
        "schema",
        "phase",
        "pair_map_sha256",
        "pair_map_count",
        "pair_name",
        "sim_step_counter",
        "episode_length",
        "dt",
        "target_write_authorized",
        "target_write_attempted",
        "scene_write_attempted",
        "physics_step_attempted",
        "latch_committed",
        "infeasible",
        "infeasible_reason",
        "held_arm",
        "selected_mask",
        "trigger_pair_ids",
        "trigger_row",
        "pair_ids",
        "exempt",
        "closing",
        "priority_p",
        "row_d",
        "row_d_min",
        "row_cls",
        "row_valid",
        "row_arm_mask",
        "row_J_F",
        "row_J_U",
    }
    for arm in ARM_KEYS:
        fields.update(
            f"{name}_{arm}" for name in ("q", "qd", "target_before", "exec_delta", "soft_limits")
        )
    return fields


def _validate_emergency_trace_contents(
    trace: Mapping[str, np.ndarray],
    *,
    expected_pair_map_authority: PairMapAuthority,
) -> dict[str, int]:
    """Validate closed trace contents against the independently issued pair map."""

    expected_pair_map_authority = _require_issued_pair_map_authority(expected_pair_map_authority)

    expected = _expected_fields()
    if set(trace) != expected:
        missing = sorted(expected - set(trace))
        extra = sorted(set(trace) - expected)
        raise ValueError(f"emergency trace field set mismatch: missing={missing}, extra={extra}")
    if any(not isinstance(value, np.ndarray) for value in trace.values()):
        raise ValueError("every emergency trace field must be a numpy array")
    if any(value.dtype == object for value in trace.values()):
        raise ValueError("emergency trace cannot contain object dtype")
    if str(trace["schema"]) != EMERGENCY_TRACE_SCHEMA:
        raise ValueError("unexpected emergency trace schema")
    if str(trace["phase"]) != EMERGENCY_TRACE_PHASE:
        raise ValueError("emergency trace must be captured before target commit")
    pair_map_sha256 = trace["pair_map_sha256"]
    if (
        pair_map_sha256.shape != ()
        or pair_map_sha256.dtype.kind != "U"
        or not _SHA256_RE.fullmatch(str(pair_map_sha256))
    ):
        raise ValueError("pair_map_sha256 must be one lowercase 64-hex string")
    if str(pair_map_sha256) != expected_pair_map_authority.sha256:
        raise ValueError("pair-map authority digest differs")
    if (
        trace["pair_map_count"].shape != ()
        or trace["pair_map_count"].dtype != np.int64
        or int(trace["pair_map_count"]) != expected_pair_map_authority.pair_count
    ):
        raise ValueError("pair-map authority count differs")

    no_write = (
        "target_write_authorized",
        "target_write_attempted",
        "scene_write_attempted",
        "physics_step_attempted",
        "latch_committed",
    )
    for key in no_write:
        if trace[key].shape != () or trace[key].dtype != np.bool_:
            raise ValueError(f"{key} must be a scalar bool")
        if bool(trace[key]):
            raise ValueError(f"{key} must be false for a pre-commit emergency")

    for key in (
        "infeasible",
        "held_arm",
        "trigger_row",
        "exempt",
        "row_valid",
        "row_arm_mask",
    ):
        if trace[key].dtype != np.bool_:
            raise ValueError(f"{key} must have dtype bool")
    for key in (
        "sim_step_counter",
        "episode_length",
        "infeasible_reason",
        "selected_mask",
        "trigger_pair_ids",
        "pair_ids",
    ):
        if trace[key].dtype != np.int64:
            raise ValueError(f"{key} must have dtype int64")

    infeasible = trace["infeasible"]
    row_d = trace["row_d"]
    if infeasible.ndim != 1 or infeasible.dtype != np.bool_:
        raise ValueError("infeasible must have shape (N,) and bool dtype")
    if row_d.ndim != 2:
        raise ValueError("row_d must have shape (N, M)")
    n, m = row_d.shape
    if infeasible.shape != (n,) or not bool(infeasible.any()):
        raise ValueError("emergency trace must contain an infeasible environment")
    if trace["episode_length"].shape != (n,):
        raise ValueError("episode_length must have shape (N,)")
    if trace["held_arm"].shape != (n, len(ARM_KEYS)):
        raise ValueError("held_arm must have shape (N, 4)")
    for key in (
        "infeasible_reason",
        "selected_mask",
        "priority_p",
    ):
        if trace[key].shape != (n,):
            raise ValueError(f"{key} must have shape (N,)")
    for key in (
        "trigger_pair_ids",
        "trigger_row",
        "pair_ids",
        "pair_name",
        "exempt",
        "closing",
        "row_d_min",
        "row_cls",
        "row_valid",
    ):
        if trace[key].shape != (n, m):
            raise ValueError(f"{key} must have shape (N, M)")
    pair_ids = trace["pair_ids"]
    if not np.array_equal(trace["row_valid"], pair_ids >= 0):
        raise ValueError("row_valid must exactly match the pair ID padding map")
    if (trace["trigger_row"] & (~trace["row_valid"] | trace["exempt"])).any():
        raise ValueError("trigger rows must be valid, non-exempt pair rows")
    expected_trigger_pair_ids = np.where(trace["trigger_row"], pair_ids, -1)
    if not np.array_equal(trace["trigger_pair_ids"], expected_trigger_pair_ids):
        raise ValueError("trigger pair IDs differ from the trigger-row pair map")
    expected_pair_names = _pair_names_for_ids(pair_ids, expected_pair_map_authority)
    if trace["pair_name"].dtype.kind != "U" or not np.array_equal(
        trace["pair_name"], expected_pair_names
    ):
        raise ValueError("pair names differ from the pinned pair map")
    if trace["row_cls"].dtype.kind != "f":
        raise ValueError("row_cls must have a floating dtype")
    safe_pair_ids = pair_ids.clip(min=0)
    entry_classes = np.asarray(
        [entry.class_id for entry in expected_pair_map_authority.entries],
        dtype=trace["row_cls"].dtype,
    )
    expected_row_cls = entry_classes[safe_pair_ids]
    if not np.array_equal(
        trace["row_cls"][trace["row_valid"]],
        expected_row_cls[trace["row_valid"]],
    ):
        raise ValueError("row classes differ from the pinned pair map")
    entry_conditional = np.asarray(
        [entry.conditional for entry in expected_pair_map_authority.entries],
        dtype=np.bool_,
    )
    fixed_dmin_row = trace["row_valid"] & ~entry_conditional[safe_pair_ids]
    entry_dmin = np.asarray(
        [entry.d_min for entry in expected_pair_map_authority.entries],
        dtype=trace["row_d_min"].dtype,
    )
    expected_row_dmin = entry_dmin[safe_pair_ids]
    if not np.array_equal(trace["row_d_min"][fixed_dmin_row], expected_row_dmin[fixed_dmin_row]):
        raise ValueError("non-conditional row d_min differs from the pinned pair map")
    if trace["row_arm_mask"].shape != (n, m, len(ARM_KEYS)):
        raise ValueError("row_arm_mask must have shape (N, M, 4)")
    if trace["row_J_F"].shape[:2] != (n, m):
        raise ValueError("row_J_F batch/row shape mismatch")
    if trace["row_J_U"].shape[:2] != (n, m):
        raise ValueError("row_J_U batch/row shape mismatch")
    for arm in ARM_KEYS:
        q = trace[f"q_{arm}"]
        if q.ndim != 2 or q.shape[0] != n:
            raise ValueError(f"q_{arm} must have shape (N, dof)")
        for name in ("qd", "target_before", "exec_delta"):
            if trace[f"{name}_{arm}"].shape != q.shape:
                raise ValueError(f"{name}_{arm} shape mismatch")
        limits = trace[f"soft_limits_{arm}"]
        if limits.ndim != 3 or limits.shape[0] not in (1, n):
            raise ValueError(f"soft_limits_{arm} must have shape (1|N, dof, 2)")
        if limits.shape[1:] != (*q.shape[1:], 2):
            raise ValueError(f"soft_limits_{arm} tail shape mismatch")
    return {"num_envs": n, "num_rows": m}


def validate_emergency_trace(
    trace: Mapping[str, np.ndarray],
    *,
    expected_trace_authority: EmergencyTraceAuthority,
) -> dict[str, int]:
    """Validate one trace against an opaque binding retained outside the trace."""

    if not isinstance(expected_trace_authority, EmergencyTraceAuthority):
        raise ValueError("expected emergency-trace authority is required")
    if expected_trace_authority not in _ISSUED_TRACE_AUTHORITIES:
        raise ValueError("emergency-trace authority was not issued for this snapshot")
    pair_map_authority = _require_issued_pair_map_authority(
        expected_trace_authority.pair_map_authority
    )
    result = _validate_emergency_trace_contents(
        trace, expected_pair_map_authority=pair_map_authority
    )
    pair_relation_sha256 = _canonical_trace_sha256(trace, field_names=_PAIR_RELATION_FIELDS)
    if pair_relation_sha256 != expected_trace_authority.pair_relation_sha256:
        raise ValueError("pair relation differs from the externally retained authority")
    snapshot_sha256 = _canonical_trace_sha256(trace)
    if snapshot_sha256 != expected_trace_authority.snapshot_sha256:
        raise ValueError("trace snapshot differs from the externally retained authority")
    return result


def _validate_precommit_counters(
    *,
    attempted_transition_index: int,
    pre_state_index: int,
    committed_transitions: int,
    sim_step_counter_before: int,
    episode_length_before: torch.Tensor,
) -> None:
    for name, value in (
        ("attempted_transition_index", attempted_transition_index),
        ("pre_state_index", pre_state_index),
        ("committed_transitions", committed_transitions),
        ("sim_step_counter_before", sim_step_counter_before),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative native int")
    if not (attempted_transition_index == pre_state_index == committed_transitions):
        raise ValueError("pre-commit transition counters must be identical")
    if (
        type(episode_length_before) is not torch.Tensor
        or episode_length_before.dtype != torch.int64
        or episode_length_before.ndim != 1
        or episode_length_before.numel() == 0
    ):
        raise ValueError("episode_length_before must be a non-empty int64 tensor")


def _contract_error_trace_arrays(
    error: TargetGuardContractError,
    *,
    attempted_transition_index: int,
    pre_state_index: int,
    committed_transitions: int,
    sim_step_counter_before: int,
    episode_length_before: torch.Tensor,
) -> dict[str, np.ndarray]:
    if type(error) is not TargetGuardContractError:
        raise TypeError("exact TargetGuardContractError is required")
    arrays = {
        "schema": np.asarray(CONTRACT_ERROR_TRACE_SCHEMA),
        "phase": np.asarray(EMERGENCY_TRACE_PHASE),
        "contract_code": np.asarray(error.contract_code),
        "field": np.asarray(error.field),
        "expected": np.asarray(error.expected),
        "observed": np.asarray(error.observed),
        "exception_module": np.asarray(type(error).__module__),
        "exception_qualname": np.asarray(type(error).__qualname__),
        "attempted_transition_index": np.asarray(attempted_transition_index, dtype=np.int64),
        "pre_state_index": np.asarray(pre_state_index, dtype=np.int64),
        "committed_transitions": np.asarray(committed_transitions, dtype=np.int64),
        "sim_step_counter_before": np.asarray(sim_step_counter_before, dtype=np.int64),
        "episode_length_before": _numpy(episode_length_before).astype(np.int64, copy=False),
        "runtime_poisoned": np.asarray(True, dtype=np.bool_),
        "target_write_authorized": np.asarray(False, dtype=np.bool_),
        "target_write_attempted": np.asarray(False, dtype=np.bool_),
        "scene_write_attempted": np.asarray(False, dtype=np.bool_),
        "physics_step_attempted": np.asarray(False, dtype=np.bool_),
        "latch_committed": np.asarray(False, dtype=np.bool_),
    }
    _validate_contract_error_trace_contents(arrays)
    return arrays


def materialize_target_guard_termination(
    error: TargetGuardBatchAbort | TargetGuardContractError,
    *,
    emergency_cache: Mapping[str, Any] | None,
    expected_pair_map_authority: PairMapAuthority | None,
    attempted_transition_index: int,
    pre_state_index: int,
    committed_transitions: int,
    sim_step_counter_before: int,
    episode_length_before: torch.Tensor,
) -> TargetGuardTerminationSnapshot:
    """Synchronously close one typed guard failure before any simulator write."""

    _validate_precommit_counters(
        attempted_transition_index=attempted_transition_index,
        pre_state_index=pre_state_index,
        committed_transitions=committed_transitions,
        sim_step_counter_before=sim_step_counter_before,
        episode_length_before=episode_length_before,
    )
    if type(error) is TargetGuardBatchAbort:
        if emergency_cache is None:
            raise ValueError("numeric abort requires its pre-commit emergency cache")
        if emergency_cache.get("decision") is not error.emergency:
            raise ValueError("abort and cache must retain the same emergency object")
        if expected_pair_map_authority is None:
            raise ValueError("numeric abort requires the issued pair-map authority")
        arrays, trace_authority = emergency_trace_arrays(
            emergency_cache,
            sim_step_counter=sim_step_counter_before,
            episode_length=episode_length_before,
            expected_pair_map_authority=expected_pair_map_authority,
        )
        frozen = _freeze_trace(arrays)
        snapshot = TargetGuardTerminationSnapshot(
            kind="numeric_emergency",
            trace=frozen,
            trace_authority=trace_authority,
            snapshot_sha256=trace_authority.snapshot_sha256,
            pair_relation_sha256=trace_authority.pair_relation_sha256,
            pair_map_sha256=expected_pair_map_authority.sha256,
            attempted_transition_index=attempted_transition_index,
            pre_state_index=pre_state_index,
            committed_transitions=committed_transitions,
            sim_step_counter_before=sim_step_counter_before,
        )
    elif type(error) is TargetGuardContractError:
        if emergency_cache is not None or expected_pair_map_authority is not None:
            raise ValueError("contract error cannot claim numeric emergency evidence")
        arrays = _contract_error_trace_arrays(
            error,
            attempted_transition_index=attempted_transition_index,
            pre_state_index=pre_state_index,
            committed_transitions=committed_transitions,
            sim_step_counter_before=sim_step_counter_before,
            episode_length_before=episode_length_before,
        )
        snapshot_sha256 = _canonical_contract_trace_sha256(arrays)
        snapshot = TargetGuardTerminationSnapshot(
            kind="contract_error",
            trace=_freeze_trace(arrays),
            trace_authority=None,
            snapshot_sha256=snapshot_sha256,
            pair_relation_sha256=None,
            pair_map_sha256=None,
            attempted_transition_index=attempted_transition_index,
            pre_state_index=pre_state_index,
            committed_transitions=committed_transitions,
            sim_step_counter_before=sim_step_counter_before,
        )
    else:
        raise TypeError("unsupported target-guard termination error")

    _ISSUED_TERMINATION_SNAPSHOTS.add(snapshot)
    validate_target_guard_termination_snapshot(snapshot)
    return snapshot


def validate_target_guard_termination_snapshot(
    snapshot: TargetGuardTerminationSnapshot,
) -> dict[str, int | str]:
    """Validate an issued immutable snapshot before runner persistence."""

    if type(snapshot) is not TargetGuardTerminationSnapshot:
        raise TypeError("exact TargetGuardTerminationSnapshot is required")
    if snapshot not in _ISSUED_TERMINATION_SNAPSHOTS:
        raise ValueError("target-guard termination snapshot was not issued")
    if any(
        not isinstance(value, np.ndarray) or value.flags.writeable
        for value in snapshot.trace.values()
    ):
        raise ValueError("termination snapshot arrays must be immutable")

    if snapshot.kind == "numeric_emergency":
        assert snapshot.trace_authority is not None
        result = validate_emergency_trace(
            snapshot.trace, expected_trace_authority=snapshot.trace_authority
        )
        if snapshot.snapshot_sha256 != snapshot.trace_authority.snapshot_sha256:
            raise ValueError("numeric snapshot digest differs from trace authority")
        if snapshot.pair_relation_sha256 != snapshot.trace_authority.pair_relation_sha256:
            raise ValueError("numeric pair relation differs from trace authority")
        if snapshot.pair_map_sha256 != snapshot.trace_authority.pair_map_authority.sha256:
            raise ValueError("numeric pair map differs from trace authority")
        if int(snapshot.trace["sim_step_counter"]) != snapshot.sim_step_counter_before:
            raise ValueError("numeric snapshot sim-step counter differs")
    elif snapshot.kind == "contract_error":
        result = _validate_contract_error_trace_contents(snapshot.trace)
        if _canonical_contract_trace_sha256(snapshot.trace) != snapshot.snapshot_sha256:
            raise ValueError("contract-error snapshot digest differs")
        for key, expected in (
            ("attempted_transition_index", snapshot.attempted_transition_index),
            ("pre_state_index", snapshot.pre_state_index),
            ("committed_transitions", snapshot.committed_transitions),
            ("sim_step_counter_before", snapshot.sim_step_counter_before),
        ):
            if int(snapshot.trace[key]) != expected:
                raise ValueError(f"contract-error {key} differs from the envelope")
    else:  # pragma: no cover - frozen dataclass constructor rejects this
        raise ValueError("unknown target-guard termination kind")
    return {"kind": snapshot.kind, **result}
