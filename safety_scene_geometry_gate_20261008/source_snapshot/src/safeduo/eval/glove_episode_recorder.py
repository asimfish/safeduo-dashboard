"""Transport-neutral assembly of formal two-operator glove recordings.

The recorder owns no transport, SDK, simulator, or clock.  A caller submits one
atomic pair of already-synchronized :class:`OperatorFrame` values per control
transition, followed by the observed post-transition arm state.  The recorder
validates and copies every value; it never interpolates, resamples, fills, or
silently reorders source frames.

``finish`` returns exogenous data accepted by the frozen D-032 episode-bundle
writer.  Capture provenance is deliberately returned as a separate object
because the bundle schema contains only causally paired replay inputs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from safeduo.eval.trajectory_task_bundle import (
    ARM_DIMS,
    SCHEMA_VERSION,
    BundleValidationError,
    validate_exogenous_input,
)


ARM_ORDER = ("F_L", "F_R", "U_L", "U_R")
OPERATOR_ORDER = ("F", "U")
OPERATOR_ARM_ORDER = MappingProxyType({"F": ("F_L", "F_R"), "U": ("U_L", "U_R")})

_RH56F2_REVOLUTE_SUFFIXES = (
    "thumb_1_joint",
    "thumb_2_joint",
    "thumb_3_joint",
    "thumb_4_joint",
    "index_1_joint",
    "index_2_joint",
    "middle_1_joint",
    "middle_2_joint",
    "ring_1_joint",
    "ring_2_joint",
    "pinky_1_joint",
    "pinky_2_joint",
)
RH56F2_LEFT_JOINT_NAMES = tuple(f"left_{name}" for name in _RH56F2_REVOLUTE_SUFFIXES)
RH56F2_RIGHT_JOINT_NAMES = tuple(f"right_{name}" for name in _RH56F2_REVOLUTE_SUFFIXES)

_JOINT_NAMES_BY_ARM = MappingProxyType(
    {
        "F_L": RH56F2_LEFT_JOINT_NAMES,
        "F_R": RH56F2_RIGHT_JOINT_NAMES,
        "U_L": RH56F2_LEFT_JOINT_NAMES,
        "U_R": RH56F2_RIGHT_JOINT_NAMES,
    }
)
_HAND_STREAM_BY_ARM = MappingProxyType({arm: f"{arm}_hand" for arm in ARM_ORDER})
_PROVENANCE_SCHEMA_VERSION = "d032.glove_recorder.v2"
_CAPTURE_MODES = {"formal", "diagnostic_only"}


class RecorderContractError(ValueError):
    """A glove frame or recorder lifecycle transition violates the contract."""


@dataclass(frozen=True)
class ArmDelta:
    """One named arm's raw teleoperation increment before any safety action."""

    arm: str
    delta: Any


@dataclass(frozen=True)
class HandTarget:
    """One named RH56F2 absolute target in canonical revolute-joint order."""

    arm: str
    joint_names: Sequence[str]
    q: Any


@dataclass(frozen=True)
class OperatorFrame:
    """One operator's complete two-arm contribution to a control transition."""

    operator: str
    seq: int
    timestamp_s: float
    arm_deltas: Sequence[ArmDelta]
    hand_targets: Sequence[HandTarget]


class RecordedEpisode:
    """Defensive snapshot of bundle input and its separate capture provenance.

    Both accessors return deep copies.  This avoids handing callers references
    which could retroactively alter the recorder's completed evidence.
    """

    __slots__ = ("__exogenous", "__provenance")

    def __init__(
        self,
        *,
        exogenous: Mapping[str, Any],
        provenance: Mapping[str, Any],
    ) -> None:
        self.__exogenous = copy.deepcopy(dict(exogenous))
        self.__provenance = copy.deepcopy(dict(provenance))

    @property
    def exogenous(self) -> dict[str, Any]:
        """Return an independently mutable copy of the frozen bundle input."""
        return copy.deepcopy(self.__exogenous)

    @property
    def provenance(self) -> dict[str, Any]:
        """Return an independently mutable copy of the frozen provenance."""
        return copy.deepcopy(self.__provenance)


@dataclass(frozen=True)
class _ValidatedFrame:
    operator: str
    seq: int
    timestamp_s: float
    arm_deltas: dict[str, np.ndarray]
    hand_targets: dict[str, np.ndarray]


@dataclass(frozen=True)
class _RecorderTimeline:
    q_ref: Mapping[str, tuple[np.ndarray, ...]]
    raw_delta: Mapping[str, tuple[np.ndarray, ...]]
    hand_targets: Mapping[str, tuple[np.ndarray, ...]]
    seq: Mapping[str, tuple[int, ...]]
    timestamps: Mapping[str, tuple[float, ...]]
    clock_skews: tuple[float, ...]
    bound_adapter: Any | None


@dataclass(frozen=True)
class _PreparedRecorderTransition:
    base_state: _RecorderTimeline
    next_state: _RecorderTimeline


def _finite_vector(value: Any, dimension: int, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise RecorderContractError(f"{label} must be numeric")
    if array.shape != (dimension,):
        raise RecorderContractError(f"{label} shape {array.shape} != ({dimension},)")
    normalized = np.asarray(array, dtype=np.float64)
    if not np.isfinite(normalized).all():
        raise RecorderContractError(f"{label} must contain only finite values")
    return normalized.copy()


def _finite_nonnegative_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise RecorderContractError(f"{label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise RecorderContractError(f"{label} must be finite and non-negative")
    return normalized


def _finite_positive_float(value: Any, label: str) -> float:
    normalized = _finite_nonnegative_float(value, label)
    if normalized == 0.0:
        raise RecorderContractError(f"{label} must be positive")
    return normalized


def _cadence_matches(
    current: float,
    previous: float,
    expected_dt: float,
    tolerance: float,
) -> bool:
    """Compare cadence with tolerance plus only subtraction round-off.

    The caller-specified tolerance is the semantic allowance.  The additional
    bound is solely the IEEE-754 error introduced when two absolute timestamps
    are subtracted; without it, a mathematically exact tolerance-boundary value
    can fail because of binary representation.
    """
    interval = current - previous
    subtraction_roundoff = 2.0 * (math.ulp(current) + math.ulp(previous))
    return abs(interval - expected_dt) <= tolerance + subtraction_roundoff


def _validate_q_reference(value: Any, label: str) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping):
        raise RecorderContractError(f"{label} must be an ordered mapping")
    if tuple(value) != ARM_ORDER:
        raise RecorderContractError(
            f"{label} arm order {tuple(value)!r} != canonical order {ARM_ORDER!r}"
        )
    return {
        arm: _finite_vector(value[arm], ARM_DIMS[arm], f"{label}.{arm}")
        for arm in ARM_ORDER
    }


def _sequence(value: Any, label: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RecorderContractError(f"{label} must be a sequence")
    return tuple(value)


def _validate_hand_targets(
    value: Any, expected_arms: tuple[str, ...], label: str
) -> dict[str, np.ndarray]:
    targets = _sequence(value, label)
    if len(targets) != len(expected_arms):
        raise RecorderContractError(
            f"{label} must contain exactly {len(expected_arms)} hand targets"
        )
    actual_arms = tuple(getattr(target, "arm", None) for target in targets)
    if actual_arms != expected_arms:
        raise RecorderContractError(
            f"{label} hand arm order {actual_arms!r} != {expected_arms!r}"
        )
    normalized: dict[str, np.ndarray] = {}
    for arm, target in zip(expected_arms, targets, strict=True):
        if not isinstance(target, HandTarget):
            raise RecorderContractError(f"{label}.{arm} must be a HandTarget")
        joint_names = tuple(target.joint_names)
        expected_names = _JOINT_NAMES_BY_ARM[arm]
        if joint_names != expected_names:
            raise RecorderContractError(
                f"{label}.{arm} joint name/order differs from actual RH56F2 canonical order"
            )
        normalized[arm] = _finite_vector(target.q, 12, f"{label}.{arm}.q")
    return normalized


def _validate_frame(value: Any, expected_operator: str) -> _ValidatedFrame:
    if not isinstance(value, OperatorFrame):
        raise RecorderContractError(
            f"operator {expected_operator} contribution must be an OperatorFrame"
        )
    if value.operator != expected_operator:
        raise RecorderContractError(
            "operator pair order/identity must be exactly F then U with no duplicates"
        )
    if isinstance(value.seq, bool) or not isinstance(value.seq, (int, np.integer)):
        raise RecorderContractError(
            f"operator {expected_operator} seq must be an integer"
        )
    seq = int(value.seq)
    if seq < 0:
        raise RecorderContractError(
            f"operator {expected_operator} seq must be non-negative"
        )
    timestamp_s = _finite_nonnegative_float(
        value.timestamp_s, f"operator {expected_operator} timestamp_s"
    )

    expected_arms = OPERATOR_ARM_ORDER[expected_operator]
    deltas = _sequence(value.arm_deltas, f"operator {expected_operator} arm_deltas")
    if len(deltas) != len(expected_arms):
        raise RecorderContractError(
            f"operator {expected_operator} must contain exactly two arm deltas"
        )
    actual_arms = tuple(getattr(delta, "arm", None) for delta in deltas)
    if actual_arms != expected_arms:
        raise RecorderContractError(
            f"operator {expected_operator} arm ownership/order {actual_arms!r} "
            f"!= {expected_arms!r}"
        )
    normalized_deltas: dict[str, np.ndarray] = {}
    for arm, delta in zip(expected_arms, deltas, strict=True):
        if not isinstance(delta, ArmDelta):
            raise RecorderContractError(
                f"operator {expected_operator}.{arm} must be ArmDelta"
            )
        normalized_deltas[arm] = _finite_vector(
            delta.delta, ARM_DIMS[arm], f"operator {expected_operator}.{arm}.delta"
        )
    normalized_hands = _validate_hand_targets(
        value.hand_targets,
        expected_arms,
        f"operator {expected_operator}.hand_targets",
    )
    return _ValidatedFrame(
        operator=expected_operator,
        seq=seq,
        timestamp_s=timestamp_s,
        arm_deltas=normalized_deltas,
        hand_targets=normalized_hands,
    )


def _trace_digest(payload: Mapping[str, Any]) -> str:
    def jsonable(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Mapping):
            return {str(key): jsonable(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [jsonable(child) for child in value]
        return value

    encoded = json.dumps(
        jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(b"d032-glove-source-trace-v1\0" + encoded).hexdigest()


def recorder_source_trace_sha256(
    exogenous: Mapping[str, Any], operators: Mapping[str, Any]
) -> str:
    """Recompute the recorder source trace from strict core replay inputs.

    ``source_trace_sha256`` is evidence, not a caller assertion.  The formal
    outer bundle uses this helper to bind the separately stored operator clock
    trace to the actual q-reference, raw-delta, and hand streams accepted by
    the frozen core-bundle validator.
    """
    try:
        normalized = validate_exogenous_input(exogenous)
    except BundleValidationError as exc:
        raise RecorderContractError(
            f"core exogenous source trace is invalid: {exc}"
        ) from exc
    if not isinstance(operators, Mapping) or tuple(operators) != OPERATOR_ORDER:
        raise RecorderContractError(
            "source trace operators must be ordered exactly F then U"
        )
    trace_operators: dict[str, Any] = {}
    for operator in OPERATOR_ORDER:
        stream = operators[operator]
        if not isinstance(stream, Mapping):
            raise RecorderContractError(
                f"source trace operator {operator} must be a mapping"
            )
        trace_operators[operator] = copy.deepcopy(dict(stream))
    return _trace_digest(
        {
            "operators": trace_operators,
            "q_ref": normalized["q_ref"],
            "raw_delta": normalized["raw_delta"],
            "hands": {
                name: {
                    "arm": hand["arm"],
                    "kind": hand["kind"],
                    "joint_names": hand["joint_names"],
                    "values": hand["values"],
                    "valid": hand["valid"],
                    **({"initial": hand["initial"]} if hand["kind"] == "delta" else {}),
                }
                for name, hand in normalized["hands"].items()
            },
        }
    )


class GloveEpisodeRecorder:
    """Assemble complete two-operator frames into one immutable replay source."""

    def __init__(
        self,
        *,
        initial_q_ref: Mapping[str, Any],
        initial_hand_targets: Sequence[HandTarget],
        max_clock_skew_s: float,
        replay_dt_s: float,
        cadence_tolerance_s: float,
        capture_mode: str = "formal",
    ) -> None:
        self._max_clock_skew_s = _finite_nonnegative_float(
            max_clock_skew_s, "max_clock_skew_s"
        )
        self._replay_dt_s = _finite_positive_float(replay_dt_s, "replay_dt_s")
        self._cadence_tolerance_s = _finite_nonnegative_float(
            cadence_tolerance_s, "cadence_tolerance_s"
        )
        if self._cadence_tolerance_s >= self._replay_dt_s:
            raise RecorderContractError(
                "cadence_tolerance_s must be smaller than replay_dt_s"
            )
        if capture_mode not in _CAPTURE_MODES:
            raise RecorderContractError(
                f"capture_mode must be one of {sorted(_CAPTURE_MODES)}"
            )
        self._capture_mode = capture_mode
        initial_q = _validate_q_reference(initial_q_ref, "initial_q_ref")
        initial_hands = _validate_hand_targets(
            initial_hand_targets, ARM_ORDER, "initial_hand_targets"
        )
        self._state = _RecorderTimeline(
            q_ref={arm: (initial_q[arm],) for arm in ARM_ORDER},
            raw_delta={arm: () for arm in ARM_ORDER},
            hand_targets={arm: (initial_hands[arm],) for arm in ARM_ORDER},
            seq={operator: () for operator in OPERATOR_ORDER},
            timestamps={operator: () for operator in OPERATOR_ORDER},
            clock_skews=(),
            bound_adapter=None,
        )
        self._finished = False
        self._completed_episode: RecordedEpisode | None = None
        self._lock = threading.RLock()

    @property
    def transition_count(self) -> int:
        """Number of complete, atomically accepted operator pairs."""
        with self._lock:
            return len(self._state.clock_skews)

    def _require_open(self) -> None:
        if self._finished:
            raise RecorderContractError("glove episode recorder is already finished")

    def append_transition(
        self,
        frames: Sequence[OperatorFrame],
        *,
        post_q_ref: Mapping[str, Any],
    ) -> None:
        """Validate and atomically append one raw transport-neutral pair."""
        with self._lock:
            prepared = self._prepare_transition_unlocked(frames, post_q_ref=post_q_ref)
            self._install_prepared_transition_unlocked(prepared)

    def _prepare_transition_unlocked(
        self,
        frames: Sequence[OperatorFrame],
        *,
        post_q_ref: Mapping[str, Any],
        bound_adapter: Any | None = None,
    ) -> _PreparedRecorderTransition:
        """Validate and preallocate one complete next recorder timeline.

        No recorder state changes here.  Every tuple/dict allocation and array
        copy needed for the transition is complete before a coordinator may
        install the returned next-state pointer.
        """
        self._require_open()
        current_state = self._state
        if current_state.bound_adapter is not None:
            if bound_adapter is not current_state.bound_adapter:
                raise RecorderContractError(
                    "recorder is bound to one adapter transaction coordinator"
                )
        elif current_state.clock_skews and bound_adapter is not None:
            raise RecorderContractError(
                "direct recorder transitions cannot later bind a formal adapter"
            )
        pair = _sequence(frames, "operator frames")
        if len(pair) != 2:
            raise RecorderContractError(
                "each transition must contain exactly operator F and operator U"
            )
        f_frame = _validate_frame(pair[0], "F")
        u_frame = _validate_frame(pair[1], "U")
        validated = {"F": f_frame, "U": u_frame}
        post = _validate_q_reference(post_q_ref, "post_q_ref")

        for operator in OPERATOR_ORDER:
            frame = validated[operator]
            if current_state.seq[operator]:
                previous_seq = current_state.seq[operator][-1]
                if frame.seq <= previous_seq:
                    raise RecorderContractError(
                        f"operator {operator} seq must be strictly monotonic: "
                        f"{frame.seq} <= {previous_seq}"
                    )
                if self._capture_mode == "formal" and frame.seq != previous_seq + 1:
                    raise RecorderContractError(
                        f"formal operator {operator} seq must be exactly previous + 1: "
                        f"{frame.seq} != {previous_seq + 1}"
                    )
            if current_state.timestamps[operator]:
                previous_timestamp = current_state.timestamps[operator][-1]
                if frame.timestamp_s <= previous_timestamp:
                    raise RecorderContractError(
                        f"operator {operator} timestamp must be strictly monotonic: "
                        f"{frame.timestamp_s} <= {previous_timestamp}"
                    )
                interval = frame.timestamp_s - previous_timestamp
                if self._capture_mode == "formal" and not _cadence_matches(
                    frame.timestamp_s,
                    previous_timestamp,
                    self._replay_dt_s,
                    self._cadence_tolerance_s,
                ):
                    raise RecorderContractError(
                        f"formal operator {operator} cadence {interval:.17g}s differs "
                        f"from replay_dt_s {self._replay_dt_s:.17g}s by more than "
                        f"{self._cadence_tolerance_s:.17g}s"
                    )
        clock_skew = abs(f_frame.timestamp_s - u_frame.timestamp_s)
        if clock_skew > self._max_clock_skew_s:
            raise RecorderContractError(
                f"operator clock skew {clock_skew:.9g}s exceeds "
                f"{self._max_clock_skew_s:.9g}s"
            )

        next_state = _RecorderTimeline(
            q_ref={arm: (*current_state.q_ref[arm], post[arm]) for arm in ARM_ORDER},
            raw_delta={
                arm: (
                    *current_state.raw_delta[arm],
                    validated[arm[0]].arm_deltas[arm],
                )
                for arm in ARM_ORDER
            },
            hand_targets={
                arm: (
                    *current_state.hand_targets[arm],
                    validated[arm[0]].hand_targets[arm],
                )
                for arm in ARM_ORDER
            },
            seq={
                operator: (*current_state.seq[operator], validated[operator].seq)
                for operator in OPERATOR_ORDER
            },
            timestamps={
                operator: (
                    *current_state.timestamps[operator],
                    validated[operator].timestamp_s,
                )
                for operator in OPERATOR_ORDER
            },
            clock_skews=(*current_state.clock_skews, clock_skew),
            bound_adapter=(
                current_state.bound_adapter
                if current_state.bound_adapter is not None
                else bound_adapter
            ),
        )
        return _PreparedRecorderTransition(
            base_state=current_state,
            next_state=next_state,
        )

    def _install_prepared_transition_unlocked(
        self, prepared: _PreparedRecorderTransition
    ) -> None:
        """Install one fully allocated timeline using one non-allocating store."""
        if self._state is not prepared.base_state:
            raise RecorderContractError("prepared recorder transition is stale")
        self._state = prepared.next_state

    def append_prepared_transition(
        self,
        *,
        adapter: Any,
        prepared: Any,
        post_q_ref: Mapping[str, Any],
    ) -> None:
        """Let an adapter commit only after this recorder fully accepts a pair.

        Lock ordering is adapter then recorder.  The adapter holds its
        generation lock while invoking the recorder callback; every recorder
        validation runs before either committed chronology is advanced.
        """
        from safeduo.eval.manus_glove_adapter import (
            ManusGloveAdapter,
            PreparedManusPair,
        )

        if not isinstance(adapter, ManusGloveAdapter) or not isinstance(
            prepared, PreparedManusPair
        ):
            raise RecorderContractError(
                "prepared transition requires a Manus adapter and issued token"
            )

        adapter._commit_prepared_to_recorder(
            prepared,
            recorder=self,
            post_q_ref=post_q_ref,
        )

    def _provenance(self, exogenous: Mapping[str, Any]) -> dict[str, Any]:
        operators: dict[str, Any] = {}
        state = self._state
        for operator in OPERATOR_ORDER:
            seq = list(state.seq[operator])
            operators[operator] = {
                "owned_arms": list(OPERATOR_ARM_ORDER[operator]),
                "seq": seq,
                "timestamp_s": list(state.timestamps[operator]),
                "sequence_gap_count": sum(
                    current != previous + 1
                    for previous, current in zip(seq, seq[1:], strict=False)
                ),
            }
        return {
            "schema_version": _PROVENANCE_SCHEMA_VERSION,
            "source_kind": "two_operator_glove",
            "transport": "neutral",
            "capture_mode": self._capture_mode,
            "formal_eligible": self._capture_mode == "formal",
            "control_transitions": self.transition_count,
            "operator_submission_order": list(OPERATOR_ORDER),
            "operators": operators,
            "max_allowed_clock_skew_s": self._max_clock_skew_s,
            "max_observed_clock_skew_s": max(state.clock_skews),
            "replay_clock_policy": {
                "clock": "capture_timestamp_s",
                "replay_dt_s": self._replay_dt_s,
                "cadence_tolerance_s": self._cadence_tolerance_s,
                "sequence_policy": "exact_prev_plus_one_per_operator",
            },
            "interpolation_applied": False,
            "synthesized_frames": 0,
            "canonical_hand_joint_names": {
                "left": list(RH56F2_LEFT_JOINT_NAMES),
                "right": list(RH56F2_RIGHT_JOINT_NAMES),
            },
            "source_trace_sha256": recorder_source_trace_sha256(exogenous, operators),
        }

    def finish(
        self,
        *,
        objects: Mapping[str, Any],
        perturbation: Mapping[str, Any],
    ) -> RecordedEpisode:
        """Validate and return bundle-compatible exogenous data and provenance."""
        with self._lock:
            return self._finish_unlocked(objects=objects, perturbation=perturbation)

    def _finish_unlocked(
        self,
        *,
        objects: Mapping[str, Any],
        perturbation: Mapping[str, Any],
    ) -> RecordedEpisode:
        self._require_open()
        if self.transition_count < 1:
            raise RecorderContractError(
                "recording must contain at least one transition"
            )
        steps = self.transition_count
        state = self._state
        exogenous = {
            "schema_version": SCHEMA_VERSION,
            "q_ref": {arm: np.stack(state.q_ref[arm], axis=0) for arm in ARM_ORDER},
            "raw_delta": {
                arm: np.stack(state.raw_delta[arm], axis=0) for arm in ARM_ORDER
            },
            "hands": {
                _HAND_STREAM_BY_ARM[arm]: {
                    "arm": arm,
                    "kind": "q",
                    "joint_names": list(_JOINT_NAMES_BY_ARM[arm]),
                    "values": np.stack(state.hand_targets[arm], axis=0),
                    "valid": np.ones((steps + 1, 12), dtype=np.bool_),
                }
                for arm in ARM_ORDER
            },
            "objects": objects,
            "perturbation": perturbation,
        }
        try:
            normalized = validate_exogenous_input(exogenous)
        except BundleValidationError as exc:
            raise RecorderContractError(
                f"assembled exogenous input is not bundle-compatible: {exc}"
            ) from exc
        provenance = self._provenance(normalized)
        episode = RecordedEpisode(exogenous=normalized, provenance=provenance)
        self._completed_episode = episode
        self._finished = True
        return episode

    def _capture_material_unlocked(
        self, *, adapter: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return recorder-issued material only for its committed adapter peer."""
        if (
            not self._finished
            or self._completed_episode is None
            or self._state.bound_adapter is not adapter
        ):
            raise RecorderContractError(
                "recorder has no finished capture jointly bound to this adapter"
            )
        return (
            self._completed_episode.exogenous,
            self._completed_episode.provenance,
        )


__all__ = [
    "ARM_ORDER",
    "OPERATOR_ARM_ORDER",
    "OPERATOR_ORDER",
    "RH56F2_LEFT_JOINT_NAMES",
    "RH56F2_RIGHT_JOINT_NAMES",
    "ArmDelta",
    "GloveEpisodeRecorder",
    "HandTarget",
    "OperatorFrame",
    "RecordedEpisode",
    "RecorderContractError",
    "recorder_source_trace_sha256",
]
